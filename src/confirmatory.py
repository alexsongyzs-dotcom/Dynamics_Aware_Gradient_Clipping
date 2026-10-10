"""Independent, resumable full-data protocol; does not change legacy experiments.

The official test set is loaded only for final confirmation. All tuning uses a
fixed stratified 5,000-example split of the official training set. Images are
cached on the selected device; augmentation and batch order use private RNGs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets

from .cifar_resnet import CifarResNet20
from .clipping import DynamicsAwareClipping, adaptive_gradient_clip_
from .models import SmallCNN


ENGINE_VERSION = 'lyx1-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_digest():
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root / f).read_bytes() for f in
        ('confirmatory.py', 'cifar_resnet.py', 'clipping.py', 'models.py'))).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def stratified_split(labels, seed=1729, val_per_class=500):
    labels = torch.as_tensor(labels).cpu()
    rng = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in torch.unique(labels, sorted=True):
        idx = torch.where(labels == label)[0]
        idx = idx[torch.randperm(len(idx), generator=rng)]
        val.append(idx[:val_per_class])
        train.append(idx[val_per_class:])
    return torch.cat(train), torch.cat(val)


def load_data(cfg, data_dir, device):
    cls = datasets.FashionMNIST if cfg['dataset'] == 'fashion_mnist' else datasets.CIFAR10
    official = cls(data_dir, train=True, download=False)
    labels = torch.as_tensor(official.targets, dtype=torch.long)
    train_idx, val_idx = stratified_split(labels, cfg['split_seed'])
    def tensors(ds):
        images = torch.as_tensor(np.asarray(ds.data)).to(device=device, dtype=torch.float32) / 255.
        images = images.unsqueeze(1) if images.ndim == 3 else images.permute(0, 3, 1, 2).contiguous()
        return images, torch.as_tensor(ds.targets, dtype=torch.long, device=device)
    images, targets = tensors(official)
    train = (images[train_idx.to(device)], targets[train_idx.to(device)])
    if cfg['stage'] == 'confirmation':
        test = cls(data_dir, train=False, download=False)
        evaluation = tensors(test)
        evaluation_name = 'official_test'
    else:
        evaluation = (images[val_idx.to(device)], targets[val_idx.to(device)])
        evaluation_name = 'training_validation'
    split_hash = hashlib.sha256(train_idx.numpy().tobytes() + val_idx.numpy().tobytes()).hexdigest()
    return train, evaluation, {'train_n':len(train_idx), 'validation_n':len(val_idx),
        'evaluation_n':len(evaluation[1]), 'evaluation_split':evaluation_name,
        'split_sha256':split_hash}


def preprocess(x, dataset, rng=None):
    if dataset == 'fashion_mnist':
        return (x - .286) / .353
    if rng is not None:
        # Integer crop offsets are per image, exactly uniform on {0,...,8}.
        padded = F.pad(x, (4, 4, 4, 4))
        offsets = torch.randint(9, (len(x), 2), device=x.device, generator=rng)
        row = offsets[:, 0, None, None] + torch.arange(32, device=x.device)[None, :, None]
        col = offsets[:, 1, None, None] + torch.arange(32, device=x.device)[None, None, :]
        x = padded.permute(0, 2, 3, 1)[torch.arange(len(x), device=x.device)[:, None, None], row, col]
        x = x.permute(0, 3, 1, 2).contiguous()
        flip = torch.rand(len(x), device=x.device, generator=rng) < .5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    mean = x.new_tensor((.4914, .4822, .4465))[None, :, None, None]
    std = x.new_tensor((.2470, .2435, .2616))[None, :, None, None]
    return (x - mean) / std


class Policy:
    def __init__(self, cfg, replay=None):
        self.name = cfg['policy']
        self.value = cfg.get('value')
        self.norms = []
        self.replay = replay
        self.controller = DynamicsAwareClipping(init_c=.5, gamma=.05, exposure_target=.1,
            osc_weight=0. if self.name == 'no_alignment' else 3.)

    def apply(self, params, norm, alignment, step):
        name = self.name
        if name in ('dagc', 'no_alignment'):
            threshold = self.controller.c
        elif name == 'none':
            threshold = norm
        elif name == 'fixed':
            threshold = self.value
        elif name == 'median':
            # Lagged lower median matches the controller's bound convention.
            values = self.norms[-200:]
            threshold = self.value * float(torch.tensor(values).median()) if values else .5
        elif name == 'autoclip':
            threshold = float(np.percentile(self.norms + [norm], self.value))
        elif name == 'replay':
            threshold = float(self.replay[step])
        elif name == 'agc':
            adaptive_gradient_clip_(params, self.value)
            clipped_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params])).item()
            self.norms.append(norm)
            return None, clipped_norm / max(norm, 1e-12), False
        else:
            raise ValueError(name)
        coefficient = min(1., threshold / max(norm, 1e-12))
        torch._foreach_mul_([p.grad for p in params], coefficient)
        lower_hit = False
        if name in ('dagc', 'no_alignment'):
            self.controller.update(norm, alignment)
            lower, _ = self.controller._bounds()
            lower_hit = math.isclose(self.controller.c, lower, rel_tol=1e-6)
        self.norms.append(norm)
        return threshold, coefficient, lower_hit


@torch.no_grad()
def evaluate(model, data, dataset):
    model.eval()
    x, y = data
    loss_sum, correct = 0., 0
    for i in range(0, len(y), 1000):
        logits = model(preprocess(x[i:i+1000], dataset))
        loss_sum += F.cross_entropy(logits, y[i:i+1000], reduction='sum').item()
        correct += (logits.argmax(1) == y[i:i+1000]).sum().item()
    model.train()
    loss = loss_sum / len(y)
    return {'accuracy':correct / len(y), 'loss':loss if math.isfinite(loss) else None}


def run(cfg, output, data_dir, device='cuda', interrupt_after_epoch=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = digest({'config':cfg, 'source':source_digest(), 'version':ENGINE_VERSION})
    result_path, checkpoint = output / 'result.json', output / 'checkpoint.pt'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['identity'] != identity:
            raise ValueError(f'Result identity mismatch: {output}')
        return result
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    train, evaluation, split = load_data(cfg, data_dir, device)
    model = (SmallCNN() if cfg['dataset'] == 'fashion_mnist' else CifarResNet20()).to(device)
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=cfg['lr'], momentum=.9, weight_decay=cfg['weight_decay'])
    replay = None
    if cfg['policy'] == 'replay':
        replay_file = Path(cfg['replay_path'])
        if hashlib.sha256(replay_file.read_bytes()).hexdigest() != cfg['replay_sha256']:
            raise ValueError('Replay checksum mismatch')
        replay = np.load(replay_file)
    policy = Policy(cfg, replay)
    previous = None
    rows, traces = [], []
    epoch_start, elapsed_before, step = 0, 0., 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        policy, previous = saved['policy'], saved['previous']
        rows, traces = saved['rows'], saved['traces']
        epoch_start, elapsed_before, step = saved['epoch'], saved['elapsed'], len(traces)
    start = time.monotonic()
    failure = None
    x, y = train
    batch_size = cfg['batch_size']
    total_steps = math.ceil(len(y) / batch_size) * cfg['epochs']
    if replay is not None and len(replay) != total_steps:
        raise ValueError('Replay length must equal the complete training budget')
    for epoch in range(epoch_start, cfg['epochs']):
        model.train()
        rng = torch.Generator(device=device).manual_seed(cfg['seed'] * 100000 + epoch + 19)
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(
            cfg['seed'] * 100000 + epoch + 7)).to(device)
        losses, correct, seen = 0., 0, 0
        for begin in range(0, len(y), batch_size):
            idx = order[begin:begin+batch_size]
            inputs = preprocess(x[idx], cfg['dataset'], rng)
            targets = y[idx]
            opt.zero_grad(set_to_none=True)
            # Cosine decay starts immediately and ends at one percent of base LR.
            lr = cfg['lr'] * (.01 + .99 * .5 * (1 + math.cos(math.pi * step / max(total_steps-1, 1))))
            opt.param_groups[0]['lr'] = lr
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            flat = torch.cat([p.grad.flatten() for p in params])
            norm_tensor = flat.norm()
            norm = norm_tensor.item()
            loss_value = loss.item()
            if not (math.isfinite(loss_value) and math.isfinite(norm)) or loss_value > 1e6:
                failure = {'epoch':epoch, 'step':step, 'reason':'nonfinite_or_loss_above_1e6'}
                break
            alignment = 1. if previous is None else (torch.dot(flat, previous) /
                (norm_tensor * previous.norm()).clamp_min(1e-12)).item()
            threshold, coefficient, lower_hit = policy.apply(params, norm, alignment, step)
            opt.step()
            previous = flat.detach()  # cat owns independent, unclipped storage
            traces.append([norm, threshold, coefficient, alignment, int(lower_hit)])
            step += 1
            losses += loss_value * len(idx)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += len(idx)
        rows.append({'epoch':epoch+1, 'train_loss': losses / max(seen,1),
                     'train_accuracy':correct / max(seen,1), 'lr':lr, 'steps':step})
        if failure:
            break
        saved = {'identity':identity, 'epoch':epoch+1, 'model':model.state_dict(),
            'optimizer':opt.state_dict(), 'policy':policy, 'previous':previous,
            'rows':rows, 'traces':traces, 'elapsed':elapsed_before+time.monotonic()-start}
        temp = checkpoint.with_suffix('.tmp')
        torch.save(saved, temp)
        os.replace(temp, checkpoint)
        atomic_json(output / 'progress.json', {'identity':identity, 'epoch':epoch+1,
            'epochs':cfg['epochs'], 'step':step, 'elapsed_seconds':saved['elapsed']})
        if interrupt_after_epoch == epoch+1:
            return None
    elapsed = elapsed_before + time.monotonic() - start
    metrics = evaluate(model, evaluation, cfg['dataset']) if not failure else {'accuracy':0., 'loss':None}
    if metrics['loss'] is None and failure is None:
        failure = {'epoch':cfg['epochs'], 'step':step, 'reason':'nonfinite_final_evaluation'}
        metrics['accuracy'] = 0.
    arr = np.array([[float('nan') if v is None else v for v in row] for row in traces], dtype=float).reshape(-1,5)
    np.savez_compressed(output / 'trace.npz', norm=arr[:,0], threshold=arr[:,1], coefficient=arr[:,2],
        alignment=arr[:,3], lower_bound_hit=arr[:,4])
    result = {'identity':identity, 'engine_version':ENGINE_VERSION, 'source_sha256':source_digest(),
        'config':cfg, 'split':split, 'status':'failed' if failure else 'complete', 'failure':failure,
        'metrics':metrics, 'epochs':rows, 'steps':step, 'expected_steps':total_steps,
        'elapsed_seconds':elapsed, 'model_parameters':sum(p.numel() for p in params),
        'mean_intensity':float(np.mean(1-arr[:,2])) if len(arr) else None,
        'clipping_fraction':float(np.mean(arr[:,2] < 1-1e-7)) if len(arr) else None,
        'lower_bound_fraction':float(np.mean(arr[:,4])) if len(arr) else None,
        'environment':{'torch':torch.__version__, 'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name() if str(device).startswith('cuda') else str(device)}}
    atomic_json(result_path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = run(json.loads(Path(args.config).read_text()), args.output, args.data_dir, args.device)
    print(json.dumps({'output':args.output, 'status':result['status'], 'accuracy':result['metrics']['accuracy']}), flush=True)
