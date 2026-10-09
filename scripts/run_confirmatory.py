"""Eight-device campaign: validation-only calibration, then frozen confirmation.

Run as `python -m scripts.run_confirmatory --root ... --data-dir ...`.
The manifest and source digest prevent accidental mixing of experiment versions.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
from src.confirmatory import atomic_json, digest, source_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--protocol', default='configs/lyx1/protocol.json')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    protocol = json.loads(Path(args.protocol).read_text())
    manifest = {'protocol':protocol, 'source_sha256':source_digest()}
    manifest['identity'] = digest(manifest)
    if (root/'manifest.json').exists():
        assert json.loads((root/'manifest.json').read_text()) == manifest, 'Campaign identity mismatch'
    atomic_json(root/'manifest.json', manifest)
    pool = Queue()
    for gpu in args.gpus.split(','):
        pool.put(gpu)
    def base(condition, stage, seed, policy, value=None):
        cfg = {k:v for k,v in condition.items() if k != 'name'}
        cfg.update(stage=stage, seed=seed, policy=policy, split_seed=protocol['split_seed'],
                   batch_size=protocol['batch_size'])
        if value is not None:
            cfg['value'] = value
        return cfg
    def task(condition, cfg):
        value = '' if 'value' not in cfg else '_'+str(cfg['value']).replace('.','p')
        name = f"{cfg['stage']}/{condition['name']}/{cfg['policy']}{value}/seed_{cfg['seed']}"
        directory = root/'raw'/name
        config = root/'configs'/Path(name+'.json')
        atomic_json(config, cfg)
        return cfg, directory, config
    def run_one(item):
        cfg, directory, config = item
        gpu = pool.get()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with (directory/'console.log').open('a') as stream:
                subprocess.run([sys.executable, '-m', 'src.confirmatory', '--config', str(config),
                    '--output', str(directory), '--data-dir', args.data_dir], env=env,
                    stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((directory/'result.json').read_text())
            print(json.dumps({'run':str(directory.relative_to(root)), 'status':result['status'],
                'accuracy':result['metrics']['accuracy'], 'seconds':result['elapsed_seconds']}), flush=True)
            return item, result
        finally:
            pool.put(gpu)
    def batch(items):
        with ThreadPoolExecutor(max_workers=pool.qsize()) as executor:
            return list(executor.map(run_one, items))
    calibrations = []
    for condition in protocol['conditions']:
        for policy, values in {**protocol['calibration_grid'], 'dagc':[None]}.items():
            for value in values:
                for seed in protocol['calibration_seeds']:
                    calibrations.append(task(condition, base(condition, 'calibration', seed, policy, value)))
    atomic_json(root/'status.json', {'phase':'calibration', 'runs':len(calibrations), 'start_unix':time.time()})
    batch(calibrations)
    selections, confirmations = {}, []
    for condition in protocol['conditions']:
        selected = {}
        for policy, values in protocol['calibration_grid'].items():
            scores = []
            for value in values:
                acc = []
                for seed in protocol['calibration_seeds']:
                    _, directory, _ = task(condition, base(condition, 'calibration', seed, policy, value))
                    result = json.loads((directory/'result.json').read_text())
                    acc.append(result['metrics']['accuracy'])
                scores.append({'value':value, 'mean_validation_accuracy':float(np.mean(acc)), 'seed_scores':acc})
            winner = max(scores, key=lambda x:x['mean_validation_accuracy'])  # grid order breaks ties
            selected[policy] = {'selected_value':winner['value'], 'candidates':scores}
        traces = []
        for seed in protocol['calibration_seeds']:
            _, directory, _ = task(condition, base(condition, 'calibration', seed, 'dagc'))
            result = json.loads((directory/'result.json').read_text())
            if result['status'] != 'complete':
                raise RuntimeError('Cannot construct a complete replay from failed DAGC calibration')
            traces.append(np.load(directory/'trace.npz')['threshold'])
        replay_path = root/'replay'/f"{condition['name']}.npy"
        replay_path.parent.mkdir(exist_ok=True)
        np.save(replay_path, np.mean(traces, axis=0))
        selected['replay_sha256'] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
        selections[condition['name']] = selected
        for policy in protocol['confirmation_policies']:
            for seed in protocol['confirmation_seeds']:
                cfg = base(condition, 'confirmation', seed, policy,
                           selected[policy]['selected_value'] if policy in selected else None)
                if policy == 'replay':
                    cfg.update(replay_path=str(replay_path), replay_sha256=selected['replay_sha256'])
                confirmations.append(task(condition, cfg))
    atomic_json(root/'selection.json', selections)
    atomic_json(root/'status.json', {'phase':'confirmation', 'runs':len(confirmations), 'start_unix':time.time()})
    batch(confirmations)
    atomic_json(root/'status.json', {'phase':'complete', 'calibration_runs':len(calibrations),
                                   'confirmation_runs':len(confirmations), 'completed_unix':time.time()})


if __name__ == '__main__':
    main()
