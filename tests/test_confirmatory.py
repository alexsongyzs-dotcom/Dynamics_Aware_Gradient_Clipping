import json
import numpy as np
import pytest
import torch

from src import confirmatory as c
from src.cifar_resnet import CifarResNet20


def test_split_is_disjoint_complete_fixed():
    labels = torch.arange(10).repeat_interleave(20)
    a, b = c.stratified_split(labels, val_per_class=5)
    assert len(set(a.tolist()) & set(b.tolist())) == 0
    assert sorted(torch.cat([a,b]).tolist()) == list(range(200))
    assert torch.bincount(labels[b]).tolist() == [5] * 10
    assert torch.equal(a, c.stratified_split(labels, val_per_class=5)[0])


def test_policy_is_causal_and_raw_gradient_survives():
    p = torch.nn.Parameter(torch.ones(2))
    policy = c.Policy({'policy':'median', 'value':.1})
    p.grad = torch.tensor([3.,4.])
    flat = torch.cat([p.grad.flatten()])
    t0, alpha, _ = policy.apply([p], 5., 1., 0)
    assert t0 == .5
    assert alpha == .1
    assert torch.equal(flat, torch.tensor([3.,4.]))
    p.grad = torch.tensor([30.,40.])
    t1, _, _ = policy.apply([p], 50., 1., 1)
    assert t1 == .5  # current 50 cannot change its own threshold


def test_augmentation_preserves_shape_and_private_rng():
    x = torch.rand(8,3,32,32)
    before = torch.get_rng_state().clone()
    a = c.preprocess(x, 'cifar10', torch.Generator().manual_seed(6))
    b = c.preprocess(x, 'cifar10', torch.Generator().manual_seed(6))
    assert a.shape == x.shape and torch.equal(a,b)
    assert torch.equal(before, torch.get_rng_state())
    assert CifarResNet20()(a).shape == (8,10)


def test_epoch_resume_matches_uninterrupted_and_rejects_stale(tmp_path, monkeypatch):
    def fake_data(cfg, data_dir, device):
        rng = torch.Generator().manual_seed(71)
        x = torch.randn(24,1,28,28,generator=rng).to(device)
        y = torch.arange(24, device=device) % 10
        return (x,y), (x[:8],y[:8]), {'evaluation_split':'synthetic'}
    monkeypatch.setattr(c, 'load_data', fake_data)
    cfg = {'dataset':'fashion_mnist', 'seed':11, 'stage':'calibration', 'policy':'dagc',
           'split_seed':1729, 'epochs':3, 'batch_size':8, 'lr':.01, 'weight_decay':1e-4}
    full = c.run(cfg, tmp_path/'full', '', 'cpu')
    assert c.run(cfg, tmp_path/'resume', '', 'cpu', interrupt_after_epoch=1) is None
    resumed = c.run(cfg, tmp_path/'resume', '', 'cpu')
    assert full['metrics'] == resumed['metrics']
    assert full['epochs'] == resumed['epochs']
    with np.load(tmp_path/'full'/'trace.npz') as a, np.load(tmp_path/'resume'/'trace.npz') as b:
        for key in a.files:
            assert np.array_equal(a[key], b[key])
    with pytest.raises(ValueError, match='identity mismatch'):
        c.run(dict(cfg, lr=.1), tmp_path/'resume', '', 'cpu')
