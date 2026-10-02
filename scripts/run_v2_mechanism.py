"""Disjoint calibration and eight-seed mechanism tests; resumable."""
import sys, json, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import train_run
OUT = ROOT / 'results' / 'v2_mechanism'
OUT.mkdir(parents=True, exist_ok=True)
BASE = dict(dataset='fashion_mnist', model='small_cnn', data_dir=str(ROOT / 'data'), epochs=15, batch_size=256, lr=0.8, momentum=0.9, weight_decay=0.0001, optimizer='sgd', device='cuda', num_workers=0, train_size=5000, test_size=2000, projection_dim=8, divergence_loss=100.0, divergence_jump=50.0)
V2 = dict(name='dagc', init_c=0.5, gamma=0.05, beta=0.9, relax=0.3, osc_weight=3.0, exposure_target=0.1)

def run(label, seed, policy, phase):
    p = OUT / f'{phase}_{label}_{seed}.json'
    if p.exists():
        return json.loads(p.read_text())
    r = train_run(dict(BASE, seed=seed, clipping=policy))
    p.write_text(json.dumps(r, allow_nan=True))
    print(f'{phase} {label} seed={seed}: {r['test_acc']:.4f} intensity={r['i_clip']:.3f}', flush=True)
    return r

def main():
    cal = [run('v2', s, V2, 'cal') for s in (800, 801, 802)]
    schedule = np.mean([r['log']['threshold'] for r in cal], axis=0).tolist()
    target = float(np.mean([r['i_clip'] for r in cal]))
    fixed = {}
    for c in (0.05, 0.1, 0.2, 0.3, 0.5):
        rs = [run(f'fixed{c}', s, dict(name='fixed', threshold=c), 'cal') for s in (800, 801, 802)]
        fixed[c] = abs(np.mean([r['i_clip'] for r in rs]) - target)
    matched = min(fixed, key=fixed.get)
    (OUT / 'frozen_protocol.json').write_text(json.dumps(dict(calibration_seeds=[800, 801, 802], evaluation_seeds=list(range(900, 908)), matched_threshold=matched, calibration_v2_intensity=target, matching_errors=fixed, schedule=schedule), indent=2))
    policies = {'v2': V2, 'no_alignment': dict(V2, osc_weight=0.0), 'no_exposure': dict(V2, relax=0.0), 'frozen': dict(V2, gamma=0.0), 'fixed_matched': dict(name='fixed', threshold=matched), 'replay': dict(name='replay', schedule=schedule), 'autoclip': dict(name='autoclip', quantile=0.1), 'rate_tracking': dict(name='rate_tracking', init_c=0.5)}
    rows = []
    for label, policy in policies.items():
        for seed in range(900, 908):
            r = run(label, seed, policy, 'eval')
            rows.append(dict(policy=label, seed=seed, test_acc=r['test_acc'], diverged=int(r['diverged']), i_clip=r['i_clip'], f_clip=r['f_clip']))
            with (OUT / 'raw_runs.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    summary = []
    paired = {(row['policy'], row['seed']): row['test_acc'] for row in rows}
    for label in policies:
        group = [row for row in rows if row['policy'] == label]
        accuracy = np.array([row['test_acc'] * 100 for row in group])
        delta = np.array([(paired['v2', row['seed']] - row['test_acc']) * 100 for row in group])
        summary.append(dict(policy=label, mean=accuracy.mean(), ci95=1.96*accuracy.std(ddof=1)/np.sqrt(8), intensity=np.mean([row['i_clip'] for row in group]), delta=delta.mean(), delta_ci95=1.96*delta.std(ddof=1)/np.sqrt(8), wins=int((delta > 0).sum())))
    with (OUT / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

if __name__ == '__main__':
    main()
