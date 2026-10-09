"""Export complete, compact evidence with checksums; checkpoints remain in place."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from src.confirmatory import atomic_json, digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root,out=Path(args.root),Path(args.output)
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='complete':
        raise RuntimeError('Refusing to export an incomplete campaign as final evidence')
    manifest=json.loads((root/'manifest.json').read_text())
    protocol=manifest['protocol']
    selection=json.loads((root/'selection.json').read_text())
    expected=status['calibration_runs']+status['confirmation_runs']
    paths=sorted((root/'raw').rglob('result.json'))
    assert len(paths)==expected
    if out.exists():
        raise FileExistsError('Evidence exports are immutable; choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    records,checks,curves=[],[],[]
    pairs=set()
    split_hashes={}
    for path in paths:
        run=json.loads(path.read_text())
        cfg=run['config']
        identity=digest({'config':cfg,'source':manifest['source_sha256'],'version':run['engine_version']})
        assert run['identity']==identity
        assert run['source_sha256']==manifest['source_sha256']
        assert run['split']['evaluation_split']==('official_test' if cfg['stage']=='confirmation' else 'training_validation')
        condition=path.relative_to(root/'raw').parts[1]
        expected_condition=next(c for c in protocol['conditions'] if c['name']==condition)
        assert all(cfg[k]==v for k,v in expected_condition.items() if k!='name')
        assert cfg['batch_size']==protocol['batch_size'] and cfg['split_seed']==protocol['split_seed']
        assert cfg['seed'] in protocol[cfg['stage']+'_seeds']
        assert run['split']['validation_n']==5000
        assert run['split']['evaluation_n']==(10000 if cfg['stage']=='confirmation' else 5000)
        assert run['split']['train_n']==(55000 if cfg['dataset']=='fashion_mnist' else 45000)
        assert run['expected_steps']==math.ceil(run['split']['train_n']/cfg['batch_size'])*cfg['epochs']
        split_hashes.setdefault(cfg['dataset'],run['split']['split_sha256'])
        assert split_hashes[cfg['dataset']]==run['split']['split_sha256']
        key=(cfg['stage'],condition,cfg['policy'],cfg.get('value'),cfg['seed'])
        assert key not in pairs
        pairs.add(key)
        if run['status']=='complete':
            assert run['steps']==run['expected_steps']
            assert [r['epoch'] for r in run['epochs']]==list(range(1,cfg['epochs']+1))
        else:
            assert run['metrics']['accuracy']==0. and run['failure']
        run['condition']=condition
        if 'replay_path' in cfg:
            cfg['replay_path']='replay/'+Path(cfg['replay_path']).name
            run['export_note']='Replay path made relative; original identity verified before export.'
        records.append(run)
        trace=path.with_name('trace.npz')
        checks.append({'run':str(path.parent.relative_to(root)),
            'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
            'identity':run['identity']})
        if cfg['stage']=='confirmation':
            with np.load(trace) as a:
                assert all(len(a[k])==run['steps'] for k in a.files)
                for b,indices in enumerate(np.array_split(np.arange(run['steps']),100)):
                    if not len(indices):continue
                    row={'condition':condition,'policy':cfg['policy'],'seed':cfg['seed'],
                         'bin':b,'step':float(indices.mean())}
                    for k in a.files:
                        values=a[k][indices]
                        row[k]=float(np.nanmean(values)) if np.isfinite(values).any() else None
                    curves.append(row)
    for condition in protocol['conditions']:
        name=condition['name']
        for policy,values in {**protocol['calibration_grid'],'dagc':[None]}.items():
            for value in values:
                actual={r['config']['seed'] for r in records if r['condition']==name and
                    r['config']['stage']=='calibration' and r['config']['policy']==policy and
                    r['config'].get('value')==value}
                assert actual==set(protocol['calibration_seeds'])
        for policy,values in protocol['calibration_grid'].items():
            scores=[]
            for value in values:
                group=[r['metrics']['accuracy'] for r in records if r['condition']==name and
                    r['config']['stage']=='calibration' and r['config']['policy']==policy and
                    r['config']['value']==value]
                scores.append(float(np.mean(group)))
            winner=values[int(np.argmax(scores))]
            assert selection[name][policy]['selected_value']==winner
        for policy in protocol['confirmation_policies']:
            group=[r for r in records if r['condition']==name and
                    r['config']['policy']==policy and r['config']['stage']=='confirmation']
            actual={r['config']['seed'] for r in group}
            assert actual==set(protocol['confirmation_seeds'])
            if policy in protocol['calibration_grid']:
                assert all(r['config']['value']==selection[name][policy]['selected_value'] for r in group)
        replay=root/'replay'/f'{name}.npy'
        calibration_traces=[np.load(p.with_name('trace.npz'))['threshold'] for p in paths
            if p.relative_to(root/'raw').parts[:3]==('calibration',name,'dagc')]
        assert np.array_equal(np.load(replay),np.mean(calibration_traces,axis=0))
        assert hashlib.sha256(replay.read_bytes()).hexdigest()==selection[name]['replay_sha256']
    (out/'seed_results.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    (out/'trace_bins.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in curves))
    for name in ('manifest.json','selection.json','status.json','calibration_carryover.json'):
        if (root/name).exists():shutil.copy2(root/name,out/name)
    shutil.copytree(root/'replay',out/'replay',dirs_exist_ok=True)
    atomic_json(out/'audit.json',{'expected_runs':expected,'verified_runs':len(records),
        'calibration_runs':sum(r['config']['stage']=='calibration' for r in records),
        'confirmation_runs':sum(r['config']['stage']=='confirmation' for r in records),
        'complete_runs':sum(r['status']=='complete' for r in records),
        'scientific_failures':sum(r['status']=='failed' for r in records),
        'identity_verified':True,'epoch_continuity_verified':True,'pair_completeness_verified':True,
        'validation_selection_verified':True,'replay_construction_verified':True,
        'artifacts':checks})
    print(json.dumps({'output':str(out),'runs':len(records)}))


if __name__=='__main__':main()
