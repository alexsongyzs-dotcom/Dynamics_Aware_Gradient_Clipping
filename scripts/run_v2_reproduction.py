"""Explicit original V2 reproduction; fresh output identity and no hidden V1 defaults."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from src.train import train_run


def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if hasattr(value, 'item'): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/v2/original_fashion_cnn.json')
    p.add_argument('--output',required=True)
    p.add_argument('--data-dir',default='data')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(300,308)))
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text()); cfg.pop('name',None)
    if cfg['clipping'].get('exposure_target') is None:
        p.error('V2 requires an explicit exposure_target; None activates V1')
    for key, default in {'beta_a':.9,'norm_window':200}.items():
        if cfg['clipping'].get(key,default)!=default:
            p.error(f'{key} is fixed by the legacy training engine; use its default {default}')
    cfg.update(device=args.device,data_dir=str(Path(args.data_dir).resolve()))
    source_files=['src/clipping.py','src/train.py','src/data.py','src/models.py','src/dynamics.py']
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files}
    manifest={'config':cfg,'seeds':args.seeds,'source_sha256':hashes,
              'scope':'Original exploratory protocol; not validation-isolated confirmation'}
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    m=out/'manifest.json'
    if m.exists() and json.loads(m.read_text())!=manifest:
        raise ValueError('Output identity mismatch: choose a fresh output directory')
    if not m.exists() and any(out.iterdir()):
        raise ValueError('Nonempty output without manifest: choose a fresh directory')
    m.write_text(json.dumps(manifest,indent=2)+'\n')
    for seed in args.seeds:
        result_path=out/f'seed_{seed}.json'
        run_cfg=dict(cfg,seed=seed)
        if result_path.exists():
            saved=json.loads(result_path.read_text())
            if saved['config']!=run_cfg:raise ValueError('Saved run configuration mismatch')
            continue
        result=clean(train_run(run_cfg))
        temp=result_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'algorithm':'DAGC-V2','config':run_cfg,'result':result},indent=2,allow_nan=False)+'\n')
        temp.replace(result_path)
        print(json.dumps({'seed':seed,'test_acc':result['test_acc'],'diverged':result['diverged']}),flush=True)


if __name__=='__main__':main()
