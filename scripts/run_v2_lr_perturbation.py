"""Frozen paired LR perturbation test, independent seeds 1000--1007."""
import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_v2_mechanism import BASE,V2
from src.train import train_run
OUT=ROOT/'results'/'v2_lr_perturbation'
OUT.mkdir(parents=True,exist_ok=True)
def main():
 protocol=json.loads((ROOT/'results/v2_mechanism/frozen_protocol.json').read_text())
 policies={'v2':V2,'no_alignment':dict(V2,osc_weight=0.),'replay':dict(name='replay',schedule=protocol['schedule'])}
 conditions={'pulse':[dict(step=100,lr=1.6),dict(step=160,lr=0.8)],'drop':[dict(step=100,lr=0.2)]}
 (OUT/'protocol.json').write_text(json.dumps(dict(seeds=list(range(1000,1008)),conditions=conditions,base_lr=0.8,steps=300),indent=2))
 rows=[]
 for condition,changes in conditions.items():
  for label,policy in policies.items():
   for seed in range(1000,1008):
    p=OUT/f'{condition}_{label}_{seed}.json'
    if p.exists():r=json.loads(p.read_text())
    else:
     r=train_run(dict(BASE,seed=seed,clipping=policy,lr_changes=changes))
     p.write_text(json.dumps(r,allow_nan=True))
     print(f'{condition} {label} {seed}: {r["test_acc"]:.4f}',flush=True)
    rows.append(dict(condition=condition,policy=label,seed=seed,test_acc=r['test_acc'],diverged=int(r['diverged']),i_clip=r['i_clip']))
    with (OUT/'raw_runs.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for condition in conditions:
  ref={r['seed']:r['test_acc'] for r in rows if r['condition']==condition and r['policy']=='v2'}
  for label in policies:
   group=[r for r in rows if r['condition']==condition and r['policy']==label]
   a=np.array([r['test_acc']*100 for r in group]);d=np.array([(ref[r['seed']]-r['test_acc'])*100 for r in group])
   summary.append(dict(condition=condition,policy=label,mean=a.mean(),ci95=1.96*a.std(ddof=1)/np.sqrt(8),delta=d.mean(),delta_ci95=1.96*d.std(ddof=1)/np.sqrt(8),wins=int((d>0).sum())))
 with (OUT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print(summary,flush=True)
if __name__=='__main__':main()
