from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,subprocess,re,shutil
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path.cwd(); paper=Path(__file__).resolve().parent; out=paper/'audit_20261010';out.mkdir(exist_ok=True); figs=paper/'figures'
backup=paper/'backups'/'before_figures_audit_20261010';backup.mkdir(parents=True,exist_ok=True)
for name in ['main.tex','main.pdf']:
 if not (backup/name).exists(): shutil.copy2(paper/name,backup/name)
script=paper/'audit_and_plot.py'
def ci(a):
 a=np.asarray(a,dtype=float);return t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan
inventory=[]; stats=[];warnings=[]
paths=list((repo/'results').rglob('raw_runs.csv'))+[repo/'results/unstable_band/eval_raw.csv',repo/'results/unstable_band/calibration_raw.csv']
for p in paths:
 d=pd.read_csv(p); keys=[c for c in ['suite','condition','dataset','model','optimizer','policy','label','family','lr','seed'] if c in d]
 dup=int(d.duplicated(keys).sum()); bad=int((~d.test_acc.between(0,1)).sum()) if 'test_acc' in d else 0
 failures=int(d.diverged.fillna(0).astype(float).sum()) if 'diverged' in d else None
 inventory.append(dict(path=str(p.relative_to(repo)),rows=len(d),duplicates=dup,invalid_accuracy=bad,reported_divergences=failures,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),failure_record='present' if failures is not None else 'missing; cannot independently certify no failures'))
 if dup or bad: warnings.append(f'{p}: duplicates={dup}, invalid_accuracy={bad}')
 if failures is None: warnings.append(f'{p.relative_to(repo)}: failure flag missing; no-failure claim is not verified')
 group=[c for c in keys if c!='seed']
 for key,g in d.groupby(group,dropna=False):
  if 'test_acc' not in g: continue
  key=key if isinstance(key,tuple) else (key,)
  stats.append(dict(path=str(p.relative_to(repo)),group=str(dict(zip(group,key))),n=len(g),seeds=','.join(map(str,sorted(g.seed.unique()))),mean=100*g.test_acc.mean(),ci95_t=ci(100*g.test_acc),mean_failed_zero=100*np.where(g.diverged.astype(bool),0,g.test_acc).mean() if 'diverged' in g else np.nan))
 # Cross-check summaries without overwriting original historical values.
 summary=p.with_name('summary.csv')
 if summary.exists() and 'test_acc_mean' in pd.read_csv(summary):
  s=pd.read_csv(summary); join=[c for c in group if c in s and c not in ['suite']]; recom=d.groupby(join,dropna=False).test_acc.mean().reset_index(name='recomputed'); merged=s.merge(recom,on=join); err=float(abs(merged.test_acc_mean-merged.recomputed).max());print(p.parent.name,'summary maximum mean discrepancy',err)
json_checks=[]
for folder in ['v2_mechanism','v2_lr_perturbation']:
 d=pd.read_csv(repo/'results'/folder/'raw_runs.csv')
 for _,r in d.iterrows():
  name=f"eval_{r.policy}_{int(r.seed)}.json" if folder=='v2_mechanism' else f"{r.condition}_{r.policy}_{int(r.seed)}.json"
  p=repo/'results'/folder/name
  if not p.exists():warnings.append('Missing JSON '+str(p));continue
  j=json.loads(p.read_text());ok=np.isclose(float(j['test_acc']),float(r.test_acc)) and bool(j['diverged'])==bool(r.diverged)
  json_checks.append(dict(path=str(p.relative_to(repo)),matches_csv=bool(ok),steps=j.get('steps_completed'),planned=j.get('steps_planned')))
  if not ok:warnings.append('JSON/CSV mismatch '+str(p))
pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False);pd.DataFrame(stats).to_csv(out/'recomputed_statistics.csv',index=False);pd.DataFrame(json_checks).to_csv(out/'json_crosschecks.csv',index=False)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');fig.savefig(figs/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
# Complete V2 exploration, all policies including no clipping and failures.
conds=[('Fashion CNN, LR .4','dagc_v2_evaluation_lr0.4'),('Fashion CNN, LR .8','dagc_v2_evaluation'),('Fashion MLP, LR .8','mlp_evaluation_lr0.8'),('MNIST CNN, LR .8','mnist_evaluation_lr0.8')]
fig,axes=plt.subplots(1,4,figsize=(12,3.4)); order=['none','fixed_tuned','agc_default','dagc_v1','dagc_v2'];labels=['None','Fixed','AGC','V1','V2']
for ax,(label,folder) in zip(axes,conds):
 d=pd.read_csv(repo/'results'/folder/'raw_runs.csv');means=[];errs=[]
 for policy in order:
  g=d[d.policy==policy];a=100*np.where(g.diverged.astype(bool),0,g.test_acc);means.append(a.mean());errs.append(ci(a))
 ax.errorbar(range(5),means,yerr=errs,fmt='o',capsize=3);ax.set_xticks(range(5),labels,rotation=30);ax.set_title(label);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Accuracy (%); failed runs = 0');fig.suptitle('Exploratory V2 evaluation: 8 seeds, Student-t 95% intervals');fig.tight_layout();save(fig,'audited_v2_benchmarks')
# Original A4 all eight settings, no claim that V1 always wins.
d=pd.read_csv(repo/'results/a4_multiseed_small/raw_runs.csv');fig,axes=plt.subplots(2,4,figsize=(12,6));
for ax,((ds,model,opt),g) in zip(axes.flat,d.groupby(['dataset','model','optimizer'])):
 a=[100*g[g.policy==p].test_acc.to_numpy() for p in ['none','fixed','agc','dagc']];ax.errorbar(range(4),[v.mean() for v in a],yerr=[ci(v) for v in a],fmt='o',capsize=3);ax.set_xticks(range(4),['None','Fixed','AGC','V1'],rotation=25);ax.set_title(f'{ds} / {model} / {opt}',fontsize=8);ax.set_ylabel('Accuracy (%)');ax.grid(axis='y',alpha=.2)
fig.suptitle('Original A4: DAGC-V1, 5 seeds per policy; Student-t 95% intervals');fig.tight_layout();save(fig,'audited_a4_v1')
d=pd.read_csv(repo/'results/a3_ablation_small/raw_runs.csv');fig,ax=plt.subplots(figsize=(7,3));order=['full_dagc','no_exposure_feedback','no_alignment_feedback','fixed_threshold','fixed_global'];a=[100*d[d.condition==p].test_acc.to_numpy() for p in order];ax.errorbar(range(5),[v.mean() for v in a],yerr=[ci(v) for v in a],fmt='o',capsize=4);ax.set_xticks(range(5),['Full V1','No exposure','No alignment','Zero gain*','Fixed global']);ax.set_ylabel('Accuracy (%)');ax.set_title('Original A3: 5 seeds; *moving bounds retained');fig.tight_layout();save(fig,'audited_a3_v1')
# Paired mechanism effects.
d=pd.read_csv(repo/'results/v2_mechanism/raw_runs.csv');pivot=d.pivot(index='seed',columns='policy',values='test_acc');policies=[p for p in pivot if p!='v2'];fig,ax=plt.subplots(figsize=(7,3.8));a=[100*(pivot.v2-pivot[p]).to_numpy() for p in policies];ax.errorbar([v.mean() for v in a],range(len(a)),xerr=[ci(v) for v in a],fmt='o',capsize=3);ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(a)),[p.replace('frozen','Zero gain (moving bounds)').replace('_',' ') for p in policies]);ax.set_xlabel('V2 minus control (percentage points), paired 95% t interval');fig.tight_layout();save(fig,'audited_mechanism')
# Multi-seed behavior from logs, no cherry-picked seed.
fig,axes=plt.subplots(1,3,figsize=(11,3));tracechecks=[]
for policy in ['v2','no_alignment','fixed_matched','replay']:
 logs=[json.loads((repo/f'results/v2_mechanism/eval_{policy}_{seed}.json').read_text())['log'] for seed in range(900,908)]
 for ax,key in zip(axes,['threshold','intensity','alignment']):
  arrays=[np.asarray(l[key],dtype=float) for l in logs];lens=[len(a) for a in arrays];assert len(set(lens))==1;stack=np.stack(arrays);ax.plot(stack.mean(0),label=policy);ax.fill_between(np.arange(stack.shape[1]),stack.mean(0)-stack.std(0,ddof=1),stack.mean(0)+stack.std(0,ddof=1),alpha=.1);ax.set_title(key);ax.set_xlabel('Training step')
 tracechecks.append(dict(policy=policy,seeds=8,steps=lens[0]))
axes[0].set_yscale('log');axes[0].legend(fontsize=7);fig.suptitle('Exploratory mechanism logs: mean +/- 1 SD across 8 seeds');fig.tight_layout();save(fig,'audited_trajectories')
# Reference sweep: stratify by gain, three seeds per cell.
d=pd.read_csv(repo/'results/dagc_v2_development/raw_runs.csv');v=d[d.label.str.contains('target=')].copy();v['tau']=v.label.str.extract(r'target=([\d.]+)').astype(float);v['gain']=v.label.str.extract(r'gamma=([\d.]+)').astype(float);fig,axes=plt.subplots(1,3,figsize=(10,3))
for gain,g in v.groupby('gain'):
 for ax,col in zip(axes,['test_acc','f_clip','i_clip']):
  s=g.groupby('tau')[col];ax.errorbar(list(s.groups),s.mean()*(100 if col=='test_acc' else 1),yerr=[ci(x)*(100 if col=='test_acc' else 1) for _,x in s],label=f'gain {gain}',capsize=3);ax.set_xlabel('Reference exposure');ax.set_title(col)
axes[0].legend();fig.suptitle('Development sweep: 3 seeds per gain/reference; 95% t intervals');fig.tight_layout();save(fig,'audited_reference_sweep')
# Perturbations paired, preserve replay null.
d=pd.read_csv(repo/'results/v2_lr_perturbation/raw_runs.csv');fig,axes=plt.subplots(1,2,figsize=(7,3))
for ax,(condition,g) in zip(axes,d.groupby('condition')):
 p=g.pivot(index='seed',columns='policy',values='test_acc');pol=[c for c in p if c!='v2'];a=[100*(p.v2-p[c]).to_numpy() for c in pol];ax.errorbar(range(len(a)),[x.mean() for x in a],yerr=[ci(x) for x in a],fmt='o',capsize=4);ax.set_xticks(range(len(a)),pol,rotation=20);ax.axhline(0,color='black',lw=.8);ax.set_title(condition);ax.set_ylabel('V2 minus control (pp)')
fig.suptitle('LR perturbations: paired 95% t intervals, 8 seeds');fig.tight_layout();save(fig,'audited_perturbations')
report='''# Experiment audit, 2026-10-10

Scope: all located legacy raw CSV tables, cross-checks of mechanism and perturbation CSV against per-run JSON, and archived 156-run screening. Remote develop/lyx1@050beca still contains no complete confirmatory evidence export; server execution has not been independently inspected.

Corrections: A3/A4 are V1, not V2. Historical normal-approximation intervals replaced with Student-t intervals in regenerated figures. V2 plot includes all four conditions and no-clipping. Failed V2 runs are zero in the regenerated benchmark, with observed-score and failure-zero means both retained in statistics. Legacy tuning reused official test data. Zero gain retains moving bounds. Gain/reference sweep has three seeds per cell, never six pooled replicates. Unverified ZClip/AdaGC reproductions are excluded from manuscript efficacy claims.

Limitations: historical A3/A4 raw tables lack failure flags and complete per-run identity/configuration records. CSV consistency does not certify scientific completeness. Existing summary statistics were not overwritten. Mechanism trajectories show between-seed SD, not confidence intervals or causal evidence. Full server traces/checkpoints and ongoing confirmation cannot be audited from the committed files.

Files: inventory.csv (source hashes, duplicates, bounds and failures), recomputed_statistics.csv (seed-level recomputation), json_crosschecks.csv (mechanism/perturbation source consistency).

Warnings:\n'''+ '\n'.join('- '+x for x in warnings)
(out/'audit_report.md').write_text(report,encoding='utf-8');print('Audited tables:',len(inventory),'JSON matches:',sum(x['matches_csv'] for x in json_checks),'/',len(json_checks));print('Warnings',warnings)
tex=(paper/'main.tex').read_text(); inserts=r'''
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/audited_v2_benchmarks.pdf}
\caption{Complete exploratory V2 evaluations over eight seeds per policy. Bars are Student-$t$ 95\% intervals. Failed runs contribute zero accuracy. All four conditions and no-clipping outcomes are retained; historical test-data reuse prevents calling these clean held-out confirmation. AGC and fixed settings follow the historical protocol, not the new equally budgeted tuning.}
\label{fig:legacy-v2}
\end{figure*}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/audited_mechanism.pdf}
\caption{Exploratory paired mechanism effects, eight seeds. Positive differences favor V2. Replay and no-alignment intervals include zero; the zero-gain control still retains moving bounds. Intervals are descriptive, without multiplicity correction.}
\label{fig:legacy-mechanism}
\end{figure}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/audited_trajectories.pdf}
\caption{Mean threshold, clipping intensity, and raw-gradient alignment over eight exploratory seeds; bands show one between-seed standard deviation. These traces describe behavior and do not identify causal signal value.}
\label{fig:legacy-traces}
\end{figure*}
'''
tex=tex.replace('\\subsection{Completed validation screening: the predictive gate did not pass}',('' if 'fig:legacy-v2' in tex else inserts)+'\n\\subsection{Completed validation screening: the predictive gate did not pass}')
extra=r'''
\clearpage
\section{Original V1 Benchmarks and Additional Exploratory Figures}
The initial A3 and A4 studies used V1 with initial threshold 1.0; they must
not be presented as V2 evaluations. A3 has five seeds per variant and A4
has five seeds per policy in each of eight dataset/model/optimizer settings.
V1 does not uniformly outperform the no-clipping baseline. Their raw tables
lack explicit failure flags, so the figures report the recorded scores rather
than asserting that all runs were failure-free. Historical normal-approximation
error bars are replaced here with Student-$t$ intervals. Complete audit records
and source hashes accompany this manuscript in \texttt{audit\_20261010/}.
\begin{figure}[h]
\centering\includegraphics[width=.85\textwidth]{figures/audited_a3_v1.pdf}
\caption{Original V1 ablations, five seeds per variant. Zero gain retains moving bounds; it is distinct from fixed global clipping.}
\end{figure}
\begin{figure}[h]
\centering\includegraphics[width=\textwidth]{figures/audited_a4_v1.pdf}
\caption{All eight original A4 settings, five seeds per policy. Error bars are Student-$t$ 95\% intervals and scores are exploratory.}
\end{figure}
\clearpage
\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{figures/audited_reference_sweep.pdf}
\caption{Development reference sweep stratified by controller gain. Each point has three seeds; gains are not pooled. Realized exposure differs from the reference and cannot be interpreted as target-rate tracking.}
\end{figure}
\begin{figure}[h]
\centering\includegraphics[width=.85\textwidth]{figures/audited_perturbations.pdf}
\caption{Exploratory learning-rate perturbations, eight paired seeds. V2-minus-replay intervals span zero; nonsignificance is not equivalence.}
\end{figure}
Single-seed smoke runs and high-learning-rate pilot selection are retained as
development artifacts, not replication evidence. The unstable-band tables
include legacy comparator implementations whose fidelity has not been verified;
they are not used to establish superiority over published ZClip or AdaGC.
'''
tex=tex.replace('\\end{document}',('' if 'Original V1 Benchmarks and Additional Exploratory Figures' in tex else extra)+'\n\\end{document}');(paper/'main.tex').write_text(tex,encoding='utf-8')
