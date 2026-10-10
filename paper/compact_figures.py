from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
paper=Path(__file__).resolve().parent;repo=paper.parent if (paper.parent/'src').exists() else paper.parent/'Experiments for AI';figs=paper/'figures';plt.rcParams.update({'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.top':False,'axes.spines.right':False})
def ci(a):return t.ppf(.975,len(a)-1)*np.std(a,ddof=1)/np.sqrt(len(a))
def save(fig,name):
 fig.savefig(figs/(name+'.pdf'),bbox_inches='tight',pad_inches=.02);fig.savefig(figs/(name+'.png'),dpi=250,bbox_inches='tight',pad_inches=.02);plt.close(fig)
conds=[('Fashion CNN, LR .4','dagc_v2_evaluation_lr0.4'),('Fashion CNN, LR .8','dagc_v2_evaluation'),('Fashion MLP, LR .8','mlp_evaluation_lr0.8'),('MNIST CNN, LR .8','mnist_evaluation_lr0.8')]
fig,axes=plt.subplots(2,2,figsize=(3.5,2.75),layout='constrained');order=['none','fixed_tuned','agc_default','dagc_v1','dagc_v2']
for ax,(name,folder) in zip(axes.flat,conds):
 d=pd.read_csv(repo/'results'/folder/'raw_runs.csv');a=[100*np.where(d[d.policy==p].diverged.astype(bool),0,d[d.policy==p].test_acc) for p in order];ax.errorbar(range(5),[v.mean() for v in a],yerr=[ci(v) for v in a],fmt='o',ms=3,capsize=2,lw=.7);ax.set_xticks(range(5),['None','Fixed','AGC','V1','V2']);ax.set_ylim(-3,103);ax.set_title(name);ax.grid(axis='y',alpha=.2)
for ax in axes[:,0]:ax.set_ylabel('Accuracy (%)')
save(fig,'audited_v2_benchmarks')
d=pd.read_csv(repo/'results/v2_mechanism/raw_runs.csv');pv=d.pivot(index='seed',columns='policy',values='test_acc');pol=[p for p in pv if p!='v2'];fig,ax=plt.subplots(figsize=(3.5,2.1),layout='constrained');a=[100*(pv.v2-pv[p]).to_numpy() for p in pol];ax.errorbar([v.mean() for v in a],range(len(a)),xerr=[ci(v) for v in a],fmt='o',ms=3,capsize=2,lw=.7);ax.axvline(0,color='black',lw=.7);labels={'autoclip':'AutoClip (untuned)','fixed_matched':'Intensity-matched fixed','frozen':'Zero gain / moving bounds','no_alignment':'No alignment','no_exposure':'No exposure','rate_tracking':'Sign-control rule','replay':'Calibration replay'};ax.set_yticks(range(len(pol)),[labels[p] for p in pol]);ax.set_xlabel('V2 minus control (pp)');save(fig,'audited_mechanism')
fig,axes=plt.subplots(3,1,figsize=(3.5,2.85),sharex=True,layout='constrained')
for policy in ['v2','no_alignment','fixed_matched','replay']:
 logs=[json.loads((repo/f'results/v2_mechanism/eval_{policy}_{seed}.json').read_text())['log'] for seed in range(900,908)]
 for ax,key in zip(axes,['threshold','intensity','alignment']):
  a=np.stack([l[key] for l in logs]);m=a.mean(0);sd=a.std(0,ddof=1);ax.plot(m,label=policy.replace('_',' '),lw=.8);ax.fill_between(np.arange(len(m)),m-sd,m+sd,alpha=.1);ax.set_ylabel(key)
axes[0].set_yscale('log');axes[0].legend(fontsize=5.5,ncol=2,loc='upper right');axes[-1].set_xlabel('Training step');save(fig,'audited_trajectories')
p=paper/'main.tex';tex=p.read_text().replace('\\begin{figure}[t]','\\begin{figure}[!htbp]').replace('\\begin{figure}[h]','\\begin{figure}[!htbp]');p.write_text(tex,encoding='utf-8')
# Keep the regeneration script portable for both the local manuscript and GitHub paper directory.
s=paper/'audit_and_plot.py';x=s.read_text();x=x.replace("paper=repo.parent/'Paper_DAGC_ICML2027'","paper=Path(__file__).resolve().parent");s.write_text(x,encoding='utf-8')
print('Rebuilt 3 compact main-text figures')
