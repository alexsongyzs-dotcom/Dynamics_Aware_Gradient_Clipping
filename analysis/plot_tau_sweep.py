"""Exploratory reference-level sweep, stratified by gain (three seeds each).

Exposure, rather than accuracy, diagnoses whether the reference is tracked.
Overlapping intervals do not establish that accuracy is insensitive to tau.
"""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'results/dagc_v2_development/raw_runs.csv'
OUT=ROOT/'results/figures/dagc_v2'


def parse(df):
    df=df.copy()
    df['tau']=df.label.str.extract(r'target=([0-9.]+)',expand=False).astype(float)
    df['gamma']=df.label.str.extract(r'gamma=([0-9.]+)',expand=False).astype(float)
    df.loc[df.label=='dagc_v1',['tau','gamma']]=[1.,.05]
    return df


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df=parse(pd.read_csv(DEV))
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    summary=[]
    for gamma,color in ((.05,'#17668a'),(.1,'#b55439')):
        rows=[]
        for tau,group in df[df.label.str.startswith('dagc_v2') & (df.gamma==gamma)].groupby('tau'):
            assert group.seed.nunique()==len(group)==3
            values=100*group.test_acc.to_numpy()
            row=dict(gamma=gamma,tau=tau,n=len(group),accuracy=values.mean(),
                ci95=float(stats.t.ppf(.975,len(group)-1)*stats.sem(values)),
                exposure=group.f_clip.mean(),intensity=group.i_clip.mean())
            rows.append(row);summary.append(row)
        result=pd.DataFrame(rows)
        axes[0].errorbar(result.tau,result.accuracy,yerr=result.ci95,fmt='o-',capsize=3,
            color=color,label=f'V2, gain {gamma:g}')
        axes[1].plot(result.tau,result.exposure,'o-',color=color,label=f'V2, gain {gamma:g}')
    v1=df[df.label=='dagc_v1']
    axes[0].errorbar([1],[100*v1.test_acc.mean()],yerr=[100*stats.t.ppf(.975,len(v1)-1)*stats.sem(v1.test_acc)],
        fmt='s',color='#555555',capsize=3,label='V1 (gain 0.05)')
    axes[1].plot([1],[v1.f_clip.mean()],'s',color='#555555',label='V1')
    axes[1].plot([0,1],[0,1],'--',color='#999999',label='Reference = exposure')
    axes[0].set_ylabel('Accuracy (%)');axes[1].set_ylabel('Realized exposure')
    axes[0].set_title('Three seeds per setting; t intervals')
    axes[1].set_title('Exposure is near saturation for V2')
    for ax in axes:
        ax.set_xlabel('Reference exposure');ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'tau_sweep_accuracy.{ext}',dpi=220)
    plt.close(fig)
    pd.DataFrame(summary).to_csv(OUT/'tau_sweep_by_gain.csv',index=False)


if __name__=='__main__':main()
