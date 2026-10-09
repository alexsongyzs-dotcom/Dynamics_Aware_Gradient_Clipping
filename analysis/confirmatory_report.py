"""Regenerate all confirmatory paper tables and figures from the exported evidence."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.statistics import mean_ci,paired_difference,holm_adjust

LABELS={'none':'No clipping','fixed':'Tuned fixed','agc':'Tuned AGC','autoclip':'Tuned AutoClip',
        'median':'Tuned median scale','dagc':'DAGC-V2','no_alignment':'No alignment','replay':'Trace replay'}
CONDITIONS={'fashion_sgd_0p8':'FashionMNIST / 0.8','cifar_sgd_0p1':'CIFAR-10 / 0.1','cifar_sgd_0p4':'CIFAR-10 / 0.4'}


def cell(values):
    m,lo,hi=mean_ci(values)
    return f'${m:.2f}\\pm{(hi-lo)/2:.2f}$'



def write_table(path, columns, header, rows):
    path.write_text('\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')


def legacy_tables(destination):
    """Recompute development intervals without pooling gains as replicates."""
    development=pd.read_csv('results/dagc_v2_development/raw_runs.csv')
    dev=[]
    for label,frame in development.groupby('label',sort=False):
        if not label.startswith('dagc_v2@'):continue
        opts=dict(item.split('=') for item in label.split('@')[1].split(','))
        m,lo,hi=mean_ci(100*frame.test_acc.to_numpy())
        dev.append({'tau':float(opts['target']),'gamma':float(opts['gamma']),
                    'n':len(frame),'mean':m,'lower':lo,'upper':hi,
                    'exposure':float(frame.f_clip.mean()),'intensity':float(frame.i_clip.mean())})
    pd.DataFrame(dev).to_csv(destination/'legacy_development_by_gain.csv',index=False)
    rows=[]
    for r in sorted(dev,key=lambda r:(r['gamma'],r['tau'])):
        rows.append(f"{r['gamma']:.2f} & {r['tau']:.2f} & {r['mean']:.2f} $\\pm$ {(r['upper']-r['lower'])/2:.2f} & {r['exposure']:.3f} & {r['intensity']:.3f} \\\\")
    write_table(destination/'legacy_development.tex', 'rrrrr', r'Gain $\gamma$ & Reference $\tau$ & Accuracy (\%), 95\% interval & Exposure & Intensity', rows)
    mech=pd.read_csv('results/v2_mechanism/raw_runs.csv')
    names={'v2':'DAGC-V2','no_alignment':'No alignment','no_exposure':'No exposure',
           'frozen':'Zero controller gain','fixed_matched':'Intensity-matched fixed',
           'replay':'Trace replay','autoclip':'AutoClip (10th percentile)','rate_tracking':'Rate tracking'}
    rows=[]
    paired=[]
    ref=mech[mech.policy=='v2'].set_index('seed').test_acc.sort_index()
    for policy,frame in mech.groupby('policy',sort=False):
        values=frame.set_index('seed').test_acc.reindex(ref.index)
        delta=paired_difference(100*ref.to_numpy(),100*values.to_numpy())
        paired.append({'policy':policy,**delta})
        rows.append(f"{names[policy]} & {cell(100*values.to_numpy())} & {delta['mean']:+.2f} & [{delta['lower']:+.2f}, {delta['upper']:+.2f}] \\\\")
    write_table(destination/'legacy_mechanism.tex', 'lrrr', r'Policy & Accuracy (\%) & $\Delta$ & Paired 95\% interval', rows)
    pd.DataFrame(paired).to_csv(destination/'legacy_mechanism_paired.csv',index=False)
    perturb=pd.read_csv('results/v2_lr_perturbation/raw_runs.csv')
    rows=[]
    for condition,frame in perturb.groupby('condition'):
        pivot=frame.pivot(index='seed',columns='policy',values='test_acc')
        for policy in ('no_alignment','replay'):
            d=paired_difference(100*pivot.v2.to_numpy(),100*pivot[policy].to_numpy())
            rows.append(f"{condition.title()} & {names[policy]} & {d['mean']:+.2f} & [{d['lower']:+.2f}, {d['upper']:+.2f}] \\\\")
    write_table(destination/'legacy_perturbations.tex', 'llrr', r'Perturbation & Comparator & Difference & Paired 95\% interval', rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',default='results/lyx1/evidence')
    p.add_argument('--paper',default='paper')
    p.add_argument('--legacy-only',action='store_true')
    args=p.parse_args()
    root,paper=Path(args.evidence),Path(args.paper)
    tables=paper/'generated';tables.mkdir(parents=True,exist_ok=True)
    legacy_tables(tables)
    if args.legacy_only:return
    manifest=json.loads((root/'manifest.json').read_text()); protocol=manifest['protocol']
    audit=json.loads((root/'audit.json').read_text())
    assert audit['verified_runs']==405 and audit['pair_completeness_verified']
    records=[json.loads(line) for line in (root/'seed_results.jsonl').read_text().splitlines()]
    flat=[]
    for r in records:
        cfg=r['config']
        flat.append({'condition':r['condition'],'stage':cfg['stage'],'policy':cfg['policy'],
            'seed':cfg['seed'],'value':cfg.get('value'),'accuracy':100*r['metrics']['accuracy'],
            'failure':r['status']=='failed','intensity':r['mean_intensity'],
            'frequency':r['clipping_fraction'],'lower_bound':r['lower_bound_fraction'],
            'seconds':r['elapsed_seconds'],'steps':r['steps']})
    df=pd.DataFrame(flat);df.to_csv(root.parent/'all_runs.csv',index=False)
    final=df[df.stage=='confirmation']
    primary,secondary,summary=[],[],[]
    for condition in protocol['conditions']:
        name=condition['name'];frame=final[final.condition==name]
        pivot=frame.pivot(index='seed',columns='policy',values='accuracy').sort_index()
        assert pivot.index.tolist()==protocol['confirmation_seeds'] and not pivot.isna().any().any()
        for policy in protocol['confirmation_policies']:
            m,lo,hi=mean_ci(pivot[policy].to_numpy())
            subset=frame[frame.policy==policy]
            summary.append({'condition':name,'policy':policy,'n':len(subset),'mean':m,'lower':lo,
                'upper':hi,'failures':int(subset.failure.sum()),'intensity':float(subset.intensity.mean()),
                'frequency':float(subset.frequency.mean()),'lower_bound':float(subset.lower_bound.mean()),
                'seconds':float(subset.seconds.mean())})
            if policy=='dagc':continue
            d={'condition':name,'comparator':policy,**paired_difference(pivot.dagc.to_numpy(),
                pivot[policy].to_numpy(),protocol['equivalence_margin_percentage_points'])}
            (primary if policy in protocol['primary_comparisons'] else secondary).append(d)
    for r,pvalue in zip(primary,holm_adjust([r['sign_flip_p'] for r in primary])):
        r['holm_p']=float(pvalue)
    pd.DataFrame(summary).to_csv(root.parent/'summary.csv',index=False)
    pd.DataFrame(primary).to_csv(root.parent/'primary_comparisons.csv',index=False)
    pd.DataFrame(secondary).to_csv(root.parent/'secondary_comparisons.csv',index=False)
    (root.parent/'analysis.json').write_text(json.dumps({'summary':summary,'primary':primary,'secondary':secondary,'superiority_criterion_met':all(r['mean']>0 and r['holm_p']<.05 for r in primary)},indent=2)+'\n')
    rows=[]
    for policy in protocol['confirmation_policies']:
        items=[LABELS[policy]]
        for c in protocol['conditions']:
            r=next(r for r in summary if r['condition']==c['name'] and r['policy']==policy)
            items.append(f"${r['mean']:.2f}\\pm{(r['upper']-r['lower'])/2:.2f}$")
        rows.append(' & '.join(items)+r' \\')
    write_table(tables/'accuracy.tex', 'lccc', r'Policy & FashionMNIST, LR 0.8 & CIFAR-10, LR 0.1 & CIFAR-10, LR 0.4', rows)
    rows=[]
    for d in primary:
        rows.append(f"{CONDITIONS[d['condition']]} & {LABELS[d['comparator']]} & {d['mean']:+.2f} & [{d['lower']:+.2f}, {d['upper']:+.2f}] & {d['wins']}/{d['n']} & {d['holm_p']:.4f} \\\\")
    write_table(tables/'paired.tex', 'llrrrr', r'Condition / initial LR & Comparator & Difference & 95\% interval & Wins & $p_H$', rows)
    rows=[]
    for r in summary:
        rows.append(f"{CONDITIONS[r['condition']]} & {LABELS[r['policy']]} & {r['failures']}/{r['n']} & {r['frequency']:.3f} & {r['intensity']:.3f} & {r['lower_bound']:.3f} \\\\")
    write_table(tables/'diagnostics.tex', 'llrrrr', r'Condition / initial LR & Policy & Failed & $f$ & $I$ & $b$', rows)
    selection=json.loads((root/'selection.json').read_text());rows=[]
    for c in protocol['conditions']:
        for policy in protocol['calibration_grid']:
            r=selection[c['name']][policy]
            scores=', '.join(f"{x['value']:g}: {100*x['mean_validation_accuracy']:.2f}" for x in r['candidates'])
            rows.append(f"{CONDITIONS[c['name']]} & {LABELS[policy]} & {scores} & {r['selected_value']:g} \\\\")
    write_table(tables/'calibration.tex', 'llll', r'Condition / initial LR & Policy & Candidate scores & Selected', rows)
    plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(6.75,2.8),sharey=True)
    comparators=protocol['primary_comparisons']+['no_alignment']
    for ax,c in zip(axes,protocol['conditions']):
        for i,policy in enumerate(comparators):
            d=next(r for r in primary+secondary if r['condition']==c['name'] and r['comparator']==policy)
            ax.errorbar(d['mean'],i,xerr=[[d['mean']-d['lower']],[d['upper']-d['mean']]],
                fmt='o',capsize=3,color='#17668a' if d['mean']>=0 else '#b55439')
        ax.axvspan(-.5,.5,color='#dddddd',alpha=.6)
        ax.axvline(0,color='black',linewidth=.7)
        ax.set_title(CONDITIONS[c['name']]);ax.set_xlabel('DAGC minus comparator (pp)')
    axes[0].set_yticks(range(len(comparators)),[LABELS[p] for p in comparators]);axes[0].invert_yaxis()
    fig.tight_layout();fig.savefig(paper/'figures/lyx1_paired.pdf');plt.close(fig)
    curves=pd.read_json(root/'trace_bins.jsonl',lines=True)
    fig,axes=plt.subplots(2,3,figsize=(6.75,4.3))
    colors={'dagc':'#17668a','no_alignment':'#73a9aa','median':'#cf7647','replay':'#7f63a9'}
    for col,c in enumerate(protocol['conditions']):
        for policy,color in colors.items():
            frame=curves[(curves.condition==c['name'])&(curves.policy==policy)]
            grouped=frame.groupby('bin')
            average=grouped[['threshold','coefficient','lower_bound_hit']].mean()
            x=(average.index+.5)/100
            axes[0,col].plot(x,average.threshold,label=LABELS[policy],color=color)
            axes[1,col].plot(x,1-average.coefficient,label=LABELS[policy],color=color)
        axes[0,col].set_title(CONDITIONS[c['name']]);axes[0,col].set_yscale('log')
        axes[1,col].set_xlabel('Fraction of training');axes[1,col].set_ylim(0,1)
    axes[0,0].set_ylabel('Applied threshold');axes[1,0].set_ylabel('Clipping intensity')
    axes[0,2].legend(fontsize=7,loc='best')
    fig.tight_layout();fig.savefig(paper/'figures/lyx1_traces.pdf');plt.close(fig)
    print(json.dumps({'runs':len(records),'primary_comparisons':primary},indent=2))


if __name__=='__main__':main()
