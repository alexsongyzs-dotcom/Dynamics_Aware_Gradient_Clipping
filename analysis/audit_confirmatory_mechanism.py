"""Post-confirmation descriptive audit of complete trajectories (no new tests).

Requires the archived full raw traces, rather than the 100-bin export.
Example: python -m analysis.audit_confirmatory_mechanism --root CAMPAIGN \
    --output results/lyx1/mechanism_audit.json
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();campaign=Path(args.root);root=campaign/'raw/confirmation'
    protocol=json.loads((campaign/'manifest.json').read_text())['protocol']
    selection=json.loads((campaign/'selection.json').read_text())
    rows=[]
    for condition in sorted(root.iterdir()):
        for seed in protocol['confirmation_seeds']:
            with np.load(condition/'dagc'/f'seed_{seed}'/'trace.npz') as a, np.load(condition/'no_alignment'/f'seed_{seed}'/'trace.npz') as b:
                state=1.;negative=0;minimum=1.;negative_unprojected=0
                for index,value in enumerate(a['alignment']):
                    state=.9*state+.1*float(value)
                    negative+=state<0;minimum=min(minimum,state)
                    negative_unprojected+=bool(state<0 and not a['lower_bound_hit'][index])
                rows.append({'condition':condition.name,'seed':seed,
                    'dagc_no_alignment_trace_equal':all(np.array_equal(a[k],b[k]) for k in a.files),
                    'negative_alignment_ema_steps':negative,
                    'negative_alignment_not_at_lower_bound':negative_unprojected,
                    'minimum_alignment_ema':minimum,'steps':len(a['alignment']),
                    'lower_bound_fraction':float(a['lower_bound_hit'].mean())})
    fixed=[]
    value=selection['cifar_sgd_0p1']['fixed']['selected_value']
    value_label=str(value).replace('.','p')
    for seed in protocol['confirmation_seeds']:
        path=root/'cifar_sgd_0p1'
        with np.load(path/'none'/f'seed_{seed}'/'trace.npz') as a, np.load(path/f'fixed_{value_label}'/f'seed_{seed}'/'trace.npz') as b:
            fixed.append({'seed':seed,'norms_identical':bool(np.array_equal(a['norm'],b['norm'])),
                'coefficients_identical':bool(np.array_equal(a['coefficient'],b['coefficient'])),
                'max_raw_norm':float(b['norm'].max()),'all_coefficients_one':bool(np.all(b['coefficient']==1.))})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'dagc_no_alignment':rows,'cifar_low_lr_fixed_none':fixed},indent=2)+'\n')


if __name__=='__main__':main()
