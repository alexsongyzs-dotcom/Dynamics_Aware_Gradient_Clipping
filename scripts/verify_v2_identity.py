"""Audit the algorithm identity independently of policy and engine labels."""
import hashlib
import json
import subprocess
import torch
from pathlib import Path
from src.clipping import DynamicsAwareClipping
from src.confirmatory import Policy, source_digest


def main():
    archived=json.loads(Path('results/lyx1/evidence/manifest.json').read_text())['source_sha256']
    files=['confirmatory.py','cifar_resnet.py','clipping.py','models.py']
    frozen=hashlib.sha256(b''.join(subprocess.check_output(['git','show','233bb79:src/'+f]) for f in files)).hexdigest()
    assert frozen==archived, 'Archived source digest does not match the reported training commit'
    normalized_match=all(Path('src',f).read_bytes().replace(b'\r\n',b'\n') ==
        subprocess.check_output(['git','show','233bb79:src/'+f]).replace(b'\r\n',b'\n') for f in files)
    assert normalized_match, 'Current training sources differ from the archived implementation'
    actual=Policy({'policy':'dagc'}).controller
    expected=DynamicsAwareClipping(init_c=.5,gamma=.05,beta=.9,relax=.3,osc_weight=3.,beta_a=.9,
        c_min_scale=.1,c_max_scale=10.,norm_window=200,exposure_target=.1)
    for name in ['gamma','beta','relax','osc_weight','beta_a','c_min_scale','c_max_scale','exposure_target','c','E','a_ema']:
        assert getattr(actual,name)==getattr(expected,name), name
    # Compare clipping and state transitions through positive and negative alignments.
    policy=Policy({'policy':'dagc'})
    parameter=torch.nn.Parameter(torch.zeros(1))
    for step,(norm,alignment) in enumerate([(1.,1.),(2.,-.9),(.2,-1.),(10.,-.8)]*80):
        parameter.grad=torch.tensor([norm])
        old=expected.c
        threshold,coefficient,_=policy.apply([parameter],norm,alignment,step)
        assert threshold==old and coefficient==min(1.,old/max(norm,1e-12))
        expected.update(norm,alignment)
        assert policy.controller.c==expected.c and policy.controller.E==expected.E
    print(json.dumps({'archived_training_source_verified':True,'confirmation_policy_is_v2':True,
        'matched_controller_steps':320,'current_source_bytes_match_archive':source_digest()==archived,
        'current_source_matches_after_line_ending_normalization':normalized_match,
        'note':'lyx1-v1 is the training engine version, not DAGC-V1'},indent=2))


if __name__=='__main__':main()
