"""Synthetic evidence tests: no model scores here are experimental results."""
import hashlib
import json
import sys
import numpy as np
import pytest
from src.confirmatory import atomic_json,digest
from scripts.export_confirmatory import main


def fixture_evidence(tmp_path):
    root=tmp_path/'campaign'
    protocol={'conditions':[{'name':'toy','dataset':'fashion_mnist','lr':.1,'epochs':1,'weight_decay':0.}],
              'batch_size':55000,'split_seed':1729,'calibration_seeds':[10],
              'confirmation_seeds':[20,21],'confirmation_policies':['dagc'], 'calibration_grid':{}}
    atomic_json(root/'manifest.json',{'protocol':protocol,'source_sha256':'synthetic-source'})
    atomic_json(root/'status.json',{'phase':'complete','calibration_runs':1,'confirmation_runs':2})
    (root/'replay').mkdir()
    np.save(root/'replay/toy.npy',np.array([.5]))
    atomic_json(root/'selection.json',{'toy':{'replay_sha256':hashlib.sha256((root/'replay/toy.npy').read_bytes()).hexdigest()}})
    for stage,seeds in (('calibration',[10]),('confirmation',[20,21])):
        for seed in seeds:
            cfg={'dataset':'fashion_mnist','lr':.1,'epochs':1,'weight_decay':0.,'batch_size':55000,
                 'split_seed':1729,'stage':stage,'seed':seed,'policy':'dagc'}
            result={'config':cfg,'engine_version':'test-fixture','source_sha256':'synthetic-source',
                'identity':digest({'config':cfg,'source':'synthetic-source','version':'test-fixture'}),
                'split':{'train_n':55000,'validation_n':5000,'evaluation_n':5000 if stage=='calibration' else 10000,
                    'evaluation_split':'training_validation' if stage=='calibration' else 'official_test','split_sha256':'fixed-fixture'},
                'steps':1,'expected_steps':1,'epochs':[{'epoch':1}],'status':'complete',
                'metrics':{'accuracy':.5},'failure':None}
            run=root/f'raw/{stage}/toy/dagc/seed_{seed}'
            atomic_json(run/'result.json',result)
            np.savez_compressed(run/'trace.npz',norm=np.array([1.]),threshold=np.array([.5]),
                coefficient=np.array([.5]),alignment=np.array([1.]),lower_bound_hit=np.array([0]))
    return root


def invoke(monkeypatch,root,out):
    monkeypatch.setattr(sys,'argv',['export','--root',str(root),'--output',str(out)])
    main()


def test_complete_evidence_and_immutable_output(tmp_path,monkeypatch):
    root=fixture_evidence(tmp_path);out=tmp_path/'export'
    invoke(monkeypatch,root,out)
    audit=json.loads((out/'audit.json').read_text())
    assert audit['verified_runs']==3 and audit['replay_construction_verified']
    with pytest.raises(FileExistsError):invoke(monkeypatch,root,out)


def test_reject_incomplete_or_mismatched_evidence(tmp_path,monkeypatch):
    root=fixture_evidence(tmp_path)
    path=root/'raw/confirmation/toy/dagc/seed_21/result.json'
    r=json.loads(path.read_text());r['split']['evaluation_split']='training_validation';atomic_json(path,r)
    with pytest.raises(AssertionError):invoke(monkeypatch,root,tmp_path/'bad-export')
    assert not (tmp_path/'bad-export/audit.json').exists()


def test_reject_modified_replay(tmp_path,monkeypatch):
    root=fixture_evidence(tmp_path)
    np.save(root/'replay/toy.npy',np.array([.6]))
    with pytest.raises(AssertionError):invoke(monkeypatch,root,tmp_path/'bad-replay')
    assert not (tmp_path/'bad-replay/audit.json').exists()
