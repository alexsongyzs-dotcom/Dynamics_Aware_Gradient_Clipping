"""Eight independent GPUs, unchanged upstream V2 runner and training sources."""
import csv, datetime, hashlib, json, os, pathlib, platform, shutil, subprocess, sys, time
B=pathlib.Path('/mnt/data/users/yuxin/workspace/Co_humanoid')
S=B/'source'; O=B/'operations'; R=B/'results'/'v2_original_recheck_lr0p8'
DATA=B/'data'; seeds=list(range(300,308))
O.mkdir(exist_ok=True); R.mkdir(parents=True,exist_ok=False)
os.chdir(S)
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
def save(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def run(cmd, name, env=None):
    with (O/name).open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(O/'status.json',dict(status='preflight',started_utc=start,port=32,experiment='Co_humanoid'))
try:
    import torch, torchvision, scipy, numpy, importlib.metadata
    environment=dict(started_utc=start,hostname=platform.node(),python=sys.version,executable=sys.executable,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','scipy','pytest']},
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu_count=torch.cuda.device_count(),
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        torch_threads=torch.get_num_threads(),environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    assert environment['gpu_count']==8
    save(O/'environment.json',environment)
    run(['nvidia-smi','-q'],'nvidia_smi_before.txt')
    run([sys.executable,'-m','pip','freeze'],'pip_freeze.txt')
    run([sys.executable,'-m','scripts.verify_v2_identity'],'v2_identity.json')
    run([sys.executable,'-m','pytest','-q'],'pytest.log')
    DATA.mkdir(exist_ok=True)
    cache=pathlib.Path('/mnt/data/users/yuxin/workspace/DAGC-develop-lyx/data/FashionMNIST')
    if cache.exists(): shutil.copytree(cache,DATA/'FashionMNIST',dirs_exist_ok=True)
    torchvision.datasets.FashionMNIST(DATA,train=True,download=True)
    torchvision.datasets.FashionMNIST(DATA,train=False,download=True)
    save(O/'dataset_sha256.json',{str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()})
    common=[sys.executable,'-u','-m','scripts.run_v2_reproduction','--data-dir',str(DATA),'--device','cuda']
    dry=subprocess.check_output(common+['--output',str(R),'--seeds',*map(str,seeds),'--dry-run'],text=True)
    manifest=json.loads(dry); save(R/'manifest.json',manifest); (O/'dry_run.json').write_text(dry)
    processes=[]; telemetry=(O/'gpu_utilization.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv','-l','2'],stdout=telemetry,stderr=subprocess.STDOUT)
    save(O/'schedule.json',dict(experiment='Co_humanoid',port=32,seeds=seeds,gpu_by_seed={str(s):i for i,s in enumerate(seeds)},source_unchanged=True))
    for gpu,seed in enumerate(seeds):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        log=(O/f'seed_{seed}.log').open('w')
        cmd=common+['--output',str(R/'shards'/f'seed_{seed}'),'--seeds',str(seed)]
        t=time.time(); p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((seed,gpu,p,log,t))
    save(O/'status.json',dict(status='running',started_utc=start,workers=[dict(seed=s,gpu=g,pid=p.pid) for s,g,p,l,t in processes]))
    print('Started all eight workers',flush=True)
    exits=[]
    while processes:
        for item in list(processes):
            seed,gpu,p,log,t=item
            if p.poll() is None: continue
            log.close(); result=R/'shards'/f'seed_{seed}'/f'seed_{seed}.json'
            exits.append(dict(seed=seed,gpu=gpu,exit_code=p.returncode,elapsed_seconds=time.time()-t,result_exists=result.exists()))
            if p.returncode==0 and result.exists(): shutil.copy2(result,R/result.name)
            processes.remove(item)
            save(O/'worker_exit_codes.json',sorted(exits,key=lambda x:x['seed']))
            print(json.dumps(exits[-1]),flush=True)
        if processes: time.sleep(2)
    monitor.terminate(); monitor.wait(timeout=10); telemetry.close()
    historical={int(r['seed']):r for r in csv.DictReader((S/'results/dagc_v2_evaluation/raw_runs.csv').open()) if r['policy']=='dagc_v2'}
    rows=[]
    for s in seeds:
        path=R/f'seed_{s}.json'
        if not path.exists(): continue
        d=json.loads(path.read_text()); v=d['result']
        assert d['config']==dict(manifest['config'],seed=s)
        assert all(len(v['log'][k])==v['steps_completed'] for k in v['log'])
        rows.append(dict(seed=s,gpu=s-300,test_acc=v['test_acc'],historical_test_acc=float(historical[s]['test_acc']),delta_pp=100*(v['test_acc']-float(historical[s]['test_acc'])),diverged=v['diverged'],steps_completed=v['steps_completed'],steps_planned=v['steps_planned'],final_loss=v['final_loss'],f_clip=v['f_clip'],i_clip=v['i_clip']))
    with (R/'paired_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    acc=[r['test_acc'] for r in rows]; old=[r['historical_test_acc'] for r in rows]
    summary=dict(experiment='Co_humanoid',port=32,source_commit=environment['source_commit'],planned_runs=8,completed_runs=len(rows),process_failures=sum(e['exit_code']!=0 for e in exits),divergences=sum(r['diverged'] for r in rows),mean_accuracy=float(numpy.mean(acc)),sample_std_accuracy=float(numpy.std(acc,ddof=1)),historical_mean_accuracy=float(numpy.mean(old)),mean_delta_pp=100*float(numpy.mean(numpy.array(acc)-old)),all_runs_complete=all(r['steps_completed']==r['steps_planned']==300 for r in rows) and len(rows)==8,scope=manifest['scope'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(R/'summary.json',summary)
    run(['nvidia-smi','-q'],'nvidia_smi_after.txt')
    save(O/'status.json',dict(status='completed' if summary['all_runs_complete'] and not summary['process_failures'] else 'incomplete',**summary))
    print(json.dumps(summary,indent=2),flush=True)
except Exception as e:
    save(O/'status.json',dict(status='failed',error=repr(e),started_utc=start))
    raise
