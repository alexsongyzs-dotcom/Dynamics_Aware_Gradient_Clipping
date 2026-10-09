"""Read-only campaign snapshot suitable for periodic monitoring."""
import argparse
import json
from pathlib import Path
from collections import Counter


def snapshot(root):
    root=Path(root)
    if not (root/'manifest.json').exists():
        return {'phase':'not_initialized'}
    manifest=json.loads((root/'manifest.json').read_text())
    status=json.loads((root/'status.json').read_text())
    counts=Counter(); failures=[]; active=[]; errors=[]
    for path in sorted((root/'raw').rglob('result.json')):
        result=json.loads(path.read_text())
        stage,condition=path.relative_to(root/'raw').parts[:2]
        counts[(stage,condition)]+=1
        if result['status']=='failed':
            failures.append({'run':str(path.parent.relative_to(root/'raw')),'reason':result['failure']})
    for path in sorted((root/'raw').rglob('progress.json')):
        if not path.with_name('result.json').exists():
            progress=json.loads(path.read_text())
            active.append({'run':str(path.parent.relative_to(root/'raw')),
                'epoch':progress['epoch'],'epochs':progress['epochs']})
    for path in sorted((root/'raw').rglob('console.log')):
        content=path.read_text()
        if 'Traceback (most recent call last)' in content:
            errors.append(str(path.parent.relative_to(root/'raw')))
    return {'phase':status['phase'],'protocol':manifest['protocol']['name'],
        'finished':sum(counts.values()),'counts':{'/'.join(k):v for k,v in counts.items()},
        'scientific_failures':failures,'checkpointed_unfinished_runs':active,'process_errors':errors}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    print(json.dumps(snapshot(parser.parse_args().root),indent=2))
