"""Deterministic naive random-weight controls, separate from the GPT-6 first trial."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import random

from wsp.direct_synthesis import TASKS, evaluate

ROOT=Path(__file__).resolve().parents[1]

def make_random(task_id: str, seed: int) -> dict:
    task=TASKS[task_id]
    rng=random.Random(int.from_bytes(sha256(f'direct-random-v1:{task_id}:{seed}'.encode()).digest()[:8],'big'))
    layers=[]
    for inn,out,activation in task.architecture:
        layers.append({'weights':[[rng.gauss(0,1) for _ in range(inn)] for _ in range(out)],
                       'bias':[rng.gauss(0,1) for _ in range(out)],'activation':activation})
    result={'schema_version':1,'task':task_id,'network':{'layers':layers}}
    if task.condition.endswith('E'): result['explanation']='Random-weight negative control, not a functional derivation.'
    return result

def run(n=100):
    rows=[]
    for task_id,task in TASKS.items():
        if task.kind=='unsatisfiable':continue
        passes=0
        for seed in range(n):
            r=evaluate(task_id,make_random(task_id,seed))
            passes+=int(r['pass'])
        rows.append({'task':task_id,'attempts':n,'exact_successes':passes})
    return {'schema_version':1,'status':'NAIVE_ARCHITECTURE_MATCHED_RANDOM_CONTROL_NOT_COMPUTE_MATCHED',
            'distribution':'independent N(0,1) for every scalar weight and bias',
            'rows':rows,'n_per_task':n,'limitations':'No training or symbolic information; weak sanity baseline and not a matched-resource comparison to GPT-6.'}

if __name__=='__main__':
    r=run()
    output=ROOT/'results'/'direct_synthesis_random_controls.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(output),'summary':[(r['task'],r['exact_successes']) for r in r['rows']]},indent=2))
