"""Offline trial protocol for sharing round-two tasks with other AI models.

Captures raw bytes of every attempt. The local disk is NOT a security boundary.
A sealed benchmark must isolate task answers and hidden evaluation elsewhere.
"""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
from .direct_challenges_v2 import TASKS,CONDITIONS,score_bytes

def timestamp():return datetime.now(timezone.utc).isoformat()
def save(path:Path,state:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n')

def start(path:Path,task_id:str,*,participant:str,model_id:str|None=None,provenance_file:Path|None=None)->dict:
    if path.exists():raise FileExistsError('Session already exists')
    if task_id not in TASKS:raise ValueError('Unknown task')
    if not participant.strip():raise ValueError('Empty participant')
    spec=TASKS[task_id].public()
    contract=CONDITIONS[TASKS[task_id].condition]
    return_state={
      'schema_version':2,'task':task_id,'participant':participant,'model_id':model_id,
      'condition':contract,'condition_id':TASKS[task_id].condition,
      'task_spec_sha256':sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      'provenance_file_sha256':sha256(provenance_file.read_bytes()).hexdigest() if provenance_file else None,
      'started_at':timestamp(),'probes':[],'final':None,'finalized':False,
      'caveat':'Local session bookkeeping, not a sealed evaluator.'}
    save(path,return_state)
    return return_state

def submit(path:Path,candidate_file:Path,*,probe:bool)->dict:
    state=json.loads(path.read_text())
    if state['finalized']:raise ValueError('Already finalized')
    limit=state['condition']['validation_probes']
    if probe and (limit==0 or len(state['probes'])>=limit):raise ValueError('No more probes permitted')
    raw=candidate_file.read_bytes()
    # For local audit, retain the exact original bytes of every submitted candidate.
    ordinal=len(state['probes'])+1 if probe else len(state['probes'])+1
    name=f"{'probe' if probe else 'final'}-{ordinal:02d}.json"
    vault=path.parent/(path.stem+'-artifacts');vault.mkdir(parents=True,exist_ok=True)
    stored=vault/name
    if stored.exists():raise FileExistsError('Refusing to overwrite attempt bytes')
    stored.write_bytes(raw)
    score=score_bytes(state['task'],raw,splits=('validation',) if probe else ('test','ood','edge'))
    record={'at':timestamp(),'name':name,'sha256':sha256(raw).hexdigest(),'bytes':len(raw),'score':score}
    if probe:state['probes'].append(record)
    else:state['final']=record;state['finalized']=True
    save(path,state)
    return record

def main():
    ap=argparse.ArgumentParser(description='Audit-friendly local round-two sessions')
    sub=ap.add_subparsers(dest='command',required=True)
    s=sub.add_parser('start');s.add_argument('task',choices=sorted(TASKS));s.add_argument('--session',required=True)
    s.add_argument('--participant',required=True);s.add_argument('--model-id');s.add_argument('--provenance-file')
    for command in ('probe','finalize'):
        s=sub.add_parser(command);s.add_argument('--session',required=True);s.add_argument('candidate')
    a=ap.parse_args()
    if a.command=='start':result=start(Path(a.session),a.task,participant=a.participant,
            model_id=a.model_id,provenance_file=Path(a.provenance_file) if a.provenance_file else None)
    else:result=submit(Path(a.session),Path(a.candidate),probe=(a.command=='probe'))
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
