"""Append-only-by-policy local trial bookkeeping for direct weight submissions.

This validates interaction accounting, NOT security isolation. The participant can
read the public evaluator/oracles locally; use a separate service for sealed work.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

from .direct_synthesis import CONDITIONS, TASKS, score_bytes, evaluate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n',encoding='utf8')


def start(path: Path, task_id: str, *, participant: str, provenance_path: Path | None = None) -> dict:
    if path.exists(): raise FileExistsError(str(path))
    if task_id not in TASKS: raise ValueError('Unknown task')
    if not participant.strip(): raise ValueError('Missing participant')
    task=TASKS[task_id]
    provenance_sha256=(sha256(provenance_path.read_bytes()).hexdigest() if provenance_path else None)
    state={'schema_version':1,'task':task_id,'condition':CONDITIONS[task.condition],
           'condition_id':task.condition,'participant':participant,
           'manifest_sha256':provenance_sha256,'started_at':_now(),
           'probes':[],'final':None,'finalized':False}
    _write(path,state)
    return state


def submit(path: Path, artifact: Path, *, is_probe: bool) -> dict:
    state=json.loads(path.read_text(encoding='utf8'))
    if state['finalized']:raise ValueError('Trial already finalized')
    quota=state['condition']['validation_probes']
    if is_probe and (quota == 0 or len(state['probes'])>=quota):
        raise ValueError('Validation probe forbidden or budget exhausted')
    raw=artifact.read_bytes()
    task_id=state['task']
    # Never inspect test/OOD/edge splits during a probe, even internally.
    score=score_bytes(task_id,raw,splits=('validation',) if is_probe else ('test','ood','edge'))
    record={'at':_now(),'artifact_sha256':sha256(raw).hexdigest(),
            'byte_size':len(raw),'score':score}
    if is_probe:state['probes'].append(record)
    else:state['final']=record;state['finalized']=True
    _write(path,state)
    return record


def main() -> None:
    import argparse
    ap=argparse.ArgumentParser(description='Local controlled direct-weight trial bookkeeping; not a security boundary')
    sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('start');a.add_argument('task',choices=sorted(TASKS));a.add_argument('--session',required=True)
    a.add_argument('--participant',required=True);a.add_argument('--provenance-file')
    for name in ('probe','finalize'):
        a=sub.add_parser(name);a.add_argument('--session',required=True);a.add_argument('candidate')
    args=ap.parse_args()
    if args.command=='start':result=start(Path(args.session),args.task,participant=args.participant,
                                           provenance_path=Path(args.provenance_file) if args.provenance_file else None)
    else:result=submit(Path(args.session),Path(args.candidate),is_probe=args.command=='probe')
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
