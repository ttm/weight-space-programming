"""Replay exact-byte, first-submission GPT-6 direct-weight results without model calls."""
from __future__ import annotations

import json
from pathlib import Path

from wsp.direct_synthesis import report

ROOT=Path(__file__).resolve().parents[1]
SUBMISSIONS=ROOT/'experiments'/'submissions'/'chatgpt-gpt6-2026-11-05'
ARCHIVED=ROOT/'experiments'/'reports'/'2026-11-05-gpt6-direct-synthesis.json'


def verify() -> dict:
    expected=json.loads(ARCHIVED.read_text(encoding='utf8'))
    observed=report(SUBMISSIONS)
    if expected != observed:
        raise AssertionError('Archived first-submission result differs from deterministic score replay')
    if expected['tasks']!=10 or expected['passed']!=9 or expected['failed']!=1:
        raise AssertionError('Unexpected pilot accounting')
    failures=[r['task'] for r in observed['records'] if not r['pass']]
    if failures!=['xor2_impossible']:
        raise AssertionError('Unexpected failed tasks')
    return {'verification':'passed','tasks':10,'passed':9,'failed':1,'failed_task':failures[0],
            'note':'Reevaluation of logged numerical artifacts, not a new model generation or independently blinded benchmark'}

if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
