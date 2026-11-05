"""Deterministic replay of immutable first submissions, report and controls."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
from wsp.direct_challenges_v2 import TASKS,report,evaluate

DIRECTORY=Path('experiments/submissions/chatgpt-gpt6-2026-11-05-round2')
REPORT=Path('experiments/reports/2026-11-05-gpt6-direct-synthesis-round2.json')
CONTROL=Path('results/direct_synthesis_round2_controls.json')

def verify()->dict:
    expected=json.loads(REPORT.read_text())
    current=report(DIRECTORY)
    if expected!=current:raise AssertionError('Round2 first-submission scores or hashes changed')
    if current['tasks']!=14 or current['passed']!=13 or current['failed']!=1:
        raise AssertionError('Unexpected first-submission outcome')
    failed=[r['task'] for r in current['rows'] if not r['passed']]
    if failed!=['annulus_l1_2']:raise AssertionError('Unexpected failure task')
    controls=json.loads(CONTROL.read_text())
    correction=Path('experiments/revisions/gpt6_round2_annulus_posthoc.json')
    if sha256(correction.read_bytes()).hexdigest()!=controls['annulus_diagnostic']['posthoc_sha256']:
        raise AssertionError('Posthoc revision altered')
    if not evaluate('annulus_l1_2',json.loads(correction.read_text()))['pass']:
        raise AssertionError('Posthoc revision failed')
    if len(controls['random_baseline']['rows'])!=13:raise AssertionError('Missing random controls')
    return {'tasks':current['tasks'],'first_attempt_success':current['passed'],
            'failures':failed,'revision_success':True,
            'random_baseline_exact':sum(x['exact_successes'] for x in controls['random_baseline']['rows']),
            'random_baseline_total':sum(x['random_trials'] for x in controls['random_baseline']['rows'])}

if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
