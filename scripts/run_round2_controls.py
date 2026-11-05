"""Development controls for GPT-6 round-two direct numeric submissions.

No model inference, paid API, gradients, or fitting. Existing submissions are immutable.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import random
from wsp.direct_challenges_v2 import TASKS, evaluate, examples
from wsp.network import forward

BASE=Path('experiments/submissions/chatgpt-gpt6-2026-11-05-round2')
REVISION=Path('experiments/revisions/gpt6_round2_annulus_posthoc.json')


def gaussian(arch:tuple,seed:int) -> dict:
    rng=random.Random(seed)
    return {'layers':[{'weights':[[rng.gauss(0,1) for _ in range(i)] for _ in range(o)],
                       'bias':[rng.gauss(0,1) for _ in range(o)],'activation':a}
                      for i,o,a in arch]}


def main() -> dict:
    attempts=30
    rows=[]
    for task_id, task in TASKS.items():
        if task.kind=='unsatisfiable':continue
        successes=0
        for seed in range(attempts):
            candidate={'schema_version':2,'task':task_id,'network':gaussian(task.architecture,seed)}
            if task.condition.endswith('E'):
                candidate['explanation']='Untrained Gaussian weight negative control.'
            successes+=bool(evaluate(task_id,candidate)['pass'])
        rows.append({'task':task_id,'random_trials':attempts,'exact_successes':successes})
    original=json.loads((BASE/'annulus_l1_2.json').read_text())
    corrected=json.loads(REVISION.read_text())
    errors=[]
    for x,y in examples('annulus_l1_2','test'):
        value=forward(original['network'],x)[0]
        if int(value>0)!=y:
            errors.append({'x':x,'expected':y,'observed_logit':value})
    if not errors:raise AssertionError('Expected floating-point annulus defect is absent')
    robustness=[]
    for seed in range(20):
        old=evaluate('annulus_l1_2',original,salt=f'round2-holdout-{seed}')
        new=evaluate('annulus_l1_2',corrected,salt=f'round2-holdout-{seed}')
        robustness.append({'salt_index':seed,
                           'first_test_accuracy':old['splits']['test']['value'],
                           'revision_test_accuracy':new['splits']['test']['value'],
                           'revision_pass':new['pass']})
    report={'schema_version':2,'status':'NO_MODEL_CALLS_DEVELOPMENT_CONTROLS',
            'random_baseline':{'distribution':'iid N(0,1)','information':'same architecture, no training','trials_per_realizable_task':attempts,'rows':rows},
            'annulus_diagnostic':{'initial_sha256':sha256((BASE/'annulus_l1_2.json').read_bytes()).hexdigest(),
                                  'posthoc_sha256':sha256(REVISION.read_bytes()).hexdigest(),
                                  'first_test_false_positives':len(errors),
                                  'first_test_examples':errors[:5],
                                  'posthoc_test':evaluate('annulus_l1_2',corrected),
                                  'additional_salts':robustness},
            'limitations':['Random weights not compute-matched to LLM','Correction is POST-HOC and forbidden as a revision of original blind B-W submission','Reference mathematical tasks and evaluator are public']}
    output=Path('results/direct_synthesis_round2_controls.json')
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'random_exact':sum(x['exact_successes'] for x in rows),
                      'random_total':attempts*len(rows),
                      'annulus_first_test_errors':len(errors),
                      'revision_additional_seed_passes':sum(r['revision_pass'] for r in robustness)},sort_keys=True))
    return report

if __name__=='__main__':main()
