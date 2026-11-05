"""Generate provider-independent prompt JSONL. No solution weights or oracle labels."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
from wsp.direct_challenges_v2 import TASKS

BASE_INSTRUCTION='Return one JSON object only, with schema_version=2 and task matching the requested id. For realizable tasks include network.layers with literal finite numerical weights and biases and precisely the declared activation and architecture. Do not submit code, gradients or an optimizer. An explanation is required exactly when the task condition says required; omit explanation entirely when forbidden. For an impossible architecture, return status=unsatisfiable, reason and any required explanation. No tools, forward passes or evaluator probes in blind conditions.'

def export(path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for task_id,task in TASKS.items():
            entry={'task':task_id,'condition_id':task.condition,'system_instruction':BASE_INSTRUCTION,
                   'task_spec':task.public(),
                   'participant_must_not_see':['reference network','other participant submissions','test labels','scoring code']}
            f.write(json.dumps(entry,sort_keys=True)+'\n')
    return len(TASKS)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='.runs/direct-round2-prompts.jsonl');a=p.parse_args()
    print(f'Wrote {export(Path(a.output))} task prompts to {a.output}')
