"""One-shot, no-tools exploratory inference with a pinned public HF language model.

This is intentionally a small CPU smoke test, not a confirmatory model benchmark.
The model sees only the public task specification and topology, never the oracle
weights. A Python evaluator subsequently scores the immutable raw response.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time

from wsp.generated import FAMILIES, make_task, evaluate_generated
from wsp.pilot import _prompt, _decode_candidate
from wsp.architecture import check_network, architecture_hash

MODEL_ID='Qwen/Qwen3-0.6B'
MODEL_REVISION='c1899de289a04d12100db370d81485cdf75e47ca'
TASK_SEED=20261105


def score_raw(task, raw: str) -> dict:
    entry={"sha256":sha256(raw.encode()).hexdigest(),"raw":raw,"error":None,"result":None}
    try:
        candidate=_decode_candidate(raw)
        if candidate.get('schema_version')!=1: raise ValueError('Bad candidate schema_version')
        if 'explanation' in candidate: raise ValueError('Explanation forbidden')
        check_network(candidate.get('network'),task.architecture)
        entry['result']=evaluate_generated(task,candidate)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        entry['error']=type(exc).__name__+': '+str(exc)
    if entry['error'] is not None:
        # Secondary, explicitly post-hoc diagnostic only. Never replaces strict score.
        entry['posthoc_extracted_json'] = None
        left, right = raw.find('{'), raw.rfind('}')
        if 0 <= left < right:
            candidate_text = raw[left:right+1]
            try:
                alternative = _decode_candidate(candidate_text)
                if alternative.get('schema_version') != 1 or 'explanation' in alternative:
                    raise ValueError('Secondary artifact has wrong schema/explanation rule')
                check_network(alternative.get('network'), task.architecture)
                secondary_score = evaluate_generated(task, alternative)
                entry['posthoc_extracted_json'] = {'score':secondary_score,'would_pass':secondary_score['all_passed']}
            except (ValueError, TypeError, KeyError, OverflowError):
                pass
    return entry


def run(*, dry_run: bool = False, max_new_tokens: int = 640) -> dict:
    tasks=[make_task(family,TASK_SEED) for family in FAMILIES]
    if dry_run:
        return {'status':'dry_run_only','tasks':[t.public() for t in tasks],
                'reference_results':[evaluate_generated(t,t.reference()) for t in tasks]}
    # Heavy dependencies are loaded only in this explicit opt-in CPU job.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
                                               torch_dtype=torch.float32)
    model.eval()
    results=[]
    for task in tasks:
        prompt=_prompt(task,'forbidden')
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                          add_generation_prompt=True,enable_thinking=False,
                                          return_tensors='pt')
        started=time.monotonic()
        with torch.inference_mode():
            output=model.generate(ids,max_new_tokens=max_new_tokens,do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        seconds=round(time.monotonic()-started,3)
        generated=output[0][ids.shape[-1]:]
        raw=tokenizer.decode(generated,skip_special_tokens=True)
        recorded=score_raw(task,raw)
        results.append({'task_id':task.task_id,'family':task.family,'task_seed':task.seed,
                        'architecture_sha256':architecture_hash(task.architecture),
                        'prompt_sha256':sha256(prompt.encode()).hexdigest(),
                        'prompt':prompt,'output_tokens':len(generated),
                        'elapsed_seconds':seconds,'submitted':recorded,
                        'passed':bool(recorded['result'] and recorded['result']['all_passed'])})
        print(json.dumps({'family':task.family,'passed':results[-1]['passed'],
                          'error':recorded['error'],'output_tokens':len(generated),
                          'seconds':seconds},sort_keys=True),flush=True)
    return {'schema_version':1,'status':'exploratory_unregistered_single_model_not_blind_family_holdout',
            'model':MODEL_ID,'model_revision':MODEL_REVISION,'sampling':'greedy',
            'tools_available_to_model':[],
            'feedback_probes':0,'explanation_condition':'forbidden',
            'task_specifications_are_visible':True,
            'scientific_limitations':'4 same-seed examples, no repeated attempts; small model; public task families; no statistical inference',
            'results':results,'successful':sum(r['passed'] for r in results),
            'total':len(results)}


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=640)
    parser.add_argument('--output',default='.runs/open-model-smoke.json')
    args=parser.parse_args()
    if not 16 <= args.max_new_tokens <= 1000:raise SystemExit('max-new-tokens out of range')
    report=run(dry_run=args.dry_run,max_new_tokens=args.max_new_tokens)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'successful':report.get('successful'),
                      'total':report.get('total'),'output':str(path)},sort_keys=True))

if __name__=='__main__':main()
