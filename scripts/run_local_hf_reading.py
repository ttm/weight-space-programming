"""Small open-model matched-topology weight-reading pilot.

For each function, compare the raw network and a function-preserving ReLU
permutation/positive rescaling. No probes or network execution tools exposed to
participant; score only after output is immutable. No confirmatory inference.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

from wsp.generated import make_task
from wsp.weight_reading import generate_bank, binary_table
MODEL_ID='Qwen/Qwen3-0.6B'
MODEL_REVISION='c1899de289a04d12100db370d81485cdf75e47ca'


def challenges() -> list[dict]:
    bank,key=generate_bank(count=3,seed=77)
    entries=[]
    for ch in bank['challenges']:
        answer=key['answers'][ch['id']]
        task=make_task('dnf4',answer['source_seed'])
        for variant,net in [('original',task.reference_network),('gauge_transformed',ch['network'])]:
            assert binary_table(net)==answer['truth_table']
            entries.append({'id':ch['id'],'variant':variant,'network':net,
                            'answer':answer['truth_table']})
    return entries


def prompt_for(network: dict) -> str:
    return ('Interpret ONLY the literal numerical neural-network weights below, without executing '
            'the network or using code, tools, APIs, or forward-pass queries. The inputs are 4 bits '
            '(x0,x1,x2,x3) and the output class is 1 iff the final scalar logit is >0; otherwise 0. '
            'Return ONLY valid JSON of exactly this form: {"truth_table":"0000000000000000"}, '
            'where the 16 characters are the predicted outputs on inputs 0000, 0001, ..., 1111 '
            'in lexicographic order. All 16 characters must be 0 or 1. No explanation. '
            'The architecture is 4 inputs, 3 ReLU hidden neurons, 1 linear output.\n'
            'NETWORK:\n'+json.dumps(network,sort_keys=True,separators=(',',':')))


def score_raw(raw: str, answer: str) -> dict:
    result={'raw':raw,'sha256':sha256(raw.encode()).hexdigest(),
            'predicted':None,'valid':False,'exact':False,'bit_accuracy':0.,'error':None}
    try:
        data=json.loads(raw)
        if not isinstance(data,dict) or set(data)!={'truth_table'}:
            raise ValueError('JSON object must contain only truth_table')
        table=data['truth_table']
        if not isinstance(table,str) or len(table)!=16 or set(table)-{'0','1'}:
            raise ValueError('truth_table must be 16 binary digits')
        result.update({'predicted':table,'valid':True,'exact':table==answer,
                       'bit_accuracy':sum(a==b for a,b in zip(table,answer))/16})
    except (ValueError,TypeError) as exc:
        result['error']=type(exc).__name__+': '+str(exc)
    return result


def run(*, dry_run: bool = False,max_new_tokens: int = 128) -> dict:
    entries=challenges()
    if dry_run:
        return {'status':'dry_run_only','pairs':3,'challenges':len(entries),
                'all_matched':all(entries[2*i]['answer']==entries[2*i+1]['answer'] for i in range(3))}
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,torch_dtype=torch.float32)
    model.eval()
    records=[]
    for entry in entries:
        prompt=prompt_for(entry['network'])
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                          add_generation_prompt=True,enable_thinking=False,
                                          return_tensors='pt')
        start=time.monotonic()
        with torch.inference_mode():
            out=model.generate(ids,max_new_tokens=max_new_tokens,do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        elapsed=round(time.monotonic()-start,3)
        tokens=out[0][ids.shape[-1]:]
        raw=tokenizer.decode(tokens,skip_special_tokens=True)
        score=score_raw(raw,entry['answer'])
        row={'id':entry['id'],'variant':entry['variant'],
             'prompt_sha256':sha256(prompt.encode()).hexdigest(),'prompt':prompt,
             'output_tokens':len(tokens),'elapsed_seconds':elapsed,'result':score}
        records.append(row)
        print(json.dumps({'variant':entry['variant'],'id':entry['id'],
                          'exact':score['exact'],'bit_accuracy':score['bit_accuracy'],
                          'valid':score['valid'],'elapsed':elapsed},sort_keys=True),flush=True)
    pair_stats=[]
    for i in range(0,len(records),2):
        a=records[i]['result'];b=records[i+1]['result']
        pair_stats.append({'id':records[i]['id'],'both_valid':a['valid'] and b['valid'],
                           'same_predicted_function':a['valid'] and b['valid'] and a['predicted']==b['predicted'],
                           'original_exact':a['exact'],'transformed_exact':b['exact']})
    return {'schema_version':1,'status':'exploratory_unregistered_weight_reading',
            'model':MODEL_ID,'model_revision':MODEL_REVISION,'decode':'greedy',
            'tools_available_to_model':[],'feedback_probes':0,
            'pairs':pair_stats,'exact':sum(r['result']['exact'] for r in records),
            'valid':sum(r['result']['valid'] for r in records),
            'total':len(records),'records':records}


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--output',default='.runs/reading.json')
    args=parser.parse_args()
    if not 16<=args.max_new_tokens<=512:raise SystemExit('bad token cap')
    result=run(dry_run=args.dry_run,max_new_tokens=args.max_new_tokens)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'exact':result.get('exact'),'valid':result.get('valid'),
                      'total':result.get('total')},sort_keys=True))

if __name__=='__main__':main()
