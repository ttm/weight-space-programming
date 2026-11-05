"""Independent mathematical task oracles for WSP direct-weight synthesis, round 2.

All tasks, networks and test seeds are public development materials, never
advertised as blinded or preregistered. No optimizer is used to select tensors.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import argparse
import json
import math
import random

from .architecture import check_network
from .network import forward, predict

CONDITIONS = {
    'B-E': {'feedback':'blind','validation_probes':0,'explanation':'required'},
    'B-W': {'feedback':'blind','validation_probes':0,'explanation':'forbidden'},
    'P-E': {'feedback':'budgeted','validation_probes':3,'explanation':'required'},
    'P-W': {'feedback':'budgeted','validation_probes':3,'explanation':'forbidden'},
}

@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    summary: str
    formula: str
    architecture: tuple[tuple[int,int,str],...]
    condition: str
    domain: str
    def public(self) -> dict:
        return {'schema_version':2,'task':self.id,'kind':self.kind,
                'summary':self.summary,'formula':self.formula,'domain':self.domain,
                'architecture': {'layers':[{'inputs':i,'outputs':o,'activation':a} for i,o,a in self.architecture]},
                'condition':CONDITIONS[self.condition],
                'output_convention': {
                    'binary':'strict positive scalar logit means 1, else 0',
                    'multilabel':'independent positive scalar logits mean 1 for each output, else 0',
                    'multiclass':'argmax over logits; ties choose lowest class index',
                    'regression':'scalar; max absolute error <= 1e-6',
                    'unsatisfiable':'return certificate status unsatisfiable and proof fields, not weights'}[self.kind]}

def t(name,kind,summary,formula,architecture,condition,domain):
    return Task(name,kind,summary,formula,tuple(architecture),condition,domain)

TASKS: dict[str,Task] = {x.id:x for x in [
    t('parity5','binary','Odd parity of five bits','1 iff sum(x0..x4) is odd',[(5,5,'relu'),(5,1,'linear')],'B-E','{0,1}^5'),
    t('exactly3of6','binary','Exactly three active bits','1 iff sum(x0..x5)==3',[(6,3,'relu'),(3,1,'linear')],'B-W','{0,1}^6'),
    t('xor_pairs_and','binary','Conjunction of two independent XORs','(x0 XOR x1) AND (x2 XOR x3)',[(4,4,'relu'),(4,1,'relu'),(1,1,'linear')],'P-W','{0,1}^4'),
    t('mux6','binary','Four-way binary selector','s0=x0,s1=x1; return d[2*s0+s1] where (d0,d1,d2,d3)=(x2,x3,x4,x5)',[(6,4,'relu'),(4,1,'linear')],'P-E','{0,1}^6'),
    t('full_adder3','multilabel','Full adder, two output bits','Input a,b,c; outputs (sum=(a+b+c)%2,carry=1 iff a+b+c>=2)',[(3,3,'relu'),(3,2,'linear')],'B-W','{0,1}^3'),
    t('compare_two_2bit','multilabel','Two 2-bit unsigned integers: three comparison indicators','Inputs (a1,a0,b1,b0); outputs (A>B,A==B,A<B), A=2*a1+a0,B=2*b1+b0',[(4,2,'relu'),(2,3,'linear')],'P-E','{0,1}^4'),
    t('trapezoid1','regression','Compact trapezoid function','f(x)=0 if x<=-2 or x>=2; f=x+2 on (-2,-1); f=1 on [-1,1]; f=2-x on (1,2)',[(1,4,'relu'),(4,1,'linear')],'B-E','real x'),
    t('max3','regression','Exact max of three reals','f(x,y,z)=max(x,y,z)',[(3,5,'relu'),(5,3,'relu'),(3,1,'linear')],'P-W','real x,y,z'),
    t('median3','regression','Exact median of three reals','f(x,y,z)=median(x,y,z), including ties and negatives',[(3,8,'relu'),(8,5,'relu'),(5,1,'linear')],'B-E','real x,y,z'),
    t('annulus_l1_2','binary','L1 annulus with inner and outer strict boundaries','1 iff 0.5 < abs(x)+abs(y) < 2, equality is class 0',[(2,4,'relu'),(4,3,'relu'),(3,1,'linear')],'B-W','real x,y'),
    t('rotated_quadrants4','multiclass','Rotated four-way quadrant classifier','u=2*x-y-0.5, v=x+2*y+0.25; classes 0 if u<=0,v<=0; 1 if u>0,v<=0; 2 if u<=0,v>0; 3 if u>0,v>0; ties choose smaller index',[(2,4,'linear')],'P-W','real x,y'),
    t('fourlevel1','multiclass','Four sequential classes by three thresholds','class 0 for x<=-1; class 1 for -1<x<=0.5; class 2 for 0.5<x<=2; class 3 for x>2',[(1,4,'linear')],'B-W','real x'),
    t('hinge2','regression','Weighted composition of two affine hinges','f(x,y)=max(0,2*x-y+1)+0.5*max(0,-x+3*y-0.25)-1.25',[(2,2,'relu'),(2,1,'linear')],'P-E','real x,y'),
    t('parity3_affine_impossible','unsatisfiable','Prove 3-bit parity cannot be an affine threshold','Odd parity on three bits, restricted architecture is a single affine scalar threshold',[(3,1,'linear')],'B-E','{0,1}^3'),
]}

def oracle(task_id: str, x: list[float]) -> int | float | list[int]:
    if task_id=='parity5':return int(sum(x)%2==1)
    if task_id=='exactly3of6':return int(sum(x)==3)
    if task_id=='xor_pairs_and':return int((x[0]!=x[1]) and (x[2]!=x[3]))
    if task_id=='mux6':return int(x[2+2*int(x[0])+int(x[1])])
    if task_id=='full_adder3':return [int(sum(x)%2==1),int(sum(x)>=2)]
    if task_id=='compare_two_2bit':
        a=2*x[0]+x[1];b=2*x[2]+x[3];return [int(a>b),int(a==b),int(a<b)]
    if task_id=='trapezoid1':return max(0.,min(1.,x[0]+2,2-x[0]))
    if task_id=='max3':return max(x)
    if task_id=='median3':return sorted(x)[1]
    if task_id=='annulus_l1_2':return int(.5<abs(x[0])+abs(x[1])<2)
    if task_id=='rotated_quadrants4':
        u=2*x[0]-x[1]-.5;v=x[0]+2*x[1]+.25
        return int(u>0)+2*int(v>0)
    if task_id=='fourlevel1':return (0 if x[0]<=-1 else 1 if x[0]<=.5 else 2 if x[0]<=2 else 3)
    if task_id=='hinge2':return max(0.,2*x[0]-x[1]+1)+.5*max(0.,-x[0]+3*x[1]-.25)-1.25
    if task_id=='parity3_affine_impossible':return int(sum(x)%2==1)
    raise KeyError(task_id)

EDGES: dict[str,list[list[float]]] = {
  'trapezoid1': [[x] for x in (-3.,-2.,-1.99999999,-1.,0.,1.,1.99999999,2.,3.)],
  'max3': [[0.,0.,0.],[1.,1.,1.],[-2.,-2.,-2.],[1.,1.,-1.],[1.,-1.,1.],[-1.,1.,1.]],
  'median3': [[0.,0.,0.],[1.,1.,1.],[-2.,-2.,-2.],[1.,1.,-1.],[1.,-1.,1.],[-1.,1.,1.],[3.,-4.,2.]],
  'annulus_l1_2': [[.5,0.],[0.,-.5],[1.,1.],[0.,2.],[0.,0.],[.50000001,0.],[0.,1.99999999],[0.,2.00000001]],
  'rotated_quadrants4': [[0.,-.5],[.25,0.],[-.25,0.],[1.,0.],[0.,0.],[-1.,1.],[2.,-3.]],
  'fourlevel1': [[x] for x in (-2.,-1.000000001,-1.,-.999999999,0.,.5-.000000001,.5,.5+.000000001,2.-.000000001,2.,2.000000001,3.)],
  'hinge2': [[0.,1.],[.25,0.],[0.,0.],[-.5,0.],[1.,1.],[-3.,3.]],
}

def examples(task_id: str, split: str, *, salt: str='') -> list[tuple[list[float],object]]:
    if task_id not in TASKS or split not in ('test','ood','edge','validation'):
        raise ValueError('Unknown task or split')
    task=TASKS[task_id]
    if task.domain.startswith('{0,1}'):
        bits=list(product((0.,1.),repeat=task.architecture[0][0]))
        if split=='validation':bits=bits[::2]
        if split in ('ood','edge'):return []
        inputs=[list(xs) for xs in bits]
    elif split=='edge':inputs=EDGES.get(task_id,[])
    else:
        seed=sha256(f'wsp-r2:{task_id}:{split}:{salt if split in ("test","ood") else ""}'.encode()).digest()
        rng=random.Random(int.from_bytes(seed[:8],'big'))
        count=64 if split=='validation' else 512
        span=3. if split!='ood' else 10.
        inputs=[[rng.uniform(-span,span) for _ in range(task.architecture[0][0])] for _ in range(count)]
    return [(x,oracle(task_id,x)) for x in inputs]

def _predict(network: dict,x: list[float],kind: str):
    if kind=='multilabel':return [int(v>0) for v in forward(network,x)]
    return predict(network,x,kind)

def evaluate(task_id: str,submission:dict,*,splits=('test','ood','edge'),salt='') -> dict:
    task=TASKS[task_id]
    if not isinstance(submission,dict) or submission.get('schema_version')!=2 or submission.get('task')!=task_id:
        raise ValueError('Incorrect schema or task')
    needed=CONDITIONS[task.condition]['explanation']
    exp=submission.get('explanation')
    if needed=='required' and (not isinstance(exp,str) or not exp.strip()):raise ValueError('Explanation required')
    if needed=='forbidden' and 'explanation' in submission:raise ValueError('Explanation forbidden')
    if task.kind=='unsatisfiable':
        if submission.get('status')!='unsatisfiable' or not isinstance(submission.get('reason'),str) or len(submission['reason'])<24 or 'network' in submission:
            raise ValueError('Expected an impossibility certificate')
        # Never claim a free-text proof was machine-proven.
        return {'task':task_id,'pass':True,'kind':task.kind,'certificate':'submitted, proof not machine verified'}
    check_network(submission.get('network'),task.public()['architecture'])
    scores={}
    for split in splits:
        data=examples(task_id,split,salt=salt)
        if not data:continue
        predictions=[_predict(submission['network'],x,task.kind) for x,_ in data]
        if task.kind=='regression':
            error=[abs(p-y) for p,(_,y) in zip(predictions,data)]
            scores[split]={'n':len(data),'metric':'max_abs_error','value':max(error),'passed':max(error)<=1e-6}
        else:
            accuracy=sum(p==y for p,(_,y) in zip(predictions,data))/len(data)
            scores[split]={'n':len(data),'metric':'accuracy','value':accuracy,'passed':accuracy==1.}
    return {'task':task_id,'pass':bool(scores) and all(q['passed'] for q in scores.values()),'kind':task.kind,'splits':scores}

def score_bytes(task_id:str,raw:bytes,*,salt='',splits=('test','ood','edge')) -> dict:
    out={'sha256':sha256(raw).hexdigest(),'nbytes':len(raw),'passed':False,'evaluation':None,'error':None}
    try:
        out['evaluation']=evaluate(task_id,json.loads(raw),salt=salt,splits=splits)
        out['passed']=out['evaluation']['pass']
    except (ValueError,TypeError,KeyError,OverflowError,ArithmeticError) as exc:
        out['error']=f'{type(exc).__name__}: {exc}'
    return out

def report(directory:Path,*,salt='') -> dict:
    raw=(directory/'manifest.json').read_bytes()
    manifest=json.loads(raw)
    if sorted(manifest['tasks'])!=sorted(TASKS):raise ValueError('Manifest task mismatch')
    rows=[]
    for task_id in manifest['tasks']:
        score=score_bytes(task_id,(directory/f'{task_id}.json').read_bytes(),salt=salt)
        rows.append({'task':task_id,'condition':TASKS[task_id].condition,**score})
    return {'schema_version':2,'status':'EXPLORATORY_SELF_AUTHORED_NOT_BLINDED',
            'manifest_sha256':sha256(raw).hexdigest(),'participant':manifest['participant'],
            'tasks':len(rows),'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),
            'rows':rows,
            'limitations':['Subject designed visible tasks and supplied weights','No independent sealed evaluator','No reliable inference sampling parameters or complete hidden reasoning log','No gradient or search was used to propose candidate weights','Public task oracle permits inspection; not a blinded benchmark']}

def main():
    ap=argparse.ArgumentParser(description='Second direct-weight synthesis benchmark, development only')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    pr=sub.add_parser('prompt');pr.add_argument('task',choices=sorted(TASKS))
    ev=sub.add_parser('evaluate');ev.add_argument('candidate')
    rp=sub.add_parser('report');rp.add_argument('directory');rp.add_argument('--output')
    args=ap.parse_args()
    if args.command=='list':res=[x.public() for x in TASKS.values()]
    elif args.command=='prompt':res=TASKS[args.task].public()
    elif args.command=='evaluate':
        raw=Path(args.candidate).read_bytes(); obj=json.loads(raw)
        res=score_bytes(obj['task'],raw)
    else:
        res=report(Path(args.directory))
        if args.output:Path(args.output).write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print(json.dumps(res,indent=2,sort_keys=True))

if __name__=='__main__':main()
