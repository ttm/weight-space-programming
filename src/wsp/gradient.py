"""Pure-Python supervised Adam baseline on generated feedforward tasks.

This is an intentionally transparent baseline, not a state-of-the-art optimized
training implementation. It consumes oracle-labeled samples, whereas an LLM
receives a full symbolic specification; compare information budgets separately.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random
from hashlib import sha256

from .baselines import sample_network
from .generated import GeneratedTask, evaluate_generated, examples
from .architecture import check_network


def _seed(*xs) -> int:
    return int.from_bytes(sha256(':'.join(map(str,xs)).encode()).digest()[:8],'big')


def _activation(name: str, z: float) -> float:
    if name=='linear':return z
    if name=='relu':return max(0.,z)
    if name=='tanh':return math.tanh(z)
    if name=='sigmoid':return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
    raise ValueError(name)


def _derivative(name: str, z: float, a: float) -> float:
    if name=='linear':return 1.
    if name=='relu':return float(z>0)
    if name=='tanh':return 1-a*a
    if name=='sigmoid':return a*(1-a)
    raise ValueError(name)


def loss_and_gradient(network: dict, x: list[float], y: int | float, kind: str) -> tuple[float,dict]:
    """Exact manual reverse-mode derivatives for the network's supported activations.

    Binary uses logistic loss on a scalar logit. Regression uses 0.5 squared error.
    Multiclass is not yet implemented in this baseline.
    """
    if kind not in ('binary','regression'):
        raise ValueError('Gradient baseline currently supports binary/regression')
    activations=[list(x)];preactivations=[]
    for layer in network['layers']:
        zs=[math.fsum([b,*(w*v for w,v in zip(row,activations[-1]))])
            for row,b in zip(layer['weights'],layer['bias'])]
        values=[_activation(layer['activation'],z) for z in zs]
        preactivations.append(zs);activations.append(values)
    if len(activations[-1])!=1:raise ValueError('Requires scalar output')
    output=activations[-1][0]
    if kind=='binary':
        # stable logistic loss = log(1 + exp(z)) - y*z
        loss=max(0.,output)+math.log1p(math.exp(-abs(output)))-float(y)*output
        derivative=_activation('sigmoid',output)-float(y)
    else:
        loss=.5*(output-float(y))**2
        derivative=output-float(y)
    errors=[derivative]
    grads=[None]*len(network['layers'])
    for li in range(len(network['layers'])-1,-1,-1):
        layer=network['layers'][li]
        dz=[e*_derivative(layer['activation'],z,a)
            for e,z,a in zip(errors,preactivations[li],activations[li+1])]
        grads[li]={'weights':[[g*inp for inp in activations[li]] for g in dz],
                   'bias':list(dz)}
        errors=[math.fsum(layer['weights'][i][j]*dz[i] for i in range(len(dz)))
                for j in range(len(activations[li]))]
    return loss, {'layers':grads}


def train(task: GeneratedTask, *, steps: int = 1000, seed: int = 0,
          learning_rate: float = .02, salt: str = '') -> dict:
    if not 1 <= steps <= 20000 or not 0 < learning_rate <= 1:
        raise ValueError('Invalid training budget or learning rate')
    rng=random.Random(_seed('adam',task.family,task.seed,seed))
    net=sample_network(task.architecture,rng,scale=.2)
    # A small positive initial bias avoids all ReLU units initially inactive.
    for layer in net['layers'][:-1]:
        if layer['activation']=='relu':layer['bias']=[.05 for _ in layer['bias']]
    training=examples(task,'train')
    beta1,beta2,epsilon=.9,.999,1e-8
    m=deepcopy(net);v=deepcopy(net)
    for state in (m,v):
        for layer in state['layers']:
            layer['weights']=[[0. for _ in row] for row in layer['weights']]
            layer['bias']=[0. for _ in layer['bias']]
    first_loss=None;last_loss=None
    for step in range(1,steps+1):
        x,y=training[rng.randrange(len(training))]
        loss,grad=loss_and_gradient(net,x,y,task.kind)
        if first_loss is None:first_loss=loss
        last_loss=loss
        bc1=1-beta1**step;bc2=1-beta2**step
        for layer,gm,mm,vv in zip(net['layers'],grad['layers'],m['layers'],v['layers']):
            for key in ('weights','bias'):
                if key=='weights':
                    for row,grow,mrow,vrow in zip(layer[key],gm[key],mm[key],vv[key]):
                        for i,g in enumerate(grow):
                            mrow[i]=beta1*mrow[i]+(1-beta1)*g
                            vrow[i]=beta2*vrow[i]+(1-beta2)*g*g
                            row[i]-=learning_rate*(mrow[i]/bc1)/(math.sqrt(vrow[i]/bc2)+epsilon)
                else:
                    for i,g in enumerate(gm[key]):
                        mm[key][i]=beta1*mm[key][i]+(1-beta1)*g
                        vv[key][i]=beta2*vv[key][i]+(1-beta2)*g*g
                        layer[key][i]-=learning_rate*(mm[key][i]/bc1)/(math.sqrt(vv[key][i]/bc2)+epsilon)
    check_network(net,task.architecture)
    result=evaluate_generated(task,{'task':task.task_id,'network':net},salt=salt)
    return {'method':'supervised_adam','task':task.task_id,'seed':seed,'steps':steps,
            'forward_backward_pairs':steps,'training_examples':len(training),
            'oracle_labels':len(training),'first_minibatch_loss':first_loss,
            'last_minibatch_loss':last_loss,'final':result,'pass':result['all_passed']}


def compare(*, seeds: int = 2, steps: tuple[int,...]=(100,500), salt: str = '') -> dict:
    from .generated import FAMILIES,make_task
    records=[]
    for family in FAMILIES:
        for seed in range(seeds):
            task=make_task(family,seed)
            for count in steps:
                record=train(task,steps=count,seed=seed,salt=salt)
                records.append({'family':family,'task_seed':seed,**record})
    summary=[]
    for count in steps:
        relevant=[r for r in records if r['steps']==count]
        summary.append({'steps':count,'trials':len(relevant),
                        'exact_successes':sum(r['pass'] for r in relevant),
                        'mean_test_accuracy_per_family':{
                            fam:sum(r['final']['splits']['test']['value'] for r in relevant
                                    if r['family']==fam)/seeds
                            for fam in FAMILIES if fam!='hinge1'}})
    return {'status':'supervised_baseline_not_matched_to_llm_compute','summary':summary,'records':records}


def main():
    import argparse,json
    parser=argparse.ArgumentParser(description='Pure Python Adam on oracle-labeled generated tasks')
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--steps',type=int,nargs='+',default=[100,500])
    parser.add_argument('--output')
    args=parser.parse_args()
    result=compare(seeds=args.seeds,steps=tuple(args.steps))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'summary':result['summary']},indent=2))

if __name__=='__main__':main()
