"""Literal human-readable transcription of ChatGPT GPT-6 weight assignments.

This program does NOT train, optimize, sample or score parameters. Never run it to
'repair' a recorded first attempt after inspecting scores. Once an artifact is
committed, hash-lock it and score the original bytes.
"""
from pathlib import Path
import json

OUT=Path('experiments/submissions/chatgpt-gpt6-2026-11-05-round2')

def net(layers):return {'layers':layers}
def layer(w,b,a):return {'weights':w,'bias':b,'activation':a}

C={}
C['parity5']=(net([
 layer([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
       [0,-1,-2,-3,-4],'relu'),
 layer([[1,-2,2,-2,2]],[-.5],'linear')]),
 'For integer s=sum(bits), ReLU(s)-2ReLU(s-1)+2ReLU(s-2)-2ReLU(s-3)+2ReLU(s-4) equals parity(s), and subtracting 0.5 gives the correct sign.')
C['exactly3of6']=(net([
 layer([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]],[-2,-3,-4],'relu'),
 layer([[1,-2,1]],[-.5],'linear')]),None)
C['xor_pairs_and']=(net([
 layer([[1,-1,0,0],[-1,1,0,0],[0,0,1,-1],[0,0,-1,1]],[0,0,0,0],'relu'),
 layer([[1,1,1,1]],[-1],'relu'),
 layer([[1]],[-.5],'linear')]),None)
C['mux6']=(net([
 layer([[-1,-1,1,0,0,0],[-1,1,0,1,0,0],[1,-1,0,0,1,0],[1,1,0,0,0,1]],
       [0,-1,-1,-2],'relu'),
 layer([[1,1,1,1]],[-.5],'linear')]),
 'Each hidden ReLU is an exact three-literal conjunction of the selected address and selected data bit, producing 1 iff satisfied, and output OR is the positive sign of their sum minus 0.5.')
C['full_adder3']=(net([
 layer([[1,1,1],[1,1,1],[1,1,1]],[0,-1,-2],'relu'),
 layer([[1,-2,2],[1,0,0]],[-.5,-1.5],'linear')]),None)
C['compare_two_2bit']=(net([
 layer([[2,1,-2,-1],[-2,-1,2,1]],[0,0],'relu'),
 layer([[1,0],[-1,-1],[0,1]],[-.5,.5,-.5],'linear')]),
 'The hidden units compute positive and negative parts of integer A-B. Their signs identify greater and lesser; equality is the strict positivity of 0.5-|A-B|.')
C['trapezoid1']=(net([
 layer([[1],[1],[1],[1]],[2,1,-1,-2],'relu'),
 layer([[1,-1,-1,1]],[0],'linear')]),
 'This is a telescoping hinge expansion with knots -2,-1,1,2; the cumulative slope is 0,+1,0,-1,0.')
C['max3']=(net([
 layer([[1,-1,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],[0,0,0,0,0],'relu'),
 layer([[1,1,-1,-1,1],[0,0,0,1,0],[0,0,0,0,1]],[0,0,0],'relu'),
 layer([[1,1,-1]],[0],'linear')]),None)
C['median3']=(net([
 layer([[1,-1,0],[-1,1,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
       [0,0,0,0,0,0,0,0],'relu'),
 layer([[0,1,0,0,-1,1,1,-1],[-1,0,0,0,-1,1,1,-1],
        [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0]],
       [0,0,0,0,0],'relu'),
 layer([[1,-1,1,-1,-1]],[0],'linear')]),
 'Let lo=min(x,y)=y-ReLU(y-x), hi=max(x,y)=y+ReLU(x-y). Then median(x,y,z)=lo+ReLU(z-lo)-ReLU(z-hi). First layer keeps positive/negative parts; second constructs the two comparisons and nonnegative passthroughs.')
C['annulus_l1_2']=(net([
 layer([[1,0],[-1,0],[0,1],[0,-1]],[0,0,0,0],'relu'),
 layer([[1,1,1,1],[1,1,1,1],[1,1,1,1]],[-.5,-1.25,-2],'relu'),
 layer([[1,-2,1]],[0],'linear')]),None)
C['rotated_quadrants4']=(net([
 layer([[-3,-1],[1,-3],[-1,3],[3,1]],[.25,-.75,.75,-.25],'linear')]),None)
C['fourlevel1']=(net([
 layer([[-3],[-1],[1],[3]],[0,2,1,-3],'linear')]),None)
C['hinge2']=(net([
 layer([[2,-1],[-1,3]],[1,-.25],'relu'),
 layer([[1,.5]],[-1.25],'linear')]),
 'The hidden units are the two affine ReLU hinges. The readout adds the first and half the second, then subtracts the prescribed intercept.')
C['parity3_affine_impossible']=(None,
 'Fix x2=0. Then the 3-bit parity function reduces to XOR(x0,x1), which is not linearly separable. Indeed the affine threshold on (0,0) requires b<=0, on (1,0),(0,1) requires w0+b>0 and w1+b>0, and on (1,1) requires w0+w1+b<=0, contradicting w0+w1+b>0.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for task_id,(network,explanation) in C.items():
        cand={'schema_version':2,'task':task_id}
        if network is not None:cand['network']=network
        else:cand.update({'status':'unsatisfiable','reason':explanation})
        if explanation is not None:cand['explanation']=explanation
        (OUT/(task_id+'.json')).write_text(json.dumps(cand,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
    manifest={'schema_version':2,'participant':{
        'self_identification':'GPT-6','surface':'ChatGPT conversation','access':'ChatGPT application',
        'inference_model_id':None,'checkpoint_sha':None,'build_id':None,'reasoning_effort_tier':None,
        'sampling_temperature':None,'sampling_seed':None,
        'local_date':'2026-11-05','timezone':'Europe/Rome',
        'tools_available':['GitHub connector','container shell','Python'],'tools_used_in_workflow':['GitHub connector','container shell'],
        'generation_method':'direct analytic assignment and static JSON transcription',
        'training_or_optimization_of_candidate_parameters':False,
        'task_visibility':'full public task specifications and oracle code were visible to subject',
        'blind_trial':False,
        'validation_probe_count_per_task':0,
        'provenance_limitations':['Backend build and exact thinking effort not exposed','Full hidden reasoning and exact model inference configuration unavailable','No independent timestamp or subject isolation']},
        'tasks':list(C),'kind':'exploratory development self-submission, first pass'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print('Wrote',len(C),'static submissions and manifest to',OUT)

if __name__=='__main__':main()
