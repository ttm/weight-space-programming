"""Matched-topology function-reading challenges and ReLU gauge transformations.

Public banks are development fixtures. A sealed trial must not disclose its key.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
import random

from .architecture import check_network
from .generated import make_task
from .network import predict


def transform(network: dict, *, seed: int) -> dict:
    """Preserve network function under hidden-unit permutations and positive ReLU scales."""
    new=deepcopy(network)
    if len(new["layers"])!=2 or new["layers"][0]["activation"]!="relu":
        raise ValueError("Requires two-layer ReLU network")
    first,second=new["layers"]
    n=len(first["bias"])
    rng=random.Random(seed)
    perm=list(range(n));rng.shuffle(perm)
    scales=[rng.choice((0.25,0.5,1.,2.,4.)) for _ in range(n)]
    first["weights"]=[[float(v*s) for v in first["weights"][j]] for j,s in zip(perm,scales)]
    first["bias"]=[float(first["bias"][j]*s) for j,s in zip(perm,scales)]
    for row in second["weights"]:
        old=list(row)
        row[:]=[old[j]/s for j,s in zip(perm,scales)]
    return new


def canonicalize(network: dict) -> dict:
    """A limited canonical form for permutation + positive rescaling (nonzero hidden units).

    Not a general neural-network canonicalization. Redundant neurons, cancellations,
    sign/rotation symmetries, and general function equivalence are not resolved.
    """
    net=deepcopy(network)
    if len(net["layers"])!=2 or net["layers"][0]["activation"]!="relu":
        raise ValueError("Requires two-layer ReLU network")
    first,second=net["layers"]
    rows=[]
    for j,(w,b) in enumerate(zip(first["weights"],first["bias"])):
        scale=max(abs(v) for v in [*w,b])
        if scale == 0: raise ValueError("Zero hidden neuron not uniquely normalizable")
        normalized=tuple(round(v/scale,12) for v in [*w,b])
        outgoing=tuple(round(r[j]*scale,12) for r in second["weights"])
        rows.append((normalized,outgoing))
    rows.sort()
    first["weights"]=[list(normal[:-1]) for normal,_ in rows]
    first["bias"]=[normal[-1] for normal,_ in rows]
    second["weights"]=[[outgoing[i] for _,outgoing in rows] for i in range(len(second["weights"]))]
    return net


def fingerprint(network: dict) -> str:
    encoded=json.dumps(canonicalize(network),sort_keys=True,separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def binary_table(network: dict) -> str:
    return "".join(str(predict(network,list(map(float,x)),"binary")) for x in product((0,1),repeat=4))


def generate_bank(*, count: int = 8, seed: int = 0) -> tuple[dict,dict]:
    if not 1 <= count <= 100: raise ValueError("Count out of range")
    rng=random.Random(seed)
    challenges=[]; answers={}
    # Prevent accidental duplicate behavior classes: all 4-bit truth tables are compared.
    seen=set()
    iterations=0
    while len(challenges)<count:
        iterations+=1
        if iterations>10000: raise RuntimeError("Cannot generate enough distinct behaviors")
        generated=make_task("dnf4",rng.randrange(1,10**9))
        net=generated.reference_network
        table=binary_table(net)
        if table in seen: continue
        seen.add(table)
        altered=transform(net,seed=rng.randrange(1,10**9))
        ident="wread_"+sha256(f"{seed}:{len(challenges)}".encode()).hexdigest()[:12]
        challenges.append({"id":ident,"network":altered,"architecture":generated.architecture,
                           "input_semantics":"four bits in lexicographic order 0000...1111",
                           "output_semantics":"binary 1 iff scalar output > 0",
                           "requested_answer":"16-digit binary truth table without forward passes"})
        answers[ident]={"truth_table":table,"source_seed":generated.seed,
                        "same_function":binary_table(altered)==table,
                        "canonical_fingerprint":fingerprint(altered),
                        "untransformed_fingerprint":fingerprint(net)}
    return {"schema_version":1,"status":"public_development_bank_not_blind","challenges":challenges}, {"schema_version":1,"answers":answers}


def evaluate_guesses(bank: dict, key: dict, guesses: dict) -> dict:
    ids={c["id"] for c in bank["challenges"]}
    if set(key["answers"]) != ids: raise ValueError("Answer key mismatch")
    details=[]
    for ident in sorted(ids):
        truth=key["answers"][ident]["truth_table"]
        prediction=guesses.get(ident)
        valid=isinstance(prediction,str) and len(prediction)==16 and set(prediction)<={"0","1"}
        correct=valid and prediction==truth
        details.append({"id":ident,"valid":valid,"exact":correct,
                        "bit_accuracy":sum(x==y for x,y in zip(prediction,truth))/16 if valid else 0.})
    return {"tasks":len(details),"exact":sum(v["exact"] for v in details),
            "mean_bit_accuracy":sum(v["bit_accuracy"] for v in details)/len(details),"details":details}


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description="Generate matched-topology public weight-reading development bank")
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--bank",required=True)
    parser.add_argument("--key",required=True)
    args=parser.parse_args()
    bank,key=generate_bank(count=args.count,seed=args.seed)
    for name,object_ in ((args.bank,bank),(args.key,key)):
        p=Path(name);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(object_,indent=2,sort_keys=True)+"\n")
    print(f"Wrote {len(bank['challenges'])} development challenges; do NOT expose answer key in sealed studies")

if __name__=="__main__":main()
