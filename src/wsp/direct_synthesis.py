"""Direct-weight synthesis benchmark v1: specifications, independent oracles and scoring.

This is a PUBLIC EXPLORATORY benchmark, not a hidden evaluation or evidence of
unseen-task generalization. Candidate weights are never trained or repaired here.
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
from .network import predict


@dataclass(frozen=True)
class Challenge:
    id: str
    kind: str
    summary: str
    formula: str
    architecture: tuple[tuple[int, int, str], ...]
    condition: str
    domain: str

    def public(self) -> dict:
        return {"schema_version": 1, "task": self.id, "kind": self.kind,
                "summary": self.summary, "formula": self.formula,
                "architecture": {"layers": [dict(inputs=a, outputs=b, activation=act)
                                            for a, b, act in self.architecture]},
                "domain": self.domain,
                "output_convention": (
                    "Binary: class 1 iff final scalar > 0; class 0 otherwise."
                    if self.kind == "binary" else
                    "Multiclass: index of greatest logit, ties choose smallest index."
                    if self.kind == "multiclass" else
                    "Regression: scalar output, maximum absolute error <= 1e-6."
                    if self.kind == "regression" else
                    "Return an unsatisfiable certificate, not numerical parameters, if no valid network exists."),
                "condition": CONDITIONS[self.condition]}


CONDITIONS = {
    "B-E": {"feedback": "blind", "validation_probes": 0, "explanation": "required"},
    "B-W": {"feedback": "blind", "validation_probes": 0, "explanation": "forbidden"},
    "P-E": {"feedback": "budgeted", "validation_probes": 3, "explanation": "required"},
    "P-W": {"feedback": "budgeted", "validation_probes": 3, "explanation": "forbidden"},
}


def arch(*layers: tuple[int, int, str]) -> tuple[tuple[int, int, str], ...]:
    return layers


TASKS = {
    "parity3": Challenge("parity3", "binary", "Odd parity of three input bits.",
        "Return 1 iff (x0+x1+x2) mod 2 == 1.", arch((3, 3, "relu"), (3, 1, "linear")), "B-E", "{0,1}^3"),
    "exactly2of4": Challenge("exactly2of4", "binary", "Exactly two of four input bits are set.",
        "Return 1 iff x0+x1+x2+x3 == 2.", arch((4, 3, "relu"), (3, 1, "linear")), "B-W", "{0,1}^4"),
    "select3": Challenge("select3", "binary", "Select one of two data bits.",
        "Inputs (s,a,b): return a when s=0, and b when s=1.", arch((3, 2, "relu"), (2, 1, "linear")), "P-W", "{0,1}^3"),
    "slanted_diamond2": Challenge("slanted_diamond2", "binary", "Classify a rotated L1 region.",
        "Return 1 iff abs(2*x-y-0.5)+abs(x+2*y+0.25) < 2.25; boundary is 0.", arch((2, 4, "relu"), (4, 1, "linear")), "B-E", "real x,y"),
    "hat1": Challenge("hat1", "regression", "Implement a triangular bump.",
        "f(x)=max(0, 1-abs(2*x-1)), for all real x.", arch((1, 3, "relu"), (3, 1, "linear")), "P-E", "real x"),
    "clip2": Challenge("clip2", "regression", "Clamp an affine function.",
        "f(x,y)=min(2, max(-1, 3*x-2*y+0.25)).", arch((2, 2, "relu"), (2, 1, "linear")), "B-W", "real x,y"),
    "shifted_absdiff2": Challenge("shifted_absdiff2", "regression", "Shifted absolute difference.",
        "f(x,y)=abs(x-y-1).", arch((2, 2, "relu"), (2, 1, "linear")), "P-E", "real x,y"),
    "threeway_controller1": Challenge("threeway_controller1", "multiclass", "Three-action controller with asymmetrical thresholds.",
        "Return class 1 when x < -0.75, class 2 when x > 1.5, else class 0. Equality stays class 0.", arch((1, 3, "linear")), "P-W", "real x"),
    "two_hats1": Challenge("two_hats1", "regression", "Sum of two nonoverlapping triangular bumps.",
        "f(x)=max(0,1-abs(x+1)) + 0.5*max(0,1-abs(x-2)).", arch((1, 6, "relu"), (6, 1, "linear")), "B-E", "real x"),
    "xor2_impossible": Challenge("xor2_impossible", "unsatisfiable", "Can one affine threshold represent XOR?",
        "On {0,1}^2 output 1 iff exactly one input is 1; only a single affine scalar logit is allowed.",
        arch((2, 1, "linear")), "B-E", "{0,1}^2"),
}


def oracle(task_id: str, x: list[float]) -> float | int:
    if task_id == "parity3": return int(sum(x) % 2 == 1)
    if task_id == "exactly2of4": return int(sum(x) == 2)
    if task_id == "select3": return int(x[1] if x[0] == 0 else x[2])
    if task_id == "slanted_diamond2": return int(abs(2*x[0]-x[1]-.5) + abs(x[0]+2*x[1]+.25) < 2.25)
    if task_id == "hat1": return max(0., 1-abs(2*x[0]-1))
    if task_id == "clip2": return min(2., max(-1., 3*x[0]-2*x[1]+.25))
    if task_id == "shifted_absdiff2": return abs(x[0]-x[1]-1)
    if task_id == "threeway_controller1": return 1 if x[0] < -.75 else (2 if x[0] > 1.5 else 0)
    if task_id == "two_hats1": return max(0.,1-abs(x[0]+1)) + .5*max(0.,1-abs(x[0]-2))
    if task_id == "xor2_impossible": return int(x[0] != x[1])
    raise ValueError(f"Unknown task {task_id}")


def _edge_inputs(task_id: str) -> list[list[float]]:
    if task_id == "slanted_diamond2":
        # Solve u=2x-y-0.5, v=x+2y+0.25 at selected points on |u|+|v|=2.25.
        points = []
        for u,v in ((2.25,0),(-2.25,0),(0,2.25),(0,-2.25),(0,0)):
            # 2x-y=u+0.5 ; x+2y=v-0.25; inverse [[2,1],[-1,2]]/5
            points.append([(2*(u+.5)+(v-.25))/5, (-(u+.5)+2*(v-.25))/5])
        return points
    if task_id == "hat1": return [[v] for v in (-2,-.00000001,0,.25,.5,.75,1,1.00000001,3)]
    if task_id == "clip2":
        # y=0, 3x+.25=-1 or 2
        return [[(-1-.25)/3,0.],[(2-.25)/3,0.],[0.,0.],[1.,2.],[-2.,-1.]]
    if task_id == "shifted_absdiff2": return [[0.,-1.],[1.,0.],[2.,1.],[-1.,3.]]
    if task_id == "threeway_controller1": return [[x] for x in (-2,-.750000001,-.75,-.749999999,0,1.499999999,1.5,1.500000001,3)]
    if task_id == "two_hats1": return [[x] for x in (-2,-1.5,-1,-.5,0,1,1.5,2,2.5,3)]
    return []


def examples(task_id: str, split: str, *, salt: str = "") -> list[tuple[list[float], int | float]]:
    task = TASKS[task_id]
    if split not in ("validation", "test", "ood", "edge"):
        raise ValueError("Unknown split")
    ndim = task.architecture[0][0]
    if task.domain.startswith("{0,1}"):
        cases = [list(map(float, bits)) for bits in product((0,1), repeat=ndim)]
        if split == "validation": cases = cases[::2]
        elif split in ("ood", "edge"): return []
    elif split == "edge":
        cases = _edge_inputs(task_id)
    else:
        key = f"direct-v1:{task_id}:{split}:{salt if split in ('test','ood') else ''}"
        rng = random.Random(int.from_bytes(sha256(key.encode()).digest()[:8], "big"))
        count = 64 if split == "validation" else 512
        limit = 3. if split != "ood" else 9.
        cases = [[rng.uniform(-limit,limit) for _ in range(ndim)] for _ in range(count)]
    return [(x, oracle(task_id,x)) for x in cases]


def evaluate(task_id: str, submission: dict, *, splits=("test","ood","edge"), salt: str = "") -> dict:
    """Score the original submitted JSON; no fitting, repair, or normalizing of weights."""
    task = TASKS[task_id]
    if not isinstance(submission, dict) or submission.get("schema_version") != 1 or submission.get("task") != task_id:
        raise ValueError("Incorrect schema_version or task")
    rule = CONDITIONS[task.condition]
    explanation = submission.get("explanation")
    if rule["explanation"] == "required" and (not isinstance(explanation, str) or not explanation.strip()):
        raise ValueError("Explanation is required for this task")
    if rule["explanation"] == "forbidden" and "explanation" in submission:
        raise ValueError("Explanation is forbidden for this task")
    if task.kind == "unsatisfiable":
        # This v1 negative control has a short known proof: four affine inequalities
        # w0+b>0, w1+b>0, b<=0, w0+w1+b<=0 imply contradiction.
        accepted = submission.get("status") == "unsatisfiable" and isinstance(submission.get("reason"), str) and len(submission["reason"].strip()) >= 12 and "network" not in submission
        if not accepted:
            raise ValueError("Negative control requires an unsatisfiable certificate, not an executable network")
        return {"task":task_id,"kind":task.kind,"pass":True,"certificate_status":"submitted; known XOR affine impossibility; proof text not machine-verified"}
    archspec = task.public()["architecture"]
    check_network(submission.get("network"), archspec)
    outcomes = {}
    for split in splits:
        pairs = examples(task_id, split, salt=salt)
        if not pairs: continue
        predictions = [predict(submission["network"], x, task.kind) for x,y in pairs]
        if task.kind == "regression":
            errors = [abs(float(p)-float(y)) for p,(_,y) in zip(predictions,pairs)]
            error = max(errors)
            passed = error <= 1e-6
            outcomes[split] = {"n":len(pairs),"metric":"max_abs_error","value":error,"mse":math.fsum(e*e for e in errors)/len(errors),"passed":passed}
        else:
            accuracy = sum(p==y for p,(_,y) in zip(predictions,pairs))/len(pairs)
            outcomes[split] = {"n":len(pairs),"metric":"accuracy","value":accuracy,"passed":accuracy == 1.}
    return {"task":task_id,"kind":task.kind,"pass":bool(outcomes) and all(x["passed"] for x in outcomes.values()),"splits":outcomes}


def score_bytes(task_id: str, raw: bytes, *, salt: str = "", splits=("test","ood","edge")) -> dict:
    """Count malformed submissions as failures and preserve exact input byte hash."""
    result = {"sha256":sha256(raw).hexdigest(),"size_bytes":len(raw),"pass":False,"evaluation":None,"error":None}
    try:
        submission=json.loads(raw)
        result["evaluation"] = evaluate(task_id, submission, splits=splits, salt=salt)
        result["pass"] = result["evaluation"]["pass"]
    except (ValueError, TypeError, KeyError, OverflowError, ArithmeticError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def report(directory: Path, *, salt: str = "") -> dict:
    manifest=json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or len(manifest.get("tasks",[])) != len(TASKS) or set(manifest["tasks"]) != set(TASKS):
        raise ValueError("Manifest must contain every task exactly once")
    records = []
    for task_id in manifest["tasks"]:
        raw=(directory / f"{task_id}.json").read_bytes()
        score=score_bytes(task_id,raw,salt=salt)
        records.append({"task":task_id,"condition":TASKS[task_id].condition,"artifact":f"{task_id}.json",**score})
    return {"schema_version":1,"status":"EXPLORATORY_SELF_SUBMISSIONS_NOT_BLIND_INDEPENDENT_TRIALS",
            "manifest_sha256":sha256((directory/"manifest.json").read_bytes()).hexdigest(),
            "participant":manifest["participant"],"tasks":len(records),"passed":sum(row["pass"] for row in records),
            "failed":sum(not row["pass"] for row in records),"records":records,
            "limitations":["Public task definitions and oracles were available to the participant", "Same assistant authored benchmark and candidate weights", "No reliable isolation of internal reasoning or external tool access", "Zero probe use in initial submissions, not proof of tool restrictions", "Tests are deterministic development tasks, not unseen or preregistered trials"]}


def main():
    parser=argparse.ArgumentParser(description="Public direct-weight synthesis task bank and evaluator")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("list")
    pr=sub.add_parser("prompt");pr.add_argument("task",choices=sorted(TASKS))
    ev=sub.add_parser("evaluate");ev.add_argument("candidate");ev.add_argument("--salt",default="")
    rp=sub.add_parser("report");rp.add_argument("directory");rp.add_argument("--output");rp.add_argument("--salt",default="")
    args=parser.parse_args()
    if args.command == "list": print(json.dumps([t.public() for t in TASKS.values()],indent=2,sort_keys=True))
    elif args.command == "prompt": print(json.dumps(TASKS[args.task].public(),indent=2,sort_keys=True))
    elif args.command == "evaluate":
        raw=Path(args.candidate).read_bytes();data=json.loads(raw)
        out=score_bytes(data["task"],raw,salt=args.salt);print(json.dumps(out,indent=2,sort_keys=True))
    else:
        out=report(Path(args.directory),salt=args.salt)
        if args.output: Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"tasks":out["tasks"],"passed":out["passed"],"failed":out["failed"]},sort_keys=True))

if __name__=="__main__": main()
