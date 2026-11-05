"""Budget-accounted random and zeroth-order search on generated tasks."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import random

from .generated import GeneratedTask, evaluate_generated
from .network import validate_network


def _seed(*args) -> int:
    return int.from_bytes(sha256(": ".join(map(str,args)).encode()).digest()[:8], "big")


def sample_network(architecture: dict, rng: random.Random, *, scale: float = 1.) -> dict:
    return {"layers": [{"weights": [[float(f"{rng.gauss(0,scale):.12g}") for _ in range(layer["inputs"])]
                                    for _ in range(layer["outputs"])],
                        "bias": [float(f"{rng.gauss(0,scale):.12g}") for _ in range(layer["outputs"])],
                        "activation": layer["activation"]} for layer in architecture["layers"]]}


def _score(task: GeneratedTask, network: dict, salt: str) -> tuple[float, bool]:
    result = evaluate_generated(task, {"task":task.task_id,"network":network}, ("validation",), salt=salt)
    v = result["splits"]["validation"]
    score = v["value"] if v["metric"] == "accuracy" else -v["value"]
    return score, result["all_passed"]


def search(task: GeneratedTask, *, method: str, budget: int, seed: int, salt: str = "", scale: float = 1.) -> dict:
    """Exactly `budget` validation evaluations, including the initial proposal.

    Hill climb evaluates perturbations of the incumbent, accepting non-worsening
    scores. No hidden oracle calls or gradient information are used. Both methods
    expose the *same validation scalar*, which may conceal individual failures.
    """
    if method not in ("random", "hill_climb") or type(budget) is not int or budget < 1:
        raise ValueError("Choose random or hill_climb and budget >= 1")
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be positive finite")
    rng = random.Random(_seed("baseline",method,task.family,task.seed,seed,budget))
    best = None
    best_score = -float("inf")
    first_passing_probe = None
    for step in range(budget):
        if method == "random" or best is None:
            trial = sample_network(task.architecture,rng,scale=scale)
        else:
            trial = deepcopy(best)
            sigma = scale * 0.3
            for layer in trial["layers"]:
                for row in layer["weights"]:
                    for i in range(len(row)):
                        row[i] = float(f"{row[i] + rng.gauss(0,sigma):.12g}")
                for i in range(len(layer["bias"])):
                    layer["bias"][i] = float(f"{layer['bias'][i] + rng.gauss(0,sigma):.12g}")
        score, passes_validation = _score(task, trial, salt)
        if first_passing_probe is None and passes_validation: first_passing_probe = step+1
        if score >= best_score:
            best_score, best = score, trial
    final = evaluate_generated(task,{"task":task.task_id,"network":best},salt=salt)
    return {"method":method,"budget":budget,"seed":seed,"validation_score":best_score,
            "validation_calls":budget,"first_passing_probe":first_passing_probe,
            "final":final, "pass":final["all_passed"]}


def compare(*, seeds: int = 4, budgets: tuple[int,...] = (1,3), salt: str = "") -> dict:
    from .generated import FAMILIES, make_task
    if not 1 <= seeds <= 100 or any(not 1 <= b <= 10000 for b in budgets):
        raise ValueError("Out of bounds")
    records = []
    for family in FAMILIES:
        for t_seed in range(seeds):
            task = make_task(family, t_seed)
            reference = evaluate_generated(task, task.reference(), salt=salt)
            if not reference["all_passed"]:
                raise AssertionError("Constructive witness failed")
            for budget in budgets:
                for method in ("random", "hill_climb"):
                    records.append({"family":family,"task_seed":t_seed,
                                    **search(task,method=method,budget=budget,seed=t_seed,salt=salt)})
    summary=[]
    for method in ("random","hill_climb"):
        for budget in budgets:
            subset=[v for v in records if v["method"]==method and v["budget"]==budget]
            summary.append({"method":method,"budget":budget,"trials":len(subset),
                            "perfect":sum(v["pass"] for v in subset)})
    return {"schema_version":1,"status":"development_baselines_not_llm_trials",
            "seeds_per_family":seeds,"budgets":list(budgets),"summary":summary,"records":records}


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description="Reproduce generated-task search baselines")
    parser.add_argument("--seeds",type=int,default=4)
    parser.add_argument("--budgets",type=int,nargs="+",default=[1,3])
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    report=compare(seeds=args.seeds,budgets=tuple(args.budgets))
    data=json.dumps(report,sort_keys=True,indent=2)+"\n"
    if args.output:
        from pathlib import Path
        p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
    print(json.dumps({"summary":report["summary"],"records":len(report["records"])},indent=2))

if __name__=="__main__":main()
