"""Novel, deterministically parameterized task families with constructive witnesses.

These are *development* tasks. The participant sees the full specification, never the
reference weights. Salted evaluation inputs are withheld by the evaluator in live trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import math
import os
import random

from .architecture import from_network, check_network
from .network import predict

FAMILIES = ("dnf4", "hinge1", "l1ball2", "threshold4")


def _rng(family: str, seed: int) -> random.Random:
    key = sha256(f"wsp-task-v1:{family}:{seed}".encode()).digest()
    return random.Random(int.from_bytes(key[:8], "big"))


@dataclass(frozen=True)
class GeneratedTask:
    family: str
    seed: int
    task_id: str
    kind: str
    description: str
    spec: dict
    architecture: dict
    reference_network: dict

    def public(self) -> dict:
        """Only this object may be sent to a participant."""
        return {"schema_version": 1, "task": self.task_id, "kind": self.kind,
                "description": self.description, "spec": self.spec,
                "architecture": self.architecture,
                "threshold_rule": "binary label 1 iff final logit > 0" if self.kind == "binary" else "regression with linear scalar output"}

    def reference(self) -> dict:
        return {"schema_version": 1, "task": self.task_id, "network": self.reference_network,
                "method": "deterministic_symbolic_compiler", "provenance": {"generator": "wsp.generated.v1"}}


def make_task(family: str, seed: int) -> GeneratedTask:
    if family not in FAMILIES or type(seed) is not int or seed < 0:
        raise ValueError(f"Invalid family or seed: {family!r} {seed!r}")
    rng = _rng(family, seed)
    if family == "dnf4":
        # Three non-repeated conjunctions of three randomly signed literals.
        clauses, seen = [], set()
        while len(clauses) < 3:
            selected = tuple(sorted(rng.sample(range(4), 3)))
            clause = tuple((i, rng.choice((0, 1))) for i in selected)
            if clause not in seen:
                seen.add(clause); clauses.append(clause)
        rows, biases = [], []
        for clause in clauses:
            row = [0.] * 4
            neg = 0
            for i, bit in clause:
                row[i] = 1. if bit else -1.
                neg += (bit == 0)
            rows.append(row)
            biases.append(float(neg - 2))
        net = {"layers": [
            {"weights": rows, "bias": biases, "activation": "relu"},
            {"weights": [[1., 1., 1.]], "bias": [-0.5], "activation": "linear"}]}
        spec = {"clauses": [[[i, bit] for i, bit in clause] for clause in clauses],
                "meaning": "OR of clauses; each clause is AND of its listed (zero-indexed bit, required value) pairs"}
        desc = "Given four input bits, return 1 iff any specified three-literal conjunction holds."
        kind = "binary"
    elif family == "hinge1":
        knots = sorted(rng.sample([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], 3))
        coeffs = [rng.choice((-2, -1, 1, 2)) for _ in knots]
        c = rng.choice((-1., -0.5, 0., 0.5, 1.))
        net = {"layers": [
            {"weights": [[1.] for _ in knots], "bias": [-float(t) for t in knots], "activation": "relu"},
            {"weights": [[float(a) for a in coeffs]], "bias": [c], "activation": "linear"}]}
        spec = {"knots": knots, "coefficients": coeffs, "constant": c,
                "formula": "constant + sum(coefficients[i]*max(0,x-knots[i]))"}
        desc = "Implement the specified 1D piecewise-linear function, for all real x."
        kind = "regression"
    elif family == "l1ball2":
        center = [rng.randint(-4, 4) / 4, rng.randint(-4, 4) / 4]
        radius = rng.choice((0.5, 1., 1.5, 2.))
        net = {"layers": [
            {"weights": [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
             "bias": [-center[0], center[0], -center[1], center[1]], "activation": "relu"},
            {"weights": [[-1., -1., -1., -1.]], "bias": [radius], "activation": "linear"}]}
        spec = {"center": center, "radius": radius,
                "formula": "1 iff abs(x-center[0])+abs(y-center[1]) < radius; equality is 0"}
        desc = "Classify whether a point is strictly inside a specified 2D L1 ball."
        kind = "binary"
    else:
        weights = [rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(4)]
        bias = rng.choice((-3, -2, -1, 0, 1, 2, 3)) + .5  # no exact ties on bits
        net = {"layers": [{"weights": [list(map(float, weights))], "bias": [bias], "activation": "linear"}]}
        spec = {"weights": weights, "bias": bias,
                "formula": "1 iff sum(weights[i]*x[i])+bias > 0; inputs are four bits"}
        desc = "Classify a weighted Boolean threshold with strict greater-than comparison."
        kind = "binary"
    key = f"{family}:{seed}"
    ident = "gen_" + sha256(key.encode()).hexdigest()[:16]
    return GeneratedTask(family, seed, ident, kind, desc, spec, from_network(net), net)


def oracle(task: GeneratedTask, x: list[float]) -> int | float:
    s = task.spec
    if task.family == "dnf4":
        return int(any(all(int(x[i]) == bit for i, bit in clause) for clause in s["clauses"]))
    if task.family == "hinge1":
        return s["constant"] + math.fsum(a * max(0., x[0] - t) for t, a in zip(s["knots"], s["coefficients"]))
    if task.family == "l1ball2":
        return int(math.fsum(abs(v-c) for v, c in zip(x, s["center"])) < s["radius"])
    return int(math.fsum(w*v for w, v in zip(s["weights"], x)) + s["bias"] > 0)


def _sample_seed(task: GeneratedTask, split: str, salt: str) -> int:
    key = f"wsp-test-v1:{task.family}:{task.seed}:{split}"
    if split in ("test", "ood"):
        key += f":{salt}"
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


def examples(task: GeneratedTask, split: str, *, salt: str = "") -> list[tuple[list[float], int | float]]:
    if split not in ("validation", "test", "ood", "edge"):
        raise ValueError("Invalid split")
    if task.family in ("dnf4", "threshold4"):
        xs = [list(xs) for xs in product((0., 1.), repeat=4)]
        if split == "validation":
            rng = random.Random(_sample_seed(task, split, salt)); rng.shuffle(xs)
            xs = xs[:8]
        if split in ("edge", "ood"):
            return []
    elif split == "edge":
        if task.family == "hinge1":
            xs = [[k] for k in task.spec["knots"]]
            xs += [[k - 1e-8] for k in task.spec["knots"]] + [[k + 1e-8] for k in task.spec["knots"]]
        else:
            cx, cy = task.spec["center"]; r = task.spec["radius"]
            xs = [[cx, cy], [cx+r, cy], [cx-r,cy], [cx,cy+r], [cx,cy-r],
                  [cx+r-1e-8,cy], [cx+r+1e-8,cy]]
    else:
        rng = random.Random(_sample_seed(task, split, salt))
        count = 64 if split == "validation" else 256
        span = 2. if split != "ood" else 5.
        ndim = 1 if task.family == "hinge1" else 2
        xs = [[rng.uniform(-span,span) for _ in range(ndim)] for _ in range(count)]
    return [(x, oracle(task,x)) for x in xs]


def evaluate_generated(task: GeneratedTask, candidate: dict, splits=("test","ood","edge"), *, salt: str = "") -> dict:
    if candidate.get("task") != task.task_id:
        raise ValueError("Candidate task ID mismatch")
    check_network(candidate.get("network"), task.architecture)
    records = {}
    for split in splits:
        data = examples(task, split, salt=salt)
        if not data:
            continue
        predictions = [predict(candidate["network"], x, task.kind) for x, _ in data]
        if task.kind == "binary":
            value = sum(a == b for a, (_, b) in zip(predictions, data)) / len(data)
            records[split] = {"metric": "accuracy", "n": len(data), "value": value, "passed": value == 1.}
        else:
            errors = [abs(a-b) for a, (_, b) in zip(predictions, data)]
            mse = math.fsum(v*v for v in errors)/len(errors)
            records[split] = {"metric": "mse", "n": len(data), "value": mse,
                              "max_abs_error": max(errors), "passed": max(errors) <= 1e-6}
    return {"task": task.task_id, "family": task.family, "splits": records,
            "all_passed": bool(records) and all(v["passed"] for v in records.values())}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Show generated research task. Reference weights optional, only for development.")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", action="store_true", help="Print private constructive witness (DEV ONLY)")
    args = parser.parse_args()
    task = make_task(args.family,args.seed)
    print(json.dumps(task.reference() if args.reference else task.public(), indent=2))

if __name__ == "__main__": main()
