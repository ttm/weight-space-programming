"""Public, deterministic task definitions. Not cryptographically hidden tests."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import random


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    inputs: int
    outputs: int
    kind: str
    oracle: str
    family: str


TASKS: dict[str, Task] = {
    "and2": Task("and2", "Compute AND on two bits", 2, 1, "binary", "1 iff both inputs are 1", "truth_table"),
    "xor2": Task("xor2", "Compute XOR on two bits", 2, 1, "binary", "1 iff exactly one input is 1", "truth_table"),
    "parity3": Task("parity3", "Odd parity on three bits", 3, 1, "binary", "1 iff the number of 1 bits is odd", "truth_table"),
    "majority5": Task("majority5", "Majority vote of five bits", 5, 1, "binary", "1 iff at least three inputs are 1", "truth_table"),
    "compare2": Task("compare2", "Strict comparison of two real numbers", 2, 1, "binary", "1 iff x > y; equality maps to 0", "continuous"),
    "absdiff2": Task("absdiff2", "Absolute difference of two real numbers", 2, 1, "regression", "|x-y|", "continuous"),
    "max2": Task("max2", "Maximum of two real numbers", 2, 1, "regression", "max(x,y)", "continuous"),
    "interval1": Task("interval1", "Membership in an open interval", 1, 1, "binary", "1 iff -0.5 < x < 0.5", "continuous"),
    "diamond2": Task("diamond2", "Membership in an open L1 ball", 2, 1, "binary", "1 iff |x| + |y| < 1", "continuous"),
    "thermostat1": Task("thermostat1", "Three-action deadband controller", 1, 3, "multiclass", "actions [hold,cool,heat]: cool if x < -0.4; heat if x > 0.4; otherwise hold", "continuous"),
}


def target(task: str, x: list[float]) -> float | int:
    if task == "and2": return int(x[0] == 1 and x[1] == 1)
    if task == "xor2": return int(sum(x) == 1)
    if task == "parity3": return int(sum(x) % 2 == 1)
    if task == "majority5": return int(sum(x) >= 3)
    if task == "compare2": return int(x[0] > x[1])
    if task == "absdiff2": return abs(x[0] - x[1])
    if task == "max2": return max(x)
    if task == "interval1": return int(abs(x[0]) < 0.5)
    if task == "diamond2": return int(sum(abs(v) for v in x) < 1)
    if task == "thermostat1": return 1 if x[0] < -0.4 else 2 if x[0] > 0.4 else 0
    raise KeyError(task)


def _seed(task: str, split: str) -> int:
    return int.from_bytes(sha256(f"wsp-v1:{task}:{split}".encode()).digest()[:8], "big")


def samples(task_name: str, split: str) -> list[tuple[list[float], float | int]]:
    if task_name not in TASKS:
        raise KeyError(f"Unknown task: {task_name}")
    task = TASKS[task_name]
    if split not in {"validation", "test", "ood", "edge"}:
        raise ValueError(f"Unknown split: {split}")
    if task.family == "truth_table":
        all_x = [list(t) for t in product((0, 1), repeat=task.inputs)]
        if split == "validation":
            # Intentionally incomplete feedback. Full truth table used for final test.
            all_x = [all_x[0], all_x[-1]] if len(all_x) == 4 else all_x[::max(1, len(all_x)//4)]
        elif split in {"ood", "edge"}:
            return []
        return [(x, target(task_name, x)) for x in all_x]
    if split == "edge":
        edges = {
            "compare2": [[0., 0.], [1., 1.], [-1., -1.], [1., -1.]],
            "absdiff2": [[0., 0.], [1., 1.], [-1., 1.]],
            "max2": [[0., 0.], [1., 1.], [-1., 1.]],
            "interval1": [[-0.5], [0.5], [0.], [-1.], [1.]],
            "diamond2": [[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [0., 0.]],
            "thermostat1": [[-0.4], [0.4], [0.], [-0.4000001], [0.4000001]],
        }
        xs = edges[task_name]
    else:
        rng = random.Random(_seed(task_name, split))
        n = 64 if split == "validation" else 512
        span = 10.0 if split == "ood" else 2.0
        xs = [[rng.uniform(-span, span) for _ in range(task.inputs)] for _ in range(n)]
    return [(x, target(task_name, x)) for x in xs]
