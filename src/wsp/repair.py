"""Explicit, auditable sparse weight edits. No optimizers or training loops."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .evaluation import evaluate
from .network import validate_network
from .tasks import TASKS


def apply_patch(candidate: dict, patch: dict) -> dict:
    if patch.get("schema_version") != 1 or patch.get("task") != candidate.get("task"):
        raise ValueError("Patch schema/task mismatch")
    edits = patch.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("Patch requires a nonempty edits list")
    revised = deepcopy(candidate)
    network = revised["network"]
    task = TASKS[revised["task"]]
    validate_network(network, task.inputs, task.outputs)
    used = set()
    for edit in edits:
        li, kind, row = edit["layer"], edit["kind"], edit["row"]
        if kind not in {"weights", "bias"}:
            raise ValueError("Only individual weights or biases may be edited")
        if isinstance(li, bool) or not isinstance(li, int) or li < 0 or li >= len(network["layers"]):
            raise ValueError("Invalid layer index")
        matrix = network["layers"][li][kind]
        if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= len(matrix):
            raise ValueError("Invalid row index")
        if kind == "weights":
            col = edit["column"]
            if isinstance(col, bool) or not isinstance(col, int) or col < 0 or col >= len(matrix[row]):
                raise ValueError("Invalid column index")
            location = (li, kind, row, col)
            old = matrix[row][col]
        else:
            location = (li, kind, row)
            old = matrix[row]
        if location in used:
            raise ValueError("Duplicate edit location")
        used.add(location)
        if old != edit["from"]:
            raise ValueError(f"Expected value {edit['from']}, found {old}")
        if kind == "weights":
            matrix[row][col] = edit["to"]
        else:
            matrix[row] = edit["to"]
    validate_network(network, task.inputs, task.outputs)
    revised["method"] = "direct_sparse_weight_edit"
    revised["explanation"] = patch.get("explanation", "Explicit numeric weight edit")
    return revised


def repair_report(source: dict, patch: dict) -> dict:
    before = evaluate(source)
    after_candidate = apply_patch(source, patch)
    after = evaluate(after_candidate)
    return {"task": source["task"], "parameters": before["parameters"], "edited_parameters": len(patch["edits"]),
            "fraction_edited": len(patch["edits"]) / before["parameters"],
            "before": before, "after": after}


def load_patch(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))
