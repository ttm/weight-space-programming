"""Minimal deterministic, dependency-free feedforward network interpreter."""
from __future__ import annotations

import json
import math
from pathlib import Path

ACTIVATIONS = {"linear", "relu", "tanh", "sigmoid"}
MAX_PARAMETERS = 100_000


def validate_network(network: dict, input_dim: int | None = None, output_dim: int | None = None) -> int:
    if not isinstance(network, dict) or not isinstance(network.get("layers"), list) or not network["layers"]:
        raise ValueError("network.layers must be a nonempty list")
    if len(network["layers"]) > 16:
        raise ValueError("Too many layers (maximum 16)")
    previous = input_dim
    count = 0
    for li, layer in enumerate(network["layers"]):
        if not isinstance(layer, dict) or layer.get("activation") not in ACTIVATIONS:
            raise ValueError(f"Layer {li}: missing/invalid activation")
        weights, bias = layer.get("weights"), layer.get("bias")
        if not isinstance(weights, list) or not weights or not isinstance(bias, list) or len(bias) != len(weights):
            raise ValueError(f"Layer {li}: weights must have a row per bias")
        if len(weights) > 4096:
            raise ValueError("Layer width exceeds 4096")
        width = len(weights[0]) if isinstance(weights[0], list) else 0
        if not width or width > 4096 or (previous is not None and width != previous):
            raise ValueError(f"Layer {li}: incorrect input dimension")
        for row in weights:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"Layer {li}: ragged weight matrix")
        for val in [*bias, *(v for row in weights for v in row)]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Layer {li}: all parameters must be finite numbers")
        count += len(bias) * (width + 1)
        if count > MAX_PARAMETERS:
            raise ValueError(f"Network exceeds {MAX_PARAMETERS} parameters")
        previous = len(bias)
    if output_dim is not None and previous != output_dim:
        raise ValueError(f"Expected {output_dim} outputs, got {previous}")
    return count


def forward(network: dict, inputs: list[float]) -> list[float]:
    values = list(inputs)
    for layer in network["layers"]:
        next_values = []
        for row, bias in zip(layer["weights"], layer["bias"]):
            z = math.fsum([bias, *(a * b for a, b in zip(row, values))])
            act = layer["activation"]
            if act == "relu":
                z = max(0.0, z)
            elif act == "tanh":
                z = math.tanh(z)
            elif act == "sigmoid":
                z = (1 / (1 + math.exp(-z))) if z >= 0 else (math.exp(z) / (1 + math.exp(z)))
            next_values.append(z)
        values = next_values
    return values


def parse_candidate(data: bytes | str) -> dict:
    candidate = json.loads(data)
    if not isinstance(candidate, dict) or candidate.get("schema_version") != 1:
        raise ValueError("Candidate requires schema_version=1")
    if not isinstance(candidate.get("task"), str):
        raise ValueError("Candidate requires task ID")
    return candidate


def load_candidate(path: str | Path) -> dict:
    return parse_candidate(Path(path).read_bytes())


def predict(network: dict, inputs: list[float], kind: str) -> float | int:
    values = forward(network, inputs)
    if kind == "binary":
        return int(values[0] > 0)
    if kind == "multiclass":
        return max(range(len(values)), key=lambda i: values[i])
    if kind == "regression":
        return values[0]
    raise ValueError(f"Unknown task kind {kind}")
