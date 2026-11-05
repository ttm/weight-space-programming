"""Exact architecture contracts for weight-space experiments.

Input/output dimensions alone do not establish a matched-architecture comparison.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


def from_network(network: dict) -> dict:
    """Return the complete layer topology, without disclosing parameter values."""
    layers = network.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("network.layers must be a nonempty list")
    return {"layers": [
        {"inputs": len(layer["weights"][0]), "outputs": len(layer["weights"]),
         "activation": layer["activation"]} for layer in layers
    ]}


def validate_architecture(architecture: dict) -> dict:
    if not isinstance(architecture, dict) or set(architecture) != {"layers"}:
        raise ValueError("Architecture must have only a layers field")
    layers = architecture["layers"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= 16:
        raise ValueError("Architecture requires 1..16 layers")
    total = 0
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"inputs", "outputs", "activation"}:
            raise ValueError(f"Invalid architecture layer {i}")
        a, b = layer["inputs"], layer["outputs"]
        if type(a) is not int or type(b) is not int or not 1 <= a <= 4096 or not 1 <= b <= 4096:
            raise ValueError(f"Invalid layer dimensions at {i}")
        if layer["activation"] not in {"linear", "relu", "sigmoid", "tanh"}:
            raise ValueError(f"Invalid activation at {i}")
        if i and a != layers[i - 1]["outputs"]:
            raise ValueError(f"Layers {i-1} and {i} are disconnected")
        total += b * (a + 1)
        if total > 100_000:
            raise ValueError("Architecture exceeds 100000 parameters")
    return architecture


def check_network(network: dict, architecture: dict) -> None:
    """Reject hidden-width, depth and activation deviations before scoring."""
    from .network import validate_network
    validate_architecture(architecture)
    validate_network(network, architecture["layers"][0]["inputs"], architecture["layers"][-1]["outputs"])
    if from_network(network) != architecture:
        raise ValueError("Candidate topology/activations differ from frozen trial architecture")


def architecture_hash(architecture: dict) -> str:
    validate_architecture(architecture)
    raw = json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()
