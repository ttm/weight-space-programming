"""Scoring, architecture-preserving baselines, and non-optimization stress tests."""
from __future__ import annotations

from copy import deepcopy
import math
import random

from .network import forward, predict, validate_network
from .tasks import TASKS, samples


def evaluate(candidate: dict, splits=("test", "ood", "edge")) -> dict:
    task_name = candidate["task"]
    task = TASKS[task_name]
    params = validate_network(candidate["network"], task.inputs, task.outputs)
    outcomes = {}
    for split in splits:
        pairs = samples(task_name, split)
        if not pairs:
            continue
        actual = [predict(candidate["network"], x, task.kind) for x, _ in pairs]
        expected = [y for _, y in pairs]
        if task.kind == "regression":
            metric, value = "mse", math.fsum((a - y) ** 2 for a, y in zip(actual, expected)) / len(pairs)
            passed = value <= 1e-12
        else:
            metric, value = "accuracy", sum(a == y for a, y in zip(actual, expected)) / len(pairs)
            passed = value == 1.0
        outcomes[split] = {"n": len(pairs), "metric": metric, "value": value, "passed": passed}
    return {"task": task_name, "parameters": params, "splits": outcomes,
            "all_passed": bool(outcomes) and all(s["passed"] for s in outcomes.values())}


def _parameters(network: dict):
    for layer in network["layers"]:
        for row in layer["weights"]:
            for i in range(len(row)):
                yield row, i
        for i in range(len(layer["bias"])):
            yield layer["bias"], i


def random_like(network: dict, seed: int) -> dict:
    rng = random.Random(seed)
    clone = deepcopy(network)
    for array, index in _parameters(clone):
        array[index] = rng.gauss(0, 1)
    return clone


def quantize(network: dict, digits: int) -> dict:
    clone = deepcopy(network)
    for array, index in _parameters(clone):
        array[index] = round(array[index], digits)
    return clone


def gaussian_noise(network: dict, sigma: float, seed: int) -> dict:
    clone, rng = deepcopy(network), random.Random(seed)
    for array, index in _parameters(clone):
        array[index] += rng.gauss(0, sigma)
    return clone


def permute_hidden(network: dict) -> dict:
    """Exchange first two hidden neurons, including their outgoing columns."""
    clone = deepcopy(network)
    if len(clone["layers"]) < 2 or len(clone["layers"][0]["bias"]) < 2:
        raise ValueError("Need a hidden layer with >=2 neurons")
    first, second = clone["layers"][:2]
    for key in ("weights", "bias"):
        first[key][0], first[key][1] = first[key][1], first[key][0]
    for row in second["weights"]:
        row[0], row[1] = row[1], row[0]
    return clone


def rescale_relu(network: dict, scale: float = 3.) -> dict:
    """Positive homogeneity: ReLU(scale*z) = scale*ReLU(z)."""
    clone = deepcopy(network)
    if len(clone["layers"]) < 2 or clone["layers"][0]["activation"] != "relu" or scale <= 0:
        raise ValueError("Need a ReLU hidden layer and positive scale")
    first, second = clone["layers"][:2]
    first["weights"][0] = [scale * v for v in first["weights"][0]]
    first["bias"][0] *= scale
    for row in second["weights"]:
        row[0] /= scale
    return clone


def equivalent(network_a: dict, network_b: dict, task_name: str, tolerance: float = 1e-9) -> bool:
    for split in ("validation", "test", "ood", "edge"):
        for x, _ in samples(task_name, split):
            av, bv = forward(network_a, x), forward(network_b, x)
            if any(not math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance) for a, b in zip(av, bv)):
                return False
    return True


def stress(candidate: dict) -> dict:
    base = evaluate(candidate)
    task, network = candidate["task"], candidate["network"]
    output = {"task": task, "original": base, "quantized": {}, "noise": {}, "invariances": {}}
    for digits in (3, 1, 0):
        output["quantized"][str(digits)] = evaluate({**candidate, "network": quantize(network, digits)})
    for sigma in (0.01, 0.1, 0.5):
        trials = [evaluate({**candidate, "network": gaussian_noise(network, sigma, seed)}) for seed in range(20)]
        output["noise"][str(sigma)] = {
            split: {"mean": math.fsum(t["splits"][split]["value"] for t in trials) / len(trials),
                    "min": min(t["splits"][split]["value"] for t in trials),
                    "max": max(t["splits"][split]["value"] for t in trials)}
            for split in base["splits"]}
    for name, transform in (("hidden_neuron_permutation", permute_hidden), ("relu_positive_rescaling", rescale_relu)):
        try:
            output["invariances"][name] = equivalent(network, transform(network), task)
        except ValueError:
            output["invariances"][name] = None
    return output


def random_baseline(candidate: dict, trials: int = 30) -> dict:
    base = evaluate(candidate)
    scores = [evaluate({**candidate, "network": random_like(candidate["network"], seed)}) for seed in range(trials)]
    return {"trials": trials, "splits": {split: {
        "metric": base["splits"][split]["metric"],
        "mean": math.fsum(v["splits"][split]["value"] for v in scores) / trials,
        "min": min(v["splits"][split]["value"] for v in scores),
        "max": max(v["splits"][split]["value"] for v in scores),
        "perfect_trials": sum(v["splits"][split]["passed"] for v in scores),
    } for split in base["splits"]}}
