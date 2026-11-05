"""Balanced, architecture-matched weight-reading tests with negative controls.

The answer key is NEVER included in a participant prompt. This is a public
exploratory bank, not a genuinely sealed or preregistered benchmark.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import random

from .architecture import check_network, from_network
from .network import predict
from .weight_reading import transform, fingerprint

FAMILIES = ("parity4", "xor_pair", "mux3", "threshold4")
INPUTS = list(product((0, 1), repeat=4))


def _seed(*parts: object) -> int:
    return int.from_bytes(sha256(": ".join(map(str, parts)).encode()).digest()[:8], "big")


def _relu_layer(rows: list[list[float]], bias: list[float], output: list[float], intercept: float) -> dict:
    assert len(rows) == len(bias) == len(output) == 4
    return {"layers": [
        {"weights": rows, "bias": bias, "activation": "relu"},
        {"weights": [output], "bias": [intercept], "activation": "linear"},
    ]}


def _truth_table(net: dict) -> str:
    return "".join(str(predict(net, list(bits), "binary")) for bits in INPUTS)


def construct(family: str, seed: int) -> dict:
    """Directly construct four different *balanced* Boolean circuit families.

    Every network has exactly the same 4 -> ReLU(4) -> linear(1) topology.
    Neither the family name nor the construction seed goes to the participant.
    """
    if family not in FAMILIES or type(seed) is not int or seed < 0:
        raise ValueError("Unknown family or invalid seed")
    rng = random.Random(_seed("balanced-v1", family, seed))
    if family == "parity4":
        rows = [[1., 1., 1., 1.] for _ in range(4)]
        net = _relu_layer(rows, [0., -1., -2., -3.], [1., -2., 2., -2.], 0.)
        # Reverse the target parity in half of the seeds, keeping the 8/8 balance.
        if seed % 2:
            net["layers"][1]["weights"][0] = [-v for v in net["layers"][1]["weights"][0]]
            net["layers"][1]["bias"][0] = .5
        # IMPORTANT: complement above must be 1 iff parity is even; with +0.5 it is
        # positive for f=0, negative for f=1 (since f is exactly 0 or 1).
    elif family == "xor_pair":
        a, b, c, d = rng.sample(range(4), 4)
        rows = [[0.] * 4 for _ in range(4)]
        rows[0][a], rows[0][b] = 1., -1.
        rows[1][a], rows[1][b] = -1., 1.
        rows[2][c] = 1.
        rows[3][d] = -1.
        net = _relu_layer(rows, [0., 0., 0., 1.], [1., 1., 0., 0.], -.5)
    elif family == "mux3":
        s, a, b, unused = rng.sample(range(4), 4)
        rows = [[0.] * 4 for _ in range(4)]
        rows[0][b] = 1.
        rows[1][a], rows[1][s] = 1., 1.
        rows[2][b], rows[2][s] = 1., 1.
        rows[3][unused] = 1.
        net = _relu_layer(rows, [0., -1., -1., .25], [1., 1., -1., 0.], -.5)
    else:
        # Search *construction parameters* deterministically to ensure every
        # threshold network has eight positives and depends on >= 3 input bits.
        # This is an oracle/reference compiler, NOT participant optimization.
        options = [-5., -3., -2., 2., 3., 5.]
        for _ in range(10000):
            w = [rng.choice(options) for _ in range(4)]
            values = sorted(sum(a * bit for a, bit in zip(w, bits)) for bits in INPUTS)
            if values[7] == values[8]:
                continue
            threshold = (values[7] + values[8]) / 2
            table = tuple(int(sum(a * bit for a, bit in zip(w, bits)) > threshold) for bits in INPUTS)
            dependent = sum(any(table[i] != table[i ^ (1 << (3-j))] for i in range(16)) for j in range(4))
            if dependent >= 3:
                break
        else:
            raise RuntimeError("Unable to construct nontrivial balanced threshold")
        rows = [[float(i == j) for i in range(4)] for j in range(4)]
        net = _relu_layer(rows, [0.] * 4, w, -threshold)
    check_network(net, from_network(net))
    table = _truth_table(net)
    if len(table) != 16 or table.count("1") != 8:
        raise AssertionError(f"Unbalanced reference for {family}: {table}")
    return net


def make_bank(*, seed: int = 20261105, per_family: int = 2) -> tuple[dict, dict]:
    """Return a public randomized presentation bank and a separate private key."""
    if type(seed) is not int or seed < 0 or type(per_family) is not int or not 1 <= per_family <= 20:
        raise ValueError("Invalid seed or per_family")
    rng = random.Random(_seed("presentation", seed))
    items, key_rows = [], {}
    for family in FAMILIES:
        for rep in range(per_family):
            reference_seed = seed + rep * 104729
            original = construct(family, reference_seed)
            altered = transform(original, seed=_seed("gauge", family, reference_seed))
            truth = _truth_table(original)
            assert _truth_table(altered) == truth
            assert fingerprint(original) == fingerprint(altered)
            group = sha256(f"{seed}:{family}:{rep}".encode()).hexdigest()[:16]
            for variant, net in (("original", original), ("gauge", altered)):
                ident = "br_" + sha256(f"{group}:{variant}".encode()).hexdigest()[:14]
                items.append({"id": ident, "network": deepcopy(net),
                              "architecture": from_network(net),
                              "input_semantics": "four binary inputs x0,x1,x2,x3; lexicographic 0000..1111",
                              "output_semantics": "class 1 iff final scalar > 0; otherwise 0"})
                key_rows[ident] = {"group": group, "variant": variant, "family": family,
                                   "reference_seed": reference_seed, "truth_table": truth,
                                   "ones": 8, "reference_fingerprint": fingerprint(original)}
    rng.shuffle(items)
    bank = {"schema_version": 1, "status": "exploratory_not_sealed", "seed": seed,
            "topology": "4 -> ReLU(4) -> linear(1)", "items": items}
    key = {"schema_version": 1, "status": "PRIVATE_REFERENCE_DO_NOT_PROMPT", "answers": key_rows}
    return bank, key


def evaluate_predictions(key: dict, guesses: dict[str, str | None]) -> dict:
    """Score strict 16-bit responses, including degenerate/constant predictors."""
    answers = key["answers"]
    if set(guesses) != set(answers):
        raise ValueError("Missing or unexpected prediction IDs")
    rows, grouped = [], {}
    for ident in sorted(answers):
        expected = answers[ident]["truth_table"]
        predicted = guesses[ident]
        valid = isinstance(predicted, str) and len(predicted) == 16 and set(predicted) <= {"0", "1"}
        exact = bool(valid and predicted == expected)
        accuracy = sum(a == b for a, b in zip(predicted, expected)) / 16 if valid else 0.
        row = {"id": ident, "group": answers[ident]["group"], "variant": answers[ident]["variant"],
               "family": answers[ident]["family"], "valid": valid, "exact": exact,
               "bit_accuracy": accuracy, "constant_prediction": valid and len(set(predicted)) == 1}
        rows.append(row)
        grouped.setdefault(row["group"], {})[row["variant"]] = (predicted if valid else None, exact)
    pairs = []
    for group in sorted(grouped):
        originals, gauges = grouped[group].get("original"), grouped[group].get("gauge")
        if originals is None or gauges is None:
            raise ValueError("Incomplete equivalence pair")
        p, q = originals[0], gauges[0]
        pairs.append({"group": group, "both_exact": originals[1] and gauges[1],
                      "agree": p is not None and q is not None and p == q,
                      "trivial_agreement": p is not None and q is not None and p == q and len(set(p)) == 1})
    return {"n_networks": len(rows), "n_functions": len(pairs),
            "valid": sum(x["valid"] for x in rows), "exact": sum(x["exact"] for x in rows),
            "mean_bit_accuracy": sum(x["bit_accuracy"] for x in rows) / len(rows),
            "constant_predictions": sum(x["constant_prediction"] for x in rows),
            "pairs_agree": sum(x["agree"] for x in pairs),
            "pairs_both_exact": sum(x["both_exact"] for x in pairs),
            "pairs_trivial_agreement": sum(x["trivial_agreement"] for x in pairs),
            "all_zero_baseline": {"exact": 0, "mean_bit_accuracy": .5, "pairs_agree": len(pairs),
                                  "pairs_both_exact": 0},
            "rows": rows, "pairs": pairs}


def main() -> None:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    bank, key = make_bank(seed=a.seed, per_family=a.per_family)
    for path, value in ((a.bank, bank), (a.key, key)):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"networks": len(bank["items"]), "functions": len(bank["items"]) // 2}))


if __name__ == "__main__":
    main()
