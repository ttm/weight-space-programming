#!/usr/bin/env python3
"""Commit reproducible negative/positive reading controls, without model inference."""
from __future__ import annotations
import json
from pathlib import Path
from wsp.balanced_reading import FAMILIES, make_bank, evaluate_predictions


def main() -> None:
    bank, key = make_bank(seed=20261105, per_family=2)
    ids = [i["id"] for i in bank["items"]]
    oracle = {ident: value["truth_table"] for ident, value in key["answers"].items()}
    constant = {ident: "0"*16 for ident in ids}
    oscore = evaluate_predictions(key,oracle)
    cscore = evaluate_predictions(key,constant)
    result = {"schema_version": 1, "status": "DEVELOPMENT_CONTROLS_NOT_MODEL_INFERENCE",
              "seed":20261105,"per_family":2,"families":list(FAMILIES),
              "unique_architectures":len({json.dumps(i["architecture"],sort_keys=True) for i in bank["items"]}),
              "n_networks":len(bank["items"]),"n_functions":len(bank["items"])//2,
              "positive_outputs_per_network":list(sorted({a["ones"] for a in key["answers"].values()})),
              "oracle":{"exact":oscore["exact"],"pairs_both_exact":oscore["pairs_both_exact"],
                        "mean_bit_accuracy":oscore["mean_bit_accuracy"]},
              "constant_zero":{"exact":cscore["exact"],"pairs_both_exact":cscore["pairs_both_exact"],
                               "pairs_agree":cscore["pairs_agree"],
                               "pairs_trivial_agreement":cscore["pairs_trivial_agreement"],
                               "mean_bit_accuracy":cscore["mean_bit_accuracy"]}}
    assert result["unique_architectures"] == 1
    assert result["positive_outputs_per_network"] == [8]
    assert result["oracle"]["pairs_both_exact"] == result["n_functions"]
    assert result["constant_zero"]["mean_bit_accuracy"] == .5
    path = Path("results/balanced_reading_controls.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":main()
