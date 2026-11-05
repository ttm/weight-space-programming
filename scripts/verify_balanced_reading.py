#!/usr/bin/env python3
"""Replay a balanced-reading artifact without model downloads or inference."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

from wsp.architecture import check_network
from wsp.balanced_reading import _truth_table, evaluate_predictions, make_bank
from run_balanced_reading import decode, prompt_for


def verify(report: dict) -> dict:
    if report.get("schema_version") != 1 or report.get("feedback_probes") != 0:
        raise ValueError("Unknown report or unexpected feedback")
    bank, key = make_bank(seed=report["seed"], per_family=report["per_family"])
    if report["public_bank"] != bank or report["answer_key_unsealed_after_run"] != key:
        raise ValueError("Public bank or oracle was modified")
    by_id = {i["id"]: i for i in bank["items"]}
    if len(report["records"]) != len(by_id) or set(i["id"] for i in report["records"]) != set(by_id):
        raise ValueError("Missing/duplicate/unexpected participant output")
    guesses = {}
    for trial in report["records"]:
        ident = trial["id"]
        item = by_id[ident]
        check_network(item["network"], item["architecture"])
        assert _truth_table(item["network"]) == key["answers"][ident]["truth_table"]
        expected_prompt = prompt_for(item, report["condition"])
        if trial["prompt"] != expected_prompt or sha256(expected_prompt.encode()).hexdigest() != trial["prompt_sha256"]:
            raise ValueError("Prompt content or hash mismatch")
        if sha256(trial["raw"].encode()).hexdigest() != trial["raw_sha256"]:
            raise ValueError("Raw response hash mismatch")
        parsed, error = decode(trial["raw"])
        if (parsed, error) != (trial["parsed"], trial["error"]):
            raise ValueError("Parser result mismatch")
        guesses[ident] = parsed
    summary = evaluate_predictions(key, guesses)
    if summary != report["summary"]:
        raise ValueError("Score mismatch")
    return {k: summary[k] for k in ("n_networks", "n_functions", "valid", "exact",
                                    "pairs_agree", "pairs_both_exact", "pairs_trivial_agreement",
                                    "mean_bit_accuracy", "constant_predictions")}


def main():
    for path in sys.argv[1:]:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        print(json.dumps({"artifact":path,"verified":verify(report)},sort_keys=True))
    if len(sys.argv)==1:
        raise SystemExit("Usage: python scripts/verify_balanced_reading.py FILE [FILE ...]")

if __name__ == "__main__":
    main()
