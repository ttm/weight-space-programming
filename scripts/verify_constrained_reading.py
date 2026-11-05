#!/usr/bin/env python3
"""Replay model or control constrained-reading outcomes from logged raw bits."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys
from wsp.balanced_reading import make_bank,evaluate_predictions
from run_constrained_reading import prompt_for,brier_score


def verify(report: dict) -> dict:
    if report.get("status") != "EXPLORATORY_POSTHOC_CONSTRAINED_DECODING_AFTER_INVALID_JSON":
        raise ValueError("Unexpected report status")
    bank,key=make_bank(seed=report["seed"],per_family=report["per_family"])
    if report["public_bank"] != bank or report["answer_key_unsealed_after_run"] != key:
        raise ValueError("Altered task bank/answer key")
    if len(report["records"]) != len(bank["items"]):
        raise ValueError("Wrong number of records")
    by_id={item["id"]:item for item in bank["items"]}
    guesses={}; ps={}
    for record in report["records"]:
        ident=record["id"]
        if ident not in by_id or ident in guesses:
            raise ValueError("Unknown or duplicate ID")
        prompt=prompt_for(by_id[ident])
        if record["prompt"]!=prompt or sha256(prompt.encode()).hexdigest()!=record["prompt_sha256"]:
            raise ValueError("Prompt mismatch")
        raw=record["raw"]
        if not isinstance(raw,str) or len(raw)!=16 or set(raw)-{"0","1"}:
            raise ValueError("Invalid output bits")
        if sha256(raw.encode()).hexdigest()!=record["raw_sha256"]:
            raise ValueError("Raw response hash mismatch")
        guesses[ident]=raw
        ps[ident]=record["probability_of_one_conditional_binary_tokens"]
    summary=evaluate_predictions(key,guesses)
    summary["conditional_binary_token_brier"]=round(brier_score(key,ps),8)
    if report["summary"]!=summary:
        raise ValueError("Score mismatch")
    return {k:summary[k] for k in ("exact","valid","pairs_agree","pairs_both_exact","mean_bit_accuracy",
                                  "constant_predictions","conditional_binary_token_brier")}


def main():
    if len(sys.argv)<2:raise SystemExit("Usage: verify_constrained_reading.py report.json ...")
    for name in sys.argv[1:]:
        print(json.dumps({"file":name,"verified":verify(json.loads(Path(name).read_text()))},sort_keys=True))

if __name__=="__main__":main()
