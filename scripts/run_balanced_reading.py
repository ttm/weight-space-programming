#!/usr/bin/env python3
"""One-shot, zero-feedback balanced weight-reading experiment.

No remote model download or inference occurs in --dry-run. Model inference is
opt-in and only scheduled by the explicitly triggered GitHub Actions workflow.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from wsp.balanced_reading import make_bank, evaluate_predictions

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CONDITIONS = ("direct", "deliberate")


def prompt_for(item: dict, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError("Invalid prompt condition")
    l1, l2 = item["network"]["layers"]
    instruction = (
        "Compute all 16 results directly from the arrays. Do not describe your reasoning."
        if condition == "direct" else
        "Silently derive each hidden neuron activation for every input, combine their contributions, "
        "check threshold ties carefully, then output your final answer only."
    )
    return (
        "Read the weights of a two-layer neural network; do NOT write new weights.\n"
        "x=(x0,x1,x2,x3) comprises exactly four binary bits. For hidden j, "
        "h_j=max(0, b1[j] + sum(W1[j][k]*x[k] for k=0..3)). "
        "Final z=b2[0] + sum(W2[0][j]*h_j for j=0..3). "
        "The predicted bit is 1 iff z>0, otherwise 0.\n"
        "Predict outputs in strict lexicographic input order: "
        "0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111.\n"
        "Return ONLY one JSON object with ONE key named truth_table whose value is a "
        "string of exactly 16 characters, each 0 or 1. Do not use markdown or additional keys.\n"
        + instruction + "\n"
        "W1=" + json.dumps(l1["weights"], separators=(",", ":")) + "\n"
        "b1=" + json.dumps(l1["bias"], separators=(",", ":")) + "\n"
        "W2=" + json.dumps(l2["weights"], separators=(",", ":")) + "\n"
        "b2=" + json.dumps(l2["bias"], separators=(",", ":")) + "\n"
    )


def decode(raw: str) -> tuple[str | None, str | None]:
    try:
        data = json.loads(raw.strip())
        if not isinstance(data, dict) or set(data) != {"truth_table"}:
            raise ValueError("Expected exactly one key truth_table")
        table = data["truth_table"]
        if not isinstance(table, str) or len(table) != 16 or set(table) - {"0", "1"}:
            raise ValueError("truth_table must be a 16-character binary string")
        return table, None
    except (ValueError, TypeError) as exc:
        return None, type(exc).__name__ + ": " + str(exc)


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=False,
    ).eval()
    return torch, tokenizer, model


def run(*, seed: int = 20261105, per_family: int = 1, condition: str = "direct",
        participant: str = "model", max_new_tokens: int = 128) -> dict:
    if condition not in CONDITIONS or participant not in ("model", "oracle", "zero"):
        raise ValueError("Unknown condition or participant")
    if max_new_tokens < 32 or max_new_tokens > 1024:
        raise ValueError("max_new_tokens out of range")
    bank, key = make_bank(seed=seed, per_family=per_family)
    if participant == "model":
        torch, tokenizer, model = load_model()
    else:
        torch, tokenizer, model = None, None, None
    records, guesses = [], {}
    for item in bank["items"]:
        prompt = prompt_for(item, condition)
        start = time.monotonic()
        output_tokens = 0
        if participant == "oracle":
            raw = json.dumps({"truth_table": key["answers"][item["id"]]["truth_table"]})
        elif participant == "zero":
            raw = json.dumps({"truth_table": "0" * 16})
        else:
            tokens = tokenizer.apply_chat_template(
                [{"role":"user", "content":prompt}], add_generation_prompt=True,
                return_tensors="pt", tokenize=True,
            )
            with torch.inference_mode():
                output = model.generate(tokens, do_sample=False, max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.eos_token_id)
            generated = output[0][tokens.shape[-1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            output_tokens = len(generated)
        duration = round(time.monotonic() - start, 4)
        prediction, error = decode(raw)
        guesses[item["id"]] = prediction
        records.append({"id":item["id"], "prompt":prompt,
                        "prompt_sha256":sha256(prompt.encode()).hexdigest(),
                        "raw":raw, "raw_sha256":sha256(raw.encode()).hexdigest(),
                        "parsed":prediction, "error":error,
                        "output_tokens":output_tokens, "elapsed_seconds":duration})
        print(json.dumps({"id":item["id"], "valid":prediction is not None,
                          "error":error, "output_tokens":output_tokens, "seconds":duration}),flush=True)
    summary = evaluate_predictions(key, guesses)
    return {"schema_version":1, "status":"EXPLORATORY_NOT_SEALED_NOT_PREREGISTERED",
            "participant":participant,
            "model":MODEL_ID if participant == "model" else None,
            "model_revision":MODEL_REVISION if participant == "model" else None,
            "sampling":"greedy" if participant == "model" else "reference_control",
            "condition":condition, "max_new_tokens":max_new_tokens,
            "tools_available_to_model":[], "feedback_probes":0,
            "task_generator":"wsp.balanced_reading.v1", "seed":seed,
            "per_family":per_family, "public_bank":bank,
            "answer_key_unsealed_after_run":key,
            "records":records, "summary":summary,
            "limitations":["Public deterministic generator and source code; no sealed task families",
                           "Small non-randomized pilot; no confidence intervals or population-level claims",
                           "Paired networks share the same true function and are not independent samples",
                           "The oracle control is code execution, not language-model comprehension"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--per-family", type=int, default=1)
    p.add_argument("--condition", choices=CONDITIONS, default="direct")
    p.add_argument("--participant", choices=("model", "oracle", "zero"), default="model")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = run(seed=args.seed, per_family=args.per_family, condition=args.condition,
                 participant=args.participant, max_new_tokens=args.max_new_tokens)
    dest = Path(args.output);dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(dest),"summary":{k:report['summary'][k] for k in
                       ('n_networks','n_functions','valid','exact','pairs_agree','pairs_both_exact','mean_bit_accuracy')}}))


if __name__ == "__main__":
    main()
