#!/usr/bin/env python3
"""Post-hoc exploratory follow-up: force 16 binary output decisions.

Study purpose: remove JSON/Markdown copying failures as a confound while retaining
zero-feedback numeric weight interpretation. Reported probability is conditional
on the *restricted* output alphabet, not full-token likelihood.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from wsp.balanced_reading import make_bank, evaluate_predictions
from run_balanced_reading import load_model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def prompt_for(item: dict) -> str:
    l1, l2 = item["network"]["layers"]
    return (
        "You will predict the Boolean output of a neural network for 16 inputs. "
        "You are NOT given desired outputs. Use only the weights below.\n"
        "Inputs x0,x1,x2,x3 are bits. For each j=0..3, "
        "h_j=max(0,b1[j]+sum(W1[j][k]*x[k] for k=0..3)). "
        "Output z=b2[0]+sum(W2[0][j]*h_j for j=0..3). "
        "Return bit 1 iff z is strictly positive; otherwise return 0.\n"
        "Consider all x in integer order 0 to 15, with x0 the MOST significant "
        "bit and x3 the least significant bit. "
        "Give exactly the resulting 16 output bits in this order, without labels, examples, "
        "input patterns, commentary, markdown or punctuation.\n"
        "W1=" + json.dumps(l1["weights"], separators=(",", ":")) + "\n"
        "b1=" + json.dumps(l1["bias"], separators=(",", ":")) + "\n"
        "W2=" + json.dumps(l2["weights"], separators=(",", ":")) + "\n"
        "b2=" + json.dumps(l2["bias"], separators=(",", ":"))
    )


def brier_score(key: dict, per_id_p1: dict[str, list[float]]) -> float:
    """P(bit=1 | token constrained to {0,1}), averaged over network views."""
    errors = []
    for ident, probabilities in per_id_p1.items():
        truth = key["answers"][ident]["truth_table"]
        if len(probabilities) != 16 or any(not 0 <= x <= 1 for x in probabilities):
            raise ValueError("Invalid probability vector")
        errors.extend((p - int(t))**2 for p,t in zip(probabilities,truth))
    return sum(errors)/len(errors)


def run(*, seed: int = 20261105, per_family: int = 1, participant: str = "model") -> dict:
    if participant not in ("model","oracle","zero"):
        raise ValueError("Invalid participant")
    bank,key = make_bank(seed=seed,per_family=per_family)
    if participant == "model":
        torch,tokenizer,model=load_model()
        t0=tokenizer.encode("0",add_special_tokens=False)
        t1=tokenizer.encode("1",add_special_tokens=False)
        if len(t0)!=1 or len(t1)!=1 or t0[0]==t1[0]:
            raise RuntimeError("Tokenizer requires single distinct tokens for '0' and '1'")
        id0,id1=t0[0],t1[0]
    else:
        torch=tokenizer=model=None
        id0=id1=None
    records=[]; guesses={}; per_id_p1={}
    for item in bank["items"]:
        prompt=prompt_for(item)
        start=time.monotonic()
        if participant == "oracle":
            raw = key["answers"][item["id"]]["truth_table"]
            conditional_p1=[float(v) for v in raw]
        elif participant == "zero":
            raw = "0"*16
            conditional_p1=[0.]*16
        else:
            ids=tokenizer.apply_chat_template([{"role":"user","content":prompt}],
                                              add_generation_prompt=True,return_tensors="pt",tokenize=True)
            with torch.inference_mode():
                generated=model.generate(
                    ids, do_sample=False, max_new_tokens=16,
                    prefix_allowed_tokens_fn=lambda batch_id, input_ids: [id0,id1],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=None,
                    return_dict_in_generate=True, output_scores=True,
                )
            suffix=generated.sequences[0][ids.shape[-1]:].tolist()
            if len(suffix)!=16 or any(token not in (id0,id1) for token in suffix):
                raise RuntimeError("Constrained decoding emitted invalid output tokens")
            raw="".join("0" if token==id0 else "1" for token in suffix)
            conditional_p1=[]
            for logits in generated.scores:
                # These scores are already restricted to {0,1} by generation.
                probs=torch.softmax(torch.stack([logits[0,id0].float(),logits[0,id1].float()]),0)
                conditional_p1.append(round(probs[1].item(),8))
        assert len(raw)==16 and set(raw)<={"0","1"}
        guesses[item["id"]]=raw
        per_id_p1[item["id"]]=conditional_p1
        seconds=round(time.monotonic()-start,4)
        records.append({"id":item["id"],"prompt":prompt,"prompt_sha256":sha256(prompt.encode()).hexdigest(),
                        "raw":raw,"raw_sha256":sha256(raw.encode()).hexdigest(),
                        "probability_of_one_conditional_binary_tokens":conditional_p1,
                        "output_tokens":len(raw) if participant=="model" else 0,
                        "elapsed_seconds":seconds})
        print(json.dumps({"id":item["id"],"generated_bits":raw,"seconds":seconds}),flush=True)
    summary=evaluate_predictions(key,guesses)
    summary["conditional_binary_token_brier"] = round(brier_score(key,per_id_p1),8)
    return {"schema_version":1,"status":"EXPLORATORY_POSTHOC_CONSTRAINED_DECODING_AFTER_INVALID_JSON",
            "participant":participant,"model":MODEL_ID if participant=="model" else None,
            "model_revision":MODEL_REVISION if participant=="model" else None,
            "sampling":"greedy_restricted_to_tokens_for_0_and_1" if participant=="model" else "control",
            "model_binary_token_ids":[id0,id1] if participant=="model" else None,
            "max_new_tokens":16,"feedback_probes":0,"tools_available_to_model":[],
            "seed":seed,"per_family":per_family,"public_bank":bank,"answer_key_unsealed_after_run":key,
            "records":records,"summary":summary,
            "limitations":["Exploratory follow-up designed after observing invalid JSON responses in the prior experiment",
                           "Restricting the output vocabulary changes the inference problem and does not establish free-form WSP",
                           "Conditional binary-token probabilities differ from unrestricted model probabilities",
                           "Four underlying functions, paired dependent views, public task family, no preregistration"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20261105)
    p.add_argument("--per-family",type=int,default=1)
    p.add_argument("--participant",choices=("model","oracle","zero"),default="model")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=run(seed=args.seed,per_family=args.per_family,participant=args.participant)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"report":str(path),"exact":report["summary"]["exact"],
                      "both_exact":report["summary"]["pairs_both_exact"],
                      "bit_accuracy":report["summary"]["mean_bit_accuracy"],
                      "conditional_brier":report["summary"]["conditional_binary_token_brier"]}))

if __name__=="__main__":main()
