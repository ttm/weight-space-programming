"""Exploratory participant runner. Stubs are development controls, NOT LLM trials.

The external model adapter is opt-in and only runs with OPENAI_API_KEY.
No model calls occur during CI, imports, or the default stub pilot.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import urllib.request

from .architecture import check_network, architecture_hash
from .baselines import sample_network
from .generated import FAMILIES, GeneratedTask, evaluate_generated, make_task


def _decode_candidate(raw: str) -> dict:
    """Strict JSON response. Invalid output counts as a failed attempt."""
    item=json.loads(raw)
    if not isinstance(item,dict): raise ValueError("Response must be a JSON object")
    return item


def _response_text(data: dict) -> str:
    # OpenAI Responses API output has an array of message items and content blocks.
    return "".join(c.get("text", "") for item in data.get("output", [])
                   if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")


def call_openai(messages: list[dict], model: str, *, max_output_tokens: int = 3000) -> tuple[str, dict]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for paid model calls")
    data = json.dumps({"model": model, "input": messages,
                       "max_output_tokens":max_output_tokens,"store":False}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/responses",data=data,
                                 headers={"Authorization":f"Bearer {key}",
                                          "Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=120) as response:
        result=json.load(response)
    raw=_response_text(result)
    if not raw:
        raise RuntimeError(f"Model response had no output_text; status={result.get('status')}")
    return raw, {"response_id":result.get("id"),"model":result.get("model"),
                 "usage":result.get("usage"),"status":result.get("status")}


def _prompt(task: GeneratedTask, explanation: str) -> str:
    requirement = ("Include a nonempty 'explanation' string." if explanation=="required"
                   else "Do not include any 'explanation' field.")
    return ("Produce a JSON object only, with keys schema_version=1, task, network. "
            "Use numerical weights, biases and activations with the EXACT architecture. "
            "No code, training, tools, optimizers, or external feedback unless provided. "
            "The network is the only executable artifact. " + requirement + "\nTASK:\n" +
            json.dumps(task.public(),sort_keys=True))


def _stub(task: GeneratedTask, participant: str, explanation: str, rng: random.Random) -> str:
    network = task.reference_network if participant == "oracle_stub" else sample_network(task.architecture,rng)
    candidate={"schema_version":1,"task":task.task_id,"network":network}
    if explanation=="required":
        candidate["explanation"]=("Symbolic witness encoded in neurons" if participant=="oracle_stub"
                                  else "Randomly sampled weights without performance guarantees")
    return json.dumps(candidate,sort_keys=True)


def run_trial(task: GeneratedTask, *, participant: str = "random_stub", model: str | None = None,
              mode: str = "blind", explanation: str = "forbidden", budget: int = 0,
              attempt: int = 0, salt: str = "", max_output_tokens: int = 3000) -> dict:
    if participant not in ("oracle_stub","random_stub","openai"):
        raise ValueError("Unsupported participant")
    if mode not in ("blind","budgeted") or explanation not in ("required","forbidden"):
        raise ValueError("Unsupported condition")
    if mode == "blind": budget=0
    elif not 1 <= budget <= 10: raise ValueError("Budgeted trial requires 1..10 validation probes")
    if participant=="openai" and (not model or not os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("OpenAI participant requires explicit --model and OPENAI_API_KEY")
    rng = random.Random(int.from_bytes(sha256(f"pilot-v1:{task.task_id}:{participant}:{mode}:{explanation}:{attempt}".encode()).digest()[:8],"big"))
    prompt=_prompt(task,explanation)
    history=[{"role":"user","content":prompt}]
    records=[]
    for call in range(budget+1):
        metadata={}
        if participant=="openai":
            try: raw,metadata=call_openai(history,model,max_output_tokens=max_output_tokens)
            except Exception as exc:
                # Provider error consumes the current attempt and is reported, not silently retried.
                raw=""; metadata={"provider_error":type(exc).__name__+": "+str(exc)[:160]}
        else:
            raw=_stub(task,participant,explanation,rng)
        item={"candidate_sha256":sha256(raw.encode()).hexdigest(),"raw":raw,"api":metadata,"error":None,"score":None}
        try:
            cand=_decode_candidate(raw)
            if cand.get("schema_version")!=1: raise ValueError("Wrong candidate schema")
            if explanation=="required" and not (isinstance(cand.get("explanation"),str) and cand["explanation"].strip()):
                raise ValueError("Explanation required")
            if explanation=="forbidden" and "explanation" in cand: raise ValueError("Explanation forbidden")
            check_network(cand.get("network"),task.architecture)
            item["score"]=evaluate_generated(task,cand,("validation",) if call<budget else ("test","ood","edge"),salt=salt)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            item["error"]=type(exc).__name__+": "+str(exc)
        records.append(item)
        if call < budget:
            # Return feedback only; no test labels, oracle or reference tensors.
            feedback = {"validation":item["score"],"error":item["error"],"remaining_probes":budget-call-1}
            history += [{"role":"assistant","content":raw},{"role":"user","content":json.dumps(feedback,sort_keys=True)+"\nGive the next candidate in exact JSON."}]
    final=records[-1]
    return {"schema_version":1,"status":"exploratory_unregistered",
            "participant":participant,"model":model,"condition":{"mode":mode,"explanation":explanation,"budget":budget},
            "task_family":task.family,"task_id":task.task_id,"task_seed":task.seed,
            "architecture_sha256":architecture_hash(task.architecture),
            "prompt_sha256":sha256(prompt.encode()).hexdigest(),
            "prompt":prompt,"attempt":attempt,"probes":records[:-1],"final":final,
            "passed":bool(final["score"] and final["score"]["all_passed"])}


def run_pilot(*, participant: str = "random_stub", model: str | None = None,
              seeds: int = 2, repeats: int = 1, salt: str = "",
              families: tuple[str,...] = FAMILIES) -> dict:
    if not 1 <= seeds <= 20 or not 1 <= repeats <= 10 or not families or any(f not in FAMILIES for f in families):
        raise ValueError("Invalid seed/repeat/family setting")
    trials=[]
    for family in families:
        for seed in range(seeds):
            task=make_task(family,seed+200)
            for mode, budget in (("blind",0),("budgeted",3)):
                for explanation in ("forbidden","required"):
                    for attempt in range(repeats):
                        trials.append(run_trial(task,participant=participant,model=model,mode=mode,
                                                explanation=explanation,budget=budget,attempt=attempt,salt=salt))
    summary=[]
    for mode in ("blind","budgeted"):
        for ex in ("forbidden","required"):
            block=[t for t in trials if t["condition"]["mode"]==mode and t["condition"]["explanation"]==ex]
            summary.append({"mode":mode,"explanation":ex,"n":len(block),"passed":sum(t["passed"] for t in block),
                            "invalid_final":sum(t["final"]["error"] is not None for t in block)})
    return {"schema_version":1,"status":"exploratory_unregistered",
            "warning":"Stubs are NOT independent LLM results; OpenAI participant requires an API key and separate compute budget.",
            "participant":participant,"model":model,"summary":summary,"trials":trials}


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description="Exploratory task runner; stubs run free, OpenAI requires API key")
    parser.add_argument("--participant",choices=("random_stub","oracle_stub","openai"),default="random_stub")
    parser.add_argument("--model",help="Required with --participant openai")
    parser.add_argument("--families",type=str,default=",".join(FAMILIES))
    parser.add_argument("--seeds",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--output")
    args=parser.parse_args()
    report=run_pilot(participant=args.participant,model=args.model,seeds=args.seeds,repeats=args.repeats,
                     salt=os.getenv("WSP_SEED_SALT", ""),
                     families=tuple(f.strip() for f in args.families.split(",")))
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"participant":args.participant,"summary":report["summary"]},indent=2))

if __name__=="__main__":main()
