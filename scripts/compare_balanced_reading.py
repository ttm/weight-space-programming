#!/usr/bin/env python3
"""Compare two exploratory prompt arms by shared *independent functions*.

This is descriptive paired analysis; n=4 functions is not statistical evidence
of broad capability, and no p-values or post-hoc selection are offered.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

from verify_balanced_reading import verify


def compare(a: dict, b: dict) -> dict:
    va, vb = verify(a), verify(b)
    if a["public_bank"] != b["public_bank"] or a["answer_key_unsealed_after_run"] != b["answer_key_unsealed_after_run"]:
        raise ValueError("Conditions must use exactly the same bank and oracle")
    if a["participant"] != b["participant"] or a["model"] != b["model"] or a["model_revision"] != b["model_revision"]:
        raise ValueError("Cannot compare different participants/checkpoints as a paired prompt study")
    if a["condition"] == b["condition"]:
        raise ValueError("Conditions must be different")
    groups = defaultdict(dict)
    for ident,row in a["answer_key_unsealed_after_run"]["answers"].items():
        groups[row["group"]][row["variant"]] = ident
    # Reflect underlying functions, not the two correlated presentations.
    by_cond = {}
    for report in (a,b):
        exact_by_id = {row["id"]:bool(row["exact"]) for row in report["summary"]["rows"]}
        by_cond[report["condition"]] = {
            g:all(exact_by_id[ident] for ident in variants.values())
            for g,variants in groups.items()
        }
    first, second = a["condition"], b["condition"]
    pair_rows = [{"group":g,first:by_cond[first][g],second:by_cond[second][g]}
                 for g in sorted(groups)]
    return {"schema_version":1,"status":"EXPLORATORY_DESCRIPTIVE_PAIRED_COMPARISON",
            "model":a["model"],"model_revision":a["model_revision"],
            "n_independent_functions":len(groups),"n_network_presentations":len(a["public_bank"]["items"]),
            "conditions":{
                first:{"strict_exact":va["exact"],"valid":va["valid"],"both_exact_pairs":va["pairs_both_exact"],
                       "trivial_agreement":va["pairs_trivial_agreement"],"mean_bit_accuracy":va["mean_bit_accuracy"]},
                second:{"strict_exact":vb["exact"],"valid":vb["valid"],"both_exact_pairs":vb["pairs_both_exact"],
                        "trivial_agreement":vb["pairs_trivial_agreement"],"mean_bit_accuracy":vb["mean_bit_accuracy"]}},
            "discordant_functions":{
                "only_"+first:sum(r[first] and not r[second] for r in pair_rows),
                "only_"+second:sum(r[second] and not r[first] for r in pair_rows)},
            "pair_rows":pair_rows,
            "caveat":"Only four original functions, paired transformed views; not sealed or preregistered; no significance claim."}


def main():
    if len(sys.argv) not in (3,4):
        raise SystemExit("Usage: compare_balanced_reading.py DIRECT.json DELIBERATE.json [OUTPUT.json]")
    a,b = (json.loads(Path(x).read_text()) for x in sys.argv[1:3])
    result=compare(a,b)
    payload=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if len(sys.argv)==4:Path(sys.argv[3]).write_text(payload)
    print(payload)


if __name__ == "__main__":main()
