"""Regenerate compact, deterministic research-control results for CI.

No external APIs or LLMs. Not a scientific model-competence study.
"""
from __future__ import annotations
import json
from pathlib import Path

from wsp.baselines import compare
from wsp.generated import FAMILIES, make_task, evaluate_generated
from wsp.pilot import run_pilot
from wsp.weight_reading import generate_bank


def main():
    checked = 0
    for family in FAMILIES:
        for seed in range(32):
            t = make_task(family, seed)
            assert evaluate_generated(t,t.reference())["all_passed"],(family,seed)
            checked += 1

    baselines = compare(seeds=4,budgets=(1,3,10))
    oracle = run_pilot(participant="oracle_stub",seeds=1,repeats=1)
    random_control = run_pilot(participant="random_stub",seeds=1,repeats=1)
    bank,key = generate_bank(count=16,seed=0)
    all_preserved = all(row["same_function"] for row in key["answers"].values())
    all_canonical = all(row["canonical_fingerprint"]==row["untransformed_fingerprint"]
                        for row in key["answers"].values())
    assert all_preserved and all_canonical
    report = {"schema_version":1,"status":"DEVELOPMENT_CONTROLS_NOT_LLM_TRIALS",
              "families":list(FAMILIES), "constructive_witnesses_verified":checked,
              "baselines":baselines["summary"],
              "stub_oracle_summary":oracle["summary"],"stub_random_summary":random_control["summary"],
              "matched_topology_reading_bank_size":len(bank["challenges"]),
              "reading_bank_gauge_preserves_behavior":all_preserved,
              "reading_bank_canonical_fingerprint_preserved":all_canonical,
              "notes":["Not blinded model generations", "No payment or provider calls", "Only integers and booleans are committed for cross-platform reproducibility"]}
    output=Path("results/development_research.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=="__main__":main()
