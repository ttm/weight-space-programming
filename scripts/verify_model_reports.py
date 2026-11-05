#!/usr/bin/env python3
"""Replay recorded model-report summaries without downloading/checkpoint inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "experiments" / "reports"


def verify() -> dict:
    synthesis = json.loads((BASE / "2026-11-05-synthesis.json").read_text())
    reading = json.loads((BASE / "2026-11-05-reading.json").read_text())
    assert synthesis["model"] == reading["model"] == "Qwen/Qwen3-0.6B"
    assert synthesis["model_revision"] == reading["model_revision"]
    assert synthesis["feedback_probes"] == reading["feedback_probes"] == 0
    assert synthesis["total"] == len(synthesis["results"]) == 4
    assert reading["total"] == len(reading["records"]) == 6
    assert reading["valid"] == 6 and reading["exact"] == 0

    for trial in synthesis["results"]:
        sub = trial["submitted"]
        assert hashlib.sha256(sub["raw"].encode()).hexdigest() == sub["sha256"]
        assert not trial["passed"] and sub["result"] is None
        assert sub.get("posthoc_extracted_json") is None
        assert sub["error"]
        assert hashlib.sha256(trial["prompt"].encode()).hexdigest() == trial["prompt_sha256"]
    assert sum(bool(t["passed"]) for t in synthesis["results"]) == synthesis["successful"] == 0

    grouped = {}
    scores = []
    for trial in reading["records"]:
        result = trial["result"]
        assert hashlib.sha256(result["raw"].encode()).hexdigest() == result["sha256"]
        assert hashlib.sha256(trial["prompt"].encode()).hexdigest() == trial["prompt_sha256"]
        assert result["valid"] and not result["exact"]
        assert result["predicted"] == "0" * 16
        scores.append(result["bit_accuracy"])
        grouped.setdefault(trial["id"], {})[trial["variant"]] = result["predicted"]
    assert len(grouped) == len(reading["pairs"]) == 3
    agreements = sum(g["original"] == g["gauge_transformed"] for g in grouped.values())
    assert agreements == 3
    assert all(p["same_predicted_function"] for p in reading["pairs"])
    assert abs(sum(scores) / len(scores) - 2 / 3) < 1e-12
    return {"model": synthesis["model"], "synthesis_valid": 0, "synthesis_total": 4,
            "reading_exact": 0, "reading_total": 6, "reading_mean_bit_accuracy": sum(scores)/len(scores),
            "reading_consistent_pairs": agreements, "reading_pair_total": 3,
            "note": "Integrity/replay validation only, not renewed inference"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
