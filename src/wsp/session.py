"""Feedback-budget bookkeeping. Honest local benchmark, not a sealed evaluation server."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .evaluation import evaluate
from .network import load_candidate
from .tasks import TASKS

MODES = {"blind", "budgeted", "open"}
EXPLANATIONS = {"required", "optional", "forbidden"}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start(path: Path, task: str, mode: str, explanation: str, budget: int = 3) -> dict:
    if path.exists():
        raise FileExistsError(f"Session exists: {path}")
    if task not in TASKS or mode not in MODES or explanation not in EXPLANATIONS or budget < 0:
        raise ValueError("Invalid task, mode, explanation or budget")
    if mode == "blind": budget = 0
    if mode == "open": budget = -1
    session = {"schema_version": 1, "task": task, "mode": mode, "explanation": explanation,
               "budget": budget, "probes": [], "finalized": False, "created_at": _now()}
    _write(path, session)
    return session


def _candidate(path: Path, session: dict) -> tuple[dict, str]:
    contents = path.read_bytes()
    candidate = load_candidate(path)
    if candidate["task"] != session["task"]:
        raise ValueError("Candidate task does not match session")
    text = candidate.get("explanation")
    if session["explanation"] == "required" and (not isinstance(text, str) or not text.strip()):
        raise ValueError("This session requires an explanation")
    if session["explanation"] == "forbidden" and text is not None:
        raise ValueError("This session forbids explanations")
    return candidate, sha256(contents).hexdigest()


def probe(session_path: Path, candidate_path: Path) -> dict:
    session = _load(session_path)
    if session["finalized"]:
        raise ValueError("Session already finalized")
    if session["mode"] == "blind":
        raise ValueError("Blind mode prohibits all validation probes")
    if session["budget"] >= 0 and len(session["probes"]) >= session["budget"]:
        raise ValueError("Validation probe budget exhausted")
    candidate, digest = _candidate(candidate_path, session)
    result = evaluate(candidate, ("validation",))
    session["probes"].append({"sha256": digest, "timestamp": _now(), "result": result})
    _write(session_path, session)
    return result


def finalize(session_path: Path, candidate_path: Path) -> dict:
    session = _load(session_path)
    if session["finalized"]:
        raise ValueError("Session already finalized: final evaluation can only be recorded once")
    candidate, digest = _candidate(candidate_path, session)
    result = evaluate(candidate)
    session["finalized"] = True
    session["final"] = {"sha256": digest, "timestamp": _now(), "result": result}
    _write(session_path, session)
    return result
