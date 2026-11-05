"""Feedback-budget bookkeeping. Honest local benchmark, not a sealed evaluation server."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from . import __version__
from .evaluation import evaluate
from .architecture import check_network, validate_architecture, architecture_hash
from .network import parse_candidate, validate_network
from .tasks import TASKS, seed_salt_sha256

MODES = {"blind", "budgeted", "open"}
EXPLANATIONS = {"required", "optional", "forbidden"}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start(path: Path, task: str, mode: str, explanation: str, budget: int = 3, *,
          participant: str | None = None, model: str | None = None, prompt_file: Path | None = None, architecture: dict | None = None, reasoning_effort: str | None = None,
          access_route: str | None = None, tools_used: list[str] | None = None) -> dict:
    if path.exists():
        raise FileExistsError(f"Session exists: {path}")
    if task not in TASKS or mode not in MODES or explanation not in EXPLANATIONS or budget < 0:
        raise ValueError("Invalid task, mode, explanation or budget")
    if architecture is not None:
        validate_architecture(architecture)
        if (architecture["layers"][0]["inputs"] != TASKS[task].inputs or
                architecture["layers"][-1]["outputs"] != TASKS[task].outputs):
            raise ValueError("Architecture input/output dimensions mismatch task")
    if mode == "blind": budget = 0
    if mode == "open": budget = -1
    if tools_used is not None and (not isinstance(tools_used, list) or
            any(not isinstance(v, str) or not v.strip() for v in tools_used)):
        raise ValueError("tools_used must be a list of nonempty strings or None")
    provenance = {"wsp_version": __version__, "participant": participant, "model": model,
                  "prompt_file": None if prompt_file is None else str(prompt_file),
                  "prompt_sha256": None if prompt_file is None else sha256(Path(prompt_file).read_bytes()).hexdigest()}
    provenance.update({"reasoning_effort": reasoning_effort, "access_route": access_route,
                       "tools_used": tools_used})
    session = {"schema_version": 1, "task": task, "mode": mode, "explanation": explanation,
               "budget": budget, "probes": [], "finalized": False, "created_at": _now(),
               "provenance": provenance}
    session["architecture"] = architecture
    session["architecture_sha256"] = architecture_hash(architecture) if architecture is not None else None
    _write(path, session)
    return session


def _submit(path: Path, session: dict, splits: tuple[str, ...]) -> dict:
    """Hash and score one artifact; a rejected artifact gives a record with an error instead of a result.

    Rejections count as attempts (PROTOCOL.md §3). A missing file submits nothing, so its
    FileNotFoundError propagates and no attempt is recorded.
    """
    contents = path.read_bytes()
    record = {"sha256": sha256(contents).hexdigest(), "timestamp": _now(), "result": None, "error": None}
    try:
        candidate = parse_candidate(contents)  # score exactly the bytes that were hashed
        if candidate["task"] != session["task"]:
            raise ValueError("Candidate task does not match session")
        text = candidate.get("explanation")
        if session["explanation"] == "required" and (not isinstance(text, str) or not text.strip()):
            raise ValueError("This session requires an explanation")
        if session["explanation"] == "forbidden" and text is not None:
            raise ValueError("This session forbids explanations")
        task = TASKS[session["task"]]
        validate_network(candidate.get("network"), task.inputs, task.outputs)
        if session.get("architecture") is not None:
            check_network(candidate.get("network"), session["architecture"])
        record["result"] = evaluate(candidate, splits)
    except (ValueError, ArithmeticError) as exc:  # bad JSON, schema, shapes, or numeric overflow
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def probe(session_path: Path, candidate_path: Path) -> dict:
    session = _load(session_path)
    if session["finalized"]:
        raise ValueError("Session already finalized")
    if session["mode"] == "blind":
        raise ValueError("Blind mode prohibits all validation probes")
    if session["budget"] >= 0 and len(session["probes"]) >= session["budget"]:
        raise ValueError("Validation probe budget exhausted")
    record = _submit(candidate_path, session, ("validation",))
    session["probes"].append(record)
    _write(session_path, session)
    if record["error"]:
        raise ValueError(f"Rejected candidate; the probe still counts against the budget. {record['error']}")
    return record["result"]


def finalize(session_path: Path, candidate_path: Path) -> dict:
    session = _load(session_path)
    if session["finalized"]:
        raise ValueError("Session already finalized: final evaluation can only be recorded once")
    record = _submit(candidate_path, session, ("test", "ood", "edge"))
    record["seed_salt_sha256"] = seed_salt_sha256()  # which test and OOD inputs produced the score
    session["finalized"] = True
    session["final"] = record
    _write(session_path, session)
    if record["error"]:
        raise ValueError(f"Rejected candidate, recorded as the failed final submission. {record['error']}")
    return record["result"]
