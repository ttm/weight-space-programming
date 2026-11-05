"""Structural provenance validation for development artifacts and trial inputs.

These fields are claims and an audit index, not cryptographic attestations of
who generated the weights. Session/runner provenance is more authoritative than
metadata written by the participant in a submission.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

FIELDS={"creator_type", "model", "reasoning_effort", "access_route", "generated_at",
        "tools_used", "prompt_reference", "evidence"}
EXCEPTION_MINIMAL="fixtures/xor2_weights_only.json"


def validate(provenance: dict) -> None:
    if not isinstance(provenance,dict) or set(provenance)!=FIELDS:
        raise ValueError("Provenance fields missing or unexpectedly present")
    if provenance["creator_type"] not in ("llm", "human", "algorithm", "unknown"):
        raise ValueError("Unknown creator type")
    for field in FIELDS-{ "creator_type", "tools_used" }:
        val=provenance[field]
        if val is not None and (not isinstance(val,str) or not val.strip()):
            raise ValueError(f"Invalid {field}")
    tools=provenance["tools_used"]
    if tools is not None and (not isinstance(tools,list) or any(not isinstance(v,str) or not v.strip() for v in tools)):
        raise ValueError("tools_used must be null or a list of nonempty strings")


def index(root: Path) -> dict:
    """Generate audit hashes for every committed JSON artifact in core folders."""
    entries=[]
    for folder in ("examples", "fixtures", "repairs"):
        for path in sorted((root/folder).glob("*.json")):
            relative=path.relative_to(root).as_posix()
            raw=path.read_bytes()
            artifact=json.loads(raw)
            provenance=artifact.get("provenance")
            if relative==EXCEPTION_MINIMAL and provenance is None:
                # This input intentionally tests a JSON artifact containing only weights.
                status="exception_minimal_weight_only"
            else:
                validate(provenance)
                status="reported_metadata_not_attestation"
            entries.append({"path":relative,"task":artifact.get("task"),
                            "sha256":sha256(raw).hexdigest(),"status":status,
                            "provenance":provenance})
    if not entries:
        raise ValueError("No artifacts found")
    return {"schema_version":1,"status":"AUDIT_METADATA_NOT_AUTHOR_AUTHENTICATION",
            "exception":EXCEPTION_MINIMAL,"artifacts":entries,
            "total":len(entries),"structured":sum(v["provenance"] is not None for v in entries)}


OWNER_REPORTED={
    "creator_type":"llm",
    "model":"GPT-6",
    "reasoning_effort":"high",
    "access_route":"ChatGPT Plus chat interface",
    "generated_at":None,
    "tools_used":None,
    "prompt_reference":None,
    "evidence":"repository owner report recorded 2026-11-05"
}
UNKNOWN_PATCH={
    "creator_type":"unknown",
    "model":None,
    "reasoning_effort":None,
    "access_route":None,
    "generated_at":None,
    "tools_used":None,
    "prompt_reference":None,
    "evidence":"repair patch origin not independently confirmed as of 2026-11-05"
}
