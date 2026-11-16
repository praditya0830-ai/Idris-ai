"""RunRecord: one envelope for every result (Fase 4).

A number without provenance is a rumour. Every answer the engine produces is wrapped in a record that
names its inputs, the kernel and solver versions, the geometry hash, the evidence level, the sources
and the checks that passed. `run_id` is derived from those facts, so the same run computed twice gets
the same id and a different run cannot pretend to be it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json

from geometry import fingerprint as FP

STATUSES = ("ok", "repaired", "quarantined", "blocked", "failed")
KERNEL_DEFAULT = {"name": "occt", "version": None}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inputs_hash(inputs) -> str:
    """Stable hash of the request/parameters (dict order and float formatting do not matter)."""
    return _sha(canonical_json(inputs))


def make(*, inputs, solver_version: str, geometry_summary: dict | None = None,
         kernel: dict | None = None, evidence: dict | None = None, sources=(),
         checks=(), status: str = "ok", kind: str = "part", created_at: str | None = None,
         extra: dict | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")
    ih = inputs_hash(inputs)
    gh = FP.fingerprint(geometry_summary) if geometry_summary else None
    record = {
        "run_id": _sha(f"{kind}|{ih}|{gh or 'no-geometry'}"),
        "kind": kind,
        "created_at": created_at or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs_hash": ih,
        "geometry_hash": gh,
        "kernel": dict(kernel or KERNEL_DEFAULT),
        "solver_version": str(solver_version),
        "evidence": evidence,
        "sources": [dict(s) for s in sources],
        "checks": [dict(c) for c in checks],
        "status": status,
    }
    if extra:
        record["extra"] = dict(extra)
    problems = validate(record)
    if problems:
        raise ValueError("; ".join(problems))
    return record


def validate(record: dict) -> list:
    """-> list of problems (empty when the record is usable)."""
    if not isinstance(record, dict):
        return ["record must be an object"]
    out = []
    for key in ("run_id", "kind", "created_at", "inputs_hash", "solver_version", "status", "checks"):
        if key not in record:
            out.append(f"missing {key}")
    if record.get("status") not in STATUSES:
        out.append(f"status {record.get('status')!r} not in {STATUSES}")
    if not isinstance(record.get("solver_version"), str) or not record.get("solver_version"):
        out.append("solver_version must be a non-empty string")
    for src in record.get("sources") or []:
        if not isinstance(src, dict) or "url" not in src:
            out.append(f"source {src!r} needs a url")
    for c in record.get("checks") or []:
        if not isinstance(c, dict) or "id" not in c or "ok" not in c:
            out.append(f"check {c!r} needs id and ok")
    return out


def body_hash(record: dict) -> str:
    """Hash of everything except the id, used to detect a store collision with different content."""
    return _sha(canonical_json({k: v for k, v in record.items() if k != "run_id"}))
