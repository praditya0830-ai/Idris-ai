"""Sidecar manifest: a STEP file that is not a dumb solid when it leaves the system (Fase 3).

`part.step` carries the shape; `part.idris.json` carries the intent: dimensions, features, evidence
level and sources. A reader who only has the geometry can stay ignorant; a reader with the sidecar can
regenerate, audit and argue with the numbers.
"""
from __future__ import annotations

import json
import os

SIDECAR_SUFFIX = ".idris.json"
FORMAT = "idris.sidecar/1"


def path_for(step_path: str) -> str:
    base = str(step_path)
    for ext in (".step", ".stp", ".iges", ".igs", ".brep"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base + SIDECAR_SUFFIX


def build(part: str, *, features=(), dims=None, run=None, evidence=None, sources=(),
          notes: str = "") -> dict:
    """-> the manifest body. `run` is an audit RunRecord (optional but recommended)."""
    return {
        "format": FORMAT,
        "part": part,
        "units": "mm",
        "features": [dict(f) for f in features],
        "dims": dict(dims or {}),
        "run_id": (run or {}).get("run_id"),
        "geometry_hash": (run or {}).get("geometry_hash"),
        "solver_version": (run or {}).get("solver_version"),
        "evidence": evidence if evidence is not None else (run or {}).get("evidence"),
        "sources": list((run or {}).get("sources") or sources),
        "notes": notes,
    }


def has_intent(manifest: dict) -> bool:
    """True only when the sidecar actually carries intent: features or dimensions plus provenance."""
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        return False
    return bool(manifest.get("features") or manifest.get("dims")) and bool(
        manifest.get("run_id") or manifest.get("evidence"))


def write(step_path: str, manifest: dict) -> str:
    out = path_for(step_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def read(step_path: str) -> dict:
    src = path_for(step_path)
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{src}: not an IDRIS sidecar (format {FORMAT!r} expected)")
    return data
