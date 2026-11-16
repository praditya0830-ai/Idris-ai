"""Canonical geometry fingerprint (Fase 0 of the enterprise architecture).

Same inputs must give the same hash on every machine, so a claim like "this part regenerates
reliably" can be checked instead of trusted. The summary is a small dictionary of physical facts,
rounded to a fixed precision and serialised with sorted keys: no dict order, no locale, no kernel.

Two kernels that agree physically therefore agree on the fingerprint; two that disagree do not.

    summary = {"volume_mm3": 4800.0, "area_mm2": 1760.0, "bbox_mm": [40, 20, 3],
               "solids": 1, "faces": 6, "edges": 12, "vertices": 8, "mass_kg": 0.0059}
    fingerprint(summary)  -> "3f2a..."
"""
from __future__ import annotations

import hashlib
import json

PRECISION = 6
# Counts are exact; sizes are compared with a relative tolerance.
FIELDS = ("volume_mm3", "area_mm2", "bbox_mm", "solids", "faces", "edges", "vertices",
          "triangles", "mass_kg")
COUNTS = ("solids", "faces", "edges", "vertices", "triangles")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or abs(v) > 1e12:                 # NaN / absurd: not a physical fact
        return None
    return round(v, PRECISION)


def canonical(summary: dict) -> dict:
    """The part of a summary that the hash covers, normalised. Unknown keys are ignored on purpose:
    a kernel that reports an extra field must not change the fingerprint of the same solid."""
    out = {}
    for k in sorted(FIELDS):
        if k not in summary:
            continue
        v = summary[k]
        if k == "bbox_mm":
            out[k] = [_num(c) for c in (v or [])][:6]
        elif isinstance(v, (list, tuple)):
            out[k] = [_num(c) for c in v]
        else:
            out[k] = _num(v)
    return out


def fingerprint(summary: dict) -> str:
    """Stable sha256 of the canonical summary (empty summary still hashes, so it is never silent)."""
    body = json.dumps(canonical(summary), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def diff(before: dict, after: dict, tol: float = 1e-3) -> list:
    """Physical differences, ignoring changes inside `tol` (relative). -> [{"field", "before", "after"}].

    Used by the STEP round-trip gate: an export/import that moves the volume by more than 0.1 % is a
    failure, not a warning."""
    a, b = canonical(before), canonical(after)
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if va == vb:
            continue
        if (k not in COUNTS and isinstance(va, (int, float)) and isinstance(vb, (int, float))):
            base = max(abs(va), abs(vb), 1e-9)
            if abs(va - vb) / base <= tol:
                continue
        out.append({"field": k, "before": va, "after": vb})
    return out


def from_solid(solid) -> dict:
    """Summary of a kernel solid. Raises geometry.kernel.KernelUnavailable when no kernel is installed."""
    from . import kernel
    return kernel.measure(solid)
