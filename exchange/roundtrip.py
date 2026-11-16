"""Round-trip gate: export -> import -> compare (Fase 3, pilar D).

An export that loses a face or 2 % of the volume is a failure, not a warning. The gate reuses
geometry.fingerprint.diff, so the tolerance policy is the same one used everywhere else.
"""
from __future__ import annotations

import os
import tempfile

from geometry import fingerprint as FP
from geometry import invariants as INV
from geometry import kernel

DEFAULT_TOL = 1e-3          # 0.1 % on sizes; counts must match exactly


class RoundTripFailed(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def compare(before: dict, after: dict, tol: float = DEFAULT_TOL) -> dict:
    diffs = FP.diff(before, after, tol)
    return {"ok": not diffs, "tol": tol, "diffs": diffs,
            "before_hash": FP.fingerprint(before), "after_hash": FP.fingerprint(after)}


def gate(before: dict, after: dict, tol: float = DEFAULT_TOL) -> dict:
    """-> {"status": "ok"|"failed", ...}; never raises, so a report can carry the failure."""
    cmp = compare(before, after, tol)
    return {"status": "ok" if cmp["ok"] else "failed", **cmp}


def check_shape(shape, tol: float = DEFAULT_TOL) -> dict:
    """Invariants of one shape plus the gate against its own summary (self-consistency)."""
    summary = kernel.measure(shape)
    checks = INV.solid_checks({**summary, "degenerate_faces": summary.get("degenerate_faces", 0)})
    return {"summary": summary, "checks": checks, "all_ok": INV.all_ok(checks),
            "geometry_hash": FP.fingerprint(summary)}


def step_roundtrip(shape, workdir: str | None = None, name: str = "part",
                   tol: float = DEFAULT_TOL) -> dict:
    """Write STEP, read it back, compare. Raises RoundTripFailed with the record attached."""
    workdir = workdir or tempfile.mkdtemp(prefix="idris-rt-")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"{name}.step")
    before = kernel.measure(shape)
    kernel.write_step(shape, path)
    again = kernel.read_step(path)
    after = kernel.measure(again)
    record = {"file": path, "before": before, "after": after,
              **gate(before, after, tol),
              "checks_in": INV.solid_checks({**before, "degenerate_faces": before.get("degenerate_faces", 0)}),
              "checks_out": INV.solid_checks({**after, "degenerate_faces": after.get("degenerate_faces", 0)})}
    if record["status"] != "ok":
        raise RoundTripFailed(
            f"round trip moved the model beyond tolerance {tol}: "
            f"{[d['field'] for d in record['diffs']]}", record)
    return record
