"""STEP/IGES in, plus feature CANDIDATES (Fase 3, pilar D).

Import is shape-level and honest: the kernel gives one shape, the summary says whether it is a valid
solid, and recognition only ever produces candidates with a confidence and a basis. A shape whose
features are not recognised is returned with `dumb_solid: True` and a reason - the system says it does
not know, instead of inventing constraints.
"""
from __future__ import annotations

import os

from geometry import fingerprint as FP
from geometry import invariants as INV
from geometry import kernel

SUPPORTED = (".step", ".stp", ".iges", ".igs", ".brep")


class ImportFailed(RuntimeError):
    """The file could not be read as a solid, and the reason is stated."""


def import_shape(path: str, k=None):
    """-> kernel shape. Raises ImportFailed for a wrong extension, missing file or bad transfer.

    `k` lets a caller pass the kernel module it is already using (the endpoints inject theirs, so a
    test can exercise the whole path without OCP installed).
    """
    kernel_module = kernel if k is None else k
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in SUPPORTED:
        raise ImportFailed(f"{path!r}: unsupported format {ext or '(none)'}; use {', '.join(SUPPORTED)}")
    if not os.path.exists(path):
        raise ImportFailed(f"{path!r}: file not found")
    try:
        shape = kernel_module.read_step(path)
    except kernel.KernelUnavailable as exc:
        raise ImportFailed(str(exc)) from None
    except Exception as exc:
        raise ImportFailed(f"{path!r}: kernel could not read the file ({exc})") from None
    if shape is None:
        raise ImportFailed(f"{path!r}: kernel returned no shape")
    return shape


def inspect(shape) -> dict:
    """Summary + invariants + fingerprint + recognition, in one record."""
    summary = kernel.measure(shape)
    checks = INV.solid_checks({**summary, "watertight": summary.get("watertight", False),
                               "degenerate_faces": summary.get("degenerate_faces", 0)})
    try:
        rec = recognise(shape)
    except kernel.KernelUnavailable as exc:
        rec = {"recognised": False, "dumb_solid": True, "candidates": [],
               "reason": f"no kernel for recognition: {exc}"}
    return {"summary": summary, "checks": checks, "all_ok": INV.all_ok(checks),
            "geometry_hash": FP.fingerprint(summary), "recognition": rec}


def face_kinds(shape) -> dict:
    """Count faces by surface type, which is what recognition reasons about."""
    kernel.require()
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import (GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                             GeomAbs_Torus)
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    names = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
             GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus"}
    counts = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = kernel.as_face(exp.Current())
        kind = names.get(BRepAdaptor_Surface(face, True).GetType(), "other")
        counts[kind] = counts.get(kind, 0) + 1
        exp.Next()
    return counts


def candidates_from_counts(counts: dict) -> dict:
    """Pure, kernel-free: the recognition rules live here so they can be tested directly."""
    counts = {k: int(v) for k, v in (counts or {}).items() if v}
    cands = []
    if counts.get("cylinder"):
        n = counts["cylinder"]
        cands.append({"feature": "hole", "count": n, "confidence": round(min(0.9, 0.4 + 0.05 * n), 2),
                      "basis": f"{n} cylindrical face(s); a cylinder may also be a boss or a pin"})
    if counts.get("torus"):
        n = counts["torus"]
        cands.append({"feature": "fillet", "count": n, "confidence": round(min(0.85, 0.35 + 0.05 * n), 2),
                      "basis": f"{n} toroidal face(s); sizes must be measured, not assumed"})
    if counts.get("cone"):
        n = counts["cone"]
        cands.append({"feature": "chamfer", "count": n, "confidence": round(min(0.8, 0.3 + 0.05 * n), 2),
                      "basis": f"{n} conical face(s); could be a countersink"})
    if counts.get("plane") and not any(c["feature"] in ("hole", "fillet") for c in cands):
        cands.append({"feature": "prismatic", "count": counts["plane"], "confidence": 0.45,
                      "basis": "planar faces only: a prismatic part, feature history unknown"})
    recognised = bool(cands)
    return {"recognised": recognised, "dumb_solid": not recognised, "candidates": cands,
            "face_kinds": counts,
            "reason": ("candidates only; nothing is converted into constraints without confirmation"
                       if recognised else
                       "no recognisable feature: imported as a dumb solid, parameters must be given by the user")}


def recognise(shape) -> dict:
    return candidates_from_counts(face_kinds(shape))
