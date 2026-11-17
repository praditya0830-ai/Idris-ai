"""Findings for a solved part, written in the code names of the team plan (docs/TEAM-PLAN.md §4).

Every entry is derived from numbers the solver already produced, and carries the measured value that
triggered it. Nothing here is a suspicion: if a rule cannot be decided from the numbers, it is not emitted.

Rules
    MEMBER_SLENDER   longest bounding dimension / shortest > 40          (a strut that will bend)
    WALL_THIN        mean wall thickness 2V/S < 1.5 mm                   (below a printable/machinable wall)
    UNSUPPORTED      |R + F| / Σ|F| > 1 %                                (the model does not balance)
    NO_LOAD_PATH     force declared, but zero stress and zero deflection  (the load never entered the part)
    MESH_COARSE      |voxel volume / mesh volume - 1| > 5 %              (stair-steps are dominating)
"""
from __future__ import annotations

import math

SLENDER_RATIO = 40.0
MIN_WALL_MM = 1.5
EQUILIBRIUM_TOL = 0.01
MESH_VOLUME_TOL_PCT = 5.0
ZERO_STRESS_MPA = 1e-6
ZERO_DISPLACEMENT_MM = 1e-9


def _bbox(points):
    xs, ys, zs = points[0::3], points[1::3], points[2::3]
    return [(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))]


def _surface_area_mm2(surface: dict) -> float:
    """Sum of the triangle areas of the part surface (mm²)."""
    pos = surface.get("positions") or []
    idx = surface.get("indices") or []
    total = 0.0
    for t in range(0, len(idx) - 2, 3):
        i0, i1, i2 = idx[t], idx[t + 1], idx[t + 2]
        ax, ay, az = pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]
        bx, by, bz = pos[3 * i1] - ax, pos[3 * i1 + 1] - ay, pos[3 * i1 + 2] - az
        cx, cy, cz = pos[3 * i2] - ax, pos[3 * i2 + 1] - ay, pos[3 * i2 + 2] - az
        nx, ny, nz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
        total += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
    return total


def _findings(code, message, measured, severity="warn"):
    return {"code": code, "message": message, "measured": measured, "severity": severity}


def _normalise(out: dict) -> dict:
    """Accept either the raw solver result or the /part-check shape (numbers under `summary`).

    Both callers must get the same findings from the same solve; a wrapped result that silently produced
    a different verdict would be worse than no review at all.
    """
    if "max_von_mises_MPa" in out or "summary" not in out:
        return out
    s = out.get("summary") or {}
    flat = dict(out)
    flat.setdefault("max_von_mises_MPa", s.get("max_von_mises_mpa"))
    flat.setdefault("max_displacement_mm", s.get("max_displacement_mm"))
    return flat


def review_part(out: dict) -> list:
    """-> findings for one part_fea.run() result (also used by /part-check)."""
    out = _normalise(out or {})
    findings = []
    surface = out.get("surface") or {}
    pos = surface.get("positions") or []
    sigma = float(out.get("max_von_mises_MPa") or 0.0)
    delta = float(out.get("max_displacement_mm") or 0.0)
    eq = float(out.get("equilibrium_error") or 0.0)
    applied = [load.get("force_N") or [0, 0, 0] for load in (out.get("applied_loads") or [])]
    applied_total = float(sum(abs(c) for f in applied for c in f))

    if pos:
        (x0, y0, z0), (x1, y1, z1) = _bbox(pos)
        dims = sorted([x1 - x0, y1 - y0, z1 - z0])
        if dims[0] > 0:
            ratio = dims[2] / dims[0]
            if ratio > SLENDER_RATIO:
                findings.append(_findings(
                    "MEMBER_SLENDER",
                    f"the part is {ratio:.0f}:1 ({dims[2]:.0f} x {dims[1]:.0f} x {dims[0]:.0f} mm); "
                    f"a member this slender bends before it is stressed, so check deflection, not only stress",
                    {"ratio": round(ratio, 2), "dims_mm": [round(d, 3) for d in dims]}))
        area = _surface_area_mm2(surface)
        volume = float((out.get("volume") or {}).get("mesh_mm3") or 0.0)
        if area > 0 and volume > 0:
            t_mean = 2.0 * volume / area
            if t_mean < MIN_WALL_MM:
                findings.append(_findings(
                    "WALL_THIN",
                    f"mean wall thickness 2V/S = {t_mean:.2f} mm is below {MIN_WALL_MM} mm; "
                    f"a wall this thin is a printing/handling problem before it is a strength problem",
                    {"mean_thickness_mm": round(t_mean, 3), "area_mm2": round(area, 2),
                     "formula": "t = 2V/S (exact for a plate, mean for anything else)"}))

    if eq > EQUILIBRIUM_TOL:
        findings.append(_findings(
            "UNSUPPORTED",
            f"reaction and load differ by {100 * eq:.1f} %; the model is not in equilibrium, "
            f"so the stress field does not describe a part that is actually held",
            {"equilibrium_error": round(eq, 6)}, severity="fail"))

    if applied_total >= 1.0 and abs(sigma) <= ZERO_STRESS_MPA and abs(delta) <= ZERO_DISPLACEMENT_MM:
        findings.append(_findings(
            "NO_LOAD_PATH",
            f"{applied_total:.0f} N was declared but the part shows neither stress nor deflection; "
            f"the load selection probably missed the part",
            {"applied_N": round(applied_total, 3), "max_von_mises_MPa": sigma,
             "max_displacement_mm": delta}, severity="fail"))

    err = float((out.get("volume") or {}).get("error_pct") or 0.0)
    if abs(err) > MESH_VOLUME_TOL_PCT:
        findings.append(_findings(
            "MESH_COARSE",
            f"voxel volume is off by {err:.1f} %; refine the element size before quoting a peak stress",
            {"volume_error_pct": err}))
    return findings


def summary(findings: list) -> dict:
    return {"count": len(findings),
            "codes": [f["code"] for f in findings],
            "fails": [f["code"] for f in findings if f["severity"] == "fail"]}
