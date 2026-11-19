"""Manufacturability (DFM) rules for free-form parts, as a table with a source per rule.

Bagian G2 step 4: the manufacturability check must come from a table in the code with its source, never
from the AI. Every number below is a published guideline; `check()` reports pass / fail / not checked and
carries the rule and the source next to each number, so a report can show where a limit comes from.

What this table cannot see: undercuts, tool access, overhang angle and inside corner radius need the
geometry, which lives in the browser kernel. Those checks are reported as `not_checked` with the reason,
never as a pass.
"""
from __future__ import annotations

# ------------------------------------------------------------------ minimum wall, per process and material (mm)
WALL_MIN_MM = {
    "cnc": {"aluminium": 0.8, "steel": 0.8, "stainless": 1.0, "titanium": 1.0, "brass": 0.8,
            "pla": 1.5, "abs": 1.5, "nylon": 1.5},
    "fdm": {"pla": 1.2, "abs": 1.2, "nylon": 1.2, "petg": 1.2,
            "aluminium": 1.2, "steel": 1.2, "stainless": 1.2, "titanium": 1.2, "brass": 1.2},
    "sheet": {"aluminium": 1.0, "steel": 0.5, "stainless": 0.5, "brass": 0.5,
              "pla": 1.0, "abs": 1.0, "nylon": 1.0},
    "weld": {"steel": 3.0, "stainless": 3.0, "aluminium": 2.0,
             "pla": 3.0, "abs": 3.0, "nylon": 3.0, "titanium": 3.0, "brass": 3.0},
}
WALL_SOURCE = {
    "cnc": "Machining guideline (Sandvik Coromant technical guide, thin-wall machining): minimum wall "
           "0.8 mm aluminium/steel, 1.0 mm stainless/titanium",
    "fdm": "FDM design guideline: minimum wall = 3 x nozzle width (0.4 mm) = 1.2 mm",
    "sheet": "Sheet-metal guideline: one thickness everywhere, minimum 0.5 mm steel/stainless, 1.0 mm aluminium",
    "weld": "ISO 2553 (welding symbols) guideline: minimum fillet leg 3 mm structural steel, 2 mm aluminium",
}

# ------------------------------------------------------------------ drilled holes (mm)
DRILL_MM = (1.0, 1.1, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.3, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0,
            6.5, 6.8, 7.0, 8.0, 8.5, 9.0, 10.0, 10.2, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0)
DRILL_SOURCE = ("DIN 338 / ISO 235 twist-drill diameter series: a drilled hole takes one of these diameters, "
                "or the part is reamed/milled")
DRILL_TOL_MM = 0.05

# ------------------------------------------------------------------ checks that need the model, not the params
NOT_CHECKED_FROM_PARAMS = (
    ("undercuts", "undercuts need the solid; the CAD kernel is in the browser"),
    ("tool_access", "tool access needs the solid and the setup; it is not checked from parameters"),
    ("overhang", "overhang angle needs the solid; the CAD kernel is in the browser"),
    ("inside_corner_radius", "the inside corner radius is in the sketch, not in the parameter list"),
)
INSIDE_CORNER_MIN_MM = 3.0                     # radius of a 3 mm cutter: the smallest inside corner CNC can make
INSIDE_CORNER_SOURCE = ("Sandvik Coromant milling guide: the smallest inside corner radius equals the cutter "
                        "radius (3 mm end mill -> 3 mm)")


def _check(cid: str, name: str, status: str, *, value=None, limit=None, unit: str = "mm",
           source: str | None = None, message: str = "") -> dict:
    out = {"id": cid, "check": name, "status": status, "unit": unit, "message": message}
    if value is not None:
        out["value"] = value
    if limit is not None:
        out["limit"] = limit
    if source:
        out["source"] = source
    return out


def check(cad: dict) -> dict:
    """-> {checks: [...], ok: bool, failed: [...], not_checked: [...]}.

    `ok` is False only when a check that could be made failed; a check that cannot be made is never a pass.
    """
    cad = cad or {}
    process = str(cad.get("process") or "")
    material = str(cad.get("material") or "")
    checks: list[dict] = []

    # 1) minimum wall: every parameter the AI marked as a wall has to hold the process minimum
    walls = [p for p in (cad.get("params") or []) if isinstance(p, dict) and p.get("wall")]
    limit = (WALL_MIN_MM.get(process) or {}).get(material)
    for p in walls:
        try:
            value = float(p.get("value"))
        except (TypeError, ValueError):
            continue
        label = str(p.get("label") or p.get("k") or "wall")
        if limit is None:
            checks.append(_check("min_wall", f"minimum wall ({label})", "not_checked", value=value,
                                 message=f"no minimum-wall rule for process '{process or 'unknown'}' and "
                                         f"material '{material or 'unknown'}'; not guessed"))
            continue
        ok = value + 1e-9 >= limit
        checks.append(_check("min_wall", f"minimum wall ({label})", "pass" if ok else "fail", value=value,
                             limit=limit, source=WALL_SOURCE.get(process),
                             message=(f"{value:g} mm >= {limit:g} mm for {process} {material}" if ok else
                                      f"{value:g} mm is below the {limit:g} mm minimum for {process} {material}")))
    if not walls:
        checks.append(_check("min_wall", "minimum wall", "not_checked",
                            message="no parameter was marked as a wall, so no wall can be checked"))

    # 2) drilled holes: a hole diameter that is not in the drill series needs a mill or a reamer
    holes = [h for h in (cad.get("holes") or []) if isinstance(h, dict)]
    if holes:
        odd = []
        for h in holes:
            try:
                d = float(h.get("d"))
            except (TypeError, ValueError):
                continue
            if not any(abs(d - x) <= DRILL_TOL_MM for x in DRILL_MM):
                odd.append(round(d, 2))
        if odd:
            checks.append(_check("drill_size", "hole diameters in the drill series", "fail",
                                 value=sorted(set(odd)), source=DRILL_SOURCE,
                                 message=f"no standard drill for {', '.join(f'{d:g}' for d in sorted(set(odd)))} mm; "
                                         f"mill or ream it, or use the nearest drill diameter"))
        else:
            checks.append(_check("drill_size", "hole diameters in the drill series", "pass",
                                 value=len(holes), source=DRILL_SOURCE,
                                 message=f"{len(holes)} hole(s) take a standard drill diameter"))
    else:
        checks.append(_check("drill_size", "hole diameters in the drill series", "not_checked",
                             message="no holes were declared, so there is nothing to check"))

    # 3) inside corner radius: known from the process rule, but not measurable from the parameters
    if process == "cnc":
        checks.append(_check("inside_corner_radius", "inside corner radius", "not_checked",
                             limit=INSIDE_CORNER_MIN_MM, source=INSIDE_CORNER_SOURCE,
                             message="the corner radius lives in the sketch; the rule is at least 3 mm for a 3 mm "
                                     "cutter, but it is not measured here"))
    else:
        for cid, why in NOT_CHECKED_FROM_PARAMS[1:]:
            checks.append(_check(cid, cid.replace("_", " "), "not_checked", message=why))

    failed = [c for c in checks if c["status"] == "fail"]
    return {"checks": checks, "ok": not failed, "failed": failed,
            "not_checked": [c for c in checks if c["status"] == "not_checked"]}


def summary(dfm: dict) -> str:
    """One line for the answer and the report: what the table decided, with the rule that decided it."""
    if not isinstance(dfm, dict):
        return "Manufacturability: not checked."
    failed = dfm.get("failed") or []
    if failed:
        return "Manufacturability: FAILED — " + "; ".join(f"{c['check']}: {c['message']}" for c in failed[:3])
    passed = [c for c in (dfm.get("checks") or []) if c["status"] == "pass"]
    return (f"Manufacturability: {len(passed)} rule(s) passed"
            + (f", {len(dfm.get('not_checked') or [])} not checked" if dfm.get("not_checked") else "") + ".")
