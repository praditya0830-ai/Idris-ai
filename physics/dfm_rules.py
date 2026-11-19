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
# Review of PR #5-#8, item 6: a source may only be quoted when it really contains the number. ISO 2553 is the
# standard for welding SYMBOLS and says nothing about a minimum fillet leg, so it is gone from this table.
# Where the document (and the table or clause) could be pinned down, the rule may fail a part; where it could
# not, the rule says so and its result is a warning, never a failure.
VERIFIED = "verified"
VERIFIED_SERIES = "standard series"
UNVERIFIED = "guideline, source not verified"

# AWS D1.1/D1.1M:2025, Clause 7.13, Table 7.7 "Minimum Fillet Weld Sizes" (Table 5.8 in the 2015/2020
# editions), leg length against the base-metal thickness T: T <= 6 mm -> 3 mm, 6 < T <= 12 -> 5 mm,
# 12 < T <= 20 -> 6 mm, T > 20 mm -> 8 mm. T governs by the thicker part joined for a non-low-hydrogen single
# pass, by the thinner part for low-hydrogen processes, and the leg need not exceed the thinner part.
WELD_SOURCE = ("AWS D1.1/D1.1M:2025 Clause 7.13, Table 7.7 (Minimum Fillet Weld Sizes; Table 5.8 in the "
               "2015/2020 editions): minimum fillet leg 3 mm up to 6 mm base-metal thickness, 5 mm up to 12 mm, "
               "6 mm up to 20 mm, 8 mm above 20 mm. The leg need not exceed the thinner part joined.")
WELD_MIN_FILLET_MM = ((6.0, 3.0), (12.0, 5.0), (20.0, 6.0), (None, 8.0))

WALL_SOURCE = {
    "cnc": {"text": "Machining guideline for thin walls: 0.8 mm aluminium/steel, 1.0 mm stainless/titanium. "
                     "No table or clause in a standard was found for these numbers, so it is a guideline.",
            "status": UNVERIFIED},
    "fdm": {"text": "FDM guideline: minimum wall = 3 x nozzle width (1.2 mm with a 0.4 mm nozzle). A rule of "
                     "thumb, not a clause in a standard.",
            "status": UNVERIFIED},
    "sheet": {"text": "Sheet-metal guideline: one thickness everywhere, minimum 0.5 mm steel/stainless, "
                       "1.0 mm aluminium. A workshop guideline, not a clause in a standard.",
              "status": UNVERIFIED},
    "weld": {"text": WELD_SOURCE, "status": VERIFIED},
}


def weld_fillet_min_mm(thickness_mm: float) -> float:
    """The AWS D1.1 Table 7.7 minimum fillet leg for a base-metal thickness. No number is invented here."""
    for limit, leg in WELD_MIN_FILLET_MM:
        if limit is None or thickness_mm <= limit:
            return leg
    return WELD_MIN_FILLET_MM[-1][1]

# ------------------------------------------------------------------ drilled holes (mm)
DRILL_MM = (1.0, 1.1, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.3, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0,
            6.5, 6.8, 7.0, 8.0, 8.5, 9.0, 10.0, 10.2, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0)
DRILL_SOURCE = ("DIN 338:2006-09 / ISO 235 twist-drill diameter series (the diameters the standard series "
                "offers). The list above is that series; the rule that a hole outside it needs reaming or "
                "milling is a design guideline built on the series.")
DRILL_SOURCE_STATUS = VERIFIED_SERIES
DRILL_TOL_MM = 0.05

# ------------------------------------------------------------------ checks that need the model, not the params
NOT_CHECKED_FROM_PARAMS = (
    ("undercuts", "undercuts need the solid; the CAD kernel is in the browser"),
    ("tool_access", "tool access needs the solid and the setup; it is not checked from parameters"),
    ("overhang", "overhang angle needs the solid; the CAD kernel is in the browser"),
    ("inside_corner_radius", "the inside corner radius is in the sketch, not in the parameter list"),
)
INSIDE_CORNER_MIN_MM = 3.0                     # radius of a 3 mm cutter: the smallest inside corner CNC can make
INSIDE_CORNER_SOURCE = ("Milling guideline, source not verified: the smallest inside corner radius equals the "
                        "cutter radius (3 mm end mill -> 3 mm). Only the rule is used; the radius itself is read "
                        "from the sketch, so this check stays not_checked.")


def _check(cid: str, name: str, status: str, *, value=None, limit=None, unit: str = "mm",
           source: str | None = None, source_status: str | None = None, message: str = "") -> dict:
    out = {"id": cid, "check": name, "status": status, "unit": unit, "message": message}
    if value is not None:
        out["value"] = value
    if limit is not None:
        out["limit"] = limit
    if source:
        out["source"] = source
    if source_status:
        out["source_status"] = source_status
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
    rule = WALL_SOURCE.get(process) or {}
    for p in walls:
        try:
            value = float(p.get("value"))
        except (TypeError, ValueError):
            continue
        label = str(p.get("label") or p.get("k") or "wall")
        if process == "weld":
            # item 6: the minimum fillet leg depends on the base-metal thickness (AWS D1.1 Table 7.7), not on
            # one fixed number; the wall in the parameter list plays that thickness here.
            leg = weld_fillet_min_mm(value)
            ok = value + 1e-9 >= leg
            checks.append(_check("min_wall", f"minimum fillet leg ({label})", "pass" if ok else "fail",
                                 value=value, limit=leg, source=WELD_SOURCE, source_status=VERIFIED,
                                 message=(f"{value:g} mm wall takes at least a {leg:g} mm fillet leg "
                                          f"(AWS D1.1 Table 7.7)" if ok else
                                          f"{value:g} mm wall needs at least a {leg:g} mm fillet leg; this leg "
                                          f"is below the AWS D1.1 Table 7.7 minimum")))
            continue
        if limit is None:
            checks.append(_check("min_wall", f"minimum wall ({label})", "not_checked", value=value,
                                 message=f"no minimum-wall rule for process '{process or 'unknown'}' and "
                                         f"material '{material or 'unknown'}'; not guessed"))
            continue
        ok = value + 1e-9 >= limit
        status = "pass" if ok else ("fail" if rule.get("status") == VERIFIED else "warn")
        message = (f"{value:g} mm >= {limit:g} mm for {process} {material}" if ok else
                   f"{value:g} mm is below the {limit:g} mm "
                   f"{'minimum' if status == 'fail' else 'guideline'} for {process} {material}")
        if status == "warn":
            message += "; the guideline's source is not verified, so this is a warning, not a failure"
        checks.append(_check("min_wall", f"minimum wall ({label})", status, value=value, limit=limit,
                             source=rule.get("text"), source_status=rule.get("status"), message=message))
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
                                 value=sorted(set(odd)), source=DRILL_SOURCE, source_status=DRILL_SOURCE_STATUS,
                                 message=f"no standard drill for {', '.join(f'{d:g}' for d in sorted(set(odd)))} mm; "
                                         f"mill or ream it, or use the nearest drill diameter"))
        else:
            checks.append(_check("drill_size", "hole diameters in the drill series", "pass",
                                 value=len(holes), source=DRILL_SOURCE, source_status=DRILL_SOURCE_STATUS,
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
    warned = [c for c in checks if c["status"] == "warn"]
    return {"checks": checks, "ok": not failed, "failed": failed, "warned": warned,
            "not_checked": [c for c in checks if c["status"] == "not_checked"]}


def summary(dfm: dict) -> str:
    """One line for the answer and the report: what the table decided, with the rule that decided it."""
    if not isinstance(dfm, dict):
        return "Manufacturability: not checked."
    failed = dfm.get("failed") or []
    warned = dfm.get("warned") or []
    parts = []
    if failed:
        parts.append("FAILED — " + "; ".join(f"{c['check']}: {c['message']}" for c in failed[:3]))
    if warned:
        parts.append(f"{len(warned)} warning(s) — " + "; ".join(f"{c['check']}: {c['message']}" for c in warned[:3]))
    if parts:
        return "Manufacturability: " + "; ".join(parts)
    passed = [c for c in (dfm.get("checks") or []) if c["status"] == "pass"]
    return (f"Manufacturability: {len(passed)} rule(s) passed"
            + (f", {len(dfm.get('not_checked') or [])} not checked" if dfm.get("not_checked") else "") + ".")
