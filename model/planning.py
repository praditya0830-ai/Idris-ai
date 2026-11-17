"""Deterministic planning and geometry-measurement review for CAD models.

This module deliberately plans from explicit part/interface data. It does not infer a
manufacturing assembly from an opaque AI prompt, and it does not claim that a mesh has
been built. Geometry findings are only emitted when the corresponding measurements are
provided by a trusted geometry/kernel step.
"""
from __future__ import annotations

from typing import Any

_NON_FUSED = {"bolt", "bearing", "slip", "revolute", "prismatic", "press_fit"}
_FUSED = {"weld", "glue", "fused", "monolithic"}


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number") from None
    if not (out == out and abs(out) != float("inf")) or out < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum:g}")
    return out


def plan_model(model: dict) -> dict:
    """Return a deterministic part/assembly plan plus findings from supplied measurements.

    Input parts are objects with ``id`` and optional ``role``, ``qty``, ``make`` and
    ``measurements``. Interfaces use ``a``, ``b``, ``type`` and optional measured
    ``gap_mm``. The caller must supply geometry-derived measurements; this function
    never pretends that descriptive metadata is a CAD kernel result.
    """
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    raw_parts = model.get("parts", [])
    raw_interfaces = model.get("interfaces", [])
    raw_needs = model.get("needs", [])
    if not isinstance(raw_parts, list) or not isinstance(raw_interfaces, list) or not isinstance(raw_needs, list):
        raise ValueError("parts, interfaces and needs must be arrays")

    parts = []
    ids = set()
    for i, part in enumerate(raw_parts):
        if not isinstance(part, dict) or not isinstance(part.get("id"), str) or not part["id"].strip():
            raise ValueError(f"parts[{i}] must have a non-empty id")
        pid = part["id"].strip()
        if pid in ids:
            raise ValueError(f"duplicate part id {pid!r}")
        ids.add(pid)
        qty = part.get("qty", 1)
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
            raise ValueError(f"parts[{i}].qty must be a positive integer")
        measurements = part.get("measurements", {})
        if not isinstance(measurements, dict):
            raise ValueError(f"parts[{i}].measurements must be an object")
        parts.append({"id": pid, "role": str(part.get("role") or "component"), "qty": qty,
                      "make": str(part.get("make") or "unspecified"),
                      **({"measurements": dict(measurements)} if measurements else {})})

    interfaces = []
    for i, interface in enumerate(raw_interfaces):
        if not isinstance(interface, dict):
            raise ValueError(f"interfaces[{i}] must be an object")
        a, b = interface.get("a"), interface.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or a not in ids or b not in ids or a == b:
            raise ValueError(f"interfaces[{i}] must connect two different declared part ids")
        kind = str(interface.get("type") or "unknown").strip().lower()
        if not kind:
            raise ValueError(f"interfaces[{i}].type must be non-empty")
        row = {"a": a, "b": b, "type": kind}
        if interface.get("at") is not None:
            at = interface["at"]
            if not isinstance(at, list) or len(at) != 3:
                raise ValueError(f"interfaces[{i}].at must contain 3 coordinates")
            row["at"] = [_number(v, f"interfaces[{i}].at") for v in at]
        if interface.get("gap_mm") is not None:
            row["gap_mm"] = _number(interface["gap_mm"], f"interfaces[{i}].gap_mm")
        interfaces.append(row)

    needs = []
    for i, need in enumerate(raw_needs):
        if not isinstance(need, str) or not need.strip():
            raise ValueError(f"needs[{i}] must be a non-empty string")
        if need.strip() not in needs:
            needs.append(need.strip())

    non_fused = [i for i in interfaces if i["type"] in _NON_FUSED]
    all_fused = bool(interfaces) and all(i["type"] in _FUSED for i in interfaces)
    if non_fused:
        kind = "assembly"
        why = "one or more interfaces are non-fused and must remain explicit"
    elif len(parts) > 1:
        kind = "assembly"
        why = f"the model declares {len(parts)} distinct parts"
    elif needs or (interfaces and not all_fused):
        kind = "assembly"
        why = "part/assembly intent or an interface is unresolved; preserve the interface and ask"
    elif len(parts) == 1:
        kind = "part"
        why = "one declared body with no relative-motion or non-fused interface"
    else:
        kind = "assembly"
        why = "no explicit monolithic part was declared; ask rather than inventing one"
        if not needs:
            needs.append("declare the part(s) and their interfaces")

    findings = []
    def add(code, part, measured, message, fix):
        findings.append({"id": code, "part": part, "measured": measured, "message": message, "fix": fix})

    for part in parts:
        m = part.get("measurements") or {}
        od = m.get("od_mm")
        length = m.get("length_mm")
        if od is not None and length is not None:
            od, length = _number(od, f"{part['id']}.od_mm", minimum=1e-12), _number(length, f"{part['id']}.length_mm")
            ratio = length / od
            if ratio > 40:
                add("MEMBER_SLENDER", part["id"], {"length_mm": length, "od_mm": od, "L_over_D": ratio},
                    f"Member length/diameter is {ratio:.1f} (limit 40).",
                    "Increase the outer diameter or add a measured support/gusset, then re-run structural analysis.")
        wall = m.get("wall_mm")
        if wall is not None:
            wall = _number(wall, f"{part['id']}.wall_mm", minimum=1e-12)
            if wall < 1.5:
                add("WALL_THIN", part["id"], {"wall_mm": wall, "minimum_mm": 1.5},
                    f"Measured wall thickness is {wall:.2f} mm, below the 1.50 mm screening minimum.",
                    "Increase the measured wall to at least 1.50 mm or select a process with a validated lower limit.")
        radius = m.get("minimum_inner_radius_mm")
        if radius is not None:
            radius = _number(radius, f"{part['id']}.minimum_inner_radius_mm")
            if radius < 0.5:
                add("CORNER_SHARP", part["id"], {"minimum_inner_radius_mm": radius, "minimum_mm": 0.5},
                    f"Measured minimum internal corner radius is {radius:.2f} mm, below R0.50 mm.",
                    "Add a fillet of at least R0.50 mm and re-check the resulting geometry.")
    for interface in interfaces:
        gap = interface.get("gap_mm")
        if gap is not None and gap < 0.2:
            add("JOINT_GAP", f"{interface['a']}↔{interface['b']}", {"gap_mm": gap, "threshold_mm": 0.2},
                f"Measured interface gap is {gap:.3f} mm, below the 0.200 mm review threshold; contact is not established by this measurement alone.",
                "Confirm the interface with kernel contact/interference analysis and define the intended fit or joint.")

    return {"kind": kind, "why": why, "parts": parts, "interfaces": interfaces,
            "needs": needs, "review": findings,
            "evidence": "metadata_and_explicit_measurements; no geometry was built by this planner"}
