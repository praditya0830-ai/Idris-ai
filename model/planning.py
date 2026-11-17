"""Deterministic CAD planning and measured review helper.

This module does not build geometry or claim kernel verification. It only accepts
explicit parts, interface metadata, and measured dimensions, then decides whether a
model should be treated as a part or an assembly and emits review findings that are
anchored to those measurements.
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
    """Return a deterministic part/assembly plan and measured review entries.

    The planner validates explicit input and is intentionally conservative: if the
    geometry is not measured or the model is ambiguous, it returns an assembly with
    a needs list instead of inventing geometry.
    """
    if not isinstance(model, dict):
        raise ValueError("model must be an object")

    raw_parts = model.get("parts", [])
    raw_interfaces = model.get("interfaces", [])
    raw_needs = model.get("needs", [])
    if not isinstance(raw_parts, list):
        raise ValueError("parts must be an array")
    if not isinstance(raw_interfaces, list):
        raise ValueError("interfaces must be an array")
    if not isinstance(raw_needs, list):
        raise ValueError("needs must be an array")

    parts = []
    ids = set()
    for idx, part in enumerate(raw_parts):
        if not isinstance(part, dict) or not isinstance(part.get("id"), str) or not part["id"].strip():
            raise ValueError(f"parts[{idx}] must have a non-empty id")
        pid = part["id"].strip()
        if pid in ids:
            raise ValueError(f"duplicate part id {pid!r}")
        ids.add(pid)

        qty = part.get("qty", 1)
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
            raise ValueError(f"parts[{idx}].qty must be a positive integer")

        measurements = part.get("measurements", {})
        if not isinstance(measurements, dict):
            raise ValueError(f"parts[{idx}].measurements must be an object")

        parts.append({
            "id": pid,
            "role": str(part.get("role") or "component"),
            "qty": qty,
            "make": str(part.get("make") or "unspecified"),
            **({"measurements": dict(measurements)} if measurements else {}),
        })

    interfaces = []
    for idx, interface in enumerate(raw_interfaces):
        if not isinstance(interface, dict):
            raise ValueError(f"interfaces[{idx}] must be an object")
        a = interface.get("a")
        b = interface.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"interfaces[{idx}] must have string a and b ids")
        if a not in ids or b not in ids or a == b:
            raise ValueError(f"interfaces[{idx}] must connect two different declared part ids")
        kind = str(interface.get("type") or "unknown").strip().lower()
        if not kind:
            raise ValueError(f"interfaces[{idx}].type must be non-empty")

        row = {"a": a, "b": b, "type": kind}
        if interface.get("at") is not None:
            at = interface["at"]
            if not isinstance(at, list) or len(at) != 3:
                raise ValueError(f"interfaces[{idx}].at must contain 3 coordinates")
            row["at"] = [_number(v, f"interfaces[{idx}].at[{i}]") for i, v in enumerate(at)]
        if interface.get("gap_mm") is not None:
            row["gap_mm"] = _number(interface["gap_mm"], f"interfaces[{idx}].gap_mm")
        interfaces.append(row)

    needs = []
    for idx, need in enumerate(raw_needs):
        if not isinstance(need, str) or not need.strip():
            raise ValueError(f"needs[{idx}] must be a non-empty string")
        value = need.strip()
        if value not in needs:
            needs.append(value)

    has_non_fused = any(i["type"] in _NON_FUSED for i in interfaces)
    has_only_fused = bool(interfaces) and all(i["type"] in _FUSED for i in interfaces)
    if has_non_fused:
        kind = "assembly"
        why = "one or more interfaces are non-fused and remain explicit"
    elif len(parts) > 1:
        kind = "assembly"
        why = f"the model declares {len(parts)} distinct part records"
    elif len(parts) == 1 and (not interfaces or has_only_fused):
        kind = "part"
        why = "one declared body with no relative-motion or non-fused interface"
    elif needs:
        kind = "assembly"
        why = "the model is unresolved; ask rather than inventing the interface"
    else:
        kind = "assembly"
        why = "ambiguous contract: defaulting to assembly is safer than a fake monolithic part"
        if not needs:
            needs.append("declare the part and its interfaces explicitly")

    review = []

    def add(code: str, part: str, measured: dict, message: str, fix: str):
        review.append({
            "id": code,
            "part": part,
            "measured": measured,
            "message": message,
            "fix": fix,
        })

    for part in parts:
        m = part.get("measurements") or {}

        od = m.get("od_mm")
        length = m.get("length_mm")
        if od is not None and length is not None:
            od = _number(od, f"{part['id']}.od_mm", minimum=1e-12)
            length = _number(length, f"{part['id']}.length_mm", minimum=0.0)
            ratio = length / od if od > 0 else float("inf")
            if ratio > 40:
                add(
                    "MEMBER_SLENDER",
                    part["id"],
                    {"length_mm": length, "od_mm": od, "L_over_D": ratio},
                    f"Member {part['id']} has L/D = {ratio:.1f}, exceeding the 40 screening limit.",
                    "Increase the outer diameter or add a gusset/support and re-run the structural check.",
                )

        wall = m.get("wall_mm")
        if wall is not None:
            wall = _number(wall, f"{part['id']}.wall_mm", minimum=1e-12)
            if wall < 1.5:
                add(
                    "WALL_THIN",
                    part["id"],
                    {"wall_mm": wall, "threshold_mm": 1.5},
                    f"Measured wall thickness is {wall:.2f} mm, below the 1.50 mm minimum.",
                    "Increase the wall thickness to at least 1.50 mm or justify an approved thinner wall.",
                )

        rmin = m.get("minimum_inner_radius_mm")
        if rmin is not None:
            rmin = _number(rmin, f"{part['id']}.minimum_inner_radius_mm", minimum=0.0)
            if rmin < 0.5:
                add(
                    "CORNER_SHARP",
                    part["id"],
                    {"minimum_inner_radius_mm": rmin, "threshold_mm": 0.5},
                    f"Measured minimum internal corner radius is {rmin:.2f} mm, below R0.50 mm.",
                    "Add a fillet of at least R0.50 mm to the internal corner.",
                )

    for interface in interfaces:
        gap = interface.get("gap_mm")
        if gap is not None and gap < 0.2:
            add(
                "JOINT_GAP",
                f"{interface['a']}↔{interface['b']}",
                {"gap_mm": gap, "threshold_mm": 0.2},
                f"Interface gap is {gap:.3f} mm, below the 0.200 mm threshold.",
                "Confirm the interface with explicit contact or interference analysis before treating it as a valid joint.",
            )

    return {
        "kind": kind,
        "why": why,
        "parts": parts,
        "interfaces": interfaces,
        "needs": needs,
        "review": review,
        "evidence": "explicit part metadata and measured values only; no geometry was built by this planner",
    }


__all__ = ["plan_model"]


if __name__ == "__main__":
    sample = {
        "parts": [{"id": "tube", "measurements": {"length_mm": 620, "od_mm": 14, "wall_mm": 1.2, "minimum_inner_radius_mm": 0.2}}],
        "interfaces": [],
        "needs": [],
    }
    print(plan_model(sample))
    