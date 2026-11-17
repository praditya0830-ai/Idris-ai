"""Strength questions answered by a solve, not by an analogy.

physics/part_fea.py already does static linear FEA on any closed mesh (voxel hex8, measured against
bar tension, Timoshenko cantilever, Kirsch and NAFEMS LE1). This module is the chat-facing wrapper:

- it builds the load case (support + load) from a name, so the chat does not have to invent selectors;
- it runs the solver and returns claims in the same shape as ai/engineer.py (value, unit, formula,
  source, evidence 3 = numerical solver);
- it cross-checks the peak stress and deflection against the closed form for the simple case, because a
  single number from one method is not evidence;
- it turns the numbers into advice (safe / thin / fails) with the number that produced it.
"""
from __future__ import annotations

import math

from physics import part_fea

# material ids the solver knows -> the bench's own table, so a number never changes meaning silently
SOLVER_MATERIALS = tuple(part_fea.MATERIALS)

CASES = {
    "cantilever_tip": {
        "label": "cantilever, load at the free end",
        "supports": [{"where": {"plane": "x", "side": "min"}}],
        "load_axis": "z",
        "load_where": {"plane": "x", "side": "max"},
        "closed_form": "cantilever",
        "formula": "σ = M·c/I with M = F·L; δ = F·L³/(3·E·I)",
    },
    "cantilever_tip_x": {
        "label": "cantilever along y, load along x at the free end",
        "supports": [{"where": {"plane": "y", "side": "min"}}],
        "load_axis": "x",
        "load_where": {"plane": "y", "side": "max"},
        "closed_form": "cantilever_x",
        "formula": "σ = M·c/I with M = F·L; δ = F·L³/(3·E·I)",
    },
    "simply_supported_center": {
        "label": "supported on both end faces, load in the middle",
        "supports": [{"where": {"box": None}}],       # filled at run time: the two end bands
        "load_axis": "z",
        "load_where": {"plane": "x", "side": None},   # filled at run time: the middle band
        "closed_form": "beam_center",
        "formula": "σ = M·c/I with M = F·L/4; δ = F·L³/(48·E·I)",
    },
}


def material_id(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    alias = {"steel_1018_cd": "steel", "steel_1018": "steel", "steel_s235": "steel", "baja": "steel",
             "al_6061_t6": "aluminium", "6061": "aluminium", "aluminium": "aluminium",
             "al_7075_t6": "aluminium", "ti_6al4v": "titanium", "titanium": "titanium",
             "nylon_pa66": "nylon", "pla": "pla", "stainless": "stainless"}
    key = alias.get(key, key)
    if key not in SOLVER_MATERIALS:
        raise ValueError(f"the solver has no material {name!r}; use one of {list(SOLVER_MATERIALS)}")
    return key


def _bbox(points):
    xs = points[0::3]
    ys = points[1::3]
    zs = points[2::3]
    return [(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))]


def _geometry_bbox(geometry: dict):
    pos = geometry.get("positions")
    if not pos:
        return None
    return _bbox([float(v) for v in pos])


def _snapped_element(geometry: dict, target_elements: int) -> float | None:
    """Use an element size that divides the longest side, so a straight part voxelizes without stair steps."""
    bb = _geometry_bbox(geometry)
    if not bb:
        return None
    (x0, y0, z0), (x1, y1, z1) = bb
    span = max(x1 - x0, y1 - y0, z1 - z0)
    vol_box = max((x1 - x0) * (y1 - y0) * (z1 - z0), 1e-9)
    h0 = (vol_box / max(int(target_elements), 8)) ** (1.0 / 3.0)
    n = max(2, int(round(span / h0)))
    return span / n


def analyze(geometry: dict, material: str = "steel", case: str = "cantilever_tip",
            force_n: float | None = None, force_vector=None, element_size_mm: float | None = None,
            target_elements: int = 20000, load_scale: float = 1.0) -> dict:
    """Run the part solver for one named case and return claims, checks and advice.

    `geometry`: {"positions": [...], "indices": [...]} or {"stl_base64": "..."} in mm.
    The force is a magnitude in N along the case's axis unless `force_vector` is given.
    """
    key = (case or "cantilever_tip").strip().lower()
    if key not in CASES:
        raise ValueError(f"unknown case {case!r}; use {sorted(CASES)}")
    spec = CASES[key]
    if force_n is None and not force_vector:
        raise ValueError("a force is needed: pass force_n (magnitude, N) or force_vector [Fx, Fy, Fz]")
    mat = material_id(material)
    payload = dict(geometry)
    payload.update({"material": mat, "target_elements": int(target_elements)})
    bbox = _geometry_bbox(geometry)
    if element_size_mm:
        payload["element_size_mm"] = float(element_size_mm)
    elif bbox:
        h = _snapped_element(geometry, target_elements)
        if h:
            payload["element_size_mm"] = h

    if key == "simply_supported_center" and bbox:
        (x0, y0, z0), (x1, y1, z1) = bbox
        span = x1 - x0
        payload["supports"] = [{"where": {"box": [[x0 - 1, y0 - 1, z0 - 1],
                                                   [x0 + 1e-3 + 0.02 * span, y1 + 1, z0 + 1]]}},
                               {"where": {"box": [[x1 - 1e-3 - 0.02 * span, y0 - 1, z0 - 1],
                                                   [x1 + 1, y1 + 1, z0 + 1]]}}]
        mid0, mid1 = x0 + 0.49 * span, x0 + 0.51 * span
        vec = [0.0, 0.0, 0.0]
        vec[{"x": 0, "y": 1, "z": 2}[spec["load_axis"]]] = -abs(float(force_n or 0))
        payload["loads"] = [{"force_N": vec, "where": {"box": [[mid0, y0 - 1, z1 - 1], [mid1, y1 + 1, z1 + 1]]}}]
    else:
        payload["supports"] = spec["supports"]
        f = [float(v) for v in force_vector] if force_vector else [0.0, 0.0, 0.0]
        if not force_vector:
            f[{"x": 0, "y": 1, "z": 2}[spec["load_axis"]]] = -abs(float(force_n))
        f = [c * float(load_scale) for c in f]
        payload["loads"] = [{"force_N": f, "where": spec["load_where"]}]

    out = part_fea.run(payload)
    s = out["surface"]
    surf_bbox = _bbox(s["positions"]) if s.get("positions") else None
    span = (surf_bbox[1][0] - surf_bbox[0][0]) if surf_bbox else None
    sy = float(out["yield_mpa"])
    sigma = float(out["max_von_mises_MPa"])
    delta = float(out["max_displacement_mm"])
    sf = float(out["min_safety_factor"])
    applied = float(sum(abs(c) for c in (out["applied_loads"][0]["force_N"]
                                         if out["applied_loads"] else [0, 0, 0])))

    cross = _cross_check(spec["closed_form"], out, span, applied)
    checks = [
        {"id": "fea.mesh_matches_volume", "ok": abs(out["volume"]["error_pct"]) <= 5.0,
         "detail": f"voxel volume {out['volume']['voxel_mm3']} mm³ vs mesh {out['volume']['mesh_mm3']} mm³ "
                   f"({out['volume']['error_pct']} %)"},
        {"id": "fea.equilibrium", "ok": float(out["equilibrium_error"]) <= 0.01,
         "detail": f"|R + F| / Σ|F| = {out['equilibrium_error']} (load reaches the supports)"},
        {"id": "fea.below_yield", "ok": sf >= 1.0,
         "detail": f"σ_max {sigma:.1f} MPa vs Sy {sy:.0f} MPa → SF {sf:.2f}"},
    ]
    advice = _advice(sf, sigma, sy, delta, span, float(out["equilibrium_error"]),
                     float(out["volume"]["error_pct"]), cross)

    claims = [
        _claim("fea.sigma_max", "peak von Mises stress", sigma, "MPa",
               "FEM: hex8 voxel model, static linear", {"material": mat, "case": key, "force_N": applied},
               "physics/part_fea.py (benchmarked: bar tension, Timoshenko cantilever, Kirsch, NAFEMS LE1)"),
        _claim("fea.delta_max", "peak displacement", delta, "mm",
               "FEM: same solve, |u| max", {"case": key, "force_N": applied}, "physics/part_fea.py"),
        _claim("fea.sf_yield", "safety factor on yield", sf, "-", "SF = Sy / σ_max",
               {"sy_mpa": sy, "sigma_mpa": sigma}, "definition"),
        _claim("fea.mass", "mass", float(out["mass_kg"]), "kg", "m = V·ρ",
               {"volume_mm3": out["volume"]["mesh_mm3"], "rho_kg_m3": out["material"]["rho"]},
               "mesh volume × solver material density"),
    ]
    if cross.get("ratio_sigma"):
        claims.append(_claim("fea.sigma_closed_form", "closed-form bending stress", cross["sigma_cf"], "MPa",
                             cross["formula_sigma"], {"span_mm": span, "force_N": applied},
                             "closed form (independent of the solve)", evidence=2))
    return {"case": key, "case_label": spec["label"], "material": out["material"]["name"],
            "claims": claims, "checks": checks, "advice": advice, "cross_check": cross,
            "mesh": out["mesh"], "volume": out["volume"], "solve_time_s": out["solve_time_s"],
            "equilibrium_error": out["equilibrium_error"], "bbox_mm": surf_bbox}


def _claim(cid, label, value, unit, formula, inputs, source, evidence=3):
    return {"id": cid, "label": label, "value": value, "unit": unit, "formula": formula,
            "inputs": inputs, "source": source, "evidence": evidence,
            "evidence_name": {1: "AI estimate", 2: "closed-form calculation", 3: "numerical solver"}[evidence]}


def _cross_check(kind: str, out: dict, span, applied: float) -> dict:
    """The same quantity from a closed form, so a wrong mesh or a wrong load path shows up as a mismatch."""
    if not span or not applied:
        return {}
    pos = out["surface"]["positions"]
    (x0, y0, z0), (x1, y1, z1) = _bbox(pos)
    E = float(out["material"]["E"])
    if kind in ("cantilever", "cantilever_x", "beam_center"):
        if kind == "cantilever_x":
            length = y1 - y0
            b = z1 - z0
            h = x1 - x0
        else:
            length = span
            b = y1 - y0
            h = z1 - z0
        I = b * h ** 3 / 12.0 if b > 0 and h > 0 else 0.0
        if I <= 0:
            return {}
        M = applied * length if kind != "beam_center" else applied * length / 4.0
        sigma_cf = M * (h / 2.0) / I
        delta_cf = (applied * length ** 3 / (3 * E * I) if kind != "beam_center"
                    else applied * length ** 3 / (48 * E * I))
        return {"kind": kind, "span_mm": round(length, 3), "I_mm4": round(I, 3),
                "sigma_cf": sigma_cf, "delta_cf": delta_cf,
                "ratio_sigma": out["max_von_mises_MPa"] / sigma_cf if sigma_cf else None,
                "ratio_delta": out["max_displacement_mm"] / delta_cf if delta_cf else None,
                "formula_sigma": "σ = M·c/I (closed form, for the box that bounds the part)"}
    return {}


def _advice(sf, sigma, sy, delta, span, eq_err, vol_err, cross) -> list:
    out = []
    if sf < 1.0:
        out.append(f"FAIL: peak {sigma:.0f} MPa is above yield {sy:.0f} MPa (SF {sf:.2f}). "
                   f"Thicken the section, shorten the span, or pick a stronger material.")
    elif sf < 1.5:
        out.append(f"THIN: SF {sf:.2f} on yield. Survives the load once, but has little room for a notch, "
                   f"a real weld or a load you did not state.")
    else:
        out.append(f"OK: SF {sf:.2f} on yield ({sigma:.0f} MPa against {sy:.0f} MPa).")
    if eq_err > 0.01:
        out.append(f"CHECK THE MODEL: reaction and load differ by {100 * eq_err:.1f} %. "
                   f"The number is not trustworthy until the load reaches a support.")
    if vol_err > 5.0:
        out.append(f"COARSE MESH: voxel volume is off by {vol_err:.1f} %. Refine element_size_mm before "
                   f"quoting a peak stress.")
    if delta and span and delta > span / 100.0:
        out.append(f"FLEXIBLE: {delta:.2f} mm over a {span:.0f} mm span is more than L/100; "
                   f"stiffness, not strength, is the limit here.")
    if cross.get("ratio_sigma"):
        r = cross["ratio_sigma"]
        if not 0.75 <= r <= 1.35:
            out.append(f"THE SOLVE AND THE CLOSED FORM DISAGREE: FEA {sigma:.0f} MPa vs closed form "
                       f"{cross['sigma_cf']:.0f} MPa (ratio {r:.2f}). Treat this as unresolved, not as a result.")
        else:
            out.append(f"Cross-check agrees: FEA {sigma:.0f} MPa vs closed form {cross['sigma_cf']:.0f} MPa "
                       f"(ratio {r:.2f}).")
    return out
