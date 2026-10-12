"""
Idris Core — design variant generator (v0)
===========================================

Sweeps the gear design space (module, pinion teeth, face width), keeps only
designs that satisfy ALL hard constraints (envelope, bending safety factor,
contact safety factor, ratio accuracy), scores them, and returns up to five
variants taken from the Pareto front (no variant is beaten on both scores by
another one).

Scores (0-100 %), both computed from numbers, nothing invented:

  physics_pct = 70 % safety margin above the required minimum
                    (full marks at 1.5x the required safety factors)
              + 30 % lightness (mass relative to the feasible set)

  design_pct  = 40 % compactness (envelope volume relative to the feasible set)
              + 30 % smoothness (transverse contact ratio: more overlap -> quieter)
              + 30 % manufacturability rules (standard module, no near-undercut
                pinion, sensible face-width / pinion-diameter ratio, tight ratio)

NOTE: "design" here is engineering design quality / manufacturability. A real
aesthetic metric (surface styling) does not exist at this stage and is not
claimed.
"""

from __future__ import annotations

from collections import Counter

from physics.gears import (
    Requirements, evaluate, stages_for_ratio,
    GEAR_MATERIALS, HOUSING_MATERIALS, STANDARD_MODULES,
)

MODULE_SWEEP = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
PINION_TEETH_SWEEP = range(17, 37)
FACE_WIDTH_FACTORS = range(8, 31, 2)       # b = k * m
MAX_WIDTH_TO_DIAMETER = 1.4                # b / pinion pitch diameter


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_low_is_good(value: float, lo: float, hi: float) -> float:
    return 1.0 if hi - lo < 1e-12 else 1.0 - (value - lo) / (hi - lo)


def _dfm_points(r: dict) -> float:
    d = r["design"]
    dp = d["module_mm"] * d["pinion_teeth"]
    pts = 0
    pts += d["module_mm"] in STANDARD_MODULES
    pts += d["pinion_teeth"] >= 18
    pts += 0.5 <= d["face_width_mm"] / dp <= 1.2
    pts += abs(r["kinematics"]["ratio_error_pct"]) <= 1.0
    return pts / 4.0


def sweep(req: Requirements) -> tuple[list[dict], Counter]:
    """Evaluate the whole design space. Returns (feasible designs, rejection counter)."""
    feasible: list[dict] = []
    rejected: Counter = Counter()
    total = 0
    for m in MODULE_SWEEP:
        for zp in PINION_TEETH_SWEEP:
            for k in FACE_WIDTH_FACTORS:
                b = round(k * m)
                if b / (m * zp) > MAX_WIDTH_TO_DIAMETER:
                    continue
                total += 1
                r = evaluate(req, m, zp, b)
                if r["feasible"]:
                    feasible.append(r)
                else:
                    for v in r["violations"]:
                        rejected[v] += 1
    rejected["_total_candidates"] = total
    return feasible, rejected


def _score(feasible: list[dict], req: Requirements) -> None:
    masses = [r["mass_kg"]["total"] for r in feasible]
    vols = [r["envelope_needed_mm"]["width"] * r["envelope_needed_mm"]["height"] * r["envelope_needed_mm"]["depth"]
            for r in feasible]
    m_lo, m_hi, v_lo, v_hi = min(masses), max(masses), min(vols), max(vols)

    for r, vol in zip(feasible, vols):
        sf_b, sf_h = r["stress"]["sf_bending"], r["stress"]["sf_contact"]
        margin = min(sf_b / req.min_sf_bending, sf_h / req.min_sf_contact)     # >= 1 for feasible designs
        safety = _clip01((margin - 1.0) / 0.5)
        lightness = _norm_low_is_good(r["mass_kg"]["total"], m_lo, m_hi)
        compact = _norm_low_is_good(vol, v_lo, v_hi)
        smooth = _clip01((r["stress"]["contact_ratio"] - 1.2) / 0.6)
        dfm = _dfm_points(r)
        physics = 0.7 * safety + 0.3 * lightness
        design = 0.4 * compact + 0.3 * smooth + 0.3 * dfm
        r["_margin"] = margin
        r["_vol"] = vol
        r["_phys"] = physics
        r["_des"] = design
        r["scores"] = {
            "physics_pct": int(round(physics * 100)),
            "design_pct": int(round(design * 100)),
            "breakdown": {
                "safety_margin": round(safety, 3), "lightness": round(lightness, 3),
                "compactness": round(compact, 3), "smoothness": round(smooth, 3),
                "manufacturability": round(dfm, 3),
            },
        }


def _pareto_layers(items: list[dict]) -> list[list[dict]]:
    """Peel off successive Pareto fronts on (physics, design); higher is better for both."""
    remaining = list(items)
    layers = []
    while remaining:
        front = [a for a in remaining
                 if not any(b["_phys"] >= a["_phys"] and b["_des"] >= a["_des"]
                            and (b["_phys"] > a["_phys"] or b["_des"] > a["_des"]) for b in remaining)]
        layers.append(front)
        remaining = [r for r in remaining if r not in front]
    return layers


def _pick_spread(front: list[dict], n: int) -> list[dict]:
    """Up to n designs evenly spread along a front sorted by physics score."""
    front = sorted(front, key=lambda r: (-r["_phys"], -r["_des"]))
    if len(front) <= n:
        return front
    idx = sorted({round(i * (len(front) - 1) / (n - 1)) for i in range(n)})
    return [front[i] for i in idx]


def _select(feasible: list[dict], n: int) -> list[dict]:
    """Prefer the Pareto front; if it has fewer than n designs, add the best of the next layers."""
    layers = _pareto_layers(feasible)
    chosen = _pick_spread(layers[0], n)
    for r in chosen:
        r["pareto_layer"] = 1
    li = 1
    seen = {(r["design"]["module_mm"], r["design"]["pinion_teeth"], r["design"]["face_width_mm"]) for r in chosen}
    while len(chosen) < n and li < len(layers):
        for r in sorted(layers[li], key=lambda r: -(r["_phys"] + r["_des"])):
            sig = (r["design"]["module_mm"], r["design"]["pinion_teeth"], r["design"]["face_width_mm"])
            if sig not in seen and len(chosen) < n:
                seen.add(sig)
                r["pareto_layer"] = li + 1
                chosen.append(r)
        li += 1
    return chosen


_CRITERIA = [   # (key, label, sort key: smaller = better), in tag priority order
    ("safety",   "Maximum structural safety margin", lambda r: -r["_margin"]),
    ("light",    "Lightweight optimized",            lambda r: r["mass_kg"]["total"]),
    ("compact",  "Compact footprint",                lambda r: r["_vol"]),
    ("smooth",   "Smoothest and easiest to make",    lambda r: -r["_des"]),
    ("balanced", "Balanced performance",             lambda r: -(r["_phys"] + r["_des"])),
]
_PHRASE = {"safety": "safety margin", "light": "weight", "compact": "footprint",
           "smooth": "smoothness / manufacturability", "balanced": "overall balance"}


def _ordinal(k: int) -> str:
    return {1: "Best", 2: "2nd best", 3: "3rd best"}.get(k, f"{k}th best")


def _label(chosen: list[dict]) -> None:
    """
    Labels are computed AFTER selection and only state facts about the chosen set:
    `strengths` = criteria where this variant is #1 among the chosen ones; `tag` = its
    first strength, or (if it wins none) its best rank, e.g. "2nd best weight".
    """
    ranks: dict[int, dict[str, int]] = {id(r): {} for r in chosen}
    for key, _label_text, sk in _CRITERIA:
        for rank, r in enumerate(sorted(chosen, key=sk), start=1):
            ranks[id(r)][key] = rank
    for r in chosen:
        rk = ranks[id(r)]
        r["strengths"] = [lbl for key, lbl, _ in _CRITERIA if rk[key] == 1]
        if r["strengths"]:
            r["tag"] = r["strengths"][0]
        else:
            key = min(rk, key=lambda k: (rk[k], [c[0] for c in _CRITERIA].index(k)))
            r["tag"] = f"{_ordinal(rk[key])} {_PHRASE[key]}"
        r["ranks_among_variants"] = rk


def _suggestions(rejected: Counter) -> list[str]:
    tips = []
    total = rejected.get("_total_candidates", 0) or 1
    if rejected["envelope"] / total > 0.4:
        tips.append("Most candidates do not fit the envelope: enlarge it, or lower the gear ratio.")
    if rejected["bending"] or rejected["contact"]:
        tips.append("Many candidates are too weak for the load: raise the input speed (lower torque), "
                    "choose a stronger gear material, or accept lower minimum safety factors.")
    if rejected["ratio"]:
        tips.append("Some candidates miss the requested ratio: allow a larger ratio tolerance.")
    return tips


def generate_variants(req: Requirements, n: int = 5) -> dict:
    if req.gear_material not in GEAR_MATERIALS:
        raise ValueError(f"unknown gear_material '{req.gear_material}', choose one of {list(GEAR_MATERIALS)}")
    if req.housing_material not in HOUSING_MATERIALS:
        raise ValueError(f"unknown housing_material '{req.housing_material}', choose one of {list(HOUSING_MATERIALS)}")
    if req.power_kw <= 0 or req.input_rpm <= 0:
        raise ValueError("power_kw and input_rpm must be positive")
    stages_for_ratio(req.gear_ratio)          # raises for unsupported ratios

    feasible, rejected = sweep(req)
    diagnostics = {
        "candidates_evaluated": rejected.pop("_total_candidates"),
        "feasible": len(feasible),
        "rejected_by": dict(rejected),
    }
    if not feasible:
        return {"variants": [], "diagnostics": diagnostics, "suggestions": _suggestions(rejected)}

    _score(feasible, req)
    chosen = _select(feasible, n)
    _label(chosen)
    chosen.sort(key=lambda r: (-r["_phys"], -r["_des"]))        # A = strongest physics ... last = strongest design

    variants = []
    for i, r in enumerate(chosen):
        env = r["envelope_needed_mm"]
        r["name"] = f"Variant {chr(ord('A') + i)}"
        r["housing_params"] = {                       # feed straight into POST /generate-housing
            "envelope_width": env["width"], "envelope_height": env["height"], "envelope_depth": env["depth"],
            "gear_ratio": r["kinematics"]["ratio_actual"], "wall_thickness": req.wall_thickness,
            "bore_diameter": r["shaft"]["output_diameter_mm"] + 2.0,
        }
        for k in ("_margin", "_vol", "_phys", "_des"):
            r.pop(k)
        variants.append(r)

    return {"variants": variants, "diagnostics": diagnostics, "suggestions": []}
