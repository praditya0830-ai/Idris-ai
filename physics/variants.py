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

from dataclasses import replace

from physics.gears import (
    Requirements, evaluate, stages_for_ratio,
    GEAR_MATERIALS, HOUSING_MATERIALS, STANDARD_MODULES,
)

try:                                   # ACE shaft + bearing check (tim solver); optional at import time
    from physics.variant_shafts import check_variant_shafts
except Exception:                      # pragma: no cover
    check_variant_shafts = None

try:                                   # ACE Goal Engine G1 (tim solver): thermal screen + engineer goals
    from physics import goal_engine
except Exception:                      # pragma: no cover
    goal_engine = None

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


# ---------------------------------------------------------------------------- repair loop
_MATERIAL_STRENGTH_ORDER = sorted(GEAR_MATERIALS, key=lambda k: GEAR_MATERIALS[k]["sigma_F_lim"])


def _envelope_for(req: Requirements, k: int = 5):
    """Envelope that fits the k smallest designs meeting strength + ratio (size unconstrained)."""
    big = replace(req, env_w=1e6, env_h=1e6, env_d=1e6)
    feas, _ = sweep(big)
    if not feas:
        return None
    vol = lambda r: r["envelope_needed_mm"]["width"] * r["envelope_needed_mm"]["height"] * r["envelope_needed_mm"]["depth"]
    best = sorted(feas, key=vol)[:k]
    return {d: max(r["envelope_needed_mm"][d] for r in best) for d in ("width", "height", "depth")}


def propose_repairs(req: Requirements, rejected: Counter, limit: int = 4) -> list[dict]:
    """
    Solver-proposed parameter changes when no design is feasible. Each proposal is re-checked
    by running the full sweep again, so `feasible_variants` is a computed count, not a guess.
    Safety factors are never lowered automatically (that would hide a weak design).
    """
    repairs = []

    def trial(change: dict, why: str):
        feas, _ = sweep(replace(req, **change))
        if feas:
            repairs.append({"change": change, "feasible_variants": len(feas), "reason": why})
        return len(feas)

    def env_change(env):
        return {"env_w": float(max(req.env_w, env["width"])), "env_h": float(max(req.env_h, env["height"])),
                "env_d": float(max(req.env_d, env["depth"]))}

    stronger = [k for k in _MATERIAL_STRENGTH_ORDER
                if GEAR_MATERIALS[k]["sigma_F_lim"] > GEAR_MATERIALS[req.gear_material]["sigma_F_lim"]]
    # 1) stronger gear material, same envelope
    for mat in stronger:
        if trial({"gear_material": mat}, f"stronger gear steel ({GEAR_MATERIALS[mat]['label']})"):
            break
    # 2) larger envelope with the current material (sized to give several variants)
    env = _envelope_for(req)
    if env:
        trial(env_change(env), f"enlarge the envelope to {env['width']}×{env['height']}×{env['depth']} mm")
    # 3) strongest material + the (smaller) envelope it needs
    if stronger:
        env2 = _envelope_for(replace(req, gear_material=stronger[-1]))
        if env2 and (not env or env2["width"] * env2["height"] * env2["depth"]
                     < env["width"] * env["height"] * env["depth"]):
            trial({"gear_material": stronger[-1], **env_change(env2)},
                  f"{GEAR_MATERIALS[stronger[-1]]['label']} and an envelope of "
                  f"{env2['width']}×{env2['height']}×{env2['depth']} mm")
    # 4) ratio tolerance, only if the ratio was a real blocker
    if rejected.get("ratio") and req.max_ratio_error < 0.05:
        trial({"max_ratio_error": 0.05}, "allow ±5 % ratio tolerance")
    # 5) thinner housing wall frees space (never below 3 mm)
    if rejected.get("envelope") and req.wall_thickness > 3.0:
        trial({"wall_thickness": 3.0}, "thinner housing wall (3 mm)")
    repairs.sort(key=lambda r: -r["feasible_variants"])
    names = {"env_w": "envelope_width", "env_h": "envelope_height", "env_d": "envelope_depth"}
    for r in repairs:                                   # API field names, as /generate-variants expects
        r["change"] = {names.get(k, k): v for k, v in r["change"].items()}
    return repairs[:limit]


def _repair_text(r: dict) -> str:
    ch = ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in r["change"].items())
    return f"Solver check: {r['reason']} ({ch}) gives {r['feasible_variants']} feasible designs."


def generate_variants(req: Requirements, n: int = 5, envelope_basis: str = "housing",
                      goals: dict | None = None, ambient_c: float = 25.0) -> dict:
    """envelope_basis "overall": the shaft ends (22 mm each side, along the depth) must also fit."""
    if envelope_basis not in ("housing", "overall"):
        raise ValueError("envelope_basis must be 'housing' or 'overall'")
    if envelope_basis == "overall":
        stick = 2 * (goal_engine.SHAFT_STICKOUT_MM if goal_engine else 22.0)
        if req.env_d - stick <= 0:
            raise ValueError(f"envelope_depth must be larger than {stick:g} mm when shaft ends are included")
        out = _generate(replace(req, env_d=req.env_d - stick), n, goals, ambient_c)
        for r in out.get("repairs", []):          # report depth in the engineer's (overall) terms
            if "envelope_depth" in r["change"]:
                r["change"]["envelope_depth"] = r["change"]["envelope_depth"] + stick
        out["suggestions"] = [s.replace("(shaft ends excluded)", "") for s in out.get("suggestions", [])]
        out["envelope_basis"] = "overall"
        out["envelope_note"] = (f"Depth checked as housing + 2 × {stick / 2:g} mm shaft ends ≤ {req.env_d:g} mm "
                                f"(housing depth ≤ {req.env_d - stick:g} mm).")
        return out
    out = _generate(req, n, goals, ambient_c)
    out["envelope_basis"] = "housing"
    return out


def _generate(req: Requirements, n: int, goals: dict | None, ambient_c: float) -> dict:
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
        repairs = propose_repairs(req, rejected)
        extra = [] if repairs else [
            "No single change inside the current design space (module ≤ 4 mm, 1–2 stages, spur gears) makes this "
            "feasible: the load is too high for this gearbox type — consider more stages, helical gears or a "
            "planetary layout (roadmap)."]
        return {"variants": [], "diagnostics": diagnostics, "repairs": repairs,
                "suggestions": [_repair_text(r) for r in repairs] + extra + _suggestions(rejected)}

    goal_note = None
    if goals and goal_engine is not None and len(feasible) <= 4000:
        # G1: screen EVERY feasible design against the engineer's goals, then pick variants from the
        # designs that meet them all (if any) — goals steer the search, they are not just a label.
        g = goals
        meeting = []
        for r in feasible:
            env = r["envelope_needed_mm"]
            r["housing_params"] = {"envelope_width": env["width"], "envelope_height": env["height"],
                                   "envelope_depth": env["depth"]}
            r["thermal_screen"] = goal_engine.thermal_screen(
                r, req.power_kw, req.input_rpm, req.housing_material, req.wall_thickness, ambient_c,
                float(g.get("oil_temp_max_c", goal_engine.OIL_LIMIT_C)), g.get("cooling_allowed"))
            r["goal_check"] = goal_engine.check_goals(r, g)
            if all(x["pass"] for x in r["goal_check"].values()):
                meeting.append(r)
        diagnostics["meeting_goals"] = len(meeting)
        if meeting:
            feasible = meeting
        else:
            goal_note = ("No feasible design meets all goals; showing the best designs on strength and size. "
                         "See goal_summary.blocking_goals and each variant's goal_check.")
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
        # ACE shaft + bearing check on the same geometry as /generate-assembly (additive fields)
        r["warnings"] = []
        if check_variant_shafts is not None:
            try:
                r["shafts"], r["warnings"] = check_variant_shafts(r, req.input_rpm)
            except Exception as e:      # never block variant generation on the extra check
                r["shafts"] = []
                r["warnings"] = [f"shaft/bearing check unavailable: {e}"]
        if goal_engine is not None:
            try:
                g = goals or {}
                r["assembly_bbox_mm"] = goal_engine.assembly_bbox(r)
                r["thermal_screen"] = goal_engine.thermal_screen(
                    r, req.power_kw, req.input_rpm, req.housing_material, req.wall_thickness, ambient_c,
                    float(g.get("oil_temp_max_c", goal_engine.OIL_LIMIT_C)), g.get("cooling_allowed"))
                if goals:
                    r["goal_check"] = goal_engine.check_goals(r, goals)
            except Exception as e:      # never block variant generation on the extra check
                r["warnings"].append(f"thermal screen unavailable: {e}")
        variants.append(r)

    out = {"variants": variants, "diagnostics": diagnostics, "suggestions": [goal_note] if goal_note else [],
           "repairs": []}
    if goals and goal_engine is not None:
        out["goal_summary"] = goal_engine.summarize(variants, goals)
    return out