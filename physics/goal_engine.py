"""ACE Goal Engine, phase G1 — engineer goals checked on every generated variant.

The variant generator (gears.py / variants.py) only knows strength, ratio and envelope. This module
adds, per variant, a fast L1 thermal screen (lumped network, same correlations as the 3D solver;
agrees with thermal_housing_3d within 0.1 % on the S1 demo variant when fed the same oil-side h)
and the cheapest computed cooling measure that meets the oil limit. With `goals` it also reports,
per goal, which variants meet it and which goal blocks — so Idris can say "nothing meets X; the
closest is Y; to get there change Z" instead of a bare FAIL.

Goals (all optional):
    oil_temp_max_c     default 90 °C (mineral gear oil, continuous)
    mass_max_kg        total gearbox mass (gears + housing [+ fins])
    sf_bending_min / sf_contact_min   extra margin above the generator's hard limits
    cooling_allowed    subset of ["natural", "fan", "fins", "efficiency", "oil_cooler"]
                       (default: everything except oil_cooler — it needs a pump circuit)
"""
import math
from .thermal_lumped import oil_temp_lumped, fin_layout, fin_mass_kg, FIN_THICKNESS_MM, FIN_MIN_GAP_MM

CONDUCTIVITY = {"Al7075-T6": 130, "Al6061-T6": 167, "Al2024-T3": 121, "A356-T6": 151, "Ti-6Al-4V": 6.7,
                "Steel-1045": 50, "Steel-4140": 42, "Steel4140": 42, "CastIron-GG25": 48, "Mg-AZ91D": 72}
DENSITY = {"Al7075-T6": 2810, "Al6061-T6": 2700, "Steel4140": 7850, "Steel-4140": 7850}
OIL_LIMIT_C = 90.0
COOLING_KINDS = ["natural", "fan", "fins", "efficiency", "oil_cooler"]
DEFAULT_ALLOWED = ["natural", "fan", "fins", "efficiency"]
SHAFT_STICKOUT_MM = 22.0          # same as assembly3d.SHAFT_STICKOUT
EFF_ASSUMPTION = ("η 99 %/stage is an assumption (ground, profile-corrected gears + low-loss bearings), not computed: "
                  "a gear/bearing loss model (ISO/TR 14179) is not in the solver yet")


def h_oil(variant, input_rpm, oil_vg=68):
    v = (variant.get("loads") or {}).get("pitch_line_velocity_ms")
    if not isinstance(v, (int, float)) or v <= 0:
        d = variant["design"]
        v = math.pi * d["module_mm"] * d["pinion_teeth"] * input_rpm / 60000
    return 200.0 * min(2.0, max(0.5, (v / 5.0) ** 0.5)) * (68 / oil_vg) ** 0.2


def _fin_sets(D_mm):
    out = []
    for gap in (12.0, 8.0):
        per_side = int(D_mm // (gap + FIN_THICKNESS_MM))
        if per_side >= 2:
            for hf in (15.0, 25.0, 35.0):
                out.append({"count": 2 * per_side, "height_mm": hf})
    return out


def thermal_screen(variant, power_kw, input_rpm, housing_material, wall_mm, ambient_c=25.0,
                   limit_c=OIL_LIMIT_C, allowed=None):
    allowed = allowed or DEFAULT_ALLOWED
    hp = variant["housing_params"]
    W, H, D = hp["envelope_width"], hp["envelope_height"], hp["envelope_depth"]
    k = CONDUCTIVITY.get(housing_material, 130)
    rho = DENSITY.get(housing_material, 2810)
    eta = float(variant["kinematics"].get("efficiency", 0.97))
    stages = int(variant["design"].get("stages", 1))
    Q = power_kw * 1000 * (1 - eta)
    Q99 = power_kw * 1000 * (1 - 0.99 ** stages)
    ho = h_oil(variant, input_rpm)

    def run(q=Q, **kw):
        return oil_temp_lumped(q, W, H, D, wall_mm, k, ho, ambient_c, **kw)

    nat = run()
    res = {"method": "L1 lumped screen (confirm with /cfd)", "heat_w": round(Q, 1),
           "oil_temp_natural_c": round(nat["oil_temp_c"], 1), "area_outer_m2": round(nat["area_outer_m2"], 4),
           "limit_c": limit_c, "pass_natural": nat["oil_temp_c"] <= limit_c, "measure": None}
    if res["pass_natural"]:
        res["measure"] = {"label": "No cooling needed (natural convection)", "change": {},
                          "oil_temp_c": res["oil_temp_natural_c"], "added_mass_kg": 0.0, "basis": "computed",
                          "assumption": None}
        return res
    # cheapest first: fan → efficiency+fan → fins (+fan) → efficiency+fan+fins → oil cooler
    tries = []
    if "fan" in allowed:
        for sp in (5.0, 10.0):
            tries.append((f"Fan {sp:g} m/s", {"cooling": "forced", "air_speed_ms": sp}, Q,
                          dict(cooling="forced", air_speed=sp), None))
    if "efficiency" in allowed and Q99 < Q:
        tries.append(("Gears/bearings 99 %/stage", {"efficiency_per_stage": 0.99}, Q99, {}, None))
        if "fan" in allowed:
            for sp in (5.0, 10.0):
                tries.append((f"99 %/stage + fan {sp:g} m/s",
                              {"efficiency_per_stage": 0.99, "cooling": "forced", "air_speed_ms": sp}, Q99,
                              dict(cooling="forced", air_speed=sp), None))
    if "fins" in allowed:
        for fs in sorted(_fin_sets(D), key=lambda f: f["count"] * f["height_mm"]):
            fl = fin_layout(fs, H, D)
            if fl["gap_mm"] < FIN_MIN_GAP_MM:
                continue
            tries.append((f"Fins {fs['count']} × {fs['height_mm']:g} mm", {"fins": fs}, Q, dict(fins=fs), fl))
            if "fan" in allowed:
                for sp in (5.0, 10.0):
                    tries.append((f"Fins {fs['count']} × {fs['height_mm']:g} mm + fan {sp:g} m/s",
                                  {"fins": fs, "cooling": "forced", "air_speed_ms": sp}, Q,
                                  dict(fins=fs, cooling="forced", air_speed=sp), fl))
            if "efficiency" in allowed and "fan" in allowed:
                tries.append((f"99 %/stage + fan 5 m/s + fins {fs['count']} × {fs['height_mm']:g} mm",
                              {"efficiency_per_stage": 0.99, "fins": fs, "cooling": "forced", "air_speed_ms": 5.0},
                              Q99, dict(fins=fs, cooling="forced", air_speed=5.0), fl))
    # power the bare housing can carry continuously at the limit (natural convection, bisection)
    lo_p, hi_p = 0.0, power_kw
    for _ in range(30):
        mid = 0.5 * (lo_p + hi_p)
        if run(mid * 1000 * (1 - eta))["oil_temp_c"] <= limit_c:
            lo_p = mid
        else:
            hi_p = mid
    res["max_power_natural_kw"] = round(lo_p, 2)
    best = {"label": "Natural convection", "change": {}, "oil_temp_c": res["oil_temp_natural_c"],
            "added_mass_kg": 0.0}
    passing = []
    closest_computed = best
    for label, change, q, kw, fl in tries:
        t = run(q, **kw)["oil_temp_c"]
        m = fin_mass_kg(fl, rho)
        assumed = "efficiency_per_stage" in change
        cand = {"label": label + (" (assumption)" if assumed else ""), "change": change, "oil_temp_c": round(t, 1),
                "added_mass_kg": round(m, 3), "basis": "assumption" if assumed else "computed",
                "assumption": EFF_ASSUMPTION if assumed else None}
        if t <= limit_c:
            passing.append(cand)
        if not assumed and t < closest_computed["oil_temp_c"]:
            closest_computed = cand
        if best is None or t < best["oil_temp_c"]:
            best = cand
    # QA Q-020: a computed measure wins over one that relies on the efficiency assumption
    comp = sorted([c for c in passing if c["basis"] == "computed"], key=lambda c: c["added_mass_kg"])
    assm = sorted([c for c in passing if c["basis"] != "computed"], key=lambda c: c["added_mass_kg"])
    if comp:
        res["measure"] = comp[0]
    elif assm:
        res["measure"] = assm[0]
        res["best_without_assumption"] = closest_computed
    if (res["measure"] is None or res["measure"]["basis"] != "computed") and "oil_cooler" in allowed:
        dT_t = (limit_c - 5) - ambient_c
        G = Q / max(nat["oil_temp_c"] - ambient_c, 1e-9)        # W/K at the natural state (screen)
        cw = 50.0 * math.ceil(max(0.0, Q - G * dT_t) / 50.0)
        t = run(oil_cooler_w=cw)["oil_temp_c"]
        res["measure"] = {"label": f"Oil cooler {cw:.0f} W", "change": {"oil_cooler_w": cw},
                          "oil_temp_c": round(t, 1), "added_mass_kg": 0.0, "basis": "computed", "assumption": None}
    if res["measure"] is None:
        res["closest"] = best
    return res


def check_goals(variant, goals):
    """Per-goal value / limit / pass for one variant (variant must already carry thermal_screen)."""
    out = {}
    ts = variant.get("thermal_screen") or {}
    meas = ts.get("measure")
    t_oil = meas["oil_temp_c"] if meas else (ts.get("closest") or {}).get("oil_temp_c", ts.get("oil_temp_natural_c"))
    computed = meas is not None and meas.get("basis", "computed") == "computed"
    out["oil_temp_max_c"] = {"value": t_oil, "limit": ts.get("limit_c"), "pass": computed,
                             "with": meas["label"] if meas else None,
                             # QA Q-020: an assumption alone never makes a goal PASS
                             "passes_only_with_assumption": meas is not None and not computed}
    if goals.get("mass_max_kg") is not None:
        m = variant["mass_kg"]["total"] + (meas["added_mass_kg"] if meas else 0.0)
        out["mass_max_kg"] = {"value": round(m, 3), "limit": goals["mass_max_kg"], "pass": m <= goals["mass_max_kg"]}
    st = variant.get("stress") or {}
    for g, key in (("sf_bending_min", "sf_bending"), ("sf_contact_min", "sf_contact")):
        if goals.get(g) is not None and key in st:
            out[g] = {"value": st[key], "limit": goals[g], "pass": st[key] >= goals[g]}
    return out


def summarize(variants, goals):
    blocking = {}
    met = []
    for v in variants:
        gc = v.get("goal_check") or {}
        if gc and all(x["pass"] for x in gc.values()):
            met.append(v["name"])
        for g, x in gc.items():
            if not x["pass"]:
                blocking[g] = blocking.get(g, 0) + 1
    return {"goals": goals, "variants_meeting_all": met,
            "blocking_goals": {g: f"{n} of {len(variants)} variants miss it" for g, n in blocking.items()}}


def assembly_bbox(variant):
    hp = variant["housing_params"]
    return {"min": [0.0, 0.0, -SHAFT_STICKOUT_MM],
            "max": [float(hp["envelope_width"]), float(hp["envelope_height"]),
                    float(hp["envelope_depth"]) + SHAFT_STICKOUT_MM]}


def assembly_warnings(variant):
    """QA Q-011: the closed box of /generate-assembly cannot be assembled when gears are larger than the bores."""
    d = variant["design"]
    tip = d["module_mm"] * (d["gear_teeth"] + 2)
    bore = float(variant["housing_params"].get("bore_diameter") or 0)
    if bore and tip > bore:
        return [f"Assembly: the housing is a closed box; the Ø{tip:.0f} mm gear cannot pass the Ø{bore:.0f} mm bore. "
                "Fix: split the housing on the shaft plane (bolted halves) or add a bolted side cover larger than "
                f"Ø{tip + 4:.0f} mm. Not yet in the 3D model (planned, G2)."]
    return []
