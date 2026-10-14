"""Autonomous shaft + bearing design loop (CAD <-> CAE feedback).

Given gears on a shaft, ACE: computes mesh loads -> reactions, M and T diagrams ->
sizes every critical section for fatigue + yield -> picks standard bearings ->
checks slopes, deflections and critical speed -> enlarges the shaft and repeats
until every check passes. Every iteration is logged so the engineer can audit it.
"""
import math
from .shaft import (SHAFT_MATERIALS, size_section, check_section, beam_response, beam_at,
                    rayleigh_critical_speed, _interp)
from .bearings import select_bearing, next_bore, STANDARD_BORES

ENGINE = "ace-shaft-0.1"
SLOPE_LIMIT_DEEP_GROOVE = 0.001   # rad, Shigley Table 7-2 (lower bound of 0.001-0.003)
SLOPE_LIMIT_SPUR_GEAR = 0.0005    # rad, uncrowned spur gears
CRITICAL_SPEED_RATIO = 2.0        # first critical >= 2x operating speed


def gear_deflection_limit(module_mm):
    """Allowable transverse deflection at a spur gear (Shigley Table 7-2, converted to module)."""
    if module_mm >= 2.4:
        return 0.25
    if module_mm >= 1.3:
        return 0.13
    return 0.076


DEFAULTS = {
    "material": "4140_QT650", "finish": "machined", "reliability": 0.99, "design_factor": 1.5,
    "criterion": "goodman", "life_hours": 5000, "bearing_reliability": 0.99,
    "application_factor": 1.2, "bearing_series": ["60", "62", "63"],
    "torque_alternating_fraction": 0.0, "axial_load_N": 0.0, "max_iterations": 30,
}


def _validate(s):
    e = []
    if s["material"] not in SHAFT_MATERIALS:
        e.append(f"unknown material; available {list(SHAFT_MATERIALS)}")
    if s.get("rpm", 0) <= 0:
        e.append("rpm must be > 0")
    xA, xB, L = s.get("xA_mm"), s.get("xB_mm"), s.get("length_mm")
    if None in (xA, xB, L) or not (0 <= xA < xB <= L):
        e.append("need 0 <= xA_mm < xB_mm <= length_mm")
    if not s.get("gears"):
        e.append("need at least one gear")
    for g in s.get("gears", []):
        if not (xA < g.get("x_mm", -1) < xB):
            e.append("gears must sit between the bearings (overhung gears: roadmap)")
        if g.get("pitch_dia_mm", 0) <= 0:
            e.append("gear pitch_dia_mm must be > 0")
    if not 0 <= s["torque_alternating_fraction"] <= 1:
        e.append("torque_alternating_fraction must be 0-1")
    return e


def _gear_loads(s, mat):
    loads, tpts, pmass = [], [], []
    for g in s["gears"]:
        T = g["torque_Nm"] * 1e3                     # N*mm, + = into shaft (driven gear)
        Wt = 2 * abs(T) / g["pitch_dia_mm"]
        Wr = Wt * math.tan(math.radians(g.get("pressure_angle_deg", 20)))
        th = math.radians(g.get("mesh_angle_deg", 0))
        uy, uz = math.cos(th), math.sin(th)          # toward mating gear
        ty, tz = -math.sin(th), math.cos(th)         # rotation tangent (+x spin)
        sgn = 1 if T > 0 else -1
        m = g.get("mass_kg")
        if m is None:
            m = mat["density"] * math.pi / 4 * g["pitch_dia_mm"] ** 2 * g.get("face_width_mm", 20) * 1e-9 * 0.8
        g["_mass"] = m
        Fy = -Wr * uy + sgn * Wt * ty - m * 9.80665
        Fz = -Wr * uz + sgn * Wt * tz
        loads.append((g["x_mm"], Fy, Fz))
        tpts.append((g["x_mm"], T))
        pmass.append((g["x_mm"], m))
        g["_Wt"], g["_Wr"] = Wt, Wr
    tc = -sum(t for _, t in tpts)
    xc = s.get("coupling_x_mm")
    if abs(tc) > 1e-6:
        if xc is None:
            raise ValueError("torques don't balance: give coupling_x_mm for the input/output coupling")
        tpts.append((xc, tc))
    return loads, tpts, pmass


def _torque_at(tpts, x):
    return abs(sum(t for xi, t in tpts if xi < x))


def _layout(s, BA, BB, dj, ds):
    xA, xB, L = s["xA_mm"], s["xB_mm"], s["length_mm"]
    a_in, b_in = xA + BA / 2, xB - BB / 2
    return [(0.0, a_in, dj), (a_in, b_in, ds), (b_in, L, dj)], a_in, b_in


def design_shaft(spec):
    s = {**DEFAULTS, **{k: v for k, v in spec.items() if v is not None}}
    s["gears"] = [dict(g) for g in s.get("gears", [])]
    errs = _validate(s)
    if errs:
        return {"ok": False, "engine": ENGINE, "errors": errs}
    mat = SHAFT_MATERIALS[s["material"]]
    try:
        loads, tpts, pmass = _gear_loads(s, mat)
    except ValueError as ex:
        return {"ok": False, "engine": ENGINE, "errors": [str(ex)]}
    kw = dict(mat=mat, n_design=s["design_factor"], finish=s["finish"],
              reliability=s["reliability"], criterion=s["criterion"])
    alt = s["torque_alternating_fraction"]
    rpm = s["rpm"]
    dj, ds, BA, BB = 10.0, 12.0, 12.0, 12.0
    log, bearings, checks, sections = [], {}, {}, []
    status = "max_iterations"

    for it in range(1, s["max_iterations"] + 1):
        segs, a_in, b_in = _layout(s, BA, BB, dj, ds)
        beam = beam_response(segs, mat["E"], s["xA_mm"], s["xB_mm"], loads)
        FrA = math.hypot(beam["y"]["RA"], beam["z"]["RA"])
        FrB = math.hypot(beam["y"]["RB"], beam["z"]["RB"])

        # --- strength: size every critical section ---
        sections, need_j, need_s = [], 0.0, 0.0
        for g in s["gears"]:
            xg, half = g["x_mm"], g.get("face_width_mm", 20) / 2
            root = g["pitch_dia_mm"] - 2.5 * g.get("module_mm", g["pitch_dia_mm"] / 20)
            g["_integral"] = ds > 0.6 * root
            locs = [(xg, "shoulder_round" if g["_integral"] else "keyseat", "gear seat"),
                    (xg - half - 1, "shoulder_round", "gear shoulder L"),
                    (xg + half + 1, "shoulder_round", "gear shoulder R")]
            for x, notch, name in locs:
                M = beam_at(beam, "M", x)
                T = max(_torque_at(tpts, x - 0.5), _torque_at(tpts, x + 0.5))
                r = size_section(M, (1 - alt) * T, notch, Ta=alt * T, **kw)
                need_s = max(need_s, r["d_min_mm"])
                sections.append({"name": name, "x_mm": x, "M_Nmm": M, "T_Nmm": T, "dia_group": "seat", **r})
        for x, name in ((a_in, "bearing A shoulder"), (b_in, "bearing B shoulder")):
            M = beam_at(beam, "M", x)
            T = _torque_at(tpts, x)
            r = size_section(M, (1 - alt) * T, "shoulder_sharp", Ta=alt * T, **kw)
            need_j = max(need_j, r["d_min_mm"])
            sections.append({"name": name, "x_mm": x, "M_Nmm": M, "T_Nmm": T, "dia_group": "journal", **r})
        # torque-only stub at coupling (journal diameter)
        if s.get("coupling_x_mm") is not None:
            T = max(_torque_at(tpts, s["coupling_x_mm"] - 0.5), _torque_at(tpts, s["coupling_x_mm"] + 0.5))
            r = size_section(beam_at(beam, "M", s["coupling_x_mm"]), (1 - alt) * T, "keyseat", Ta=alt * T, **kw)
            need_j = max(need_j, r["d_min_mm"])
            sections.append({"name": "coupling keyseat", "x_mm": s["coupling_x_mm"],
                             "M_Nmm": beam_at(beam, "M", s["coupling_x_mm"]), "T_Nmm": T,
                             "dia_group": "journal", **r})

        # --- bearings ---
        new_dj = next_bore(max(dj, need_j))
        if new_dj is None:
            status = "no_standard_bearing"
            break
        bA = bB = None
        while new_dj is not None:
            bA = select_bearing(new_dj, FrA, s["axial_load_N"], rpm, s["life_hours"],
                                s["bearing_reliability"], s["application_factor"], s["bearing_series"])
            bB = select_bearing(new_dj, FrB, 0.0, rpm, s["life_hours"],
                                s["bearing_reliability"], s["application_factor"], s["bearing_series"])
            if bA and bB and bA["d_mm"] == bB["d_mm"]:
                break
            if bA and bB:  # different bores: both on the larger
                new_dj = max(bA["d_mm"], bB["d_mm"])
                continue
            nb = [b for b in STANDARD_BORES if b > new_dj]
            new_dj = nb[0] if nb else None
        if not (bA and bB):
            status = "no_standard_bearing"
            break
        new_dj = bA["d_mm"]
        Dmax = max(bA["D_mm"], bB["D_mm"])
        shoulder = new_dj + 0.25 * (Dmax - new_dj)       # bearing abutment height (approx.)
        new_ds = math.ceil(max(ds, need_s, shoulder))
        if new_dj != dj or new_ds != ds or bA["B_mm"] != BA or bB["B_mm"] != BB:
            log.append({"iter": it, "action": "resize for strength/bearings",
                        "journal_mm": new_dj, "seat_mm": new_ds,
                        "bearings": [bA["designation"], bB["designation"]]})
            dj, ds, BA, BB = new_dj, new_ds, bA["B_mm"], bB["B_mm"]
            continue
        bearings = {"A": {**bA, "x_mm": s["xA_mm"]}, "B": {**bB, "x_mm": s["xB_mm"]}}

        # --- stiffness + dynamics ---
        checks = {}
        for key, xb in (("bearing_A_slope", s["xA_mm"]), ("bearing_B_slope", s["xB_mm"])):
            v = beam_at(beam, "slope", xb)
            checks[key] = {"value": v, "limit": SLOPE_LIMIT_DEEP_GROOVE, "unit": "rad",
                           "pass": v <= SLOPE_LIMIT_DEEP_GROOVE}
        for i, g in enumerate(s["gears"]):
            v = beam_at(beam, "slope", g["x_mm"])
            checks[f"gear{i+1}_slope"] = {"value": v, "limit": SLOPE_LIMIT_SPUR_GEAR, "unit": "rad",
                                          "pass": v <= SLOPE_LIMIT_SPUR_GEAR}
            lim = gear_deflection_limit(g.get("module_mm", 2))
            v = beam_at(beam, "defl", g["x_mm"])
            checks[f"gear{i+1}_deflection"] = {"value": v, "limit": lim, "unit": "mm", "pass": v <= lim}
        nc = rayleigh_critical_speed(segs, mat["E"], mat["density"], s["xA_mm"], s["xB_mm"], pmass)
        checks["critical_speed"] = {"value": nc, "limit": CRITICAL_SPEED_RATIO * rpm, "unit": "rpm",
                                    "pass": nc >= CRITICAL_SPEED_RATIO * rpm}
        failed = [k for k, c in checks.items() if not c["pass"]]
        if not failed:
            status = "converged"
            break
        new_ds = math.ceil(ds * 1.1)
        if new_ds > 2.2 * dj:
            nb = [b for b in STANDARD_BORES if b > dj]
            if not nb:
                status = "stiffness_not_achievable"
                break
            dj = nb[0]
        log.append({"iter": it, "action": f"stiffen: failed {failed}", "journal_mm": dj, "seat_mm": new_ds})
        ds = new_ds

    if status != "converged":
        return {"ok": False, "engine": ENGINE, "status": status, "iterations": log,
                "errors": [{"no_standard_bearing": "Loads exceed the bearing catalog (bore <= 60 mm). "
                            "Reduce load/life or use roller bearings (roadmap).",
                            "stiffness_not_achievable": "Shaft cannot be stiff enough within catalog sizes.",
                            "max_iterations": "Design loop did not converge."}[status]]}

    # final verification of all sections at chosen diameters
    final = []
    for sec in sections:
        d = dj if sec["dia_group"] == "journal" else ds
        notch = sec["notch"]
        T = sec["T_Nmm"]
        r = check_section(d, sec["M_Nmm"], (1 - alt) * T, notch, mat, s["finish"], s["reliability"],
                          s["criterion"], Ta=alt * T)
        final.append({"name": sec["name"], "x_mm": sec["x_mm"], "M_Nm": sec["M_Nmm"] / 1e3,
                      "T_Nm": T / 1e3, "d_required_mm": sec["d_min_mm"], "d_used_mm": d,
                      "notch": notch, "Kf": r["Kf"], "Kfs": r["Kfs"], "Se_MPa": r["Se_MPa"],
                      "n_fatigue": r["n_fatigue"], "n_yield": r["n_yield"],
                      "pass": min(r["n_fatigue"], r["n_yield"]) >= s["design_factor"] - 1e-6})
    segs, _, _ = _layout(s, BA, BB, dj, ds)
    beam = beam_response(segs, mat["E"], s["xA_mm"], s["xB_mm"], loads)
    mass = sum(mat["density"] * math.pi / 4 * d * d * (x1 - x0) * 1e-9 for x0, x1, d in segs)
    warnings = []
    for i, g in enumerate(s["gears"]):
        if g.get("_integral"):
            warnings.append(f"gear{i+1}: seat Ø{ds} mm too large for a keyed hub -> make it integral "
                            "(teeth cut on the shaft)")
    for k, b in bearings.items():
        if b["ndm_mm_rpm"] > 5e5:
            warnings.append(f"bearing {k}: n*dm = {b['ndm_mm_rpm']:.0f} mm*rpm, check catalog speed rating / oil lubrication")
    if alt == 0:
        warnings.append("torque assumed steady; for reversing actuator duty set torque_alternating_fraction=1")

    L = s["length_mm"]
    xs = [L * i / 100 for i in range(101)]
    diagrams = {"x_mm": xs,
                "M_Nm": [beam_at(beam, "M", x) / 1e3 for x in xs],
                "T_Nm": [_torque_at(tpts, x) / 1e3 for x in xs],
                "deflection_mm": [beam_at(beam, "defl", x) for x in xs]}
    res = {
        "ok": True, "engine": ENGINE, "status": status,
        "method": ("Fatigue: DE-%s with Marin factors and Neuber notch sensitivity (Shigley ch.6-7); "
                   "deflection: Euler-Bernoulli numerical integration; critical speed: Rayleigh; "
                   "bearings: ISO 281 basic rating with Weibull reliability (Shigley ch.11)") % s["criterion"].capitalize(),
        "material": {"key": s["material"], **mat},
        "geometry": {"length_mm": L, "journal_dia_mm": dj, "seat_dia_mm": ds,
                     "segments": [{"x0_mm": a, "x1_mm": b, "dia_mm": d} for a, b, d in segs],
                     "mass_kg": mass},
        "gear_loads": [{"x_mm": g["x_mm"], "Wt_N": g["_Wt"], "Wr_N": g["_Wr"], "mass_kg": g["_mass"],
                        "integral": g.get("_integral", False)} for g in s["gears"]],
        "bearings": bearings, "sections": final, "checks": checks,
        "iterations": log, "warnings": warnings, "diagrams": diagrams,
    }
    return _round(res)


def _round(x):
    if isinstance(x, float):
        return x if math.isinf(x) else round(x, 6 if abs(x) < 0.01 else 3)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


def shafts_for_gear_stage(stage, **opts):
    """Design input + output shafts for one spur stage (output of the gear solver).

    stage keys accepted: input_torque_Nm|torque_in_Nm, input_rpm|rpm_in, ratio (or gear teeth),
    pitch_dia_pinion_mm, pitch_dia_gear_mm, face_width_mm, module_mm, efficiency (0.98).
    """
    T = stage.get("input_torque_Nm", stage.get("torque_in_Nm"))
    rpm = stage.get("input_rpm", stage.get("rpm_in"))
    dp, dg = stage.get("pitch_dia_pinion_mm"), stage.get("pitch_dia_gear_mm")
    F = stage.get("face_width_mm", 20)
    m = stage.get("module_mm", 2)
    eff = stage.get("efficiency", 0.98)
    if None in (T, rpm, dp, dg) or T <= 0 or rpm <= 0:
        return {"ok": False, "engine": ENGINE,
                "errors": ["need input_torque_Nm, input_rpm, pitch_dia_pinion_mm, pitch_dia_gear_mm"]}
    ratio = dg / dp
    stub, gap = 30.0, 20.0
    span = F + 2 * gap
    xA = stub + 8
    xB = xA + span
    L = xB + 12
    base = {"xA_mm": xA, "xB_mm": xB, "length_mm": L, "coupling_x_mm": stub / 2}
    base.update({k: v for k, v in opts.items() if v is not None})
    gear_common = {"x_mm": xA + span / 2, "face_width_mm": F, "module_mm": m}
    inp = design_shaft({**base, "rpm": rpm, "gears": [
        {**gear_common, "pitch_dia_mm": dp, "torque_Nm": -T, "mesh_angle_deg": 0}]})
    out = design_shaft({**base, "rpm": rpm / ratio, "gears": [
        {**gear_common, "pitch_dia_mm": dg, "torque_Nm": T * ratio * eff, "mesh_angle_deg": 180}]})
    return {"ok": inp.get("ok", False) and out.get("ok", False), "engine": ENGINE,
            "center_distance_mm": (dp + dg) / 2, "input_shaft": inp, "output_shaft": out}
