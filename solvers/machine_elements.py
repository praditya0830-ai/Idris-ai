"""Machine-element solvers registered in the ACE framework (L0 / L1)."""
from ace_core import SolverSpec, Benchmark, Check, register
from physics.shaft import (SHAFT_MATERIALS, SURFACE, check_section, endurance_limit,
                           fatigue_kf, von_mises_stresses, fatigue_safety, yield_safety)
from physics.bearings import required_C10, life_hours, equivalent_load
from physics.shaft_design import shafts_for_gear_stage

IN, LBF, KPSI = 25.4, 4.448222, 6.894757
CRITERIA = ["goodman", "gerber", "asme", "soderberg"]


# ---------------------------------------------------------------- shaft section (L0)
def _run_section(p):
    Kf = fatigue_kf(p["Kt"], p["notch_radius_mm"], p["Sut_MPa"]) if p.get("Kf") is None else p["Kf"]
    Kfs = fatigue_kf(p["Kts"], p["notch_radius_mm"], p["Sut_MPa"], torsion=True) if p.get("Kfs") is None else p["Kfs"]
    se = endurance_limit(p["Sut_MPa"], p["diameter_mm"], p["finish"], p["reliability"])
    Se = p.get("Se_MPa") or se["Se"]
    d = p["diameter_mm"]
    sa, sm = von_mises_stresses(d, p["Ma_Nm"] * 1e3, p["Mm_Nm"] * 1e3,
                                p["Ta_Nm"] * 1e3, p["Tm_Nm"] * 1e3, Kf, Kfs)
    n = fatigue_safety(sa, sm, Se, p["Sut_MPa"], p["Sy_MPa"], p["criterion"])
    ny = yield_safety(d, (p["Ma_Nm"] + p["Mm_Nm"]) * 1e3, (p["Ta_Nm"] + p["Tm_Nm"]) * 1e3,
                      Kf, Kfs, p["Sy_MPa"])
    return {"outputs": {"Kf": Kf, "Kfs": Kfs, "Se_MPa": Se, "marin": se,
                        "sigma_a_MPa": sa, "sigma_m_MPa": sm, "n_fatigue": n, "n_yield": ny},
            "checks": [Check("fatigue safety factor", n, p["design_factor"], "-", "min", p["criterion"]),
                       Check("yield safety factor", ny, p["design_factor"], "-", "min")]}


SECTION_SCHEMA = {
    "diameter_mm": {"type": "number", "exclusive_min": 0, "max": 2000, "required": True, "unit": "mm"},
    "Ma_Nm": {"type": "number", "min": 0, "default": 0.0, "unit": "N*m", "desc": "alternating bending moment"},
    "Mm_Nm": {"type": "number", "min": 0, "default": 0.0, "unit": "N*m", "desc": "mean bending moment"},
    "Ta_Nm": {"type": "number", "min": 0, "default": 0.0, "unit": "N*m", "desc": "alternating torque"},
    "Tm_Nm": {"type": "number", "min": 0, "default": 0.0, "unit": "N*m", "desc": "mean torque"},
    "Sut_MPa": {"type": "number", "exclusive_min": 0, "max": 3000, "required": True, "unit": "MPa"},
    "Sy_MPa": {"type": "number", "exclusive_min": 0, "max": 3000, "required": True, "unit": "MPa"},
    "Kt": {"type": "number", "min": 1, "max": 10, "default": 1.0},
    "Kts": {"type": "number", "min": 1, "max": 10, "default": 1.0},
    "notch_radius_mm": {"type": "number", "min": 0, "max": 500, "default": 0.0, "unit": "mm"},
    "Kf": {"type": "number", "min": 1, "max": 10, "desc": "override computed Kf"},
    "Kfs": {"type": "number", "min": 1, "max": 10, "desc": "override computed Kfs"},
    "Se_MPa": {"type": "number", "exclusive_min": 0, "max": 2000, "desc": "override endurance limit"},
    "finish": {"type": "string", "enum": list(SURFACE), "default": "machined"},
    "reliability": {"type": "number", "min": 0.5, "max": 0.999999, "default": 0.99},
    "criterion": {"type": "string", "enum": CRITERIA, "default": "goodman"},
    "design_factor": {"type": "number", "min": 1, "max": 10, "default": 1.5},
}

register(SolverSpec(
    name="shaft_section", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Fatigue + yield safety factor of one shaft cross-section under bending and torsion.",
    method="DE-Goodman/Gerber/ASME/Soderberg with Marin factors and Neuber notch sensitivity (Shigley 10e ch.6-7)",
    schema=SECTION_SCHEMA, run=_run_section,
    benchmarks=[
        Benchmark("shigley-7-1-goodman", "Shigley 10e, Example 7-1 (shoulder, Goodman n=1.62)",
                  {"diameter_mm": 1.1 * IN, "Ma_Nm": 1260 * LBF * IN / 1e3, "Tm_Nm": 1100 * LBF * IN / 1e3,
                   "Sut_MPa": 105 * KPSI, "Sy_MPa": 82 * KPSI, "Kf": 1.58, "Kfs": 1.39,
                   "Se_MPa": 29.3 * KPSI, "criterion": "goodman"},
                  {"outputs.n_fatigue": 1.62}, 0.015),
        Benchmark("shigley-7-1-gerber", "Shigley 10e, Example 7-1 (Gerber n=1.87)",
                  {"diameter_mm": 1.1 * IN, "Ma_Nm": 1260 * LBF * IN / 1e3, "Tm_Nm": 1100 * LBF * IN / 1e3,
                   "Sut_MPa": 105 * KPSI, "Sy_MPa": 82 * KPSI, "Kf": 1.58, "Kfs": 1.39,
                   "Se_MPa": 29.3 * KPSI, "criterion": "gerber"},
                  {"outputs.n_fatigue": 1.87}, 0.015),
        Benchmark("shigley-7-1-marin", "Shigley 10e, Example 7-1 (Se = 29.3 kpsi, Kf = 1.58)",
                  {"diameter_mm": 1.1 * IN, "Sut_MPa": 105 * KPSI, "Sy_MPa": 82 * KPSI,
                   "Kt": 1.68, "notch_radius_mm": 0.11 * IN, "Ma_Nm": 100},
                  {"outputs.Se_MPa": 29.3 * KPSI, "outputs.Kf": 1.58}, 0.015),
    ],
    limitations=["Kt must come from a chart or FEA for the real geometry",
                 "infinite-life design; finite-life (S-N) not yet included"],
    example={"diameter_mm": 25, "Ma_Nm": 30, "Tm_Nm": 20, "Sut_MPa": 760, "Sy_MPa": 655,
             "Kt": 1.7, "Kts": 1.5, "notch_radius_mm": 2.5},
))


# ---------------------------------------------------------------- bearing rating (L0)
def _run_bearing(p):
    eq = equivalent_load(p["Fr_N"], p["Fa_N"], p["C0_kN"] * 1e3) if p.get("C0_kN") else \
        {"P": p["Fr_N"], "X": 1.0, "Y": 0.0, "e": None}
    C = required_C10(eq["P"], p["rpm"], p["life_hours"], p["reliability"], p["application_factor"], p["ball"])
    out = {"P_N": eq["P"], "X": eq["X"], "Y": eq["Y"], "C10_required_kN": C / 1e3}
    checks = []
    if p.get("C10_catalog_kN"):
        L = life_hours(p["C10_catalog_kN"] * 1e3, eq["P"], p["rpm"], p["reliability"],
                       p["application_factor"], p["ball"])
        out["life_hours"] = L
        checks.append(Check("bearing life", L, p["life_hours"], "h", "min"))
    return {"outputs": out, "checks": checks}


register(SolverSpec(
    name="bearing_rating", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Required catalog rating C10 / predicted life of a rolling bearing.",
    method="ISO 281 basic rating life with 3-parameter Weibull reliability (Shigley 10e ch.11)",
    schema={
        "Fr_N": {"type": "number", "min": 0, "required": True, "unit": "N"},
        "Fa_N": {"type": "number", "min": 0, "default": 0.0, "unit": "N"},
        "C0_kN": {"type": "number", "exclusive_min": 0, "desc": "static rating, needed if Fa > 0"},
        "rpm": {"type": "number", "exclusive_min": 0, "max": 200000, "required": True},
        "life_hours": {"type": "number", "exclusive_min": 0, "required": True, "unit": "h"},
        "reliability": {"type": "number", "min": 0.5, "max": 0.9999, "default": 0.9},
        "application_factor": {"type": "number", "min": 1, "max": 5, "default": 1.0},
        "ball": {"type": "boolean", "default": True},
        "C10_catalog_kN": {"type": "number", "exclusive_min": 0, "desc": "to compute life"},
    },
    run=_run_bearing,
    benchmarks=[Benchmark("shigley-11-1", "Shigley 10e, Example 11-1 (C10 = 16.0 kN)",
                          {"Fr_N": 2000, "rpm": 1725, "life_hours": 5000, "reliability": 0.9},
                          {"outputs.C10_required_kN": 16.0}, 0.02)],
    limitations=["basic rating only: no a_ISO lubrication/contamination life modification"],
    example={"Fr_N": 1500, "rpm": 6000, "life_hours": 5000, "reliability": 0.99},
))


# ---------------------------------------------------------------- shaft + bearing system (L1)
def _run_system(p):
    st = {k: p[k] for k in ("input_torque_Nm", "input_rpm", "pitch_dia_pinion_mm",
                            "pitch_dia_gear_mm", "face_width_mm", "module_mm", "efficiency")}
    opts = {k: p[k] for k in ("material", "life_hours", "design_factor", "criterion",
                              "torque_alternating_fraction", "bearing_reliability")}
    r = shafts_for_gear_stage(st, **opts)
    checks, warnings, errors, its = [], [], [], []
    for side in ("input_shaft", "output_shaft"):
        s = r[side]
        if not s.get("ok"):
            errors += [f"{side}: {e}" for e in s.get("errors", [])]
            continue
        for sec in s["sections"]:
            checks.append(Check(f"{side} · {sec['name']} fatigue", sec["n_fatigue"],
                                p["design_factor"], "-", "min"))
        for k, c in s["checks"].items():
            kind = "min" if k == "critical_speed" else "max"
            checks.append(Check(f"{side} · {k}", c["value"], c["limit"], c["unit"], kind))
        for k, b in s["bearings"].items():
            checks.append(Check(f"{side} · bearing {k} {b['designation']} life",
                                b["life_h_at_reliability"], p["life_hours"], "h", "min"))
        warnings += [f"{side}: {w}" for w in s["warnings"]]
        its += [{"shaft": side, **i} for i in s["iterations"]]
    return {"ok": r["ok"], "outputs": r, "checks": checks, "warnings": warnings,
            "errors": errors, "iterations": its}


register(SolverSpec(
    name="shaft_bearing", version="0.1.0", domain="machine_elements", fidelity="L1",
    summary=("Designs input and output shafts plus deep-groove bearings for one spur gear stage, "
             "iterating until strength, stiffness, critical speed and bearing life all pass."),
    method=("Section fatigue (Shigley ch.6-7) + Euler-Bernoulli deflection + Rayleigh critical speed "
            "+ ISO 281 bearing life, closed CAD<->CAE loop"),
    schema={
        "input_torque_Nm": {"type": "number", "exclusive_min": 0, "max": 1e5, "required": True},
        "input_rpm": {"type": "number", "exclusive_min": 0, "max": 1e5, "required": True},
        "pitch_dia_pinion_mm": {"type": "number", "exclusive_min": 0, "max": 2000, "required": True},
        "pitch_dia_gear_mm": {"type": "number", "exclusive_min": 0, "max": 5000, "required": True},
        "face_width_mm": {"type": "number", "exclusive_min": 0, "max": 500, "default": 20},
        "module_mm": {"type": "number", "exclusive_min": 0, "max": 50, "default": 2},
        "efficiency": {"type": "number", "min": 0.5, "max": 1, "default": 0.98},
        "material": {"type": "string", "enum": list(SHAFT_MATERIALS), "default": "4140_QT650"},
        "life_hours": {"type": "number", "exclusive_min": 0, "max": 2e5, "default": 5000},
        "design_factor": {"type": "number", "min": 1, "max": 5, "default": 1.5},
        "criterion": {"type": "string", "enum": CRITERIA, "default": "goodman"},
        "torque_alternating_fraction": {"type": "number", "min": 0, "max": 1, "default": 0.0,
                                        "desc": "1 = fully reversing torque (actuators)"},
        "bearing_reliability": {"type": "number", "min": 0.5, "max": 0.9999, "default": 0.99},
    },
    run=_run_system,
    benchmarks=[Benchmark(
        "regression-vtol-20Nm", "ACE regression baseline v0.1 (self-consistency, not a published value)",
        {"input_torque_Nm": 20, "input_rpm": 6000, "pitch_dia_pinion_mm": 28,
         "pitch_dia_gear_mm": 112, "face_width_mm": 24.5, "module_mm": 1.75},
        {"outputs.input_shaft.geometry.journal_dia_mm": 25,
         "outputs.output_shaft.geometry.journal_dia_mm": 17}, 0.0)],
    limitations=["gears between bearings only (no overhung pinion yet)",
                 "deep-groove ball bearings only; catalog values approximate (verify with SKF/NSK)",
                 "Kt values are first-iteration estimates (Shigley Table 7-1)"],
    example={"input_torque_Nm": 20, "input_rpm": 6000, "pitch_dia_pinion_mm": 28,
             "pitch_dia_gear_mm": 112, "face_width_mm": 24.5, "module_mm": 1.75,
             "torque_alternating_fraction": 1.0},
))
