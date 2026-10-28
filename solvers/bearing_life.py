"""ISO 281:2007 modified rating life solver (a_ISO)."""
from ace_core import SolverSpec, Benchmark, Check, register
from physics.bearing_life import a_iso, modified_life, E_C, nu1
from physics.bearings import DEEP_GROOVE


def _run_aiso(p):
    a, note = a_iso(p["kappa"], p["e_C"], p["Cu_kn"] * 1e3, p["P_kn"] * 1e3)
    return {"outputs": {"a_ISO": a, "note": note}, "warnings": [note] if note else []}


def _run_life(p):
    d, D, B, C, C0 = DEEP_GROOVE[p["designation"]]
    r = modified_life(d, D, C * 1e3, C0 * 1e3, p["radial_load_n"], p["rpm"], p["viscosity_mm2_s"], p["contamination"])
    checks = []
    if p.get("required_life_h") and r["L_nm_h"] is not None:
        checks.append(Check("modified rating life L_nm", r["L_nm_h"], p["required_life_h"], "h", "min"))
    warns = [r["note"]] if r["note"] else []
    return {"outputs": {"designation": p["designation"], **r}, "checks": checks, "warnings": warns}


register(SolverSpec(
    name="bearing_aiso", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="ISO 281:2007 life modification factor a_ISO for radial ball bearings.",
    method="a_ISO = 0.1·[1 − (2.5671 − A/κ^B)^0.83·(e_C·C_u/P)^(1/3)]^−9.3, three κ ranges, a_ISO ≤ 50",
    schema={"kappa": {"type": "number", "exclusive_min": 0, "default": 1.0},
            "e_C": {"type": "number", "min": 0, "max": 1, "default": 0.5},
            "Cu_kn": {"type": "number", "exclusive_min": 0, "default": 0.5},
            "P_kn": {"type": "number", "exclusive_min": 0, "default": 2.0}},
    run=_run_aiso,
    benchmarks=[Benchmark("iso281-6206-example",
                          "Bearing 6206: κ = 0.8, e_C = 0.2078, C_u = 0.509 kN, P = 9.7 kN → a_ISO = 0.331 "
                          "(Evolution of rolling bearing life rating through the standardization, MTM 2020, Table 3)",
                          {"kappa": 0.8, "e_C": 0.2078, "Cu_kn": 0.509, "P_kn": 9.7}, {"outputs.a_ISO": 0.331}, 0.01)],
    limitations=["radial ball bearings", "κ < 0.1 not covered; κ > 4 treated as 4"],
    example={"kappa": 1.0, "e_C": 0.5, "Cu_kn": 0.5, "P_kn": 2.0},
))

register(SolverSpec(
    name="bearing_life_iso281", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Modified rating life L_nm = a_ISO · L10 of a catalogue deep-groove ball bearing (90 % reliability).",
    method="ISO 281:2007: κ = ν/ν1, C_u = C0/22, e_C by cleanliness class, a_ISO (benchmarked in bearing_aiso)",
    schema={"designation": {"type": "string", "enum": sorted(DEEP_GROOVE), "default": "6206"},
            "radial_load_n": {"type": "number", "exclusive_min": 0, "default": 2000.0},
            "rpm": {"type": "number", "exclusive_min": 0, "default": 3000.0},
            "viscosity_mm2_s": {"type": "number", "exclusive_min": 0, "default": 20.0},
            "contamination": {"type": "string", "enum": list(E_C), "default": "normal_cleanliness"},
            "required_life_h": {"type": "number", "exclusive_min": 0}},
    run=_run_life,
    benchmarks=[Benchmark("iso281-6206-cu",
                          "C_u of 6206 from C0 = 11.2 kN: C0/22 = 0.509 kN, as used in the MTM 2020 ISO 281 example",
                          {"designation": "6206", "radial_load_n": 9700.0, "rpm": 3000.0, "viscosity_mm2_s": 20.0},
                          {"outputs.Cu_n": 509.0}, 0.002)],
    limitations=["catalogue C, C0 are approximate SKF values", "pure radial load (spur gears)",
                 "bearing temperature taken equal to the oil temperature"],
    example={"designation": "6206", "radial_load_n": 2000.0, "rpm": 3000.0, "viscosity_mm2_s": 20.0},
))
