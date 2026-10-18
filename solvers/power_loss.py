"""Gear power-loss solvers (Tahap 2.1): Ohlendorf gear loss factor, Schlenk mesh friction, oil viscosity."""
from ace_core import SolverSpec, Benchmark, register
from physics.power_loss import gear_loss_factor, walther


def _run_hv(p):
    r = gear_loss_factor(p["module_mm"], p["z1"], p["z2"], p["x1"], p["x2"], p.get("da1_mm"), p.get("da2_mm"),
                         p.get("center_distance_mm"))
    warn = [] if r["valid"] else ["Ohlendorf H_V is only valid for 1 ≤ ε_α ≤ 2 and ε_1, ε_2 ≤ 1"]
    return {"outputs": r, "warnings": warn}


def _run_visc(p):
    nu = walther(p["nu40_mm2_s"], p["nu100_mm2_s"])(p["temp_c"])
    return {"outputs": {"nu_mm2_s": nu}}


register(SolverSpec(
    name="gear_loss_factor", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Ohlendorf gear loss factor H_V of a spur gear pair (ISO/TR 14179-2 / FVA).",
    method="H_V = π(u+1)/(z1·u·cosβ_b)·(1 − ε_α + ε_1² + ε_2²), addendum contact ratios from the tip circles",
    schema={"module_mm": {"type": "number", "exclusive_min": 0, "default": 2.0},
            "z1": {"type": "integer", "min": 8, "default": 17}, "z2": {"type": "integer", "min": 8, "default": 51},
            "x1": {"type": "number", "default": 0.0}, "x2": {"type": "number", "default": 0.0},
            "da1_mm": {"type": "number", "exclusive_min": 0}, "da2_mm": {"type": "number", "exclusive_min": 0},
            "center_distance_mm": {"type": "number", "exclusive_min": 0}},
    run=_run_hv,
    benchmarks=[Benchmark(
        "fzg-c-gear-ohlendorf",
        "FZG type-C test gear (z = 16/24, m = 4.5 mm, a = 91.5 mm, x = 0.1817/0.1715, da = 82.46/118.36 mm): "
        "H_V (Ohlendorf) = 0.1959, Mech. Sci. 6, 81–88 (2015), Table 2",
        {"module_mm": 4.5, "z1": 16, "z2": 24, "x1": 0.1817, "x2": 0.1715, "da1_mm": 82.46, "da2_mm": 118.36,
         "center_distance_mm": 91.5},
        {"outputs.H_V": 0.1959}, 0.01)],
    limitations=["spur gears (β = 0)", "valid for 1 ≤ ε_α ≤ 2 and ε_1, ε_2 ≤ 1"],
    example={"module_mm": 2.0, "z1": 17, "z2": 51},
))

register(SolverSpec(
    name="oil_viscosity", version="0.1.0", domain="tribology", fidelity="L0",
    summary="Kinematic viscosity of a mineral oil at temperature from ν40 and ν100 (ASTM D341 / Walther).",
    method="log log(ν + 0.7) = A − B log T (T in K)",
    schema={"nu40_mm2_s": {"type": "number", "exclusive_min": 0, "default": 68.0},
            "nu100_mm2_s": {"type": "number", "exclusive_min": 0, "default": 8.8},
            "temp_c": {"type": "number", "min": -20, "max": 200, "default": 80.0}},
    run=_run_visc,
    benchmarks=[Benchmark(
        "mineral-oil-min10-60c",
        "Mineral gear oil MIN-10: ν40 = 94.1, ν100 = 10.6 → ν60 = 38.4 mm²/s measured (Lubricants 8 (2020) 5, "
        "Table 2); D341 is within 3 % for mineral oils (synthetics deviate more)",
        {"nu40_mm2_s": 94.1, "nu100_mm2_s": 10.6, "temp_c": 60.0}, {"outputs.nu_mm2_s": 38.4}, 0.03)],
    limitations=["mineral oils; PAO/PAG deviate 3–7 % at 60 °C"],
    example={"nu40_mm2_s": 68.0, "nu100_mm2_s": 8.8, "temp_c": 90.0},
))

# The combined mesh loss (P_A · μ_mz · H_V) is not registered as its own solver: the framework requires a
# published numeric benchmark, and none was found for μ_mz (Schlenk). It is used inside /cfd (loss_model)
# and the variant thermal screen, always labelled as a lower bound.
