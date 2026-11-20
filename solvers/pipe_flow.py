"""Steady single-phase pipe flow (v0.39 P2e): POST /solve/pipe_flow_darcy.

Darcy-Weisbach pressure loss with the friction factor from Poiseuille in the laminar regime and from the
Colebrook equation in the turbulent one, plus explicit minor losses (sum of K). This is not CFD: it is a
one-dimensional correlation with a stated range of validity, which is exactly what a line sizing needs.

Benchmarks: the exact Hagen-Poiseuille result for laminar flow, and Colebrook factors read off the Moody chart
(Moody 1944; the same values are tabulated in Crane TP-410).
"""
import math

from ace_core import SolverSpec, Benchmark, Check, register

FLUIDS = {
    # name: (density kg/m3, dynamic viscosity Pa s, source)
    "water_20c": (998.2, 1.002e-3, "IAPWS-IF97 / CRC Handbook, water at 20 degC"),
    "water_50c": (988.0, 5.47e-4, "IAPWS-IF97 / CRC Handbook, water at 50 degC"),
    "air_20c": (1.204, 1.825e-5, "CRC Handbook, dry air at 20 degC and 101.325 kPa"),
    "oil_iso_vg46": (860.0, 4.1e-2, "ISO 3448 VG46 at 40 degC: 46 mm2/s, 860 kg/m3 (ISO 11158 typical)"),
}
RE_LAMINAR = 2300.0
RE_TURBULENT = 4000.0


def colebrook(re: float, rel_rough: float, tol: float = 1e-12, iters: int = 100) -> tuple:
    """Friction factor from the Colebrook equation by fixed-point iteration on 1/sqrt(f)."""
    x = 1.0 / math.sqrt(0.02)                          # start from f = 0.02
    for k in range(1, iters + 1):
        x_new = -2.0 * math.log10(rel_rough / 3.7 + 2.51 * x / re)
        if abs(x_new - x) < tol:
            return 1.0 / x_new ** 2, k
        x = x_new
    return 1.0 / x ** 2, iters


def run(p):
    fluid = str(p.get("fluid") or "").lower()
    if fluid and fluid not in FLUIDS:
        raise ValueError(f"unknown fluid '{fluid}'; use one of {sorted(FLUIDS)} or give density and viscosity")
    rho = float(p.get("density_kg_m3") or (FLUIDS[fluid][0] if fluid else 0) or 0)
    mu = float(p.get("viscosity_Pa_s") or (FLUIDS[fluid][1] if fluid else 0) or 0)
    if rho <= 0 or mu <= 0:
        raise ValueError("give a fluid, or density_kg_m3 and viscosity_Pa_s")
    d = float(p["diameter_mm"]) / 1000.0
    length = float(p["length_mm"]) / 1000.0
    rough = float(p.get("roughness_mm") or 0.0) / 1000.0
    v = p.get("velocity_m_s")
    q = p.get("flow_m3_h")
    if v is None and q is None:
        raise ValueError("give velocity_m_s or flow_m3_h")
    area = math.pi * d ** 2 / 4.0
    if v is None:
        v = float(q) / 3600.0 / area
    v = float(v)
    if v <= 0:
        raise ValueError("the velocity must be positive")
    k_sum = float(p.get("minor_loss_K") or 0.0)
    re = rho * v * d / mu
    rel = rough / d
    if re < RE_LAMINAR:
        f, regime, method, iters = 64.0 / re, "laminar", "f = 64/Re (Hagen-Poiseuille)", 0
    elif re < RE_TURBULENT:
        f, regime, method, iters = 64.0 / re, "transitional", (
            "f = 64/Re shown for reference: in the transitional band 2300 < Re < 4000 neither the laminar nor "
            "the Colebrook factor is reliable, so a line should be sized outside it"), 0
    else:
        f, iters = colebrook(re, rel)
        regime, method = "turbulent", "Colebrook equation (1/sqrt f = -2 log10(eps/(3.7 D) + 2.51/(Re sqrt f)))"
    dyn = 0.5 * rho * v ** 2
    dp_friction = f * (length / d) * dyn
    dp_minor = k_sum * dyn
    dp = dp_friction + dp_minor
    warnings = []
    if regime == "transitional":
        warnings.append("Re is in the transitional band (2300-4000): the friction factor, and therefore the "
                        "pressure loss, is between the laminar and turbulent values; size the line outside it.")
    if rel > 0.05:
        warnings.append(f"relative roughness eps/D = {rel:.3f} is beyond the range the Colebrook chart is used "
                        f"for (and the fully-rough limit applies)")
    if re < RE_LAMINAR and rel > 0.0:
        warnings.append("roughness has no effect in the laminar regime; it is ignored, which is correct "
                        "for Hagen-Poiseuille")
    checks = [Check("pressure drop", dp, float(p.get("max_pressure_drop_Pa") or 1e12), "Pa", "max")]
    outputs = {
        "regime": regime, "method": method, "iterations": iters,
        "velocity_m_s": round(v, 6), "flow_m3_h": round(v * area * 3600.0, 4),
        "reynolds": round(re, 2), "relative_roughness": round(rel, 6),
        "friction_factor": round(f, 6),
        "pressure_drop_Pa": round(dp, 3), "pressure_drop_bar": round(dp / 1e5, 6),
        "friction_drop_Pa": round(dp_friction, 3), "minor_drop_Pa": round(dp_minor, 3),
        "head_loss_m": round(dp / (rho * 9.80665), 4),
        "fluid": {"density_kg_m3": rho, "viscosity_Pa_s": mu,
                  "source": FLUIDS[fluid][2] if fluid else "user values"},
        "not_checked": ["3D flow field, separation, elbows as geometry (only a K value is used)",
                        "compressibility, heat transfer, two-phase flow, cavitation",
                        "pipe wall roughness of a real used pipe (only the value you give)"],
    }
    return {"outputs": outputs, "checks": checks, "warnings": warnings}


def _water(**kw):
    return dict({"fluid": "water_20c", "diameter_mm": 50.0, "length_mm": 1000.0}, **kw)


register(SolverSpec(
    name="pipe_flow_darcy", version="1.0.0", domain="cfd", fidelity="L1",
    summary="Steady single-phase pipe flow: Reynolds number, friction factor (Poiseuille or Colebrook), "
            "Darcy-Weisbach pressure drop with minor losses, and the regime it was computed in.",
    method="Darcy-Weisbach; f = 64/Re below Re 2300; Colebrook solved by fixed-point iteration above Re 4000; "
           "minor losses from a sum of K",
    schema={
        "fluid": {"type": "string", "enum": list(FLUIDS)},
        "density_kg_m3": {"type": "number", "exclusive_min": 0, "max": 25000},
        "viscosity_Pa_s": {"type": "number", "exclusive_min": 0, "max": 100},
        "diameter_mm": {"type": "number", "exclusive_min": 0, "max": 5000, "required": True},
        "length_mm": {"type": "number", "exclusive_min": 0, "max": 1e7, "required": True},
        "roughness_mm": {"type": "number", "min": 0, "max": 50, "default": 0.0},
        "velocity_m_s": {"type": "number", "exclusive_min": 0, "max": 200},
        "flow_m3_h": {"type": "number", "exclusive_min": 0, "max": 1e6},
        "minor_loss_K": {"type": "number", "min": 0, "max": 1000, "default": 0.0},
        "max_pressure_drop_Pa": {"type": "number", "exclusive_min": 0},
    },
    run=run,
    benchmarks=[
        Benchmark("hagen-poiseuille-laminar",
                  "Laminar flow in a round pipe: dp = 32 mu L V / D^2 (Hagen-Poiseuille, exact; the same result "
                  "as f = 64/Re in Darcy-Weisbach). Water 20 degC, D 10 mm, L 1 m, V 0.1 m/s (Re = 1000)",
                  {"fluid": "water_20c", "diameter_mm": 10.0, "length_mm": 1000.0, "velocity_m_s": 0.1},
                  {"outputs.pressure_drop_Pa": 32.06}, 0.005),
        Benchmark("colebrook-moody-re1e5-eps0.001",
                  "Colebrook friction factor read off the Moody chart (Moody 1944, Trans. ASME 66; tabulated in "
                  "Crane TP-410): Re = 1e5, eps/D = 0.001 -> f = 0.0223",
                  _water(roughness_mm=0.05, velocity_m_s=2.0),
                  {"outputs.friction_factor": 0.0223}, 0.02),
        Benchmark("colebrook-moody-re1e6-eps0.0001",
                  "Colebrook friction factor read off the Moody chart (Moody 1944; Crane TP-410): Re = 1e6, "
                  "eps/D = 0.0001 -> f = 0.0134",
                  _water(roughness_mm=0.005, velocity_m_s=20.0),
                  {"outputs.friction_factor": 0.0134}, 0.02),
    ],
    limitations=[
        "one-dimensional: fully developed flow in a straight round pipe; the flow field is not computed",
        "single phase, incompressible, steady, isothermal; no cavitation, no two-phase, no compressibility",
        "minor losses only through a sum of K values you supply (no fittings database here)",
        "transitional band 2300 < Re < 4000: the factor is shown for reference and flagged, not recommended",
        "general 3D CFD (ducts, separation, wakes, heat transfer) is not in IDRIS",
    ],
    example=_water(roughness_mm=0.05, velocity_m_s=2.0, minor_loss_K=2.5),
))
