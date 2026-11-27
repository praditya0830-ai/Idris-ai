"""Thermal, lumped / 1D (audit 27 Nov, Batch 3): steady heat through a flat wall with convection on both sides.

This is the honest thermal answer IDRIS can give without a 3D field solver: one-dimensional steady conduction
through a plane wall, with a convection coefficient on each face. It is *not* 3D FEA and it says so in every
reply (`fidelity` and `not_checked`); what it does give is exact for the model it states:

    q = dT / R,   R = 1/(h_hot A) + L/(k A) + 1/(h_cold A)

from which the two surface temperatures follow. Benchmarked against that closed form (the same formula an
engineer checks by hand), and against pure conduction when the convection coefficients are very large.

h comes from the engineer, or from a named correlation the user picks (natural convection in air, forced air,
water) — each carries its source and its range of validity, and the reply repeats both.
"""
import math

from ace_core import SolverSpec, Benchmark, Check, register

# A small, sourced table of convection coefficients. Values are order-of-magnitude ranges from Incropera &
# DeWitt, Fundamentals of Heat and Mass Transfer, Table 1.1 (typical values of h, W/m2K). The solver uses the
# middle of the stated range and says so; an engineer's own h always wins.
H_TABLE = {
    "air_natural": (5.0, 25.0, "Incropera & DeWitt, Table 1.1: free convection, gases, 5-25 W/m2K"),
    "air_forced": (25.0, 250.0, "Incropera & DeWitt, Table 1.1: forced convection, gases, 25-250 W/m2K"),
    "water_forced": (100.0, 15000.0, "Incropera & DeWitt, Table 1.1: forced convection, water, 100-15000 W/m2K"),
    "oil_forced": (50.0, 3000.0, "Incropera & DeWitt, Table 1.1: forced convection, oils, 50-3000 W/m2K"),
}


def _h(name: str, own: float | None, which: str) -> tuple:
    """-> (h, source). The engineer's number wins; a named correlation gives the middle of its stated range."""
    if own is not None:
        if own <= 0:
            raise ValueError(f"{which}_W_m2K must be positive")
        return float(own), "you"
    key = str(name or "").strip().lower()
    if key not in H_TABLE:
        raise ValueError(f"give {which}_W_m2K, or name a correlation: {sorted(H_TABLE)}")
    lo, hi, src = H_TABLE[key]
    return math.sqrt(lo * hi), f"table: {key} (mid-range {math.sqrt(lo * hi):.1f} of {lo:g}-{hi:g}) — {src}"


def run(p):
    length = float(p["thickness_mm"]) / 1000.0          # m
    area = float(p["surface_area_m2"])                  # m2, from the kernel in the Studio
    if length <= 0 or area <= 0:
        raise ValueError("thickness_mm and surface_area_m2 must be positive")
    k = float(p["k_W_mK"])
    if k <= 0:
        raise ValueError("k_W_mK must be positive")
    t_hot = float(p["T_hot_C"])
    t_cold = float(p["T_cold_C"])
    d_t = t_hot - t_cold
    if abs(d_t) < 1e-12:
        raise ValueError("T_hot_C and T_cold_C are equal, so there is no heat flow to solve for")
    h_hot, src_hot = _h(p.get("correlation_hot"), p.get("h_hot_W_m2K"), "h_hot")
    h_cold, src_cold = _h(p.get("correlation_cold"), p.get("h_cold_W_m2K"), "h_cold")

    r_conv_hot = 1.0 / (h_hot * area)
    r_wall = length / (k * area)
    r_conv_cold = 1.0 / (h_cold * area)
    r_total = r_conv_hot + r_wall + r_conv_cold
    q = d_t / r_total                                   # W, positive from hot face to cold face
    flux = q / area
    t_hot_surface = t_hot - q * r_conv_hot
    t_cold_surface = t_hot_surface - q * r_wall

    checks = []
    if p.get("max_temperature_C") is not None:
        checks.append(Check("hot surface temperature", round(t_hot_surface, 3),
                            float(p["max_temperature_C"]), "degC", "max"))
    if p.get("max_heat_flow_W") is not None:
        checks.append(Check("heat flow", round(abs(q), 3), float(p["max_heat_flow_W"]), "W", "max"))

    outputs = {
        "model": "steady 1D conduction through a flat wall, convection on both faces",
        "fidelity": "lumped / 1D, not 3D FEA",
        "heat_flow_W": round(q, 4),
        "heat_flux_W_m2": round(flux, 3),
        "thickness_m": round(length, 6),
        "surface_area_m2": round(area, 6),
        "conductivity_W_mK": k,
        "h_hot_W_m2K": round(h_hot, 3), "h_hot_source": src_hot,
        "h_cold_W_m2K": round(h_cold, 3), "h_cold_source": src_cold,
        "resistance_K_W": {"hot_convection": round(r_conv_hot, 8), "wall": round(r_wall, 8),
                           "cold_convection": round(r_conv_cold, 8), "total": round(r_total, 8)},
        "temperature_C": {"hot_fluid": t_hot, "hot_surface": round(t_hot_surface, 3),
                          "cold_surface": round(t_cold_surface, 3), "cold_fluid": t_cold},
        "method": "q = dT / (1/(h_hot A) + L/(k A) + 1/(h_cold A)); surface temperatures from the same "
                  "series resistance (Incropera & DeWitt, chapters 3 and 1)",
        "warnings": [] if p.get("h_hot_W_m2K") or p.get("h_cold_W_m2K") else
                    ["h is the middle of a published range, not a computed coefficient: give your own h "
                     "when the flow is known (a correlation is an order-of-magnitude answer)."],
        "not_checked": ["3D temperature field, hot spots and gradients inside the part (needs the FEA worker)",
                        "radiation, contact resistance between layers, temperature-dependent k and h",
                        "transient warm-up and cool-down",
                        "a part shape other than a flat wall of the stated area and thickness"],
    }
    return {"outputs": outputs, "checks": checks,
            "warnings": outputs["warnings"] + ["This is a 1D lumped model: it is not a 3D thermal FEA."]}


register(SolverSpec(
    name="thermal_lumped", version="1.0.0", domain="thermal", fidelity="L0",
    summary="Steady heat flow through a flat wall: one-dimensional conduction plus convection on both faces, "
            "with the two surface temperatures and every resistance shown. Labelled 'lumped / 1D, not 3D FEA'.",
    method="Series thermal resistance: q = dT / (1/(h_hot A) + L/(k A) + 1/(h_cold A)) "
           "(Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, ch. 3); h from the engineer or the "
           "mid-range of a published table (ch. 1, Table 1.1)",
    schema={
        "thickness_mm": {"type": "number", "exclusive_min": 0, "max": 10000, "unit": "mm", "required": True},
        "surface_area_m2": {"type": "number", "exclusive_min": 0, "max": 1e6, "unit": "m2", "required": True},
        "k_W_mK": {"type": "number", "exclusive_min": 0, "max": 5000, "unit": "W/(m·K)", "required": True},
        "T_hot_C": {"type": "number", "min": -273.15, "max": 5000, "unit": "degC", "required": True},
        "T_cold_C": {"type": "number", "min": -273.15, "max": 5000, "unit": "degC", "required": True},
        "h_hot_W_m2K": {"type": "number", "exclusive_min": 0, "max": 1e6, "unit": "W/(m2·K)"},
        "h_cold_W_m2K": {"type": "number", "exclusive_min": 0, "max": 1e6, "unit": "W/(m2·K)"},
        "correlation_hot": {"type": "string", "enum": list(H_TABLE)},
        "correlation_cold": {"type": "string", "enum": list(H_TABLE)},
        "max_temperature_C": {"type": "number", "min": -273.15, "max": 5000, "unit": "degC"},
        "max_heat_flow_W": {"type": "number", "exclusive_min": 0, "unit": "W"},
    },
    run=run,
    benchmarks=[
        Benchmark(
            "flat-wall-series-resistance",
            "Flat wall, A = 1 m2, L = 10 mm, k = 50 W/mK, h = 10 W/m2K on both faces, 100 K across it: "
            "R = 0.1 + 0.0002 + 0.1 = 0.2002 K/W, so q = 100 / 0.2002 = 499.5005 W (closed form)",
            {"thickness_mm": 10.0, "surface_area_m2": 1.0, "k_W_mK": 50.0, "T_hot_C": 120.0,
             "T_cold_C": 20.0, "h_hot_W_m2K": 10.0, "h_cold_W_m2K": 10.0},
            {"outputs.heat_flow_W": 499.5005, "outputs.temperature_C.hot_surface": 70.05,
             "outputs.temperature_C.cold_surface": 69.95}, 0.001),
        Benchmark(
            "conduction-only-limit",
            "When convection is very large the wall is almost the only resistance: with h = 1e6 W/m2K the "
            "series resistance is 2e-6 + 2e-4 + 2e-6 = 2.02e-4 K/W, so q = 100 / 2.02e-4 = 495 049.5 W, "
            "and the faces sit within 0.5 K of the fluids (closed form)",
            {"thickness_mm": 10.0, "surface_area_m2": 1.0, "k_W_mK": 50.0, "T_hot_C": 120.0,
             "T_cold_C": 20.0, "h_hot_W_m2K": 1e6, "h_cold_W_m2K": 1e6},
            {"outputs.heat_flow_W": 495049.505}, 0.001),
    ],
    limitations=["steady state only; no transient warm-up",
                 "flat wall only: no fins, cylinders, or 3D shape effects",
                 "radiation, contact resistance and temperature-dependent properties are not modelled",
                 "h is an input or the middle of a published range, not a computed coefficient"],
    example={"thickness_mm": 10.0, "surface_area_m2": 0.05, "k_W_mK": 50.0, "T_hot_C": 120.0,
             "T_cold_C": 20.0, "correlation_hot": "air_natural", "correlation_cold": "air_natural"},
))
