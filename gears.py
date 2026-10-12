"""
Idris Core — CAE layer v0: spur gear train physics (closed-form formulas)
==========================================================================

Explicit, deterministic engineering formulas — no AI guessing. This is the
"CAE AI" first stage from the architecture: closed-form standard formulas
first (fast, transparent, verifiable), detailed FEA later.

What is modelled (spur gears, 20 deg pressure angle, full depth):
  * Torque / speed through 1 or 2 stages (2-stage = "reverted" layout, input
    and output shafts coaxial, both stages identical)
  * Tooth BENDING stress  : Lewis equation with AGMA-style load factors
  * Tooth CONTACT stress  : Hertz equation (ZE * ZH form, as in ISO 6336/AGMA)
  * Safety factors vs. material endurance limits
  * Contact ratio (smoothness / noise proxy)
  * Required housing envelope, masses, shaft diameter

IMPORTANT LIMITS (this is a screening tool, not a certification):
  * Material strengths below are REPRESENTATIVE catalogue-level values and
    must be replaced with certified data (ISO 6336 / AGMA 2001 tables or
    supplier data) before any real design release.
  * Load factors (KA, Km), life factor (=1), reliability factor (=1) are
    fixed simple assumptions, documented below.
  * The housing is NOT structurally analysed here (needs FEA — next layer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ALPHA = math.radians(20.0)          # pressure angle
ZE = 189.8                          # sqrt(MPa), steel-on-steel elastic coefficient
ZH = math.sqrt(2.0 / (math.sin(ALPHA) * math.cos(ALPHA)))   # ~2.495 for 20 deg
ETA_STAGE = 0.98                    # efficiency per spur mesh (typical 0.97-0.99)
K_A = 1.25                          # application factor (moderate shock, e.g. rotor loads)
K_M = 1.30                          # load distribution factor (face-width misalignment)
Q_V = 7                             # AGMA transmission accuracy number
CLEARANCE = 6.0                     # mm, gear-to-wall / bearing space
STAGE_GAP = 6.0                     # mm, axial gap between the two stages
TAU_SHAFT = 50.0                    # MPa, allowable shear stress for shaft sizing (conservative)

# Representative values (see module docstring warning)
GEAR_MATERIALS = {
    "42CrMo4":   {"label": "42CrMo4 quenched & tempered",      "sigma_F_lim": 340.0, "sigma_H_lim": 1050.0, "density": 7850.0},
    "17CrNiMo6": {"label": "17CrNiMo6 carburized & ground",    "sigma_F_lim": 460.0, "sigma_H_lim": 1500.0, "density": 7850.0},
    "AMS6265":   {"label": "AMS 6265 (9310) carburized, aero", "sigma_F_lim": 500.0, "sigma_H_lim": 1650.0, "density": 7850.0},
}
HOUSING_MATERIALS = {
    "Al7075-T6":  {"label": "Aluminium 7075-T6", "density": 2810.0},
    "Al6061-T6":  {"label": "Aluminium 6061-T6", "density": 2700.0},
    "Steel4140":  {"label": "Steel 4140",        "density": 7850.0},
}
STANDARD_MODULES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)   # ISO 54 first-choice series


@dataclass
class Requirements:
    power_kw: float
    input_rpm: float
    gear_ratio: float
    env_w: float
    env_h: float
    env_d: float
    wall_thickness: float = 4.0
    gear_material: str = "17CrNiMo6"
    housing_material: str = "Al7075-T6"
    min_sf_bending: float = 1.4     # typical industrial minimum (ISO 6336 practice 1.25-1.6)
    min_sf_contact: float = 1.1     # typical industrial minimum (1.0-1.3)
    max_ratio_error: float = 0.03   # 3 %


def stages_for_ratio(ratio: float) -> int:
    """Single mesh up to ~6.5:1, two meshes (reverted) up to ~42:1. Beyond that: unsupported in this POC."""
    if ratio <= 1.0:
        raise ValueError("gear_ratio must be greater than 1")
    if ratio <= 6.5:
        return 1
    if ratio <= 42.0:
        return 2
    raise ValueError("gear_ratio above 42:1 needs 3+ stages, not supported yet")


# Lewis form factor Y for 20 deg full-depth teeth (tabulated textbook values, e.g. Shigley
# "Mechanical Engineering Design", Table 14-2). Used as: sigma_F = Ft * K / (b * m * Y).
_LEWIS_Y_TABLE = [
    (12, 0.245), (13, 0.261), (14, 0.277), (15, 0.290), (16, 0.296), (17, 0.303), (18, 0.309),
    (19, 0.314), (20, 0.322), (21, 0.328), (22, 0.331), (24, 0.337), (26, 0.346), (28, 0.353),
    (30, 0.359), (34, 0.371), (38, 0.384), (43, 0.397), (50, 0.409), (60, 0.422), (75, 0.435),
    (100, 0.447), (150, 0.460), (300, 0.472), (400, 0.480),
]


def lewis_form_factor(z: int) -> float:
    """Lewis form factor Y(z), linear interpolation in the tabulated values (clamped at the ends)."""
    tbl = _LEWIS_Y_TABLE
    if z <= tbl[0][0]:
        return tbl[0][1]
    if z >= tbl[-1][0]:
        return tbl[-1][1]
    for (z0, y0), (z1, y1) in zip(tbl, tbl[1:]):
        if z0 <= z <= z1:
            return y0 + (y1 - y0) * (z - z0) / (z1 - z0)
    raise AssertionError("unreachable")


def dynamic_factor(v_ms: float) -> float:
    """AGMA dynamic factor Kv for transmission accuracy Qv, v = pitch-line velocity in m/s."""
    B = 0.25 * (12 - Q_V) ** (2.0 / 3.0)
    A = 50 + 56 * (1 - B)
    return ((A + math.sqrt(200.0 * v_ms)) / A) ** B


def contact_ratio(m: float, zp: int, zg: int) -> float:
    """Transverse contact ratio of a standard external spur pair (addendum = m)."""
    rp, rg = m * zp / 2.0, m * zg / 2.0
    rap, rag = rp + m, rg + m
    rbp, rbg = rp * math.cos(ALPHA), rg * math.cos(ALPHA)
    a = rp + rg
    path = math.sqrt(rap**2 - rbp**2) + math.sqrt(rag**2 - rbg**2) - a * math.sin(ALPHA)
    return path / (math.pi * m * math.cos(ALPHA))


def shaft_diameter_mm(torque_nm: float) -> float:
    """Minimum solid-shaft diameter from torsion only, rounded up to the next 5 mm."""
    d = (16.0 * torque_nm * 1000.0 / (math.pi * TAU_SHAFT)) ** (1.0 / 3.0)
    return float(math.ceil(d / 5.0) * 5.0)


def evaluate(req: Requirements, m: float, zp: int, b: float) -> dict:
    """Evaluate one gear design (module m [mm], pinion teeth zp, face width b [mm])."""
    gm = GEAR_MATERIALS[req.gear_material]
    hm = HOUSING_MATERIALS[req.housing_material]
    stages = stages_for_ratio(req.gear_ratio)

    # --- ratio / geometry -------------------------------------------------
    u_target = req.gear_ratio ** (1.0 / stages)
    zg = max(int(round(zp * u_target)), zp + 1)
    u = zg / zp
    ratio_actual = u ** stages
    ratio_error = ratio_actual / req.gear_ratio - 1.0

    dp, dg = m * zp, m * zg                       # pitch diameters
    dap, dag = m * (zp + 2), m * (zg + 2)         # tip diameters
    a = m * (zp + zg) / 2.0                       # center distance

    # --- loads (worst-loaded mesh is the LAST stage: highest torque) --------
    t_in = 9549.0 * req.power_kw / req.input_rpm                      # N*m at input shaft
    eta_total = ETA_STAGE ** stages
    t_pinion = t_in * (u * ETA_STAGE) ** (stages - 1)                 # N*m on last-stage pinion
    n_pinion = req.input_rpm / u ** (stages - 1)                      # rpm of last-stage pinion
    ft = 2000.0 * t_pinion / dp                                       # N, tangential force at pitch circle
    v = math.pi * dp * n_pinion / 60000.0                             # m/s pitch-line velocity
    kv = dynamic_factor(v)
    k = K_A * kv * K_M

    # --- stresses -----------------------------------------------------------
    yp, yg = lewis_form_factor(zp), lewis_form_factor(zg)
    sigma_f_p = ft * k / (b * m * yp)
    sigma_f_g = ft * k / (b * m * yg)
    sigma_f = max(sigma_f_p, sigma_f_g)
    sigma_h = ZE * ZH * math.sqrt(k * ft / (b * dp) * (u + 1.0) / u)
    sf_bend = gm["sigma_F_lim"] / sigma_f
    sf_cont = gm["sigma_H_lim"] / sigma_h
    eps = contact_ratio(m, zp, zg)

    # --- housing envelope the gear train needs -------------------------------
    t = req.wall_thickness
    if stages == 1:
        w_need = a + dap / 2.0 + dag / 2.0 + 2 * CLEARANCE + 2 * t
        h_need = max(dap, dag) + 2 * CLEARANCE + 2 * t
        d_need = b + 2 * CLEARANCE + 2 * t
    else:   # reverted: input/output coaxial (axis A), intermediate shaft (axis B) at distance a
        w_need = a + dag + 2 * CLEARANCE + 2 * t
        h_need = dag + 2 * CLEARANCE + 2 * t
        d_need = 2 * b + STAGE_GAP + 2 * CLEARANCE + 2 * t
    w_need, h_need, d_need = math.ceil(w_need), math.ceil(h_need), math.ceil(d_need)
    fits = w_need <= req.env_w and h_need <= req.env_h and d_need <= req.env_d

    # --- masses ---------------------------------------------------------------
    gear_vol_mm3 = stages * math.pi / 4.0 * b * 0.85 * (dp**2 + dg**2)    # 0.85: teeth/root vs pitch cylinder
    gears_kg = gear_vol_mm3 * 1e-9 * gm["density"]
    housing_vol_mm3 = (w_need * h_need * d_need
                       - (w_need - 2 * t) * (h_need - 2 * t) * (d_need - 2 * t))
    housing_kg = housing_vol_mm3 * 1e-9 * hm["density"]

    t_out = t_in * ratio_actual * eta_total
    d_shaft_out = shaft_diameter_mm(t_out)

    reasons = []
    if abs(ratio_error) > req.max_ratio_error:
        reasons.append("ratio")
    if not fits:
        reasons.append("envelope")
    if sf_bend < req.min_sf_bending:
        reasons.append("bending")
    if sf_cont < req.min_sf_contact:
        reasons.append("contact")

    return {
        "design": {"module_mm": m, "pinion_teeth": zp, "gear_teeth": zg, "face_width_mm": b,
                   "stages": stages, "center_distance_mm": round(a, 3)},
        "kinematics": {"ratio_target": req.gear_ratio, "ratio_actual": round(ratio_actual, 4),
                       "ratio_error_pct": round(ratio_error * 100, 3),
                       "input_torque_nm": round(t_in, 3), "output_torque_nm": round(t_out, 3),
                       "output_rpm": round(req.input_rpm / ratio_actual, 2), "efficiency": round(eta_total, 4)},
        "loads": {"critical_stage": stages, "tangential_force_n": round(ft, 2),
                  "pitch_line_velocity_ms": round(v, 3), "K_total": round(k, 4), "K_v": round(kv, 4)},
        "stress": {"bending_mpa": round(sigma_f, 2), "contact_mpa": round(sigma_h, 2),
                   "bending_limit_mpa": gm["sigma_F_lim"], "contact_limit_mpa": gm["sigma_H_lim"],
                   "sf_bending": round(sf_bend, 3), "sf_contact": round(sf_cont, 3),
                   "contact_ratio": round(eps, 3)},
        "envelope_needed_mm": {"width": w_need, "height": h_need, "depth": d_need},
        "mass_kg": {"gears": round(gears_kg, 3), "housing": round(housing_kg, 3),
                    "total": round(gears_kg + housing_kg, 3)},
        "shaft": {"output_diameter_mm": d_shaft_out},
        "feasible": not reasons,
        "violations": reasons,
    }
