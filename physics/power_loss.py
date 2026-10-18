"""Gearbox power losses (Tahap 2.1) — load-dependent part per ISO/TR 14179-2 style models.

Units: mm, N, N·m, rpm, W, °C. Screening level.

What is computed (each with its source)
- Gear mesh, load-dependent:  P_VZP = P_A · μ_mz · H_V                          (ISO/TR 14179-2; FVA)
    H_V  (Ohlendorf)  = π (u + 1) / (z1 · u · cos β_b) · (1 − ε_α + ε_1² + ε_2²)
                        valid for 1 ≤ ε_α ≤ 2 and ε_1, ε_2 ≤ 1
    μ_mz (Schlenk)    = 0.048 · (F_bt/b / (v_ΣC · ρ_redC))^0.2 · η_oil^−0.05 · Ra^0.25 · X_L
                        F_bt/b in N/mm, v_ΣC = 2 v_t sin α_wt in m/s, ρ_redC in mm (pitch point),
                        η_oil dynamic viscosity in mPa·s at oil temperature, Ra = (Ra1 + Ra2)/2 in µm,
                        X_L = 1.0 for mineral oil.
    Sources for both formulas (identical in each): FVA-Workbench KnowledgeBase "Power loss";
    Paschold et al., Forsch. Ingenieurwes. 86 (2022), doi 10.1007/s10010-021-00532-4, eq. 6, 7, 11.
    Benchmark: H_V of the FZG type-C gear = 0.1959 (Wimmer / Mech. Sci. 6, 81–88, 2015, Table 2).
- Rolling bearings, load-dependent: M = μ · F · d / 2 with μ = 0.0015 for deep-groove ball bearings
    (constant-friction-coefficient method; NTN catalogue Table 10.1: 1.0–1.5 × 10⁻³; the upper value
    is used).
- Radial lip seals: P_VD = 7.69 · 10⁻⁶ · d_sh² · n_sh  (W; ISO/TR 14179-2, as quoted by Paschold 2022 eq. 14).
- Oil viscosity at temperature: ASTM D341 (Walther) through ν40 and ν100 of a typical mineral gear oil.
    Benchmark: mineral oil MIN-10 (ν40 = 94.1, ν100 = 10.6 mm²/s) → ν60 = 38.4 mm²/s measured
    (Lubricants 8 (2020) 5, Table 2); D341 gives 37.6 (−2.2 %).

What is NOT computed yet (stated in every result)
- No-load losses: gear churning / squeezing in the oil bath and the speed-dependent drag of the
  bearings. The published models (Mauz, Changenet, ISO/TR 14179-1/-2) disagree by large factors and
  no numeric example was found to benchmark one. The computed total is therefore a LOWER BOUND of
  the real losses; at high pitch-line speed the no-load part can be of the same order as the mesh loss.
"""
import math

ALPHA = math.radians(20.0)
MU_BEARING_DGBB = 0.0015
SEAL_K = 7.69e-6
# typical mineral gear oils, VI ≈ 95–97 (e.g. Mobilgear 600 XP data sheet): ν40, ν100 in mm²/s
OIL_NU = {68: (68.0, 8.8), 100: (100.0, 11.2), 150: (150.0, 14.7), 220: (220.0, 19.0), 320: (320.0, 24.1)}
RHO15 = 0.88                 # g/cm³, mineral gear oil
RA_DEFAULT_UM = 0.5          # ground / finished gear flanks (state as an assumption)


def walther(nu40, nu100):
    """ASTM D341: log log (ν + 0.7) = A − B log T. Returns ν(T °C) in mm²/s."""
    T1, T2 = 313.15, 373.15
    f = lambda n: math.log10(math.log10(n + 0.7))
    B = (f(nu40) - f(nu100)) / (math.log10(T2) - math.log10(T1))
    A = f(nu40) + B * math.log10(T1)
    return lambda T: 10 ** (10 ** (A - B * math.log10(T + 273.15))) - 0.7


def oil_props(vg, T_c):
    nu40, nu100 = OIL_NU.get(int(vg), (float(vg), None))
    if nu100 is None:
        raise ValueError(f"no viscosity data for ISO VG {vg}")
    nu = walther(nu40, nu100)(T_c)
    rho = RHO15 * (1 - 0.00065 * (T_c - 15.0))
    return {"nu_mm2_s": nu, "rho_g_cm3": rho, "eta_mpas": nu * rho}


def contact_ratios(m, z1, z2, x1=0.0, x2=0.0):
    """Transverse contact ratio and its addendum parts (spur, standard centre distance if x1 + x2 = 0)."""
    d1, d2 = m * z1, m * z2
    db1, db2 = d1 * math.cos(ALPHA), d2 * math.cos(ALPHA)
    da1, da2 = d1 + 2 * m * (1 + x1), d2 + 2 * m * (1 + x2)
    a0 = (d1 + d2) / 2
    a = a0 + m * (x1 + x2)
    awt = math.acos(a0 * math.cos(ALPHA) / a)
    pb = math.pi * m * math.cos(ALPHA)
    e1 = (0.5 * math.sqrt(da1 ** 2 - db1 ** 2) - 0.5 * db1 * math.tan(awt)) / pb
    e2 = (0.5 * math.sqrt(da2 ** 2 - db2 ** 2) - 0.5 * db2 * math.tan(awt)) / pb
    return e1, e2, e1 + e2, awt


def gear_loss_factor(m, z1, z2, x1=0.0, x2=0.0, da1=None, da2=None, a=None):
    """Ohlendorf H_V (spur). Optional explicit tip diameters and centre distance (as tabulated)."""
    if da1 is None and da2 is None:
        e1, e2, ea, awt = contact_ratios(m, z1, z2, x1, x2)
    else:
        d1, d2 = m * z1, m * z2
        db1, db2 = d1 * math.cos(ALPHA), d2 * math.cos(ALPHA)
        a = a or (d1 + d2) / 2 + m * (x1 + x2)
        awt = math.acos((d1 + d2) / 2 * math.cos(ALPHA) / a)
        pb = math.pi * m * math.cos(ALPHA)
        e1 = (0.5 * math.sqrt(da1 ** 2 - db1 ** 2) - 0.5 * db1 * math.tan(awt)) / pb
        e2 = (0.5 * math.sqrt(da2 ** 2 - db2 ** 2) - 0.5 * db2 * math.tan(awt)) / pb
        ea = e1 + e2
    u = z2 / z1
    hv = math.pi * (u + 1) / (z1 * u) * (1 - ea + e1 ** 2 + e2 ** 2)
    valid = 1.0 <= ea <= 2.0 and e1 <= 1.0 and e2 <= 1.0
    return {"H_V": hv, "eps_alpha": ea, "eps_1": e1, "eps_2": e2, "alpha_wt_deg": math.degrees(awt),
            "valid": valid}


def schlenk_mu(Ft_n, b_mm, v_t, d1, d2, awt, eta_mpas, ra_um, XL=1.0):
    Fbt_b = Ft_n / math.cos(ALPHA) / b_mm
    v_sum = 2 * v_t * math.sin(awt)
    r1, r2 = 0.5 * d1 * math.sin(awt), 0.5 * d2 * math.sin(awt)
    rho_red = r1 * r2 / (r1 + r2)
    mu = 0.048 * (Fbt_b / (v_sum * rho_red)) ** 0.2 * eta_mpas ** -0.05 * ra_um ** 0.25 * XL
    return mu, {"Fbt_per_b_n_mm": Fbt_b, "v_sum_c_ms": v_sum, "rho_red_c_mm": rho_red}


def mesh_loss(m, z1, z2, b, T_pinion_nm, n_pinion_rpm, oil, ra_um=RA_DEFAULT_UM):
    d1, d2 = m * z1, m * z2
    g = gear_loss_factor(m, z1, z2)
    Ft = 2000.0 * T_pinion_nm / d1
    v_t = math.pi * d1 * n_pinion_rpm / 60000.0
    mu, aux = schlenk_mu(Ft, b, v_t, d1, d2, math.radians(g["alpha_wt_deg"]), oil["eta_mpas"], ra_um)
    P_A = Ft * v_t
    return {"P_in_w": P_A, "loss_w": P_A * mu * g["H_V"], "mu_mz": mu, **g, "Ft_n": Ft, "v_t_ms": v_t, **aux}


def bearing_loss(F_n, d_mm, n_rpm, mu=MU_BEARING_DGBB):
    M = mu * F_n * d_mm / 2.0                        # N·mm
    return M * 2 * math.pi * n_rpm / 60.0 / 1000.0    # W


def seal_loss(d_mm, n_rpm):
    return SEAL_K * d_mm ** 2 * n_rpm


def gearbox_losses(L, input_rpm, oil_vg, T_oil_c, bearings=None, seals=None, ra_um=RA_DEFAULT_UM):
    """Load-dependent losses of the layout L (physics.housing.build_layout) at oil temperature T_oil_c.

    bearings: [(shaft index, k (0/1), bore d_mm)] from the shaft plan; default: the shaft diameter.
    seals:    [(shaft diameter mm, rpm)]; default: input and output shafts.
    """
    from .housing import per_shaft_reactions
    oil = oil_props(oil_vg, T_oil_c)
    m = L["module_mm"]
    zp, zg = L["teeth"]
    b = L["face_width_mm"]
    meshes = []
    for s in range(L["stages"]):
        sh = L["shafts"][s]
        meshes.append({"stage": s + 1, **mesh_loss(m, zp, zg, b, sh["torque"], input_rpm * sh["rpm_factor"], oil, ra_um)})
    reac = per_shaft_reactions(L)
    brg = []
    for si, s in enumerate(L["shafts"]):
        n = input_rpm * s["rpm_factor"]
        for k in (0, 1):
            d = next((bd for i, kk, bd in (bearings or []) if i == si and kk == k), s["d"])
            F = float(math.hypot(*reac[si][k]))
            brg.append({"shaft": s["name"], "support": k, "radial_n": F, "bore_mm": d, "rpm": n,
                        "loss_w": bearing_loss(F, d, n)})
    if seals is None:
        seals = [(L["shafts"][0]["d"], input_rpm), (L["shafts"][-1]["d"], input_rpm * L["shafts"][-1]["rpm_factor"])]
    sl = [{"shaft_mm": d, "rpm": n, "loss_w": seal_loss(d, n)} for d, n in seals]
    P_in = 2 * math.pi * input_rpm / 60.0 * L["shafts"][0]["torque"]
    tot_mesh = sum(x["loss_w"] for x in meshes)
    tot_brg = sum(x["loss_w"] for x in brg)
    tot_seal = sum(x["loss_w"] for x in sl)
    total = tot_mesh + tot_brg + tot_seal
    return {"model": "ISO/TR 14179-2 load-dependent losses (Schlenk μ_mz, Ohlendorf H_V; bearings μ = 0.0015; "
                     "seals ISO/TR 14179-2)",
            "oil_temp_c": T_oil_c, "oil": {"iso_vg": oil_vg, **{k: round(v, 3) for k, v in oil.items()}},
            "ra_um": ra_um, "input_power_w": P_in,
            "mesh": meshes, "bearings": brg, "seals": sl,
            "mesh_w": tot_mesh, "bearings_w": tot_brg, "seals_w": tot_seal, "total_load_dependent_w": total,
            "efficiency_load_dependent": 1 - total / P_in if P_in > 0 else None,
            "no_load_losses": "not computed (gear churning and bearing drag): the total is a LOWER BOUND",
            "assumptions": [f"flank roughness Ra = {ra_um:g} µm (ground gears)",
                            "mineral gear oil, typical ν40/ν100 (VI ≈ 95), X_L = 1",
                            "deep-groove ball bearings, μ = 0.0015 (catalogue upper value)"]}


def rounded(r):
    """Compact, rounded copy for API output."""
    out = {k: v for k, v in r.items() if k not in ("mesh", "bearings", "seals")}
    out["oil_temp_c"] = round(r["oil_temp_c"], 1)
    for k in ("mesh_w", "bearings_w", "seals_w", "total_load_dependent_w", "input_power_w"):
        out[k] = round(r[k], 1)
    if r.get("efficiency_load_dependent") is not None:
        out["efficiency_load_dependent"] = round(r["efficiency_load_dependent"], 5)
    out["mesh"] = [{"stage": x["stage"], "loss_w": round(x["loss_w"], 1), "mu_mz": round(x["mu_mz"], 4),
                    "H_V": round(x["H_V"], 4), "eps_alpha": round(x["eps_alpha"], 3), "v_t_ms": round(x["v_t_ms"], 2),
                    "valid": x["valid"]} for x in r["mesh"]]
    out["bearings"] = [{**{k: x[k] for k in ("shaft", "support", "rpm")}, "radial_n": round(x["radial_n"], 1),
                        "bore_mm": x["bore_mm"], "loss_w": round(x["loss_w"], 2)} for x in r["bearings"]]
    out["seals"] = [{"shaft_mm": x["shaft_mm"], "rpm": round(x["rpm"], 1), "loss_w": round(x["loss_w"], 2)}
                    for x in r["seals"]]
    return out
