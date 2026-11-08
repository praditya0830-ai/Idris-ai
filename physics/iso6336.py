"""ISO 6336 load-capacity rating of one cylindrical gear pair (pitting chain), checkable factor by factor.

Purpose: let an engineer re-rate a gear pair they already designed (their own numbers in, every factor out),
and compare against their tool. Validated against the published worked example
ISO/TR 6336-30 Example 1 (single helical, case carburized) - see solvers/gear_pair_iso.py.

Scope (honest):
  * External spur and helical pairs, involute, profile shift allowed, steel/steel.
  * ISO 6336-1:2019 Method B for K_v (all speed ranges) and K_Halpha; single/mesh stiffness c', c_gamma.
  * K_Hbeta is an INPUT (from the engineer's tool or measurement). Computing it from shaft deflection
    (ISO 6336-1 Annex E) is the next solver step.  K_Fbeta is derived from K_Hbeta (N_F exponent).
  * ISO 6336-2:2019 Method B: Z_H, Z_E, Z_eps, Z_beta, Z_B/Z_D, Z_NT, Z_L, Z_v, Z_R, Z_X; Z_W = 1 (same-hardness pair).
  * Tolerances f_pb, f_falpha from ISO 1328-1:2013 when the measured values are not given.
  * Not covered: tooth-root bending of this pair (separate module, x = 0 only), internal gears,
    double helical, limited pitting permitted, load spectra, scuffing, micropitting.
"""
from __future__ import annotations

import math

STEEL_E = 206000.0
STEEL_NU = 0.3
STEEL_RHO = 7.83e-6          # kg/mm^3


def _inv(a):
    return math.tan(a) - a


def _inv_solve(target):
    a = 0.5
    for _ in range(60):
        f = _inv(a) - target
        a -= f / (math.tan(a) ** 2)
    return a


def _round_1328(v):
    """ISO 1328-1:2013 rounding of tolerance values: > 10 um to 1 um, otherwise to 0.1 um."""
    return float(round(v)) if v > 10.0 else round(v, 1)


def tol_1328(d, mn, grade):
    """Single pitch tolerance f_pT and profile form tolerance f_falphaT [um], ISO 1328-1:2013."""
    s = math.sqrt(2.0) ** (grade - 5)
    f_pt = (0.001 * d + 0.4 * mn + 5.0) * s
    f_fa = (2.5 * math.sqrt(mn) + 0.17 * math.sqrt(d) + 0.5) * s
    return _round_1328(f_pt), _round_1328(f_fa)


def _c_zl(s_hlim):
    s = min(max(s_hlim, 850.0), 1200.0)
    return 0.08 * (s - 850.0) / 350.0 + 0.83


def _c_zr(s_hlim):
    if s_hlim < 850.0:
        return 0.15
    if s_hlim > 1200.0:
        return 0.08
    return 0.32 - 0.0002 * s_hlim


def life_factor_contact(n_l, z_10=0.85):
    """Z_NT for St, V, GGG, GTS, Eh, IF, limited pitting NOT permitted (ISO 6336-2 Table/Fig.):
    1.6 at <= 1e5, 1.0 at 5e7, z_10 at 1e10 (0.85 default; 1.0 with optimum lubrication/material/manufacture);
    log-log interpolation between the points."""
    if n_l <= 1e5:
        return 1.6
    if n_l <= 5e7:
        return math.exp(math.log(1.6) * math.log(5e7 / n_l) / math.log(5e7 / 1e5))
    if n_l <= 1e10:
        return math.exp(math.log(z_10) * math.log(n_l / 5e7) / math.log(1e10 / 5e7))
    return z_10


def rate_pair(p: dict) -> dict:
    mn, an, beta = p["mn"], math.radians(p.get("alpha_n_deg", 20.0)), math.radians(p.get("beta_deg", 0.0))
    z1, z2 = int(p["z1"]), int(p["z2"])
    x1, x2 = p.get("x1", 0.0), p.get("x2", 0.0)
    b = p["b"]
    ha, hf = p.get("haP", 1.0), p.get("hfP", 1.25)
    ka = p.get("K_A", 1.0)
    k_hb = p["K_Hbeta"]
    grade = p.get("accuracy_grade", 6)
    E, nu, rho = p.get("E", STEEL_E), p.get("nu", STEEL_NU), p.get("density", STEEL_RHO)
    hard = p.get("material_type", "case_hardened")
    s_hlim1, s_hlim2 = p["sigma_Hlim1"], p.get("sigma_Hlim2", p["sigma_Hlim1"])
    assumptions = []

    # ---- geometry ------------------------------------------------------------------------------
    u = z2 / z1
    at = math.atan(math.tan(an) / math.cos(beta))
    bb = math.asin(math.sin(beta) * math.cos(an))
    d1, d2 = z1 * mn / math.cos(beta), z2 * mn / math.cos(beta)
    db1, db2 = d1 * math.cos(at), d2 * math.cos(at)
    a0 = (d1 + d2) / 2.0
    if p.get("a_w"):
        a_w = p["a_w"]
        awt = math.acos(a0 * math.cos(at) / a_w)
    else:
        awt = _inv_solve(_inv(at) + 2.0 * math.tan(an) * (x1 + x2) / (z1 + z2))
        a_w = a0 * math.cos(at) / math.cos(awt)
    da1, da2 = d1 + 2 * mn * (ha + x1), d2 + 2 * mn * (ha + x2)
    if p.get("da1"):
        da1 = p["da1"]
    if p.get("da2"):
        da2 = p["da2"]
    df1, df2 = d1 - 2 * mn * (hf - x1), d2 - 2 * mn * (hf - x2)
    pbt = math.pi * mn * math.cos(at) / math.cos(beta)
    g_a = 0.5 * math.sqrt(da1**2 - db1**2) + 0.5 * math.sqrt(da2**2 - db2**2) - a_w * math.sin(awt)
    e_a = g_a / pbt
    e_b = b * math.sin(beta) / (math.pi * mn)
    e_g = e_a + e_b
    zn1 = z1 / (math.cos(bb) ** 2 * math.cos(beta))
    zn2 = z2 / (math.cos(bb) ** 2 * math.cos(beta))

    # ---- loads ---------------------------------------------------------------------------------
    t1, n1 = p["torque_1_nm"], p["n1_rpm"]
    ft = 2000.0 * t1 / d1
    v = math.pi * d1 * n1 / 60000.0
    w = ka * ft / b                                           # N/mm

    # ---- stiffness (ISO 6336-1 §9, Method B) ----------------------------------------------------
    C = (0.04723, 0.15551, 0.25791, -0.00635, -0.11654, -0.00193, -0.24188, 0.00529, 0.00182)
    q = (C[0] + C[1] / zn1 + C[2] / zn2 + C[3] * x1 + C[4] * x1 / zn1 + C[5] * x2 + C[6] * x2 / zn2
         + C[7] * x1**2 + C[8] * x2**2)
    c_th = 1.0 / q
    c_m, c_r = 0.8, 1.0
    c_b = (1 + 0.5 * (1.2 - hf)) * (1 - 0.02 * (20.0 - math.degrees(an)))
    c1 = c_th * c_m * c_r * c_b * math.cos(beta)
    if w < 100.0:
        c1 *= (w / 100.0) ** 0.25
    c_ga = c1 * (0.75 * e_a + 0.25)
    c_gb = 0.85 * c_ga
    if c_r == 1.0:
        assumptions.append("Solid gear blanks (C_R = 1); a webbed blank lowers mesh stiffness.")

    # ---- deviations and running-in -------------------------------------------------------------
    d_tol = max(d1, d2) if p.get("tolerance_on", "wheel") == "wheel" else d1
    f_pt, f_fa = tol_1328(d_tol, mn, grade)
    f_pb = p.get("f_pb_um") or round(f_pt * math.cos(at), 2)
    f_fa = p.get("f_falpha_um") or f_fa
    if not p.get("f_pb_um"):
        assumptions.append(f"f_pb = f_pT·cos α_t from ISO 1328-1:2013 grade {grade} (d = {d_tol:.1f} mm), "
                           "not a measured value.")
    if hard == "case_hardened":
        y_a = min(0.075 * f_pb, 3.0)
    else:
        y_a = 160.0 / min(s_hlim1, s_hlim2) * f_pb
        if v > 10.0:
            y_a = min(y_a, 6400.0 / min(s_hlim1, s_hlim2))
        elif v > 5.0:
            y_a = min(y_a, 12800.0 / min(s_hlim1, s_hlim2))
    y_f = y_a
    ca = p.get("C_a_um", 0.0)

    # ---- dynamic factor K_v, Method B ----------------------------------------------------------
    qb1, qb2 = p.get("bore_ratio1", 0.0), p.get("bore_ratio2", 0.0)
    dm1, dm2 = (da1 + df1) / 2.0, (da2 + df2) / 2.0
    m_red = (math.pi / 8.0) * (dm1 / db1) ** 2 * dm1**2 / (
        1.0 / ((1 - qb1**4) * rho) + 1.0 / ((1 - qb2**4) * rho * u**2))
    n_e1 = 30000.0 / (math.pi * z1) * math.sqrt(c_ga / m_red)
    N = n1 / n_e1
    n_s = 0.85 if w >= 100.0 else 0.5 + 0.35 * math.sqrt(w / 100.0)
    bp = c1 * max(f_pb - y_a, 0.0) / w
    bf = c1 * max(f_fa - y_f, 0.0) / w
    bk = abs(1.0 - c1 * ca / w)
    cv1 = 0.32
    cv2 = 0.34 if e_g <= 2 else 0.57 / (e_g - 0.3)
    cv3 = 0.23 if e_g <= 2 else 0.096 / (e_g - 1.56)
    cv4 = 0.90 if e_g <= 2 else (0.57 - 0.05 * e_g) / (e_g - 1.44)
    cv5 = 0.47
    cv6 = 0.47 if e_g <= 2 else 0.12 / (e_g - 1.74)
    if e_g <= 1.5:
        cv7 = 0.75
    elif e_g <= 2.5:
        cv7 = 0.125 * math.sin(math.pi * (e_g - 2.0)) + 0.875
    else:
        cv7 = 1.0
    kv_res = cv1 * bp + cv2 * bf + cv4 * bk + 1.0
    kv_sup = cv5 * bp + cv6 * bf + cv7
    if N <= n_s:
        kv, kv_range = N * (cv1 * bp + cv2 * bf + cv3 * bk) + 1.0, "subcritical"
    elif N <= 1.15:
        kv, kv_range = kv_res, "main resonance (avoid)"
    elif N < 1.5:
        kv, kv_range = kv_sup + (kv_res - kv_sup) / 0.35 * (1.5 - N), "intermediate"
    else:
        kv, kv_range = kv_sup, "supercritical"
    if kv_range != "subcritical":
        assumptions.append(f"K_v in the {kv_range} range: formula implemented, not yet checked against a worked example.")

    # ---- transverse load factors, Method B -------------------------------------------------------
    z_eps = (math.sqrt((4 - e_a) / 3.0) if e_b == 0 else
             math.sqrt((4 - e_a) / 3.0 * (1 - e_b) + e_b / e_a) if e_b < 1 else math.sqrt(1.0 / e_a))
    w_h = ft * ka * kv * k_hb / b
    if e_g <= 2:
        k_ha = e_g / 2.0 * (0.9 + 0.4 * c_ga * (f_pb - y_a) / w_h)
    else:
        k_ha = 0.9 + 0.4 * math.sqrt(2 * (e_g - 1) / e_g) * c_ga * (f_pb - y_a) / w_h
    k_ha = min(max(k_ha, 1.0), e_g / (e_a * z_eps**2))
    k_fa = min(max(k_ha, 1.0), e_g / (0.25 * e_a + 0.75))

    # ---- face load factor for root stress from K_Hbeta ----------------------------------------
    h1, h2 = (da1 - df1) / 2.0, (da2 - df2) / 2.0
    bh = max(min(b / h1, b / h2), 3.0)
    n_f = bh**2 / (1 + bh + bh**2)
    k_fb = k_hb ** n_f

    # ---- contact stress (ISO 6336-2) -----------------------------------------------------------
    z_h = math.sqrt(2 * math.cos(bb) * math.cos(awt) / (math.cos(at) ** 2 * math.sin(awt)))
    z_e = math.sqrt(1.0 / (math.pi * 2 * (1 - nu**2) / E))
    z_beta = 1.0 / math.sqrt(math.cos(beta))
    if e_b >= 1:
        z_b = z_d = 1.0
    else:
        g1 = math.sqrt(da1**2 / db1**2 - 1) - 2 * math.pi / z1
        g2 = math.sqrt(da2**2 / db2**2 - 1) - (e_a - 1) * 2 * math.pi / z2
        m1 = math.tan(awt) / math.sqrt(g1 * g2)
        g1b = math.sqrt(da1**2 / db1**2 - 1) - (e_a - 1) * 2 * math.pi / z1
        g2b = math.sqrt(da2**2 / db2**2 - 1) - 2 * math.pi / z2
        m2 = math.tan(awt) / math.sqrt(g1b * g2b)
        z_b = max(1.0, m1 - e_b * (m1 - 1))
        z_d = max(1.0, m2 - e_b * (m2 - 1))
    s_h0 = z_h * z_e * z_eps * z_beta * math.sqrt(ft / (d1 * b) * (u + 1) / u)
    kk = math.sqrt(ka * kv * k_hb * k_ha)
    s_h1, s_h2 = z_b * s_h0 * kk, z_d * s_h0 * kk

    # ---- permissible contact stress -----------------------------------------------------------
    l_h = p.get("life_h", 20000.0)
    nl1 = 60.0 * n1 * l_h * p.get("contacts_per_rev1", 1)
    nl2 = nl1 / u * p.get("contacts_per_rev2", 1) / p.get("contacts_per_rev1", 1)
    z10 = p.get("Z_NT_at_1e10", 0.85)
    znt1, znt2 = life_factor_contact(nl1, z10), life_factor_contact(nl2, z10)
    s_hlim_min = min(s_hlim1, s_hlim2)
    czl = _c_zl(s_hlim_min)
    nu40 = p.get("nu40", 220.0)
    z_l = czl + 4 * (1 - czl) / (1.2 + 134.0 / nu40) ** 2
    czv = czl + 0.02
    z_v = czv + 2 * (1 - czv) / math.sqrt(0.8 + 32.0 / v)
    rz1 = p.get("Rz1_um") or 6.0 * p.get("Ra1_um", p.get("Ra_um", 0.8))
    rz2 = p.get("Rz2_um") or 6.0 * p.get("Ra2_um", p.get("Ra_um", 0.8))
    if not p.get("Rz1_um"):
        assumptions.append("Flank R_z = 6·R_a (ISO/TR 6336-30 approximation) because R_z was not given.")
    rho1, rho2 = 0.5 * db1 * math.tan(awt), 0.5 * db2 * math.tan(awt)
    rho_red = rho1 * rho2 / (rho1 + rho2)
    rz10 = 0.5 * (rz1 + rz2) * (10.0 / rho_red) ** (1.0 / 3.0)
    z_r = (3.0 / rz10) ** _c_zr(s_hlim_min)
    if hard == "case_hardened":
        z_x = 1.0 if mn < 10 else max(1.05 - 0.005 * mn, 0.9)
    else:
        z_x = 1.0
    z_w = 1.0
    common = z_l * z_v * z_r * z_w * z_x
    s_hmin = p.get("S_Hmin", 1.0)
    shp1 = s_hlim1 * znt1 * common / s_hmin
    shp2 = s_hlim2 * znt2 * common / s_hmin
    sh1 = s_hlim1 * znt1 * common / s_h1
    sh2 = s_hlim2 * znt2 * common / s_h2

    r = lambda x, n=5: round(x, n)
    return {
        "outputs": {
            "geometry": {"d1": r(d1, 4), "d2": r(d2, 4), "a_w": r(a_w, 4), "alpha_t_deg": r(math.degrees(at)),
                         "alpha_wt_deg": r(math.degrees(awt)), "beta_b_deg": r(math.degrees(bb)),
                         "da1": r(da1, 4), "da2": r(da2, 4), "df1": r(df1, 4), "df2": r(df2, 4),
                         "eps_alpha": r(e_a), "eps_beta": r(e_b), "eps_gamma": r(e_g),
                         "z_n1": r(zn1, 4), "z_n2": r(zn2, 4), "u": r(u)},
            "loads": {"F_t": r(ft, 2), "v": r(v), "w_N_per_mm": r(w, 3), "N_L1": nl1, "N_L2": round(nl2)},
            "stiffness": {"c_th": r(c_th), "c_prime": r(c1), "c_gamma_alpha": r(c_ga), "c_gamma_beta": r(c_gb),
                          "C_B": r(c_b), "C_M": c_m, "C_R": c_r},
            "deviations_um": {"f_pT": f_pt, "f_pb": f_pb, "f_falpha": f_fa, "y_alpha": r(y_a, 3), "C_a": ca},
            "factors": {"K_A": ka, "K_v": r(kv), "K_v_range": kv_range, "N": r(N), "n_E1": r(n_e1, 1),
                        "m_red": m_red, "B_p": r(bp), "B_f": r(bf), "B_k": r(bk),
                        "K_Hbeta": k_hb, "K_Fbeta": r(k_fb), "N_F": r(n_f), "K_Halpha": r(k_ha), "K_Falpha": r(k_fa),
                        "Z_H": r(z_h), "Z_E": r(z_e, 4), "Z_eps": r(z_eps), "Z_beta": r(z_beta),
                        "Z_B": r(z_b), "Z_D": r(z_d), "Z_NT1": r(znt1), "Z_NT2": r(znt2), "Z_L": r(z_l),
                        "Z_v": r(z_v), "Z_R": r(z_r), "Z_W": z_w, "Z_X": z_x, "rho_red": r(rho_red, 4),
                        "R_z10": r(rz10, 4)},
            "contact": {"sigma_H0": r(s_h0, 3), "sigma_H1": r(s_h1, 3), "sigma_H2": r(s_h2, 3),
                        "sigma_HP1": r(shp1, 3), "sigma_HP2": r(shp2, 3), "S_H1": r(sh1), "S_H2": r(sh2),
                        "S_Hmin": s_hmin, "ok": min(sh1, sh2) >= s_hmin},
        },
        "assumptions": assumptions + [
            "K_Hβ is taken as given (input), not computed from shaft/housing deflection.",
            "Z_W = 1 (both gears the same hardness class).",
            "Limited pitting not permitted (Z_NT curve without the permitted-pitting branch).",
        ],
        "standard": "ISO 6336-1:2019 (Method B), ISO 6336-2:2019 (Method B), ISO 1328-1:2013",
    }
