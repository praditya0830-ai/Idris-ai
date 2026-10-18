"""Fast thermal screen of a gearbox housing (L1) + cooling-fin model shared with the 3D solver.

Lumped network: oil --(h_oil·A_in)--> wall --(t/k·A)--> outer skin --((h_c+h_r)·A_out)--> ambient.
Same correlations as thermal_housing_3d (Churchill-Chu natural / flat-plate forced + radiation), so the
screen and the 3D result use one physics; the 3D FEM stays the reference (L2).

Fins: straight rectangular fins on the two side walls (x = 0 and x = W, vertical, running along
the housing height) or on the top face (running along the width). Fin efficiency η = tanh(m·Lc)/(m·Lc), m = √(2h/(k·t)), Lc = L + t/2
(Incropera eq. 3.89 / 3.92). Under natural convection the air between fins is a vertical channel:
h from Bar-Cohen & Rohsenow (1984) parallel-plate correlation
    Nu_s = [576/El² + 2.873/El^½]^-½ ,  El = Ra_s · s / L_fin .
Under forced air the fins are assumed parallel to the flow (h of the base plate). Radiation is
counted on the base envelope only (fins mostly see each other) — conservative.
"""
import math
from .thermal3d import AIR, air_props, h_natural_air, h_forced_air

FIN_THICKNESS_MM = 3.0        # cast / machined aluminium fin, practical minimum
FIN_MIN_GAP_MM = 4.0          # below this, casting and dirt make fins useless
G = 9.81
ALPHA_AIR = 2.22e-5           # thermal diffusivity of air ≈ 300 K, m²/s


def h_channel_natural(gap_mm, L_fin_mm, T_s, T_amb):
    """Average h between vertical parallel plates, natural convection (Bar-Cohen & Rohsenow)."""
    s = max(gap_mm, 0.5) * 1e-3
    Lf = max(L_fin_mm, 5.0) * 1e-3
    dT = max(T_s - T_amb, 0.1)
    beta = 1.0 / (0.5 * (T_s + T_amb) + 273.15)
    a = air_props(0.5 * (T_s + T_amb))
    Ra_s = G * beta * dT * s ** 3 / (a["nu"] * a["alpha"])
    El = max(Ra_s * s / Lf, 1e-9)
    Nu = (576.0 / El ** 2 + 2.873 / El ** 0.5) ** -0.5
    return Nu * a["k"] / s


def fin_layout(fins, H_mm, D_mm, W_mm=None):
    """Geometry of `fins` = {count, height_mm[, thickness_mm][, location]}.
    location "sides" (default): vertical fins on the two side walls (x = 0, x = W), length = H, the
                                 housing gets 2 × height wider.
    location "top": fins on the top face running along the width, length = W, the housing gets
                    height taller; under natural convection the air rises through a channel as
                    tall as the fin."""
    n = int(fins.get("count", 0))
    hf = float(fins.get("height_mm", 0.0))
    tf = float(fins.get("thickness_mm", FIN_THICKNESS_MM))
    loc = fins.get("location", "sides")
    if n <= 0 or hf <= 0:
        return None
    if loc == "top":
        per_row = n
        length = float(W_mm if W_mm else H_mm)
        channel = hf
    else:
        loc = "sides"
        per_row = max(1, math.ceil(n / 2))
        length = float(H_mm)
        channel = float(H_mm)
    gap = (D_mm - per_row * tf) / per_row
    return {"count": n, "height_mm": hf, "thickness_mm": tf, "location": loc, "per_side": per_row,
            "gap_mm": gap, "length_mm": length, "channel_len_mm": channel}


def fin_conv_factor(fl, h_base, k, A_out_m2, cooling, T_s, T_amb):
    """Factor on the outer CONVECTIVE conductance that the fins give (≥ 1 normally)."""
    if fl is None:
        return 1.0, None
    if fl["gap_mm"] <= 0:
        return 1.0, 0.0
    h_f = h_channel_natural(fl["gap_mm"], fl.get("channel_len_mm", fl["length_mm"]), T_s, T_amb) if cooling == "natural" else h_base
    tf = fl["thickness_mm"] * 1e-3
    Lc = (fl["height_mm"] + 0.5 * fl["thickness_mm"]) * 1e-3
    m = math.sqrt(2 * h_f / (k * tf))
    eta = math.tanh(m * Lc) / (m * Lc)
    Lf = fl["length_mm"] * 1e-3
    n = fl["count"]
    # between-fin base area keeps h_base; fin surfaces get h_f·η; the fin root covers base area
    added = n * Lf * (2 * Lc * eta * h_f - tf * h_base)
    return 1.0 + added / (h_base * A_out_m2), eta


def fin_mass_kg(fl, rho):
    if fl is None:
        return 0.0
    return fl["count"] * fl["length_mm"] * fl["height_mm"] * fl["thickness_mm"] * 1e-9 * rho


def box_areas(W, H, D, t, web_mm=0.0, cover_mm=None, extra_out_mm2=0.0):
    """Outer and inner (oil-wetted) areas of the closed box, m². A central web (2-stage layout v4) of
    thickness web_mm splits the inside into two oil chambers: its two faces are wetted too."""
    wi, hi, di = W - 2 * t, H - 2 * t, D - 2 * (cover_mm or t)
    A_out = (2 * (W * H + W * D + H * D) + float(extra_out_mm2 or 0.0)) * 1e-6
    web = max(0.0, float(web_mm or 0.0))
    A_in = (2 * wi * hi * (2 if web else 1) + 2 * (wi + hi) * (di - web)) * 1e-6
    return A_out, A_in


def oil_temp_lumped(Q_w, W, H, D, t, k, h_oil, T_amb=25.0, cooling="natural", air_speed=0.0,
                    fins=None, oil_cooler_w=0.0, emissivity=0.85, web_mm=0.0, cover_mm=None,
                    extra_out_mm2=0.0):
    """Steady oil temperature of a closed box housing (mm, W, W/mK, W/m²K). Returns a dict."""
    A_out, A_in = box_areas(W, H, D, t, web_mm, cover_mm, extra_out_mm2)
    Q = max(0.0, Q_w - max(0.0, oil_cooler_w))
    fl = fin_layout(fins, H, D, W) if fins else None
    R_in = 1.0 / (h_oil * A_in)
    R_wall = (t * 1e-3) / (k * 0.5 * (A_in + A_out))
    T_s = T_amb + 20.0
    f = 1.0
    eta = None
    for _ in range(60):
        if cooling == "forced":
            hc, hr, _ = h_forced_air(air_speed, W * 1e-3, T_s, T_amb, emissivity)
        else:
            hc, hr = h_natural_air(T_s - T_amb, H * 1e-3, T_s, T_amb, emissivity)
        f, eta = fin_conv_factor(fl, hc, k, A_out, cooling, T_s, T_amb)
        G_out = (hc * f + hr) * A_out
        T_new = T_amb + Q / G_out
        if abs(T_new - T_s) < 0.01:
            T_s = T_new
            break
        T_s = 0.5 * (T_s + T_new)
    T_oil = T_s + Q * (R_in + R_wall)
    return {"oil_temp_c": T_oil, "skin_temp_c": T_s, "area_outer_m2": A_out, "area_wetted_m2": A_in,
            "h_conv": hc, "h_rad": hr, "fin_factor": f, "fin_efficiency": eta, "heat_w": Q,
            "fin_gap_mm": fl["gap_mm"] if fl else None}
