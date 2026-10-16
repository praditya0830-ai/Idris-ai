"""Tooth-form factors from real generated tooth geometry (ISO 6336-3, Method C style).

The root fillet of a rack-generated spur gear is a trochoid. We compute it exactly from the
standard ISO rack (hf = 1.25 m, root radius rho = 0.38 m, alpha = 20 deg), find the critical
section with the 30 deg tangent, and derive

    Y_F : tooth form factor (load at the tooth tip)      Y_F = 6 (hF/m) cos(aFen) / ((sFn/m)^2 cos(alpha))
    Y_S : stress correction factor (notch + shear/compression)   [ISO 6336-3 eq. (Method C)]

All lengths in units of module m; profile shift x = 0.

v2 (tim solver, 16 Oct 2026): the bending arm hFe is measured to the intersection of the load line
with the tooth centreline, as ISO 6336-3 defines it (v1 used the height of the tip point, which made
Y_F ≈ 10 % too high and Y_S ≈ 2 % too low). Benchmarked against the ISO 6336-3 closed-form
equations (iso_closed_form below): Y_F and Y_S agree within 0.5 % for z = 17…200.
"""
from __future__ import annotations

import math
from functools import lru_cache

ALPHA = math.radians(20.0)
H_F0 = 1.25          # rack addendum  -> root circle at r - 1.25 m
RHO_F0 = 0.38        # rack tip radius


def _inv(a: float) -> float:
    return math.tan(a) - a


@lru_cache(maxsize=512)
def tooth_form(z: int, n: int = 600) -> dict:
    r = z / 2.0
    rb, ra = r * math.cos(ALPHA), r + 1.0
    # arc centre in rack coordinates (right-hand arc), see derivation in docs
    yc = -H_F0 + RHO_F0
    xc = math.pi / 4.0 + yc * math.tan(ALPHA) - RHO_F0 / math.cos(ALPHA)
    rot = math.pi / z                       # rotates the tooth axis onto +y
    pts, gam = [], []
    for i in range(n + 1):
        g = math.radians(-90.0) + (math.radians(90.0) - ALPHA) * i / n        # -90 deg .. -alpha
        xr, yr = xc + RHO_F0 * math.cos(g), yc + RHO_F0 * math.sin(g)
        x0 = xr - yr * math.cos(g) / math.sin(g) if abs(math.sin(g)) > 1e-9 else xr   # normal meets pitch line
        th = math.pi / 2.0 - x0 / r
        px = r * math.cos(th) + (xr - x0) * math.sin(th) + yr * math.cos(th)
        py = r * math.sin(th) - (xr - x0) * math.cos(th) + yr * math.sin(th)
        pts.append((px * math.cos(rot) - py * math.sin(rot), px * math.sin(rot) + py * math.cos(rot)))
        gam.append(g)
    # tangent angle to the tooth axis along the fillet; find the 30 deg point (from the root upward)
    target = math.radians(30.0)
    idx = None
    for i in range(1, n):
        dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        ang = math.atan2(abs(dx), abs(dy))           # angle between tangent and the tooth axis (y)
        if ang <= target:
            idx = i
            break
    if idx is None:
        raise ValueError(f"no 30 deg tangent point found for z={z}")
    xt, yt = pts[idx]
    s_fn = 2.0 * abs(xt)
    # curvature radius of the fillet at the critical point (three-point circle)
    (x1, y1), (x2, y2), (x3, y3) = pts[idx - 1], pts[idx], pts[idx + 1]
    a_, b_, c_ = math.dist((x2, y2), (x3, y3)), math.dist((x1, y1), (x3, y3)), math.dist((x1, y1), (x2, y2))
    area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    rho_f = a_ * b_ * c_ / (2.0 * area2) if area2 > 1e-14 else RHO_F0
    # load point at the tip: tooth half-thickness angle at ra
    a_a = math.acos(rb / ra)
    eta = math.pi / (2.0 * z) + _inv(ALPHA) - _inv(a_a)
    a_fen = a_a - eta
    # bending arm: from the critical section to where the LINE OF ACTION of the tip load crosses the
    # tooth centreline (ISO 6336-3: hFe/m = ½[(cos γ − sin γ·tan αFen)·da/m − …]), not to the tip point
    y_load = ra * (math.cos(eta) - math.sin(eta) * math.tan(a_fen))
    h_fe = y_load - yt
    yf = 6.0 * h_fe * math.cos(a_fen) / (s_fn ** 2 * math.cos(ALPHA))
    L = s_fn / h_fe
    qs = s_fn / (2.0 * rho_f)
    ys = (1.2 + 0.13 * L) * qs ** (1.0 / (1.21 + 2.3 / L))
    return {"YF": yf, "YS": ys, "sFn": s_fn, "hFe": h_fe, "rhoF": rho_f, "alphaFen_deg": math.degrees(a_fen),
            "root_radius": min(math.hypot(*p) for p in pts)}


def iso_closed_form(z: int, x: float = 0.0) -> dict:
    """ISO 6336-3 analytical tooth-form equations (30° tangent, load at the tip), used as the benchmark."""
    a, hfP, rho = ALPHA, H_F0, RHO_F0
    E = math.pi / 4 - hfP * math.tan(a) - (1 - math.sin(a)) * rho / math.cos(a)
    G = rho - hfP + x
    H = 2 / z * (math.pi / 2 - E) - math.pi / 3
    th = math.pi / 6
    for _ in range(100):
        th = 2 * G / z * math.tan(th) - H
    s_fn = z * math.sin(math.pi / 3 - th) + math.sqrt(3) * (G / math.cos(th) - rho)
    rho_f = rho + 2 * G * G / (math.cos(th) * (z * math.cos(th) ** 2 - 2 * G))
    da, db = z + 2 + 2 * x, z * math.cos(a)
    a_en = math.acos(db / da)
    g_e = (0.5 * math.pi + 2 * math.tan(a) * x) / z + _inv(a) - _inv(a_en)
    a_fen = a_en - g_e
    h_fe = 0.5 * ((math.cos(g_e) - math.sin(g_e) * math.tan(a_fen)) * da - z * math.cos(math.pi / 3 - th)
                  - G / math.cos(th) + rho)
    yf = 6 * h_fe * math.cos(a_fen) / (s_fn ** 2 * math.cos(a))
    L = s_fn / h_fe
    ys = (1.2 + 0.13 * L) * (s_fn / (2 * rho_f)) ** (1 / (1.21 + 2.3 / L))
    return {"YF": yf, "YS": ys, "sFn": s_fn, "hFe": h_fe, "rhoF": rho_f}
