"""Rolling-bearing selection (Shigley 10th ed., ch.11; ISO 281 basic rating).

Catalog values are APPROXIMATE SKF deep-groove ball bearing data (d, D, B in mm;
C, C0 in kN). Verify against the current SKF/NSK catalog before ordering parts.
"""
import math

# designation: (d, D, B, C_kN, C0_kN)
DEEP_GROOVE = {
    # 60xx — light, thin section
    "6000": (10, 26, 8, 4.75, 1.96), "6001": (12, 28, 8, 5.4, 2.36),
    "6002": (15, 32, 9, 5.85, 2.85), "6003": (17, 35, 10, 6.37, 3.25),
    "6004": (20, 42, 12, 9.95, 5.0), "6005": (25, 47, 12, 11.9, 6.55),
    "6006": (30, 55, 13, 13.8, 8.3), "6007": (35, 62, 14, 16.8, 10.2),
    "6008": (40, 68, 15, 17.8, 11.6), "6009": (45, 75, 16, 22.1, 14.6),
    "6010": (50, 80, 16, 22.9, 16.0),
    # 62xx — medium
    "6200": (10, 30, 9, 5.4, 2.36), "6201": (12, 32, 10, 7.28, 3.1),
    "6202": (15, 35, 11, 8.06, 3.75), "6203": (17, 40, 12, 9.95, 4.75),
    "6204": (20, 47, 14, 13.5, 6.55), "6205": (25, 52, 15, 14.8, 7.8),
    "6206": (30, 62, 16, 20.3, 11.2), "6207": (35, 72, 17, 27.0, 15.3),
    "6208": (40, 80, 18, 32.5, 19.0), "6209": (45, 85, 19, 35.1, 21.6),
    "6210": (50, 90, 20, 37.1, 23.2), "6211": (55, 100, 21, 46.2, 29.0),
    "6212": (60, 110, 22, 55.3, 36.0),
    # 63xx — heavy
    "6300": (10, 35, 11, 8.52, 3.4), "6301": (12, 37, 12, 10.8, 4.15),
    "6302": (15, 42, 13, 11.9, 5.4), "6303": (17, 47, 14, 14.3, 6.55),
    "6304": (20, 52, 15, 16.8, 7.8), "6305": (25, 62, 17, 23.4, 11.6),
    "6306": (30, 72, 19, 29.6, 16.0), "6307": (35, 80, 21, 35.1, 19.0),
    "6308": (40, 90, 23, 42.3, 24.0), "6309": (45, 100, 25, 55.3, 31.5),
    "6310": (50, 110, 27, 65.0, 38.0),
}
STANDARD_BORES = sorted({v[0] for v in DEEP_GROOVE.values()})

# Shigley Table 11-1, deep-groove ball bearings: (Fa/C0, e, Y2) with X2 = 0.56
_XY = [(0.014, 0.19, 2.30), (0.021, 0.21, 2.15), (0.028, 0.22, 1.99), (0.042, 0.24, 1.85),
       (0.056, 0.26, 1.71), (0.070, 0.27, 1.63), (0.084, 0.28, 1.55), (0.110, 0.30, 1.45),
       (0.170, 0.34, 1.31), (0.280, 0.38, 1.15), (0.420, 0.42, 1.04), (0.560, 0.44, 1.00)]

# Weibull parameters for manufacturer "1e6 rev" ratings (Shigley Ex. 11-3)
X0, THETA_MINUS_X0, B_WEIBULL = 0.02, 4.439, 1.483


def next_bore(d):
    for b in STANDARD_BORES:
        if b >= d - 1e-9:
            return b
    return None


def equivalent_load(Fr, Fa, C0_N, V=1.0):
    """P = X V Fr + Y Fa (Shigley eq. 11-12)."""
    if Fa <= 0:
        return {"P": V * Fr, "X": 1.0, "Y": 0.0, "e": None}
    r = Fa / C0_N
    r = min(max(r, _XY[0][0]), _XY[-1][0])
    for (r0, e0, y0), (r1, e1, y1) in zip(_XY, _XY[1:]):
        if r0 <= r <= r1:
            t = (r - r0) / (r1 - r0)
            e, Y = e0 + t * (e1 - e0), y0 + t * (y1 - y0)
            break
    if Fa / (V * Fr if Fr > 0 else 1e-9) <= e:
        return {"P": V * Fr, "X": 1.0, "Y": 0.0, "e": e}
    return {"P": 0.56 * V * Fr + Y * Fa, "X": 0.56, "Y": Y, "e": e}


def _weibull_den(R):
    return X0 + THETA_MINUS_X0 * math.log(1 / R) ** (1 / B_WEIBULL)


def required_C10(P, rpm, life_h, reliability=0.9, af=1.0, ball=True):
    """Catalog rating needed, N (Shigley eq. 11-10)."""
    a = 3.0 if ball else 10 / 3
    xD = life_h * 60 * rpm / 1e6
    return af * P * (xD / _weibull_den(reliability)) ** (1 / a)


def life_hours(C10, P, rpm, reliability=0.9, af=1.0, ball=True):
    a = 3.0 if ball else 10 / 3
    if P <= 0:
        return math.inf
    xD = (C10 / (af * P)) ** a * _weibull_den(reliability)
    return xD * 1e6 / (60 * rpm)


def select_bearing(bore_min, Fr, Fa, rpm, life_h, reliability=0.99, af=1.2,
                   series=("60", "62", "63"), static_sf=1.0):
    """Smallest bore >= bore_min, then first series in preference order that passes."""
    for bore in STANDARD_BORES:
        if bore < bore_min - 1e-9:
            continue
        for s in series:
            for desig, (d, D, B, C, C0) in DEEP_GROOVE.items():
                if d != bore or not desig.startswith(s):
                    continue
                eq = equivalent_load(Fr, Fa, C0 * 1e3)
                Creq = required_C10(eq["P"], rpm, life_h, reliability, af)
                P0 = max(Fr, 0.6 * Fr + 0.5 * Fa)
                s0 = math.inf if P0 == 0 else C0 * 1e3 / P0
                if C * 1e3 >= Creq and s0 >= static_sf:
                    return {"designation": desig, "d_mm": d, "D_mm": D, "B_mm": B,
                            "C_kN": C, "C0_kN": C0, "Fr_N": Fr, "Fa_N": Fa,
                            "P_N": eq["P"], "X": eq["X"], "Y": eq["Y"],
                            "C10_required_kN": Creq / 1e3,
                            "life_h_at_reliability": life_hours(C * 1e3, eq["P"], rpm, reliability, af),
                            "reliability": reliability, "static_safety_s0": s0,
                            "ndm_mm_rpm": rpm * (d + D) / 2}
    return None
