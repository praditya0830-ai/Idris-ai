"""Shaft strength, deflection and critical speed (Shigley 10th ed., ch.6-7).

Pure stdlib. Units: mm, N, N*mm, MPa, rad, kg, rpm.
All formulas are closed-form or plain numerical integration of Euler-Bernoulli
beam theory -- no ML. Material values are approximate handbook values.
"""
import math
from bisect import bisect_right
from statistics import NormalDist

G = 9.80665e3  # mm/s^2

SHAFT_MATERIALS = {
    "1020_CD": {"name": "AISI 1020 cold-drawn", "Sut": 470, "Sy": 390, "E": 207000, "density": 7850},
    "1045_CD": {"name": "AISI 1045 cold-drawn", "Sut": 630, "Sy": 530, "E": 207000, "density": 7850},
    "4140_QT650": {"name": "AISI 4140 Q&T 650C", "Sut": 760, "Sy": 655, "E": 207000, "density": 7850},
    "4140_QT425": {"name": "AISI 4140 Q&T 425C", "Sut": 1250, "Sy": 1140, "E": 207000, "density": 7850},
    "17-4PH_H1025": {"name": "17-4PH stainless H1025", "Sut": 1070, "Sy": 1000, "E": 196000, "density": 7800},
}

# Marin surface factor ka = a * Sut^b  (Shigley Table 6-2, Sut in MPa)
SURFACE = {"ground": (1.58, -0.085), "machined": (4.51, -0.265), "cold_drawn": (4.51, -0.265),
           "hot_rolled": (57.7, -0.718), "forged": (272.0, -0.995)}

# First-iteration stress concentration estimates (Shigley Table 7-1): (Kt, Kts, r/d)
NOTCHES = {
    "none": (1.0, 1.0, None),
    "shoulder_sharp": (2.7, 2.2, 0.02),   # bearing shoulder, small fillet
    "shoulder_round": (1.7, 1.5, 0.10),   # well-rounded fillet
    "keyseat": (2.14, 3.0, 0.02),         # end-mill keyseat
    "ring_groove": (5.0, 3.0, 0.01),      # retaining-ring groove
}


# ---------------- fatigue ----------------

def marin_ka(Sut, finish="machined"):
    a, b = SURFACE[finish]
    return a * Sut ** b


def marin_kb(d):
    """Size factor, rotating round shaft (Shigley eq. 6-20), d in mm."""
    if d <= 2.79:
        return 1.0
    if d <= 51:
        return 1.24 * d ** -0.107
    return 1.51 * d ** -0.157


def marin_ke(reliability):
    """Reliability factor ke = 1 - 0.08 z_a (Shigley eq. 6-28)."""
    if reliability <= 0.5:
        return 1.0
    return 1 - 0.08 * NormalDist().inv_cdf(reliability)


def endurance_limit(Sut, d, finish="machined", reliability=0.99, kd=1.0, kf_misc=1.0):
    Se_prime = 0.5 * Sut if Sut <= 1400 else 700.0
    ka, kb, ke = marin_ka(Sut, finish), marin_kb(d), marin_ke(reliability)
    return {"Se": ka * kb * kd * ke * kf_misc * Se_prime, "Se_prime": Se_prime,
            "ka": ka, "kb": kb, "kd": kd, "ke": ke}


def neuber_sqrt_a(Sut, torsion=False):
    """Neuber constant sqrt(a) in sqrt(mm) (Shigley eq. 6-35a/b, fitted in kpsi)."""
    S = min(max(Sut / 6.894757, 50), 250)
    if torsion:
        v = 0.190 - 2.51e-3 * S + 1.35e-5 * S**2 - 2.67e-8 * S**3
    else:
        v = 0.246 - 3.08e-3 * S + 1.51e-5 * S**2 - 2.67e-8 * S**3
    return v * math.sqrt(25.4)


def fatigue_kf(Kt, r, Sut, torsion=False):
    if Kt <= 1 or not r:
        return 1.0
    q = 1 / (1 + neuber_sqrt_a(Sut, torsion) / math.sqrt(r))
    return 1 + q * (Kt - 1)


def von_mises_stresses(d, Ma, Mm, Ta, Tm, Kf, Kfs):
    """Alternating / mean von Mises stress for combined bending + torsion [MPa]."""
    c = math.pi * d**3
    sa = math.sqrt((32 * Kf * Ma / c) ** 2 + 3 * (16 * Kfs * Ta / c) ** 2)
    sm = math.sqrt((32 * Kf * Mm / c) ** 2 + 3 * (16 * Kfs * Tm / c) ** 2)
    return sa, sm


def fatigue_safety(sa, sm, Se, Sut, Sy, criterion="goodman"):
    if sa <= 0 and sm <= 0:
        return math.inf
    if criterion == "goodman":
        return 1 / (sa / Se + sm / Sut)
    if criterion == "soderberg":
        return 1 / (sa / Se + sm / Sy)
    if criterion == "asme":
        return 1 / math.sqrt((sa / Se) ** 2 + (sm / Sy) ** 2)
    if criterion == "gerber":
        if sm <= 0:
            return Se / sa
        if sa <= 0:
            return Sut / sm
        return 0.5 * (Sut / sm) ** 2 * (sa / Se) * (-1 + math.sqrt(1 + (2 * sm * Se / (Sut * sa)) ** 2))
    raise ValueError(criterion)


def yield_safety(d, Mmax, Tmax, Kf, Kfs, Sy):
    c = math.pi * d**3
    s = math.sqrt((32 * Kf * Mmax / c) ** 2 + 3 * (16 * Kfs * Tmax / c) ** 2)
    return math.inf if s == 0 else Sy / s


def check_section(d, Ma, Tm, notch, mat, finish="machined", reliability=0.99,
                  criterion="goodman", Mm=0.0, Ta=0.0):
    Kt, Kts, rd = NOTCHES[notch]
    r = rd * d if rd else None
    Kf = fatigue_kf(Kt, r, mat["Sut"])
    Kfs = fatigue_kf(Kts, r, mat["Sut"], torsion=True)
    se = endurance_limit(mat["Sut"], d, finish, reliability)
    sa, sm = von_mises_stresses(d, Ma, Mm, Ta, Tm, Kf, Kfs)
    return {"d_mm": d, "notch": notch, "Kt": Kt, "Kts": Kts, "Kf": Kf, "Kfs": Kfs,
            "Se_MPa": se["Se"], "marin": se, "sigma_a_MPa": sa, "sigma_m_MPa": sm,
            "n_fatigue": fatigue_safety(sa, sm, se["Se"], mat["Sut"], mat["Sy"], criterion),
            "n_yield": yield_safety(d, Ma + Mm, Ta + Tm, Kf, Kfs, mat["Sy"])}


def size_section(Ma, Tm, notch, mat, n_design=1.5, finish="machined", reliability=0.99,
                 criterion="goodman", Mm=0.0, Ta=0.0):
    """Smallest diameter meeting n_design in fatigue AND yield (fixed-point on d^3)."""
    if Ma + Mm + Ta + Tm <= 0:
        return {"d_min_mm": 0.0, "converged": True}
    d = 20.0
    for _ in range(100):
        res = check_section(d, Ma, Tm, notch, mat, finish, reliability, criterion, Mm, Ta)
        n = min(res["n_fatigue"], res["n_yield"])
        d_new = d * (n_design / n) ** (1 / 3)
        if abs(d_new - d) < 1e-6:
            break
        d = d_new
    res = check_section(d, Ma, Tm, notch, mat, finish, reliability, criterion, Mm, Ta)
    res["d_min_mm"] = d
    return res


# ---------------- beam (deflection / slope) ----------------

def _I(d):
    return math.pi * d**4 / 64


def diameter_at(segments, x):
    starts = [s[0] for s in segments]
    i = max(0, min(bisect_right(starts, x) - 1, len(segments) - 1))
    return segments[i][2]


def _interp(xs, ys, x):
    i = bisect_right(xs, x)
    if i <= 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    x0, x1 = xs[i - 1], xs[i]
    return ys[i - 1] if x1 == x0 else ys[i - 1] + (ys[i] - ys[i - 1]) * (x - x0) / (x1 - x0)


def beam_response(segments, E, xA, xB, loads, n=1500):
    """Stepped shaft on two simple supports at xA, xB.

    segments: [(x0, x1, d), ...] contiguous from 0 to L.
    loads   : [(x, Fy, Fz), ...] point forces in N.
    Returns per-plane moment, slope, deflection and support reactions.
    """
    L = segments[-1][1]
    pts = [L * i / (n - 1) for i in range(n)]
    special = [0.0, L, xA, xB] + [p[0] for p in loads] + [s[0] for s in segments]
    xs = sorted(set(round(v, 9) for v in pts + special))
    EI = [E * _I(diameter_at(segments, x)) for x in xs]
    out = {"x": xs}
    for k, ax in ((1, "y"), (2, "z")):
        F = [(p[0], p[k]) for p in loads if p[k]]
        RB = -sum(f * (x - xA) for x, f in F) / (xB - xA)
        RA = -sum(f for _, f in F) - RB
        allp = F + [(xA, RA), (xB, RB)]
        M = [sum(f * (x - xi) for xi, f in allp if xi < x) for x in xs]
        kap = [m / ei for m, ei in zip(M, EI)]
        th0, y0 = [0.0], [0.0]
        for i in range(1, len(xs)):
            h = xs[i] - xs[i - 1]
            th0.append(th0[-1] + 0.5 * h * (kap[i] + kap[i - 1]))
        for i in range(1, len(xs)):
            h = xs[i] - xs[i - 1]
            y0.append(y0[-1] + 0.5 * h * (th0[i] + th0[i - 1]))
        yA, yB = _interp(xs, y0, xA), _interp(xs, y0, xB)
        C1 = -(yB - yA) / (xB - xA)
        C2 = -yA - C1 * xA
        out[ax] = {"M": M, "slope": [t + C1 for t in th0],
                   "defl": [v + C1 * x + C2 for v, x in zip(y0, xs)], "RA": RA, "RB": RB}
    return out


def beam_at(beam, key, x):
    """Resultant magnitude of 'M', 'slope' or 'defl' at x (both planes)."""
    xs = beam["x"]
    return math.hypot(_interp(xs, beam["y"][key], x), _interp(xs, beam["z"][key], x))


def rayleigh_critical_speed(segments, E, density, xA, xB, point_masses=(), n_lumps=80):
    """First critical speed [rpm] by Rayleigh's method with gravity deflections (Shigley eq. 7-23)."""
    L = segments[-1][1]
    masses = []
    for i in range(n_lumps):
        a, b = L * i / n_lumps, L * (i + 1) / n_lumps
        xm = 0.5 * (a + b)
        d = diameter_at(segments, xm)
        masses.append((xm, density * math.pi / 4 * d**2 * (b - a) * 1e-9))
    masses += list(point_masses)
    beam = beam_response(segments, E, xA, xB, [(x, -m * G * 1e-3, 0.0) for x, m in masses])
    num = den = 0.0
    for x, m in masses:
        y = abs(_interp(beam["x"], beam["y"]["defl"], x))
        num += m * y
        den += m * y * y
    if den == 0:
        return math.inf
    omega = math.sqrt(G * num / den)          # rad/s (G in mm/s^2, y in mm)
    return omega * 60 / (2 * math.pi)
