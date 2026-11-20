"""Linear eigen-buckling of a slender member (v0.39 P2b): POST /solve/buckling_beam.

A beam-column finite element: Euler-Bernoulli bending stiffness plus the geometric (initial-stress) stiffness
from a unit axial force. The eigenproblem (K + lambda*Kg) phi = 0 gives the load factor lambda: the axial load
that buckles the member, times the load it was given.

Support conditions are the four textbook ones at each end (fixed, pinned, free, guided), so all four Euler
column cases are benchmarked here against pi^2 EI/(K L)^2 with K = 2, 1, 0.699 and 0.5.

What this is NOT: plate or shell buckling (a flat plate in compression buckles at a stress from Timoshenko &
Gere Table 9-1, which needs a plate element this module does not have), lateral-torsional buckling of an
I-beam, or a non-conservative estimate for an imperfect structure -- linear buckling ignores imperfection and
is therefore an upper bound.
"""
import math

import numpy as np

from ace_core import SolverSpec, Benchmark, Check, register

ENDPOINTS = ("fixed", "pinned", "free", "guided")
_SEL = {"type": "object"}


def _section(p):
    """Second moment of area and the section depth per bending plane."""
    kind = str(p.get("section") or "circle").lower()
    if p.get("I_mm4"):
        return float(p["I_mm4"]), float(p["I_mm4"]), None
    if kind == "rect":
        b, h = float(p["width_mm"]), float(p["height_mm"])
        return b * h ** 3 / 12.0, h * b ** 3 / 12.0, (b, h)
    d = float(p["diameter_mm"])
    i = math.pi * d ** 4 / 64.0
    return i, i, (d,)


def _end_dofs(kind):
    """Which of (w, theta) is held at an end."""
    return {"fixed": (True, True), "pinned": (True, False), "free": (False, False),
            "guided": (False, True)}[kind]


def _ke_kg(le, ei, p_axial):
    le2 = le * le
    ke = ei / le ** 3 * np.array([
        [12, 6 * le, -12, 6 * le],
        [6 * le, 4 * le2, -6 * le, 2 * le2],
        [-12, -6 * le, 12, -6 * le],
        [6 * le, 2 * le2, -6 * le, 4 * le2]], float)
    kg = p_axial / (30 * le) * np.array([
        [36, 3 * le, -36, 3 * le],
        [3 * le, 4 * le2, -3 * le, -le2],
        [-36, -3 * le, 36, -3 * le],
        [3 * le, -le2, -3 * le, 4 * le2]], float)
    return ke, kg


def _plane(ei, n_el, length, left, right, n_modes):
    le = length / n_el
    nd = 2 * (n_el + 1)
    K = np.zeros((nd, nd))
    Kg = np.zeros((nd, nd))
    for e in range(n_el):
        ke, kg = _ke_kg(le, ei, 1.0)
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(idx, idx)] += ke
        Kg[np.ix_(idx, idx)] += kg
    free = np.ones(nd, bool)
    for node, kind in ((0, left), (n_el, right)):
        w_fixed, t_fixed = _end_dofs(kind)
        if w_fixed:
            free[2 * node] = False
        if t_fixed:
            free[2 * node + 1] = False
    fr = np.flatnonzero(free)
    if len(fr) < 2:
        raise ValueError("these support conditions leave nothing free to buckle")
    A = K[np.ix_(fr, fr)]
    # A compressive axial force SOFTENS the member: (Ke - P*Kg_unit) phi = 0, so with Kg_unit assembled for a
    # unit compressive force the buckling load is the generalized eigenvalue of  K phi = lambda * Kg_unit phi.
    B = Kg[np.ix_(fr, fr)]
    try:
        from scipy.linalg import eig as _geig
        vals, vecs = _geig(A, B, right=True)
        vals, vecs = np.real(vals), np.real(vecs)
    except Exception:                           # numpy only: 1/lambda are the eigenvalues of A^-1 B
        mu, vecs = np.linalg.eig(np.linalg.solve(A, B))
        mu = np.real(mu)
        good = np.abs(mu) > 1e-12
        vals, vecs = np.where(good, 1.0 / np.where(good, mu, 1.0), 0.0), np.real(vecs)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    out = []
    for k in range(len(vals)):
        if len(out) >= n_modes:
            break
        lam, vec = float(vals[k]), vecs[:, k]
        if lam <= 1e-12:                        # a non-positive factor is not a compressive buckling load
            continue
        full = np.zeros(nd)
        full[fr] = vec
        scale = np.abs(full).max() or 1.0
        out.append({"load_factor": lam, "shape": full / scale})
    if not out:
        raise ValueError("no positive buckling load was found for these support conditions")
    return out


def run(p, progress=None):
    length = float(p["length_mm"])
    n_el = int(p.get("elements") or 40)
    left, right = str(p.get("left") or "fixed").lower(), str(p.get("right") or "free").lower()
    if left not in ENDPOINTS or right not in ENDPOINTS:
        raise ValueError(f"supports must be one of {ENDPOINTS}")
    e = float(p.get("E_MPa") or 210000.0)
    i1, i2, sec = _section(p)
    n_modes = int(p.get("n_modes") or 3)
    progress and progress("Assembling", 0.2)
    m1 = _plane(e * i1, n_el, length, left, right, n_modes)
    if abs(i1 - i2) < 1e-9 * max(1.0, i1):
        planes = [("strong axis", m1)]
        weak = m1
    else:
        m2 = _plane(e * i2, n_el, length, left, right, n_modes)
        planes = [("strong axis (I = %.4g mm4)" % i1, m1), ("weak axis (I = %.4g mm4)" % i2, m2)]
        weak = m2 if i2 < i1 else m1
    crit = float(weak[0]["load_factor"])
    k_eff = math.pi * math.sqrt(e * min(i1, i2) / crit) / length
    modes = []
    for name, ms in planes:
        modes.append({"plane": name, "load_factors_N": [round(float(m["load_factor"]), 2) for m in ms]})
    progress and progress("Done", 1.0)
    outputs = {
        "critical_load_N": round(crit, 2),
        "critical_stress_MPa": round(crit / float(p.get("area_mm2") or 1.0), 4) if p.get("area_mm2") else None,
        "effective_length_factor": round(k_eff, 4),
        "boundary": {"left": left, "right": right},
        "section": {"I_mm4": round(min(i1, i2), 4), "dimensions": sec},
        "modes": modes,
        "not_checked": ["imperfection sensitivity / buckling curve", "lateral-torsional buckling of an open section",
                        "plate or shell local buckling", "post-buckling behaviour"],
    }
    checks = ([Check("slenderness (L / r)", float(p["slenderness"]), 1e9, "-", "max",
                     "informational only: no slenderness limit is imposed here")] if p.get("slenderness") else [])
    warnings = [
        "Linear eigen-buckling: imperfections, initial bow and residual stress are not modelled, so this load "
        "is an upper bound. Apply a buckling factor (e.g. from the governing code) before design.",
        "Euler-Bernoulli beam element: shear deformation is ignored (fine for a slender member, wrong for a "
        "short stubby one).",
        "Bending only: no torsion, no lateral-torsional buckling, no plate or shell buckling.",
    ]
    return {"outputs": outputs, "checks": checks, "warnings": warnings}


def _euler(e, i, length, k):
    return math.pi ** 2 * e * i / (k * length) ** 2


_E, _D, _L = 210000.0, 20.0, 2000.0
_I = math.pi * _D ** 4 / 64.0
BASE = {"length_mm": _L, "diameter_mm": _D, "E_MPa": _E, "elements": 40}

register(SolverSpec(
    name="buckling_beam", version="1.0.0", domain="structural_stability", fidelity="L1",
    summary="Linear eigen-buckling of a slender prismatic member: critical axial load and effective length "
            "factor for clamped / pinned / free / guided ends, benchmarked against the four Euler column cases.",
    method="Beam-column finite element: Euler-Bernoulli bending stiffness + geometric stiffness from a unit "
           "axial force, generalized eigenproblem (K + lambda Kg) phi = 0, dense LAPACK",
    schema={
        "length_mm": {"type": "number", "exclusive_min": 0, "required": True},
        "diameter_mm": {"type": "number", "exclusive_min": 0},
        "section": {"type": "string", "enum": ["circle", "rect"], "default": "circle"},
        "width_mm": {"type": "number", "exclusive_min": 0},
        "height_mm": {"type": "number", "exclusive_min": 0},
        "area_mm2": {"type": "number", "exclusive_min": 0},
        "I_mm4": {"type": "number", "exclusive_min": 0},
        "E_MPa": {"type": "number", "exclusive_min": 0, "max": 1e6, "default": 210000.0},
        "slenderness": {"type": "number", "exclusive_min": 0},
        "left": {"type": "string", "enum": list(ENDPOINTS), "default": "fixed"},
        "right": {"type": "string", "enum": list(ENDPOINTS), "default": "free"},
        "elements": {"type": "integer", "min": 4, "max": 400, "default": 40},
        "n_modes": {"type": "integer", "min": 1, "max": 6, "default": 3},
    },
    run=run,
    benchmarks=[
        Benchmark("euler-fixed-free", "Euler column, fixed-free: Pcr = pi^2 EI/(2L)^2 (Timoshenko & Gere, "
                                      "Theory of Elastic Stability, 2nd ed., sec. 2.2)",
                  dict(BASE, left="fixed", right="free"),
                  {"outputs.critical_load_N": round(_euler(_E, _I, _L, 2.0), 2)}, 0.01),
        Benchmark("euler-pinned-pinned", "Euler column, pinned-pinned: Pcr = pi^2 EI/L^2 (Timoshenko & Gere "
                                         "sec. 2.2)",
                  dict(BASE, left="pinned", right="pinned"),
                  {"outputs.critical_load_N": round(_euler(_E, _I, _L, 1.0), 2)}, 0.01),
        Benchmark("euler-fixed-pinned", "Euler column, fixed-pinned: Pcr = 20.19 EI/L^2, i.e. K = 0.699 "
                                        "(Timoshenko & Gere sec. 2.2)",
                  dict(BASE, left="fixed", right="pinned"),
                  {"outputs.critical_load_N": round(_euler(_E, _I, _L, 0.6991557), 2)}, 0.01),
        Benchmark("euler-fixed-fixed", "Euler column, fixed-fixed: Pcr = 4 pi^2 EI/L^2, i.e. K = 0.5 "
                                       "(Timoshenko & Gere sec. 2.2)",
                  dict(BASE, left="fixed", right="fixed"),
                  {"outputs.critical_load_N": round(_euler(_E, _I, _L, 0.5), 2)}, 0.01),
    ],
    limitations=[
        "linear eigen-buckling: an upper bound, no imperfection, no residual stress, no buckling curve",
        "Euler-Bernoulli beam element: slender members only (no shear deformation)",
        "bending of a prismatic member only: no torsion, no lateral-torsional buckling, no plate/shell buckling",
        "constant axial force along the member (a column under its own weight or a varying axial force is not "
        "modelled)",
        "uniform section; a stepped member is not modelled",
    ],
    example=dict(BASE, left="pinned", right="pinned"),
))
