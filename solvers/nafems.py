"""NAFEMS standard benchmarks (V&V of the ACE finite elements) and thin/thick-walled pressure vessels.

Added 18 Oct 2026 from Arya's "Idris Core master specification" (section 3 V&V, section 4A vessels):
only the parts that are correct and cheap to run are used.

- NAFEMS LE1 (The Standard NAFEMS Benchmarks, R0015): elliptic membrane, plane stress, target
  σ_yy = 92.7 MPa at point D (2.0, 0). Solved with the same Q4 incompatible-modes element as fea_plane.
- Thin-walled vessel (ASME BPVC-style membrane stresses): σ_h = p·D/(2t), σ_l = p·D/(4t), D = mean diameter.
  Checked against the exact Lamé thick-cylinder solution (Timoshenko & Goodier, Theory of Elasticity),
  which is also returned so the engineer sees the thin-wall error.
"""
import math
import numpy as np
from ace_core import SolverSpec, Benchmark, Check, register
from physics.fem2d import solve

LE1_TARGET = 92.7      # MPa, NAFEMS LE1 reference value at D


def le1_model(nt, nr, E=210e3, nu=0.3, t=0.1, p=10.0):
    """Quarter elliptic membrane: inner ellipse (x/2)²+(y/1)²=1, outer (x/3.25)²+(y/2.75)²=1 (m, MPa as in
    the NAFEMS definition; units are consistent), uniform outward pressure p on the outer edge, symmetry
    on x = 0 and y = 0. Mapped mesh: lines join equal parametric angles on both ellipses."""
    nodes = []
    for i in range(nt + 1):
        th = math.pi / 2 * i / nt
        pi_ = np.array([2.0 * math.cos(th), 1.0 * math.sin(th)])
        po = np.array([3.25 * math.cos(th), 2.75 * math.sin(th)])
        for j in range(nr + 1):
            nodes.append(pi_ + (po - pi_) * j / nr)
    nodes = np.array(nodes)
    idx = lambda i, j: i * (nr + 1) + j
    elems = [[idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)] for i in range(nt) for j in range(nr)]
    e0 = nodes[elems[0]]
    area = 0.5 * sum(e0[k, 0] * e0[(k + 1) % 4, 1] - e0[(k + 1) % 4, 0] * e0[k, 1] for k in range(4))
    if area < 0:
        elems = [e[::-1] for e in elems]
    F = np.zeros((len(nodes), 2))
    for i in range(nt):
        a, b = idx(i, nr), idx(i + 1, nr)
        d = nodes[b] - nodes[a]
        L = float(np.linalg.norm(d))
        n = np.array([d[1], -d[0]]) / L
        if np.dot(n, nodes[a] + nodes[b]) < 0:
            n = -n
        F[a] += 0.5 * p * t * L * n
        F[b] += 0.5 * p * t * L * n
    fixed = [(idx(0, j), 1) for j in range(nr + 1)] + [(idx(nt, j), 0) for j in range(nr + 1)]
    loads = [(k, F[k, 0], F[k, 1]) for k in range(len(nodes)) if np.any(F[k])]
    r = solve(nodes, np.array(elems), E, nu, thickness=t, fixed=fixed, loads=loads)
    return float(r["stress"][idx(0, 0), 1]), len(nodes)


def _run_le1(p):
    n = {"coarse": (12, 6), "medium": (24, 12), "fine": (48, 24)}[p["mesh"]]
    s, nn = le1_model(*n)
    return {"outputs": {"sigma_yy_D_MPa": s, "reference_MPa": LE1_TARGET, "error_pct": 100 * (s / LE1_TARGET - 1),
                        "mesh": {"elements": n[0] * n[1], "nodes": nn, "type": "Q4 incompatible modes"}},
            "warnings": ["NAFEMS reference values: The Standard NAFEMS Benchmarks (R0015), test LE1"]}


def vessel(p_mpa, d_mean_mm, t_mm):
    """Thin-wall membrane stresses and the exact Lamé stresses (inner fibre) of a closed cylinder."""
    sh_thin = p_mpa * d_mean_mm / (2 * t_mm)
    sl_thin = p_mpa * d_mean_mm / (4 * t_mm)
    ri, ro = 0.5 * d_mean_mm - 0.5 * t_mm, 0.5 * d_mean_mm + 0.5 * t_mm
    k = ro ** 2 / (ro ** 2 - ri ** 2)
    sh_lame_in = p_mpa * ri ** 2 / (ro ** 2 - ri ** 2) * (1 + ro ** 2 / ri ** 2)
    sh_lame_mean = p_mpa * ri ** 2 / (ro ** 2 - ri ** 2) * (1 + ro ** 2 / (0.5 * d_mean_mm) ** 2)
    sl_lame = p_mpa * ri ** 2 / (ro ** 2 - ri ** 2)
    return {"hoop_thin_MPa": sh_thin, "longitudinal_thin_MPa": sl_thin,
            "hoop_lame_inner_MPa": sh_lame_in, "hoop_lame_mean_radius_MPa": sh_lame_mean,
            "longitudinal_lame_MPa": sl_lame, "radial_inner_MPa": -p_mpa,
            "thin_wall_error_at_inner_pct": 100 * (sh_thin / sh_lame_in - 1), "t_over_D": t_mm / d_mean_mm,
            "thin_wall_valid": t_mm / d_mean_mm <= 0.05}


def _run_vessel(p):
    r = vessel(p["pressure_mpa"], p["mean_diameter_mm"], p["wall_mm"])
    sy, fos = p.get("yield_mpa"), p["factor_of_safety"]
    checks, warns = [], []
    if not r["thin_wall_valid"]:
        warns.append("t/D > 0.05: use the Lamé (thick-wall) stresses, the thin-wall formula is unconservative")
    if sy:
        vm = math.sqrt(0.5 * ((r["hoop_lame_inner_MPa"] - r["longitudinal_lame_MPa"]) ** 2
                              + (r["longitudinal_lame_MPa"] - r["radial_inner_MPa"]) ** 2
                              + (r["radial_inner_MPa"] - r["hoop_lame_inner_MPa"]) ** 2))
        ms = sy / (vm * fos) - 1
        r.update(von_mises_inner_MPa=vm, margin_of_safety=ms)
        checks.append(Check("margin of safety (yield, Lamé inner fibre)", ms, 0.0, "-", "min",
                            f"MS = S_y/(σ_vm·FoS) − 1, FoS = {fos:g}"))
        if ms < 0:
            warns.append(f"Margin of safety {ms:.2f} < 0: the wall is too thin for FoS {fos:g}")
    return {"outputs": r, "checks": checks, "warnings": warns}


register(SolverSpec(
    name="nafems_le1", version="0.1.0", domain="structural", fidelity="L2",
    summary="NAFEMS LE1 elliptic membrane: verification of the ACE plane-stress element against the NAFEMS "
            "reference σ_yy = 92.7 MPa at point D.",
    method="Q4 incompatible-modes plane stress (same element as fea_plane), mapped mesh, nodal-averaged stress",
    schema={"mesh": {"type": "string", "enum": ["coarse", "medium", "fine"], "default": "fine"}},
    run=_run_le1,
    benchmarks=[Benchmark("nafems-le1-fine", "The Standard NAFEMS Benchmarks (R0015), LE1: σ_yy at D = 92.7 MPa",
                          {"mesh": "fine"}, {"outputs.sigma_yy_D_MPa": LE1_TARGET}, 0.02)],
    limitations=["verification case only (fixed geometry)", "coarse mesh is 6 % low: use fine for the check"],
    example={"mesh": "fine"},
))

register(SolverSpec(
    name="pressure_vessel", version="0.1.0", domain="structural", fidelity="L0",
    summary="Closed cylindrical pressure vessel / pipe: thin-wall hoop and longitudinal stress (p·D/2t, p·D/4t) "
            "with the exact Lamé thick-wall stresses and a NASA-style margin of safety.",
    method="Membrane theory (ASME BPVC-style thin wall, D = mean diameter); Lamé exact solution for the check; "
           "MS = S_y/(σ_vm·FoS) − 1",
    schema={"pressure_mpa": {"type": "number", "exclusive_min": 0, "default": 1.0},
            "mean_diameter_mm": {"type": "number", "exclusive_min": 0, "default": 100.0},
            "wall_mm": {"type": "number", "exclusive_min": 0, "default": 2.0},
            "yield_mpa": {"type": "number", "exclusive_min": 0},
            "factor_of_safety": {"type": "number", "min": 1.0, "max": 5.0, "default": 1.5}},
    run=_run_vessel,
    benchmarks=[Benchmark(
        "lame-thin-wall-limit",
        "Lamé exact thick-cylinder solution (Timoshenko & Goodier): for a thin wall the maximum (inner-fibre) "
        "hoop stress equals the membrane value p·D_mean/(2t). D/t = 50, p = 1 MPa → 25.0 MPa both (Lamé 25.01)",
        {"pressure_mpa": 1.0, "mean_diameter_mm": 100.0, "wall_mm": 2.0},
        {"outputs.hoop_thin_MPa": 25.0, "outputs.hoop_lame_inner_MPa": 25.0}, 0.002)],
    limitations=["closed-end cylinder, internal pressure only, no openings, no external loads",
                 "no ASME code allowable (S) or weld efficiency (E) applied: screening only"],
    example={"pressure_mpa": 1.0, "mean_diameter_mm": 100.0, "wall_mm": 2.0, "yield_mpa": 250.0},
))
