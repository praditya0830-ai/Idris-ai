"""Shell FEA for thin-walled and welded sheet structures (v0.33). POST /weldment-check (snow plow blade family).

Two solvers:
  fea_shell_cases   textbook plate and shell problems, run by name: the proof that the element is right
  fea_plow_blade    the parametric snow plow blade under loads the engineer gives
"""
import math

import numpy as np

from ace_core import SolverSpec, Benchmark, Check, register
from physics import plow_shell as ps
from physics import shell_fea as sf


# ------------------------------------------------------------------ textbook cases
def _plate(n, clamped):
    a, t, E, nu, q = 1000.0, 10.0, 200000.0, 0.3, 0.01
    pts = np.linspace(0, a, n + 1)
    nodes, quads, nid = sf.grid(pts, pts, lambda u, v: (u, v, 0.0))
    m = sf.ShellModel(nodes, quads, t, E, nu)
    for i in range(n + 1):
        for j in range(n + 1):
            ex, ey = i in (0, n), j in (0, n)
            if not (ex or ey):
                continue
            k = nid(i, j)
            if clamped:
                m.fix([k])
            else:                                   # hard simple support: w = 0 and no rotation along the edge
                m.fix([k], (0, 1, 2, 5))
                if ex:
                    m.fix([k], (3,))
                if ey:
                    m.fix([k], (4,))
    m.add_pressure(range(len(quads)), q, (0, 0, 1))
    r = m.solve()
    D = E * t ** 3 / (12 * (1 - nu * nu))
    centre = [e for e, qd in enumerate(quads) if nid(n // 2, n // 2) in qd]
    return {"deflection_coefficient": float(r["displacement"][nid(n // 2, n // 2), 2] / (q * a ** 4 / D)),
            "centre_moment_coefficient": float(r["von_mises"][centre].mean() * t * t / 6 / (q * a * a)),
            "equilibrium_error": r["equilibrium_error"], "elements": len(quads)}


def _scordelis(n):
    R, L, t, E, nu, phi = 25.0, 50.0, 0.25, 4.32e8, 0.0, math.radians(40)
    nodes, quads, nid = sf.grid(np.linspace(0, L, n + 1), np.linspace(-phi, phi, n + 1),
                                lambda x, a: (x, R * math.sin(a), R * math.cos(a)))
    m = sf.ShellModel(nodes, quads, t, E, nu)
    for j in range(n + 1):
        m.fix([nid(0, j), nid(n, j)], (1, 2))       # rigid end diaphragms
    m.fix([nid(n // 2, n // 2)], (0,))
    m.add_pressure(range(len(quads)), 90.0, (0, 0, -1))
    r = m.solve()
    return {"free_edge_deflection": float(-r["displacement"][nid(n // 2, 0), 2]),
            "equilibrium_error": r["equilibrium_error"], "elements": len(quads)}


def _tbeam(n):
    L, bf, tf, hw, tw, E, nu, P = 1000.0, 100.0, 5.0, 50.0, 5.0, 210000.0, 0.3, 1000.0
    xs = np.linspace(0, L, 5 * n + 1)
    fl = sf.grid(xs, np.linspace(-bf / 2, bf / 2, n + 1), lambda x, y: (x, y, 0.0))[:2] + (tf,)
    wb = sf.grid(xs, np.linspace(0, -hw, n // 2 + 1), lambda x, z: (x, 0.0, z))[:2] + (tw,)
    nodes, quads, th, _ = sf.merge([fl, wb])
    m = sf.ShellModel(nodes, quads, th, E, nu)
    m.fix(np.where(np.abs(nodes[:, 0]) < 1e-9)[0])
    tip = np.where((np.abs(nodes[:, 0] - L) < 1e-9) & (np.abs(nodes[:, 1]) < 1e-9))[0]
    m.add_force(tip, (0, 0, -P))
    r = m.solve()
    return {"tip_deflection_mm": float(-r["displacement"][tip, 2].mean()),
            "equilibrium_error": r["equilibrium_error"], "elements": len(quads)}


def _tbeam_reference():
    L, bf, tf, hw, tw, E, nu, P = 1000.0, 100.0, 5.0, 50.0, 5.0, 210000.0, 0.3, 1000.0
    Af, Aw = bf * tf, hw * tw
    zc = Aw * (-hw / 2) / (Af + Aw)
    I = bf * tf ** 3 / 12 + Af * zc ** 2 + tw * hw ** 3 / 12 + Aw * (-hw / 2 - zc) ** 2
    return P * L ** 3 / (3 * E * I) + P * L / (E / (2 * (1 + nu)) * Aw)


CASES = {"plate_simply_supported": lambda n: _plate(n, False), "plate_clamped": lambda n: _plate(n, True),
         "scordelis_lo_roof": _scordelis, "welded_t_beam": _tbeam}


def _run_case(p, progress):
    n = int(p.get("n") or 16)
    n += n % 2
    o = CASES[p["case"]](n)
    o["case"], o["n"] = p["case"], n
    return {"outputs": o, "checks": [Check("force equilibrium error", o["equilibrium_error"], 1e-6, "-", "max",
                                           "solver self-check")], "warnings": []}


register(SolverSpec(
    name="fea_shell_cases", version="0.1.0", domain="structural_fea", fidelity="L2",
    summary="Textbook plate and shell problems solved with the IDRIS flat shell element: the evidence behind every "
            "welded-sheet check.",
    method="4-node flat shell: MITC4 Mindlin-Reissner plate (Bathe & Dvorkin 1985) + Q4 membrane with centre-integrated "
           "shear + rigid-body-invariant drilling penalty; sparse direct solve",
    schema={"case": {"type": "string", "enum": list(CASES), "default": "plate_simply_supported"},
            "n": {"type": "integer", "min": 4, "max": 48, "default": 16}},
    run=_run_case, progress=True,
    benchmarks=[
        Benchmark("plate-simply-supported-uniform",
                  "Square plate a/t = 100, simply supported, uniform pressure: w = 0.00406 q a^4 / D and "
                  "M = 0.0479 q a^2 at the centre (Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, "
                  "Table 8, nu = 0.3)",
                  {"case": "plate_simply_supported", "n": 16},
                  {"outputs.deflection_coefficient": 0.00406, "outputs.centre_moment_coefficient": 0.0479}, 0.01),
        Benchmark("plate-clamped-uniform",
                  "Square plate a/t = 100, all edges clamped, uniform pressure: w = 0.00126 q a^4 / D and "
                  "M = 0.0231 q a^2 at the centre (same book, Table 35)",
                  {"case": "plate_clamped", "n": 16},
                  {"outputs.deflection_coefficient": 0.00126, "outputs.centre_moment_coefficient": 0.0231}, 0.015),
        Benchmark("scordelis-lo-roof",
                  "Cylindrical roof R 25, L 50, t 0.25, 40 deg half angle, rigid end diaphragms, self-weight 90 per "
                  "area: vertical deflection at the middle of the free edge 0.3024 (MacNeal & Harder 1985, "
                  "A proposed standard set of problems to test finite element accuracy)",
                  {"case": "scordelis_lo_roof", "n": 16}, {"outputs.free_edge_deflection": 0.3024}, 0.02),
        Benchmark("welded-t-beam-cantilever",
                  "Cantilever T-beam from two plates sharing a weld line (flange 100x5, web 50x5, L 1000, 1 kN at the "
                  "tip): beam theory PL^3/3EI + PL/(G A_web). Checks the plate-to-rib joint.",
                  {"case": "welded_t_beam", "n": 8}, {"outputs.tip_deflection_mm": _tbeam_reference()}, 0.02),
    ],
    limitations=["linear elastic, small displacement, static", "planar quadrilateral elements only",
                 "a weld line is a rigid joint: weld size and weld stress are not modelled"],
    example={"case": "plate_simply_supported", "n": 16},
))


# ------------------------------------------------------------------ snow plow blade
def _run_plow(p, progress):
    try:
        o = ps.analyze(p, progress)
    except ValueError as e:
        return {"ok": False, "errors": [str(e)]}
    df = float(p.get("design_factor") or 2.0)
    o["design_factor"] = df
    o["margin_of_safety"] = o["min_safety_factor"] / df - 1
    o["reaction_x_N"] = o["reaction_N"][0]
    Sy = o["yield_mpa"]
    warnings = [
        "Loads are the engineer's: IDRIS has no validated snow, wind or impact load model.",
        f"Supports: rear edge of ribs {o['supports']['mount_ribs']} fully clamped ({o['supports']['source']}). A real "
        "push frame is neither rigid nor attached along the whole edge.",
        "The cutting edge bar is left out of the stiffness (safe side for the sheet); welds are rigid joints.",
        "Peak stress sits at clamped edges, load edges and weld lines and rises with mesh refinement: judge the sheet "
        "by von_mises_p99_MPa together with the peak.",
    ]
    if o["max_von_mises_MPa"] > Sy:
        warnings.append(f"Peak stress {o['max_von_mises_MPa']:.0f} MPa is above yield {Sy:g} MPa: this linear result "
                        "no longer describes the real blade there (it would yield).")
    t_sheet = o["dimensions_mm"]["t"]
    if o["max_displacement_mm"] > t_sheet:
        warnings.append(f"Deflection {o['max_displacement_mm']:.1f} mm is larger than the sheet thickness "
                        f"{t_sheet:g} mm: small-displacement theory is at its limit; buckling is not checked.")
    mc = o.get("mesh_convergence")
    if mc and not mc["converged"]:
        warnings.append("Not mesh-converged at this element size (change over 5 % on refinement): treat as screening "
                        "or send a smaller element_size_mm.")
    checks = [Check("blade safety factor on yield (peak von Mises)", o["min_safety_factor"], df, "-", "min"),
              Check("blade safety factor on yield (99th percentile von Mises)", o["safety_factor_p99"], df, "-", "min",
                    "less sensitive to singular points than the peak"),
              Check("force equilibrium error", o["equilibrium_error"], 1e-6, "-", "max", "solver self-check")]
    if p.get("max_deflection_mm"):
        checks.append(Check("max deflection", o["max_displacement_mm"], float(p["max_deflection_mm"]), "mm", "max"))
    return {"outputs": o, "checks": checks, "warnings": warnings}


_num = lambda lo, hi, **kw: {"type": "number", "min": lo, "max": hi, **kw}
PLOW_EX = {"W": 2400, "pressure_kPa": 5.0, "steel": "S355", "convergence_study": False}

register(SolverSpec(
    name="fea_plow_blade", version="0.1.0", domain="structural_fea", fidelity="L2",
    summary="Shell FEA of the parametric welded snow plow blade (moldboard + ribs) under the engineer's pressure and "
            "edge loads: von Mises, deflection, safety factor on yield, mesh-convergence check.",
    method="Mid-surface shell model built from the blade parameters; 4-node flat shell elements (see fea_shell_cases "
           "for the element benchmarks); ribs welded to the moldboard by shared nodes; sparse direct solve",
    schema={
        **{k: _num(lo, hi, **({"default": dv} if dv is not None else {})) for k, (_, lo, hi, dv) in ps.PLOW_INPUTS.items()},
        "steel": {"type": "string", "enum": list(ps.STEELS), "default": "S355"},
        "pressure_kPa": _num(0, 500),
        "edge_force_N": {"type": "number", "min": 0, "max": 1e6},
        "edge_zone_mm": {"type": "array", "items_type": "number", "min_items": 2, "max_items": 2},
        "mount_ribs": {"type": "array", "items_type": "integer", "min_items": 1, "max_items": 9},
        "mount_z_mm": {"type": "array", "items_type": "number", "min_items": 2, "max_items": 2},
        "element_size_mm": _num(10, 200),
        "design_factor": _num(1.0, 10.0, default=2.0),
        "max_deflection_mm": {"type": "number", "exclusive_min": 0},
        "convergence_study": {"type": "boolean", "default": True},
    },
    run=_run_plow, progress=True,
    benchmarks=[
        Benchmark("plow-pressure-resultant",
                  "Blade 2400 x 900 under 5 kPa on the curved front face: the support reaction along the travel "
                  "direction equals pressure x projected area = 5e-3 x 2400 x 900 = 10800 N (exact for a pressure on "
                  "any curved surface). Checks the mesh, the load and the supports, not the stress.",
                  PLOW_EX, {"outputs.reaction_x_N": 10800.0}, 0.001),
    ],
    limitations=["linear elastic, small displacement, static; no buckling, fatigue, impact dynamics or plasticity",
                 "loads only from the engineer (pressure on the moldboard, force on the cutting edge)",
                 "supports: clamped rear edges of the chosen ribs; push frame, trip mechanism and bolts not modelled",
                 "cutting edge bar excluded from stiffness; weld lines rigid, weld stress not checked",
                 f"at most {ps.MAX_ELEMENTS} elements"],
    example=PLOW_EX,
))
