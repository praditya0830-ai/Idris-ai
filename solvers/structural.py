"""Structural FEA solvers (L2) registered in the ACE framework."""
import math
import numpy as np
from ace_core import SolverSpec, Benchmark, Check, register
from physics.fem2d import (solve, mesh_rectangle, mesh_plate_with_hole, edge_traction_loads,
                           HAVE_SCIPY, MAX_DENSE_DOFS)

MESH = {"coarse": 1, "medium": 2, "fine": 3}
MAX_NODES_CUSTOM = 3000


def _summary(nodes, elems, r):
    vm = r["von_mises"]
    disp = np.linalg.norm(r["u"], axis=1)
    i, j = int(np.argmax(vm)), int(np.argmax(disp))
    return {
        "max_von_mises_MPa": float(vm[i]), "max_von_mises_at_mm": nodes[i].tolist(),
        "max_principal_MPa": float(r["s1"].max()), "min_principal_MPa": float(r["s2"].min()),
        "max_displacement_mm": float(disp[j]), "max_displacement_at_mm": nodes[j].tolist(),
        "nodes": len(nodes), "elements": len(elems), "dofs": r["ndof"],
    }


def _field(nodes, elems, r):
    """Per-node results for the 3D viewport (colour map + deformed shape)."""
    return {"nodes_mm": nodes.tolist(), "elements": elems.tolist(),
            "displacement_mm": r["u"].tolist(), "von_mises_MPa": r["von_mises"].tolist(),
            "sigma_x_MPa": r["stress"][:, 0].tolist(), "sigma_y_MPa": r["stress"][:, 1].tolist(),
            "tau_xy_MPa": r["stress"][:, 2].tolist()}


def _cantilever(p, level):
    L, H, P = p["length_mm"], p["height_mm"], p["load_N"]
    ny = 2 * level + 2
    nx = max(4, round(ny * L / H))
    nodes, elems = mesh_rectangle(L, H, nx, ny)
    fixed = [(k, d) for k, (x, y) in enumerate(nodes) if x == 0 for d in (0, 1)]
    right = [k for k, (x, y) in enumerate(nodes) if abs(x - L) < 1e-9]
    loads = [(k, 0.0, -P / len(right)) for k in right]
    r = solve(nodes, elems, p["E_MPa"], p["nu"], p["thickness_mm"], fixed, loads)
    t = p["thickness_mm"]
    I, G = t * H**3 / 12, p["E_MPa"] / (2 * (1 + p["nu"]))
    theory = {"tip_deflection_beam_theory_mm": P * L**3 / (3 * p["E_MPa"] * I) + P * L / (5 / 6 * G * H * t),
              "root_bending_stress_beam_theory_MPa": P * L * (H / 2) / I}
    extra = {"tip_deflection_mm": float(-r["u"][right, 1].mean()), **theory}
    return nodes, elems, r, extra


def _plate_hole(p, level):
    a, B, s0 = p["hole_radius_mm"], p["half_width_mm"], p["stress_MPa"]
    nodes, elems, sets = mesh_plate_with_hole(a, B, 8 * level, 9 * level + 1, 1.12 if level > 1 else 1.2)
    fixed = [(k, 0) for k in sets["x0"]] + [(k, 1) for k in sets["y0"]]
    loads = edge_traction_loads(nodes, sets["right"], s0, 0.0, p["thickness_mm"])
    r = solve(nodes, elems, p["E_MPa"], p["nu"], p["thickness_mm"], fixed, loads)
    k = sets["hole_top"]
    extra = {"sigma_x_hole_edge_MPa": float(r["stress"][k, 0]),
             "Kt_gross": float(r["stress"][k, 0] / s0),
             "sigma_y_at_hole_side_MPa": float(r["stress"][sets["y0"][0], 1]),
             "model": "quarter model with symmetry (x=0, y=0)"}
    return nodes, elems, r, extra


def _custom(p):
    nodes = np.array(p["nodes_mm"], float)
    elems = np.array(p["elements"], int)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes_mm must be a list of [x, y]")
    if len(nodes) > MAX_NODES_CUSTOM:
        raise ValueError(f"at most {MAX_NODES_CUSTOM} nodes")
    if elems.ndim != 2 or elems.shape[1] != 4 or elems.min() < 0 or elems.max() >= len(nodes):
        raise ValueError("elements must be [n1,n2,n3,n4] with valid node indices (counter-clockwise)")
    fixed = []
    for s in p["supports"]:
        if not 0 <= s["node"] < len(nodes):
            raise ValueError(f"support node {s['node']} out of range")
        if s["x"]:
            fixed.append((s["node"], 0))
        if s["y"]:
            fixed.append((s["node"], 1))
    loads = []
    for l in p["loads"]:
        if not 0 <= l["node"] < len(nodes):
            raise ValueError(f"load node {l['node']} out of range")
        loads.append((l["node"], l["Fx_N"], l["Fy_N"]))
    r = solve(nodes, elems, p["E_MPa"], p["nu"], p["thickness_mm"], fixed, loads, p["plane"])
    return nodes, elems, r, {}


def _run(p):
    case = p["case"]
    warnings, its = [], []
    if case == "custom":
        for k in ("nodes_mm", "elements", "supports", "loads"):
            if not p.get(k):
                return {"ok": False, "errors": [f"custom case needs '{k}'"]}
        nodes, elems, r, extra = _custom(p)
        warnings.append("custom mesh: no automatic mesh-convergence study; refine and compare yourself")
    else:
        if case == "cantilever":
            need = ("length_mm", "height_mm", "load_N")
        else:
            need = ("hole_radius_mm", "half_width_mm", "stress_MPa")
        miss = [k for k in need if p.get(k) is None]
        if miss:
            return {"ok": False, "errors": [f"case '{case}' needs {miss}"]}
        if case == "plate_with_hole" and not p["hole_radius_mm"] < 0.8 * p["half_width_mm"]:
            return {"ok": False, "errors": ["hole_radius_mm must be < 0.8 * half_width_mm"]}
        build = _cantilever if case == "cantilever" else _plate_hole
        lvl = MESH[p["mesh"]]
        nodes, elems, r, extra = build(p, lvl)
        # solution verification: re-solve on a neighbouring mesh level and compare peak stress
        lvl2 = lvl - 1 if lvl > 1 else lvl + 1
        n2, e2, r2, x2 = build(p, lvl2)
        a, b = r["von_mises"].max(), r2["von_mises"].max()
        change = abs(b - a) / max(a, b) if max(a, b) else 0.0
        extra["mesh_convergence"] = {"peak_von_mises_this_mesh_MPa": float(a),
                                     "peak_von_mises_compare_mesh_MPa": float(b),
                                     "compare_mesh": [k for k, v in MESH.items() if v == lvl2][0],
                                     "compare_mesh_nodes": len(n2),
                                     "relative_change": float(change)}
        its.append({"iter": 1, "action": f"solve {p['mesh']} mesh", "nodes": len(nodes)})
        its.append({"iter": 2, "action": "re-solve neighbouring mesh for convergence check", "nodes": len(n2),
                    "peak_change_pct": round(100 * change, 2)})
        if change > 0.05:
            warnings.append(f"peak stress changed {100 * change:.1f}% between mesh levels: "
                            "result not mesh-converged, use mesh='fine'")
    out = _summary(nodes, elems, r)
    out.update(extra)
    if p["include_field"]:
        out["field"] = _field(nodes, elems, r)
    checks = []
    if p.get("yield_MPa"):
        checks.append(Check("von Mises vs yield / design factor", out["max_von_mises_MPa"],
                            p["yield_MPa"] / p["design_factor"], "MPa", "max"))
        checks.append(Check("safety factor on yield", p["yield_MPa"] / max(out["max_von_mises_MPa"], 1e-12),
                            p["design_factor"], "-", "min"))
    if p.get("max_displacement_mm"):
        checks.append(Check("max displacement", out["max_displacement_mm"], p["max_displacement_mm"], "mm", "max"))
    if not checks:
        warnings.append("no yield_MPa given: stresses reported without a pass/fail verdict")
    return {"outputs": out, "checks": checks, "warnings": warnings, "iterations": its}


SCHEMA = {
    "case": {"type": "string", "enum": ["cantilever", "plate_with_hole", "custom"], "required": True},
    "E_MPa": {"type": "number", "exclusive_min": 0, "max": 1.2e6, "default": 205000.0, "unit": "MPa"},
    "nu": {"type": "number", "min": 0, "max": 0.49, "default": 0.29},
    "thickness_mm": {"type": "number", "exclusive_min": 0, "max": 1e4, "default": 1.0, "unit": "mm"},
    "plane": {"type": "string", "enum": ["stress", "strain"], "default": "stress"},
    "mesh": {"type": "string", "enum": list(MESH), "default": "fine"},
    "yield_MPa": {"type": "number", "exclusive_min": 0, "max": 5000, "unit": "MPa"},
    "design_factor": {"type": "number", "min": 1, "max": 10, "default": 1.5},
    "max_displacement_mm": {"type": "number", "exclusive_min": 0, "unit": "mm"},
    "include_field": {"type": "boolean", "default": True},
    # cantilever
    "length_mm": {"type": "number", "exclusive_min": 0, "max": 1e5, "unit": "mm"},
    "height_mm": {"type": "number", "exclusive_min": 0, "max": 1e5, "unit": "mm"},
    "load_N": {"type": "number", "min": -1e9, "max": 1e9, "unit": "N", "desc": "tip load, + = downward"},
    # plate with hole
    "hole_radius_mm": {"type": "number", "exclusive_min": 0, "max": 1e5, "unit": "mm"},
    "half_width_mm": {"type": "number", "exclusive_min": 0, "max": 1e5, "unit": "mm"},
    "stress_MPa": {"type": "number", "min": -1e5, "max": 1e5, "unit": "MPa", "desc": "remote tension"},
    # custom mesh
    "nodes_mm": {"type": "array", "max_items": MAX_NODES_CUSTOM,
                 "items_rule": {"type": "array", "items_type": "number", "min_items": 2, "max_items": 2}},
    "elements": {"type": "array", "max_items": 3 * MAX_NODES_CUSTOM,
                 "items_rule": {"type": "array", "items_type": "integer", "min_items": 4, "max_items": 4}},
    "supports": {"type": "array", "max_items": MAX_NODES_CUSTOM,
                 "items": {"node": {"type": "integer", "min": 0, "required": True},
                           "x": {"type": "boolean", "default": True},
                           "y": {"type": "boolean", "default": True}}},
    "loads": {"type": "array", "max_items": MAX_NODES_CUSTOM,
              "items": {"node": {"type": "integer", "min": 0, "required": True},
                        "Fx_N": {"type": "number", "default": 0.0},
                        "Fy_N": {"type": "number", "default": 0.0}}},
}

register(SolverSpec(
    name="fea_plane", version="0.1.0", domain="structural_fea", fidelity="L2",
    summary=("2D linear-elastic finite-element stress analysis (plane stress/strain): "
             "parametric cantilever, plate with hole, or a custom quad mesh."),
    method=("FEM, 4-node quads with incompatible modes (QM6), 2x2 Gauss, nodal-averaged "
            "stress recovery, automatic mesh-refinement convergence check"),
    schema=SCHEMA, run=_run,
    benchmarks=[
        Benchmark("cantilever-timoshenko", "Timoshenko beam theory: δ = PL³/3EI + PL/(κGA), κ = 5/6, L/H = 10",
                  {"case": "cantilever", "length_mm": 100, "height_mm": 10, "load_N": 100,
                   "E_MPa": 200000, "nu": 0.3, "include_field": False},
                  {"outputs.tip_deflection_mm": 100 * 100**3 / (3 * 200000 * 10**3 / 12)
                   + 100 * 100 / (5 / 6 * 200000 / 2.6 * 10)}, 0.02),
        Benchmark("kirsch-hole-Kt", "Kirsch (1898) infinite plate with circular hole: Kt = 3 (d/W = 0.05)",
                  {"case": "plate_with_hole", "hole_radius_mm": 1, "half_width_mm": 20, "stress_MPa": 1,
                   "mesh": "fine", "include_field": False},
                  {"outputs.Kt_gross": 3.0}, 0.025),
        Benchmark("kirsch-hole-side", "Kirsch: σ_y = −σ at the hole side (θ = 0)",
                  {"case": "plate_with_hole", "hole_radius_mm": 1, "half_width_mm": 20, "stress_MPa": 1,
                   "mesh": "fine", "include_field": False},
                  {"outputs.sigma_y_at_hole_side_MPa": -1.0}, 0.03),
    ],
    limitations=["2D only (3D solid FEA is roadmap L2b)", "linear elastic, small displacement",
                 "no contact, no plasticity, no dynamics/modal yet",
                 f"custom meshes up to {MAX_NODES_CUSTOM} nodes"
                 + ("" if HAVE_SCIPY else f" (dense solver: ~{MAX_DENSE_DOFS // 2} nodes)")],
    example={"case": "plate_with_hole", "hole_radius_mm": 5, "half_width_mm": 50,
             "stress_MPa": 100, "yield_MPa": 655},
))
