"""3D structural FEA solvers (L2): generic solid block (engine verification) and gearbox housing."""
import os
import time
import numpy as np
from ace_core import SolverSpec, Benchmark, Check, register
from physics.fem3d import box_model
from ._example import EXAMPLE_VARIANT
from physics.housing import (HOUSING_MATERIALS, find_material, build_layout, build_model,
                             bearing_loads, fixture_nodes)

MESH_DIVISIONS = {"coarse": 1, "standard": 2, "fine": 3}      # elements through the wall
REG_VM = 10.0747          # regression value, set below after first validated run
MAX_ELEMENTS = int(os.environ.get("IDRIS_FEA_MAX_ELEMENTS", "60000"))
KNOWN_HP_KEYS = {"envelope_width", "envelope_height", "envelope_depth", "wall_thickness", "gear_ratio",
                 "bore_diameter"}
FRAME_NOTE = ("Same coordinate frame as /generate-assembly: housing x∈[0,W], y∈[0,H], z∈[0,D] mm, "
              "shafts along +z at y = H/2.")


# ------------------------------------------------------------------ solid block (verification)
def _run_block(p):
    L, W, H, h = p["length_mm"], p["width_mm"], p["height_mm"], p["element_size_mm"]
    if L / h * W / h * H / h > MAX_ELEMENTS:
        return {"ok": False, "errors": [f"too many elements (> {MAX_ELEMENTS}); increase element_size_mm"]}
    m = box_model(L, W, H, h)
    x = m.nodes
    Lr = x[:, 0].max()
    end = np.where(np.abs(x[:, 0] - Lr) < 1e-9)[0]
    y, z = x[end, 1], x[end, 2]
    w = (np.where((y < 1e-9) | (np.abs(y - y.max()) < 1e-9), 0.5, 1.0)
         * np.where((z < 1e-9) | (np.abs(z - z.max()) < 1e-9), 0.5, 1.0))
    loads = np.zeros((m.n_nodes, 3))
    loads[end, 0] = p["axial_load_N"] * w / w.sum()
    loads[end, 1] = -p["tip_load_N"] * w / w.sum()
    r = m.solve(p["E_MPa"], p["nu"], np.where(x[:, 0] < 1e-9)[0], loads)
    mid = np.abs(x[:, 0] - round(Lr / 2 / m.h) * m.h) < 1e-9
    out = {"tip_deflection_y_mm": float(-r["u"][end, 1].mean()),
           "tip_displacement_x_mm": float(r["u"][end, 0].mean()),
           "sigma_x_mid_span_MPa": float(np.median(r["stress"][mid, 0])),
           "max_von_mises_MPa": float(r["von_mises"].max()),
           "elements": len(m.elems), "nodes": m.n_nodes, "cg_iterations": r["iterations"]}
    return {"outputs": out}


register(SolverSpec(
    name="fea_solid_block", version="0.1.0", domain="structural_fea", fidelity="L2",
    summary="3D FEA of a solid rectangular block clamped at x=0 with an axial and/or tip load (engine check).",
    method="3D FEM, 8-node hexahedra with incompatible modes, matrix-free PCG",
    schema={
        "length_mm": {"type": "number", "exclusive_min": 0, "max": 1e4, "required": True},
        "width_mm": {"type": "number", "exclusive_min": 0, "max": 1e4, "required": True},
        "height_mm": {"type": "number", "exclusive_min": 0, "max": 1e4, "required": True},
        "element_size_mm": {"type": "number", "exclusive_min": 0, "required": True},
        "axial_load_N": {"type": "number", "default": 0.0},
        "tip_load_N": {"type": "number", "default": 0.0, "desc": "+ = downward (-y)"},
        "E_MPa": {"type": "number", "exclusive_min": 0, "default": 200000.0},
        "nu": {"type": "number", "min": 0, "max": 0.49, "default": 0.3},
    },
    run=_run_block,
    benchmarks=[
        Benchmark("bar-tension", "Uniaxial bar: δ = FL/EA, σ = F/A (exact)",
                  {"length_mm": 100, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5,
                   "axial_load_N": 1000},
                  {"outputs.sigma_x_mid_span_MPa": 10.0, "outputs.tip_displacement_x_mm": 0.005}, 0.01),
        Benchmark("cantilever-3d-timoshenko", "Timoshenko beam: δ = PL³/3EI + PL/(κGA), L/H = 10",
                  {"length_mm": 100, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5,
                   "tip_load_N": 1000},
                  {"outputs.tip_deflection_y_mm": 1000 * 100**3 / (3 * 200000 * 10**4 / 12)
                   + 1000 * 100 / (5 / 6 * 200000 / 2.6 * 100)}, 0.02),
    ],
    limitations=["voxel (cube) elements only", "linear elastic, small displacement"],
    example={"length_mm": 100, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5, "tip_load_N": 1000},
))


# ------------------------------------------------------------------ gearbox housing
def _run_housing(p, progress):
    warnings = []
    mat = find_material(p["housing_material"])
    if mat is None:
        return {"ok": False, "errors": [f"Unknown housing material '{p['housing_material']}'. "
                                        f"Available: {', '.join(HOUSING_MATERIALS)}"]}
    E, nu, Sy, rho, note = HOUSING_MATERIALS[mat]
    if note:
        warnings.append(f"{mat}: {note}")
    hp = p.get("housing_params") or {}
    extra = sorted(set(hp) - KNOWN_HP_KEYS)
    if extra:
        warnings.append(f"housing_params keys not used by the solver yet: {extra}")
    progress("Meshing", 0.05)
    try:
        L = build_layout(p["design"], p["kinematics"], p.get("loads") or {}, p.get("shaft") or {}, hp)
    except (KeyError, ValueError, TypeError) as ex:
        return {"ok": False, "errors": [f"Cannot build the housing model from this variant: {ex}"]}
    model = build_model(L, p["mesh"])
    h = model.h
    ne = len(model.elems)
    if ne > MAX_ELEMENTS:
        smaller = [k for k, v in MESH_DIVISIONS.items() if v < MESH_DIVISIONS[p["mesh"]]]
        return {"ok": False, "errors": [
            f"Mesh '{p['mesh']}' needs {ne:,} elements but this server allows {MAX_ELEMENTS:,} "
            f"(free-tier RAM/CPU). Please choose mesh '{smaller[-1] if smaller else 'coarse'}'."]}
    if MESH_DIVISIONS[p["mesh"]] == 1:
        warnings.append("coarse mesh: one element through the wall and ~2× wall-thickness elements in-plane; "
                        "peak stresses are approximate (use 'standard' for final numbers)")
    progress("Applying loads", 0.15)
    loads, applied = bearing_loads(model, L)
    fixed = fixture_nodes(model, L, p["fixture"])
    if len(fixed) == 0:
        return {"ok": False, "errors": [f"No nodes found for fixture '{p['fixture']}'"]}
    progress("Solving", 0.2)
    t0 = time.perf_counter()
    expected_iters = 600 * MESH_DIVISIONS[p["mesh"]]

    def cg_progress(it, res):
        progress("Solving", min(0.85, 0.2 + 0.65 * it / (it + expected_iters)))

    r = model.solve(E, nu, fixed, loads, progress=cg_progress)
    solve_s = time.perf_counter() - t0
    progress("Post-processing", 0.9)
    ids, tris = model.surface()
    pos = model.nodes[ids]
    vm = r["von_mises"][ids]
    disp = r["u"][ids]
    reac = r["reactions"][fixed].sum(0)
    app = loads.sum(0)
    tot = np.abs(loads).sum() or 1.0
    balance = float(np.linalg.norm(reac + app) / tot)
    vmax = float(r["von_mises"].max())
    sf = Sy / max(vmax, 1e-12)
    # precision chosen for the viewport: 0.01 mm geometry, 0.01 MPa stress, 1e-5 mm displacement
    surface = {"positions": np.round(pos, 2).ravel(), "indices": tris.ravel(),
               "von_mises": np.round(vm, 2), "displacement": np.round(disp, 5).ravel()}
    warnings.append(FRAME_NOTE)
    warnings.append("Voxel mesh: curved bores are stair-stepped; peak stress at bore edges is approximate")
    if any(a["seat"].startswith("blind") for a in applied):
        warnings.append("Shafts without a bore in the assembly (e.g. intermediate shaft) are assumed to sit in "
                        "blind bearing seats on the inner wall face.")
    out = {
        "surface": surface, "yield_mpa": Sy, "material": mat,
        "mesh": {"elements": ne, "nodes": model.n_nodes, "type": "hex8-im (rectilinear brick)",
                 "element_size_mm": round(h, 3), "surface_triangles": int(len(tris))},
        "solve_time_s": round(solve_s, 2), "cg_iterations": r["iterations"],
        "applied_loads": applied,
        "max_von_mises_MPa": vmax, "max_displacement_mm": float(np.linalg.norm(r["u"], axis=1).max()),
        "min_safety_factor": sf, "fixture": p["fixture"],
        "model": {"envelope_mm": [L["W"], L["H"], L["D"]], "wall_mm": L["wall_mm"],
                  "shafts": [{"name": s["name"], "x_mm": round(s["x"], 2), "diameter_mm": s["d"],
                              "torque_Nm": round(s["torque"], 2)} for s in L["shafts"]],
                  "bores": [{"wall": b["wall"], "x_mm": round(b["x"], 2), "diameter_mm": b["d"]} for b in L["bores"]],
                  "gear_mesh_forces": [{"stage": me["stage"] + 1, "tangential_n": round(me["Wt"], 1),
                                        "radial_n": round(me["Wr"], 1), "z_mm": round(me["z"], 2)}
                                       for me in L["meshes"]],
                  "approx_mass_kg": float(model.evol.sum()) * 1e-9 * rho},
        "equilibrium_error": balance,
    }
    checks = [Check("housing safety factor on yield", sf, p["design_factor"], "-", "min"),
              Check("force equilibrium error", balance, 1e-3, "-", "max", "solver self-check")]
    progress("Post-processing", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


VARIANT_DESIGN = {"module_mm": {"type": "number", "exclusive_min": 0, "max": 50, "required": True},
                  "pinion_teeth": {"type": "integer", "min": 6, "max": 400, "required": True},
                  "gear_teeth": {"type": "integer", "min": 6, "max": 2000, "required": True},
                  "face_width_mm": {"type": "number", "exclusive_min": 0, "max": 1000, "required": True},
                  "stages": {"type": "integer", "min": 1, "max": 2, "default": 1},
                  "center_distance_mm": {"type": "number", "exclusive_min": 0}}
VARIANT_KIN = {"input_torque_nm": {"type": "number", "exclusive_min": 0, "max": 1e6, "required": True},
               "efficiency": {"type": "number", "min": 0.3, "max": 1, "default": 0.97}}

register(SolverSpec(
    name="fea_housing_3d", version="0.1.0", domain="structural_fea", fidelity="L2", progress=True,
    summary=("3D static FEA of a gearbox housing under bearing loads computed from the gear forces "
             "of a /generate-variants variant; returns a coloured surface for the viewport."),
    method=("Bearing reactions from gear mesh forces (lever rule per shaft), cosine-distributed on the "
            "bores; 3D FEM with 8-node incompatible-mode hexahedra on a voxel mesh; matrix-free PCG"),
    schema={
        "analysis": {"type": "string", "enum": ["static"], "default": "static"},
        "design": {"type": "object", "required": True, "fields": VARIANT_DESIGN, "allow_extra": True},
        "kinematics": {"type": "object", "required": True, "fields": VARIANT_KIN, "allow_extra": True},
        "loads": {"type": "object", "default": {}},
        "shaft": {"type": "object", "default": {}},
        "housing_params": {"type": "object", "default": {}},
        "housing_material": {"type": "string", "default": "Al7075-T6"},
        "fixture": {"type": "string", "enum": ["base", "flange"], "default": "base"},
        "mesh": {"type": "string", "enum": list(MESH_DIVISIONS), "default": "standard"},
        "design_factor": {"type": "number", "min": 1, "max": 10, "default": 1.5},
    },
    run=_run_housing,
    benchmarks=[Benchmark(
        "regression-housing-2stage-coarse", "ACE regression v0.2 on a real /generate-variants variant (self-consistency)",
        {**EXAMPLE_VARIANT, "mesh": "coarse"},
        {"outputs.max_von_mises_MPa": REG_VM}, 0.02)],
    limitations=["housing = closed box of /generate-assembly (no ribs, feet, fillets or bearing bosses)",
                 "bearing loads from gear forces only (no gravity, no thermal, no bolt preload)",
                 "1-stage or 2-stage reverted spur layouts (as /generate-assembly)",
                 f"server cap {MAX_ELEMENTS:,} elements (set IDRIS_FEA_MAX_ELEMENTS on a larger plan)"],
    example={**EXAMPLE_VARIANT, "fixture": "base", "mesh": "coarse"},
))
