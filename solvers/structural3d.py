"""3D structural FEA solvers (L2): generic solid block (engine verification) and gearbox housing."""
import os
import time
import numpy as np
from ace_core import SolverSpec, Benchmark, Check, register
from physics.fem3d import box_model
from ._example import EXAMPLE_VARIANT
from physics.variant_shafts import designed_housing_params
from physics.housing import (HOUSING_MATERIALS, find_material, build_layout, build_model,
                             bearing_loads, fixture_nodes)

MESH_DIVISIONS = {"coarse": 1, "standard": 2, "fine": 3}      # elements through the wall
REG_VM = 9.2362    # v0.11: layout v4 (central ribbed web, two bearings per shaft, deeper housing); v0.9 was 11.7776
MAX_ELEMENTS = int(os.environ.get("IDRIS_FEA_MAX_ELEMENTS", "60000"))
KNOWN_HP_KEYS = {"envelope_width", "envelope_height", "envelope_depth", "wall_thickness", "gear_ratio",
                 "bore_diameter", "bearing_widths", "web_mm", "front_bore_diameter", "back_bore_diameter",
                 "web_hole_diameter", "split_cover", "stickout_mm", "web_hub_diameter"}
FRAME_NOTE = ("Same coordinate frame as /generate-assembly: housing x∈[0,W], y∈[0,H], z∈[0,D] mm, "
              "shafts along +z at y = H/2. Body, central web and bolted covers are analysed as one rigidly joined "
              "part; bores = bearing outside diameter; bearing bosses, web pockets, bolts and plugs are not meshed "
              "(the plain wall is analysed, which is conservative for the seats).")


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
        L = build_layout(p["design"], p["kinematics"], p.get("loads") or {}, p.get("shaft") or {},
                         designed_housing_params(p, hp, p.get("input_rpm")))
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
        warnings.append("Bearings in closed bosses (e.g. the intermediate shaft) load the inner wall face around "
                        "the axis; web bearings load the face of the central web.")
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
    if not p.get("_inner"):
        out["options"] = _fea_options(p, L, mat, vmax, sf, out["model"]["approx_mass_kg"], solve_s, progress)
        out["outer_size_mm"], out["fits_envelope"], _why = _envelope_fit(p, [L["W"], L["H"], L["D"]])
    progress("Post-processing", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


def _envelope_fit(p, size):
    """(outer size incl. shaft ends if basis overall, fits?, reason) against the request's envelope_mm."""
    env = p.get("envelope_mm") or {}
    size = [float(size[0]), float(size[1]), float(size[2]) + (44.0 if p.get("envelope_basis") == "overall" else 0.0)]
    if not all(isinstance(env.get(k), (int, float)) and env[k] > 0 for k in ("width", "height", "depth")):
        return size, None, None
    lim = [env["width"], env["height"], env["depth"]]
    bad = [f"{n} {v:.0f} > {l:g} mm" for n, v, l in zip(("width", "height", "depth"), size, lim) if v > l + 0.5]
    return size, not bad, ("exceeds the envelope: " + ", ".join(bad)) if bad else None


def wall_change_hp(p, L, dt):
    """housing_params with the wall changed by dt mm and the cavity kept: outer box grows by 2·dt."""
    hp = dict(p.get("housing_params") or {})
    hp.update(wall_thickness=round(L["wall_mm"] + dt, 2), envelope_width=round(L["W"] + 2 * dt, 2),
              envelope_height=round(L["H"] + 2 * dt, 2), envelope_depth=round(L["D"] + 2 * dt, 2))
    return hp


MIN_WALL_MM = 3.0
FAST_ENOUGH_S = 15.0          # extra solves only when the base solve was quick (keeps /fea under ~60 s on Render)


def _fea_options(p, L, mat, vmax, sf, mass, solve_s, progress):
    """Computed design changes (UI 'Try' rows, same shape as /cfd options). Change keys merge into the request."""
    E0, nu0, Sy0, rho0, _ = HOUSING_MATERIALS[mat]
    target = p["design_factor"]
    opts = []

    def add(kind, label, change, sf_new, dmass, note, basis="computed", size=None):
        size, fits, why = _envelope_fit(p, size or [L["W"], L["H"], L["D"]])
        opts.append({"kind": kind, "label": label + (" — exceeds envelope" if fits is False else ""),
                     "change": change, "safety_factor": round(sf_new, 2), "pass": bool(sf_new >= target and fits is not False),
                     "added_mass_kg": round(dmass, 3), "note": note, "basis": basis, "outer_size_mm": size,
                     "fits_envelope": fits, "envelope_violation": why})

    def resolve(dt):
        hp = wall_change_hp(p, L, dt)
        r = _run_housing({**p, "housing_params": hp, "_inner": True}, lambda *a, **k: None)
        if not r.get("outputs"):
            return None, hp
        o = r["outputs"]
        return (Sy0 / max(o["max_von_mises_MPa"], 1e-12), o["model"]["approx_mass_kg"], o["model"]["envelope_mm"]), hp

    can_resolve = solve_s <= FAST_ENOUGH_S
    if sf < target:
        if can_resolve:
            for dt in (1.0, 2.0, 4.0):
                progress("Solving", 0.9)
                res, hp = resolve(dt)
                if res:
                    add("wall", f"Wall {L['wall_mm'] + dt:g} mm (+{dt:g})", {"housing_params": hp}, res[0], res[1] - mass,
                        "outer box grows by 2× the extra wall; cavity unchanged", size=res[2])
                    if res[0] >= target:
                        break
        for m, (E, nu, Sy, rho, note) in HOUSING_MATERIALS.items():
            if Sy > Sy0 and abs(nu - nu0) < 0.05 and m != mat:
                add("material", f"Housing material {m}", {"housing_material": m}, Sy / vmax, mass * (rho / rho0 - 1),
                    "same stress field (linear elastic, uniform material); SF = yield / σ_vM,max" + (f"; {note}" if note else ""))
    elif sf >= 2 * target:
        # over-designed: computed ways to save mass while keeping SF ≥ target
        if can_resolve and L["wall_mm"] - 1.0 >= MIN_WALL_MM:
            progress("Solving", 0.9)
            res, hp = resolve(-1.0)
            if res:
                add("wall", f"Thinner wall {L['wall_mm'] - 1:g} mm (−1)", {"housing_params": hp}, res[0], res[1] - mass,
                    f"saves mass; {MIN_WALL_MM:g} mm is the practical minimum for machined/cast aluminium", size=res[2])
        for m in ("Mg-AZ91D", "A356-T6"):
            if m != mat and m in HOUSING_MATERIALS:
                E, nu, Sy, rho, note = HOUSING_MATERIALS[m]
                add("material", f"Housing material {m}", {"housing_material": m}, Sy / vmax, mass * (rho / rho0 - 1),
                    "same stress field (linear elastic); " + ("cast, cheaper in series" if m == "A356-T6" else
                                                              "lightest structural metal; check corrosion protection"))
    opts.sort(key=lambda o: (not o["pass"], o["added_mass_kg"]))
    return opts


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
        "envelope_mm": {"type": "object", "default": {}, "allow_extra": False,
                        "fields": {"width": {"type": "number", "min": 0, "max": 1e5},
                                   "height": {"type": "number", "min": 0, "max": 1e5},
                                   "depth": {"type": "number", "min": 0, "max": 1e5}}},
        "envelope_basis": {"type": "string", "enum": ["housing", "overall"], "default": "housing"},
        "input_rpm": {"type": "number", "exclusive_min": 0, "max": 1e5},
    },
    run=_run_housing,
    benchmarks=[Benchmark(
        "regression-housing-2stage-coarse", "ACE regression v0.11 on a real /generate-variants variant, manufacturable layout v4 (central web, bearing seats), bores = bearing OD (self-consistency)",
        {**EXAMPLE_VARIANT, "mesh": "coarse"},
        {"outputs.max_von_mises_MPa": REG_VM}, 0.02)],
    limitations=["housing = closed box of /generate-assembly (no ribs, feet, fillets or bearing bosses)",
                 "bearing loads from gear forces only (no gravity, no thermal, no bolt preload)",
                 "1-stage or 2-stage reverted spur layouts (as /generate-assembly)",
                 f"server cap {MAX_ELEMENTS:,} elements (set IDRIS_FEA_MAX_ELEMENTS on a larger plan)"],
    example={**EXAMPLE_VARIANT, "fixture": "base", "mesh": "coarse"},
))
