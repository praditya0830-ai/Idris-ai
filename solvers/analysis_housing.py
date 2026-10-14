"""Housing analyses for the UI's Analysis tab: modal (L2), thermal (L2), design review (L0)."""
import os
import time
import numpy as np
from ace_core import SolverSpec, Benchmark, Check, register
from physics.fem3d import VoxelModel, box_model
from physics.housing import (HOUSING_MATERIALS, find_material, build_layout, build_model, fixture_nodes,
                             wetted_face_mask)
from physics.modal3d import natural_frequencies, natural_frequencies_fast
from physics.thermal3d import ThermalModel, h_natural_air, h_forced_air
from physics.review import review
from .structural3d import VARIANT_DESIGN, VARIANT_KIN, MESH_DIVISIONS

MAX_MODAL_ELEMENTS = int(os.environ.get("IDRIS_MODAL_MAX_ELEMENTS", "8000"))
MAX_THERMAL_ELEMENTS = int(os.environ.get("IDRIS_THERMAL_MAX_ELEMENTS", "40000"))
# thermal conductivity W/(m·K) of the housing materials
CONDUCTIVITY = {"Al7075-T6": 130, "Al6061-T6": 167, "Al2024-T3": 121, "A356-T6": 151, "Ti-6Al-4V": 6.7,
                "Steel-1045": 50, "Steel-4140": 42, "CastIron-GG25": 48, "Mg-AZ91D": 72}
OIL_GRADES = {"ISO VG 68": 68, "ISO VG 100": 100, "ISO VG 150": 150, "ISO VG 220": 220}
OIL_LIMIT_C = 90.0          # common continuous limit for mineral gear oils
EMISSIVITY = 0.85           # painted / anodised housing (bare machined Al ≈ 0.1–0.2)

VARIANT_SCHEMA = {
    "design": {"type": "object", "required": True, "fields": VARIANT_DESIGN, "allow_extra": True},
    "kinematics": {"type": "object", "required": True, "fields": VARIANT_KIN, "allow_extra": True},
    "loads": {"type": "object", "default": {}},
    "shaft": {"type": "object", "default": {}},
    "housing_params": {"type": "object", "default": {}},
    "housing_material": {"type": "string", "default": "Al7075-T6"},
}
EXAMPLE_VARIANT = {"design": {"module_mm": 2, "pinion_teeth": 18, "gear_teeth": 51, "face_width_mm": 48,
                              "stages": 2, "center_distance_mm": 69},
                   "kinematics": {"input_torque_nm": 23.9, "efficiency": 0.96},
                   "shaft": {"output_diameter_mm": 30}, "housing_material": "Al7075-T6"}


def _material(p):
    mat = find_material(p["housing_material"])
    if mat is None:
        raise ValueError(f"Unknown housing material '{p['housing_material']}'. "
                         f"Available: {', '.join(HOUSING_MATERIALS)}")
    return mat


def _layout(p):
    try:
        return build_layout(p["design"], p["kinematics"], p.get("loads") or {}, p.get("shaft") or {},
                            p.get("housing_params") or {})
    except (KeyError, ValueError, TypeError) as ex:
        raise ValueError(f"Cannot build the housing model from this variant: {ex}")


def _surface(model, L):
    ids, tris = model.surface()
    return ids, tris, np.round(model.nodes[ids] + L["offset"], 2).ravel()


FRAME_NOTE = ("Coordinate frame: input shaft axis at x=0, y=0 along +z, inner front face at z=0 "
              "(to be aligned with /generate-assembly; housing_params.frame_offset_mm shifts it)")


# ------------------------------------------------------------------ modal
def _run_modal(p, progress):
    try:
        mat = _material(p)
        L = _layout(p)
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    E, nu, Sy, rho, _ = HOUSING_MATERIALS[mat]
    progress("Meshing", 0.05)
    pre_warn = []
    mesh_used = p["mesh"]
    while True:
        h = L["wall_mm"] / MESH_DIVISIONS[mesh_used]
        model = build_model(L, h)
        ne = len(model.elems)
        if ne <= MAX_MODAL_ELEMENTS:
            break
        if mesh_used == "coarse":
            return {"ok": False, "errors": [
                f"This housing needs {ne:,} elements even on mesh 'coarse'; the server limit for modal analysis "
                f"is {MAX_MODAL_ELEMENTS:,}."]}
        pre_warn.append(f"Mesh '{mesh_used}' ({ne:,} elements) is above the server limit for eigenvalue problems "
                        f"({MAX_MODAL_ELEMENTS:,}); solved on a coarser mesh. For this housing type the first "
                        "frequencies change by less than 2 % between 'coarse' and 'standard'.")
        mesh_used = {"fine": "standard", "standard": "coarse"}[mesh_used]
    progress("Applying loads", 0.1)
    fixed = fixture_nodes(model, L, p["fixture"])
    progress("Solving", 0.15)
    t0 = time.perf_counter()

    def cb(it, res):
        progress("Solving", min(0.85, 0.15 + 0.7 * it / (it + 300)))

    f, shapes, info = natural_frequencies_fast(model, E, nu, rho, fixed, p["modes"], progress=cb)
    solve_s = time.perf_counter() - t0
    progress("Post-processing", 0.9)
    ids, tris, pos = _surface(model, L)
    modes = [{"frequency_hz": round(float(fk), 2), "displacement": np.round(sh[ids], 5).ravel(),
              "effective_mass_fraction_xyz": [round(v, 4) for v in info["effective_mass_fraction_xyz"][k]]}
             for k, (fk, sh) in enumerate(zip(f, shapes))]
    # excitation frequencies from the variant (shaft speeds and gear mesh 1x, 2x)
    kin = p["kinematics"]
    excit = []
    rpm_out = kin.get("output_rpm")
    ratio = kin.get("ratio_actual")
    if isinstance(rpm_out, (int, float)) and isinstance(ratio, (int, float)):
        rpm_in = rpm_out * ratio
        i_st = p["design"]["gear_teeth"] / p["design"]["pinion_teeth"]
        rpms = [rpm_in / i_st ** k for k in range(L["stages"] + 1)]
        for k, r in enumerate(rpms):
            excit.append({"source": f"shaft {k + 1} speed", "hz": r / 60})
        for s in range(L["stages"]):
            gm = rpms[s] / 60 * p["design"]["pinion_teeth"]
            excit += [{"source": f"stage {s + 1} gear mesh 1x", "hz": gm},
                      {"source": f"stage {s + 1} gear mesh 2x", "hz": 2 * gm}]
    warnings = pre_warn + [FRAME_NOTE, "Mode shapes are normalised to max |u| = 1 (arbitrary scale).",
                "Housing modelled empty: mass and stiffness of gears, shafts and oil are not included "
                "(real frequencies are usually somewhat lower)."]
    checks = []
    for e in excit:
        near = [m for m in modes if abs(m["frequency_hz"] - e["hz"]) <= 0.1 * e["hz"]]
        for m in near:
            warnings.append(f"Mode at {m['frequency_hz']:.0f} Hz is within ±10 % of {e['source']} ({e['hz']:.0f} Hz): resonance risk.")
    if excit:
        margin = min(abs(m["frequency_hz"] - e["hz"]) / e["hz"] for m in modes for e in excit)
        checks.append(Check("min separation mode ↔ excitation", margin, 0.1, "-", "min",
                            "±10 % rule; UI also flags this"))
    out = {"surface": {"positions": pos, "indices": tris.ravel(), "modes": modes},
           "mesh": {"elements": ne, "nodes": model.n_nodes, "type": "hex8-im (voxel)",
                    "element_size_mm": round(h, 3), "surface_triangles": int(len(tris))},
           "solve_time_s": round(solve_s, 2), "frequencies_hz": [m["frequency_hz"] for m in modes],
           "excitations_hz": excit, "eigen_method": info["method"], "material": mat,
           "housing_mass_kg": info["total_mass_kg"], "fixture": p["fixture"], "mesh_level": mesh_used}
    progress("Post-processing", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


register(SolverSpec(
    name="modal_housing_3d", version="0.1.0", domain="structural_dynamics", fidelity="L2", progress=True,
    summary="Natural frequencies and mode shapes of the gearbox housing, with resonance check against shaft and gear-mesh frequencies.",
    method=("3D FEM, 8-node incompatible-mode hexahedra on a voxel mesh, lumped mass; generalized eigenproblem by "
            "shift-invert Lanczos (scipy) or LOBPCG with block-Jacobi preconditioner (numpy)"),
    schema={**VARIANT_SCHEMA,
            "analysis": {"type": "string", "enum": ["modal"], "default": "modal"},
            "modes": {"type": "integer", "enum": [4, 6, 10], "default": 6},
            "fixture": {"type": "string", "enum": ["base", "flange"], "default": "base"},
            "mesh": {"type": "string", "enum": list(MESH_DIVISIONS), "default": "coarse"}},
    run=_run_modal,
    benchmarks=[Benchmark("regression-housing-modal-coarse",
                          "ACE regression v0.1; LOBPCG and scipy eigsh agree to 4 digits (self-consistency)",
                          {**EXAMPLE_VARIANT, "modes": 4, "mesh": "coarse"},
                          {"outputs.frequencies_hz.0": 1617.0}, 0.01)],
    limitations=["empty housing (no gears/shafts/oil mass)", "linear, undamped, small vibrations",
                 f"server cap {MAX_MODAL_ELEMENTS:,} elements (use mesh 'coarse')"],
    example={**EXAMPLE_VARIANT, "modes": 4, "mesh": "coarse"},
))


def _run_block_modal(p):
    m = box_model(p["length_mm"], p["width_mm"], p["height_mm"], p["element_size_mm"])
    if len(m.elems) > MAX_MODAL_ELEMENTS:
        return {"ok": False, "errors": ["too many elements; increase element_size_mm"]}
    fixed = np.where(m.nodes[:, 0] < 1e-9)[0]
    f, _, info = natural_frequencies_fast(m, p["E_MPa"], p["nu"], p["density_kg_m3"], fixed, 4)
    return {"outputs": {"frequencies_hz": [float(v) for v in f], "eigen_method": info["method"],
                        "elements": len(m.elems)}}


register(SolverSpec(
    name="modal_solid_block", version="0.1.0", domain="structural_dynamics", fidelity="L2",
    summary="Natural frequencies of a cantilevered solid block (engine verification).",
    method="Same eigen engine as modal_housing_3d",
    schema={"length_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "width_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "height_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "element_size_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "E_MPa": {"type": "number", "exclusive_min": 0, "default": 200000.0},
            "nu": {"type": "number", "min": 0, "max": 0.49, "default": 0.3},
            "density_kg_m3": {"type": "number", "exclusive_min": 0, "default": 7850.0}},
    run=_run_block_modal,
    benchmarks=[Benchmark("cantilever-euler-bernoulli-f1",
                          "Euler-Bernoulli cantilever: f₁ = (1.8751²/2π)·√(EI/ρAL⁴), L/H = 20 (Blevins, Formulas for Natural Frequency, Table 8-1)",
                          {"length_mm": 200, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5},
                          {"outputs.frequencies_hz.0": 203.8}, 0.02),
                Benchmark("cantilever-euler-bernoulli-f2",
                          "Euler-Bernoulli cantilever 2nd bending mode: β₂L = 4.6941",
                          {"length_mm": 200, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5},
                          {"outputs.frequencies_hz.2": 1277.5}, 0.02)],
    limitations=["voxel elements"],
    example={"length_mm": 200, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5},
))


# ------------------------------------------------------------------ thermal
def _run_thermal(p, progress):
    try:
        mat = _material(p)
        L = _layout(p)
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    warnings = [FRAME_NOTE]
    k = CONDUCTIVITY[mat]
    eta = float(p["kinematics"].get("efficiency", 0.97))
    Q = p["power_kw"] * 1000 * (1 - eta)
    progress("Meshing", 0.05)
    h = L["wall_mm"] / MESH_DIVISIONS[p["mesh"]]
    model = build_model(L, h)
    ne = len(model.elems)
    if ne > MAX_THERMAL_ELEMENTS:
        return {"ok": False, "errors": [f"Mesh '{p['mesh']}' needs {ne:,} elements but this server allows "
                                        f"{MAX_THERMAL_ELEMENTS:,}. Please choose a coarser mesh."]}
    th = ThermalModel(model, k)
    progress("Applying loads", 0.15)
    wet = wetted_face_mask(th, L)
    A_in = wet.sum() * th.face_area_mm2 * 1e-6
    A_out = (~wet).sum() * th.face_area_mm2 * 1e-6
    # oil-side coefficient: splash lubrication estimate scaled with pitch speed and viscosity
    v = (p.get("loads") or {}).get("pitch_line_velocity_ms")
    if not isinstance(v, (int, float)) or v <= 0:
        dp = p["design"]["module_mm"] * p["design"]["pinion_teeth"]
        v = 3.1416 * dp * p["input_rpm"] / 60000
    h_in = 200.0 * min(2.0, max(0.5, (v / 5.0) ** 0.5)) * (68 / OIL_GRADES[p["oil"]]) ** 0.2
    I = L["inner"]
    t = L["wall_mm"]
    Ho = (I["y1"] - I["y0"] + 2 * t) * 1e-3
    Wo = (I["x1"] - I["x0"] + 2 * t) * 1e-3
    Ta = float(p["ambient_c"])
    T_s = Ta + 20
    G_in = th.node_conductance(wet, h_in)
    progress("Solving", 0.2)
    t0 = time.perf_counter()
    for it in range(8):
        if p["cooling"] == "forced":
            if p["air_speed_ms"] <= 0:
                return {"ok": False, "errors": ["cooling 'forced' needs air_speed_ms > 0 (2, 5 or 10)"]}
            hc, hr, Re = h_forced_air(p["air_speed_ms"], Wo, T_s, Ta, EMISSIVITY)
        else:
            hc, hr = h_natural_air(T_s - Ta, Ho, T_s, Ta, EMISSIVITY)
        G_out = th.node_conductance(~wet, hc + hr)
        T, T_oil, iters = th.solve_oil_balance(Q, G_in, G_out, Ta)
        out_nodes = G_out > 0
        T_new = float((G_out[out_nodes] * T[out_nodes]).sum() / G_out[out_nodes].sum())
        progress("Solving", 0.2 + 0.08 * (it + 1))
        if abs(T_new - T_s) < 0.05:
            T_s = T_new
            break
        T_s = T_new
    solve_s = time.perf_counter() - t0
    progress("Post-processing", 0.9)
    q_out = float((G_out * (T - Ta)).sum())
    ids, tris, pos = _surface(model, L)
    if p["power_kw"] * (1 - eta) * 1000 < 1:
        warnings.append("Heat load is below 1 W: temperatures stay at ambient.")
    warnings += [f"Oil-side heat transfer coefficient {h_in:.0f} W/m²K is an engineering estimate for splash "
                 "lubrication (scaled with pitch speed and viscosity), not a CFD result.",
                 f"Outer surface: {'forced' if p['cooling'] == 'forced' else 'natural'} convection correlation "
                 f"(h_c = {hc:.1f} W/m²K) + radiation (ε = {EMISSIVITY}, h_r = {hr:.1f} W/m²K).",
                 "Heat source = P × (1 − η) delivered to a well-mixed oil volume; local hot spots at bearings "
                 "and the gear mesh are not resolved."]
    if T_oil > OIL_LIMIT_C:
        warnings.append(f"Oil temperature {T_oil:.0f} °C exceeds {OIL_LIMIT_C:.0f} °C: add fins, forced air, an oil "
                        "cooler, or a higher-efficiency gear set.")
    checks = [Check("oil temperature", T_oil, OIL_LIMIT_C, "°C", "max", "mineral gear oil continuous limit"),
              Check("heat balance error", abs(q_out - Q) / max(Q, 1e-9), 0.01, "-", "max", "solver self-check")]
    out = {"surface": {"positions": pos, "indices": tris.ravel(), "temperature": np.round(T[ids], 2)},
           "oil_temp_c": round(T_oil, 2), "heat_dissipated_w": round(q_out, 2), "heat_generated_w": round(Q, 2),
           "max_surface_temp_c": round(float(T.max()), 2), "mean_outer_surface_temp_c": round(T_s, 2),
           "h_oil_w_m2k": round(h_in, 1), "h_outer_convection_w_m2k": round(hc, 2),
           "h_outer_radiation_w_m2k": round(hr, 2), "area_outer_m2": round(A_out, 4),
           "area_wetted_m2": round(A_in, 4), "material": mat, "conductivity_w_mk": k,
           "mesh": {"elements": ne, "nodes": model.n_nodes, "type": "hex8 (voxel)",
                    "element_size_mm": round(h, 3), "surface_triangles": int(len(tris))},
           "solve_time_s": round(solve_s, 2), "outer_iterations": it + 1}
    progress("Post-processing", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


register(SolverSpec(
    name="thermal_housing_3d", version="0.1.0", domain="thermal", fidelity="L2", progress=True,
    summary="Steady-state oil and housing temperatures of a gearbox: gear losses P·(1−η) into the oil, conduction through the walls, convection + radiation to ambient.",
    method=("3D conduction FEM (8-node hexahedra, voxel mesh) coupled to a lumped oil node; outer convection from "
            "McAdams (natural) / flat-plate Nusselt (forced) correlations + linearised radiation, iterated to convergence"),
    schema={**VARIANT_SCHEMA,
            "analysis": {"type": "string", "enum": ["thermal"], "default": "thermal"},
            "power_kw": {"type": "number", "min": 0, "max": 10000, "required": True},
            "input_rpm": {"type": "number", "exclusive_min": 0, "max": 1e5, "default": 6000.0},
            "ambient_c": {"type": "number", "min": -40, "max": 80, "default": 25.0},
            "cooling": {"type": "string", "enum": ["natural", "forced"], "default": "natural"},
            "air_speed_ms": {"type": "number", "min": 0, "max": 50, "default": 0.0},
            "oil": {"type": "string", "enum": list(OIL_GRADES), "default": "ISO VG 68"},
            "mesh": {"type": "string", "enum": list(MESH_DIVISIONS), "default": "coarse"}},
    run=_run_thermal,
    benchmarks=[Benchmark("regression-housing-thermal", "ACE regression v0.1 + heat-balance self-check",
                          {**EXAMPLE_VARIANT, "power_kw": 15, "input_rpm": 6000, "ambient_c": 25},
                          {"outputs.heat_dissipated_w": 600.0}, 0.01)],
    limitations=["not CFD: convection from correlations, oil assumed well mixed",
                 "no fins, no sun load, housing emissivity assumed 0.85 (painted)"],
    example={**EXAMPLE_VARIANT, "power_kw": 15, "input_rpm": 6000, "ambient_c": 25, "cooling": "natural",
             "air_speed_ms": 0, "oil": "ISO VG 68"},
))


def _run_slab(p):
    n_t = max(2, round(p["thickness_mm"] / p["element_size_mm"]))
    h = p["thickness_mm"] / n_t
    m = VoxelModel(np.ones((n_t, 3, 3), bool), h)
    th = ThermalModel(m, p["k_W_mK"])
    left, right = th.normals[:, 0] < -0.5, th.normals[:, 0] > 0.5
    G1, G2 = th.node_conductance(left, p["h_hot"]), th.node_conductance(right, p["h_cold"])
    T, _ = th.solve_linear(G1 + G2, G1 * p["T_hot_c"] + G2 * p["T_cold_c"])
    area_m2 = 9 * h * h * 1e-6
    return {"outputs": {"heat_flux_W_m2": float((G2 * (T - p["T_cold_c"])).sum() / area_m2),
                        "hot_surface_temp_c": float(T[m.nodes[:, 0] < 1e-9].mean())}}


register(SolverSpec(
    name="thermal_slab", version="0.1.0", domain="thermal", fidelity="L2",
    summary="Plane wall between two fluids (engine verification of conduction + convection).",
    method="Same conduction/convection engine as thermal_housing_3d",
    schema={"thickness_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "element_size_mm": {"type": "number", "exclusive_min": 0, "default": 1.0},
            "k_W_mK": {"type": "number", "exclusive_min": 0, "required": True},
            "h_hot": {"type": "number", "exclusive_min": 0, "required": True},
            "h_cold": {"type": "number", "exclusive_min": 0, "required": True},
            "T_hot_c": {"type": "number", "required": True},
            "T_cold_c": {"type": "number", "required": True}},
    run=_run_slab,
    benchmarks=[Benchmark("plane-wall-resistance", "Incropera, Fundamentals of Heat and Mass Transfer, eq. 3.19: "
                          "q = ΔT / (1/h₁ + L/k + 1/h₂) (exact)",
                          {"thickness_mm": 10, "k_W_mK": 50, "h_hot": 500, "h_cold": 20, "T_hot_c": 100, "T_cold_c": 20},
                          {"outputs.heat_flux_W_m2": 80 / (1 / 500 + 0.01 / 50 + 1 / 20),
                           "outputs.hot_surface_temp_c": 100 - 80 / (1 / 500 + 0.01 / 50 + 1 / 20) / 500}, 0.001)],
    limitations=["1D slab only"],
    example={"thickness_mm": 10, "k_W_mK": 50, "h_hot": 500, "h_cold": 20, "T_hot_c": 100, "T_cold_c": 20},
))


# ------------------------------------------------------------------ review
CHECKS = ["interference", "fits", "dfm", "assembly"]


def _run_review(p, progress):
    try:
        mat = _material(p)
        L = _layout(p)
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    progress("Solving", 0.3)
    t0 = time.perf_counter()
    bad = [c for c in p["checks"] if c not in CHECKS]
    if bad:
        return {"ok": False, "errors": [f"unknown checks {bad}; choose from {CHECKS}"]}
    findings = review(p["design"], p["kinematics"], p.get("shaft") or {}, L, mat, p["checks"], p.get("parts"))
    progress("Post-processing", 1.0)
    counts = {s: sum(f["severity"] == s for f in findings) for s in ("critical", "warning", "info")}
    names = {pt.get("name") for pt in (p.get("parts") or []) if isinstance(pt, dict)}
    warnings = [FRAME_NOTE]
    if not names:
        warnings.append("No parts list received: findings use generic part names.")
    return {"outputs": {"findings": findings, "counts": counts, "checks_run": p["checks"],
                        "checked_parts": len(names) or None, "solve_time_s": round(time.perf_counter() - t0, 3)},
            "checks": [Check("critical findings", counts["critical"], 0, "-", "max")],
            "warnings": warnings}


register(SolverSpec(
    name="review_gearbox", version="0.1.0", domain="design_review", fidelity="L0", progress=True,
    summary="Pre-manufacturing design review of a gearbox variant: interference, fits, DFM and assembly, each finding with a fix and a 3D location.",
    method="Rule base: AGMA/ISO gear proportions, SKF bearing fit practice, casting/machining DFM guidelines, assembly sequence logic",
    schema={**VARIANT_SCHEMA,
            "analysis": {"type": "string", "enum": ["review"], "default": "review"},
            "parts": {"type": "array", "default": [], "max_items": 5000},
            "checks": {"type": "array", "items_type": "string", "default": CHECKS}},
    run=_run_review,
    benchmarks=[Benchmark("regression-review-2stage", "ACE regression v0.1: example variant must flag the closed-housing assembly problem",
                          {**EXAMPLE_VARIANT}, {"outputs.counts.critical": 1}, 0.0)],
    limitations=["rules on the parametric layout, not a geometric clash detection on the STL parts yet",
                 "part names matched heuristically from the parts list"],
    example={**EXAMPLE_VARIANT, "checks": CHECKS},
))
