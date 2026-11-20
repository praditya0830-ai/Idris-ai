"""Natural frequencies of any part from its triangle mesh (v0.39 P2a): POST /solve/modal_part_3d.

Geometry path is the same as /part-check: a closed triangle mesh is voxelized, and the eigenproblem
K phi = omega^2 M phi is solved on the voxel solid by the engine in physics/modal3d.py (shift-invert Lanczos
when scipy is present, LOBPCG otherwise). Supports are clamped, selected on the surface like /part-check.

What it is for: a part whose lowest natural frequencies must be compared with what excites it (motor, gear
mesh, road input). What it is not: a damped or forced response, and not an electromagnetic excitation.
"""
import numpy as np

from ace_core import SolverSpec, Benchmark, Check, register
from physics import modal3d, part_fea as pf

_SEL = {"type": "object"}


def _run(p, progress):
    tri = pf.triangles_from_input(p)
    vol = pf.mesh_volume(tri)
    if vol <= 0:
        raise ValueError("the geometry has no volume (is the mesh closed?)")
    mat_key = str(p.get("material") or "aluminium").lower()
    mat = dict(pf.MATERIALS.get(mat_key, pf.MATERIALS["aluminium"]))
    for k_in, k in (("E_MPa", "E"), ("nu", "nu"), ("density_kg_m3", "rho")):
        if p.get(k_in):
            mat[k] = float(p[k_in])
    target = int(p.get("target_elements") or 20000)
    h = float(p.get("element_size_mm") or (vol / target) ** (1 / 3))
    n_modes = int(p.get("n_modes") or 6)
    progress("Meshing", 0.05)
    filled, origin = pf.voxelize(tri, h)
    ne = int(filled.sum())
    if ne < 8:
        raise ValueError("geometry too small for this element size (fewer than 8 elements)")
    model = pf.VoxelModel(filled, h=h, origin=origin)
    ids, tris, area_w = pf._surface_areas(model)
    fixed = []
    for s in p.get("supports") or []:
        m = pf.select(model, ids, s.get("where") or {}, h)
        if not m.any():
            raise ValueError(f"support {s.get('where')} selects no surface of the part")
        fixed.append(ids[m])
    if not fixed:
        raise ValueError("at least one support is needed")
    fixed = np.unique(np.concatenate(fixed))
    progress("Eigen-solve", 0.25)
    f, shapes, info = modal3d.natural_frequencies_fast(model, mat["E"], mat["nu"], mat["rho"], fixed, n_modes,
                                                       progress=lambda it, r: progress("Eigen-solve", 0.4))
    progress("Post-processing", 0.95)
    vvox = float(model.evol.sum())
    modes = []
    for k in range(len(f)):
        u = shapes[k]
        modes.append({"mode": k + 1, "frequency_hz": round(float(f[k]), 2),
                      "displacement": np.round(u[ids], 6).ravel().tolist(),
                      "max_displacement_mm": round(float(np.linalg.norm(u, axis=1).max()), 6),
                      "effective_mass_fraction_xyz": [round(float(x), 4)
                                                      for x in info["effective_mass_fraction_xyz"][k]]})
    out = {
        "frequencies_hz": [round(float(x), 2) for x in f],
        "lowest_hz": round(float(f[0]), 2) if len(f) else None,
        "modes": modes, "material": mat, "mass_kg": round(float(info.get("total_mass_kg") or 0), 4),
        "eigen_method": info.get("method"), "converged": bool(info.get("converged")),
        "iterations": info.get("iterations"),
        "mesh": {"type": "hex8 voxel", "element_size_mm": round(float(h), 4), "elements": ne,
                 "nodes": model.n_nodes, "surface_triangles": int(len(tris)), "fixed_nodes": int(len(fixed))},
        "volume": {"mesh_mm3": round(vol, 2), "voxel_mm3": round(vvox, 2),
                   "error_pct": round(100 * (vvox / vol - 1), 2)},
        "surface": {"positions": np.round(model.nodes[ids], 2).ravel().tolist(), "indices": tris.ravel().tolist()},
        "not_checked": ["damping", "forced response", "electromagnetic or gear-mesh excitation (only the "
                        "structure's own frequencies are computed)"],
    }
    checks = [Check("voxel volume error", abs(out["volume"]["error_pct"]), 5.0, "%", "max",
                    "geometry fidelity of the voxel mesh vs the exact part volume")]
    if p.get("min_frequency_hz"):
        checks.append(Check("lowest natural frequency", float(f[0]), float(p["min_frequency_hz"]), "Hz", "min"))
    warnings = ["Voxel mesh: curved surfaces are stair-stepped, so frequencies move by a few percent; refine "
                "element_size_mm when a mode is close to the margin.",
                "Supports are fully clamped (no sliding, no bolt flexibility); a clamped edge is stiffer than a "
                "bolted joint, so the computed frequencies are an upper bound for that joint."]
    if not info.get("converged"):
        warnings.append(f"the eigen-solver did not converge (residual {info.get('residual')}); treat the modes "
                        f"as a screening estimate")
    if abs(out["volume"]["error_pct"]) > 5:
        warnings.append(f"Voxel volume differs {out['volume']['error_pct']:+.1f} % from the part: refine "
                        f"element_size_mm.")
    progress("Done", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


def _box_mesh(L, W, H):
    from assembly3d import build_box
    v, f = build_box(0, L, 0, W, 0, H)
    T = np.asarray(v, float)[np.asarray(f)]
    return {"positions": T.reshape(-1).tolist(), "indices": list(range(len(T) * 3))}


BEAM = {**_box_mesh(200, 10, 10), "material": "steel", "element_size_mm": 2.5,
        "n_modes": 6, "supports": [{"where": {"plane": "x", "side": "min"}}]}
# Euler-Bernoulli cantilever, beta1 L = 1.8751, beta2 L = 4.6941, beta3 L = 7.8548; the 3rd bending mode of a
# square section is the 3rd listed mode only for a beam that bends in one plane (the second bending plane
# repeats modes 1-3 first), so the check uses the torsion-free sequence the engine actually returns.
_E, _NU, _RHO = 210000.0, 0.3, 7850.0
_A, _I = 10 * 10.0, 10 * 10.0 ** 3 / 12
_L = 200.0


def _f_euler(beta_l):
    return beta_l ** 2 / (2 * np.pi) * np.sqrt(_E * _I / (_RHO * 1e-12 * _A * _L ** 4))


PLATE = {**_box_mesh(100, 100, 4), "material": "steel", "element_size_mm": 2.0, "n_modes": 4,
         "supports": [{"where": {"plane": "x", "side": "min"}}, {"where": {"plane": "x", "side": "max"}},
                      {"where": {"plane": "y", "side": "min"}}, {"where": {"plane": "y", "side": "max"}}]}

register(SolverSpec(
    name="modal_part_3d", version="1.0.0", domain="structural_dynamics", fidelity="L2",
    summary="Natural frequencies and mode shapes of any part from its triangle mesh: clamped supports, "
            "reported modes, mass, effective modal mass, and how far the voxel volume is off the part.",
    method="Ray-cast voxel mesh (hex8) -> generalized eigenproblem K phi = omega^2 M phi, lumped mass, "
           "shift-invert Lanczos (scipy) or LOBPCG; same engine as modal_solid_block / modal_housing_3d",
    schema={
        "stl_base64": {"type": "string", "max_len": 30_000_000},
        "positions": {"type": "array", "items_type": "number", "max_items": 3_000_000},
        "indices": {"type": "array", "items_type": "integer", "max_items": 3_000_000},
        "material": {"type": "string", "enum": list(pf.MATERIALS), "default": "aluminium"},
        "E_MPa": {"type": "number", "exclusive_min": 0, "max": 1e6},
        "nu": {"type": "number", "min": 0, "max": 0.49},
        "density_kg_m3": {"type": "number", "exclusive_min": 0, "max": 25000},
        "supports": {"type": "array", "min_items": 1, "max_items": 12, "items_type": "object"},
        "element_size_mm": {"type": "number", "exclusive_min": 0, "max": 500},
        "target_elements": {"type": "integer", "min": 500, "max": 60000, "default": 20000},
        "n_modes": {"type": "integer", "min": 1, "max": 12, "default": 6},
        "min_frequency_hz": {"type": "number", "exclusive_min": 0},
    },
    run=_run, progress=True,
    benchmarks=[
        Benchmark("cantilever-mesh-euler-f1",
                  "Euler-Bernoulli cantilever, beta1 L = 1.8751: f1 = (beta1 L)^2/2pi sqrt(EI/(rho A L^4)) "
                  "(Blevins, Formulas for Natural Frequency, Table 8-1), 200x10x10 mm steel bar from a triangle "
                  "mesh, clamped face",
                  BEAM, {"outputs.frequencies_hz.0": round(float(_f_euler(1.8751)), 1)}, 0.05),
        Benchmark("cantilever-mesh-euler-f2",
                  "Euler-Bernoulli cantilever, beta2 L = 4.6941 (Blevins Table 8-1): the 2nd bending mode of the "
                  "same bar. A square section bends equally in both planes, so the engine lists f1 twice and f2 "
                  "is mode 3 (same sequence as modal_solid_block)",
                  BEAM, {"outputs.frequencies_hz.2": round(float(_f_euler(4.6941)), 1)}, 0.05),
        Benchmark("clamped-edge-square-plate-leissa",
                  "Square plate with all four edges clamped, fundamental: omega a^2 sqrt(rho h/D) = 35.99 "
                  "(Leissa, Vibration of Plates, NASA SP-160, 1969, Table 4-1; Blevins Table 11-4). 100x100x4 mm "
                  "steel plate; the four edge faces (the full 4 mm thickness) are clamped, which is slightly "
                  "stiffer than a clamped edge on the mid-surface",
                  PLATE, {"outputs.frequencies_hz.0": round(float(35.99 / (2 * np.pi) * np.sqrt(
                      _E * 4.0 ** 3 / (12 * (1 - _NU ** 2)) / (_RHO * 1e-12 * 4.0)) / 100.0 ** 2), 1)}, 0.04),
    ],
    limitations=[
        "linear, undamped, free vibration of a clamped part: no forced or transient response, no damping",
        "voxel (cube) elements: curved faces stair-stepped, frequencies move by a few percent",
        "clamped supports only: a bolted or sliding joint is softer than clamped, so computed frequencies are "
        "an upper bound for it",
        "no pre-stress, no rotation (no centrifugal stiffening), no contact",
        "at most 60000 elements (Render 512 MB)",
    ],
    example=dict(BEAM, element_size_mm=4.0, n_modes=4),
))
