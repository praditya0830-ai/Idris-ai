"""Static FEA of any part (Mark 1 #7): triangle mesh from the browser -> voxel hex FEA. POST /part-check."""
import math
import os

import numpy as np

from ace_core import SolverSpec, Benchmark, Check, register
from physics import part_fea as pf
from physics.gci import celik_gci, converged

MAX_ELEMENTS = int(os.environ.get("IDRIS_FEA_MAX_ELEMENTS", "60000"))
_SEL = {"type": "object"}


def _run(p, progress):
    p = dict(p, _max_elements=MAX_ELEMENTS)
    warnings = [pf.MATERIAL_NOTE,
                "Voxel mesh: curved faces are stair-stepped; peak stress at sharp corners, small holes and at the "
                "supports is mesh-dependent (singular) - read von_mises_p995_MPa, max_principal_MPa and the "
                "convergence study; the safety factor is the peak stress / strength of the criterion above.",
                "Supports are fully clamped (no sliding, no bolt flexibility); loads are spread evenly over the "
                "selected area."]
    try:
        o = pf.run(p, progress)
    except ValueError as e:
        return {"ok": False, "errors": [str(e)]}
    df = float(p.get("design_factor") or 2.0)
    crit = o.get("criterion") or {}
    sf = o.get("min_safety_factor")
    o["design_factor"] = df
    o["margin_of_safety"] = (sf / df - 1) if sf is not None else None
    # the label travels with the number: a brittle part is not judged on yield, and the check says which rule it is
    warnings.append(f"Strength criterion: {crit.get('label')}; strength {o['strength_MPa']:g} MPa "
                    f"({crit.get('strength_basis')}).")
    checks = []
    if sf is not None:
        checks.append(Check(f"part safety factor \u2014 {crit.get('label')}", sf, df, "-", "min"))
    else:
        warnings.append(o.get("safety_note") or "no safety factor was computed for this load case")
    checks += [Check("force equilibrium error", o["equilibrium_error"], 1e-3, "-", "max", "solver self-check"),
               Check("voxel volume error", abs(o["volume"]["error_pct"]), 5.0, "%", "max",
                     "geometry fidelity of the voxel mesh vs the exact part volume")]
    if p.get("max_deflection_mm"):
        checks.append(Check("max deflection", o["max_displacement_mm"], float(p["max_deflection_mm"]), "mm", "max"))
    if p.get("convergence_study"):
        h1 = o["mesh"]["element_size_mm"]
        hs, res = [h1], [o]
        for r_ in (1.3, 1.69):
            progress("Convergence study", 0.9)
            try:
                res.append(pf.run(p, _h=h1 * r_))
                hs.append(h1 * r_)
            except ValueError as e:
                warnings.append(f"convergence level skipped: {e}")
                break
        if len(res) == 3:
            mc = {}
            for key, label in (("max_displacement_mm", "peak displacement"), ("von_mises_p995_MPa", "99.5 % von Mises")):
                rep = celik_gci([r[key] for r in res], hs)
                rep["converged"] = converged(rep)
                mc[key] = rep
                if not rep["converged"]:
                    warnings.append(f"{label} not converged on these meshes ({rep.get('convergence')}); "
                                    "treat it as screening.")
            o["mesh_convergence"] = mc
        for r in res[1:]:
            r.pop("surface", None)
    if abs(o["volume"]["error_pct"]) > 5:
        warnings.append(f"Voxel volume differs {o['volume']['error_pct']:+.1f} % from the part: refine element_size_mm.")
    if o["margin_of_safety"] is not None and o["margin_of_safety"] < 0:
        warnings.append(f"Margin of safety {o['margin_of_safety']:.2f} < 0 at design factor {df:g}.")
    progress("Post-processing", 1.0)
    return {"outputs": o, "checks": checks, "warnings": warnings}


def _box_mesh(L, W, H):
    from assembly3d import build_box
    v, f = build_box(0, L, 0, W, 0, H)
    T = np.asarray(v, float)[np.asarray(f)]
    return {"positions": T.reshape(-1).tolist(), "indices": list(range(len(T) * 3))}


def _rod_mesh(d, L, n=128):
    from assembly3d import build_shaft
    v, f = build_shaft(d, d / 2, d / 2, 0, L, n=n)
    T = np.asarray(v, float)[np.asarray(f)]
    return {"positions": T.reshape(-1).tolist(), "indices": list(range(len(T) * 3))}


_STEEL = {"E_MPa": 200000.0, "nu": 0.3, "yield_MPa": 250.0}
BOX_EX = {**_box_mesh(100, 10, 10), **_STEEL, "element_size_mm": 1.25,
          "supports": [{"where": {"plane": "x", "side": "min"}}],
          "loads": [{"force_N": [0, -1000, 0], "where": {"plane": "x", "side": "max"}}]}
_I, _A, _k = math.pi * 10**4 / 64, math.pi * 25, 6 * 1.3 / (7 + 6 * 0.3)
ROD_EX = {**_rod_mesh(10, 100), **_STEEL, "element_size_mm": 0.7,
          "supports": [{"where": {"plane": "z", "side": "min"}}],
          "loads": [{"force_N": [1000, 0, 0], "where": {"plane": "z", "side": "max"}}]}

register(SolverSpec(
    name="fea_part_3d", version="0.1.0", domain="structural_fea", fidelity="L2",
    summary="Static FEA of any part from its triangle mesh (STL or positions/indices): supports, loads, "
            "von Mises, displacement, safety factor, optional mesh-convergence study.",
    method="Ray-cast voxelization (Z rays through voxel centres) -> 8-node hexahedra with incompatible modes, "
           "matrix-free PCG (same engine as fea_solid_block / fea_housing_3d)",
    schema={
        "stl_base64": {"type": "string", "max_len": 30_000_000},
        "positions": {"type": "array", "items_type": "number", "max_items": 3_000_000},
        "indices": {"type": "array", "items_type": "integer", "max_items": 3_000_000},
        "material": {"type": "string", "enum": list(pf.MATERIALS), "default": "aluminium"},
        "E_MPa": {"type": "number", "exclusive_min": 0, "max": 1e6},
        "nu": {"type": "number", "min": 0, "max": 0.49},
        "yield_MPa": {"type": "number", "exclusive_min": 0, "max": 1e4},
        "density_kg_m3": {"type": "number", "exclusive_min": 0, "max": 25000},
        "supports": {"type": "array", "min_items": 1, "max_items": 12, "items_type": "object"},
        "loads": {"type": "array", "min_items": 1, "max_items": 12, "items_type": "object"},
        "element_size_mm": {"type": "number", "exclusive_min": 0, "max": 500},
        "target_elements": {"type": "integer", "min": 500, "max": MAX_ELEMENTS, "default": 20000},
        "design_factor": {"type": "number", "min": 1.0, "max": 10.0, "default": 2.0},
        "max_deflection_mm": {"type": "number", "exclusive_min": 0},
        "convergence_study": {"type": "boolean", "default": False},
    },
    run=_run, progress=True,
    benchmarks=[
        Benchmark("cantilever-box-timoshenko",
                  "Square bar 100x10x10 from a triangle mesh, clamped face, 1 kN tip load: "
                  "Timoshenko δ = PL³/3EI + PL/(κGA), κ = 5/6",
                  BOX_EX, {"outputs.max_displacement_mm": 1000 * 100**3 / (3 * 200000 * 10**4 / 12)
                           + 1000 * 100 / (5 / 6 * 200000 / 2.6 * 100)}, 0.02),
        Benchmark("cantilever-round-bar-voxelized",
                  "Round bar Ø10 x 100 (128-facet mesh, voxelized, 0.7 mm): δ = PL³/3EI + PL/(κGA), "
                  "κ = 6(1+ν)/(7+6ν) (circular section) - checks the stair-stepped geometry",
                  ROD_EX, {"outputs.max_displacement_mm": 1000 * 100**3 / (3 * 200000 * _I)
                           + 1000 * 100 / (_k * 200000 / 2.6 * _A)}, 0.03),
    ],
    limitations=["linear elastic, small displacement, static", "voxel (cube) elements: curved faces stair-stepped",
                 "clamped supports only; no contact, no bolt preload, no pressure or gravity loads yet",
                 "peak stress at sharp corners/supports is singular: use p99.5 stress and the convergence study",
                 f"at most {MAX_ELEMENTS} elements (Render 512 MB)"],
    example=dict(BOX_EX, element_size_mm=2.5),
))
