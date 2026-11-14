"""Shell FEA of the parametric snow plow blade (physics/parts_library.py, template plow_blade). mm, N, MPa.

Model: moldboard = cylindrical panel on its mid-surface (radius R), ribs = flat plates in planes y = const welded to
the back of the panel along the whole arc. Flat 4-node shell elements (physics/shell_fea.py).
Not in the model: the cutting edge bar (its stiffness is ignored, which is the safe side for the sheet; its mass is
counted), the push frame, trip springs, welds (a weld line is a rigid joint), bolts, contact, buckling, fatigue.
Loads come from the engineer. IDRIS has no validated snow or impact load model and does not invent one.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import shell_fea as sf
from .parts_library import PLOW_INPUTS, STEEL_RHO, plow_geometry

# EN 10025-2:2019 Table 6, minimum yield strength ReH for nominal thickness <= 16 mm
STEELS = {"S235": {"name": "S235JR", "Sy": 235.0}, "S355": {"name": "S355J2", "Sy": 355.0}}
E_STEEL, NU_STEEL = 210000.0, 0.3
MAX_ELEMENTS = 12000


def _breaks(points: list, h: float) -> np.ndarray:
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil((b - a) / h - 1e-9)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def build(v: dict, h: float):
    """-> (ShellModel, info). Groups: 0 = moldboard, 1.. = ribs."""
    W, H, c, t, dr, tr = (float(v[k]) for k in ("W", "H", "c", "t", "dr", "tr"))
    g = plow_geometry(v)
    R, n = g["R"], g["ribs"]
    a = math.asin(H / (2 * R))
    xc, zc = R - c, H / 2
    rib_y = [i * (W - tr) / (n - 1) + tr / 2 for i in range(n)]
    ys = _breaks([0.0] + rib_y + [W], h)
    m = max(4, 2 * int(math.ceil(2 * a * R / h / 2)))
    phis = np.linspace(-a, a, m + 1)
    pt = lambda r, f, y: (xc - r * math.cos(f), y, zc + r * math.sin(f))
    parts = [sf.grid(ys, phis, lambda y, f: pt(R, f, y))[:2] + (t,)]
    r_out = R + t / 2 + dr
    rs = np.linspace(R, r_out, max(2, int(math.ceil((r_out - R) / h))) + 1)
    for y in rib_y:
        parts.append(sf.grid(rs, phis, lambda r, f, y=y: pt(r, f, y))[:2] + (tr,))
    nodes, quads, th, grp = sf.merge(parts, tol=1e-5)
    model = sf.ShellModel(nodes, quads, th, E_STEEL, NU_STEEL)
    rad = np.hypot(nodes[:, 0] - xc, nodes[:, 2] - zc)
    ang = np.arctan2(nodes[:, 2] - zc, xc - nodes[:, 0])
    info = {"R": R, "half_angle": a, "rib_y": rib_y, "r_out": r_out, "xc": xc, "zc": zc, "group": grp,
            "rad": rad, "ang": ang, "geom": g}
    return model, info


def _run_once(v: dict, loads: dict, mounts: list, mount_z, h: float) -> dict:
    model, info = build(v, h)
    if len(model.quads) > MAX_ELEMENTS:
        raise ValueError(f"mesh of {len(model.quads)} elements is over the limit {MAX_ELEMENTS}: use a larger element size")
    nodes, grp = model.nodes, info["group"]
    tol = 1e-4
    # supports: rear (outer) edge of the mounting ribs, fully clamped
    fixed = np.zeros(len(nodes), bool)
    for i in mounts:
        on = (np.abs(nodes[:, 1] - info["rib_y"][i]) < tol) & (np.abs(info["rad"] - info["r_out"]) < 1e-3)
        if mount_z:
            on &= (nodes[:, 2] >= mount_z[0] - tol) & (nodes[:, 2] <= mount_z[1] + tol)
        fixed |= on
    if fixed.sum() < 2:
        raise ValueError("the mounting zone holds fewer than 2 nodes: widen mount_z_mm or use a finer mesh")
    model.fix(np.where(fixed)[0])
    applied = []
    sheet = np.where(grp == 0)[0]
    p = float(loads.get("pressure_kPa") or 0.0) * 1e-3
    if p:
        # snow pushes on the concave (front) face: along the outward radial direction, i.e. away from the arc centre
        normals = model.element_normals()[sheet]
        cen = nodes[model.quads[sheet]].mean(axis=1)
        outward = np.stack([cen[:, 0] - info["xc"], np.zeros(len(cen)), cen[:, 2] - info["zc"]], axis=1)
        sign = np.sign(np.einsum("ij,ij->i", normals, outward))
        tot = np.zeros(3)
        for e, s in zip(sheet, sign):
            tot += model.add_pressure([e], p * s)
        applied.append({"load": "pressure on the moldboard front face", "pressure_kPa": p * 1e3,
                        "resultant_N": [round(float(x), 1) for x in tot]})
    ef = loads.get("edge_force_N")
    if ef:
        vec = np.array(ef, float) if isinstance(ef, (list, tuple)) else np.array([-float(ef), 0.0, 0.0])
        edge = (np.abs(info["rad"] - info["R"]) < 1e-3) & (np.abs(info["ang"] + info["half_angle"]) < 1e-6)
        zone = loads.get("edge_zone_mm")
        if zone:
            edge &= (nodes[:, 1] >= zone[0] - tol) & (nodes[:, 1] <= zone[1] + tol)
        ids = np.where(edge)[0]
        if len(ids) < 2:
            raise ValueError("edge_zone_mm holds fewer than 2 nodes on the bottom edge")
        model.add_force(ids, vec)
        applied.append({"load": "force on the bottom (cutting) edge", "force_N": [round(float(x), 1) for x in vec],
                        "zone_mm": [float(nodes[ids, 1].min()), float(nodes[ids, 1].max())]})
    if not applied:
        raise ValueError("no load given: send pressure_kPa and/or edge_force_N (IDRIS does not guess loads)")
    r = model.solve()
    vm = r["von_mises"]
    return {"model": model, "info": info, "r": r, "vm": vm, "applied": applied, "fixed_nodes": int(fixed.sum()),
            "vm_sheet": vm[grp == 0], "vm_ribs": vm[grp > 0]}


def analyze(p: dict, progress=lambda *a, **k: None) -> dict:
    t0 = time.perf_counter()
    v = {}
    for k, (lab, lo, hi, dv) in PLOW_INPUTS.items():
        x = p.get(k)
        if x in (None, ""):
            x = dv if dv is not None else (round(0.18 * float(v["H"]), 1) if k == "c" else None)
        if x is None:
            raise ValueError(f"{lab} is required")
        x = float(x)
        if not lo <= x <= hi:
            raise ValueError(f"{lab} must be between {lo:g} and {hi:g} (got {x:g})")
        v[k] = x
    if v["c"] > 0.45 * v["H"]:
        raise ValueError("curl depth is more than 45 % of the height")
    grade = str(p.get("steel") or "S355").upper()
    if grade not in STEELS:
        raise ValueError(f"steel must be one of {sorted(STEELS)}")
    Sy = STEELS[grade]["Sy"]
    g = plow_geometry(v)
    n = g["ribs"]
    mounts = p.get("mount_ribs")
    if not mounts:
        q = [min(range(n), key=lambda i, f=f: abs(i / (n - 1) - f)) for f in (0.25, 0.75)]
        mounts = sorted(set(q)) if len(set(q)) == 2 else [0, n - 1]
        mount_src = "default: the two ribs nearest the quarter points"
    else:
        mounts = sorted({int(i) - 1 for i in mounts})
        mount_src = "engineer's choice"
        if not mounts or mounts[0] < 0 or mounts[-1] >= n:
            raise ValueError(f"mount_ribs must be rib numbers from 1 to {n}")
    mz = p.get("mount_z_mm")
    loads = {k: p.get(k) for k in ("pressure_kPa", "edge_force_N", "edge_zone_mm")}
    h = float(p.get("element_size_mm") or max(20.0, min(60.0, v["W"] / 60)))
    progress("Shell FEA", 0.2)
    a = _run_once(v, loads, mounts, mz, h)
    r, vm = a["r"], a["vm"]
    vmax, v99 = float(vm.max()), float(np.percentile(vm, 99.0))
    out = {
        "template": "plow_blade", "dimensions_mm": v, "steel": {"grade": STEELS[grade]["name"], "yield_MPa": Sy,
                                                               "E_MPa": E_STEEL, "nu": NU_STEEL,
                                                               "source": "EN 10025-2:2019 Table 6, t <= 16 mm (minimum ReH)"},
        "yield_mpa": Sy, "mass_kg": round(g["mass_kg"], 2),
        "supports": {"mount_ribs": [i + 1 for i in mounts], "source": mount_src, "zone_z_mm": mz,
                     "assumption": "rear edge of the mounting ribs fully clamped", "fixed_nodes": a["fixed_nodes"]},
        "applied_loads": a["applied"], "reaction_N": [round(float(x), 1) for x in r["reaction"]],
        "equilibrium_error": r["equilibrium_error"],
        "max_von_mises_MPa": vmax, "von_mises_p99_MPa": v99,
        "max_von_mises_sheet_MPa": float(a["vm_sheet"].max()), "max_von_mises_ribs_MPa": float(a["vm_ribs"].max()),
        "max_displacement_mm": r["max_displacement"],
        "min_safety_factor": Sy / vmax if vmax > 0 else float("inf"),
        "safety_factor_p99": Sy / v99 if v99 > 0 else float("inf"),
        "mesh": {"type": "MITC4 flat shell", "element_size_mm": round(h, 2), "elements": int(len(a["model"].quads)),
                 "nodes": int(len(a["model"].nodes)), "dof": r["dof"]},
    }
    if p.get("convergence_study", True):
        progress("Convergence study", 0.6)
        b = _run_once(v, loads, mounts, mz, h / 1.5)
        vb = b["vm"]
        d_disp = abs(b["r"]["max_displacement"] / r["max_displacement"] - 1)
        d_p99 = abs(float(np.percentile(vb, 99.0)) / v99 - 1)
        d_max = abs(float(vb.max()) / vmax - 1)
        out["mesh_convergence"] = {
            "finer_element_size_mm": round(h / 1.5, 2), "finer_elements": int(len(b["model"].quads)),
            "change_max_displacement": round(d_disp, 4), "change_von_mises_p99": round(d_p99, 4),
            "change_max_von_mises": round(d_max, 4),
            "converged": bool(d_disp < 0.05 and d_p99 < 0.05),
            "note": "relative change when the element size is divided by 1.5; peak stress at clamped edges and weld "
                    "lines keeps rising with refinement (singular) and is not a convergence measure"}
    # surface for the viewer, in the CAD model's coordinates (the CAD part extends along -Y)
    model = a["model"]
    nodal, cnt = np.zeros(len(model.nodes)), np.zeros(len(model.nodes))
    for q, s in zip(model.quads, vm):
        nodal[q] += s
        cnt[q] += 1
    nodal /= np.maximum(cnt, 1)
    pos = model.nodes * np.array([1.0, -1.0, 1.0])
    disp = r["displacement"] * np.array([1.0, -1.0, 1.0])
    tris = np.concatenate([model.quads[:, [0, 1, 2]], model.quads[:, [0, 2, 3]]])
    out["surface"] = {"positions": np.round(pos, 2).ravel().tolist(), "indices": tris.ravel().tolist(),
                      "von_mises": np.round(nodal, 2).tolist(), "displacement": np.round(disp, 5).ravel().tolist()}
    out["solve_time_s"] = round(time.perf_counter() - t0, 2)
    progress("Post-processing", 1.0)
    return out
