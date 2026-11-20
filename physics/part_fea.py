"""FEA for any part (Mark 1 #7, ACE step 4): triangle mesh from the browser kernel -> voxel hex mesh -> static
linear FEA with the same engine as the housing (physics/fem3d.VoxelModel, benchmarks: bar tension, Timoshenko
cantilever, Kirsch, NAFEMS LE1).

Geometry: the part's closed triangle surface (STL bytes or positions/indices). It is voxelized by ray casting along
Z through voxel centres (inside = odd number of crossings above the centre). Curved faces become stair-stepped;
the voxel volume is compared with the exact mesh volume and reported.

Boundary conditions are picked on the SURFACE with selectors, in part coordinates (mm):
    {"plane": "x"|"y"|"z", "side": "min"|"max"}    a flat end face of the bounding box
    {"box": [[x0, y0, z0], [x1, y1, z1]]}           all surface nodes inside a box (e.g. a clicked face's bbox,
                                                     a bolt hole)
    {"cylinder": {"center": [x, y, z], "axis": "z", "radius": r, "length": l}}   surface nodes on/inside a hole
supports: [{"where": selector}]  -> clamped (ux = uy = uz = 0)
loads:    [{"force_N": [Fx, Fy, Fz], "where": selector}] -> total force spread over the selected surface area
"""
from __future__ import annotations

import base64
import math
import struct
import time

import numpy as np

from physics.fem3d import VoxelModel

# Typical handbook values for screening. E [MPa], nu [-], Sy [MPa], rho [kg/m^3].
# Every row says where its density and its strength come from (`source`), because a mass or a safety factor
# without a source is a guess with a decimal point. `Sy` is the 0.2 % proof stress for metals that yield; cast
# iron does not yield, so its row carries the minimum tensile strength and says so (`brittle`).
MATERIALS = {
    "aluminium": {"name": "Aluminium 6061-T6", "E": 68900.0, "nu": 0.33, "Sy": 276.0, "rho": 2700.0,
                  "source": {"density": "ASM Handbook Vol. 2: 6061-T6, 2.70 g/cm^3",
                             "strength": "ASM Handbook Vol. 2: 6061-T6, Rp0.2 276 MPa",
                             "stiffness": "ASM Handbook Vol. 2: 6061-T6, E 68.9 GPa"}},
    "steel": {"name": "Structural steel S235", "E": 210000.0, "nu": 0.3, "Sy": 235.0, "rho": 7850.0,
              "source": {"density": "EN 10025-2 S235JR: 7.85 g/cm^3",
                         "strength": "EN 10025-2 S235JR: ReH min 235 MPa (t <= 16 mm)",
                         "stiffness": "EN 10025-2: E 210 GPa"}},
    "stainless": {"name": "Stainless 304 (annealed)", "E": 193000.0, "nu": 0.29, "Sy": 205.0, "rho": 8000.0,
                  "source": {"density": "ASM Handbook Vol. 2: AISI 304, 8.00 g/cm^3",
                             "strength": "ASTM A240/A276 304: Rp0.2 min 205 MPa (annealed)",
                             "stiffness": "ASM Handbook Vol. 2: AISI 304, E 193 GPa"}},
    "titanium": {"name": "Ti-6Al-4V", "E": 114000.0, "nu": 0.34, "Sy": 880.0, "rho": 4430.0,
                 "source": {"density": "ASM Handbook Vol. 2: Ti-6Al-4V, 4.43 g/cm^3",
                            "strength": "ASM Handbook Vol. 2: Ti-6Al-4V annealed, Rp0.2 880 MPa",
                            "stiffness": "ASM Handbook Vol. 2: Ti-6Al-4V, E 114 GPa"}},
    "brass": {"name": "Brass CW614N (CuZn39Pb3)", "E": 97000.0, "nu": 0.34, "Sy": 200.0, "rho": 8470.0,
              "source": {"density": "EN 12164 CW614N: 8.47 g/cm^3",
                         "strength": "EN 12164 CW614N: Rp0.2 is temper dependent (about 200-450 MPa); "
                                     "200 MPa is the softest temper, a lower bound",
                         "stiffness": "EN 12164 CW614N: E about 97 GPa"}},
    "cast_iron": {"name": "Cast iron GJL-250 (grey)", "E": 105000.0, "nu": 0.26, "Sy": 250.0, "rho": 7200.0,
                  "brittle": True,
                  "source": {"density": "EN 1561 GJL-250: 7.20 g/cm^3",
                             "strength": "EN 1561 GJL-250: Rm min 250 MPa in tension. Brittle: there is no yield "
                                         "point, so the safety factor here is Rm / von Mises, not a yield check",
                             "stiffness": "EN 1561 GJL-250: E 105 GPa (105-120 GPa for grey iron)"}},
    "pla": {"name": "PLA (FDM, 100 % infill)", "E": 3500.0, "nu": 0.36, "Sy": 45.0, "rho": 1240.0,
            "source": {"density": "filament datasheets: PLA 1.24 g/cm^3",
                       "strength": "filament datasheets: tensile about 45 MPa printed at 100 % infill; no standard "
                                   "specimen, and the bond between layers is weaker",
                       "stiffness": "filament datasheets: E about 3.5 GPa"}},
    "abs": {"name": "ABS (FDM, 100 % infill)", "E": 2200.0, "nu": 0.35, "Sy": 35.0, "rho": 1050.0,
            "source": {"density": "filament datasheets: ABS 1.05 g/cm^3",
                       "strength": "filament datasheets: tensile about 35 MPa printed at 100 % infill; no standard "
                                   "specimen, and the bond between layers is weaker",
                       "stiffness": "filament datasheets: E about 2.2 GPa"}},
    "nylon": {"name": "Nylon PA12 (SLS/FDM)", "E": 1700.0, "nu": 0.39, "Sy": 45.0, "rho": 1010.0,
              "source": {"density": "EOS PA 2200 datasheet: 1.01 g/cm^3",
                         "strength": "EOS PA 2200 datasheet: tensile 45-48 MPa (dry); 45 MPa is used, and a wet "
                                     "part is weaker",
                         "stiffness": "EOS PA 2200 datasheet: E 1.7 GPa"}},
}
MATERIAL_NOTE = ("Typical handbook values for screening; printed plastics depend strongly on infill, layer "
                 "direction and temperature (FDM parts are weaker across layers). Use certified data for sign-off."
                 " Cast iron GJL-250 is brittle: its reported safety factor is tensile strength / von Mises, not a "
                 "yield check.")


def material_catalog():
    """The one material table for the UI selector and the engineering report."""
    return {"materials": [{"id": key, **row} for key, row in MATERIALS.items()], "note": MATERIAL_NOTE}



# ------------------------------------------------------------------ geometry input
def triangles_from_stl(data: bytes) -> np.ndarray:
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data) and n > 0:
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].astype(float)
    txt = data.decode("ascii", "ignore")
    vs = [list(map(float, ln.split()[1:4])) for ln in txt.splitlines() if ln.strip().startswith("vertex")]
    if len(vs) < 3 or len(vs) % 3:
        raise ValueError("could not read the STL (binary or ASCII)")
    return np.array(vs, float).reshape(-1, 3, 3)


def triangles_from_input(p: dict) -> np.ndarray:
    if p.get("stl_base64"):
        tri = triangles_from_stl(base64.b64decode(p["stl_base64"].split(",", 1)[-1]))
    elif p.get("positions") and p.get("indices"):
        pos = np.asarray(p["positions"], float).reshape(-1, 3)
        idx = np.asarray(p["indices"], int).reshape(-1, 3)
        if idx.max() >= len(pos) or idx.min() < 0:
            raise ValueError("indices point outside positions")
        tri = pos[idx]
    else:
        raise ValueError("send the part as stl_base64, or as positions + indices (triangles)")
    if not np.isfinite(tri).all():
        raise ValueError("geometry contains NaN/inf")
    return tri


def mesh_volume(tri: np.ndarray) -> float:
    return float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


# ------------------------------------------------------------------ voxelization (ray casting along Z)
def voxelize(tri: np.ndarray, h: float):
    lo = tri.reshape(-1, 3).min(0)
    hi = tri.reshape(-1, 3).max(0)
    n = np.maximum(1, np.ceil((hi - lo) / h - 1e-9).astype(int))
    origin = lo - (n * h - (hi - lo)) / 2.0                       # centre the grid on the part
    nx, ny, nz = n
    eps = h * np.array([0.0137, 0.0219])                         # keep rays off exact edges/vertices (CAD grids)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    keep = np.abs(area2) > 1e-12
    a, b, c, area2 = a[keep], b[keep], c[keep], area2[keep]
    txy_lo = np.minimum(np.minimum(a, b), c)[:, :2]
    txy_hi = np.maximum(np.maximum(a, b), c)[:, :2]
    # ray k sits at origin + (k + 0.5) h + eps; candidate rays inside the triangle's xy box
    i0 = np.clip(np.ceil((txy_lo[:, 0] - origin[0] - eps[0]) / h - 0.5), 0, nx - 1).astype(int)
    i1 = np.clip(np.floor((txy_hi[:, 0] - origin[0] - eps[0]) / h - 0.5), -1, nx - 1).astype(int)
    j0 = np.clip(np.ceil((txy_lo[:, 1] - origin[1] - eps[1]) / h - 0.5), 0, ny - 1).astype(int)
    j1 = np.clip(np.floor((txy_hi[:, 1] - origin[1] - eps[1]) / h - 0.5), -1, ny - 1).astype(int)
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    cnt = ni * nj
    total = int(cnt.sum())
    filled = np.zeros((nx, ny, nz), bool)
    if total == 0:
        return filled, origin
    t = np.repeat(np.arange(len(a)), cnt)
    off = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ii = i0[t] + off // np.repeat(nj, cnt)
    jj = j0[t] + off % np.repeat(nj, cnt)
    px = origin[0] + (ii + 0.5) * h + eps[0]
    py = origin[1] + (jj + 0.5) * h + eps[1]
    A, B, C, D2 = a[t], b[t], c[t], area2[t]
    w1 = ((B[:, 0] - px) * (C[:, 1] - py) - (C[:, 0] - px) * (B[:, 1] - py)) / D2
    w2 = ((C[:, 0] - px) * (A[:, 1] - py) - (A[:, 0] - px) * (C[:, 1] - py)) / D2
    w3 = 1.0 - w1 - w2
    ok = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
    z = w1 * A[:, 2] + w2 * B[:, 2] + w3 * C[:, 2]
    col = (ii * ny + jj)[ok]
    z = z[ok]
    order = np.lexsort((z, col))
    col, z = col[order], z[order]
    starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
    ends = np.r_[starts[1:], len(col)]
    zc = origin[2] + (np.arange(nz) + 0.5) * h
    for s, e in zip(starts, ends):
        zz = z[s:e]
        if len(zz) > 1:                                          # a ray through a shared edge counts it once
            zz = zz[np.r_[True, np.diff(zz) > 1e-9 * max(h, 1.0)]]
        if len(zz) % 2:                                          # open/non-manifold column: drop last hit
            zz = zz[:-1]
        ci = col[s]
        line = np.zeros(nz, bool)
        for k in range(0, len(zz), 2):
            line |= (zc > zz[k]) & (zc < zz[k + 1])
        filled[ci // ny, ci % ny] = line
    return filled, origin


# ------------------------------------------------------------------ boundary-condition selectors
def _surface_areas(model):
    ids, tris = model.surface()
    pos = model.nodes[ids]
    P = pos[tris]
    ar = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    w = np.bincount(tris.ravel(), weights=np.repeat(ar / 3.0, 3), minlength=len(ids))
    return ids, tris, w


def select(model, ids, where: dict, h: float):
    x = model.nodes[ids]
    tol = 0.51 * h
    if "plane" in where:
        ax = "xyz".index(where["plane"])
        v = x[:, ax]
        target = v.min() if where.get("side", "min") == "min" else v.max()
        return np.abs(v - target) <= tol
    if "box" in where:
        lo, hi = np.asarray(where["box"], float)
        lo, hi = np.minimum(lo, hi) - tol, np.maximum(lo, hi) + tol
        return np.all((x >= lo) & (x <= hi), axis=1)
    if "cylinder" in where:
        cy = where["cylinder"]
        ax = "xyz".index(cy.get("axis", "z"))
        c = np.asarray(cy["center"], float)
        d = x - c
        along = d[:, ax]
        rad = np.linalg.norm(np.delete(d, ax, axis=1), axis=1)
        L = float(cy.get("length", 1e9))
        return (rad <= float(cy["radius"]) + tol) & (np.abs(along) <= L / 2 + tol)
    raise ValueError("selector needs plane, box or cylinder")


# ------------------------------------------------------------------ analysis
def run(p: dict, progress=lambda *a, **k: None, _h=None) -> dict:
    t0 = time.perf_counter()
    tri = triangles_from_input(p)
    vol = mesh_volume(tri)
    if vol <= 0:
        raise ValueError("the geometry has no volume (is the mesh closed?)")
    mat_key = str(p.get("material") or "aluminium").lower()
    mat = dict(MATERIALS.get(mat_key, MATERIALS["aluminium"]))
    for k_in, k in (("E_MPa", "E"), ("nu", "nu"), ("yield_MPa", "Sy"), ("density_kg_m3", "rho")):
        if p.get(k_in):
            mat[k] = float(p[k_in])
            mat["name"] = mat["name"] + " (user values)"
    target = int(p.get("target_elements") or 20000)
    h = _h or p.get("element_size_mm") or (vol / target) ** (1 / 3)
    progress("Meshing", 0.05)
    filled, origin = voxelize(tri, h)
    ne = int(filled.sum())
    max_el = int(p.get("_max_elements") or 60000)
    if ne > max_el:
        raise ValueError(f"{ne} elements > limit {max_el}; increase element_size_mm")
    if ne < 8:
        raise ValueError("geometry too small for this element size (fewer than 8 elements)")
    model = VoxelModel(filled, h=h, origin=origin)
    ids, tris, area_w = _surface_areas(model)
    fixed, applied, loads = [], [], np.zeros((model.n_nodes, 3))
    for s in p.get("supports") or []:
        m = select(model, ids, s.get("where") or {}, h)
        if not m.any():
            raise ValueError(f"support {s.get('where')} selects no surface of the part")
        fixed.append(ids[m])
    if not fixed:
        raise ValueError("at least one support is needed")
    fixed = np.unique(np.concatenate(fixed))
    for ld in p.get("loads") or []:
        m = select(model, ids, ld.get("where") or {}, h)
        if not m.any():
            raise ValueError(f"load {ld.get('where')} selects no surface of the part")
        F = np.asarray(ld.get("force_N") or [0, 0, 0], float)
        w = area_w[m] / area_w[m].sum()
        loads[ids[m]] += w[:, None] * F[None, :]
        applied.append({"force_N": F.round(3).tolist(), "nodes": int(m.sum()),
                        "area_mm2": round(float(area_w[m].sum()), 2), "where": ld.get("where")})
    if not applied:
        raise ValueError("at least one load is needed")
    progress("Solving", 0.2)
    r = model.solve(mat["E"], mat["nu"], fixed, loads,
                    progress=lambda it, res: progress("Solving", min(0.85, 0.2 + 0.65 * it / (it + 400))))
    vm = r["von_mises"]
    disp = np.linalg.norm(r["u"], axis=1)
    reac = r["reactions"][fixed].sum(0)
    app = loads.sum(0)
    bal = float(np.linalg.norm(reac + app) / (np.abs(loads).sum() or 1.0))
    vmax = float(vm.max())
    v995 = float(np.percentile(r["von_mises_elem"], 99.5))
    sf = mat["Sy"] / max(vmax, 1e-12)
    vvox = float(model.evol.sum())
    pos = model.nodes[ids]
    surface = {"positions": np.round(pos, 2).ravel().tolist(), "indices": tris.ravel().tolist(),
               "von_mises": np.round(vm[ids], 2).tolist(),
               "displacement": np.round(r["u"][ids], 5).ravel().tolist()}
    return {
        "surface": surface, "material": mat, "yield_mpa": mat["Sy"],
        "max_von_mises_MPa": vmax, "von_mises_p995_MPa": v995, "max_displacement_mm": float(disp.max()),
        "min_safety_factor": sf,
        "mesh": {"type": "hex8-im voxel", "element_size_mm": round(float(h), 4), "elements": ne,
                 "nodes": model.n_nodes, "surface_triangles": int(len(tris)), "fixed_nodes": int(len(fixed))},
        "volume": {"mesh_mm3": round(vol, 2), "voxel_mm3": round(vvox, 2),
                   "error_pct": round(100 * (vvox / vol - 1), 2)},
        "mass_kg": vol * 1e-9 * mat["rho"], "applied_loads": applied,
        "reaction_N": (-reac).round(3).tolist(), "equilibrium_error": bal,
        "cg_iterations": r["iterations"], "solve_time_s": round(time.perf_counter() - t0, 2),
    }
