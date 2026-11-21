"""Manufacturability of a free-form part, measured from its mesh (v0.39, DFM for any part).

The rules live in `physics/dfm_rules.py` and carry a source per number. What that module cannot do is see the
geometry: it reads a parameter list. Here the geometry is read instead, so the same rules can be applied to a
part nobody parametrised:

    wall thickness   ray cast from the surface along its inward normal to the first OTHER triangle it meets
                     (Moller-Trumbore, accelerated by a uniform triangle grid)
    overhang         FDM faces whose normal is more than `overhang_deg` off the build direction
    draft            cast faces whose normal is almost perpendicular to the pull direction
    holes            diameters declared by the caller, checked against the DIN 338 / ISO 235 drill series

Every number below is measured, and every limit comes from `dfm_rules` with its source and source status. A
measurement has a resolution: it is reported next to the number, never hidden. The exact cast is read to the
mesh's own accuracy; when a mesh is too large for it, the voxel fallback reports the cell it used, and a check
whose limit is finer than a quarter of that cell is `not_checked`, not a pass. Checks that need more than a
triangle mesh (undercuts, tool access, inside corner radius from the sketch) stay `not_checked` with the reason --
never a pass.

v0.39.1: the grid edge used to come from the SHORTEST SIDE OF THE BOUNDING BOX (`size.min()/8`), which is not a
wall; an 80 x 60 x 60 bracket was measured with a 7.5 mm cell, coarser than the 5 mm wall it was asked about.
The measurement no longer goes through a grid at all.
"""
from __future__ import annotations

import numpy as np

from . import dfm_rules
from .part_fea import mesh_volume, triangles_from_input, voxelize

MAX_SAMPLES = 4000
MAX_STEPS = 260                  # voxel fallback only
VOXEL_TARGET = 2_000_000         # voxels in the fallback grid
VOXEL_AXIS_CELLS = 400           # fallback cell = diagonal / 400, coarsened to stay inside the voxel budget
EXACT_MAX_TRIANGLES = 150_000    # above this the exact cast is slower than the request budget allows
GRID_ENTRY_BUDGET = 2_000_000    # triangle -> cell entries in the acceleration grid
MIN_RESOLUTION_MM = 1e-4         # smallest resolution reported; below this the mesh itself is the limit
BUILD_DIR = (0.0, 0.0, 1.0)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _normals_areas(tri):
    """Valid triangles, their outward normals and areas (degenerate triangles dropped)."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-12
    tri, n, ln = tri[keep], n[keep], ln[keep]
    return tri, n / ln[:, None], 0.5 * ln


class _TriGrid:
    """Uniform grid of triangle bounding boxes, for exact ray casting in numpy.

    Built once per mesh. Every triangle is registered in each cell its bounding box touches; asking a cell for
    its triangles is one binary search, so a ray can be marched cell by cell without a Python object per triangle.
    """

    def __init__(self, tri):
        self.tri = tri
        pts = tri.reshape(-1, 3)
        self.lo, self.hi = pts.min(0), pts.max(0)
        ext = np.maximum(self.hi - self.lo, 1e-9)
        n = max(1, int(round((len(tri) / 6.0) ** (1.0 / 3.0))))
        h = float(ext.max()) / n
        for _ in range(24):
            self.shape = np.maximum(1, np.ceil(ext / h).astype(int))
            i0 = np.clip(np.floor((tri.min(1) - self.lo) / h).astype(int), 0, self.shape - 1)
            i1 = np.clip(np.floor((tri.max(1) - self.lo) / h).astype(int), 0, self.shape - 1)
            span = i1 - i0 + 1
            total = int(np.prod(span, axis=1).sum())
            if total <= GRID_ENTRY_BUDGET:
                break
            h *= 1.5
        self.h = h
        per = np.prod(span, axis=1)
        start = np.cumsum(per) - per
        tid = np.repeat(np.arange(len(tri)), per)
        off = np.arange(total) - np.repeat(start, per)
        sy = span[tid, 1].astype(np.int64)
        sz = span[tid, 2].astype(np.int64)
        di = off // (sy * sz)
        rem = off - di * sy * sz
        dj = rem // sz
        dk = rem - dj * sz
        ii = i0[tid, 0] + di
        jj = i0[tid, 1] + dj
        kk = i0[tid, 2] + dk
        key = (ii * self.shape[1] + jj) * self.shape[2] + kk
        order = np.argsort(key, kind="stable")
        self.keys = key[order]
        self.tids = tid[order]

    def cell(self, i, j, k):
        key = (i * self.shape[1] + j) * self.shape[2] + k
        a = int(np.searchsorted(self.keys, key, "left"))
        b = int(np.searchsorted(self.keys, key, "right"))
        return self.tids[a:b] if b > a else None

    def first_hit(self, o, d, t_max, t_min):
        """Distance from `o` along `d` to the nearest triangle, or None if the ray leaves the grid first."""
        d_safe = np.where(np.abs(d) < 1e-12, 1e-12, d)      # keeps the slab test finite for an axis-parallel ray
        ta, tb = (self.lo - o) / d_safe, (self.hi - o) / d_safe
        t_enter = max(float(np.max(np.minimum(ta, tb))), 0.0)
        t_exit = min(float(np.min(np.maximum(ta, tb))), float(t_max))
        if t_enter > t_exit:
            return None
        cell = np.clip(np.floor((o + d * t_enter - self.lo) / self.h).astype(int), 0, self.shape - 1)
        step = np.where(np.abs(d) < 1e-12, 0, np.where(d > 0, 1, -1)).astype(int)
        edge = self.lo + (cell + (step > 0)) * self.h
        with np.errstate(divide="ignore", invalid="ignore"):
            t_next = np.where(step != 0, (edge - o) / d_safe, np.inf)
            t_delta = np.where(step != 0, self.h / np.abs(d_safe), np.inf)
        t_cell = t_enter
        while True:
            cand = self.cell(int(cell[0]), int(cell[1]), int(cell[2]))
            t_stop = min(float(np.min(t_next)), t_exit)
            if cand is not None:
                a, b, c = self.tri[cand, 0], self.tri[cand, 1], self.tri[cand, 2]
                t, hit = _ray_tri(o, d, a, b, c, max(t_cell, t_min))
                # a triangle registered in this cell may reach into a later one: only a hit inside this cell counts
                hit &= t <= t_stop + 1e-12 * max(1.0, abs(t_stop))
                if hit.any():
                    return float(t[hit].min())
            k = int(np.argmin(t_next))
            if t_next[k] > t_exit:
                return None
            cell[k] += step[k]
            if cell[k] < 0 or cell[k] >= self.shape[k]:
                return None
            t_cell = float(t_next[k])
            t_next[k] += t_delta[k]


def _ray_tri(o, d, a, b, c, t_min):
    """Moller-Trumbore, vectorised over triangles. -> (t, hit)."""
    e1 = b - a
    e2 = c - a
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - a
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_min)
    return t, hit


def _exact_wall(tri, cen, d):
    """Exact ray cast from each sample into the material. -> (t, measured, resolution_mm)."""
    diag = float(np.linalg.norm(tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0)))
    eps = max(diag * 1e-9, 1e-9)                       # step off the triangle we start on
    o = cen + d * eps
    grid = _TriGrid(tri)
    t = np.full(len(o), np.inf)
    for i in range(len(o)):
        hit = grid.first_hit(o[i], d[i], diag, eps)
        if hit is not None:
            t[i] = hit
    return t, np.isfinite(t), max(eps, MIN_RESOLUTION_MM)


def _voxel_wall(tri, cen, d):
    """Fallback for a mesh too large for the exact cast: march the ray through a voxel grid.

    -> (t, measured, resolution_mm). The cell is diagonal / 400 (coarsened to fit the voxel budget), never a
    fraction of the bounding box's shortest side.
    """
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    size = np.maximum(hi - lo, 1e-6)
    diag = float(np.linalg.norm(size))
    h = max(diag / VOXEL_AXIS_CELLS, 0.02)
    for _ in range(40):
        n = np.ceil(size / h).astype(int)
        if float(np.prod(n)) <= VOXEL_TARGET:
            break
        h *= 1.3
    filled, origin = voxelize(tri, h)
    shape = np.array(filled.shape)

    def occ(p):
        ij = np.floor((p - origin) / h).astype(int)
        ok = np.all((ij >= 0) & (ij < shape), axis=1)
        out = np.zeros(len(p), bool)
        out[ok] = filled[ij[ok, 0], ij[ok, 1], ij[ok, 2]]
        return out, ok

    d = d.copy()
    p = cen + d * (0.5 * h)
    inside, okg = occ(p)
    wrong = okg & ~inside                            # triangles wound the other way: flip, do not drop
    d[wrong] *= -1
    p[wrong] = cen[wrong] + d[wrong] * (0.5 * h)
    inside, okg = occ(p)
    alive = inside & okg
    t = np.full(len(p), 0.5 * h)
    for _ in range(MAX_STEPS):
        if not alive.any():
            break
        p[alive] += d[alive] * h
        inp, ok = occ(p)
        t[alive] += h
        alive &= inp & ok
    measured = np.isfinite(t) & (t < 0.5 * h + MAX_STEPS * h)
    return t, measured, float(h)


def wall_thickness(tri, max_samples: int = MAX_SAMPLES, method: str | None = None) -> tuple:
    """Local wall thickness: from the surface inward to the first other surface of the part.

    `method` is "exact" (ray-triangle, the default) or "voxel" (the fallback, automatically used above
    EXACT_MAX_TRIANGLES). The exact value is as accurate as the mesh; the voxel value is quantised by the cell
    it reports as `resolution_mm`.
    """
    tri, nrm, area = _normals_areas(tri)
    if not len(tri):
        raise ValueError("the mesh has no usable triangle")
    # a consistently wound mesh has outward normals; a mesh wound the other way would send every ray outward
    signed = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    if signed < 0:
        nrm = -nrm
    s = len(tri)
    # sample by AREA, with replacement: a big flat face must weigh more than a small triangle on a chamfer.
    # Uniform-per-triangle sampling would report the thin direction of a plate as its long one.
    rng = np.random.default_rng(0)
    pick = rng.choice(s, size=max_samples, replace=True, p=area / area.sum())
    cen = tri[pick].mean(1)
    d = -nrm[pick]
    use = method or ("voxel" if s > EXACT_MAX_TRIANGLES else "exact")
    if use == "voxel":
        t, measured, h = _voxel_wall(tri, cen, d)
        how = (f"voxel fallback: ray cast through a voxel grid of {h:g} mm cells; each value is quantised by "
               f"resolution_mm (+/- one cell)")
    else:
        t, measured, h = _exact_wall(tri, cen, d)
        how = ("exact ray-triangle intersection (Moller-Trumbore) from the surface along its inward normal, "
               "accelerated by a uniform triangle grid; the value is as accurate as the mesh")
    vals = t[measured]
    cap = float(np.linalg.norm(tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0)))
    out = {
        "method": how,
        "method_id": "exact-ray-triangle" if use == "exact" else "voxel-fallback",
        "resolution_mm": round(float(h), 6),
        "samples": int(len(vals)),
        "samples_total": int(len(pick)),
        "unresolved": int((~measured).sum()),
        "min_mm": round(float(vals.min()), 3) if len(vals) else None,
        "p5_mm": round(float(np.percentile(vals, 5)), 3) if len(vals) else None,
        "median_mm": round(float(np.median(vals)), 3) if len(vals) else None,
        "max_mm": round(float(vals.max()), 3) if len(vals) else None,
        "at_least_mm": round(float(0.5 * h + MAX_STEPS * h), 3) if use == "voxel" else None,
    }
    return out, tri[pick], t, measured, h


def overhang(tri, build=(0, 0, 1), overhang_deg: float = 45.0) -> dict:
    """FDM overhang: faces that lean under the build direction by more than `overhang_deg`."""
    tri, nrm, area = _normals_areas(tri)
    b = _unit(build)
    with np.errstate(invalid="ignore"):
        ang = np.degrees(np.arccos(np.clip(-nrm @ b, -1.0, 1.0)))   # 0 deg = facing straight down
    bad = ang < float(overhang_deg)
    tot = float(area.sum()) or 1.0
    return {"build_direction": [round(float(x), 3) for x in b],
            "threshold_deg": float(overhang_deg),
            "area_mm2": round(float(area[bad].sum()), 2),
            "area_fraction": round(float(area[bad].sum()) / tot, 4),
            "faces": int(bad.sum()),
            "worst_deg": round(float(ang.min()), 2) if len(ang) else None}


def draft(tri, pull=(0, 0, 1), draft_deg: float = 1.0) -> dict:
    """Cast: faces whose normal is within `draft_deg` of perpendicular to the pull direction will rub."""
    tri, nrm, area = _normals_areas(tri)
    p = _unit(pull)
    with np.errstate(invalid="ignore"):
        tilt = np.degrees(np.arcsin(np.clip(np.abs(nrm @ p), -1.0, 1.0)))   # 0 deg = a wall parallel to the pull
    bad = tilt < float(draft_deg)
    tot = float(area.sum()) or 1.0
    return {"pull_direction": [round(float(x), 3) for x in p],
            "draft_deg": float(draft_deg),
            "area_mm2": round(float(area[bad].sum()), 2),
            "area_fraction": round(float(area[bad].sum()) / tot, 4),
            "faces": int(bad.sum())}


def _check(cid, name, status, *, value=None, limit=None, unit=None, source=None, source_status=None, message=""):
    out = {"id": cid, "check": name, "status": status, "unit": unit, "message": message}
    if value is not None:
        out["value"] = value
    if limit is not None:
        out["limit"] = limit
    if source:
        out["source"] = source
    if source_status:
        out["source_status"] = source_status
    return out


def measure(payload: dict) -> dict:
    """-> the measured part + the rule findings. Raises ValueError when the mesh cannot be used."""
    tri = triangles_from_input(payload)
    vol = mesh_volume(tri)
    if vol <= 0:
        raise ValueError("the geometry has no volume (is the mesh closed?)")
    process = str(payload.get("process") or "cnc").lower()
    material = str(payload.get("material") or "aluminium").lower()

    wall, _samp, per_sample, measured, res = wall_thickness(tri)
    checks: list[dict] = []

    # 1) minimum wall, from the rules table, against the thickness measured here
    limit_wall = (dfm_rules.WALL_MIN_MM.get(process) or {}).get(material)
    rule = dfm_rules.WALL_SOURCE.get(process) or {}
    ref = wall["p5_mm"] if wall["p5_mm"] is not None else wall["min_mm"]
    if ref is None:
        checks.append(_check("min_wall", "minimum wall (measured)", "not_checked",
                             message="no wall could be measured from this mesh"
                                     + (f": {wall['unresolved']} of {wall['samples_total']} rays hit no other "
                                        f"surface, so the mesh is not closed along them"
                                        if wall["unresolved"] else "")))
    elif limit_wall is None:
        checks.append(_check("min_wall", "minimum wall (measured)", "not_checked", value=ref, unit="mm",
                             message=f"no minimum-wall rule for process '{process}' and material "
                                     f"'{material}'; not guessed"))
    elif float(res) > float(limit_wall) / 4.0:
        # a measurement coarser than a quarter of the limit can neither pass nor fail the wall
        checks.append(_check("min_wall", "minimum wall (measured)", "not_checked", value=ref, unit="mm",
                             limit=limit_wall, source=rule.get("text"), source_status=rule.get("status"),
                             message=f"measurement too coarse for this wall limit: resolution {res:g} mm is "
                                     f"more than a quarter of the {limit_wall:g} mm limit"))
    else:
        ok = ref + float(res) + 1e-9 >= limit_wall      # the measurement's resolution is allowed for
        status = "pass" if ok else ("fail" if rule.get("status") == dfm_rules.VERIFIED else "warn")
        msg = (f"measured 5th-percentile wall {ref:g} mm (resolution {res:g} mm) >= {limit_wall:g} mm for "
               f"{process} {material}" if ok else
               f"measured 5th-percentile wall {ref:g} mm (resolution {res:g} mm) is below the {limit_wall:g} mm "
               f"{'minimum' if status == 'fail' else 'guideline'} for {process} {material}")
        if status == "warn":
            msg += "; the guideline's source is not verified, so this is a warning, not a failure"
        checks.append(_check("min_wall", "minimum wall (measured)", status, value=ref, limit=limit_wall,
                             unit="mm", source=rule.get("text"), source_status=rule.get("status"), message=msg))

    # 2) hole diameters: declared holes are checked against the drill series; the mesh is not asked to
    #    recognise a hole, because a faceted bore can also be a slot, a fillet or a tessellation artefact
    holes = [x for x in (payload.get("holes") or []) if isinstance(x, dict)]
    if holes:
        rule_r = dfm_rules.check({"process": process, "material": material, "params": [], "holes": holes})
        drill = [c for c in rule_r["checks"] if c["id"] == "drill_size"]
        checks.extend(drill)
    else:
        checks.append(_check("drill_size", "hole diameters in the drill series", "not_checked",
                             message="no holes were declared; a hole cannot be recognised reliably in a "
                                     "triangle mesh, so send {\"holes\": [{\"d\": ...}]} to have them checked"))

    # 3) process-specific geometry
    if process == "fdm":
        oh = overhang(tri, payload.get("build_direction") or BUILD_DIR,
                      float(payload.get("overhang_deg") or 45.0))
        msg = (f"{100 * oh['area_fraction']:.1f} % of the surface ({oh['area_mm2']:g} mm2) needs support: "
               f"faces lean more than {oh['threshold_deg']:g} deg from the build direction")
        checks.append(_check("overhang", "supported overhang", "warn" if oh["area_fraction"] > 0.02 else "pass",
                             unit="fraction", value=oh["area_fraction"],
                             source="FDM practice: 45 deg from vertical is printable without support. A rule of "
                                    "thumb, not a clause in a standard.",
                             source_status=dfm_rules.UNVERIFIED, message=msg))
    else:
        oh = None
        for cid, why in (("overhang", "overhang is an FDM rule; this part is not FDM"),
                         ("support", "support is an FDM rule; this part is not FDM")):
            checks.append(_check(cid, cid, "not_checked", message=why))

    if process == "cast":
        dr = draft(tri, payload.get("pull_direction") or BUILD_DIR, float(payload.get("draft_deg") or 1.0))
        msg = (f"{100 * dr['area_fraction']:.1f} % of the surface ({dr['area_mm2']:g} mm2) is within "
               f"{dr['draft_deg']:g} deg of parallel to the pull direction, so it needs draft")
        checks.append(_check("draft", "draft on walls parallel to the pull", "warn" if dr["area_fraction"] else "pass",
                             value=dr["area_fraction"], unit="fraction",
                             source="Casting practice: 1-3 deg draft on walls parallel to the pull. A workshop "
                                    "guideline, not a clause in a standard.",
                             source_status=dfm_rules.UNVERIFIED, message=msg))
    else:
        dr = None
        checks.append(_check("draft", "draft", "not_checked", message="draft is a casting rule; this part is not cast"))

    for cid, why in (("undercuts", "undercuts need the solid and the tool approach; they are not measurable from a "
                                   "triangle mesh alone"),
                     ("inside_corner_radius", "the inside corner radius would have to be recognised in the mesh; "
                                              "the sketch that made it is not sent here")):
        checks.append(_check(cid, cid.replace("_", " "), "not_checked", message=why))

    failed = [c for c in checks if c["status"] == "fail"]
    warned = [c for c in checks if c["status"] == "warn"]

    # 3D contour: per-vertex measured thickness, so the Studio can colour the model
    tri_all, _n, _a = _normals_areas(tri)
    pos_all = tri_all.reshape(-1, 3)
    pos_s = _samp.reshape(-1, 3)               # the sampled triangles, with replacement
    key_all = np.round(pos_all / 1e-6).astype(np.int64)
    key_s = np.round(pos_s / 1e-6).astype(np.int64)
    _u, inv = np.unique(np.vstack([key_all, key_s]), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)          # numpy 2 returns it shaped like the input
    inv_s = inv[len(key_all):]
    vert_t = np.full(len(_u), np.inf)
    assigned = per_sample.copy()
    assigned[~measured] = np.inf
    np.minimum.at(vert_t, inv_s, np.repeat(assigned, 3))
    vmin = float(vert_t[np.isfinite(vert_t)].min()) if np.isfinite(vert_t).any() else None
    vmax = float(vert_t[np.isfinite(vert_t)].max()) if np.isfinite(vert_t).any() else None
    thickness = [None if not np.isfinite(v) else round(float(v), 3) for v in vert_t]

    return {
        "ok": not failed,
        "process": process, "material": material,
        "volume_mm3": round(vol, 2), "surface_area_mm2": round(float(_a.sum()), 2),
        "wall": wall,
        "overhang": oh, "draft": dr,
        "checks": checks, "failed": failed, "warned": warned,
        "not_checked": [c for c in checks if c["status"] == "not_checked"],
        "surface": {"positions": np.round(_u / 1e6, 3).ravel().tolist(),
                    "thickness": thickness, "min_mm": vmin, "max_mm": vmax},
    }


def summary(m: dict) -> str:
    """One line for an answer or a report."""
    if not isinstance(m, dict):
        return "Manufacturability: not measured."
    wall = m.get("wall") or {}
    bits = [f"measured wall {wall.get('min_mm')}-{wall.get('median_mm')} mm (5th pct {wall.get('p5_mm')} mm, "
            f"{wall.get('method_id') or 'measured'}, resolution {wall.get('resolution_mm')} mm)"]
    failed = m.get("failed") or []
    warned = m.get("warned") or []
    if failed:
        bits.append("FAILED — " + "; ".join(f"{c['check']}: {c['message']}" for c in failed[:3]))
    if warned:
        bits.append(f"{len(warned)} warning(s) — " + "; ".join(f"{c['check']}: {c['message']}" for c in warned[:2]))
    if not failed and not warned:
        bits.append("every check that could be made passed")
    return "Manufacturability (" + m.get("process", "?") + "/" + m.get("material", "?") + "): " + "; ".join(bits) + "."
