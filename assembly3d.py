"""
Idris Core — 3D assembly builder (v3)
======================================

Turns one gearbox variant (module, teeth, face width, stages, housing envelope)
into a set of closed triangle meshes, one per part:

  * Housing        hollow box, wall t, with a shaft bore in each end plate
                   (front and back bores may be at different positions)
  * Gears          true involute spur-gear teeth (20 deg pressure angle,
                   addendum m, dedendum 1.25 m), phased so they mesh
  * Shafts         plain cylinders sized from torsion (physics.gears)

Everything is built directly from the numbers with NumPy only (no CAD kernel).
Shaft axes are parallel to Z; housing frame: x in [0, W], y in [0, H], z in [0, D].

What is NOT modelled (honest limits of this POC): tooth root fillets, keyways,
bearings, seals, fasteners, gear hubs/webs. Shafts overlap the gear bores exactly
(they are separate bodies, as in a real assembly before boolean union).
"""

from __future__ import annotations

import math

import numpy as np

from physics.gears import ALPHA, CLEARANCE, ETA_STAGE, STAGE_GAP, shaft_diameter_mm

TWO_PI = 2.0 * math.pi
SHAFT_STICKOUT = 22.0       # mm the shafts stick out of the housing


# --------------------------------------------------------------------------- mesh helpers
def _orient(verts: np.ndarray, tris: np.ndarray, expected) -> np.ndarray:
    """Flip winding so that triangle normals agree (in sum) with expected(centroids)."""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    cen = (p0 + p1 + p2) / 3.0
    if np.einsum("ij,ij->i", nrm, expected(cen)).sum() < 0:
        tris = tris[:, [0, 2, 1]]
    return tris


def _band(verts, a, b, expected) -> np.ndarray:
    """Triangulate the strip between two equal-length closed index loops."""
    n = len(a)
    t = []
    for i in range(n):
        j = (i + 1) % n
        t.append((a[i], a[j], b[j]))
        t.append((a[i], b[j], b[i]))
    return _orient(verts, np.array(t, dtype=np.int64), expected)


def _stitch(verts, loop_a, loop_b, center, expected) -> np.ndarray:
    """
    Triangulate a planar ring between two closed loops of (possibly) different vertex
    counts, both star-shaped around `center` (x, y). Vertices are merged by polar angle,
    so the two loops must be angularly dense (a few degrees per step).
    """
    def prep(idx):
        ang = np.arctan2(verts[idx, 1] - center[1], verts[idx, 0] - center[0]) % TWO_PI
        ang = np.round(ang, 9)                     # exact ties (radial segments) must keep their loop order
        ang = np.where(ang >= TWO_PI - 1e-9, 0.0, ang)
        order = np.lexsort((np.arange(len(ang)), ang))
        return np.asarray(idx)[order], ang[order]

    ia, aa = prep(np.asarray(loop_a))
    ib, ab = prep(np.asarray(loop_b))
    # rotate loop b so that it starts at the last vertex whose angle <= aa[0]
    k = np.searchsorted(ab, aa[0], side="right") - 1
    if k < 0:
        k = len(ab) - 1
    ib = np.roll(ib, -k)
    ab = np.roll(ab, -k)
    # unwrap angles so they increase monotonically starting from loop start
    def unwrap(a):
        a = a.copy()
        for i in range(1, len(a)):
            while a[i] < a[i - 1] - 1e-12:
                a[i] += TWO_PI
        return a
    aa = unwrap(aa)
    ab = unwrap(ab)
    na, nb = len(ia), len(ib)
    ext_a = np.append(aa, aa[0] + TWO_PI)
    ext_b = np.append(ab, ab[0] + TWO_PI)
    i = j = 0
    tris = []
    while i < na or j < nb:
        adv_a = i < na and (j >= nb or ext_a[i + 1] <= ext_b[j + 1] + 1e-12)
        if adv_a:
            tris.append((ia[i % na], ia[(i + 1) % na], ib[j % nb]))
            i += 1
        else:
            tris.append((ib[j % nb], ia[i % na], ib[(j + 1) % nb]))
            j += 1
    return _orient(verts, np.array(tris, dtype=np.int64), expected)


def _const(v):
    arr = np.array(v, dtype=float)
    return lambda c: np.tile(arr, (len(c), 1))


def _radial(cx, cy, sign=1.0):
    return lambda c: sign * np.stack([c[:, 0] - cx, c[:, 1] - cy, np.zeros(len(c))], axis=1)


def mesh_volume_mm3(verts: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))


def _circle(cx, cy, r, z, n=48, rot=0.0):
    a = np.linspace(0, TWO_PI, n, endpoint=False) + rot
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)], axis=1)


class _Builder:
    """Collects vertex blocks and faces into one mesh."""
    def __init__(self):
        self.v = []
        self.f = []
        self.n = 0

    def add(self, pts: np.ndarray) -> np.ndarray:
        idx = np.arange(self.n, self.n + len(pts))
        self.v.append(pts)
        self.n += len(pts)
        return idx

    def verts(self):
        return np.vstack(self.v)

    def face(self, tris):
        self.f.append(tris)

    def result(self):
        return self.verts(), np.vstack(self.f)


# --------------------------------------------------------------------------- housing
def _rect_points(x0, x1, y0, y1, centers, segs, z) -> np.ndarray:
    """
    Points on the rectangle boundary: its 4 corners + the hits of rays cast from every
    given centre at `segs` evenly spaced angles. Sorted counter-clockwise along the
    perimeter, de-duplicated. Returns (N, 3).
    """
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for (cx, cy) in centers:
        for a in np.linspace(0, TWO_PI, segs, endpoint=False):
            c, s = math.cos(a), math.sin(a)
            tx = ((x1 - cx) / c if c > 1e-12 else (x0 - cx) / c) if abs(c) > 1e-12 else math.inf
            ty = ((y1 - cy) / s if s > 1e-12 else (y0 - cy) / s) if abs(s) > 1e-12 else math.inf
            d = min(tx, ty)
            pts.append((cx + d * c, cy + d * s))
    w, h = x1 - x0, y1 - y0
    per = 2 * (w + h)

    def s_of(p):
        x, y = p
        if abs(y - y0) < 1e-7:
            return x - x0
        if abs(x - x1) < 1e-7:
            return w + (y - y0)
        if abs(y - y1) < 1e-7:
            return w + h + (x1 - x)
        return 2 * w + h + (y1 - y)

    pts.sort(key=s_of)
    keep = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - keep[-1][0], p[1] - keep[-1][1]) > 1e-6:
            keep.append(p)
    if math.hypot(keep[0][0] - keep[-1][0], keep[0][1] - keep[-1][1]) < 1e-6:
        keep.pop()
    return np.array([(x, y, z) for x, y in keep], dtype=float)


def build_housing(W, H, D, t, front_bore, back_bore, segs=96):
    """
    Hollow box with one circular bore in each end plate.
    front_bore / back_bore = (cx, cy, diameter) in the housing frame.
    Returns (verts, faces, bore_polygon_area_mm2_list).
    """
    for nm, (cx, cy, d) in (("front", front_bore), ("back", back_bore)):
        r = d / 2.0
        if d <= 0:
            raise ValueError(f"{nm} bore diameter must be positive")
        if cx - r < t * 1.0 or cx + r > W - t or cy - r < t or cy + r > H - t:
            raise ValueError(f"{nm} bore does not fit inside the housing walls")
    if 2 * t >= min(W, H, D):
        raise ValueError("wall_thickness too large for this envelope")

    b = _Builder()
    centers = [(front_bore[0], front_bore[1]), (back_bore[0], back_bore[1])]
    o_pts = lambda z: _rect_points(0, W, 0, H, centers, segs, z)
    i_pts = lambda z: _rect_points(t, W - t, t, H - t, centers, segs, z)
    O0, O1 = b.add(o_pts(0.0)), b.add(o_pts(D))
    I0, I1 = b.add(i_pts(t)), b.add(i_pts(D - t))
    fx, fy, fd = front_bore
    bx, by, bd = back_bore
    C0, B0 = b.add(_circle(fx, fy, fd / 2, 0.0, segs)), b.add(_circle(fx, fy, fd / 2, t, segs))
    C1, B1 = b.add(_circle(bx, by, bd / 2, D, segs)), b.add(_circle(bx, by, bd / 2, D - t, segs))
    V = b.verts()
    cxr, cyr = W / 2, H / 2
    b.face(_stitch(V, O0, C0, (fx, fy), _const((0, 0, -1))))            # outer front plate
    b.face(_stitch(V, O1, C1, (bx, by), _const((0, 0, 1))))             # outer back plate
    b.face(_band(V, O0, O1, _radial(cxr, cyr, 1)))                      # outer side walls
    b.face(_band(V, C0, B0, _radial(fx, fy, -1)))                       # front bore wall
    b.face(_band(V, C1, B1, _radial(bx, by, -1)))                       # back bore wall
    b.face(_stitch(V, I0, B0, (fx, fy), _const((0, 0, 1))))             # cavity floor
    b.face(_stitch(V, I1, B1, (bx, by), _const((0, 0, -1))))            # cavity ceiling
    b.face(_band(V, I0, I1, _radial(cxr, cyr, -1)))                     # cavity side walls
    verts, faces = b.result()
    return verts, faces


# --------------------------------------------------------------------------- gears
def _inv(a):
    return math.tan(a) - a


def gear_profile(m: float, z: int, rot: float = 0.0, flank_pts: int = 8):
    """
    Closed 2-D tooth outline (counter-clockwise), standard 20 deg full-depth involute.
    Returns (angles, radii) arrays and the geometry dict.
    """
    rp = m * z / 2.0
    rb = rp * math.cos(ALPHA)
    ra = rp + m
    rf = rp - 1.25 * m
    inv_p = _inv(ALPHA)
    half_pitch = math.pi / (2.0 * z)

    def psi(r):                      # half tooth angle at radius r
        r = max(r, rb)
        return half_pitch + inv_p - _inv(math.acos(rb / r))

    while psi(ra) < 0.004 and ra > rp:          # keep the tip from becoming a knife edge
        ra -= 0.01 * m
    r0 = max(rb, rf)
    rs = np.linspace(r0, ra, flank_pts)
    ang, rad = [], []
    for i in range(z):
        phi = TWO_PI * i / z + rot
        if rf < r0:                                  # radial segment from root up to the base circle
            ang.append(phi - psi(r0)); rad.append(rf)
        for r in rs:                                 # right flank
            ang.append(phi - psi(r)); rad.append(r)
        for a_ in np.linspace(phi - psi(ra), phi + psi(ra), 4)[1:-1]:   # tip land
            ang.append(a_); rad.append(ra)
        for r in rs[::-1]:                           # left flank
            ang.append(phi + psi(r)); rad.append(r)
        if rf < r0:
            ang.append(phi + psi(r0)); rad.append(rf)
        nxt = TWO_PI * (i + 1) / z + rot - psi(r0)   # root arc to the next tooth
        start = phi + psi(r0)
        for a_ in np.linspace(start, nxt, 5)[1:-1]:
            ang.append(a_); rad.append(rf)
    geo = {"rp": rp, "rb": rb, "ra": ra, "rf": rf}
    return np.array(ang), np.array(rad), geo


def build_gear(m, z, b_face, bore_d, cx, cy, z0, rot=0.0):
    """Closed mesh of one spur gear; axis parallel to Z, front face at z0, back face at z0 + b_face."""
    ang, rad, geo = gear_profile(m, z, rot)
    n = len(ang)
    xs, ys = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
    P0 = np.stack([xs, ys, np.full(n, z0)], axis=1)
    P1 = np.stack([xs, ys, np.full(n, z0 + b_face)], axis=1)
    nb = 48
    b = _Builder()
    ia, ib = b.add(P0), b.add(P1)
    ja = b.add(_circle(cx, cy, bore_d / 2, z0, nb))
    jb = b.add(_circle(cx, cy, bore_d / 2, z0 + b_face, nb))
    V = b.verts()
    b.face(_stitch(V, ia, ja, (cx, cy), _const((0, 0, -1))))            # front face
    b.face(_stitch(V, ib, jb, (cx, cy), _const((0, 0, 1))))             # back face
    b.face(_band(V, ia, ib, _radial(cx, cy, 1)))                        # tooth surface
    b.face(_band(V, ja, jb, _radial(cx, cy, -1)))                       # bore
    verts, faces = b.result()
    return verts, faces, geo, (xs, ys)


def build_shaft(d, cx, cy, z_start, z_end, n=48):
    """Solid cylinder with flat end caps."""
    b = _Builder()
    ia, ib = b.add(_circle(cx, cy, d / 2, z_start, n)), b.add(_circle(cx, cy, d / 2, z_end, n))
    ca, cb = b.add(np.array([[cx, cy, z_start]])), b.add(np.array([[cx, cy, z_end]]))
    V = b.verts()
    b.face(_band(V, ia, ib, _radial(cx, cy, 1)))
    cap = lambda loop, c, nz: _orient(V, np.array([(c[0], loop[i], loop[(i + 1) % n]) for i in range(n)]), _const(nz))
    b.face(cap(ia, ca, (0, 0, -1)))
    b.face(cap(ib, cb, (0, 0, 1)))
    return b.result()


# --------------------------------------------------------------------------- assembly
def build_assembly(design: dict, kinematics: dict, shaft: dict, housing_params: dict) -> dict:
    """
    design / kinematics / shaft / housing_params are the dicts produced by
    physics.variants.generate_variants() for one variant.
    Returns {"parts": [...], "warnings": [...], "bbox": {...}}.
    """
    m = float(design["module_mm"])
    zp, zg = int(design["pinion_teeth"]), int(design["gear_teeth"])
    b = float(design["face_width_mm"])
    stages = int(design["stages"])
    a = m * (zp + zg) / 2.0
    W = float(housing_params["envelope_width"])
    H = float(housing_params["envelope_height"])
    D = float(housing_params["envelope_depth"])
    t = float(housing_params["wall_thickness"])
    if not (0 < m <= 12 and 8 <= zp <= 200 and 8 <= zg <= 400 and 0 < b <= 400 and stages in (1, 2)):
        raise ValueError("unsupported gear parameters for 3D assembly")

    dap, dag = m * (zp + 2), m * (zg + 2)
    u = zg / zp
    t_in = float(kinematics["input_torque_nm"])
    d_in = shaft_diameter_mm(t_in)
    d_out = float(shaft["output_diameter_mm"])
    d_mid = shaft_diameter_mm(t_in * u * ETA_STAGE) if stages == 2 else None
    warnings = []

    yc = H / 2.0
    z1 = t + CLEARANCE
    if stages == 1:
        x_a = t + CLEARANCE + dap / 2.0
        x_b = x_a + a
        x_right = x_b + dag / 2.0 + CLEARANCE + t
        d_needed = b + 2 * CLEARANCE + 2 * t
    else:
        x_a = t + CLEARANCE + dag / 2.0
        x_b = x_a + a
        x_right = x_b + dag / 2.0 + CLEARANCE + t
        d_needed = 2 * b + STAGE_GAP + 2 * CLEARANCE + 2 * t
    if x_right > W + 1.0 or d_needed > D + 1.0 or max(dap, dag) + 2 * CLEARANCE + 2 * t > H + 1.0:
        raise ValueError("housing envelope is too small for this gear layout")
    shift = (W - x_right) / 2.0           # centre the gear train in the (ceil-rounded) envelope
    x_a += shift
    x_b += shift
    z2 = z1 + b + STAGE_GAP

    gear_meta, parts = {}, []

    def add_part(name, kind, verts, faces, **meta):
        parts.append({"name": name, "kind": kind, "verts": verts, "faces": faces, **meta})

    def gear(name, mm, z, bore, cx, z0, rot):
        v, f, geo, xy = build_gear(mm, z, b, bore, cx, yc, z0, rot)
        if geo["rf"] - bore / 2.0 < 2.0:
            warnings.append(f"{name}: root circle is only {geo['rf'] - bore / 2.0:.1f} mm from the bore — "
                            f"in practice this pinion would be cut directly on the shaft.")
        add_part(name, "gear", v, f, teeth=z, module_mm=mm, face_width_mm=b, bore_mm=bore,
                 pitch_diameter_mm=2 * geo["rp"], tip_diameter_mm=2 * geo["ra"])

    def shaft_part(name, d, cx, z_start, z_end):
        v, f = build_shaft(d, cx, yc, z_start, z_end)
        add_part(name, "shaft", v, f, diameter_mm=d, length_mm=z_end - z_start)

    if stages == 1:
        hv, hf = build_housing(W, H, D, t, (x_a, yc, d_in + 2), (x_b, yc, d_out + 2))
        add_part("Housing", "housing", hv, hf)
        shaft_part("Input shaft", d_in, x_a, -SHAFT_STICKOUT, z1 + b)
        shaft_part("Output shaft", d_out, x_b, z1, D + SHAFT_STICKOUT)
        gear("Pinion", m, zp, d_in, x_a, z1, 0.0)
        gear("Gear", m, zg, d_out, x_b, z1, math.pi - math.pi / zg)
    else:
        hv, hf = build_housing(W, H, D, t, (x_a, yc, d_in + 2), (x_a, yc, d_out + 2))
        add_part("Housing", "housing", hv, hf)
        shaft_part("Input shaft", d_in, x_a, -SHAFT_STICKOUT, z1 + b)
        shaft_part("Intermediate shaft", d_mid, x_b, 0.4 * t, D - 0.4 * t)
        shaft_part("Output shaft", d_out, x_a, z2, D + SHAFT_STICKOUT)
        gear("Pinion (stage 1)", m, zp, d_in, x_a, z1, 0.0)
        gear("Gear (stage 1)", m, zg, d_mid, x_b, z1, math.pi - math.pi / zg)
        gear("Pinion (stage 2)", m, zp, d_mid, x_b, z2, math.pi)
        gear("Gear (stage 2)", m, zg, d_out, x_a, z2, -math.pi / zg)

    allv = np.vstack([p["verts"] for p in parts])
    bbox = {"min": allv.min(axis=0).round(3).tolist(), "max": allv.max(axis=0).round(3).tolist()}
    return {"parts": parts, "warnings": warnings, "bbox": bbox,
            "layout": {"axis_a_mm": [round(x_a, 3), round(yc, 3)], "axis_b_mm": [round(x_b, 3), round(yc, 3)],
                       "center_distance_mm": round(a, 3), "shafts_mm": {"input": d_in, "intermediate": d_mid, "output": d_out}}}
