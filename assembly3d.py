"""
Idris Core — 3D assembly builder (v3)
======================================

Turns one gearbox variant (module, teeth, face width, stages, housing envelope)
into a set of closed triangle meshes, one per part:

  * Housing        body (front wall + side walls) + bolted end cover (back wall), wall t,
                   bores sized to the bearing outside diameter
  * Bearings       ring envelopes of the selected deep-groove bearings in the two through-bores
  * Gears          true involute spur-gear teeth (20 deg pressure angle,
                   addendum m, dedendum 1.25 m), phased so they mesh
  * Shafts         stepped (journal / seat) from the ACE shaft solver; torsion-only fallback

Everything is built directly from the numbers with NumPy only (no CAD kernel).
Shaft axes are parallel to Z; housing frame: x in [0, W], y in [0, H], z in [0, D].

What is NOT modelled (honest limits of this POC): tooth root fillets, keyways,
bearings, seals, fasteners, gear hubs/webs. Shafts overlap the gear bores exactly
(they are separate bodies, as in a real assembly before boolean union).
"""

from __future__ import annotations

import math
from physics.bolted_joint import METRIC

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


def build_housing_split(W, H, D, t, front_bore, back_bore, segs=96):
    """
    Assemblable housing (QA Q-001/Q-011): a cup-shaped BODY (front wall + side walls, open at the
    back) and a bolted END COVER (the back wall, z in [D - t, D]). Gears and shafts go in through the
    open back, then the cover closes it. Together they occupy exactly the closed box of build_housing.
    Returns ((body_verts, body_faces), (cover_verts, cover_faces)).
    """
    for nm, (cx, cy, d) in (("front", front_bore), ("back", back_bore)):
        r = d / 2.0
        if d <= 0:
            raise ValueError(f"{nm} bore diameter must be positive")
        if cx - r < t or cx + r > W - t or cy - r < t or cy + r > H - t:
            raise ValueError(f"{nm} bore does not fit inside the housing walls")
    if 2 * t >= min(W, H, D):
        raise ValueError("wall_thickness too large for this envelope")
    Db = D - t
    fx, fy, fd = front_bore
    bx, by, bd = back_bore
    cxr, cyr = W / 2, H / 2
    # body
    b = _Builder()
    centers = [(fx, fy)]
    O0, O1 = b.add(_rect_points(0, W, 0, H, centers + [(cxr, cyr)], segs, 0.0)), \
        b.add(_rect_points(0, W, 0, H, centers + [(cxr, cyr)], segs, Db))
    I0, I1 = b.add(_rect_points(t, W - t, t, H - t, centers + [(cxr, cyr)], segs, t)), \
        b.add(_rect_points(t, W - t, t, H - t, centers + [(cxr, cyr)], segs, Db))
    C0, B0 = b.add(_circle(fx, fy, fd / 2, 0.0, segs)), b.add(_circle(fx, fy, fd / 2, t, segs))
    V = b.verts()
    b.face(_stitch(V, O0, C0, (fx, fy), _const((0, 0, -1))))            # outer front plate
    b.face(_band(V, O0, O1, _radial(cxr, cyr, 1)))                      # outer side walls
    b.face(_band(V, C0, B0, _radial(fx, fy, -1)))                       # front bore wall
    b.face(_stitch(V, I0, B0, (fx, fy), _const((0, 0, 1))))             # cavity floor
    b.face(_band(V, I0, I1, _radial(cxr, cyr, -1)))                     # cavity side walls
    b.face(_stitch(V, O1, I1, (cxr, cyr), _const((0, 0, 1))))           # joint face (cover seats here)
    body = b.result()
    # cover
    c = _Builder()
    P0 = c.add(_rect_points(0, W, 0, H, [(bx, by)], segs, Db))
    P1 = c.add(_rect_points(0, W, 0, H, [(bx, by)], segs, D))
    K0, K1 = c.add(_circle(bx, by, bd / 2, Db, segs)), c.add(_circle(bx, by, bd / 2, D, segs))
    V = c.verts()
    c.face(_stitch(V, P1, K1, (bx, by), _const((0, 0, 1))))             # outer face
    c.face(_stitch(V, P0, K0, (bx, by), _const((0, 0, -1))))            # inner face
    c.face(_band(V, P0, P1, _radial(cxr, cyr, 1)))                      # rim
    c.face(_band(V, K0, K1, _radial(bx, by, -1)))                       # bore wall
    cover = c.result()
    return body, cover


def build_box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned closed box."""
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int64)
    return v, f


def _merge(meshes):
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


def build_plate_with_holes(x0, x1, y0, y1, z0, z1, holes, segs=64):
    """Closed plate x∈[x0,x1], y∈[y0,y1], z∈[z0,z1] with circular through-holes [(cx, cy, d)].
    Split into vertical strips with at most one hole each (every strip is star-shaped around its hole,
    so the ring triangulation of build_housing applies); strips touch, they do not overlap."""
    hs = sorted(holes, key=lambda h: h[0])
    cuts = [x0] + [0.5 * (hs[i][0] + hs[i + 1][0]) for i in range(len(hs) - 1)] + [x1]
    parts = []
    for i in range(len(cuts) - 1):
        a, b_ = cuts[i], cuts[i + 1]
        if not hs:
            parts.append(build_box(a, b_, y0, y1, z0, z1))
            continue
        cx, cy, d = hs[i]
        bld = _Builder()
        P0 = bld.add(_rect_points(a, b_, y0, y1, [(cx, cy)], segs, z0))
        P1 = bld.add(_rect_points(a, b_, y0, y1, [(cx, cy)], segs, z1))
        C0, C1 = bld.add(_circle(cx, cy, d / 2, z0, segs)), bld.add(_circle(cx, cy, d / 2, z1, segs))
        V = bld.verts()
        mx, my = 0.5 * (a + b_), 0.5 * (y0 + y1)
        bld.face(_stitch(V, P0, C0, (cx, cy), _const((0, 0, -1))))
        bld.face(_stitch(V, P1, C1, (cx, cy), _const((0, 0, 1))))
        bld.face(_band(V, P0, P1, _radial(mx, my, 1)))
        bld.face(_band(V, C0, C1, _radial(cx, cy, -1)))
        parts.append(bld.result())
    return _merge(parts)


def build_tube(W, H, t, z0, z1):
    """Side walls only: rectangular annulus x∈[0,W]×y∈[0,H] minus the cavity, z∈[z0,z1] (4 closed bars)."""
    return _merge([build_box(0, W, 0, t, z0, z1), build_box(0, W, H - t, H, z0, z1),
                   build_box(0, t, t, H - t, z0, z1), build_box(W - t, W, t, H - t, z0, z1)])


def build_cylinder_axis(axis, c, r, h0, h1, n=32):
    """Solid cylinder along x, y or z (axis = 0/1/2), centre line through c (3-vector), from h0 to h1."""
    v, f = build_shaft(2 * r, 0.0, 0.0, h0, h1, n)
    out = np.zeros_like(v)
    if axis == 2:
        out = v + np.array([c[0], c[1], 0.0])
    elif axis == 0:
        out[:, 0], out[:, 1], out[:, 2] = v[:, 2], v[:, 0] + c[1], v[:, 1] + c[2]
    else:
        out[:, 1], out[:, 2], out[:, 0] = v[:, 2], v[:, 0] + c[2], v[:, 1] + c[0]
    if axis != 2:                                   # the axis permutation is a rotation: winding stays valid
        pass
    return out, f


def build_ring(d_in, d_out, cx, cy, z0, z1, n=48):
    """Hollow cylinder (bearing envelope: bore d_in, outside d_out, width z1 - z0)."""
    b = _Builder()
    o0, o1 = b.add(_circle(cx, cy, d_out / 2, z0, n)), b.add(_circle(cx, cy, d_out / 2, z1, n))
    i0, i1 = b.add(_circle(cx, cy, d_in / 2, z0, n)), b.add(_circle(cx, cy, d_in / 2, z1, n))
    V = b.verts()
    b.face(_band(V, o0, o1, _radial(cx, cy, 1)))
    b.face(_band(V, i0, i1, _radial(cx, cy, -1)))
    b.face(_band(V, i0, o0, _const((0, 0, -1))))
    b.face(_band(V, i1, o1, _const((0, 0, 1))))
    return b.result()


def build_stepped_shaft(segments, cx, cy, n=48):
    """Stepped shaft = abutting solid cylinders [(z0, z1, d), ...] in one mesh."""
    vs, fs, off = [], [], 0
    for z0, z1, d in segments:
        if z1 - z0 < 1e-6:
            continue
        v, f = build_shaft(d, cx, cy, z0, z1, n)
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


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
ORING_CORD = 2.62          # mm, ISO 3601 / AS568-1xx cross-section
ORING_GROOVE_W = 3.56      # mm, face-seal groove width (Parker O-Ring Handbook, static face seal)
ORING_GROOVE_D = 1.96      # mm, groove depth (≈ 25 % squeeze)


def _lip_boxes(W, H, t, L, walls, tc, D):
    """Internal sealing lip (land = wall + lip) at the bolted joints."""
    lip = L.get("lip") or {"width_mm": 3.0, "depth_mm": 3.0, "walls": list(walls)}
    w, dz = lip["width_mm"], lip["depth_mm"]
    out = []
    for wall in walls:
        z0, z1 = (tc, tc + dz) if wall == "front" else (D - tc - dz, D - tc)
        out += [build_box(t, W - t, t, t + w, z0, z1), build_box(t, W - t, H - t - w, H - t, z0, z1),
                build_box(t, t + w, t + w, H - t - w, z0, z1), build_box(W - t - w, W - t, t + w, H - t - w, z0, z1)]
    return out


def _oring_frame(W, H, t, L, z_face, wall):
    """O-ring drawn as a square-section rectangular frame on the sealing land, in the cover groove."""
    lip = (L.get("lip") or {}).get("width_mm", 3.0)
    c = (t + lip) / 2.0                    # centre line on the land (from the outer face)
    a = ORING_CORD / 2.0
    z0, z1 = (z_face - ORING_GROOVE_D, z_face) if wall == "front" else (z_face, z_face + ORING_GROOVE_D)
    return _merge([build_box(c - a, W - c + a, c - a, c + a, z0, z1),
                   build_box(c - a, W - c + a, H - c - a, H - c + a, z0, z1),
                   build_box(c - a, c + a, c + a, H - c - a, z0, z1),
                   build_box(W - c - a, W - c + a, c + a, H - c - a, z0, z1)])


def build_assembly(design: dict, kinematics: dict, shaft: dict, housing_params: dict,
                   materials: dict | None = None) -> dict:
    """
    Manufacturable gearbox (layout v4) for one variant of physics.variants.generate_variants():
    housing body + bolted cover(s) (+ central bearing web for 2 stages), bearing bosses, every shaft in two
    deep-groove bearings, lip seals where shafts leave, cover bolts in corner columns, oil plugs.
    Returns {"parts": [...], "warnings": [...], "notes": [...], "bbox": {...}, "layout": {...}}.
    """
    m = float(design["module_mm"])
    zp, zg = int(design["pinion_teeth"]), int(design["gear_teeth"])
    b = float(design["face_width_mm"])
    stages = int(design["stages"])
    if not (0 < m <= 12 and 8 <= zp <= 200 and 8 <= zg <= 400 and 0 < b <= 400 and stages in (1, 2)):
        raise ValueError("unsupported gear parameters for 3D assembly")
    from physics.variant_shafts import shaft_plan
    from physics.goal_engine import assembly_notes
    variant = {"design": design, "kinematics": kinematics, "shaft": shaft, "housing_params": housing_params,
               "loads": {}, "materials": materials or {}}
    plan = shaft_plan(variant)
    if plan is None:
        raise ValueError("no shaft + bearing design exists for this variant (fatigue / L10 life); "
                         "it cannot be built as a manufacturable gearbox")
    L = plan["L"]
    W, H, D, t, yc = L["W"], L["H"], L["D"], L["wall_mm"], L["yc"]
    tc = L.get("cover_mm", t)                      # end walls / bolted covers (≥ side wall)
    x_a, x_b, z1, z2 = L["x_a"], L["x_b"], L["z1"], L["z2"]
    a = L["center_distance_mm"]
    web = L.get("web")
    warnings = list(plan["notes"])
    notes = []
    parts = []

    def add_part(name, kind, verts, faces, **meta):
        parts.append({"name": name, "kind": kind, "verts": verts, "faces": faces, **meta})

    names = [sh["name"] for sh in L["shafts"]]
    axes = [sh["x"] for sh in L["shafts"]]
    if stages == 1:
        mounts = {0: [("Pinion", zp, z1, 0.0, True)], 1: [("Gear", zg, z1, math.pi - math.pi / zg, False)]}
    else:
        mounts = {0: [("Pinion (stage 1)", zp, z1, 0.0, True)],
                  1: [("Gear (stage 1)", zg, z1, math.pi - math.pi / zg, False),
                      ("Pinion (stage 2)", zp, z2, math.pi, True)],
                  2: [("Gear (stage 2)", zg, z2, -math.pi / zg, False)]}

    # ---- shafts: journal at the bearings, seat under the gears (ACE shaft solver), integral pinions
    gear_bores, profiles = {}, []
    for si, nm in enumerate(names):
        ps = plan["shafts"][si]
        za_, zb_ = L["shafts"][si]["z0"], L["shafts"][si]["z1"]
        journal, seat = float(ps["journal_mm"]), float(ps["seat_mm"])
        zones = []
        for gname, zt, gz0, _rot, is_pin in mounts[si]:
            d_zone = seat
            root = m * (zt - 2.5)
            if is_pin and root - d_zone < 4.0:
                d_zone = math.floor(root - 4.0)
                notes.append(f"{gname}: cut directly on the shaft (integral pinion shaft, Ø{d_zone:.0f} mm core): "
                             f"its root Ø{root:.1f} mm leaves no room for a keyed Ø{seat:.0f} mm seat.")
            zones.append((gz0, gz0 + b, d_zone))
            gear_bores[gname] = d_zone
        zones.sort()
        segs, z = [], za_
        for k, (g0, g1, dz) in enumerate(zones):
            if g0 > z:
                prev = zones[k - 1][2] if k else None
                segs.append((z, g0, min(prev, dz) if prev else min(journal, dz)))
            segs.append((max(g0, z), g1, dz))
            z = g1
        if zb_ > z:
            segs.append((z, zb_, min(journal, zones[-1][2])))
        profiles.append(segs)

    # ---- housing walls: bearing holes (through-bores where a shaft leaves, capped pockets elsewhere)
    brg_at = {"front": [], "back": [], "web": []}
    for si, ps in enumerate(plan["shafts"]):
        for br in ps["bearings"]:
            brg_at[br["seat"]].append((si, br))
    exit_shafts = {0: "front", len(names) - 1: "back"}
    holes = {w: [(axes[si], yc, float(br["D"])) for si, br in brg_at[w]] for w in ("front", "back")}
    if stages == 1:
        body = [build_tube(W, H, t, 0.0, D - tc), build_plate_with_holes(t, W - t, t, H - t, 0.0, tc, holes["front"])]
        body += _lip_boxes(W, H, t, L, ("back",), tc, D)
        add_part("Housing", "housing", *_merge(body), role="body: side walls + integral front wall, open at the back; "
                 "internal sealing lip at the cover joint")
        cv, cf = build_plate_with_holes(0, W, 0, H, D - tc, D, holes["back"])
        add_part("Housing cover", "housing", cv, cf, role=f"bolted back cover, {tc:g} mm, O-ring groove in its face",
                 thickness_mm=tc)
        covers = [("back", D - tc, D)]
        col_spans = {"back": [(tc, D - tc)]}
    else:
        body = [build_tube(W, H, t, tc, D - tc)]
        body += _lip_boxes(W, H, t, L, ("front", "back"), tc, D)
        b_in = [br for si, br in brg_at["web"] if si == 0][0]
        b_out = [br for si, br in brg_at["web"] if si == 2][0]
        hole_b = (x_b, yc, float(L["web_hole_mm"]))
        zw0, zw1 = web["z0"], web["z1"]
        # ribbed web: plate across the section at the septum + a bearing hub around axis A
        zp0, zp1 = web.get("plate", (zw0, zw1))
        hub_r = web.get("hub_d", float(max(b_in["D"], b_out["D"])) + 12.0) / 2.0
        p_in, p_out = zw0 + float(b_in["B"]), zw1 - float(b_out["B"])      # pocket bottoms (septum faces)
        for z0_, z1_ in ((zp0, min(zp1, p_in)), (max(zp0, p_in), min(zp1, p_out)), (max(zp0, p_out), zp1)):
            if z1_ - z0_ < 1e-6:
                continue
            zm = 0.5 * (z0_ + z1_)
            hl = [hole_b]
            if zm < p_in:
                hl.append((x_a, yc, float(b_in["D"])))
            elif zm > p_out:
                hl.append((x_a, yc, float(b_out["D"])))
            body.append(build_plate_with_holes(t, W - t, t, H - t, z0_, z1_, hl))
        if zp0 > zw0 + 1e-6:
            body.append(build_ring(float(b_in["D"]), 2 * hub_r, x_a, yc, zw0, min(zp0, p_in)))
            if zp0 > p_in + 1e-6:                     # septum part of the hub below the plate
                body.append(build_cylinder_axis(2, (x_a, yc, 0), hub_r, p_in, zp0, 48))
        if zp1 < zw1 - 1e-6:
            if zp1 < p_out - 1e-6:
                body.append(build_cylinder_axis(2, (x_a, yc, 0), hub_r, zp1, p_out, 48))
            body.append(build_ring(float(b_out["D"]), 2 * hub_r, x_a, yc, max(zp1, p_out), zw1))
        add_part("Housing", "housing", *_merge(body),
                 role="body: side walls + integral central web carrying the inner bearings of the input and output shafts")
        fv, ff = build_plate_with_holes(0, W, 0, H, 0.0, tc, holes["front"])
        add_part("Housing front cover", "housing", fv, ff, role=f"bolted front cover, {tc:g} mm, O-ring groove in its face",
                 thickness_mm=tc)
        cv, cf = build_plate_with_holes(0, W, 0, H, D - tc, D, holes["back"])
        add_part("Housing cover", "housing", cv, cf, role=f"bolted back cover, {tc:g} mm, O-ring groove in its face",
                 thickness_mm=tc)
        covers = [("front", 0.0, tc), ("back", D - tc, D)]
        col_spans = {"front": [(tc, zp0)], "back": [(zp1, D - tc)]}

    # ---- cover joints (Tahap 1.1): screws, dowels and columns sized by the bolted-joint solver
    joints = plan.get("joints") or {}
    cols, bolts, orings, jspec = [], [], [], []
    for wall, c0, c1 in covers:
        j = joints.get(wall)
        if not j or not j.get("pass"):
            warnings.append(f"The {wall} cover joint could not be sized (see layout.joints).")
            continue
        d_s = METRIC[j["size"]][0]
        dk = METRIC[j["size"]][2]
        for x0_, y0_, w_, h_ in j.get("column_rects") or [[a, b, c, c] for a, b, c in j["columns"]]:
            for z0_, z1_ in col_spans[wall]:
                cols.append(build_box(x0_, x0_ + w_, y0_, y0_ + h_, z0_, z1_))
        for cxc, cyc in j["positions"]:
            if wall == "back":
                bolts.append(build_cylinder_axis(2, (cxc, cyc, 0), dk / 2, D, D + d_s, 24))
            else:
                bolts.append(build_cylinder_axis(2, (cxc, cyc, 0), dk / 2, -d_s, 0.0, 24))
        z_face = c1 if wall == "front" else c0          # cover face that touches the body
        orings.append(_oring_frame(W, H, t, L, z_face, wall))
        jspec.append(f"{wall}: {j['spec']}; bonded sealing washers DIN 7603 under the screw heads")
    if cols:
        add_part("Housing corner columns", "housing", *_merge(cols),
                 role="columns with tapped holes for the cover screws (size from the bolted-joint check)")
    if bolts:
        add_part("Cover bolts", "fastener", *_merge(bolts), spec="; ".join(jspec))
    if orings:
        add_part("Cover O-rings", "seal", *_merge(orings),
                 spec=f"O-ring {ORING_CORD:g} mm cord (NBR 70 / FKM), face-seal groove {ORING_GROOVE_W:g} × "
                      f"{ORING_GROOVE_D:g} mm in each cover, groove corners R ≥ {3 * ORING_CORD:.0f} mm "
                      "(groove not cut in the STL)")

    # ---- bearing bosses, bearings, seals
    bosses, seals = [], []
    for wall in ("front", "back"):
        for si, br in brg_at[wall]:
            B, Dbr = float(br["B"]), float(br["D"])
            wb = max(4.0, 0.12 * Dbr)
            out_len = max(0.0, B - tc)
            exits = exit_shafts.get(si) == wall
            extra = 7.0 if exits else 3.0                 # seal width, or the closing cap
            if wall == "front":
                z_lo, z_hi = -(out_len + extra), 0.0
            else:
                z_lo, z_hi = D, D + out_len + extra
            bosses.append(build_ring(Dbr, Dbr + 2 * wb, axes[si], yc, z_lo, z_hi))
            if not exits:                                  # blind pocket: closing cap
                cap = (z_lo, z_lo + 3.0) if wall == "front" else (z_hi - 3.0, z_hi)
                bosses.append(build_cylinder_axis(2, (axes[si], yc, 0), Dbr / 2, cap[0], cap[1], 48))
            else:                                          # lip seal outside the bearing
                outside = [sg[2] for sg in profiles[si] if (sg[0] < 0 if wall == "front" else sg[1] > D)]
                d_sh = min(outside) if outside else float(br["d"])
                sz = (z_lo, z_lo + 7.0) if wall == "front" else (z_hi - 7.0, z_hi)
                seals.append(build_ring(d_sh, Dbr, axes[si], yc, sz[0], sz[1]))
    add_part("Bearing bosses", "housing", *_merge(bosses),
             role="bearing seats: outer ring flush with the inner wall face, boss outside (nothing intrudes "
                  "into the gear space)")
    for si, ps in enumerate(plan["shafts"]):
        for br in ps["bearings"]:
            z_b = br["z"]
            v, f = build_ring(br["d"], br["D"], axes[si], yc, z_b - br["B"] / 2, z_b + br["B"] / 2)
            where = {"front": "front wall", "back": "back wall", "web": "central web"}[br["seat"]]
            add_part(f"Bearing {br['designation']} ({names[si].lower()}, {where})", "bearing", v, f,
                     designation=br["designation"], bore_mm=br["d"], outer_mm=br["D"], width_mm=br["B"])
    if seals:
        add_part("Shaft seals", "seal", *_merge(seals), spec="radial lip seals (FKM), 7 mm wide")

    # ---- oil: low breather/fill on top; flush (DIN 908-type) level + magnetic drain plugs in the back cover,
    # placed clear of the bearing bosses (nothing sticks out sideways, so the width envelope is unchanged)
    dag = m * (zg + 2)
    y_level = max(t + 9.0, min(H - t - 9.0, yc - dag / 2 + 2.5 * m + 3.0))
    y_drain = t + 8.0
    boss_r = [(axes[si], float(br["D"]) / 2 + max(4.0, 0.12 * float(br["D"]))) for si, br in brg_at["back"]]

    def free(x, y, r):
        return t + r <= x <= W - t - r and all(math.hypot(x - bx, y - yc) > br_ + r + 2 for bx, br_ in boss_r)
    xs_try = [W / 2 + k * 6.0 for k in range(-25, 26)]
    x_drain = next((x for x in sorted(xs_try, key=lambda x: abs(x - W / 2)) if free(x, y_drain, 8.0)), None)
    x_level = next((x for x in sorted(xs_try, key=lambda x: -abs(x - W / 2))
                    if free(x, y_level, 7.0) and (x_drain is None or abs(x - x_drain) > 18)), None)
    plugs = [build_cylinder_axis(1, (W / 2, 0, D / 2), 8.0, H, H + 6.0)]
    spec = ["breather/fill M16x1.5 on top"]
    if x_drain is not None:
        plugs.append(build_cylinder_axis(2, (x_drain, y_drain, 0), 8.0, D, D + 2.0))
        spec.append("magnetic drain M12x1.5 (flush, back cover, lowest point)")
    if x_level is not None:
        plugs.append(build_cylinder_axis(2, (x_level, y_level, 0), 7.0, D, D + 2.0))
        spec.append("oil level M12x1.5 (flush, back cover)")
    if x_drain is None or x_level is None:
        warnings.append("No room for a drain or level plug on the back cover next to the bearing bosses.")
    add_part("Oil plugs", "plug", *_merge(plugs), spec="; ".join(spec) + "; threads in internal cast bosses")

    # ---- shafts and gears
    for si, nm in enumerate(names):
        v, f = build_stepped_shaft(profiles[si], axes[si], yc)
        add_part(nm, "shaft", v, f, diameter_mm=max(sg[2] for sg in profiles[si]),
                 length_mm=L["shafts"][si]["z1"] - L["shafts"][si]["z0"],
                 segments_mm=[[round(a_, 2), round(b2, 2), d_] for a_, b2, d_ in profiles[si]])

    def gear(name, mm, z, bore, cx, z0, rot):
        v, f, geo, xy = build_gear(mm, z, b, bore, cx, yc, z0, rot)
        add_part(name, "gear", v, f, teeth=z, module_mm=mm, face_width_mm=b, bore_mm=bore,
                 pitch_diameter_mm=2 * geo["rp"], tip_diameter_mm=2 * geo["ra"])
    for si in mounts:
        for gname, zt, gz0, rot, _p in mounts[si]:
            gear(gname, m, zt, gear_bores[gname], axes[si], gz0, rot)
    notes += assembly_notes({"design": design, "housing_params": plan["hp"]})
    d_in, d_out = L["shafts"][0]["d"], L["shafts"][-1]["d"]
    d_mid = L["shafts"][1]["d"] if stages == 2 else None
    front_d = plan["front_bore_mm"]
    back_d = plan["back_bore_mm"]
    allv = np.vstack([p["verts"] for p in parts])
    bbox = {"min": allv.min(axis=0).round(3).tolist(), "max": allv.max(axis=0).round(3).tolist()}
    result = {"parts": parts, "warnings": warnings, "notes": notes, "bbox": bbox, "housing_params": plan["hp"],
            "layout": {"axis_a_mm": [round(x_a, 3), round(yc, 3)], "axis_b_mm": [round(x_b, 3), round(yc, 3)],
                       "center_distance_mm": round(a, 3), "shafts_mm": {"input": d_in, "intermediate": d_mid, "output": d_out},
                       "bores_mm": {"front": front_d, "back": back_d},
                       "shaft_design": "ACE shaft solver (Shigley fatigue + ISO 281)",
                       "web_mm": web["t"] if web else None, "web": web, "stickout_mm": list(L["stickout"]), "manufacturable": plan["manufacturable"],
                       "cover_mm": tc, "joints": joints,
                       "housing_box_mm": [float(W), float(H), float(D)],
                       "overall_box_mm": [round(bbox["max"][i] - bbox["min"][i], 1) for i in range(3)]}}
    from physics.bom import build_bom
    result["bom"] = build_bom(result, plan, materials)
    return result