"""
Idris Core — Geometry Module (v2: exact mesh)
==============================================

Parametric gearbox HOUSING generator.

The housing is a hollow rectangular box (wall thickness `t` on every side)
with a circular bore through both end plates, where the shaft passes.

v1 built this from a voxel grid + marching cubes, which quantized every
dimension to the voxel size (e.g. a 150 mm depth came out as 148.6 mm, and
a 3 mm wall was ~5% off in volume). v2 builds the closed triangle mesh
directly from the parameters, so:

  * dimensions are exact (bounding box == requested envelope),
  * the bore is a true (finely segmented) circle, not an octagon,
  * the mesh is watertight and ~50x smaller (about 1.6k triangles),
  * no dependency beyond NumPy.

Topology (8 surfaces, every edge loop shared by exactly two of them):
  outer end faces (z=0, z=D)      ring: outer rectangle -> bore circle
  outer side band                 outer rectangle z=0 -> z=D
  bore walls (two, length t)      bore circle z=0 -> z=t, z=D-t -> z=D
  inner cavity end faces          ring: inner rectangle -> bore circle
  inner cavity side band          inner rectangle z=t -> z=D-t

This still proves the pipeline (parameters -> solid -> standard file ->
computed properties). Real STEP/B-Rep output (OpenCascade) and gear teeth
are later milestones; the function signature stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HousingResult:
    verts_mm: np.ndarray       # (N, 3) float, millimeters
    faces: np.ndarray          # (M, 3) int, vertex indices
    volume_mm3: float
    bbox_mm: dict               # {"width":..., "height":..., "depth":...}
    bore_diameter_mm: float


def _angles(half_w_outer: float, half_h_outer: float,
            half_w_inner: float, half_h_inner: float, segments: int) -> np.ndarray:
    """
    Sorted angles (0..2pi) shared by every loop. Includes the corner angles of
    both rectangles so rectangle corners are exact vertices.
    """
    uniform = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    extra = []
    for hw, hh in ((half_w_outer, half_h_outer), (half_w_inner, half_h_inner)):
        c = np.arctan2(hh, hw)
        extra += [c, np.pi - c, np.pi + c, 2.0 * np.pi - c]
    ang = np.sort(np.concatenate([uniform, np.array(extra)]))
    keep = np.concatenate([[True], np.diff(ang) > 1e-9])
    return ang[keep]


def _rect_loop(cx, cy, hw, hh, z, ang) -> np.ndarray:
    """Points where rays from the center at `ang` hit the rectangle boundary."""
    c, s = np.cos(ang), np.sin(ang)
    with np.errstate(divide="ignore"):
        tx = np.where(np.abs(c) > 1e-12, hw / np.abs(c), np.inf)
        ty = np.where(np.abs(s) > 1e-12, hh / np.abs(s), np.inf)
    d = np.minimum(tx, ty)
    return np.stack([cx + d * c, cy + d * s, np.full_like(ang, z)], axis=1)


def _circle_loop(cx, cy, r, z, ang) -> np.ndarray:
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full_like(ang, z)], axis=1)


def _band(verts: np.ndarray, a: np.ndarray, b: np.ndarray, expected) -> np.ndarray:
    """
    Triangulate the strip between two index loops `a` and `b` (same length,
    closed). Winding is auto-oriented so triangle normals agree with
    `expected(centroid) -> direction` (the outward normal of the solid).
    """
    n = len(a)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((a[i], a[j], b[j]))
        tris.append((a[i], b[j], b[i]))
    tris = np.array(tris, dtype=np.int64)

    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0
    score = np.einsum("ij,ij->i", normals, expected(centroids)).sum()
    if score < 0:
        tris = tris[:, [0, 2, 1]]
    return tris


def _mesh_volume_mm3(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh (divergence theorem)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def generate_housing(
    envelope_width: float,
    envelope_height: float,
    envelope_depth: float,
    gear_ratio: float,
    wall_thickness: float,
    segments: int = 96,
    bore_diameter: float | None = None,
) -> HousingResult:
    """
    Build a parametric gearbox housing shell with a circular shaft bore.

    All parameters are millimeters except gear_ratio (dimensionless).
    The shaft axis is Z (the depth direction); the bore is centered in X/Y.

    `bore_diameter` comes from a simple placeholder heuristic for this POC
    (NOT an AGMA/engineering calculation — that is the CAE layer, a later
    milestone): a larger gear_ratio implies a smaller output shaft relative
    to the envelope.
    """
    W, H, D, t = float(envelope_width), float(envelope_height), float(envelope_depth), float(wall_thickness)

    if min(W, H, D) <= 0 or t <= 0 or gear_ratio <= 0:
        raise ValueError("all dimensions, wall_thickness and gear_ratio must be positive")
    if 2 * t >= min(W, H):
        raise ValueError("wall_thickness is too large for the given envelope — cavity would vanish")
    if 2 * t >= D:
        raise ValueError("wall_thickness is too large for the given depth — cavity would vanish")

    if bore_diameter is not None:
        # explicit bore (e.g. sized from the output shaft by the physics layer)
        bore_d = float(bore_diameter)
        if bore_d <= 0:
            raise ValueError("bore_diameter must be positive")
        if bore_d > min(W, H) - 4 * t:
            raise ValueError("bore_diameter is too large for this envelope and wall_thickness")
    else:
        bore_d = max(min(W, H) * 0.28 / (gear_ratio ** 0.5), t * 2.5)
        bore_d = min(bore_d, min(W, H) - 4 * t)
        if bore_d <= 0:
            raise ValueError("envelope is too small for this wall_thickness — no room for a bore")
    r = bore_d / 2.0

    cx, cy = W / 2.0, H / 2.0
    ang = _angles(W / 2, H / 2, W / 2 - t, H / 2 - t, segments)
    n = len(ang)

    # Eight closed loops (each n points), stacked into one vertex array
    loops = {
        "O0": _rect_loop(cx, cy, W / 2, H / 2, 0.0, ang),         # outer rect, z=0
        "O1": _rect_loop(cx, cy, W / 2, H / 2, D, ang),           # outer rect, z=D
        "C0": _circle_loop(cx, cy, r, 0.0, ang),                  # bore edge, z=0
        "C1": _circle_loop(cx, cy, r, D, ang),                    # bore edge, z=D
        "B0": _circle_loop(cx, cy, r, t, ang),                    # bore edge, z=t
        "B1": _circle_loop(cx, cy, r, D - t, ang),                # bore edge, z=D-t
        "I0": _rect_loop(cx, cy, W / 2 - t, H / 2 - t, t, ang),   # cavity rect, z=t
        "I1": _rect_loop(cx, cy, W / 2 - t, H / 2 - t, D - t, ang),  # cavity rect, z=D-t
    }
    names = list(loops)
    verts = np.vstack([loops[k] for k in names])
    idx = {k: np.arange(i * n, (i + 1) * n) for i, k in enumerate(names)}

    axis_out = lambda c: np.stack([c[:, 0] - cx, c[:, 1] - cy, np.zeros(len(c))], axis=1)   # away from axis
    axis_in = lambda c: -axis_out(c)                                                        # toward axis
    const = lambda v: (lambda c: np.tile(np.array(v, dtype=float), (len(c), 1)))

    faces = np.vstack([
        _band(verts, idx["O0"], idx["C0"], const((0, 0, -1))),   # outer end face z=0
        _band(verts, idx["O1"], idx["C1"], const((0, 0, 1))),    # outer end face z=D
        _band(verts, idx["O0"], idx["O1"], axis_out),            # outer side walls
        _band(verts, idx["C0"], idx["B0"], axis_in),             # bore wall, front plate
        _band(verts, idx["C1"], idx["B1"], axis_in),             # bore wall, back plate
        _band(verts, idx["I0"], idx["B0"], const((0, 0, 1))),    # cavity floor (faces into cavity)
        _band(verts, idx["I1"], idx["B1"], const((0, 0, -1))),   # cavity ceiling
        _band(verts, idx["I0"], idx["I1"], axis_in),             # cavity side walls
    ])

    return HousingResult(
        verts_mm=verts,
        faces=faces,
        volume_mm3=abs(_mesh_volume_mm3(verts, faces)),
        bbox_mm={
            "width": float(verts[:, 0].max() - verts[:, 0].min()),
            "height": float(verts[:, 1].max() - verts[:, 1].min()),
            "depth": float(verts[:, 2].max() - verts[:, 2].min()),
        },
        bore_diameter_mm=float(bore_d),
    )


def export_mesh_to_stl(verts: np.ndarray, faces: np.ndarray, filepath: str, name: str = "idris_core_housing") -> None:
    """
    Binary STL writer (no external dependency). Layout: 80-byte header,
    uint32 triangle count, then per triangle 12 float32 (normal + 3 vertices)
    and a 2-byte attribute field.
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths[:, None]

    rec = np.zeros(len(faces), dtype=[("data", "<f4", (12,)), ("attr", "<u2")])
    rec["data"] = np.hstack([normals, v0, v1, v2]).astype("<f4")

    header = name.encode("utf-8")[:80].ljust(80, b" ")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(np.array([len(faces)], dtype="<u4").tobytes())
        f.write(rec.tobytes())
