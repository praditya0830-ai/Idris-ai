"""
Idris Core — POC Geometry Module
=================================

Parametric gearbox HOUSING generator (Step 1 of the Idris Core backend POC).

IMPORTANT — environment note:
This sandbox has no outbound internet access to PyPI/conda (verified: the
agent egress proxy returns a policy-denial / 403 on pypi.org), so
`pythonocc-core` (which needs conda-forge) and `cadquery` could not be
installed. To still deliver a real, working, end-to-end pipeline, this
module builds the solid with a different (but equally valid for a POC)
method:

    1. Rasterize the housing as a 3D boolean voxel grid (outer box minus
       inner cavity, minus a cylindrical bore) using pure NumPy.
    2. Extract a triangulated surface mesh from that voxel grid with the
       Marching Cubes algorithm (scikit-image, already available here).
    3. Export the mesh to a binary STL file (hand-written exporter, no
       extra dependency).

This proves the exact same pipeline the real product needs — natural
parameters in, solid 3D geometry out, a standard file a CAD viewer can
open, plus computed volume/bounding-box data — without needing
OpenCascade. When this moves to a real dev environment with internet
access, swap this module's internals for pythonocc-core/STEP export; the
function signature and API contract below do not need to change.

Scope (matches the original spec): a simple rectangular housing SHELL with
a circular bore for the gear shaft. No gear teeth yet — that is a later
step.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from skimage import measure


@dataclass
class HousingResult:
    verts_mm: np.ndarray       # (N, 3) float, in millimeters
    faces: np.ndarray          # (M, 3) int, vertex indices
    volume_mm3: float
    bbox_mm: dict               # {"width":..., "height":..., "depth":...}
    bore_diameter_mm: float


def _voxel_grid_for_housing(
    envelope_width: float,
    envelope_height: float,
    envelope_depth: float,
    wall_thickness: float,
    bore_diameter: float,
    resolution: int = 120,
) -> tuple[np.ndarray, float]:
    """
    Build a boolean voxel grid representing:
      outer box  MINUS  inner cavity (inset by wall_thickness on all sides)
                 MINUS  a cylindrical bore running through the depth axis

    Returns (grid, voxel_size_mm) where grid[i, j, k] == True means solid
    material at that voxel.
    """
    # voxel_size chosen so the longest envelope dimension maps to `resolution` voxels
    longest = max(envelope_width, envelope_height, envelope_depth)
    voxel_size = longest / resolution

    nx = max(int(round(envelope_width / voxel_size)), 4)
    ny = max(int(round(envelope_height / voxel_size)), 4)
    nz = max(int(round(envelope_depth / voxel_size)), 4)

    # Cell-centered coordinates in mm, origin at the box corner
    xs = (np.arange(nx) + 0.5) * voxel_size
    ys = (np.arange(ny) + 0.5) * voxel_size
    zs = (np.arange(nz) + 0.5) * voxel_size
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Outer solid box
    outer = np.ones_like(X, dtype=bool)

    # Inner cavity: box inset by wall_thickness on every side, open (no cavity
    # floor cut needed since marching cubes just needs a closed manifold —
    # here we cut the full inset box so the housing is a hollow shell)
    t = wall_thickness
    inner_cavity = (
        (X > t) & (X < envelope_width - t) &
        (Y > t) & (Y < envelope_height - t) &
        (Z > t) & (Z < envelope_depth - t)
    )

    # Cylindrical bore for the gear shaft, centered on the housing, running
    # through the full depth (Z axis) — this is where the shaft would pass
    cx, cy = envelope_width / 2.0, envelope_height / 2.0
    r = bore_diameter / 2.0
    bore = ((X - cx) ** 2 + (Y - cy) ** 2) <= (r ** 2)

    solid = outer & (~inner_cavity) & (~bore)

    # Pad with one empty voxel border on all sides so marching_cubes produces
    # a closed surface instead of clipping at the array boundary
    solid = np.pad(solid, pad_width=1, mode="constant", constant_values=False)

    return solid, voxel_size


def _mesh_volume_mm3(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh via the divergence theorem."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    vol = np.sum(np.einsum("ij,ij->i", v0, cross)) / 6.0
    return abs(float(vol))


def generate_housing(
    envelope_width: float,
    envelope_height: float,
    envelope_depth: float,
    gear_ratio: float,
    wall_thickness: float,
    resolution: int = 70,
) -> HousingResult:
    """
    Build a parametric gearbox housing shell with a circular shaft bore.

    Parameters are all in millimeters except gear_ratio (dimensionless).

    `bore_diameter` is derived with a simple placeholder heuristic for this
    POC (NOT an AGMA/engineering calculation yet — that closed-form-formula
    validation layer is a later ACE milestone): a bigger gear_ratio implies
    a smaller output shaft relative to the envelope.
    """
    if wall_thickness * 2 >= min(envelope_width, envelope_height):
        raise ValueError("wall_thickness is too large for the given envelope — cavity would vanish")

    bore_diameter = max(
        min(envelope_width, envelope_height) * 0.28 / (gear_ratio ** 0.5),
        wall_thickness * 2.5,  # never thinner than a structurally sane minimum
    )
    bore_diameter = min(bore_diameter, min(envelope_width, envelope_height) - 4 * wall_thickness)

    grid, voxel_size = _voxel_grid_for_housing(
        envelope_width, envelope_height, envelope_depth, wall_thickness, bore_diameter, resolution
    )

    verts, faces, _normals, _values = measure.marching_cubes(grid.astype(np.float32), level=0.5)

    # Marching cubes returns verts in voxel index space (including the pad
    # offset) — convert back to real millimeters
    verts_mm = (verts - 1.0) * voxel_size  # subtract the 1-voxel pad border

    volume = _mesh_volume_mm3(verts_mm, faces)

    bbox_mm = {
        "width": float(verts_mm[:, 0].max() - verts_mm[:, 0].min()),
        "height": float(verts_mm[:, 1].max() - verts_mm[:, 1].min()),
        "depth": float(verts_mm[:, 2].max() - verts_mm[:, 2].min()),
    }

    return HousingResult(
        verts_mm=verts_mm,
        faces=faces,
        volume_mm3=volume,
        bbox_mm=bbox_mm,
        bore_diameter_mm=float(bore_diameter),
    )


def export_mesh_to_stl(verts: np.ndarray, faces: np.ndarray, filepath: str, name: str = "idris_core_housing") -> None:
    """
    Hand-written binary STL exporter (no numpy-stl/trimesh dependency
    available in this sandbox). Binary STL format:
      80-byte header, 4-byte uint32 triangle count, then per triangle:
      12 bytes normal (3x float32) + 36 bytes vertices (9x float32) + 2 bytes padding
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    normals = (normals.T / norms).T

    with open(filepath, "wb") as f:
        header = name.encode("utf-8")[:80]
        header = header + b" " * (80 - len(header))
        f.write(header)
        f.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *v0[i]))
            f.write(struct.pack("<3f", *v1[i]))
            f.write(struct.pack("<3f", *v2[i]))
            f.write(struct.pack("<H", 0))
