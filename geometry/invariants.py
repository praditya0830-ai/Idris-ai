"""Hard geometry checks: the gate before any shape leaves the server (Fase 0 / pilar A).

Two levels, because the system produces two things:
  * mesh_checks()   - the triangle mesh the browser builds and the FEA consumes;
  * solid_checks()  - a kernel summary (one solid, watertight, positive volume, real extents).

Every check returns {"id", "ok", "detail"}. A design that fails here is quarantined with the reason
stated; it is never passed on as "ok". Check ids are part of the contract (docs/ARCH-CONTRACT.md),
so they are stable strings, not human sentences.
"""
from __future__ import annotations

MIN_VOLUME_MM3 = 1e-6
MIN_TRIANGLE_AREA_MM2 = 1e-12


def _check(check_id: str, ok: bool, detail: str = "") -> dict:
    return {"id": check_id, "ok": bool(ok), "detail": detail}


def _area2d(a, b, c) -> float:
    """Twice the triangle area via the cross product (no numpy: this runs anywhere)."""
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return (cx * cx + cy * cy + cz * cz) ** 0.5


def mesh_checks(positions, indices) -> list:
    """Manifold, closed, oriented-enough mesh. positions = [[x,y,z], ...]; indices = flat triples."""
    positions = list(positions or [])
    indices = list(indices or [])
    n, tri = len(positions), len(indices) // 3
    out = [_check("mesh.nonempty", n >= 3 and tri >= 1, f"{n} vertices, {tri} triangles")]
    if not out[0]["ok"]:
        return out

    bad = [i for i in indices if not isinstance(i, int) or not (0 <= i < n)]
    out.append(_check("mesh.indices_in_range", not bad,
                      "" if not bad else f"{len(bad)} indices outside 0..{n - 1}"))
    if bad:
        return out

    degenerate = 0
    for t in range(tri):
        a, b, c = (positions[indices[3 * t + k]] for k in range(3))
        if _area2d(tuple(a), tuple(b), tuple(c)) / 2.0 <= MIN_TRIANGLE_AREA_MM2:
            degenerate += 1
    out.append(_check("mesh.no_degenerate_triangles", degenerate == 0,
                      "" if not degenerate else f"{degenerate} zero-area triangles"))

    edges = {}
    for t in range(tri):
        i, j, k = indices[3 * t:3 * t + 3]
        for u, v in ((i, j), (j, k), (k, i)):
            key = (u, v) if u < v else (v, u)
            edges[key] = edges.get(key, 0) + 1
    boundary = sum(1 for c in edges.values() if c == 1)
    over = sum(1 for c in edges.values() if c > 2)
    out.append(_check("mesh.manifold", over == 0,
                      "" if over == 0 else f"{over} edges shared by more than two triangles"))
    out.append(_check("mesh.closed", boundary == 0,
                      "" if boundary == 0 else f"{boundary} boundary edges (the surface is open)"))
    return out


def solid_checks(summary: dict) -> list:
    """Invariants for a kernel solid, from geometry.fingerprint'able facts."""
    s = summary or {}
    vol = s.get("volume_mm3")
    bbox = s.get("bbox_mm") or []
    faces = s.get("faces") or 0
    out = [
        _check("solid.single", s.get("solids") == 1, f"solids = {s.get('solids')!r}"),
        _check("solid.watertight", s.get("watertight") is True, f"watertight = {s.get('watertight')!r}"),
        _check("solid.volume_positive", isinstance(vol, (int, float)) and vol > MIN_VOLUME_MM3,
               f"volume = {vol!r} mm3"),
        _check("solid.bbox_positive",
               len(bbox) in (3, 6) and all(isinstance(c, (int, float)) and c > 0 for c in bbox[:3]),
               f"bbox = {bbox!r}"),
        _check("solid.has_faces", int(faces) > 0, f"faces = {faces!r}"),
        _check("solid.no_degenerate_faces", int(s.get("degenerate_faces") or 0) == 0,
               f"degenerate faces = {s.get('degenerate_faces')!r}"),
    ]
    return out


def all_ok(checks) -> bool:
    return all(c["ok"] for c in checks or [])


def failures(checks) -> list:
    return [c for c in checks or [] if not c["ok"]]
