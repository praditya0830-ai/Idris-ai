"""Fase 0: the fingerprint is deterministic and the invariant gate actually rejects bad geometry."""
import pytest

from geometry import fingerprint as F
from geometry import invariants as INV
from geometry import kernel

CUBE_SUMMARY = {"volume_mm3": 4800.0, "area_mm2": 1760.0, "bbox_mm": [40.0, 20.0, 3.0],
                "solids": 1, "faces": 6, "edges": 12, "vertices": 8, "mass_kg": 0.005904}

GOLDEN = {
    "bracket": {"volume_mm3": 4800.0, "area_mm2": 1760.0, "bbox_mm": [40, 20, 3],
                "solids": 1, "faces": 6, "edges": 12, "vertices": 8},
    "flange": {"volume_mm3": 12880.5, "area_mm2": 6031.2, "bbox_mm": [70, 70, 12],
               "solids": 1, "faces": 14, "edges": 28, "vertices": 22},
    "spacer": {"volume_mm3": 2261.9, "area_mm2": 1884.9, "bbox_mm": [12, 12, 40],
               "solids": 1, "faces": 4, "edges": 6, "vertices": 8},
    "housing": {"volume_mm3": 210455.0, "area_mm2": 61234.0, "bbox_mm": [152, 108, 96],
                "solids": 1, "faces": 61, "edges": 138, "vertices": 70},
    "unknown_part": {"volume_mm3": 999.0, "area_mm2": 640.0, "bbox_mm": [10, 10, 10],
                     "solids": 1, "faces": 6, "edges": 12, "vertices": 8},
}


def cube_mesh():
    """Closed unit cube: 8 vertices, 12 triangles, every edge shared by exactly two triangles."""
    p = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return p, idx


def test_fingerprint_is_deterministic_and_order_independent():
    for name, s in GOLDEN.items():
        assert F.fingerprint(s) == F.fingerprint(dict(s)), name
        shuffled = dict(reversed(list(s.items())))
        assert F.fingerprint(shuffled) == F.fingerprint(s), name


def test_all_golden_parts_hash_differently():
    hashes = {name: F.fingerprint(s) for name, s in GOLDEN.items()}
    assert len(set(hashes.values())) == len(hashes)


def test_unrelated_fields_do_not_change_the_hash():
    a = F.fingerprint(CUBE_SUMMARY)
    b = F.fingerprint({**CUBE_SUMMARY, "kernel": "occt 8.0", "notes": "built twice"})
    assert a == b


def test_a_real_change_changes_the_hash():
    assert F.fingerprint(CUBE_SUMMARY) != F.fingerprint({**CUBE_SUMMARY, "volume_mm3": 4800.6})


def test_diff_ignores_noise_and_reports_real_moves():
    near = {**CUBE_SUMMARY, "volume_mm3": 4800.0 * 1.0004}          # 0.04 %: inside tolerance
    assert F.diff(CUBE_SUMMARY, near) == []
    moved = F.diff(CUBE_SUMMARY, {**CUBE_SUMMARY, "volume_mm3": 5000.0})
    assert [d["field"] for d in moved] == ["volume_mm3"]
    counts = F.diff(CUBE_SUMMARY, {**CUBE_SUMMARY, "faces": 7})
    assert counts[0]["field"] == "faces"                            # counts are exact, never tolerated


def test_closed_cube_mesh_passes_every_check():
    p, idx = cube_mesh()
    checks = INV.mesh_checks(p, idx)
    assert INV.all_ok(checks), INV.failures(checks)


def test_open_mesh_is_rejected_as_not_closed():
    p, idx = cube_mesh()
    checks = INV.mesh_checks(p, idx[:-3])                           # drop one triangle
    assert not INV.all_ok(checks)
    failed = {c["id"] for c in INV.failures(checks)}
    # a hole leaves boundary edges but every remaining edge is still shared by at most two faces
    assert failed == {"mesh.closed"}, failed


def test_non_manifold_mesh_is_rejected():
    p, idx = cube_mesh()
    checks = INV.mesh_checks(p, idx + idx[:3])                     # same triangle twice
    failed = {c["id"] for c in INV.failures(checks)}
    assert "mesh.manifold" in failed, failed


def test_degenerate_triangle_and_bad_index_are_rejected():
    p, idx = cube_mesh()
    p2 = p + [[0, 0, 0]]
    checks = INV.mesh_checks(p2, idx + [0, 0, 8])                  # vertices 0 and 8 coincide
    failed = {c["id"] for c in INV.failures(checks)}
    assert "mesh.no_degenerate_triangles" in failed, failed
    bad_index = INV.mesh_checks(p, idx + [0, 1, 99])
    assert "mesh.indices_in_range" in {c["id"] for c in INV.failures(bad_index)}


def test_empty_mesh_fails_loudly():
    checks = INV.mesh_checks([], [])
    assert checks[0]["id"] == "mesh.nonempty" and not checks[0]["ok"]


def test_solid_checks_accept_a_good_summary():
    s = {**CUBE_SUMMARY, "watertight": True, "degenerate_faces": 0}
    assert INV.all_ok(INV.solid_checks(s))


@pytest.mark.parametrize("broken,check_id", [
    ({"solids": 2}, "solid.single"),
    ({"watertight": False}, "solid.watertight"),
    ({"volume_mm3": 0.0}, "solid.volume_positive"),
    ({"bbox_mm": [40, 0, 3]}, "solid.bbox_positive"),
    ({"faces": 0}, "solid.has_faces"),
    ({"degenerate_faces": 2}, "solid.no_degenerate_faces"),
])
def test_solid_checks_name_the_broken_invariant(broken, check_id):
    s = {**CUBE_SUMMARY, "watertight": True, "degenerate_faces": 0, **broken}
    ids = {c["id"] for c in INV.failures(INV.solid_checks(s))}
    assert check_id in ids


def test_kernel_is_honest_when_it_is_missing():
    if kernel.available():
        pytest.skip("OCP is installed here; the unavailable path cannot be exercised")
    assert kernel.available() is False
    for call in (lambda: kernel.box(1, 1, 1), lambda: kernel.measure(object()),
                 lambda: kernel.read_step("nope.step")):
        with pytest.raises(kernel.KernelUnavailable):
            call()
