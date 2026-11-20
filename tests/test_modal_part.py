"""Modal analysis of any part (P2a): Euler-Bernoulli beam, clamped-edge plate, and the honest limits."""
import math

import solvers  # noqa: F401
from ace_core import registry
from ace_core.registry import verify
from solvers.modal_part import BEAM, PLATE, _box_mesh, _L, _E, _I, _RHO, _A


def _run(**kw):
    r = registry.run("modal_part_3d", {**BEAM, **kw})
    assert r["ok"], r.get("errors")
    return r["outputs"]


def test_benchmarks_pass_against_blevins_and_leissa():
    r = verify("modal_part_3d")
    assert r["all_pass"] and r["passed"] == 3


def test_the_cantilever_matches_euler_bernoulli():
    o = _run()
    f1 = 1.8751 ** 2 / (2 * math.pi) * math.sqrt(_E * _I / (_RHO * 1e-12 * _A * _L ** 4))
    assert abs(o["frequencies_hz"][0] - f1) / f1 < 0.05
    assert o["lowest_hz"] == o["frequencies_hz"][0] and len(o["modes"]) == len(o["frequencies_hz"])


def test_each_mode_carries_a_shape_on_the_surface_nodes():
    o = _run(n_modes=3)
    pos = o["surface"]["positions"]
    for m in o["modes"]:
        assert len(m["displacement"]) == len(pos) and m["max_displacement_mm"] > 0


def test_a_frequency_floor_becomes_a_failing_check_when_the_part_is_too_soft():
    below = registry.run("modal_part_3d", dict(BEAM, min_frequency_hz=400.0))
    assert below["ok"] and below["verdict"] == "FAIL"
    above = registry.run("modal_part_3d", dict(BEAM, min_frequency_hz=100.0))
    assert above["verdict"] in ("PASS", "OK")


def test_the_answer_says_what_it_did_not_compute():
    o = _run()
    text = " ".join(o["not_checked"]).lower()
    assert "forced response" in text and "electromagnetic" in text
    r = registry.run("modal_part_3d", BEAM)
    assert any("clamped" in w for w in r["warnings"])


def test_a_mesh_needs_a_support_and_a_closed_volume():
    no_support = registry.run("modal_part_3d", {k: v for k, v in BEAM.items() if k != "supports"})
    assert no_support["ok"] is False
    bad = {"positions": [0, 0, 0, 1, 0, 0, 0, 1, 0], "indices": [0, 1, 2],
           "supports": [{"where": {"plane": "x", "side": "min"}}]}
    assert registry.run("modal_part_3d", bad)["ok"] is False


def test_the_voxel_volume_error_is_reported_next_to_the_frequencies():
    o = _run()
    assert "error_pct" in o["volume"] and abs(o["volume"]["error_pct"]) < 5
    assert o["mass_kg"] > 0 and o["eigen_method"]
