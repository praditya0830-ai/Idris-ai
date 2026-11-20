"""Eigen-buckling of a slender member (P2b): the four Euler column cases, and nothing more than that."""
import math

import pytest

import solvers  # noqa: F401
from ace_core import registry
from ace_core.registry import verify
from solvers.buckling_beam import _euler, _E, _I, _L

BASE = {"length_mm": _L, "diameter_mm": 20.0, "E_MPa": _E, "elements": 40}


def _run(**kw):
    r = registry.run("buckling_beam", {**BASE, **kw})
    assert r["ok"], r.get("errors")
    return r["outputs"]


def test_all_four_euler_benchmarks_pass():
    r = verify("buckling_beam")
    assert r["all_pass"] and r["passed"] == 4


@pytest.mark.parametrize("left,right,k", [("fixed", "free", 2.0), ("pinned", "pinned", 1.0),
                                          ("fixed", "fixed", 0.5), ("fixed", "pinned", 0.6991557)])
def test_each_support_condition_matches_pi2ei_over_kl2(left, right, k):
    o = _run(left=left, right=right)
    assert math.isclose(o["critical_load_N"], _euler(_E, _I, _L, k), rel_tol=0.01)
    assert math.isclose(o["effective_length_factor"], k, rel_tol=0.01)


def test_a_guided_end_is_accepted_and_gives_a_sliding_column():
    o = _run(left="fixed", right="guided")
    # fixed-guided buckles as a fixed-fixed column of half the length: K = 1 for the whole member
    assert math.isclose(o["critical_load_N"], _euler(_E, _I, _L, 1.0), rel_tol=0.02)


def test_a_rectangular_section_buckles_about_its_weak_axis():
    o = _run(section="rect", width_mm=40.0, height_mm=10.0, left="pinned", right="pinned")
    weak = 40.0 * 10.0 ** 3 / 12
    assert math.isclose(o["critical_load_N"], _euler(_E, weak, _L, 1.0), rel_tol=0.01)
    assert len(o["modes"]) == 2                       # both bending planes are reported


def test_the_answer_says_linear_buckling_is_an_upper_bound():
    o = _run()
    text = " ".join(o["not_checked"]).lower()
    assert "plate" in text and "lateral-torsional" in text
    r = registry.run("buckling_beam", BASE)
    assert any("upper bound" in w for w in r["warnings"])


def test_a_slenderness_value_only_annotates_and_does_not_raise_a_failure():
    r = registry.run("buckling_beam", dict(BASE, slenderness=140.0))
    assert r["ok"] and r["verdict"] in ("PASS", "OK")


def test_an_unknown_support_is_refused():
    r = registry.run("buckling_beam", dict(BASE, left="welded"))
    assert r["ok"] is False and r["errors"]
