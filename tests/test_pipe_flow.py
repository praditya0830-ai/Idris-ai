"""Pipe flow (P2e): Poiseuille in the laminar regime, Colebrook in the turbulent one, and the band between."""
import math

import solvers  # noqa: F401
from ace_core import registry
from ace_core.registry import verify
from solvers.pipe_flow import colebrook

WATER = {"fluid": "water_20c", "diameter_mm": 50.0, "length_mm": 1000.0}


def _run(**kw):
    r = registry.run("pipe_flow_darcy", {**WATER, **kw})
    assert r["ok"], r.get("errors")
    return r["outputs"]


def test_benchmarks_pass_against_poiseuille_and_the_moody_chart():
    r = verify("pipe_flow_darcy")
    assert r["all_pass"] and r["passed"] == 3


def test_laminar_pressure_drop_is_the_exact_hagen_poiseuille_value():
    o = _run(diameter_mm=10.0, velocity_m_s=0.1)
    mu, rho, d, v, L = 1.002e-3, 998.2, 0.01, 0.1, 1.0
    exact = 128 * mu * L * (v * math.pi * d ** 2 / 4) / (math.pi * d ** 4)
    assert o["regime"] == "laminar" and math.isclose(o["pressure_drop_Pa"], exact, rel_tol=1e-3)
    # the solver envelope rounds a number above 1e-2 to four decimals, so a factor of 0.06424 comes back 0.0642
    assert math.isclose(o["friction_factor"], 64.0 / o["reynolds"], rel_tol=2e-3)


def test_colebrook_matches_the_published_factors_and_converges():
    for rel, expected in ((1e5, 0.0223), (1e6, 0.0134)):
        f, iters = colebrook(rel, 0.001 if rel == 1e5 else 0.0001)
        assert abs(f - expected) / expected < 0.02 and iters < 60


def test_the_transitional_band_is_flagged_not_used():
    o = _run(velocity_m_s=0.07)          # Re ~ 3500 for water at 20 degC in a 50 mm pipe
    assert 2300 <= o["reynolds"] < 4000 and o["regime"] == "transitional"
    assert any("transitional band" in w for w in registry.run("pipe_flow_darcy",
                                                             dict(WATER, velocity_m_s=0.07))["warnings"])


def test_roughness_is_ignored_in_the_laminar_regime_and_said_so():
    rough = registry.run("pipe_flow_darcy", dict(WATER, diameter_mm=10.0, velocity_m_s=0.1, roughness_mm=0.05))
    smooth = registry.run("pipe_flow_darcy", dict(WATER, diameter_mm=10.0, velocity_m_s=0.1))
    assert rough["outputs"]["pressure_drop_Pa"] == smooth["outputs"]["pressure_drop_Pa"]
    assert any("laminar regime" in w for w in rough["warnings"])


def test_minor_losses_add_to_the_friction_loss():
    base = _run(velocity_m_s=2.0, roughness_mm=0.05)
    withk = _run(velocity_m_s=2.0, roughness_mm=0.05, minor_loss_K=2.5)
    dyn = 0.5 * 998.2 * 2.0 ** 2
    assert math.isclose(withk["minor_drop_Pa"], 2.5 * dyn, rel_tol=1e-6)
    assert math.isclose(withk["pressure_drop_Pa"], base["pressure_drop_Pa"] + 2.5 * dyn, rel_tol=1e-6)


def test_flow_rate_and_velocity_agree_and_one_of_them_is_required():
    o = _run(velocity_m_s=2.0)
    assert math.isclose(o["flow_m3_h"], 2.0 * math.pi * 0.05 ** 2 / 4 * 3600, rel_tol=1e-4)
    assert registry.run("pipe_flow_darcy", WATER)["ok"] is False
    assert registry.run("pipe_flow_darcy", dict(WATER, velocity_m_s=1.0))["ok"] is True


def test_the_solver_says_what_it_is_not():
    o = _run(velocity_m_s=2.0)
    text = " ".join(o["not_checked"]).lower()
    assert "3d flow field" in text and "compressibility" in text
