"""Tahap 2.1: load-dependent gearbox losses (ISO/TR 14179-2 style) and their use as heat source."""
import math
from ace_core.analysis import handle_post
from physics.power_loss import gear_loss_factor, walther, oil_props, bearing_loss, seal_loss, mesh_loss
from solvers._example import EXAMPLE_VARIANT as V

BASE = {**V, "power_kw": 11.19, "input_rpm": 6000}


def test_ohlendorf_fzg_c_benchmark():
    r = gear_loss_factor(4.5, 16, 24, 0.1817, 0.1715, 82.46, 118.36, 91.5)
    assert abs(r["H_V"] / 0.1959 - 1) < 0.01 and r["valid"]


def test_walther_mineral_oil_benchmark():
    assert abs(walther(94.1, 10.6)(60.0) / 38.4 - 1) < 0.03
    assert abs(walther(94.1, 10.6)(40.0) - 94.1) < 1e-6 and abs(walther(94.1, 10.6)(100.0) - 10.6) < 1e-6


def test_schlenk_trends():
    oil = oil_props(68, 80)
    lo = mesh_loss(2.0, 17, 51, 32, 10.0, 6000, oil)["mu_mz"]
    hi = mesh_loss(2.0, 17, 51, 32, 40.0, 6000, oil)["mu_mz"]
    slow = mesh_loss(2.0, 17, 51, 32, 10.0, 1500, oil)["mu_mz"]
    assert lo < hi and lo < slow                      # more load / lower speed → more friction
    assert 0.02 < lo < 0.1                            # plausible range for mineral oil


def test_bearing_and_seal_formulas():
    # M = μ F d / 2, P = M ω; seal P = 7.69e-6 d² n
    assert abs(bearing_loss(1000, 20, 3000) - 0.0015 * 1000 * 20 / 2 * 2 * math.pi * 3000 / 60 / 1000) < 1e-12
    assert abs(seal_loss(20, 3000) - 7.69e-6 * 400 * 3000) < 1e-12


def test_cfd_loss_model_heat_balance():
    c, a = handle_post("/cfd?sync=1", {**BASE, "loss_model": "constant"})
    c2, b = handle_post("/cfd?sync=1", {**BASE, "loss_model": "iso14179_load"})
    assert c == c2 == 200
    ra, rb = a["result"], b["result"]
    assert ra["power_loss"]["used_as_heat_source"] is False and rb["power_loss"]["used_as_heat_source"] is True
    # the computed loss is evaluated at the oil temperature it produces (self-consistent within 1 %)
    assert abs(rb["heat_generated_w"] / rb["power_loss"]["total_load_dependent_w"] - 1) < 0.01
    assert rb["oil_temp_c"] < ra["oil_temp_c"]        # lower bound vs conservative constant efficiency
    assert any("LOWER BOUND" in w for w in rb["warnings"])


def test_generator_reports_loss_estimate_next_to_constant():
    from physics.variants import generate_variants
    from tests.test_manufacturable import S1
    v = generate_variants(S1, compare_basis=False)["variants"][0]
    ts = v["thermal_screen"]
    e = ts["load_loss_estimate"]
    assert "lower bound" in e["basis"] and 0 < e["load_losses_w"] < ts["heat_w"]
    assert e["oil_temp_natural_c"] < ts["oil_temp_natural_c"]
