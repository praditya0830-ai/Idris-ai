"""Thermal, lumped / 1D (Batch 3): the closed-form flat wall, checked against the same formula by hand.

The benchmark is the analytic result itself (q = dT / R with R = 1/(hA) + L/(kA) + 1/(hA)), so a wrong
resistance network cannot pass; the conduction-only case pins the wall term, and every reply must say that this
is 1D and not a 3D thermal FEA.
"""
import math

import solvers  # noqa: F401
from ace_core import registry
from ace_core.registry import verify

WALL = {"thickness_mm": 10.0, "surface_area_m2": 1.0, "k_W_mK": 50.0, "T_hot_C": 120.0, "T_cold_C": 20.0,
        "h_hot_W_m2K": 10.0, "h_cold_W_m2K": 10.0}


def _run(**kw):
    r = registry.run("thermal_lumped", {**WALL, **kw})
    assert r["ok"], r.get("errors")
    return r["outputs"]


def test_the_benchmarks_pass_against_the_closed_form():
    r = verify("thermal_lumped")
    assert r["all_pass"], r
    assert r["passed"] == 2


def test_a_flat_wall_matches_the_analytic_answer_to_a_tenth_of_a_percent():
    o = _run()
    length, area, k, h = 0.010, 1.0, 50.0, 10.0
    r = 1 / (h * area) + length / (k * area) + 1 / (h * area)
    q = 100.0 / r
    assert math.isclose(o["heat_flow_W"], q, rel_tol=1e-3)
    assert math.isclose(o["temperature_C"]["hot_surface"], 120.0 - q / (h * area), abs_tol=0.01)
    assert math.isclose(o["temperature_C"]["cold_surface"], 120.0 - q * (1 / (h * area) + length / (k * area)),
                        abs_tol=0.01)
    assert math.isclose(o["heat_flux_W_m2"], q / area, rel_tol=1e-3)


def test_the_wall_is_almost_the_only_resistance_when_convection_is_very_large():
    o = _run(h_hot_W_m2K=1e6, h_cold_W_m2K=1e6)
    r = 1e-6 + 0.010 / 50.0 + 1e-6
    assert math.isclose(o["heat_flow_W"], 100.0 / r, rel_tol=1e-3)
    assert abs(o["temperature_C"]["cold_surface"] - 20.0) < 0.6


# The four ranges printed in Incropera et al., 6th ed., Table 1.1 (typical h, W/m2K). Anything else in H_TABLE
# is a number with Incropera's name on it that Incropera does not print — the audit's WAJIB 4.
TABLE_11 = {
    "air_natural": (2.0, 25.0),        # free convection, gases
    "liquid_natural": (50.0, 1000.0),  # free convection, liquids
    "air_forced": (25.0, 250.0),       # forced convection, gases
    "liquid_forced": (100.0, 20000.0),  # forced convection, liquids
}


def test_every_h_table_row_is_a_range_printed_in_table_1_1():
    from solvers.thermal_lumped import H_TABLE
    assert set(H_TABLE) == set(TABLE_11)
    for name, (lo, hi) in TABLE_11.items():
        got_lo, got_hi, src = H_TABLE[name]
        assert (got_lo, got_hi) == (lo, hi), name
        assert "Incropera" in src and "Table 1.1" in src
    assert "oil_forced" not in H_TABLE, "Table 1.1 has no oil row, so Incropera's name cannot be on one"
    assert "water_forced" not in H_TABLE, "the table says liquids (100-20,000), not water 100-15,000"


def test_h_from_a_table_is_named_with_its_source_and_its_limit():
    o = _run(h_hot_W_m2K=None, h_cold_W_m2K=None, correlation_hot="air_natural",
             correlation_cold="air_natural")
    assert o["h_hot_source"].startswith("table: air_natural") and "Incropera" in o["h_hot_source"]
    assert "geometric mean" in o["h_hot_source"] and "2-25" in o["h_hot_source"]
    assert math.isclose(o["h_hot_W_m2K"], math.sqrt(2.0 * 25.0), abs_tol=0.01)
    r = registry.run("thermal_lumped", {**WALL, "h_hot_W_m2K": None, "h_cold_W_m2K": None,
                                       "correlation_hot": "air_natural", "correlation_cold": "air_natural"})
    assert any("geometric mean" in w for w in r["warnings"])


def test_one_h_from_the_engineer_and_one_from_the_table_still_warns():
    """The old condition (h_hot or h_cold given) silenced the warning whenever only one side was a table."""
    r = registry.run("thermal_lumped", {**WALL, "h_hot_W_m2K": 42.0, "h_cold_W_m2K": None,
                                       "correlation_cold": "liquid_forced"})
    assert r["ok"], r.get("errors")
    o = r["outputs"]
    assert o["h_hot_W_m2K"] == 42.0 and o["h_hot_source"] == "you"
    assert math.isclose(o["h_cold_W_m2K"], math.sqrt(100.0 * 20000.0), abs_tol=0.01)
    assert any("geometric mean" in w and "h_cold" in w for w in r["warnings"])
    assert any("This is a 1D lumped model" in w for w in r["warnings"])


def test_two_h_values_from_the_engineer_need_no_table_warning():
    r = registry.run("thermal_lumped", WALL)
    assert r["ok"]
    assert not any("published table" in w for w in r["warnings"])


def test_the_reply_says_it_is_one_dimensional_and_not_fea():
    o = _run()
    assert "1D" in o["fidelity"] and "not 3D FEA" in o["fidelity"]
    assert any("3D temperature field" in n for n in o["not_checked"])
    r = registry.run("thermal_lumped", WALL)
    assert "thermal" in r["solver"] and r["fidelity"] == "L0"


def test_a_wall_with_no_temperature_difference_or_no_h_is_refused_with_a_reason():
    r = registry.run("thermal_lumped", {**WALL, "T_hot_C": 20.0})
    assert not r["ok"] and any("no heat flow" in e for e in r["errors"])
    r = registry.run("thermal_lumped", {**WALL, "thickness_mm": -1})
    assert not r["ok"]
    r = registry.run("thermal_lumped", {k: v for k, v in WALL.items() if not k.startswith("h_")})
    assert not r["ok"] and any("correlation" in e or "h_hot" in e for e in r["errors"])
