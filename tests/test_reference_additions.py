"""Reference additions of 18 Oct 2026 (from Arya's master specification, correct parts only)."""
import math
from ace_core import registry
import solvers  # noqa: F401
from physics.iso2768 import deviation
from physics.iso286 import IT, RANGES


def test_nafems_le1():
    out = registry.run("nafems_le1", {"mesh": "fine"})["outputs"]
    assert abs(out["sigma_yy_D_MPa"] / 92.7 - 1) < 0.02


def test_pressure_vessel_thin_vs_lame_and_margin():
    o = registry.run("pressure_vessel", {"pressure_mpa": 1.0, "mean_diameter_mm": 100.0, "wall_mm": 2.0})["outputs"]
    assert abs(o["hoop_thin_MPa"] - 25.0) < 1e-9 and abs(o["longitudinal_thin_MPa"] - 12.5) < 1e-9
    assert abs(o["hoop_lame_inner_MPa"] / o["hoop_thin_MPa"] - 1) < 0.002
    thick = registry.run("pressure_vessel", {"pressure_mpa": 10.0, "mean_diameter_mm": 50.0, "wall_mm": 10.0,
                                             "yield_mpa": 250.0, "factor_of_safety": 1.5})
    assert not thick["outputs"]["thin_wall_valid"] and thick["outputs"]["hoop_lame_inner_MPa"] > thick["outputs"]["hoop_thin_MPa"]
    assert abs(thick["outputs"]["margin_of_safety"] - (250 / (thick["outputs"]["von_mises_inner_MPa"] * 1.5) - 1)) < 1e-3


def test_iso2768_table_not_formula():
    assert deviation(25) == 0.2 and deviation(100) == 0.3 and deviation(300) == 0.5 and deviation(5, "f") == 0.05
    assert deviation(1500, "c") == 3.0 and deviation(2, "v") is None


def test_iso286_table_matches_tolerance_factor():
    # i = 0.45·∛D + 0.001·D (µm), D = geometric mean of the size step; IT6 = 10 i, IT7 = 16 i (ISO 286-1)
    lows = (1,) + RANGES[:-1]
    for k, (lo, hi) in enumerate(zip(lows, RANGES)):
        if lo < 3:
            continue
        D = math.sqrt(lo * hi)
        i = 0.45 * D ** (1 / 3) + 0.001 * D
        assert abs(IT[6][k] - 10 * i) <= 1.0 and abs(IT[7][k] - 16 * i) <= 1.5, (lo, hi)


def test_fea_reports_margin_of_safety():
    from ace_core.analysis import handle_post
    from solvers._example import EXAMPLE_VARIANT as V
    c, b = handle_post("/fea?sync=1", {**V, "mesh": "coarse", "design_factor": 1.5})
    s = b["result"]["summary"]
    assert abs(s["margin_of_safety"] - (s["min_safety_factor"] / 1.5 - 1)) < 1e-3 * max(1.0, abs(s["margin_of_safety"]))
