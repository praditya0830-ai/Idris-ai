"""Mesh-convergence study (v0.17): GCI per Celik et al. 2008, and its use in the housing FEA."""
import math
from physics.gci import celik_gci, converged, two_level


def test_celik_2008_table1():
    rep = celik_gci([6.063, 5.972, 5.863], [1.0, 1.5, 1.5 * 1.333])
    assert rep["convergence"] == "monotonic"
    assert abs(rep["observed_order"] - 1.53) < 0.01
    assert abs(rep["phi_extrapolated"] - 6.1685) < 1e-3
    assert abs(rep["e_a21"] - 0.015) < 5e-4 and abs(rep["e_ext21"] - 0.017) < 5e-4
    assert abs(rep["gci_fine21"] - 0.022) < 5e-4
    assert converged(rep)


def test_second_order_series_recovers_order_and_exact_value():
    h = [1.0, 1.7, 2.6]                              # non-constant ratio on purpose
    phi = [10.0 + 0.3 * x ** 2 for x in h]
    rep = celik_gci(phi, h)
    assert abs(rep["observed_order"] - 2.0) < 1e-6
    assert abs(rep["phi_extrapolated"] - 10.0) < 1e-6
    assert rep["in_asymptotic_range"]


def test_oscillatory_gets_no_gci():
    rep = celik_gci([10.98, 11.11, 6.03], [2.49, 3.36, 6.17])
    assert rep["convergence"] == "oscillatory" and "gci_fine21" not in rep and not converged(rep)
    assert math.isclose(rep["spread_rel"], (11.11 - 6.03) / 10.98, rel_tol=1e-9)


def test_meshes_must_get_finer():
    assert celik_gci([1.0, 1.1, 1.2], [2.0, 1.5, 1.0])["convergence"] == "invalid"


def test_two_levels_report_change_only():
    rep = two_level(7.0, 4.4)
    assert rep["convergence"] == "two meshes only" and abs(rep["relative_change"] - 2.6 / 7.0) < 1e-12


def test_housing_fea_reports_mesh_convergence(monkeypatch):
    # cap the server so only coarse + standard fit: exercises the two-level path quickly
    import solvers  # noqa: F401
    import solvers.structural3d as s3
    from ace_core.analysis import handle_post
    from physics.gears import Requirements
    from physics.variants import generate_variants
    req = Requirements(power_kw=11.0, input_rpm=6000, gear_ratio=6, env_w=250, env_h=200, env_d=180,
                       wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                       min_sf_bending=1.4, min_sf_contact=1.1)
    v = generate_variants(req, compare_basis=False)["variants"][0]
    monkeypatch.setattr(s3, "MAX_ELEMENTS", 25000)
    c, b = handle_post("/fea?sync=1", {**v, "mesh": "coarse", "convergence_study": True})
    assert c == 200, b
    mc = b["result"]["mesh_convergence"]
    assert [lv["mesh"] for lv in mc["levels"]] == ["standard", "coarse"] and mc["skipped_levels"] == ["fine"]
    assert mc["quantities"]["max_von_mises_MPa"]["convergence"] == "two meshes only"
    worst = max(lv["max_von_mises_MPa"] for lv in mc["levels"])
    assert math.isclose(mc["safety_factor_worst_level"], b["result"]["yield_mpa"] / worst, rel_tol=1e-3)
    assert any("every mesh level" in w for w in b["result"]["warnings"])
    # off by default: no extra solves
    c, b = handle_post("/fea?sync=1", {**v, "mesh": "coarse"})
    assert b["result"]["mesh_convergence"] is None
