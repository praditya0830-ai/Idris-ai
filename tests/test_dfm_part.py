"""DFM for any part (P3): the measured wall, the sourced rules, and what is honestly not measured."""
import numpy as np

import solvers  # noqa: F401
from ace_core.http_routes import handle_post
from ace_core.registry import verify
from physics import dfm_part as dp, dfm_rules
from solvers.dfm_part import BLOCK, PLATE, _box_mesh


def test_benchmarks_pass():
    r = verify("dfm_part")
    assert r["all_pass"] and r["passed"] == 2


def test_a_plate_measures_its_own_thickness():
    m = dp.measure(dict(PLATE))
    assert m["wall"]["min_mm"] is not None
    # the grid quantises the measurement to one voxel; the plate is 2 mm
    assert abs(m["wall"]["median_mm"] - 2.0) <= m["wall"]["resolution_mm"]
    assert m["wall"]["resolution_mm"] > 0


def test_the_wall_check_uses_the_measured_value_and_a_sourced_limit():
    m = dp.measure(dict(BLOCK, process="fdm", material="pla"))
    wall = [c for c in m["checks"] if c["id"] == "min_wall"][0]
    assert wall["value"] == m["wall"]["p5_mm"] and wall["limit"] == dfm_rules.WALL_MIN_MM["fdm"]["pla"]
    assert wall["source_status"] == dfm_rules.UNVERIFIED


def test_a_thin_wall_below_the_guideline_is_a_warning_not_a_failure():
    # a 0.5 mm plate in PLA on FDM: below the 1.2 mm guideline, whose source is unverified
    thin = {**_box_mesh(20, 20, 0.5), "process": "fdm", "material": "pla"}
    m = dp.measure(thin)
    wall = [c for c in m["checks"] if c["id"] == "min_wall"][0]
    assert wall["status"] == "warn", wall
    assert m["ok"] is True                       # a guideline with an unverified source never fails a part


def test_a_declared_hole_outside_the_drill_series_fails_and_names_the_rule():
    m = dp.measure(dict(PLATE, holes=[{"d": 6.0}, {"d": 6.2}]))
    drill = [c for c in m["checks"] if c["id"] == "drill_size"][0]
    assert drill["status"] == "fail" and drill["value"] == [6.2]
    assert "DIN 338" in drill["source"] and m["ok"] is False


def test_holes_that_are_not_declared_are_not_guessed():
    m = dp.measure(dict(PLATE))
    drill = [c for c in m["checks"] if c["id"] == "drill_size"][0]
    assert drill["status"] == "not_checked" and "declared" in drill["message"]


def test_overhang_is_measured_for_fdm_and_not_claimed_for_other_processes():
    fdm = dp.measure(dict(BLOCK, process="fdm", material="pla"))
    oh = [c for c in fdm["checks"] if c["id"] == "overhang"][0]
    assert oh["status"] in ("pass", "warn") and fdm["overhang"]["area_fraction"] >= 0
    cnc = dp.measure(dict(BLOCK, process="cnc", material="aluminium"))
    assert [c for c in cnc["checks"] if c["id"] == "overhang"][0]["status"] == "not_checked"


def test_draft_is_a_cast_rule_and_says_the_guideline_is_unverified():
    m = dp.measure(dict(BLOCK, process="cast", material="cast_iron"))
    dr = [c for c in m["checks"] if c["id"] == "draft"][0]
    assert dr["status"] in ("pass", "warn") and m["draft"] is not None
    cnc = dp.measure(dict(BLOCK, process="cnc", material="aluminium"))
    assert [c for c in cnc["checks"] if c["id"] == "draft"][0]["status"] == "not_checked"


def test_undercuts_and_inside_corners_stay_not_checked():
    m = dp.measure(dict(PLATE))
    ids = {c["id"]: c["status"] for c in m["checks"]}
    assert ids["undercuts"] == "not_checked" and ids["inside_corner_radius"] == "not_checked"


def test_the_contour_carries_the_measured_thickness_per_vertex():
    m = dp.measure(dict(BLOCK))
    pos, th = m["surface"]["positions"], m["surface"]["thickness"]
    assert len(pos) == 3 * len(th) and any(t is not None for t in th)
    finite = [t for t in th if t is not None]
    assert min(finite) > 0 and max(finite) >= min(finite)


def test_the_route_answers_with_the_solver_envelope():
    code, body = handle_post("/dfm/part", dict(BLOCK))
    assert code == 200 and body["ok"] is True and body["solver"] == "dfm_part"
    assert body["outputs"]["wall"]["min_mm"] > 0 and body["outputs"]["summary"].startswith("Manufacturability")


def test_a_mesh_with_no_volume_is_refused():
    bad = {"positions": [0, 0, 0, 1, 0, 0, 0, 1, 0], "indices": [0, 1, 2], "process": "cnc", "material": "steel"}
    code, body = handle_post("/dfm/part", bad)
    assert code == 400 and body["ok"] is False


def test_sls_and_cast_now_have_a_wall_rule_that_is_marked_a_guideline():
    for process in ("sls", "cast"):
        assert process in dfm_rules.WALL_MIN_MM and process in dfm_rules.WALL_SOURCE
        assert dfm_rules.WALL_SOURCE[process]["status"] == dfm_rules.UNVERIFIED
    assert dfm_rules.WALL_MIN_MM["cast"]["cast_iron"] > 0
    assert np.isfinite(dfm_rules.weld_fillet_min_mm(10.0))
