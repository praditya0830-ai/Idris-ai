"""DFM for any part (P3): the measured wall, the sourced rules, and what is honestly not measured.

v0.39.1 adds the checks the UI team found: the wall measurement must not be coarser than the wall it is asked
about, and a count must not be labelled millimetres.
"""
import time

import numpy as np

import solvers  # noqa: F401
from ace_core.http_routes import handle_post
from ace_core.registry import verify
from physics import dfm_part as dp, dfm_rules
from physics.part_fea import triangles_from_input
from solvers.dfm_part import BLOCK, PLATE, _box_mesh


def _extrude(poly, tris, y0, y1):
    """A closed prism: a 2D outline in (x, z), triangulated, extruded from y0 to y1."""
    poly = np.asarray(poly, float)
    n = len(poly)
    V = np.vstack([np.column_stack([poly[:, 0], np.full(n, y0), poly[:, 1]]),
                   np.column_stack([poly[:, 0], np.full(n, y1), poly[:, 1]])])
    F = []
    for a, b, c in tris:
        F.append((a, c, b))                       # bottom cap, normal -y
        F.append((a + n, b + n, c + n))           # top cap, normal +y
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, j + n))
        F.append((i, j + n, i + n))
    return {"positions": V.ravel().tolist(), "indices": np.asarray(F).ravel().tolist()}


def _l_bracket():
    """80 x 60 x 60 L-bracket, every wall 5 mm: the part the UI reported as "7,5 +/- 5 mm"."""
    poly = [(0, 0), (80, 0), (80, 5), (5, 5), (5, 60), (0, 60)]
    return _extrude(poly, [(3, 4, 5), (3, 5, 0), (3, 0, 1), (3, 1, 2)], 0, 60)


def _tube(r_out, r_in, y0, y1, n=64):
    """A closed tube: outer wall, bore and two end rings."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = []
    for r, y in ((r_out, y0), (r_out, y1), (r_in, y0), (r_in, y1)):
        for a in ang:
            V.append([r * np.cos(a), y, r * np.sin(a)])
    V = np.asarray(V, float)

    def quad(a, b, c, d, out):
        tri = V[[a, b, c]]
        if float(np.cross(tri[1] - tri[0], tri[2] - tri[0]) @ np.asarray(out)) < 0:
            a, b, c, d = d, c, b, a
        return [(a, b, c), (a, c, d)]

    F = []
    for i in range(n):
        j = (i + 1) % n
        mid = 0.5 * (ang[i] + ang[j])
        rad = (np.cos(mid), 0.0, np.sin(mid))
        F += quad(i, j, n + j, n + i, rad)
        F += quad(2 * n + i, 3 * n + i, 3 * n + j, 2 * n + j, [-x for x in rad])
        F += quad(i, 2 * n + i, 2 * n + j, j, (0, -1, 0))
        F += quad(n + i, n + j, 3 * n + j, 3 * n + i, (0, 1, 0))
    return {"positions": V.ravel().tolist(), "indices": np.asarray(F).ravel().tolist()}


def _sphere(r=50.0, nt=100, nph=100):
    th, ph = np.linspace(0, np.pi, nt + 1), np.linspace(0, 2 * np.pi, nph + 1)
    V = np.array([[r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)]
                  for t in th for p in ph])
    F = []
    for i in range(nt):
        for j in range(nph):
            a = i * (nph + 1) + j
            F += [(a, a + nph + 1, a + nph + 2), (a, a + nph + 2, a + 1)]
    return {"positions": V.ravel().tolist(), "indices": np.asarray(F).ravel().tolist()}


def _wall(m):
    return [c for c in m["checks"] if c["id"] == "min_wall"][0]


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


# ------------------------------------------------- v0.39.1: the wall is measured, not voxel-quantised
def test_a_5_mm_plate_measures_5_mm_not_one_voxel():
    m = dp.measure(dict(_box_mesh(80, 60, 5), process="cnc", material="aluminium"))
    assert m["wall"]["method_id"] == "exact-ray-triangle"
    assert abs(m["wall"]["min_mm"] - 5.0) <= 0.1
    assert abs(m["wall"]["p5_mm"] - 5.0) <= 0.1 and abs(m["wall"]["median_mm"] - 5.0) <= 0.1
    assert m["wall"]["resolution_mm"] <= 0.01          # the mesh, not a 7.5 mm grid cell


def test_an_l_bracket_with_5_mm_walls_reads_5_mm():
    # the UI saw "7,5 +/- 5 mm": the old grid edge was size.min()/8 = 60/8 = 7.5 mm, coarser than the wall
    m = dp.measure({**_l_bracket(), "process": "cnc", "material": "aluminium"})
    wall = m["wall"]
    assert abs(wall["p5_mm"] - 5.0) <= 0.1, wall
    assert abs(wall["median_mm"] - 5.0) <= 0.1, wall
    assert wall["resolution_mm"] < 1.0
    assert _wall(m)["status"] == "pass" and _wall(m)["value"] == wall["p5_mm"]


def test_a_1_mm_wall_on_a_200_mm_part_is_measured_not_reported_as_at_least():
    m = dp.measure(dict(_box_mesh(200, 200, 1), process="cnc", material="aluminium"))
    wall = m["wall"]
    assert abs(wall["min_mm"] - 1.0) <= 0.05 and abs(wall["median_mm"] - 1.0) <= 0.05
    assert wall["p5_mm"] is not None and wall["at_least_mm"] is None
    assert wall["unresolved"] == 0


def test_a_tube_wall_is_the_wall_between_the_bore_and_the_outside():
    m = dp.measure({**_tube(10, 8, 0, 40), "process": "cnc", "material": "aluminium"})
    assert abs(m["wall"]["p5_mm"] - 2.0) <= 0.1, m["wall"]
    assert abs(m["wall"]["median_mm"] - 2.0) <= 0.1, m["wall"]


def test_a_20k_triangle_mesh_is_measured_in_under_five_seconds():
    tri = triangles_from_input(_sphere())
    assert len(tri) == 20000
    t0 = time.perf_counter()
    wall, *_ = dp.wall_thickness(tri)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"{elapsed:.2f} s for {len(tri)} triangles"
    # a sphere of radius 50: a ray cast inward crosses the diameter, not the radius
    assert abs(wall["median_mm"] - 100.0) <= 1.0, wall


def test_the_answer_says_which_method_measured_the_wall():
    m = dp.measure(dict(PLATE))
    assert "exact ray-triangle" in m["wall"]["method"]
    assert m["wall"]["resolution_mm"] > 0 and m["wall"]["samples"] > 0
    assert "exact-ray-triangle" in dp.summary(m)


def test_the_voxel_fallback_takes_its_cell_from_the_diagonal_not_the_shortest_side(monkeypatch):
    monkeypatch.setattr(dp, "EXACT_MAX_TRIANGLES", 0)          # force the fallback
    wall, *_ = dp.wall_thickness(triangles_from_input(_box_mesh(80, 60, 5)))
    assert wall["method_id"] == "voxel-fallback"
    # 80 x 60 x 5: the shortest side would give 0.625 mm, the diagonal gives about 0.25 mm
    assert 0.05 < wall["resolution_mm"] < 0.3, wall


def test_a_measurement_coarser_than_a_quarter_of_the_limit_is_not_checked(monkeypatch):
    monkeypatch.setattr(dp, "EXACT_MAX_TRIANGLES", 0)
    m = dp.measure({**_box_mesh(200, 200, 10), "process": "fdm", "material": "pla"})
    wall = _wall(m)
    assert m["wall"]["resolution_mm"] > dfm_rules.WALL_MIN_MM["fdm"]["pla"] / 4.0
    assert wall["status"] == "not_checked" and "too coarse" in wall["message"], wall
    assert m["ok"] is True                                  # not a pass and not a failure


# ------------------------------------------------- v0.39.1: a unit belongs to the quantity it measures
def test_every_finding_states_the_unit_of_what_it_measured():
    m = dp.measure(dict(BLOCK, process="fdm", material="pla", holes=[{"d": 6.0}]))
    units = {c["id"]: c["unit"] for c in m["checks"]}
    assert units["min_wall"] == "mm"
    assert units["drill_size"] == "count"                   # one hole, not one millimetre
    assert units["overhang"] == "fraction"
    assert units["draft"] is None and units["undercuts"] is None
    assert units["inside_corner_radius"] is None
    allowed = {"mm", "deg", "count", "ratio", "fraction", None}
    assert all(c["unit"] in allowed for c in m["checks"]), units


def test_a_hole_count_is_never_labelled_millimetres():
    ok = dfm_rules.check({"process": "cnc", "material": "aluminium", "holes": [{"d": 6.0}, {"d": 8.0}]})
    good = [c for c in ok["checks"] if c["id"] == "drill_size"][0]
    assert good["status"] == "pass" and good["value"] == 2 and good["unit"] == "count"
    bad = dfm_rules.check({"process": "cnc", "material": "aluminium", "holes": [{"d": 6.2}]})
    offenders = [c for c in bad["checks"] if c["id"] == "drill_size"][0]
    # the failing value is the list of diameters, so that one is in millimetres
    assert offenders["value"] == [6.2] and offenders["unit"] == "mm"
    nothing = dfm_rules.check({"process": "cnc", "material": "aluminium"})
    assert [c for c in nothing["checks"] if c["id"] == "drill_size"][0]["unit"] is None


def test_the_solver_check_carries_the_unit_too():
    code, body = handle_post("/dfm/part", dict(BLOCK, process="fdm", material="pla"))
    assert code == 200
    units = {f["id"]: f["unit"] for f in body["outputs"]["findings"]}
    assert units["min_wall"] == "mm" and units["overhang"] == "fraction"


# ------------------------------------------------------ X1: the three cases the UI team wrote down
def test_a_60x40x5_plate_measures_5_mm_within_half_a_millimetre():
    m = dp.measure({**_box_mesh(60, 40, 5), "process": "cnc", "material": "aluminium"})
    wall = m["wall"]
    assert wall["method_id"] == "exact-ray-triangle"
    assert abs(wall["p5_mm"] - 5.0) <= 0.5, wall
    assert abs(wall["median_mm"] - 5.0) <= 0.5, wall


def _l_bracket_60_40_45():
    """60 x 40 x 45 L-bracket, every wall 5 mm: the bracket from the UI report, at their dimensions."""
    poly = [(0, 0), (60, 0), (60, 5), (5, 5), (5, 45), (0, 45)]
    return _extrude(poly, [(3, 4, 5), (3, 5, 0), (3, 0, 1), (3, 1, 2)], 0, 40)


def test_an_l_bracket_of_60x40x45_reads_its_5_mm_wall_within_point_6():
    m = dp.measure({**_l_bracket_60_40_45(), "process": "cnc", "material": "aluminium"})
    wall = m["wall"]
    assert abs(wall["p5_mm"] - 5.0) <= 0.6, wall
    assert abs(wall["median_mm"] - 5.0) <= 0.6, wall
    assert wall["resolution_mm"] < 1.0 and wall["method_id"] == "exact-ray-triangle"


def test_a_100_mm_cube_is_measured_in_under_ten_seconds():
    t0 = time.perf_counter()
    m = dp.measure({**_box_mesh(100, 100, 100), "process": "cnc", "material": "aluminium"})
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"{elapsed:.2f} s for a 100 mm cube"
    assert abs(m["wall"]["p5_mm"] - 100.0) <= 1.0, m["wall"]


def test_the_drill_check_lists_each_diameter_and_the_drill_that_makes_it():
    m = dp.measure(dict(PLATE, holes=[{"d": 6.0}, {"d": 6.0}, {"d": 6.2}]))
    drill = [c for c in m["checks"] if c["id"] == "drill_size"][0]
    rows = {r["d_mm"]: r for r in drill["diameters_mm"]}
    assert rows[6.0]["standard_drill_mm"] == 6.0 and rows[6.0]["in_series"] is True
    assert rows[6.0]["holes"] == 2                     # the same diameter twice is one row, two holes
    assert rows[6.2]["in_series"] is False and rows[6.2]["standard_drill_mm"] == 6.0
    assert drill["status"] == "fail" and drill["unit"] == "mm"


def test_a_passing_drill_check_carries_the_same_table():
    ok = dfm_rules.check({"process": "cnc", "material": "aluminium", "holes": [{"d": 8.0}, {"d": 8.0}]})
    drill = [c for c in ok["checks"] if c["id"] == "drill_size"][0]
    assert drill["status"] == "pass" and drill["unit"] == "count"
    assert drill["diameters_mm"] == [{"d_mm": 8.0, "standard_drill_mm": 8.0, "in_series": True, "holes": 2}]
