"""S19: the sheet-metal and surface features that used to be refused are real steps now.

Three things are checked here, all without an AI or a browser:

  * the formulas the kernel's geometry matches (bend allowance, hem gap, jog minimum rise, punch radius, cap volume);
  * the deterministic reader - a size that was not typed is never invented, and a feature that cannot be built is
    refused with a sentence about this part;
  * the ladder: a chat sentence about a hem, a jog and an emboss turns into a features rung whose tree the Studio can
    build, with no `part_spec` injected by the test.
"""
import math

import feature_recipe as fr
import part_spec
import sheet_features as sf
from design_ladder import design


def test_bend_allowance_and_hem_gap_match_the_kernel_formulas():
    assert math.isclose(sf.bend_allowance(90, 2, 0.44, 2), (math.pi / 2) * (2 + 0.44 * 2), rel_tol=1e-12)
    assert math.isclose(sf.bend_allowance(180, 2, 0.44, 2), math.pi * (2 + 0.44 * 2), rel_tol=1e-12)
    # the return layer of a 180 degree fold sits 2R + t above the plate, so the sheet never touches itself
    assert sf.hem_gap(2, 2) == 6
    assert sf.hem_gap(2, 2) > 2
    assert math.isclose(sf.hem_developed(2, 2, 0.44, 10), math.pi * (2 + 0.88) + 10, rel_tol=1e-12)


def test_jog_minimum_rise_and_allowance():
    assert sf.jog_min_rise(2, 2) == 2 + 2 + 3
    ok, why = sf.check_jog(2, 2, 100, 5, 200)
    assert not ok and "above 7.00 mm" in why
    ok, why = sf.check_jog(2, 2, 100, 15, 200)
    assert ok, why
    assert math.isclose(sf.jog_developed(2, 2, 0.44), 2 * (math.pi / 2) * (2 + 0.88), rel_tol=1e-12)


def test_forming_tool_radius_and_cap_volume():
    assert math.isclose(sf.dome_radius(16, 3), (64 + 9) / 6, rel_tol=1e-12)
    assert math.isclose(sf.dome_volume(16, 3), math.pi * 3 / 6 * (3 * 64 + 9), rel_tol=1e-12)
    ok, why = sf.check_form(16, 10, 50, 30, 100, 60)
    assert not ok and "deeper than its own radius" in why
    ok, why = sf.check_form(16, 3, 50, 30, 100, 60)
    assert ok, why
    # x and y are read in the sheet's own frame (0..L, 0..W), which is how the kernel builds it
    ok, why = sf.check_form(16, 3, 0, 0, 100, 60)
    assert not ok and "not fully on the sheet" in why


def test_hem_longer_than_half_the_sheet_is_refused():
    ok, why = sf.check_hem(2, 2, 80, 200, 120)
    assert not ok and "fold past the middle" in why


MSG = ("a sheet metal tray 200x120x2 mm with a 12 mm hem on the +X edge, "
       "a jog rise 15 mm at 100 mm, and an emboss \u00d816 x 3 mm deep")


def test_reader_takes_only_the_sizes_that_were_typed():
    spec = part_spec.from_message(MSG)
    assert spec["shape_class"] == "sheet"
    sheet = spec["sheet_features"]
    assert [f["type"] for f in sheet["features"]] == ["hem", "jog", "form"]
    assert all(f["ok"] for f in sheet["features"]), [f["why"] for f in sheet["features"]]
    hem = next(f for f in sheet["features"] if f["type"] == "hem")
    assert hem["p"] == {"edge": "x+", "F": 12.0}
    jog = next(f for f in sheet["features"] if f["type"] == "jog")
    assert jog["p"] == {"d": 100.0, "h": 15.0}
    form = next(f for f in sheet["features"] if f["type"] == "form")
    assert form["p"]["D"] == 16.0 and form["p"]["depth"] == 3.0 and form["p"]["kind"] == "emboss"
    assert form["p"]["x"] == 100.0 and form["p"]["y"] == 60.0     # the middle of the 200x120 sheet
    # every number says where it came from: the typed sizes are `you`, the K is the workshop table, the rest computed
    by_k = {n["k"]: n for n in sheet["numbers"]}
    assert by_k["hem_length_mm"]["source"] == "you"
    assert by_k["k_factor"]["source"].startswith("table:")
    assert by_k["bend_radius_mm"]["source"] == "computed" and by_k["bend_radius_mm"]["method"]
    assert "method" in by_k["punch_radius_mm"] and by_k["punch_radius_mm"]["from"]


def test_a_missing_size_is_asked_never_guessed():
    got = sf.parse("a sheet metal plate 200x120x2 mm with a hem", {"length_mm": 200, "width_mm": 120,
                                                                   "thickness_mm": 2})
    assert not got["features"]
    assert got["questions"] and got["questions"][0]["k"] == "hem_length_mm"


def test_ladder_builds_the_sheet_tree_without_any_injected_spec():
    spec = part_spec.from_message(MSG)                 # the normal chat door, deterministic parser
    out = design(MSG, {"part_spec": spec})
    assert out["ok"] and out["rung"] == "features", out.get("report")
    steps = out["cad"]["feature_tree"]
    assert [s["type"] for s in steps] == ["smbase", "hem", "jog", "form"]
    ok, errs = fr.validate({"steps": steps}, {k: v for k, v in out["cad"]["sources"].items()})
    assert ok, errs
    base = steps[0]["p"]
    assert base["L"] == 200 and base["W"] == 120 and base["t"] == 2 and base["R"] == 2
    assert steps[1]["p"]["F"] == 12 and steps[2]["p"]["h"] == 15 and steps[3]["p"]["D"] == 16


def test_validation_refuses_a_hem_that_would_fold_past_the_middle():
    rec = {"steps": [
        {"id": "base", "type": "smbase", "p": {"L": 200, "W": 120, "t": 2, "R": 2}},
        {"id": "hem1", "type": "hem", "p": {"edge": "x+", "F": 90}},
    ]}
    ok, errs = fr.validate(rec, {})
    assert not ok
    assert any("fold past the middle" in e for e in errs), errs


def test_validation_refuses_a_jog_too_short_to_clear_its_bends():
    rec = {"steps": [
        {"id": "base", "type": "smbase", "p": {"L": 200, "W": 120, "t": 2, "R": 2}},
        {"id": "jog1", "type": "jog", "p": {"d": 100, "h": 5}},
    ]}
    ok, errs = fr.validate(rec, {})
    assert not ok and any("cannot clear both 90 degree bends" in e for e in errs), errs


def test_validation_accepts_the_analysis_steps_and_checks_their_sizes():
    rec = {"steps": [
        {"id": "base", "type": "smbase", "p": {"L": 100, "W": 60, "t": 2, "R": 2}},
        {"id": "bend", "type": "hem", "p": {"edge": "x+", "F": 10}},
        {"id": "m", "type": "motion", "p": {"axis": "Z", "angle": 90, "steps": 8}},
        {"id": "sz", "type": "sizes", "p": {"step": "base", "param": "L", "value": 120}},
        {"id": "c", "type": "curv", "p": {"kind": "curv"}},
    ]}
    ok, errs = fr.validate(rec, {})
    assert ok, errs
    rec["steps"][3]["p"]["value"] = 0
    ok, errs = fr.validate(rec, {})
    assert not ok and any("value above 0" in e for e in errs), errs


def test_boundary_surface_needs_exactly_four_curves():
    rec = {"steps": [
        {"id": "sk", "type": "sketch", "p": {"prof": "ents", "plane": "XZ",
                                             "ents": [{"t": "line", "a": [0, 0], "b": [10, 0]},
                                                      {"t": "line", "a": [10, 0], "b": [10, 10]},
                                                      {"t": "line", "a": [10, 10], "b": [0, 10]},
                                                      {"t": "line", "a": [0, 10], "b": [0, 0]}]}},
        {"id": "patch", "type": "surfbound", "p": {"sketch": "sk", "t": 0, "solid": "sheet"}},
    ]}
    ok, errs = fr.validate(rec, {})
    assert ok, errs
    rec["steps"][0]["p"]["ents"] = rec["steps"][0]["p"]["ents"][:3]
    ok, errs = fr.validate(rec, {})
    assert not ok and any("four boundary curves" in e for e in errs), errs
