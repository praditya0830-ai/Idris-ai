"""Bagian H tests: the goal map is truthful about which quantities IDRIS can measure, the shape families keep
the engineer's sizes locked, and a ranking never uses a number that was not measured by a named solver.
"""
import json

import shape_ext
from ace_core import objectives as obj

QUIET = "desain motor listrik yang tidak bising"


def _fake_call(payload):
    return lambda system, user: json.dumps(payload)


# ------------------------------------------------------------------ H1: goal -> measurable quantities
def test_a_quiet_motor_maps_to_its_quantities_with_the_real_solver_status():
    m = obj.measurable(QUIET)
    assert m["goals"] == ["quiet"]
    by_key = {q["key"]: q for q in m["quantities"]}
    assert by_key["cogging_torque"]["status"] == "not_available", "no cogging solver exists in this build"
    assert by_key["torque_ripple"]["status"] == "not_available"
    assert by_key["fan_blade_tone"]["status"] == "not_available"
    # review item 5: there is no modal solver for a free part in this build, and the motor's electromagnetic
    # excitation frequency is not computed by IDRIS at all, so a "gap" would be a number with one half missing
    assert by_key["resonance_gap"]["status"] == "not_available"
    assert by_key["resonance_gap"]["solver"] is None
    assert "Cogging torque" in m["not_checked"]


def test_mass_is_measured_by_the_kernel_not_estimated():
    m = obj.measurable("plat ringan 200 mm")
    by_key = {q["key"]: q for q in m["quantities"]}
    assert by_key["mass"]["status"] == "kernel"
    assert "density" in by_key["mass"]["benchmark"]
    assert m["scoreable"] == ["mass"]


def test_no_measurable_goal_says_so_instead_of_scoring():
    m = obj.measurable(QUIET)
    assert m["scoreable"] == [], "nothing in the quiet-motor goal can be measured in this build"
    assert m["note"] == obj.NOT_CHECKED_NOTE
    m2 = obj.measurable("kursi yang nyaman")
    assert m2["goals"] == [] and m2["quantities"] == []
    assert "No measurable goal" in m2["note"]


# ------------------------------------------------------------------ H2: shape families
def test_families_are_parametric_and_the_engineer_size_is_locked():
    payload = {"families": [
        {"name": "Circular ribs", "principle": "ribs break the ring mode", "changes": ["rib direction"],
         "design_space": [{"name": "thickness", "min": 200, "max": 200, "unit": "mm", "locked": True},
                          {"name": "rib_height", "min": 4, "max": 18, "unit": "mm"}]},
        {"name": "Finned housing", "principle": "fins add radiating area", "changes": ["fin pattern"],
         "design_space": [{"name": "fin_gap", "min": 6, "max": 20, "unit": "mm"}]},
        {"name": "Skewed magnets", "principle": "skew lowers the cogging order",
         "changes": ["magnet skew"], "design_space": [{"name": "skew_deg", "min": 0, "max": 12, "unit": "deg"}]},
    ]}
    out = obj.families("plat 200 mm " + QUIET, 3, call=_fake_call(payload))
    assert out["ok"] is True and len(out["families"]) == 3
    assert out["locked_sizes"] == [200.0]
    thick = out["families"][0]["design_space"][0]
    assert thick["locked"] is True and thick["min"] == 200.0
    assert out["families"][0]["design_space"][1]["locked"] is False
    assert out["distinct_names"] is True


def test_a_short_family_list_is_reported_not_padded():
    payload = {"families": [{"name": "Only one", "principle": "p", "design_space": []}]}
    out = obj.families(QUIET, 3, call=_fake_call(payload))
    assert out["ok"] is False and len(out["families"]) == 1
    assert "3 were asked for" in out["error"], "IDRIS does not invent the missing families"


def test_a_broken_reply_is_a_clear_error():
    out = obj.families(QUIET, 3, call=lambda s, u: "sorry, I cannot")
    assert out["ok"] is False and out["families"] == []
    assert "structured data" in out["error"]


def test_the_family_count_is_limited_to_the_allowed_set():
    out = shape_ext.handle_post("/shape/families", {"request": QUIET, "count": 4})
    assert out[0] == 422 and out[1]["allowed"] == [3, 5, 7, 10]


# ------------------------------------------------------------------ H4: ranking from measured values only
def _measured():
    return {q["key"]: q for q in obj.measurable("plat ringan dan kaku " + QUIET)["quantities"]}


def test_ranking_uses_measured_values_and_names_the_solver():
    out = obj.rank([
        {"name": "A", "values": {"mass": 2.0, "max_deflection": 0.9}, "sources": {"mass": "kernel"},
         "checks": {"pass": True}},
        {"name": "B", "values": {"mass": 1.0, "max_deflection": 0.4}, "sources": {"mass": "kernel"},
         "checks": {"pass": True}},
    ], None, _measured())
    assert [r["name"] for r in out["ranking"]] == ["B", "A"], "lighter and stiffer wins on equal weights"
    assert out["ranking"][0]["measured_by"]["mass"] == "kernel"
    assert "fea_part_3d" in out["ranking"][0]["measured_by"]["max_deflection"]


def test_a_variant_without_a_measured_value_is_not_ranked():
    out = obj.rank([
        {"name": "A", "values": {"mass": 2.0}, "checks": {"pass": True}},
        {"name": "B", "values": {"cogging_torque": 0.1}, "checks": {"pass": True}},
    ], None, _measured())
    assert [r["name"] for r in out["ranking"]] == ["A"]
    assert out["not_ranked"][0]["name"] == "B"
    assert "no measured objective value" in out["not_ranked"][0]["reason"], \
        "a quantity without a solver can never enter a score"


def test_a_failed_gate_is_excluded_with_its_reason():
    out = obj.rank([
        {"name": "A", "values": {"mass": 1.0}, "checks": {"pass": False, "reason": "SF 1.1 < 2.0"}},
        {"name": "B", "values": {"mass": 3.0}, "checks": {"pass": True}},
    ], None, _measured())
    assert [r["name"] for r in out["ranking"]] == ["B"]
    assert "SF 1.1 < 2.0" in out["not_ranked"][0]["reason"]


def test_weights_are_honoured():
    results = [{"name": "A", "values": {"mass": 1.0, "max_deflection": 0.9}, "checks": {"pass": True}},
               {"name": "B", "values": {"mass": 3.0, "max_deflection": 0.1}, "checks": {"pass": True}}]
    light = obj.rank(results, {"mass": 10.0, "max_deflection": 0.1}, _measured())
    stiff = obj.rank(results, {"mass": 0.1, "max_deflection": 10.0}, _measured())
    assert [r["name"] for r in light["ranking"]] == ["A", "B"]
    assert [r["name"] for r in stiff["ranking"]] == ["B", "A"]


def test_nothing_measured_means_nothing_ranked():
    out = obj.rank([{"name": "A", "values": {}}], None, _measured())
    assert out["ranking"] == []
    assert "nothing is ranked" in out["summary"]
