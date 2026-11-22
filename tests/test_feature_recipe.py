"""Feature recipes: templates and the pre-build checks that refuse a bad tree before a kernel call."""
import feature_recipe


def _nums(**over):
    n = {"length_mm": 100, "width_mm": 60, "thickness_mm": 8}
    n.update(over)
    return n


def _holes():
    return [{"d_mm": 8, "x_mm": -38, "y_mm": -18}, {"d_mm": 8, "x_mm": 38, "y_mm": -18},
            {"d_mm": 8, "x_mm": -38, "y_mm": 18}, {"d_mm": 8, "x_mm": 38, "y_mm": 18}]


def test_plate_template_is_six_ordered_steps():
    rec = feature_recipe.template_for("prismatic", _nums(), _holes())
    assert rec and len(rec["steps"]) == 6
    assert [s["type"] for s in rec["steps"]] == ["sketch", "push", "hole", "hole", "hole", "hole"]
    ok, errs = feature_recipe.validate(rec, _nums())
    assert ok, errs


def test_hole_before_the_solid_is_refused():
    rec = feature_recipe.template_for("prismatic", _nums(), _holes())
    bad = {"steps": [rec["steps"][0], rec["steps"][2], rec["steps"][1]]}
    ok, errs = feature_recipe.validate(bad, _nums())
    assert not ok and any("order" in e for e in errs)


def test_hole_outside_the_face_is_refused():
    rec = feature_recipe.template_for("prismatic", _nums(), [{"d_mm": 8, "x_mm": 60, "y_mm": 0}])
    ok, errs = feature_recipe.validate(rec, _nums())
    assert not ok and any("outside the face" in e for e in errs)


def test_fillet_bigger_than_the_material_is_refused():
    bad = feature_recipe.template_for("prismatic", _nums(), [], fillet=20)
    ok, errs = feature_recipe.validate(bad, _nums())
    assert not ok and any("too big" in e for e in errs)
    good = feature_recipe.template_for("prismatic", _nums(), [], fillet=2)
    assert feature_recipe.validate(good, _nums())[0]


def test_open_sketch_from_lines_is_refused():
    rec = {"steps": [{"id": "s", "type": "sketch", "p": {"prof": "ents", "ents": [
        {"a": [0, 0]}, {"a": [10, 0]}, {"a": [10, 10]}]}},
        {"id": "p", "type": "push", "p": {"sketch": "s", "dist": 5}}]}
    ok, errs = feature_recipe.validate(rec, {})
    assert not ok and any("do not close" in e for e in errs)


def test_no_template_for_a_freeform_shape():
    assert feature_recipe.template_for("freeform", _nums(), _holes()) is None
