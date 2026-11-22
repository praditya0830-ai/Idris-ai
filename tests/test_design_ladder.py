"""The design ladder: rungs, shape routing, number sources, and the failure report (no silent box)."""
import design_ladder


def _code_ok(*a, **k):
    return {"ok": True, "cad": {"title": "x", "code": "return { body: makeBox([0,0,0],[100,60,8]) };", "params": []},
            "answer": "code", "read_by": "test"}


def _code_fail(*a, **k):
    return {"ok": False, "cad": None, "answer": "the kernel refused it", "detail": "kernel: zero-length segment"}


def _bracket(**over):
    spec = {"id": "bracket", "name": "Mounting bracket", "kind": "functional", "material": "steel",
            "shape_class": "prismatic",
            "numbers": [{"k": "length_mm", "value": 100, "unit": "mm", "source": "you"},
                        {"k": "width_mm", "value": 60, "unit": "mm", "source": "you"},
                        {"k": "thickness_mm", "value": 8, "unit": "mm", "source": "you"}],
            "holes": [{"d_mm": 8, "x_mm": -38, "y_mm": -18}, {"d_mm": 8, "x_mm": 38, "y_mm": -18},
                      {"d_mm": 8, "x_mm": -38, "y_mm": 18}, {"d_mm": 8, "x_mm": 38, "y_mm": 18}]}
    spec.update(over)
    return spec


def test_prismatic_part_is_built_from_a_recipe_first():
    r = design_ladder.design("mounting bracket 100x60x8 with four M8 holes", {"part_spec": _bracket()},
                             code_fn=_code_ok)
    assert r["ok"] and r["rung"] == "features" and r["fidelity"] == "features"
    assert [s["type"] for s in r["cad"]["feature_tree"]] == ["sketch", "push", "hole", "hole", "hole", "hole"]
    assert all(n["source"] for n in r["cad"]["numbers"])
    assert r["report"] is None


def test_bad_fillet_is_refused_before_the_kernel_and_code_takes_over():
    spec = _bracket(fillet_mm=20)
    r = design_ladder.design("bracket with a big round", {"part_spec": spec}, code_fn=_code_ok)
    assert r["rung"] == "code" and r["features_error"] and "fillet" in r["features_error"].lower()


def test_freeform_starts_with_code_on_purpose():
    spec = _bracket(shape_class="freeform", name="Moldboard")
    r = design_ladder.design("curved moldboard", {"part_spec": spec}, code_fn=_code_ok)
    assert r["rung"] == "code" and "free-form" in r["features_error"]


def test_shape_class_from_words_when_the_plan_has_none():
    cls = design_ladder.shape_class_of({}, "a double curved fairing for the nose")
    assert cls["shape_class"] == "freeform" and cls["source"] == "words"


def test_failure_is_a_report_and_never_a_box_design(monkeypatch):
    import cad_fallback
    monkeypatch.setattr(cad_fallback, "placeholder_cad", lambda m: None)
    r = design_ladder.design("bracket 100x60x8", {"part_spec": _bracket(shape_class="freeform")},
                             code_fn=_code_fail)
    assert r["ok"] is False and r["rung"] == "failed" and r["fidelity"] == "failed" and r["cad"] is None
    rep = r["report"]
    assert rep["part_id"] == "bracket" and rep["rungs_tried"][-1]["rung"] == "code"
    assert rep["requested"]["dims"]["length_mm"] == 100 and rep["suggestion"]


def test_envelope_is_only_a_labelled_failed_preview(monkeypatch):
    import cad_fallback
    monkeypatch.setattr(cad_fallback, "placeholder_cad",
                        lambda m: {"title": "envelope", "code": "return {}", "read_by": "local envelope"})
    r = design_ladder.design("bracket", {"part_spec": _bracket(shape_class="freeform")}, code_fn=_code_fail)
    assert r["fidelity"] == "failed"
    assert r["preview"]["label"] == "failed preview — not a design"


def test_computed_number_needs_a_method_and_inputs():
    spec = _bracket(numbers=[{"k": "thickness_mm", "value": 6, "unit": "mm", "source": "computed"},
                             {"k": "width_mm", "value": 60, "source": "computed", "method": "wall = 0.5*bolt_d (rule of thumb, ASME B1.1 clearance)",
                              "from": ["hole_d_mm"]}],
                    holes=[])
    got = design_ladder.normalise_numbers(spec)
    assert [n["k"] for n in got["numbers"]] == ["width_mm"]
    assert got["numbers"][0]["method"] and got["numbers"][0]["from_"] == ["hole_d_mm"]
    assert got["questions"][0]["k"] == "thickness_mm" and got["dropped"][0]["k"] == "thickness_mm"


def test_guess_is_not_a_source():
    got = design_ladder.normalise_numbers({"numbers": [{"k": "length_mm", "value": 90, "source": "guess"}]})
    assert got["numbers"] == [] and got["questions"] and got["dropped"][0]["k"] == "length_mm"


def test_table_number_is_marked_assumed():
    got = design_ladder.normalise_numbers({"numbers": [{"k": "bolt", "value": 8,
                                                       "source": "table:ISO 273 medium"}]})
    assert got["numbers"][0]["source"] == "table:ISO 273 medium" and got["numbers"][0]["assumed"] is True
