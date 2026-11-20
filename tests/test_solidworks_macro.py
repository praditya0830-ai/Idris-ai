"""SolidWorks rebuild macro (P5b): the feature tree becomes parametric API calls -- and is not claimed as run."""
from engine_routes import handle_post
from physics.sw_macro import check_vba, macro_from_model

MODEL = {
    "model_id": "bracket", "revision": 4,
    "features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10, "direction": "+normal"}},
        {"id": "f3", "op": "hole", "deps": ["f2"], "params": {"x": 10, "y": 5, "d": 6, "through": True}},
        {"id": "f4", "op": "fillet", "deps": ["f3"], "params": {"radius": 2, "edges": "all"}},
        {"id": "f5", "op": "pattern", "deps": ["f4"], "params": {"mode": "linear", "count": 3, "dx": 15}},
        {"id": "f6", "op": "import", "params": {"path": "part.step"}},
    ],
}


def test_the_macro_is_valid_vba_by_our_own_checker():
    rep = macro_from_model(MODEL, name="bracket")
    ok, problems = check_vba(rep["vba"])
    assert ok, problems
    assert rep["vba"].startswith("Option Explicit")


def test_every_feature_is_either_a_call_or_a_stated_skip():
    rep = macro_from_model(MODEL, name="bracket")
    ops = {f["op"] for f in rep["translated"]}
    assert {"sketch", "extrude", "hole", "fillet", "pattern"} <= ops
    assert [s["op"] for s in rep["skipped"]] == ["import"]
    assert "NOT TRANSLATED (import)" in rep["vba"]
    assert rep["vba"].count("' [") >= 6                     # every feature is tagged in the text


def test_lengths_are_metres_and_the_dimensions_are_real_dimensions():
    rep = macro_from_model(MODEL, name="bracket")
    text = rep["vba"]
    assert "CreateCornerRectangle(-0.02, -0.01, 0, 0.02, 0.01, 0)" in text
    assert "FeatureExtrusion2(" in text and "0.01, 0.01" in text        # 10 mm depth, twice (forward/backward)
    assert "AddDimension2(" in text and "swDim.SystemValue = 0.04" in text
    assert "CreateCircleByRadius(0.01, 0.005, 0, 0.003)" in text       # hole d=6 mm at (10, 5)


def test_the_answer_says_it_was_never_run_in_solidworks():
    rep = macro_from_model(MODEL, name="bracket")
    assert any("NOT executed in SolidWorks" in n for n in rep["notes"])
    assert any("Hole Wizard" in m["what"] for m in rep["manual"])


def test_a_model_without_a_feature_tree_is_refused_with_a_reason():
    try:
        macro_from_model({"features": []}, name="ai-part")
        assert False
    except ValueError as e:
        assert "no feature tree" in str(e)


def test_the_route_returns_the_macro_with_its_checks():
    code, body = handle_post("/exchange/solidworks-macro", {"model": MODEL, "name": "bracket"})
    assert code == 200 and body["success"] is True
    assert body["filename"].endswith("_rebuild.bas")
    assert body["vba_check"]["ok"] is True
    assert body["not_executed"].startswith("this macro has not been run in SolidWorks")
    assert body["checks"][0]["ok"] is True


def test_the_route_tells_a_code_written_part_to_use_step_instead():
    code, body = handle_post("/exchange/solidworks-macro", {"model": {"features": []}})
    assert code == 400 and "STEP" in body["hint"]
