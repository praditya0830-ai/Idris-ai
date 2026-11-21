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


# ------------------------------------------------------------ X2: the features v67 left in "Not translated"
PLATE = {
    "model_id": "plate", "revision": 8,
    "features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 80, "h": 50}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "sketch", "deps": ["f2"],
         "params": {"plane": "XY", "profile": "circle", "w": 20, "cx": 20, "cy": 5}},
        {"id": "f4", "op": "extrude", "deps": ["f3"], "params": {"distance": 3, "op": "cut"}},
        {"id": "f5", "op": "hole", "deps": ["f4"],
         "params": {"x": -20, "y": 0, "d": 6, "through": False, "depth": 8, "from": "top", "z_top": 10}},
        {"id": "f6", "op": "hole", "deps": ["f5"],
         "params": {"x": 30, "y": -10, "d": 5, "through": True, "cbore_d": 9, "cbore_depth": 4}},
        {"id": "f7", "op": "chamfer", "deps": ["f6"], "params": {"distance": 1}},
        {"id": "f8", "op": "mirror", "deps": ["f7"], "params": {"plane": "YZ"}},
    ],
}


def test_every_feature_of_the_plate_is_translated_and_nothing_is_left_out():
    rep = macro_from_model(PLATE, name="plate")
    assert rep["skipped"] == []
    assert [t["op"] for t in rep["translated"]] == ["sketch", "extrude", "sketch", "extrude", "hole", "hole",
                                                     "chamfer", "mirror"]
    assert rep["translated"][3]["kind"] == "cut"
    ok, problems = check_vba(rep["vba"])
    assert ok, problems


def test_an_off_centre_sketch_is_drawn_at_its_centre_not_at_the_origin():
    text = macro_from_model(PLATE)["vba"]
    assert "CreateCircleByRadius(0.02, 0.005, 0, 0.01)" in text        # 20 mm, 5 mm, r 10 mm
    assert "centre (20, 5) mm" in text


def test_an_extruded_cut_is_a_cut_and_not_a_boss():
    text = macro_from_model(PLATE)["vba"]
    assert "cut extrude 3 mm" in text
    assert "FeatureCut4(True, False, .False., 0, 0, 0.003, 0.003" in text
    assert "FeatureExtrusion2(" in text                    # the boss of f2 is still a boss


def test_a_blind_hole_from_the_top_starts_on_a_plane_at_z_top():
    text = macro_from_model(PLATE)["vba"]
    assert "from the top face (z = 10 mm), depth 8 mm" in text
    assert "InsertRefPlane(8, 0.01, 0, 0, 0, 0)" in text   # 8 = swRefPlaneReferenceConstraint_Distance
    assert "FeatureCut4(True, False, False, 0, 0, 0.008, 0.008" in text


def test_a_counterbore_becomes_two_cuts_and_says_so():
    rep = macro_from_model(PLATE)
    assert "CreateCircleByRadius(0.03, -0.01, 0, 0.0025)" in rep["vba"]      # the through hole, d = 5
    assert "CreateCircleByRadius(0.03, -0.01, 0, 0.0045)" in rep["vba"]      # the counterbore, d = 9
    assert "FeatureCut4(True, False, False, 1, 0, 0, 0" in rep["vba"]      # through all
    assert "FeatureCut4(True, False, False, 0, 0, 0.004, 0.004" in rep["vba"]  # 4 mm deep
    assert any("two cuts" in m["what"] for m in rep["manual"])


def test_a_chamfer_is_an_equal_distance_chamfer_on_the_selected_edges():
    text = macro_from_model(PLATE)["vba"]
    assert "InsertFeatureChamfer(0, 18, 0.001, 0, 0.001, 0.001, 0.001, 0)" in text
    assert "chamfer 1 mm (equal distance)" in text


def test_a_mirror_selects_the_feature_above_it_and_the_named_plane():
    text = macro_from_model(PLATE)["vba"]
    assert 'SelectByID2("Right Plane", "PLANE", 0, 0, 0, True, 2, Nothing, 0)' in text
    assert "FeatureByPositionReverse(1)" in text and "swFeat.Select2(False, 1)" in text
    assert "InsertMirrorFeature2(False, False, False, False, 0)" in text


def test_a_hole_from_the_top_without_z_top_is_a_stated_skip_not_a_guess():
    model = {"features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "hole", "deps": ["f2"],
         "params": {"x": 0, "y": 0, "d": 6, "through": False, "depth": 4, "from": "top"}},
    ]}
    rep = macro_from_model(model)
    assert [s["op"] for s in rep["skipped"]] == ["hole"] and "z_top" in rep["skipped"][0]["reason"]
    ok, problems = check_vba(rep["vba"])
    assert ok, problems


def test_a_not_translated_note_from_the_studio_is_carried_as_a_skip():
    model = {"features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "not_translated", "deps": ["f2"],
         "params": {"what": "tube", "why": "no macro mapping for this primitive"}},
    ]}
    rep = macro_from_model(model)
    assert rep["skipped"] == [{"feature": "f3", "op": "tube",
                               "reason": "no macro mapping for this primitive"}]
    assert "NOT TRANSLATED (tube)" in rep["vba"]


def test_the_route_returns_the_full_plate_without_a_skip():
    code, body = handle_post("/exchange/solidworks-macro", {"model": PLATE, "name": "plate"})
    assert code == 200 and body["success"] is True
    assert body["vba_check"]["ok"] is True and len(body["translated"]) == 8
    assert body["not_executed"].startswith("this macro has not been run in SolidWorks")
