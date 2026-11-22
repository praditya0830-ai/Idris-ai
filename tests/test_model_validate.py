"""`POST /model/validate` for the graph the Studio sends to the SolidWorks macro (X3).

The macro path and the validator have to speak one contract: every op the Studio can send is either accepted
with its parameters resolved, or rejected with a message that names the feature and the field. A dimension that
cannot be resolved is an error in the answer, not a 500.
"""
import engine_routes as ER
from model import ops as O

# the plate from the UI team: a box 80 x 50 x 10, a boss O20 h15, a hole O6.6 at x = 30, a fillet r2
PLATE = {
    "model_id": "plate", "revision": 3,
    "dims": {"L": 80, "W": 50, "t": 10, "boss": 15, "Lhalf": "L/2"},
    "features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": "L", "h": "W"}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": "t"}},
        {"id": "f3", "op": "sketch", "deps": ["f2"],
         "params": {"plane": "XY", "profile": "circle", "w": 20, "cx": -20, "cy": 0, "offset": "t"}},
        {"id": "f4", "op": "extrude", "deps": ["f3"], "params": {"distance": "boss"}},
        {"id": "f5", "op": "hole", "deps": ["f4"], "params": {"x": 30, "y": 0, "d": 6.6, "through": True}},
        {"id": "f6", "op": "fillet", "deps": ["f5"], "params": {"radius": 2, "edges": "all"}},
    ],
}


def test_every_op_the_studio_sends_is_in_the_graph_contract():
    assert {"sketch", "extrude", "revolve", "hole", "fillet", "pattern", "chamfer", "mirror",
            "not_translated"} <= set(O.OPS)


def test_the_plate_validates_with_its_dimensions_resolved():
    status, body = ER.handle_post("/model/validate", {"model": PLATE})
    assert status == 200 and body["success"] is True and body["ok"] is True
    assert body["errors"] == [] and body["problems"] == []
    assert body["order"] == ["f1", "f2", "f3", "f4", "f5", "f6"]
    assert body["features"] == 6 and body["rejected"] == 0
    assert body["dims"]["Lhalf"] == 40.0                    # an expression over another dimension
    assert body["model"]["features"][0]["params"]["w"] == 80.0
    assert body["model"]["features"][1]["params"]["distance"] == 10.0
    assert body["model"]["features"][4]["params"]["d"] == 6.6


def test_the_resolved_plate_goes_straight_into_the_macro_route():
    # what the Studio does: validate, then send the resolved model to the macro
    status, checked = ER.handle_post("/model/validate", {"model": PLATE})
    assert status == 200 and checked["ok"] is True
    code, body = ER.handle_post("/exchange/solidworks-macro", {"model": checked["model"], "name": "plate"})
    assert code == 200 and [s["op"] for s in body["skipped"]] == []
    assert len(body["translated"]) == 6
    assert "CreateCornerRectangle(-0.04, -0.025, 0, 0.04, 0.025, 0)" in body["macro"]   # 80 x 50 mm


def test_a_hole_from_the_top_without_z_top_names_the_field():
    bad = {"model_id": "x", "features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "hole", "deps": ["f2"],
         "params": {"x": 0, "y": 0, "d": 6, "through": False, "depth": 4, "from": "top"}},
    ]}
    status, body = ER.handle_post("/model/validate", {"model": bad})
    assert status == 200 and body["ok"] is False
    assert body["problems"][0]["feature"] == "f3" and body["problems"][0]["op"] == "hole"
    assert "z_top" in body["problems"][0]["error"]


def test_a_half_counterbore_is_rejected_with_a_readable_reason():
    bad = {"features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "hole", "deps": ["f2"], "params": {"x": 0, "y": 0, "d": 5, "cbore_d": 9}},
    ]}
    status, body = ER.handle_post("/model/validate", {"model": bad})
    assert status == 200 and body["ok"] is False
    assert "cbore_depth" in body["errors"][0] and body["problems"][0]["feature"] == "f3"


def test_an_unknown_field_is_named_with_the_fields_that_are_allowed():
    bad = {"features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10, "speed": 3}},
    ]}
    status, body = ER.handle_post("/model/validate", {"model": bad})
    assert status == 200 and body["ok"] is False
    assert "unknown field(s) ['speed']" in body["errors"][0]


def test_a_dimension_that_cannot_be_resolved_is_an_error_not_a_crash():
    bad = {"model_id": "x", "dims": {"a": "a + 1"}, "features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": "a", "h": 20}},
    ]}
    status, body = ER.handle_post("/model/validate", {"model": bad})
    assert status == 200 and body["ok"] is False
    assert body["problems"][0]["feature"] == "f1" and "dimension cycle" in body["errors"][0]


def test_a_hole_on_an_axis_is_valid():
    # the plate's hole sits at y = 0 and the boss at x = -20: a coordinate is not a length, so 0 is legal
    ok = {"features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 80, "h": 50}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "hole", "deps": ["f2"], "params": {"x": 0, "y": 0, "d": 6, "through": True}},
    ]}
    status, body = ER.handle_post("/model/validate", {"model": ok})
    assert status == 200 and body["ok"] is True, body["errors"]


def test_the_exchange_only_ops_are_accepted_and_named_as_such():
    model = {"model_id": "m", "features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 40, "h": 20}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
        {"id": "f3", "op": "chamfer", "deps": ["f2"], "params": {"distance": 1}},
        {"id": "f4", "op": "mirror", "deps": ["f3"], "params": {"plane": "YZ"}},
        {"id": "f5", "op": "not_translated", "deps": ["f4"],
         "params": {"what": "tube", "why": "no macro mapping for this primitive"}},
    ]}
    status, body = ER.handle_post("/model/validate", {"model": model})
    assert status == 200 and body["ok"] is True and body["errors"] == []
    assert body["exchange_only"] == ["f3", "f4", "f5"]


def test_a_body_without_features_is_refused_with_a_problem_list():
    status, body = ER.handle_post("/model/validate", {"hello": 1})
    assert status == 400 and body["success"] is False and "features" in body["error"]
