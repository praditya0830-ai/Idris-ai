"""Engine endpoints: handler behaviour plus the main.py wiring that exposes them.

The handlers are tested as plain functions (no server), which is why they return (status, payload).
The wiring test parses main.py because that is exactly the bug this project already shipped once:
a handler existed but the route was not reachable.
"""
import base64
import json
import os
import re

import pytest

import engine_routes as ER
from geometry import kernel

MODEL = {"model_id": "bracket", "dims": {"L": 40, "W": 20, "t": 3},
         "features": [
             {"id": "f1", "op": "sketch", "deps": [], "params": {"plane": "XZ", "w": "L", "h": "W"}},
             {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": "t"}},
             {"id": "f3", "op": "hole", "deps": ["f2"], "params": {"d": 4, "x": "L/2", "y": "W/2"}}]}

ASM = {"model_id": "demo", "tree": {"id": "root", "kind": "assembly", "children": [
    {"id": "housing", "kind": "part", "part": "housing", "triangles": 12000, "mass_kg": 6.4},
    {"id": "sub-0", "kind": "subassembly", "children": [
        {"id": f"b-{i}", "kind": "instance", "part": "bolt", "triangles": 480, "mass_kg": 0.012}
        for i in range(20)]}]}}


def _main_source():
    path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    return open(path, encoding="utf-8").read()


# ------------------------------------------------------------------- wiring
def test_main_py_exposes_every_engine_route():
    src = _main_source()
    post = src[src.index("def do_POST"):]
    allow = post[post.index("POST_ROUTES = ("):post.index(")", post.index("POST_ROUTES = ("))]
    for path in ("/model/validate", "/assembly/manifest", "/exchange/step"):
        assert f'"{path}"' in allow, path
    assert 'startswith("/geometry/")' in src
    assert 'startswith(("/model/", "/assembly/", "/exchange/"))' in post
    get_branch = src[src.index("def do_GET"):src.index("def do_POST")]
    assert "handle_get" in get_branch and "handle_post" in post


def test_route_lists_match_the_handlers():
    assert ER.handle_get("/geometry/checks") is not None
    assert ER.handle_get("/geometry/kernel") is not None
    assert ER.handle_get("/geometry/selftest") is not None
    assert ER.handle_get("/geometry/nope") is None
    assert ER.handle_post("/model/validate", {}) is not None
    assert ER.handle_post("/assembly/manifest", {}) is not None
    assert ER.handle_post("/exchange/step", {}) is not None
    assert ER.handle_post("/model/nope", {}) is None


# ---------------------------------------------------------------------- GET
def test_geometry_checks_lists_the_contract():
    status, body = ER.handle_get("/geometry/checks")
    assert status == 200 and body["success"]
    ids = {c["id"] for c in body["checks"]}
    assert {"mesh.closed", "mesh.manifold", "solid.watertight", "solid.volume_positive"} <= ids
    assert all(c["means"] for c in body["checks"])


def test_kernel_endpoint_reports_the_truth_of_this_machine():
    status, body = ER.handle_get("/geometry/kernel")
    assert status == 200
    assert body["available"] is kernel.available()
    assert ("no OCP" in body["detail"]) == (not body["available"])


def test_selftest_is_honest_and_says_what_it_exercised():
    status, body = ER.handle_get("/geometry/selftest")
    assert status == 200 and body["success"]
    checks = {c["id"]: c for c in body["checks"]}
    for cid in ("mesh.closed_cube", "mesh.rejects_open_surface", "fingerprint.deterministic",
                "model.partial_regeneration", "model.quarantine_and_block",
                "assembly.manifest_compact", "assembly.mass_matches_bom"):
        assert checks[cid]["ok"] is True, (cid, checks[cid])
    assert body["kernel"] is kernel.available()
    if not kernel.available():
        assert checks["kernel.available"]["ok"] is False
        assert "NOT exercised" in checks["kernel.available"]["detail"]
    assert body["pass"] == all(c["ok"] for c in body["checks"])


# --------------------------------------------------------------------- POST
def test_model_validate_accepts_a_good_model_and_resolves_expressions():
    status, body = ER.handle_post("/model/validate", {"model": MODEL})
    assert status == 200 and body["ok"] is True and body["errors"] == []
    assert body["order"] == ["f1", "f2", "f3"]
    assert body["model"]["features"][2]["params"]["x"] == 20.0


def test_model_validate_names_each_rejected_feature():
    bad = {"model_id": "x", "dims": {"L": 40}, "features": [
        {"id": "f1", "op": "teleport", "deps": [], "params": {}},
        {"id": "f2", "op": "hole", "deps": ["nope"], "params": {"d": 1, "x": 1, "y": 1}}]}
    status, body = ER.handle_post("/model/validate", bad)
    assert status == 200 and body["ok"] is False
    joined = " | ".join(body["errors"])
    assert "unknown op" in joined and "unknown dependency" in joined


def test_model_validate_rejects_a_body_that_is_not_a_model():
    status, body = ER.handle_post("/model/validate", {"hello": 1})
    assert status == 400 and not body["success"]


def test_assembly_manifest_is_structure_only():
    status, body = ER.handle_post("/assembly/manifest", {"assembly": ASM, "distance_mm": 100.0})
    assert status == 200 and body["success"]
    assert body["instances"] == 21 and body["unique_geometry"] == 2
    assert body["parts"]["bolt"]["instances"] == 20
    assert body["lod"]["distance_mm"] == 100.0
    blob = json.dumps(body)
    assert "positions" not in blob and "indices" not in blob


def test_assembly_manifest_rejects_a_broken_tree():
    status, body = ER.handle_post("/assembly/manifest", {"tree": {"kind": "galaxy", "id": "x"}})
    assert status == 400 and not body["success"]
    status, body = ER.handle_post("/assembly/manifest", {"tree": {"id": "a", "kind": "assembly",
                                                                  "children": [{"id": "b", "part": "p"},
                                                                               {"id": "b", "part": "p"}]}})
    assert status == 400 and "duplicate" in body["error"]


def test_exchange_step_needs_a_payload():
    status, body = ER.handle_post("/exchange/step", {})
    assert status == 400 and "step_base64" in body["error"]


def test_exchange_step_refuses_without_a_kernel_instead_of_guessing():
    if kernel.available():
        pytest.skip("OCP installed; the no-kernel refusal cannot be exercised")
    status, body = ER.handle_post("/exchange/step", {
        "filename": "part.step", "step_base64": base64.b64encode(b"ISO-10303-21;\n").decode()})
    assert status == 503
    assert body["error"] == "kernel_unavailable"
    assert "no substitute answer" in body["detail"]
