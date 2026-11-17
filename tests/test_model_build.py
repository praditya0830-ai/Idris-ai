"""Kernel-backed builder and the endpoints it powers, tested with a stand-in kernel.

The stand-in is not a mock of the maths: it returns real arithmetic for the volume a rectangle or a
circle sweeps, so hole placement and the invariant gate are exercised for real. OCP itself is checked
on the live server (GET /geometry/selftest, POST /model/build), because a sandbox without OCP must not
pretend it built a solid.
"""
import base64
import copy
import json
import math
import os

import pytest

import engine_routes as ER
from model import builder as B
from model.graph import Graph

PLANES = {"XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
          "XZ": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
          "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))}


def _area(face):
    if face["profile"] == "circle":
        return math.pi * (face["w"] / 2.0) ** 2
    if face["profile"] == "polygon":
        return 3.0 * math.sqrt(3.0) / 2.0 * (face["w"] / 2.0) ** 2
    return face["w"] * face["h"]


class FakeKernel:
    """Enough of geometry.kernel to exercise model/builder.py without OpenCascade."""

    available = staticmethod(lambda: True)
    version = staticmethod(lambda: "fake-1.0")
    require = staticmethod(lambda: None)
    step_reads = 0
    calls = []

    @staticmethod
    def reset():
        FakeKernel.step_reads = 0
        FakeKernel.calls = []

    @staticmethod
    def plane_axes(plane):
        try:
            return PLANES[plane]
        except KeyError:
            raise ValueError(f"unknown sketch plane {plane!r}") from None

    @staticmethod
    def sketch_face(plane, profile, w, h=None):
        FakeKernel.calls.append(("sketch_face", plane, profile, w, h))
        return {"kind": "face", "plane": plane, "profile": profile, "w": float(w),
                "h": float(w if h is None else h)}

    @staticmethod
    def prism(face, vector):
        FakeKernel.calls.append(("prism", list(vector)))
        n = PLANES[face["plane"]][2]
        length = abs(sum(a * b for a, b in zip(vector, n)))
        if length < 1e-12:
            raise ValueError("extrude: the distance resolves to zero")
        e1, e2, _ = PLANES[face["plane"]]
        ext = [abs(e1[i]) * face["w"] + abs(e2[i]) * face["h"] + abs(n[i]) * length for i in range(3)]
        return {"kind": "solid", "volume": _area(face) * length, "bbox": ext,
                "plane": face["plane"], "profile": face["profile"],
                "size": [face["w"], face["h"]]}

    @staticmethod
    def translate(shape, vector):
        return dict(shape, moved=list(vector), bbox=list(shape["bbox"]))

    @staticmethod
    def rotate(shape, axis, angle_deg, origin=(0.0, 0.0, 0.0)):
        return dict(shape, rotated=[axis, angle_deg])

    @staticmethod
    def loft(bottom, top):
        return {"kind": "solid", "volume": _area(bottom) * 100.0, "bbox": list(bottom["bbox"]),
                "plane": bottom["plane"], "profile": bottom["profile"], "size": list(bottom["size"])}

    @staticmethod
    def revolve(face, axis, angle_deg):
        return {"kind": "solid", "volume": _area(face) * 300.0 * float(angle_deg) / 360.0,
                "bbox": [face["w"], face["w"], face["w"]], "plane": face["plane"],
                "profile": face["profile"], "size": [face["w"], face["h"]]}

    @staticmethod
    def cylinder_along(radius, height, origin, direction):
        return {"kind": "cylinder", "volume": math.pi * float(radius) ** 2 * float(height),
                "origin": list(origin), "direction": list(direction),
                "radius": float(radius), "height": float(height)}

    @staticmethod
    def boolean(op, a, b):
        if op == "cut":
            inside = all(-2.0 <= b["origin"][i] <= a["bbox"][i] + 2.0 for i in range(3))
            return dict(a, volume=a["volume"] - (b["volume"] if inside else 0.0))
        if op == "fuse":
            return dict(a, volume=a["volume"] + b["volume"])
        return a

    @staticmethod
    def fillet(shape, radius, edges=None):
        if float(radius) > 1e6:
            raise RuntimeError("fillet failed in the kernel (radius too large for the geometry)")
        return dict(shape)

    @staticmethod
    def edge_filter(shape, selector="all"):
        if selector in (None, "all"):
            return None
        if selector == "none":
            return []
        return [0, 1]

    @staticmethod
    def bbox_range(shape):
        return [0.0, 0.0, 0.0], list(shape["bbox"])

    @staticmethod
    def measure(shape):
        if shape.get("kind") != "solid":
            raise RuntimeError("measure: not a solid")
        return {"volume_mm3": float(shape["volume"]), "area_mm2": 0.0,
                "bbox_mm": [float(c) for c in shape["bbox"]],
                "solids": 1, "faces": 6, "edges": 12, "vertices": 8,
                "degenerate_faces": 0, "watertight": True}

    @staticmethod
    def tessellate(shape, deflection=0.1, angle=0.5):
        positions, indices = ER._box_mesh([float(c) for c in shape["bbox"]])
        return positions, indices

    @staticmethod
    def read_step(path):
        FakeKernel.step_reads += 1
        with open(path, "rb") as fh:
            data = fh.read()
        return {"kind": "solid", "volume": 1000.0, "bbox": [10.0, 10.0, 10.0], "bytes": len(data)}


class NoKernel:
    available = staticmethod(lambda: False)
    version = staticmethod(lambda: None)
    require = staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no kernel")))


BRACKET = {"model_id": "bracket", "dims": {"L": 40, "W": 20, "t": 3},
           "features": [
               {"id": "f1", "op": "sketch", "deps": [], "params": {"plane": "XY", "w": "L", "h": "W"}},
               {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": "t"}},
               {"id": "f3", "op": "hole", "deps": ["f2"], "params": {"d": 4, "x": 10, "y": 10}}]}

ASM = {"model_id": "demo", "tree": {"id": "root", "kind": "assembly", "children": [
    {"id": "plate", "part": "plate", "triangles": 100, "mass_kg": 1.0},
    {"id": "b-1", "part": "bolt", "triangles": 480, "mass_kg": 0.012},
    {"id": "b-2", "part": "bolt", "triangles": 480, "mass_kg": 0.012},
    {"id": "b-3", "part": "bolt", "triangles": 480, "mass_kg": 0.012}]}}

BOLT = {"filename": "bolt.step", "step_base64": base64.b64encode(b"ISO-10303-21;\nfake bolt\n").decode()}


def _graph(model=None):
    return Graph.from_dict(copy.deepcopy(model or BRACKET))


# ------------------------------------------------------------------- the builder
def test_a_bracket_is_built_with_real_volume_and_passes_the_invariants():
    out = B.build(_graph(), k=FakeKernel)
    assert out["status"] == "ok"
    assert out["regenerated"] == ["f1", "f2", "f3"]
    slab = 40 * 20 * 3
    drilled = math.pi * 2.0 ** 2 * 5.0          # the drill spans the 3 mm plate plus 1 mm each side
    summary = out["features"]["f3"]["summary"]
    assert summary["volume_mm3"] == pytest.approx(slab - drilled)
    assert summary["bbox_mm"] == pytest.approx([40.0, 20.0, 3.0])
    assert all(c["ok"] for c in out["features"]["f3"]["checks"])
    assert out["final"]["feature"] == "f3"
    assert out["features"]["f3"]["fingerprint"]


def test_the_hole_is_drilled_along_the_sketch_normal_not_world_z():
    model = copy.deepcopy(BRACKET)
    model["features"][0]["params"]["plane"] = "XZ"
    FakeKernel.reset()
    B.build(Graph.from_dict(model), k=FakeKernel)
    vectors = [c[1] for c in FakeKernel.calls if c[0] == "prism"]
    assert vectors == [[0.0, -3.0, 0.0]]      # XZ normal is -Y: the extrusion follows the plane


def test_a_hole_that_misses_the_part_is_quarantined_and_its_dependents_are_blocked():
    model = copy.deepcopy(BRACKET)
    model["features"][2]["params"].update({"x": 500, "y": 500})
    model["features"].append({"id": "f4", "op": "fillet", "deps": ["f3"], "params": {"radius": 1}})
    out = B.build(Graph.from_dict(model), k=FakeKernel)
    assert out["status"] == "quarantined"
    assert out["quarantined"][0]["id"] == "f3"
    assert "removed no material" in out["quarantined"][0]["reason"]
    assert out["blocked"] == [{"id": "f4", "reason": "blocked by f3"}]


def test_a_fillet_the_kernel_rejects_is_quarantined_with_the_kernel_reason():
    model = copy.deepcopy(BRACKET)
    model["features"].append({"id": "f4", "op": "fillet", "deps": ["f3"],
                              "params": {"radius": 2e6, "edges": "vertical"}})
    out = B.build(Graph.from_dict(model), k=FakeKernel)
    assert out["status"] == "quarantined"
    assert "radius too large" in out["quarantined"][0]["reason"]


def test_a_fillet_selector_that_matches_no_edge_is_an_error_not_a_silent_no_op():
    model = copy.deepcopy(BRACKET)
    model["features"].append({"id": "f4", "op": "fillet", "deps": ["f3"],
                              "params": {"radius": 1, "edges": "top"}})
    fake = type("K", (FakeKernel,), {"edge_filter": staticmethod(lambda shape, selector="all": [])})
    out = B.build(Graph.from_dict(model), k=fake)
    assert out["status"] == "quarantined" and "no edge" in out["quarantined"][0]["reason"]


def test_only_the_cone_of_influence_is_rebuilt_when_a_parameter_changes():
    g = _graph()
    state = {}
    first = B.build(g, k=FakeKernel, collect=state)
    g.update("f2", {"distance": 5.0})
    second = B.build(g, k=FakeKernel, cache=dict(state), keys=first["keys"])
    assert second["reused"] == ["f1"]
    assert second["regenerated"] == ["f2", "f3"]
    assert second["changed"] == ["f2"]          # the direct change; dependents follow from the graph
    assert second["features"]["f3"]["summary"]["volume_mm3"] == pytest.approx(40 * 20 * 5 - math.pi * 4 * 7)


def test_an_opaque_node_is_refused_instead_of_executed():
    model = {"model_id": "x", "features": [{"id": "c1", "op": "opaque", "deps": [],
                                            "params": {"code": "print('hello')"}}]}
    out = B.build(Graph.from_dict(model), k=FakeKernel)
    assert out["status"] == "quarantined"
    assert "NOT executed" in out["quarantined"][0]["reason"]


def test_updating_a_value_replaces_the_old_expression():
    """A field passed to update() must win over the expression it was first written as."""
    g = Graph("m", dims={"t": 3})
    g.add("f1", "sketch", [], {"plane": "XY", "w": 10, "h": 10})
    g.add("f2", "extrude", ["f1"], {"distance": "t"})
    assert g.get("f2").params["distance"] == 3.0
    g.update("f2", {"distance": 5.0})
    assert g.get("f2").params["distance"] == 5.0
    assert "exprs" not in g.get("f2").params
    g.update("f2", {"distance": "t*2"})
    assert g.get("f2").params["distance"] == 6.0
    assert g.get("f2").params["exprs"] == {"distance": "t*2"}


# ----------------------------------------------------------------- the endpoints
def test_build_endpoint_refuses_honestly_without_a_kernel(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", NoKernel)
    status, body = ER.handle_post("/model/build", {"model": BRACKET})
    assert status == 503 and body["error"] == "kernel_unavailable"
    assert "nothing is simulated" in body["detail"]


def test_build_endpoint_returns_the_solids_and_a_run_record(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", FakeKernel)
    ER._BUILDS.clear()
    status, body = ER.handle_post("/model/build", {"model": BRACKET})
    assert status == 200 and body["success"] and body["status"] == "ok"
    assert "state" not in body                      # kernel shapes never reach JSON
    json.dumps(body)                                # and the rest is JSON-clean
    run = body["run"]
    assert run["kind"] == "model" and run["status"] == "ok" and run["geometry_hash"]
    assert run["checks"] and all(c["ok"] for c in run["checks"])
    assert run["kernel"] == {"name": "occt", "version": "fake-1.0"}


def test_build_endpoint_rebuilds_only_what_changed(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", FakeKernel)
    ER._BUILDS.clear()
    ER.handle_post("/model/build", {"model": BRACKET})
    edited = copy.deepcopy(BRACKET)
    edited["features"][1]["params"]["distance"] = 5
    status, body = ER.handle_post("/model/build", {"model": edited})
    assert status == 200
    assert body["reused"] == ["f1"]
    assert body["regenerated"] == ["f2", "f3"]


def test_node_endpoint_serves_a_closed_mesh_and_one_step_read(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", FakeKernel)
    FakeKernel.reset()
    status, body = ER.handle_post("/assembly/node", {"assembly": copy.deepcopy(ASM), "node": "b-2",
                                                     "parts": {"bolt": BOLT}})
    assert status == 200 and body["success"]
    assert body["id"] == "b-2" and body["part"] == "bolt"
    assert body["geometry_included"] is True
    assert len(body["positions"]) == 8 and body["triangles"] == 12
    assert all(c["ok"] for c in body["checks"])     # a closed box mesh passes mesh_checks
    assert FakeKernel.step_reads == 1
    assert body["run"]["kind"] == "assembly_node" and body["run"]["geometry_hash"]


def test_node_endpoint_sends_a_box_proxy_when_the_viewer_is_far_away(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", FakeKernel)
    status, body = ER.handle_post("/assembly/node", {"assembly": copy.deepcopy(ASM), "node": "b-1",
                                                     "distance_mm": 5000, "parts": {"bolt": BOLT}})
    assert status == 200 and body["level"] == "bbox"
    assert body["triangles"] == 12 and "bbox proxy" in body["note"]


def test_node_endpoint_names_the_missing_geometry(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", FakeKernel)
    status, body = ER.handle_post("/assembly/node", {"assembly": copy.deepcopy(ASM), "node": "b-1"})
    assert status == 400 and "no geometry for part" in body["error"]


def test_node_endpoint_rejects_an_unknown_node(monkeypatch):
    monkeypatch.setattr(ER, "KERNEL", FakeKernel)
    status, body = ER.handle_post("/assembly/node", {"assembly": copy.deepcopy(ASM), "node": "ghost",
                                                     "parts": {"bolt": BOLT}})
    assert status == 400 and "no node" in body["error"]


def test_manifest_carries_checks_and_a_run_record():
    status, body = ER.handle_post("/assembly/manifest", {"assembly": copy.deepcopy(ASM)})
    assert status == 200
    ids = {c["id"] for c in body["checks"]}
    assert {"assembly.mass_matches_bom", "assembly.under_triangle_budget"} <= ids
    assert body["run"]["kind"] == "assembly"
    blob = json.dumps(body)
    assert "positions" not in blob and "indices" not in blob


# --------------------------------------------------------------------- the wiring
def test_main_py_exposes_the_new_routes():
    path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    src = open(path, encoding="utf-8").read()
    post = src[src.index("def do_POST"):]
    allow = post[post.index("POST_ROUTES = ("):post.index(")", post.index("POST_ROUTES = ("))]
    for route in ("/model/build", "/assembly/node"):
        assert f'"{route}"' in allow, route
    assert set(ER.POST_PATHS) <= {p.strip().strip('"') for p in allow.replace("\n", " ").split(",")}


def test_handler_lists_match_the_routes():
    for path in ER.POST_PATHS:
        assert ER.handle_post(path, {}) is not None, path
    assert ER.handle_post("/model/nope", {}) is None
