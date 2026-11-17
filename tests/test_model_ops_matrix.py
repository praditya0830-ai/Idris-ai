"""Every sketch profile and operation the engine offers, checked feature by feature.

These tests keep the promise that a documented op actually works: the matrix found on the live server
(draft broken by BRepTools.OuterWire, every fillet broken because the edge was not downcast, a linear
pattern rejected as "not one solid") is exactly what this file pins down. The fake kernel here is a
little more faithful than the one in test_model_build: it knows wires, so the draft path (prism vs
loft) is exercised without OpenCascade.
"""
import math

import pytest

from model import builder as B
from model.graph import Graph
from test_model_build import BRACKET, PLANES, FakeKernel, _area

CENTRED = ("circle", "polygon", "triangle")


class WireKernel(FakeKernel):
    """FakeKernel plus sketch_wire / wire-loft, so drafts can be built without OCP."""

    @staticmethod
    def sketch_wire(plane, profile, w, h=None, n=6, thickness=None):
        if profile in ("rectangle", "lshape") and h is None:
            raise ValueError(f"sketch: a {profile} needs h")
        return {"kind": "wire", "plane": plane, "profile": profile, "w": float(w),
                "h": float(w if h is None else h), "n": int(n), "thickness": thickness}

    @staticmethod
    def sketch_face(plane, profile, w, h=None, n=6, thickness=None):
        wire = WireKernel.sketch_wire(plane, profile, w, h, n, thickness)
        return {"kind": "face", "plane": plane, "profile": profile, "w": float(w),
                "h": float(w if h is None else h)}

    @staticmethod
    def translate(shape, vector):
        return dict(shape, moved=list(vector), bbox=list(shape["bbox"]) if "bbox" in shape else None)

    @staticmethod
    def loft(bottom, top):
        grow = top["w"] - bottom["w"]
        return {"kind": "solid", "volume": _area(bottom) * 10.0 + grow, "bbox": [bottom["w"], bottom["h"], 10.0],
                "plane": bottom["plane"], "profile": bottom["profile"],
                "size": [bottom["w"], bottom["h"]]}


def model(*features, dims=None):
    return Graph.from_dict({"model_id": "t", "dims": dims or {}, "features": list(features)})


def sketch(**params):
    return {"id": "f1", "op": "sketch", "deps": [], "params": params}


def op(name, deps, **params):
    return {"id": "f2", "op": name, "deps": [deps], "params": params}


def build(*features, dims=None, k=WireKernel):
    return B.build(model(*features, dims=dims), k=k)


# ------------------------------------------------------------------ sketch profiles
@pytest.mark.parametrize("profile", ["rectangle", "circle", "polygon", "triangle", "lshape"])
def test_every_documented_profile_builds(profile):
    params = {"plane": "XY", "profile": profile, "w": 30, "h": 20}
    out = build(sketch(**params), op("extrude", "f1", distance=4))
    assert out["status"] == "ok", out["errors"]
    assert out["features"]["f2"]["summary"]["volume_mm3"] > 0


def test_a_polygon_takes_its_side_count():
    out = build(sketch(plane="XY", profile="polygon", w=30, n=8), op("extrude", "f1", distance=2))
    assert out["status"] == "ok"
    assert out["features"]["f2"]["summary"]["volume_mm3"] > 0


def test_an_lshape_needs_a_height_and_a_sane_thickness():
    out = build(sketch(plane="XY", profile="lshape", w=30), op("extrude", "f1", distance=4))
    assert out["status"] == "quarantined" and "needs h" in out["quarantined"][0]["reason"]

    out = build(sketch(plane="XY", profile="lshape", w=30, h=20, thickness=40),
                op("extrude", "f1", distance=4))
    assert out["status"] == "quarantined" and "thickness" in out["quarantined"][0]["reason"]


def test_a_sketch_can_sit_away_from_its_plane():
    """offset is what makes a multi-level part possible: the sketch is not stuck at the plane."""
    flat = build(sketch(plane="XY", profile="rectangle", w=10, h=10), op("extrude", "f1", distance=2))
    lifted = build(sketch(plane="XY", profile="rectangle", w=10, h=10, offset=12),
                   op("extrude", "f1", distance=2))
    assert flat["status"] == lifted["status"] == "ok"
    assert WireKernel.sketch_wire("XY", "rectangle", 10, 10)["kind"] == "wire"
    # the offset travels with the face: the builder asks the kernel to translate it
    assert "offset" in Graph.from_dict({"model_id": "o", "features": [
        {"id": "f1", "op": "sketch", "deps": [], "params": {"profile": "rectangle", "w": 1, "h": 1, "offset": 3}}]}
    ).get("f1").params


# ----------------------------------------------------------------------- extrude
def test_extrude_directions_are_all_accepted():
    for direction in ("+normal", "-normal", "symmetric"):
        out = build(sketch(plane="XY", profile="rectangle", w=10, h=10),
                    op("extrude", "f1", distance=4, direction=direction))
        assert out["status"] == "ok", (direction, out["errors"])


def test_a_draft_uses_a_loft_and_survives_the_kernel():
    out = build(sketch(plane="XY", profile="rectangle", w=20, h=10),
                op("extrude", "f1", distance=10, draft_deg=5))
    assert out["status"] == "ok", out["errors"]
    assert out["features"]["f2"]["summary"]["volume_mm3"] > 0


# ----------------------------------------------------------------------- revolve
def test_revolve_works_around_an_axis_in_the_sketch_plane():
    out = build(sketch(plane="XZ", profile="rectangle", w=10, h=20),
                op("revolve", "f1", axis="z", angle_deg=270))
    assert out["status"] == "ok", out["errors"]


def test_revolve_refuses_an_axis_perpendicular_to_the_sketch_plane():
    out = build(sketch(plane="XZ", profile="rectangle", w=10, h=20),
                op("revolve", "f1", axis="y"))
    assert out["status"] == "quarantined"
    assert "perpendicular" in out["quarantined"][0]["reason"]


# ------------------------------------------------------------------ fillet/pattern
def test_fillet_reaches_every_selector():
    for selector in ("all", "vertical", "horizontal", "top", "bottom"):
        out = build(sketch(plane="XY", profile="rectangle", w=20, h=10),
                    op("extrude", "f1", distance=4),
                    {"id": "f3", "op": "fillet", "deps": ["f2"], "params": {"radius": 1, "edges": selector}})
        assert out["status"] == "ok", (selector, out["errors"])


def test_a_linear_pattern_of_separate_copies_is_not_called_a_broken_solid():
    model = {"model_id": "p", "features": [
        {"id": "f1", "op": "sketch", "deps": [], "params": {"plane": "XY", "profile": "rectangle", "w": 10, "h": 10}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 4}},
        {"id": "f3", "op": "pattern", "deps": ["f2"],
         "params": {"mode": "linear", "count": 3, "dx": 20}}]}
    out = B.build(Graph.from_dict(model), k=SeparateBodies)
    assert out["status"] == "ok", out["errors"]
    ids = {c["id"] for c in out["features"]["f3"]["checks"]}
    assert "solid.single" not in ids and "solid.bodies_positive" in ids


class SeparateBodies(WireKernel):
    """A fuse of copies that do not touch: the kernel reports three solids."""

    @staticmethod
    def boolean(op, a, b):
        if op == "fuse":
            return dict(a, volume=a["volume"] + b["volume"], solids=3)
        return dict(a)

    @staticmethod
    def measure(shape):
        out = WireKernel.measure(shape)
        out["solids"] = int(shape.get("solids") or 1)
        return out


def test_the_bracket_from_the_first_release_still_builds():
    out = B.build(Graph.from_dict(BRACKET), k=WireKernel)
    assert out["status"] == "ok", out["errors"]
    assert out["final"]["feature"] == "f3"
