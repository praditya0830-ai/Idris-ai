"""Fase 1: the feature graph rebuilds only what changed, and never builds on a broken parent."""
import pytest

from model import expr as E
from model.graph import Feature, Graph, GraphError, regenerate

DIMS = {"L": 40, "W": 20, "t": 3, "h": 4.5, "pitch": 8}


def build():
    g = Graph("bracket-1", dims=DIMS)
    g.add("f1", "sketch", [], {"plane": "XZ", "profile": "rectangle", "w": "L", "h": "W"})
    g.add("f2", "extrude", ["f1"], {"distance": "t"})
    g.add("f3", "hole", ["f2"], {"d": "h", "x": "L*0.65", "y": "W/2", "through": True})
    g.add("f4", "pattern", ["f2"], {"mode": "linear", "count": 3, "dx": "pitch"})
    g.add("f5", "fillet", ["f3"], {"radius": 1.0, "edges": "vertical"})
    return g


class Counter:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def __call__(self, feature, dep_values):
        self.calls.append(feature.id)
        if feature.id in self.fail:
            raise RuntimeError(f"kernel refused {feature.id}")
        return {"id": feature.id, "deps": dep_values}


def test_expressions_are_resolved_to_numbers_on_insert():
    g = build()
    assert g.get("f1").params["w"] == 40.0
    assert g.get("f3").params["x"] == 26.0
    assert g.get("f3").params["exprs"]["x"] == "L*0.65"
    assert set(g.get("f3").params["exprs"]) == {"d", "x", "y"}   # every expression is preserved


def test_expression_language_is_arithmetic_only():
    assert E.evaluate("L*0.65", {"L": 40}) == 26.0
    assert E.evaluate("(W/2 - 1.5)**2", {"W": 20}) == 72.25
    for bad in ("__import__('os')", "L.real", "[1][0]", "L if L else 1", "open('x')"):
        with pytest.raises(E.ExprError):
            E.evaluate(bad, {"L": 1})
    with pytest.raises(E.ExprError):
        E.evaluate("Q*2", {"L": 1})                    # undefined dimension
    with pytest.raises(E.ExprError):
        E.evaluate("L/0", {"L": 1})
    with pytest.raises(E.ExprError):
        E.evaluate("L", {"L": "M", "M": "L"})         # cycle through dimensions


def test_unknown_op_and_unknown_param_are_rejected():
    g = Graph("x", dims=DIMS)
    with pytest.raises(GraphError, match="unknown op"):
        g.add("a", "teleport", [], {})
    with pytest.raises(GraphError, match="unknown field"):
        g.add("a", "extrude", [], {"distance": 3, "colour": "red"})


def test_missing_dependency_and_cycle_are_rejected():
    g = build()
    with pytest.raises(GraphError, match="unknown dependency"):
        g.add("z", "fillet", ["nope"], {"radius": 1})
    with pytest.raises(GraphError, match="cycle"):
        g.update("f1", deps=["f5"])                    # f1 -> f5 -> f3 -> f2 -> f1


def test_partial_regeneration_touches_only_the_cone_of_influence():
    g = build()
    c = Counter()
    first = regenerate(g, evaluate=c, cache={})
    assert first["status"] == "ok"
    assert first["regenerated"] == ["f1", "f2", "f3", "f4", "f5"]

    cache = {fid: f"cached-{fid}" for fid in g.topo()}
    c2 = Counter()
    g.update("f3", {"d": 5.0})
    out = regenerate(g, changed=["f3"], evaluate=c2, cache=cache)
    assert out["regenerated"] == ["f3", "f5"]
    assert out["reused"] == ["f1", "f2", "f4"]
    assert c2.calls == ["f3", "f5"]


def test_changing_a_shared_parent_rebuilds_both_branches():
    g = build()
    cache = {fid: fid for fid in g.topo()}
    c = Counter()
    g.update("f2", {"distance": 4.0})
    out = regenerate(g, changed="f2", evaluate=c, cache=cache)
    assert out["regenerated"] == ["f2", "f3", "f4", "f5"]
    assert out["reused"] == ["f1"]


def test_a_failed_feature_is_quarantined_and_its_children_are_blocked():
    g = build()
    c = Counter(fail={"f2"})
    out = regenerate(g, evaluate=c, cache={})
    assert out["status"] == "quarantined"
    assert [q["id"] for q in out["quarantined"]] == ["f2"]
    assert "kernel refused f2" in out["quarantined"][0]["reason"]
    assert [b["id"] for b in out["blocked"]] == ["f3", "f4", "f5"]
    assert "f5" not in c.calls                     # nothing is built on the broken parent
    assert out["state"].get("f1") is not None


def test_removal_needs_cascade_and_reports_the_victims():
    g = build()
    with pytest.raises(GraphError, match="dependents"):
        g.remove("f2")
    assert g.remove("f2", cascade=True) == ["f2", "f3", "f4", "f5"]
    assert list(g._features) == ["f1"]


def test_round_trip_through_json_keeps_the_model():
    g = build()
    data = g.to_dict()
    again = Graph.from_dict(data)
    assert again.to_dict()["features"] == data["features"]
    assert again.topo() == g.topo()
    assert again.resolved_dims() == g.resolved_dims()


def test_insertion_order_is_stable_across_runs():
    a, b = build(), build()
    assert a.topo() == b.topo() == ["f1", "f2", "f3", "f4", "f5"]
    assert a.to_dict()["features"] == b.to_dict()["features"]
