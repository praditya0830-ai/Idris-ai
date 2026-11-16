"""Fase 2: a 500-instance assembly stays small, aggregates correctly and loads geometry once."""
import json

import pytest

from assembly import lod
from assembly.resolver import Resolver
from assembly.tree import Assembly, AssemblyError, node_from_dict

BOLT_MASS = 0.012
BOLT_TRIS = 480


def gearbox(subassemblies=25, bolts_each=20):
    """25 subassemblies x 20 bolts + one housing: 500 instanced bolts, 2 unique geometries."""
    children = [{"id": "housing", "kind": "part", "part": "housing", "triangles": 12000,
                 "mass_kg": 6.4}]
    for s in range(subassemblies):
        kids = [{"id": f"bolt-{s}-{b}", "kind": "instance", "part": "bolt",
                 "triangles": BOLT_TRIS, "mass_kg": BOLT_MASS,
                 "transform": {"translate": [s * 10.0, b * 8.0, 0.0]}} for b in range(bolts_each)]
        children.append({"id": f"sub-{s}", "kind": "subassembly", "children": kids})
    return {"model_id": "gearbox", "tree": {"id": "root", "kind": "assembly", "children": children}}


def test_instancing_shares_geometry_but_counts_every_part():
    a = Assembly.from_dict(gearbox())
    assert a.instance_count() == 501
    assert a.unique_geometry() == 2
    rows = {r["part"]: r for r in a.bom()}
    assert rows["bolt"]["count"] == 500
    assert rows["bolt"]["mass_total"] == pytest.approx(6.0)
    assert rows["housing"]["count"] == 1


def test_mass_matches_the_naive_sum():
    a = Assembly.from_dict(gearbox())
    naive = 6.4 + 500 * BOLT_MASS
    assert a.mass_kg() == pytest.approx(naive, rel=1e-9)


def test_manifest_is_compact_and_carries_no_geometry():
    a = Assembly.from_dict(gearbox())
    r = Resolver(a, load_part=lambda pid, level: {"positions": [0, 0, 0], "part": pid},
                 revision=7)
    m = r.manifest()
    blob = json.dumps(m, separators=(",", ":"))
    assert len(blob.encode()) < 2_000_000
    assert "positions" not in blob and "indices" not in blob
    assert m["parts"]["bolt"]["instances"] == 500
    assert m["instances"] == 501 and m["unique_geometry"] == 2
    assert r.manifest_bytes() == len(blob.encode())


def test_geometry_is_loaded_once_per_unique_part_then_cached():
    a = Assembly.from_dict(gearbox())
    r = Resolver(a, load_part=lambda pid, level: {"part": pid}, revision=1)
    first = r.node("bolt-0-0")
    again = r.node("bolt-24-19")
    assert first["part"] == "bolt" and again["part"] == "bolt"
    assert r.stats["loads"] == 1 and r.stats["cache_hits"] == 1
    assert again["cached"] is True
    r.node("housing")
    assert r.stats["loads"] == 2


def test_a_new_revision_invalidates_the_cache():
    a = Assembly.from_dict(gearbox(1, 2))
    r = Resolver(a, load_part=lambda pid, level: {"part": pid}, revision=1)
    r.node("housing")
    r.revision = 2
    assert r.node("housing")["cached"] is False
    assert r.stats["loads"] == 2


def test_lod_policy_is_predictable():
    assert lod.level_for(12000, distance_mm=None) == "full"
    assert lod.level_for(12000, distance_mm=5000.0) == "bbox"
    assert lod.level_for(12000, distance_mm=100.0, budget=1000, share=1.0) == "decimated"
    assert lod.level_for(0) == "full"
    assert lod.triangles_for("bbox", 12000) == 12
    assert lod.triangles_for("decimated", 12000) < 12000


def test_plan_states_whether_the_payload_is_over_budget():
    a = Assembly.from_dict(gearbox(2, 2))
    near = lod.plan(a, distance_mm=100.0, budget=100_000)
    far = lod.plan(a, distance_mm=5000.0, budget=100_000)
    assert near["over_budget"] is False
    assert far["triangles_total"] < near["triangles_total"]
    tight = lod.plan(a, distance_mm=100.0, budget=1000)
    assert tight["over_budget"] is True
    assert {n["level"] for n in far["nodes"]} == {"bbox"}


def test_bad_trees_are_rejected():
    with pytest.raises(AssemblyError, match="duplicate"):
        Assembly(node_from_dict({"id": "r", "kind": "assembly",
                                 "children": [{"id": "x", "part": "p"},
                                              {"id": "x", "part": "p"}]}))
    with pytest.raises(AssemblyError, match="unknown kind"):
        node_from_dict({"id": "r", "kind": "galaxy"})
    a = Assembly.from_dict(gearbox(1, 1))
    with pytest.raises(AssemblyError, match="no node"):
        a.find("nope")
    r = Resolver(a, load_part=lambda pid, level: pid)
    with pytest.raises(ValueError, match="not a geometry node"):
        r.node("sub-0")
