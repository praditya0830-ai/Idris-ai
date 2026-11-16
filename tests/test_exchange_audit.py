"""Fase 3 + 4: exchange honesty, provenance records, immutable store, proposal validation."""
import json

import pytest

import ai
import assembly
import audit
import exchange
import geometry
import model
from ai import propose
from audit import run as R
from audit.store import Store, StoreError
from exchange import roundtrip, sidecar, step
from geometry import kernel

SUMMARY = {"volume_mm3": 4800.0, "area_mm2": 1760.0, "bbox_mm": [40.0, 20.0, 3.0],
           "solids": 1, "faces": 6, "edges": 12, "vertices": 8, "watertight": True,
           "degenerate_faces": 0}


def test_new_packages_are_the_ones_in_this_repo():
    for mod in (ai, assembly, audit, exchange, model):
        roots = list(getattr(mod, "__path__", []))
        assert roots and any("Idris-ai" in p for p in roots), getattr(mod, "__name__", mod)


# ---------------------------------------------------------------- exchange
def test_face_counts_become_candidates_with_confidence_and_basis():
    rec = step.candidates_from_counts({"plane": 6, "cylinder": 4, "torus": 2})
    assert rec["recognised"] is True and rec["dumb_solid"] is False
    kinds = {c["feature"]: c for c in rec["candidates"]}
    assert kinds["hole"]["count"] == 4 and 0.4 <= kinds["hole"]["confidence"] <= 0.9
    assert kinds["fillet"]["count"] == 2
    assert "confirmation" in rec["reason"]
    for c in rec["candidates"]:
        assert c["basis"]                                   # a claim without a basis is not allowed


def test_unrecognised_shape_is_reported_as_a_dumb_solid():
    rec = step.candidates_from_counts({})
    assert rec["recognised"] is False and rec["dumb_solid"] is True
    assert "dumb solid" in rec["reason"]
    assert step.candidates_from_counts({"plane": 6})["candidates"][0]["feature"] == "prismatic"


def test_import_rejects_unsupported_extension_and_missing_files():
    with pytest.raises(step.ImportFailed, match="unsupported format"):
        step.import_shape("part.obj")
    with pytest.raises(step.ImportFailed, match="not found"):
        step.import_shape("/tmp/definitely-missing-idris.step")


def test_import_says_kernel_is_missing_instead_of_failing_obscurely(tmp_path):
    if kernel.available():
        pytest.skip("OCP installed; the missing-kernel path cannot be exercised")
    f = tmp_path / "part.step"
    f.write_text("ISO-10303-21;\n")
    with pytest.raises(step.ImportFailed, match="kernel"):
        step.import_shape(str(f))


def test_roundtrip_gate_tolerates_noise_and_fails_real_loss():
    same = dict(SUMMARY)
    near = {**SUMMARY, "volume_mm3": SUMMARY["volume_mm3"] * 1.00005}
    assert roundtrip.gate(same, near)["status"] == "ok"
    lost = roundtrip.gate(same, {**SUMMARY, "volume_mm3": 4600.0, "faces": 5})
    assert lost["status"] == "failed"
    assert {d["field"] for d in lost["diffs"]} == {"volume_mm3", "faces"}


def test_sidecar_carries_intent_and_says_when_it_does_not(tmp_path):
    run = R.make(inputs={"wall": 4}, solver_version="0.35.0", geometry_summary=SUMMARY,
                 evidence={"level": 3, "method": "3D FEM"},
                 sources=[{"k": "T", "url": "https://example.org/x", "quote": "q"}])
    man = sidecar.build("bracket", features=[{"id": "f1", "op": "sketch"}],
                        dims={"wall": 4}, run=run)
    assert sidecar.has_intent(man) is True
    assert man["sources"][0]["url"] == "https://example.org/x"
    path = sidecar.write(str(tmp_path / "bracket.step"), man)
    assert path.endswith("bracket.idris.json")
    assert sidecar.read(str(tmp_path / "bracket.step"))["dims"] == {"wall": 4}
    assert sidecar.has_intent(sidecar.build("x")) is False          # no features, no provenance


# ------------------------------------------------------------------- audit
def test_run_ids_are_deterministic_and_geometry_sensitive():
    kw = dict(inputs={"wall": 4, "module": 2}, solver_version="0.35.0", geometry_summary=SUMMARY)
    a = R.make(**kw)
    b = R.make(**{**kw, "created_at": "2026-11-16T00:00:00Z"})
    assert a["run_id"] == b["run_id"]
    c = R.make(**{**kw, "geometry_summary": {**SUMMARY, "volume_mm3": 5000.0}})
    assert c["run_id"] != a["run_id"]
    assert a["status"] == "ok" and a["geometry_hash"]


def test_record_validation_names_what_is_wrong():
    rec = R.make(inputs={}, solver_version="0.35.0")
    assert R.validate(rec) == []
    assert any("status" in p for p in R.validate({**rec, "status": "great"}))
    assert any("solver_version" in p for p in R.validate({**rec, "solver_version": ""}))
    assert any("url" in p for p in R.validate({**rec, "sources": [{"k": "T"}]}))
    with pytest.raises(ValueError):
        R.make(inputs={}, solver_version="0.35.0", status="great")


def test_store_is_immutable_and_reports_identical_resaves(tmp_path):
    store = Store(str(tmp_path / "runs"))
    rec = R.make(inputs={"a": 1}, solver_version="0.35.0", geometry_summary=SUMMARY)
    first = store.save(rec)
    assert first["existing"] is False
    assert store.save(rec)["existing"] is True
    tampered = {**rec, "status": "repaired"}
    with pytest.raises(StoreError, match="immutable"):
        store.save(tampered)
    assert store.get(rec["run_id"])["status"] == "ok"
    assert store.ids() == [rec["run_id"]]
    assert len(store.list(status="ok")) == 1 and store.list(status="failed") == []


def test_report_escapes_untrusted_text_and_states_evidence():
    rec = R.make(inputs={}, solver_version="0.35.0", geometry_summary=SUMMARY,
                 evidence={"level": 3, "level_name": "numerical solver result",
                           "not_yet": ["no experimental validation (level 5)"]},
                 checks=[{"id": "solid.watertight", "ok": True, "detail": ""}],
                 sources=[{"k": "T", "url": "https://x", "quote": "<script>alert(1)</script>"}])
    html = audit.report.page([rec])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "level 3" in html and "no experimental validation" in html


# ---------------------------------------------------------------- proposals
def test_proposal_accepts_good_features_and_rejects_each_bad_one_by_name():
    proposal = {"features": [
        {"id": "f1", "op": "sketch", "params": {"plane": "XY", "w": "L", "h": "L/2"}},
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 3}},
        {"id": "f3", "op": "teleport", "deps": ["f2"], "params": {}},
        {"id": "f4", "op": "hole", "deps": ["f2"], "params": {"d": 4, "x": 1, "y": 2, "colour": "red"}},
        {"id": "f5", "op": "hole", "deps": ["f2"], "params": {"d": "unknown_dim", "x": 1, "y": 2}},
        {"op": "fillet", "params": {"radius": 1}},
    ]}
    out = propose.validate_proposal(proposal, dims={"L": 40})
    assert [a["id"] for a in out["accepted"]] == ["f1", "f2"]
    assert out["ok"] is False
    reasons = {r.get("id"): r["reason"] for r in out["rejected"]}
    assert "unknown op" in reasons["f3"]
    assert "unknown field" in reasons["f4"]
    assert "undefined dimension" in reasons["f5"]
    assert None in reasons                                  # the feature without an id
    assert out["graph"].get("f1").params["h"] == 20.0


def test_apply_proposal_only_adds_what_survives_validation():
    g = model.Graph("m", dims={"L": 40})
    g.add("f1", "sketch", [], {"w": 10, "h": 5})
    out = propose.apply_proposal(g, {"features": [
        {"id": "f1", "op": "extrude", "deps": [], "params": {"distance": 1}},   # duplicate id
        {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 2}},
        {"id": "f3", "op": "fillet", "deps": ["missing"], "params": {"radius": 1}},
    ]})
    assert out["applied"] == ["f2"]
    assert {r.get("id") for r in out["rejected"]} == {"f1", "f3"}
    assert "f2" in g and len(g) == 2
