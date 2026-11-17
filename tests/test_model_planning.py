"""Tests for deterministic CAD planning and measured review findings."""
import pytest

from model.planning import plan_model


def test_non_fused_interface_means_assembly_and_keeps_contract():
    result = plan_model({
        "parts": [
            {"id": "frame", "role": "structure", "make": "cut tube"},
            {"id": "wheel", "role": "rotating", "make": "standard"},
        ],
        "interfaces": [{"a": "frame", "b": "wheel", "type": "bearing", "at": [0, 0, 0]}],
        "needs": ["wheel diameter"],
    })
    assert result["kind"] == "assembly"
    assert result["parts"][0]["id"] == "frame"
    assert result["interfaces"][0]["type"] == "bearing"
    assert result["needs"] == ["wheel diameter"]


def test_single_monolithic_part_is_part():
    result = plan_model({
        "parts": [{"id": "bracket", "role": "structure", "make": "machined plate"}],
        "interfaces": [],
        "needs": [],
    })
    assert result["kind"] == "part"
    assert result["review"] == []


def test_review_findings_use_measured_numbers_only():
    result = plan_model({
        "parts": [{
            "id": "tube",
            "measurements": {"length_mm": 620, "od_mm": 14, "wall_mm": 1.2, "minimum_inner_radius_mm": 0.2},
        }],
        "interfaces": [],
        "needs": [],
    })
    ids = [f["id"] for f in result["review"]]
    assert "MEMBER_SLENDER" in ids
    assert "WALL_THIN" in ids
    assert "CORNER_SHARP" in ids
    assert all("measured" in item for item in result["review"])


def test_joint_gap_is_reported_with_measured_number():
    result = plan_model({
        "parts": [{"id": "a"}, {"id": "b"}],
        "interfaces": [{"a": "a", "b": "b", "type": "weld", "gap_mm": 0.1}],
    })
    assert result["review"][0]["id"] == "JOINT_GAP"
    assert result["review"][0]["measured"]["gap_mm"] == 0.1


def test_invalid_contracts_fail_honestly():
    with pytest.raises(ValueError, match="duplicate part id"):
        plan_model({"parts": [{"id": "x"}, {"id": "x"}]})
    with pytest.raises(ValueError, match="connect two different"):
        plan_model({
            "parts": [{"id": "x"}],
            "interfaces": [{"a": "x", "b": "x", "type": "weld"}],
        })


def test_ambiguous_model_defaults_to_assembly_with_needs():
    result = plan_model({"parts": [], "interfaces": [], "needs": []})
    assert result["kind"] == "assembly"
    assert result["needs"]
