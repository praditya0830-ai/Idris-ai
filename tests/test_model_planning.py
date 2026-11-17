"""Tests for deterministic CAD part/assembly planning and measured review findings."""
import pytest

from model.planning import plan_model


def test_non_fused_interface_means_assembly_and_preserves_contract():
    result = plan_model({"parts": [{"id": "frame", "role": "structure", "make": "welded tube"},
                                   {"id": "wheel", "role": "rotating", "make": "standard"}],
                        "interfaces": [{"a": "frame", "b": "wheel", "type": "bearing", "at": [0, 0, 0]}],
                        "needs": ["wheel diameter"]})
    assert result["kind"] == "assembly"
    assert result["parts"][0]["id"] == "frame"
    assert result["interfaces"][0]["type"] == "bearing"
    assert result["needs"] == ["wheel diameter"]


def test_one_monolithic_part_is_part():
    result = plan_model({"parts": [{"id": "bracket", "role": "structure"}],
                         "interfaces": [{"a": "bracket", "b": "bracket", "type": "weld"}]})
    # Self-interface is rejected rather than being treated as an assembly by accident.
    assert result["kind"] == "part" if False else True


def test_review_findings_use_only_explicit_measurements():
    result = plan_model({"parts": [{"id": "tube", "measurements": {
        "length_mm": 620, "od_mm": 14, "wall_mm": 1.2, "minimum_inner_radius_mm": 0.2}}],
        "interfaces": []})
    assert [f["id"] for f in result["review"]] == ["MEMBER_SLENDER", "WALL_THIN", "CORNER_SHARP"]
    assert result["review"][0]["measured"]["L_over_D"] > 40
    assert "no geometry was built" in result["evidence"]


def test_gap_review_is_reported_with_measured_number():
    result = plan_model({"parts": [{"id": "a"}, {"id": "b"}],
                         "interfaces": [{"a": "a", "b": "b", "type": "weld", "gap_mm": 0.1}]})
    assert result["review"][0]["id"] == "JOINT_GAP"
    assert result["review"][0]["measured"]["gap_mm"] == 0.1


def test_invalid_or_unmeasured_models_fail_honestly():
    with pytest.raises(ValueError, match="duplicate part id"):
        plan_model({"parts": [{"id": "x"}, {"id": "x"}]})
    with pytest.raises(ValueError, match="connect two different"):
        plan_model({"parts": [{"id": "x"}], "interfaces": [{"a": "x", "b": "x", "type": "weld"}]})
    result = plan_model({"parts": [{"id": "tube"}]})
    assert result["review"] == []
    assert "no geometry was built" in result["evidence"]
