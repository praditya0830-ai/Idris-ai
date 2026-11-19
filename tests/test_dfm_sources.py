"""Review of PR #5-#8, item 6: every DFM number names the document that really contains it.

The rule under test: a source is only quoted when it holds the number. A welding symbol standard does not set a
fillet leg, so it is not quoted any more; the AWS D1.1 table does, and its limit depends on the plate thickness.
A figure nobody could point to a table for stays a guideline, and a guideline may only warn — never fail a part.
"""
import pytest

from physics import dfm_rules


def _wall(process, material, value, thickness_key="t"):
    return dfm_rules.check({"process": process, "material": material,
                            "params": [{"k": thickness_key, "value": value, "wall": True, "label": "wall"}]})


def _check(out, cid="min_wall"):
    return [c for c in out["checks"] if c["id"] == cid][0]


# ------------------------------------------------------------------ the welding rule
def test_the_weld_rule_no_longer_claims_iso_2553():
    text = dfm_rules.WALL_SOURCE["weld"]["text"]
    assert "ISO 2553" not in text, "ISO 2553 is about welding symbols, it sets no fillet size"
    assert "AWS D1.1" in text and "Table 7.7" in text


@pytest.mark.parametrize("thickness,leg", [(1.0, 3.0), (6.0, 3.0), (8.0, 5.0), (12.0, 5.0), (16.0, 6.0),
                                           (20.0, 6.0), (30.0, 8.0)])
def test_the_weld_leg_follows_the_aws_table_by_thickness(thickness, leg):
    assert dfm_rules.weld_fillet_min_mm(thickness) == leg


def test_a_wall_at_or_above_the_table_minimum_passes():
    out = _wall("weld", "steel", 7.0)                     # 7 mm wall: the table asks for 5 mm
    c = _check(out)
    assert c["status"] == "pass" and c["limit"] == 5.0
    assert c["source_status"] == dfm_rules.VERIFIED


def test_a_leg_below_the_aws_minimum_fails_because_that_source_is_verified():
    out = _wall("weld", "steel", 2.0)
    c = _check(out)
    assert c["status"] == "fail" and c["limit"] == 3.0
    assert c["source_status"] == dfm_rules.VERIFIED
    assert out["ok"] is False


# ------------------------------------------------------------------ the guidelines
def test_a_rule_without_a_verifiable_source_warns_and_never_fails():
    out = _wall("cnc", "aluminium", 0.5)
    c = _check(out)
    assert c["status"] == "warn" and c["source_status"] == dfm_rules.UNVERIFIED
    assert out["ok"] is True and out["failed"] == [] and out["warned"] == [c]


def test_every_process_entry_states_a_status_and_a_text():
    for process, rule in dfm_rules.WALL_SOURCE.items():
        assert rule["text"].strip(), process
        assert rule["status"] in (dfm_rules.VERIFIED, dfm_rules.UNVERIFIED), process


def test_a_wall_above_the_guideline_still_passes():
    out = _wall("fdm", "pla", 2.0)
    assert _check(out)["status"] == "pass" and out["ok"] is True


def test_a_process_and_material_without_a_rule_is_not_guessed():
    out = _wall("cnc", "unobtainium", 5.0)
    c = _check(out)
    assert c["status"] == "not_checked" and "not guessed" in c["message"]


# ------------------------------------------------------------------ the drill series
def test_the_drill_rule_names_the_standard_it_uses():
    assert "DIN 338" in dfm_rules.DRILL_SOURCE and "ISO 235" in dfm_rules.DRILL_SOURCE
    out = dfm_rules.check({"process": "cnc", "material": "aluminium",
                           "holes": [{"d": 3.7}]})
    c = _check(out, "drill_size")
    assert c["status"] == "fail" and c["source_status"] == dfm_rules.VERIFIED_SERIES


def test_a_standard_drill_diameter_passes():
    out = dfm_rules.check({"process": "cnc", "material": "aluminium", "holes": [{"d": 6.0}, {"d": 8.5}]})
    assert _check(out, "drill_size")["status"] == "pass"


# ------------------------------------------------------------------ the one-line summary
def test_the_summary_states_a_warning_as_a_warning():
    line = dfm_rules.summary(_wall("cnc", "aluminium", 0.5))
    assert "warning" in line and "FAILED" not in line
