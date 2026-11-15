"""POST /report (engineering package HTML) and GET /materials."""
import json

import pytest

import report
from physics.part_fea import MATERIALS, material_catalog

PAYLOAD = {
    "title": "Gearbox S1 - Variant B",
    "project": "VTOL actuator",
    "author": "A. Engineer",
    "summary": {"Mass": "6.49 kg", "Min FoS": 2.1, "Verdict": "PASS"},
    "requirements": {"power_kw": 11.19, "input_rpm": 6000, "gear_ratio": 8},
    "variant": {"name": "Variant B",
                "design": {"module_mm": 2.0, "pinion_teeth": 17, "gear_teeth": 48},
                "kinematics": {"ratio_error_pct": -0.346},
                "mass_kg": {"total": 6.494}},
    "checks": [{"name": "Bending safety", "value": 1.62, "limit": 1.4, "unit": "-", "pass": True},
               {"name": "Contact safety", "value": 1.05, "limit": 1.1, "unit": "-", "pass": False}],
    "analyses": {"fea": {"summary": {"max_von_mises_mpa": 84.2, "margin_of_safety": 2.1}},
                 "modal": {"frequencies_hz": [812.0, 1204.0]},
                 "thermal": {"oil_temp_c": 85.0}},
    "review": {"counts": {"critical": 0, "warning": 1},
               "findings": [{"severity": "warning", "check": "dfm", "part": "Housing",
                             "message": "wall 0.9 mm", "fix": "use 1.5 mm"}]},
    "bom": [{"part": "Housing", "qty": 1, "material": "Al7075-T6"},
            {"part": "Cover", "qty": 1, "material": "Al7075-T6"}],
    "solvers": [{"name": "gear_pair_iso6336", "version": "0.2.0", "fidelity": "L0",
                 "method": "ISO 6336 Method B",
                 "benchmarks": [{"source": "ISO/TR 6336-30 Example 1"}]}],
}


def test_render_contains_all_sections():
    page = report.render(PAYLOAD)
    assert page.startswith("<!doctype html>")
    for text in ("Gearbox S1 - Variant B", "VTOL actuator", "A. Engineer",
                 "Requirements", "Variant", "Design checks", "Analyses", "Design review",
                 "Bill of materials", "Solver evidence",
                 "ISO 6336 Method B", "ISO/TR 6336-30 Example 1", "Housing", "wall 0.9 mm",
                 "812", "84.2"):
        assert text in page, text
    assert "PASS" in page and "FAIL" in page
    assert page.count("<table") >= 4


def test_render_escapes_user_text():
    page = report.render({
        "title": '<script>alert("x")</script>',
        "review": {"findings": [{"severity": "critical", "message": '<img src=x onerror=alert(1)>'}]},
        "bom": [{"part": '<b onmouseover="x">evil</b>'}],
    })
    assert "<script>alert" not in page
    assert "<img src=x" not in page
    assert "<b onmouseover" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;img src=x" in page


def test_render_empty_payload_is_still_a_report():
    page = report.render({})
    assert "IDRIS Engineering Report" in page
    assert "no summary supplied" in page
    assert "no requirements supplied" in page
    assert "no solver evidence" in page or "No solver evidence" in page


def test_render_rejects_non_mapping():
    with pytest.raises(ValueError):
        report.render(["not", "an", "object"])  # type: ignore[arg-type]


def test_render_rejects_oversized_payload():
    with pytest.raises(ValueError):
        report.render({"title": "x" * (report.MAX_PAYLOAD_BYTES + 1)})


def test_render_summary_derived_when_missing():
    page = report.render({"variant": {"mass_kg": {"total": 6.49}},
                          "checks": [{"name": "a", "pass": True}, {"name": "b", "pass": False}]})
    assert "6.49 kg" in page
    assert "1/2 pass" in page


def test_material_catalog_covers_simulator_materials():
    cat = material_catalog()
    ids = {m["id"] for m in cat["materials"]}
    assert ids == set(MATERIALS)
    assert {"aluminium", "steel", "titanium"} <= ids
    row = next(m for m in cat["materials"] if m["id"] == "titanium")
    assert {"name", "E", "nu", "Sy", "rho"} <= set(row)
    assert row["Sy"] == 880.0
    assert cat["note"]


def test_payload_is_json_serialisable():
    # the route hands the request body straight to render(); make sure the
    # documented shape survives a JSON round-trip unchanged (no tuples/sets)
    assert json.loads(json.dumps(PAYLOAD)) == PAYLOAD
