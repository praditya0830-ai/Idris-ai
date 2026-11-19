"""Bagian D2 tests: the report carries every mandatory section, escapes the engineer's text, says plainly when
no strength check was run, and names the solver and its benchmark next to a result.
"""
import os

import part_report
import report_ext

CAD = {"title": "Bracket", "params": [{"k": "L", "label": "Length", "value": 40.0, "unit": "mm", "source": "user"},
                                      {"k": "t", "label": "Thickness", "value": 5.0, "unit": "mm",
                                       "source": "default: minimum for CNC aluminium"}]}


def test_all_mandatory_sections_are_present():
    out = part_report.build({"request": "bracket 40 mm aluminium", "cad": CAD})
    doc = out["html"]
    for heading in ("1. The request", "2. Inputs and where each number comes from", "3. Material",
                    "4. Analysis settings", "5. Strength result", "8. Not checked by IDRIS"):
        assert heading in doc, heading
    assert out["filename"].startswith("report_part_") and out["filename"].endswith(".html")
    assert out["download_url"] == "/download/" + out["filename"]
    assert os.path.isfile(os.path.join(part_report.OUTPUT_DIR, out["filename"]))


def test_the_engineer_text_is_escaped():
    out = part_report.build({"request": "<script>alert(1)</script>", "cad": CAD})
    assert "<script>alert(1)</script>" not in out["html"]
    assert "&lt;script&gt;" in out["html"]


def test_without_a_load_the_strength_section_says_so():
    doc = part_report.build({"request": "bracket", "cad": CAD})["html"]
    assert "No load was given; no strength check was run." in doc
    assert "PASS" not in doc and "Validated" not in doc


def test_every_input_names_its_source():
    doc = part_report.build({"request": "bracket", "cad": CAD})["html"]
    assert "user (the engineer stated it)" in doc
    assert "default: minimum for CNC aluminium" in doc


def test_a_fea_result_names_the_solver_and_its_benchmark():
    body = {"request": "bracket 40 mm", "cad": CAD,
            "part_check": {"solver": "fea_part_3d", "von_mises_max_mpa": 118.4, "sf": 2.1,
                           "displacement_max_mm": 0.21,
                           "mesh": {"elements": 41250, "nodes": 68310, "element_type": "C3D10"},
                           "loads": [{"force_N": 900}], "supports": [{"where": "two bolt holes"}],
                           "convergence": {"h1_mm": 2.0, "h2_mm": 1.0, "delta_pct": 3.4}}}
    doc = part_report.build(body)["html"]
    assert "118.4 MPa" in doc and "2.1" in doc and "0.21 mm" in doc
    assert "fea_part_3d" in doc
    assert "no published benchmark for this solver" in doc or "benchmark" in doc
    assert "C3D10" in doc and "two bolt holes" in doc and "delta_pct" in doc


def test_a_candidate_formula_is_marked_as_unused():
    body = {"request": "bracket", "cad": CAD,
            "references": {"items": [{"title": "Beam bending", "formula": "sigma = M*c/I",
                                      "source": "https://example.org/beam"}]}}
    doc = part_report.build(body)["html"]
    assert "unverified candidate - not used for any number" in doc
    assert "NOT used to produce any number" in doc


def test_the_not_checked_section_is_never_empty():
    doc = part_report.build({"request": "bracket", "cad": CAD})["html"]
    assert "fatigue and buckling" in doc
    assert "manufacturability (DFM)" in doc


def test_an_empty_body_is_a_clear_400():
    status, body = report_ext.handle_post("/report/part", {})
    assert status == 400 and "request sentence" in body["error"]


def test_an_unknown_route_is_not_handled_here():
    assert report_ext.handle_post("/report/other", {}) is None
    assert report_ext.handle_get("/report/part") is None
