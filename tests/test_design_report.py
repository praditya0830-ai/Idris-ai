"""POST /design/report and GET /design/reports (doc: a failed report is always kept, 25 Nov 2026).

The store is a small JSONL file; these tests point IDRIS_REPORT_DIR at a temporary folder, so nothing is written
into the checkout and the real store is never read.
"""
import json

import design_report
import http_ext

REPORT = {"part_name": "bracket", "shape_class": "plate",
          "rungs_tried": [{"rung": "features", "error": "no template fits"}],
          "suggestion": "give the plate thickness"}


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv(design_report.STORE_ENV, str(tmp_path))
    return tmp_path


def test_the_route_is_registered_through_the_small_module_hook():
    assert http_ext.known_post("/design/report")
    assert http_ext.handle_get("/design/reports?limit=5") is not None


def test_a_failed_report_is_saved_and_listed(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    status, body = http_ext.handle_post("/design/report", {
        "report": REPORT, "message": "make a bracket", "topic": "bracket", "solver_version": "0.43.0"})
    assert status == 200 and body["ok"] and body["id"].startswith("rep-")
    got = http_ext.handle_get("/design/reports?limit=50")
    assert got[0] == 200 and got[1]["count"] == 1
    row = got[1]["reports"][0]
    assert row["id"] == body["id"]
    assert row["report"]["part_name"] == "bracket"
    assert row["message"] == "make a bracket" and row["topic"] == "bracket"
    assert row["solver_version"] == "0.43.0"
    assert (tmp_path / design_report.FILE_NAME).exists()


def test_newest_first_and_the_limit_is_honoured(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    ids = []
    for i in range(3):
        _s, body = http_ext.handle_post("/design/report", {"report": {**REPORT, "part_name": f"p{i}"}})
        ids.append(body["id"])
    rows = http_ext.handle_get("/design/reports?limit=2")[1]["reports"]
    assert len(rows) == 2 and rows[0]["id"] == ids[-1] and rows[1]["id"] == ids[-2]


def test_a_body_without_a_report_is_refused_with_the_reason(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    status, body = http_ext.handle_post("/design/report", {"message": "no report here"})
    assert status == 400 and not body["ok"] and "report" in body["error"]
    assert http_ext.handle_get("/design/reports")[1]["count"] == 0


def test_the_message_is_capped_so_the_store_cannot_be_flooded(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    _s, body = http_ext.handle_post("/design/report", {"report": REPORT, "message": "x" * 9000})
    row = http_ext.handle_get("/design/reports")[1]["reports"][0]
    assert row["id"] == body["id"]
    assert len(row["message"]) == design_report.MAX_MESSAGE_CHARS


def test_an_oversized_report_is_refused_instead_of_stored(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    big = {**REPORT, "suggestion": "y" * (design_report.MAX_REPORT_BYTES + 100)}
    status, body = http_ext.handle_post("/design/report", {"report": big})
    assert status == 400 and "keep it under" in body["error"]


def test_the_stored_rows_hold_no_request_identity_fields(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    http_ext.handle_post("/design/report", {"report": REPORT, "message": "m", "ip": "10.0.0.1",
                                           "email": "a@b.c", "user": "someone"})
    row = http_ext.handle_get("/design/reports")[1]["reports"][0]
    assert set(row) == {"id", "saved_at", "report", "message", "topic", "solver_version"}
    assert "10.0.0.1" not in json.dumps(row)


def test_another_path_is_not_answered_by_this_module():
    assert design_report.handle_post("/report", {}) is None
    assert design_report.handle_get("/design/report") is None
