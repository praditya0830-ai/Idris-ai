"""Review of PR #5-#8, item 8: the job and the browser each do half of the circle, and neither lies.

No AI and no kernel: the code writer is a fake and the browser is a test thread that posts the same JSON the
Studio posts. The rules under test: the job says what it is waiting for, it uses the numbers it is given, a
wait nobody answers stops it with a plain reason, and a round number that is not the one waiting is refused.
"""
import threading
import time

import pytest

import design_job as dj
import http_ext


def _cad(hole=4.5, user_length=40.0, req=None):
    return {"title": "L bracket", "kind": "functional", "process": "cnc", "material": "aluminium",
            "note": "", "code": "// step: base plate\nconst body = makeBox([0, 0, 0], [1, 2, 3]);\nreturn { body };",
            "params": [{"k": "L", "label": "Length", "unit": "mm", "value": user_length,
                        "min": 10, "max": 200, "source": "user"},
                       {"k": "t", "label": "Wall", "unit": "mm", "value": 3.0,
                        "min": 0.4, "max": 12, "source": "default", "wall": True}],
            "expect": [{"label": "Length", "axis": "x", "value": user_length, "tol": 0.5}],
            "holes": [{"x": 20, "y": 10, "z_top": 3, "axis": "z", "d": hole}],
            "variants": [], "missing": [], "requirements": req, "read_by": "fake (model)"}


def _writer(hole=4.5, user_length=40.0, req=None):
    return lambda msg, ctx, prov: {"ok": True, "cad": _cad(hole, user_length, req), "answer": "", "attempts": []}


LOAD = {"loads": [{"force_N": [0, 0, -100], "where": {"plane": "z", "side": "min"}, "label": "Load"}],
        "supports": [{"where": {"plane": "z", "side": "min"}}], "design_factor": 2.0}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("IDRIS_JOB_FEEDBACK_WAIT_S", "5")
    dj._STORE.clear()
    dj._WAIT.clear()
    yield
    dj._STORE.clear()
    dj._WAIT.clear()


def _wait_state(job_id, need, timeout=8.0):
    """Poll the job like the Studio does, until it says it waits for `need`."""
    deadline = time.time() + timeout
    seen = None
    while time.time() < deadline:
        seen = dj.status(job_id) or {}
        if seen.get("waiting_for") == need:
            return seen
        time.sleep(0.02)
    raise AssertionError(f"the job never waited for {need!r}; last status: {seen}")


def _run_in_thread(result, **kw):
    t = threading.Thread(target=lambda: result.update(dj.run(kw.pop("request", "bracket L 40"), **kw)))
    t.start()
    return t


# ------------------------------------------------------------------ the circle closes
def test_the_job_waits_for_the_browser_and_then_uses_its_measurements():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="jr1", feedback=dj.browser_feedback("jr1"))

    st = _wait_state("jr1", "browser_build")
    assert st["waiting_round"] == 1
    assert "measurements" in st["how_to_continue"], "the status must say what the Studio has to send"

    status, body = dj.handle_feedback("/design/job/jr1/feedback",
                                      {"round": 1, "measurements": {"x": 40.0, "y": 30.0, "z": 5.0}})
    assert status == 200 and body["accepted"] == ["measurements"]
    t.join(10)

    assert out["label"] == "Geometry only", out["checks"]
    size = [c for c in out["checks"] if c["id"] == "kernel_size"][0]
    assert size["status"] == "pass" and size["source"] == "the CAD kernel in the browser"
    assert [e["step"] for e in out["events"]][:3] == ["plan", "code", "build"]


def test_measurements_from_the_browser_can_fail_the_engineer_size(monkeypatch):
    monkeypatch.setenv("IDRIS_DESIGN_ROUNDS", "1")          # one round: the mismatch is the answer
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="jr2", feedback=dj.browser_feedback("jr2"))
    _wait_state("jr2", "browser_build")
    dj.handle_feedback("/design/job/jr2/feedback", {"round": 1, "measurements": {"x": 55.0}})
    t.join(10)

    size = [c for c in out["checks"] if c["id"] == "kernel_size"][0]
    assert size["status"] == "fail"
    assert "55 mm vs 40 mm" in size["message"]
    assert out["label"] == "Not validated" and out["ok"] is False


def test_the_strength_numbers_from_the_browser_decide_the_safety_factor(monkeypatch):
    monkeypatch.setenv("IDRIS_DESIGN_ROUNDS", "1")
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(req=LOAD), job_id="jr3", feedback=dj.browser_feedback("jr3"))

    _wait_state("jr3", "browser_build")
    dj.handle_feedback("/design/job/jr3/feedback", {"round": 1, "measurements": {"x": 40.0}})

    st = _wait_state("jr3", "fea")                           # Part Stress has not answered yet
    assert "part_check" in st["how_to_continue"]
    dj.handle_feedback("/design/job/jr3/feedback",
                       {"round": 1, "part_check": {"sf": 1.2, "source": "Part Stress (browser)"}})
    t.join(10)

    fea = [c for c in out["checks"] if c["id"] == "fea"][0]
    assert fea["status"] == "fail" and fea["value"] == 1.2
    assert fea["source"] == "Part Stress (browser)", "the number is quoted with where it came from"
    assert out["label"] == "Not validated"


# ------------------------------------------------------------------ honesty when nobody answers
def test_a_wait_nobody_answers_stops_the_job_with_a_plain_reason(monkeypatch):
    monkeypatch.setenv("IDRIS_JOB_FEEDBACK_WAIT_S", "0.3")
    started = time.time()
    res = dj.run("bracket L 40", cad_fn=_writer(), job_id="jr4", feedback=dj.browser_feedback("jr4"))
    assert time.time() - started < 5, "the wait must be bounded"
    assert "waiting for the browser timed out" in res["answer"]
    assert "Nothing is reported as a pass" in res["answer"]
    assert res["stop_reason"] and res["label"] == "Not validated"
    assert not [c for c in res["checks"] if c["id"] == "kernel_size" and c["status"] == "pass"]


def test_the_wait_can_be_turned_off_and_the_round_says_the_build_is_not_confirmed(monkeypatch):
    monkeypatch.setenv("IDRIS_JOB_FEEDBACK_WAIT_S", "0")
    res = dj.run("bracket L 40", cad_fn=_writer(), job_id="jr5", feedback=dj.browser_feedback("jr5"))
    assert res["label"] == "Geometry only"
    build = [e for e in res["events"] if e["step"] == "build"][0]
    assert build["status"] == "not_checked"


# ------------------------------------------------------------------ the route
def test_the_route_refuses_a_round_that_is_not_the_one_waiting():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="jr6", feedback=dj.browser_feedback("jr6"))
    _wait_state("jr6", "browser_build")
    status, body = dj.handle_feedback("/design/job/jr6/feedback", {"round": 3, "measurements": {"x": 40.0}})
    assert status == 409 and body["round"] == 1 and body["waiting_for"] == "browser_build"
    dj.handle_feedback("/design/job/jr6/feedback", {"round": 1, "measurements": {"x": 40.0}})
    t.join(10)
    assert out["label"] == "Geometry only"


def test_the_route_refuses_an_empty_payload_and_an_unknown_job():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="jr7", feedback=dj.browser_feedback("jr7"))
    _wait_state("jr7", "browser_build")
    status, body = dj.handle_feedback("/design/job/jr7/feedback", {"round": 1})
    assert status == 400 and "at least one of" in body["error"]
    dj.handle_feedback("/design/job/jr7/feedback", {"round": 1, "measurements": {"x": 40.0}})
    t.join(10)

    status, body = dj.handle_feedback("/design/job/nosuchjob/feedback", {"round": 1, "measurements": {}})
    assert status == 404 and "not waiting" in body["error"]


def test_http_ext_recognises_only_the_feedback_path_and_delegates_it():
    assert http_ext.is_design_feedback("/design/job/abc123/feedback")
    assert http_ext.is_design_feedback("/design/job/abc123/feedback/")
    assert not http_ext.is_design_feedback("/design/job")
    assert not http_ext.is_design_feedback("/design/job/abc123")
    assert http_ext.known_post("/design/job/abc123/feedback")
    assert not http_ext.known_post("/design/job/abc123")
    status, body = http_ext.handle_post("/design/job/abc123/feedback", {"round": 1, "measurements": {}})
    assert status == 404 and body["success"] is False


def test_a_job_with_an_injected_writer_never_waits():
    """The default: a job whose code writer was injected has no browser, so nothing waits at all."""
    started = time.time()
    res = dj.run("bracket L 40", cad_fn=_writer())
    assert time.time() - started < 5 and res["label"] == "Geometry only"
    assert dj.waiting_for(None) is None
