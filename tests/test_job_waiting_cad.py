"""UI v66 A: the design job sends the Studio the code of the round it is waiting for.

Without it the circle cannot turn: the job said "waiting for browser_build" and the Studio knew it had to build
something but not what. These tests pin the contract (`waiting_cad`, `waiting_cad_hash`, `rounds_data`) and the
one rule that protects the numbers: feedback measured on another round's code is refused.
"""
import hashlib
import json
import threading
import time

import pytest

import design_job as dj


def _cad(t=3.0, user_length=40.0, req=None):
    return {"title": "L bracket", "kind": "functional", "process": "cnc", "material": "aluminium", "note": "",
            "code": ("// step: base plate\nconst body = makeBox([0, 0, 0], [1, 2, 3]);\n"
                     f"// step: rib\nconst rib = makeBox([0, 0, 0], [1, 2, {t}]);\nreturn {{ body: body.fuse(rib) }};"),
            "params": [{"k": "L", "label": "Length", "unit": "mm", "value": user_length, "min": 10, "max": 200,
                        "source": "user"},
                       {"k": "t", "label": "Wall", "unit": "mm", "value": t, "min": 0.4, "max": 12,
                        "source": "default", "wall": True}],
            "expect": [{"label": "Length", "axis": "x", "value": user_length, "tol": 0.5}],
            "holes": [], "variants": [], "missing": [], "requirements": req, "read_by": "fake (model)"}


def _writer(t=3.0, req=None):
    return lambda msg, ctx, prov: {"ok": True, "cad": _cad(t, req=req), "answer": "", "attempts": []}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("IDRIS_JOB_FEEDBACK_WAIT_S", "5")
    dj._STORE.clear()
    dj._WAIT.clear()
    yield
    dj._STORE.clear()
    dj._WAIT.clear()


def _wait_state(job_id, need="browser_build", timeout=8.0):
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


# ------------------------------------------------------------------ what the Studio is sent
def test_the_status_carries_the_cad_and_its_hash_for_the_round_being_waited_for():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="wc1", feedback=dj.browser_feedback("wc1"))
    st = _wait_state("wc1")

    cad = st["waiting_cad"]
    assert cad and cad["code"] and cad["steps"], "the Studio cannot build what it was not sent"
    assert [s["name"] for s in cad["steps"]] == ["base plate", "rib"]
    assert cad["params"][0]["k"] == "L" and cad["expect"][0]["axis"] == "x"
    assert st["waiting_cad_hash"] == hashlib.sha256(cad["code"].encode()).hexdigest()
    assert st["waiting_cad_hash"] in st["how_to_continue"]
    assert len(json.dumps(st)) < 60_000, "a status is polled often: it must stay small"

    dj.handle_feedback("/design/job/wc1/feedback", {"round": 1, "measurements": {"x": 40.0}})
    t.join(10)


def test_feedback_measured_on_another_round_is_refused_with_the_hash_to_build():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="wc2", feedback=dj.browser_feedback("wc2"))
    st = _wait_state("wc2")
    good = st["waiting_cad_hash"]

    code, body = dj.handle_feedback("/design/job/wc2/feedback",
                                    {"round": 1, "cad_hash": "0" * 64, "measurements": {"x": 40.0}})
    assert code == 409 and body["waiting_cad_hash"] == good
    assert "different code" in body["error"]

    code, body = dj.handle_feedback("/design/job/wc2/feedback",
                                    {"round": 1, "cad_hash": good, "measurements": {"x": 40.0}})
    assert code == 200 and body["cad_hash"] == good
    t.join(10)


def test_the_hash_is_optional_so_an_older_studio_still_closes_the_circle():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="wc3", feedback=dj.browser_feedback("wc3"))
    _wait_state("wc3")
    code, _body = dj.handle_feedback("/design/job/wc3/feedback", {"round": 1, "measurements": {"x": 40.0}})
    assert code == 200
    t.join(10)


# ------------------------------------------------------------------ what each round was, without its code
def test_rounds_data_summarises_every_round_and_names_what_the_repair_changed():
    calls = {"n": 0}

    def writer(msg, ctx, prov):
        calls["n"] += 1
        return {"ok": True, "cad": _cad(t=3.0 if calls["n"] == 1 else 4.0), "answer": "", "attempts": []}

    def review(cad_now):
        return {"verdict": "revise" if calls["n"] == 1 else "ok", "notes": ["wall 3 mm is thin for cnc"]}

    res = dj.run("bracket L 40", cad_fn=writer, review=review, job_id="rd1")
    rounds = res["rounds_data"]
    assert [r["round"] for r in rounds] == [1, 2]
    assert rounds[0]["changed"] == [], "nothing was changed before the first round"
    assert rounds[0]["label"] == "Not validated" and rounds[0]["cad_hash"]
    assert "code" not in rounds[0], "rounds_data never carries the code of an old round"
    changed = rounds[1]["changed"]
    assert changed == [{"param": "t", "from": 3.0, "to": 4.0, "why": changed[0]["why"]}]
    assert "visual review" in changed[0]["why"] and "wall 3 mm" in changed[0]["why"]
    assert rounds[1]["cad_hash"] != rounds[0]["cad_hash"]
    assert any(c["status"] == "pass" for c in rounds[1]["checks"])


def test_the_status_shows_the_rounds_that_have_finished_while_the_job_still_runs():
    out = {}
    t = _run_in_thread(out, cad_fn=_writer(), job_id="rd2", feedback=dj.browser_feedback("rd2"))
    _wait_state("rd2")
    dj.handle_feedback("/design/job/rd2/feedback", {"round": 1, "measurements": {"x": 55.0}})
    t.join(10)
    st = dj.status("rd2")
    assert st["status"] == "done" and st["result"]["rounds_data"][0]["round"] == 1
    assert st["result"]["rounds_data"][0]["checks"], "the checks of the round are kept as they ran"
