"""Batch 1 (audit 27 Nov): an assembly job stops at the review gate before it assembles anything.

Every part is designed, then the job holds at `awaiting_review` with one row per part: rung, the sizes and
their source, the assumptions, and the failure report when a part failed. `POST /assembly/job/<id>/review`
approves (and only then places the parts), redesigns one part with the engineer's own sizes and comes back to
the gate, or skips a failed part — which is written into the answer, not silently ignored.

The gate is opt-in (`"review_gate": true` on /assembly/design): the Studio that is live today only stops polling
on done/failed, so the old flow must keep working until its review panel ships. A redesign answers 202 and runs
in the background in its own assembly slot, so the HTTP call never hangs and two designs never run at once.
"""
import time

import assembly_ext
import assembly_plan
import http_ext

PLAN = {
    "parts": [
        {"id": "frame", "name": "Frame", "role": "carries the load", "qty": 1, "standard": False,
         "dims": {"length_mm": 200}, "sources": {"length_mm": "you"},
         "assumptions": ["wall thickness 2 mm until the engineer says otherwise"]},
        {"id": "bracket", "name": "Bracket", "role": "holds the frame", "qty": 2, "standard": False,
         "dims": {"length_mm": 60, "width_mm": 40},
         "sources": {"length_mm": "you", "width_mm": "table: ISO 2768 medium"}},
        {"id": "bolt_m6", "name": "M6 bolt", "role": "fastens", "qty": 8, "standard": True},
    ],
    "interfaces": [{"between": ["frame", "bracket"], "kind": "bolted", "params": {"d_mm": 6},
                    "source": "table: ISO 4017 M6"}],
    "frames": [{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "bracket", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "bolt_m6", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]}],
    "sources": {}, "questions": [],
}

FEATURES = {"feature_tree": [{"type": "sketch"}, {"type": "push"}], "code": "const body = null;"}
CODE = {"code": "const body = makeBox([0,0,0],[60,40,5]);"}


def _designer(fail=(), calls=None):
    """A designer that fails the parts named in `fail`, and builds the rest from a feature recipe."""
    def d(prompt, prov=None, ctx=None):
        if calls is not None:
            calls.append((prompt, ctx))
        name = str((ctx or {}).get("assembly_part") or "")
        if name in fail or (not name and any(w in prompt for w in fail)):
            return {"ok": False, "cad": None, "answer": "the writer gave no usable size for this part"}
        return {"ok": True, "cad": dict(FEATURES), "fidelity": "features"}
    return d


def _slow(inner, seconds):
    """The same designer, but it takes its time: a redesign must not hold the HTTP request open for it."""
    def d(prompt, prov=None, ctx=None):
        time.sleep(seconds)
        return inner(prompt, prov, ctx)
    return d


def _run(designer, plan=None, job_id=None, review_gate=True):
    plan = plan if plan is not None else assembly_plan.validate(PLAN)[0]
    job_id = job_id or f"gate-{time.time_ns()}"
    job = {"id": job_id, "request": "a frame with a bracket", "plan": plan, "parts": [], "events": [],
           "status": "queued", "started": time.time(), "finished": None, "error": None, "assembly": None,
           "stopped": None, "provider": None, "designer": designer, "review": None, "skipped": [],
           "redesigns": [], "review_gate": review_gate, "review_since": None}
    assembly_ext._JOBS[job_id] = job
    assembly_ext.run_job(job, designer=designer)
    return job


def _wait(job_id, want, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        st = assembly_ext.status(job_id)
        if st and st["status"] == want:
            return st
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never reached {want}: "
                         f"{(assembly_ext.status(job_id) or {}).get('status')}")


def test_the_job_stops_at_the_gate_and_places_nothing():
    job = _run(_designer())
    st = assembly_ext.status(job["id"])
    assert st["status"] == assembly_ext.REVIEW_STATUS == "awaiting_review"
    assert st["assembly"] is None                         # nothing is placed before the review
    review = st["review"]
    assert review["can_assemble"] and review["blocked"] == []
    rows = {r["id"]: r for r in review["parts"]}
    assert rows["frame"]["rung"] == "features" and rows["bracket"]["rung"] == "features"
    assert rows["bolt_m6"]["rung"] == "standard"           # bought, never designed, and never blocks the gate
    frame = rows["frame"]
    assert frame["dims"] == [{"k": "length_mm", "value": 200.0, "unit": "mm", "source": "you",
                              "method": None}]
    assert frame["assumptions"] == ["wall thickness 2 mm until the engineer says otherwise"]
    assert rows["bracket"]["dims"][1]["source"] == "table: ISO 2768 medium"
    assert st["events"][-1]["step"] == "review" and "Review them before assembly" in st["events"][-1]["message"]


def test_without_the_flag_the_job_finishes_as_the_live_studio_expects():
    """Review WAJIB 1: the Studio on main only stops polling at done/failed, so the gate must be opt-in."""
    job = _run(_designer(), review_gate=False)
    st = assembly_ext.status(job["id"])
    assert st["status"] == "done" and job["finished"] is not None
    assert st["assembly"] is not None and st["assembly"]["parts"]
    assert st["review"] is None and st["review_gate"] is False
    assert st["eta_s"] == 0.0


def test_the_design_route_only_gates_when_review_gate_is_true(monkeypatch):
    started = []

    def fake_start(request, plan, provider=None, designer=None, review_gate=False):
        started.append(review_gate)
        return "jobid"

    monkeypatch.setattr(assembly_ext, "start_job", fake_start)
    raw = assembly_plan.validate(PLAN)[0]
    raw.pop("unsourced", None)
    code, out = assembly_ext.handle_post("/assembly/design", {"request": "a frame", "plan": PLAN})
    assert code == 200 and out["job_id"] == "jobid" and started == [False]
    code, out = assembly_ext.handle_post("/assembly/design",
                                         {"request": "a frame", "plan": PLAN, "review_gate": True})
    assert code == 200 and started == [False, True]


def test_a_review_nobody_answers_is_dropped_after_24_hours_and_says_why():
    """Review WAJIB 2: an awaiting_review job has finished=None, so the old cleanup never freed it."""
    job = _run(_designer())
    assert assembly_ext.status(job["id"])["status"] == assembly_ext.REVIEW_STATUS
    job["review_since"] = time.time() - assembly_ext.REVIEW_TTL_S - 1
    code, out = assembly_ext.handle_get(f"/assembly/job/{job['id']}")
    assert code == 404 and out["expired"] is True
    assert out["error"] == "review expired after 24 h, start the assembly again"
    assert job["id"] not in assembly_ext._JOBS
    assert assembly_ext.status(job["id"]) is None
    code, out = assembly_ext.handle_post(f"/assembly/job/{job['id']}/review", {"action": "approve"})
    assert code == 404 and out["expired"] is True


def test_a_young_review_is_not_dropped_and_a_finished_job_keeps_its_own_ttl():
    job = _run(_designer())
    job["review_since"] = time.time() - 60.0
    assert assembly_ext.status(job["id"])["status"] == assembly_ext.REVIEW_STATUS
    done = _run(_designer(), review_gate=False)
    done["finished"] = time.time() - 60.0
    assert assembly_ext.status(done["id"])["status"] == "done"


def test_a_failed_part_blocks_assembly_until_it_is_skipped_and_the_skip_is_recorded():
    job = _run(_designer(fail=("bracket",)))
    st = assembly_ext.status(job["id"])
    rows = {r["id"]: r for r in st["review"]["parts"]}
    assert rows["bracket"]["rung"] == "failed"
    assert "no usable size" in rows["bracket"]["failed_report"]["error"]
    assert st["review"]["blocked"] == ["bracket"] and not st["review"]["can_assemble"]

    code, out = assembly_ext.handle_post(f"/assembly/job/{job['id']}/review", {"action": "approve"})
    assert code == 409 and out["blocked"] == ["bracket"]
    assert "skip" in out["error"]
    assert assembly_ext.status(job["id"])["assembly"] is None

    code, out = assembly_ext.handle_post(f"/assembly/job/{job['id']}/review",
                                         {"action": "approve", "skip": ["bracket"]})
    assert code == 200 and out["status"] == "done"
    assert out["skipped"] == ["bracket"] and out["assembly"] is not None
    assert any("skipped bracket" in e["message"] for e in out["events"])


def test_a_redesign_answers_202_at_once_and_rebuilds_only_that_part_in_the_background():
    calls = []
    job = _run(_slow(_designer(calls=calls), 0.3))
    assert len(calls) == 2                                 # frame + bracket; the standard bolt is bought

    t0 = time.time()
    code, out = assembly_ext.handle_post(f"/assembly/job/{job['id']}/review",
                                         {"action": "redesign", "part_id": "bracket",
                                          "changes": {"dims": {"width_mm": 50}}})
    answered_in = time.time() - t0
    assert code == 202 and out["status"] == "redesigning"
    assert answered_in < 1.0, "the redesign must not run inside the request"
    assert out["assembly"] is None                         # still at the gate
    assert "poll GET" in out["how_to_continue"]
    assert len(calls) == 2                                 # the design had not even started yet, or just did

    out = _wait(job["id"], assembly_ext.REVIEW_STATUS)
    assert len(calls) == 3                                 # one part again, not the whole plan
    assert calls[-1][1]["assembly_part"] == "bracket"
    assert calls[-1][1]["part_spec"]["dims"]["width_mm"] == 50.0
    assert calls[-1][1]["part_spec"]["numbers"][1]["source"] == "you"
    rows = {r["id"]: r for r in out["review"]["parts"]}
    assert rows["bracket"]["dims"][1]["value"] == 50.0 and rows["bracket"]["dims"][1]["source"] == "you"
    assert out["review"]["redesigns"][0]["part_id"] == "bracket"
    assert out["assembly"] is None
    assert job["review_since"] is not None                 # the 24 h clock restarts at the gate

    code, out = assembly_ext.handle_post(f"/assembly/job/{job['id']}/review", {"action": "approve"})
    assert code == 200 and out["assembly"] is not None


def test_a_redesign_waits_for_a_free_slot_like_any_other_design():
    """Review WAJIB 3: a hand redesign is a design, so DEFAULT_MAX_JOBS = 1 covers it too."""
    calls = []
    job = _run(_designer(calls=calls))
    held = assembly_ext.max_jobs()
    for _ in range(held):
        assembly_ext._acquire_slot()                       # another assembly holds every slot
    try:
        code, out = assembly_ext.handle_post(f"/assembly/job/{job['id']}/review",
                                             {"action": "redesign", "part_id": "bracket",
                                              "changes": {"length_mm": 70}})
        assert code == 202 and out["status"] == "redesigning"
        time.sleep(0.3)
        assert assembly_ext.status(job["id"])["status"] == "redesigning"
        assert len(calls) == 2, "a redesign must not design while every slot is taken"
    finally:
        for _ in range(held):
            assembly_ext._release_slot()
    out = _wait(job["id"], assembly_ext.REVIEW_STATUS)
    assert len(calls) == 3 and out["review"]["redesigns"][0]["changes"] == {"length_mm": 70.0}


def test_a_redesign_needs_a_number_and_a_source_the_engineer_stands_behind():
    job = _run(_designer())
    url = f"/assembly/job/{job['id']}/review"
    code, out = assembly_ext.handle_post(url, {"action": "redesign", "part_id": "bracket",
                                               "changes": {"width_mm": "wide"}})
    assert code == 400 and "not a number" in out["error"]
    code, out = assembly_ext.handle_post(url, {"action": "redesign", "part_id": "bracket",
                                               "changes": {"dims": {"width_mm": 50}, "source": "table: ISO 2768"}})
    assert code == 400 and "\"you\"" in out["error"]
    code, out = assembly_ext.handle_post(url, {"action": "redesign", "part_id": "bracket", "changes": {}})
    assert code == 400 and "at least one size" in out["error"]
    code, out = assembly_ext.handle_post(url, {"action": "redesign", "part_id": "bolt_m6",
                                               "changes": {"length_mm": 20}})
    assert code == 400 and "standard part" in out["error"]
    assert assembly_ext.status(job["id"])["status"] == "awaiting_review"   # nothing moved


def test_the_gate_is_a_known_route_and_only_answers_at_the_gate():
    assert http_ext.known_post("/assembly/job/abc123/review")
    assert not http_ext.known_post("/assembly/job/abc123/approve")
    job = _run(_designer())
    url = f"/assembly/job/{job['id']}/review"
    assert assembly_ext.handle_post(url, {"action": "approve"})[0] == 200
    code, out = assembly_ext.handle_post(url, {"action": "approve"})
    assert code == 409 and "not awaiting_review" in out["error"]
    assert assembly_ext.handle_post("/assembly/job/nope/review", {"action": "approve"})[0] == 404
