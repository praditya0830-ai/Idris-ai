"""The trace on a /chat answer (doc: how IDRIS got here, 25 Nov 2026).

The engine is not called and no kernel runs: the route, the reference lookup and the design ladder are faked, so
these tests pin what the trace says about a turn — the order of the steps, that a step which never ran is absent,
and that a failed part carries a failed step with the ladder's own reasons.
"""
import io
import json

import cad_router
import chat_trace
import llm
import main
from research import reference_flow


class _Fake(main.Handler):
    """One POST /chat on a handler, without a socket: `sent` holds exactly what the browser would have received."""

    def __init__(self, body):
        raw = json.dumps(body).encode("utf-8")
        self.path = "/chat"
        self.client_address = ("127.0.0.1", 50000)
        self.headers = {"Content-Length": str(len(raw)), "X-Forwarded-For": "127.0.0.1"}
        self.rfile = io.BytesIO(raw)
        self.sent = []

    def _send_json(self, status, payload):
        self.sent.append((status, payload))


ROUTE = {"route": "free_design", "kind": "part", "library_id": None, "query": {}, "concepts": [],
         "reason": "a new part with sizes it can build", "read_by": "cad_router"}

BUILT = {"success": True, "answer": "4 feature steps for the bracket.", "rung": "features", "fidelity": "features",
         "design_shape_class": {"shape_class": "plate"}, "cad": {"title": "Bracket"}, "cad_attempts": []}


def _chat(monkeypatch, *, message="a bracket for the snow plow", route=None, refs=None, cad=None, concepts=None):
    monkeypatch.setattr(llm, "check_rate", lambda *a, **k: None)
    monkeypatch.setattr(cad_router, "route", lambda *a, **k: route if route is not None else dict(ROUTE))
    monkeypatch.setattr(reference_flow, "for_chat", lambda *a, **k: refs)
    monkeypatch.setattr(main.Handler, "_chat_cad", lambda self, *a, **k: cad if cad is not None else dict(BUILT))
    monkeypatch.setattr(main.Handler, "_chat_concepts",
                        lambda self, *a, **k: (concepts if concepts is not None else
                                               (dict(cad) if cad is not None else dict(BUILT))))
    h = _Fake({"message": message, "concept_count": 1})
    main.Handler.do_POST(h)
    assert h.sent, "the handler sent no answer"
    return h.sent[-1]


def _steps(answer):
    return [row["step"] for row in answer["trace"]]


def test_a_design_turn_reads_briefs_routes_and_builds_in_that_order(monkeypatch):
    status, answer = _chat(monkeypatch)
    assert status == 200
    steps = _steps(answer)
    assert steps[0] == chat_trace.STEP_READ
    assert chat_trace.STEP_BRIEF in steps
    assert steps.index(chat_trace.STEP_ROUTE) < steps.index("Built with features")
    assert steps[-1] == "Built with features"


def test_the_brief_step_says_what_was_checked(monkeypatch):
    _status, answer = _chat(monkeypatch)
    brief = next(row for row in answer["trace"] if row["step"] == chat_trace.STEP_BRIEF)
    assert "no open brief" in brief["detail"]
    assert brief["source"] == "brief.py"


def test_the_route_step_carries_the_reason_and_the_shape_class(monkeypatch):
    _status, answer = _chat(monkeypatch)
    route = next(row for row in answer["trace"] if row["step"] == chat_trace.STEP_ROUTE)
    assert "free_design" in route["detail"]
    assert "a new part with sizes it can build" in route["detail"]
    assert "shape class: plate" in route["detail"]
    assert route["source"] == "cad_router"


def test_no_reference_step_when_the_lookup_never_ran(monkeypatch):
    _status, answer = _chat(monkeypatch, refs=None)
    assert chat_trace.STEP_REFERENCES not in _steps(answer)


def test_a_reference_lookup_that_ran_is_shown_with_its_mode_and_count(monkeypatch):
    refs = {"mode": "solver", "items": [{"title": "beam bending"}], "solvers": [{"name": "beam"}]}
    _status, answer = _chat(monkeypatch, refs=refs)
    row = next(r for r in answer["trace"] if r["step"] == chat_trace.STEP_REFERENCES)
    assert 'mode "solver"' in row["detail"] and "1 item(s)" in row["detail"]
    assert row["source"] == "/references/design"


def test_a_failed_part_is_a_failed_step_with_the_ladder_reasons(monkeypatch):
    failed = {"success": True, "answer": "I could not build this part.", "rung": "failed", "cad": None,
              "design_report": {"part_name": "bracket", "shape_class": "plate",
                                "rungs_tried": [{"rung": "features", "error": "no template fits"},
                                                {"rung": "code", "error": "the code did not build"}],
                                "suggestion": "give the plate thickness"}}
    _status, answer = _chat(monkeypatch, cad=failed)
    last = answer["trace"][-1]
    assert last["status"] == "failed"
    assert last["step"] == "Build failed"
    assert "no template fits" in last["detail"] and "the code did not build" in last["detail"]


def test_a_reply_that_is_not_a_design_still_carries_the_read_and_route(monkeypatch):
    talk = {"route": "other", "kind": None, "library_id": None, "query": {}, "concepts": [],
            "reason": "a greeting", "read_by": "cad_router"}
    monkeypatch.setattr(llm, "discuss", lambda *a, **k: {"success": True, "answer": "Hello."})
    status, answer = _chat(monkeypatch, message="hi", route=talk)
    assert status == 200
    steps = _steps(answer)
    assert steps[0] == chat_trace.STEP_READ and chat_trace.STEP_ROUTE in steps
    assert "Built with features" not in steps and "Build failed" not in steps


def test_the_trace_is_capped_and_rows_are_plain_json():
    trace = chat_trace.Trace()
    for i in range(30):
        trace.add(f"step {i}", "x" * 900, "src")
    rows = trace.rows()
    assert len(rows) == chat_trace.MAX_STEPS
    assert rows[-1]["step"] == "step 29"
    assert all(len(row["detail"]) <= chat_trace.MAX_DETAIL_CHARS for row in rows)
    json.dumps(rows)


def test_update_never_invents_a_step_that_did_not_run():
    trace = chat_trace.Trace()
    assert trace.update(chat_trace.STEP_ROUTE, append=" · x") is None
    assert trace.rows() == []
    trace.add(chat_trace.STEP_ROUTE, "route")
    trace.update(chat_trace.STEP_ROUTE, append=" · shape class: plate")
    assert trace.rows()[0]["detail"] == "route · shape class: plate"
