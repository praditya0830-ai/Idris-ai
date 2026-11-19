"""Bagian E on /chat: "design whole bicycle" used to become one solid and a timeout. Now the sentence becomes a
plan of parts, and a job that designs the parts one at a time. These tests use a fake planner and a fake job: no AI
is called, and no part is designed.
"""
import os

import assembly_ext
import assembly_plan
import cad_router
import main

PLAN = {
    "parts": [{"id": "frame", "name": "Frame", "role": "carries the load", "qty": 1, "needs": [], "standard": False},
              {"id": "wheel", "name": "Wheel", "role": "rolls", "qty": 2, "needs": ["rim diameter"], "standard": False}],
    "interfaces": [{"between": ["frame", "wheel"], "kind": "press", "params": {"diameter_mm": 44.5},
                    "source": "standard: ISO 5775"}],
    "frames": [{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "wheel", "origin_mm": [500, 0, 0], "axis": [1, 0, 0]}],
    "sources": {"diameter_mm": "standard: ISO 5775"},
    "questions": [],
}


def _planner(monkeypatch, plan=None, ok=True, error=None, errors=None):
    """A plan without an AI call."""
    monkeypatch.setattr(assembly_plan, "plan_for",
                        lambda request, provider=None: {
                            "ok": ok, "plan": plan, "errors": errors or [],
                            "estimate": assembly_plan.estimate(plan) if plan else None, "error": error})


def _fake_job(monkeypatch, started=None):
    started = started if started is not None else []
    monkeypatch.setattr(assembly_ext, "start_job",
                        lambda request, plan, provider=None: started.append(request) or "job123")
    monkeypatch.setattr(assembly_ext, "status", lambda job_id: {"status": "queued", "eta_s": 0.0})
    return started


def test_whole_bicycle_becomes_a_plan_and_a_job(monkeypatch):
    _planner(monkeypatch, PLAN, ok=True)
    started = _fake_job(monkeypatch)
    out = main.Handler._chat_assembly(None, "design whole bicycle 170 cm, 700c", None)
    assert out["intent"] == "assembly" and out["success"]
    assert [p["id"] for p in out["assembly"]["plan"]["parts"]] == ["frame", "wheel"]
    assert out["assembly"]["job_id"] == "job123" and out["assembly"]["poll"] == "/assembly/design/job123"
    assert started == ["design whole bicycle 170 cm, 700c"]          # the job starts in this same reply
    est = out["assembly"]["estimate"]
    assert est["parts_to_design"] == 2 and est["eta_s"] > 0          # counted from the plan, not promised
    assert out["assembly"]["not_checked"]                              # the assembly as a system is not analysed


def test_a_question_holds_the_job_back(monkeypatch):
    plan = {**PLAN, "questions": ["Which wheel size?"]}
    _planner(monkeypatch, plan, ok=True)
    started = _fake_job(monkeypatch)
    out = main.Handler._chat_assembly(None, "whole bicycle", None)
    assert out["assembly"]["job_id"] is None and started == []
    assert out["assembly"]["questions"] == ["Which wheel size?"]
    assert "Which wheel size?" in out["answer"]


def test_a_plan_without_an_ai_starts_no_job(monkeypatch):
    _planner(monkeypatch, None, ok=False, error="no AI provider is configured on the server")
    started = _fake_job(monkeypatch)
    out = main.Handler._chat_assembly(None, "whole bicycle", None)
    assert out["assembly"]["job_id"] is None and started == []
    assert "no AI provider is configured" in out["answer"]           # the real reason, not a generic failure


def test_a_confirmed_plan_starts_the_job_without_asking_again(monkeypatch):
    started = _fake_job(monkeypatch)
    monkeypatch.setattr(assembly_plan, "plan_for",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no second plan may be written")))
    out = main.Handler._chat_assembly(None, "yes, start it", {"assembly_plan": PLAN, "assembly_request": "whole bicycle"})
    assert out["assembly"]["job_id"] == "job123"
    assert started == ["whole bicycle"]                              # the request, not the confirmation sentence
    assert out["assembly"]["plan"]["parts"][1]["qty"] == 2


def test_a_broken_confirmed_plan_starts_no_job(monkeypatch):
    started = _fake_job(monkeypatch)
    bad = {**PLAN, "frames": [{"part_id": "saddle", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]}]}
    out = main.Handler._chat_assembly(None, "go", {"assembly_plan": bad})
    assert out["assembly"]["job_id"] is None and started == []
    assert out["assembly"]["errors"] and "saddle" in out["assembly"]["errors"][0]


def test_the_router_can_say_assembly_and_the_chat_asks_it_first():
    assert "assembly" in cad_router.ROUTES
    src = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
    # the assembly routes must be allowed and must be answered before the /assembly/ prefix reaches engine_routes
    assert '"/assembly/plan", "/assembly/design"' in src
    post = src[src.index("def do_POST"):]
    assert post.index("assembly_ext.handle_post") < post.index('base.startswith(("/model/", "/assembly/"')
    chat = src[src.index('if base == "/chat":'):]
    assert chat.index('routed.get("route") == "assembly"') < chat.index("self._chat_cad(msg, ctx")
    # a plan sent back is recognised before the router reads the message again
    assert chat.index('ctx.get("assembly_plan")') < chat.index("cad_router.route(")
