"""Bagian E on /chat (branch `solver/chat-assembly-v2`): "whole bicycle" becomes a plan of parts, and nothing is
designed until the engineer answers Start?.

The assembly route is faked here: these tests pin what the chat does with its answer — the plan, the questions and
the estimate on the first message, the job only on `context.assembly_confirm: true`, and every guard the route
already has (a size with no source, a full queue, a spent AI budget) answered instead of bypassed. No AI is called
and no part is designed.
"""
import json
import os

import ai_budget
import assembly_ext
import assembly_plan
import cad_router
import main

BIKE = {
    "parts": [
        {"id": "frame", "name": "Frame", "role": "carries the rider", "qty": 1, "needs": ["tube diameter"],
         "standard": False},
        {"id": "fork", "name": "Fork", "role": "holds the front wheel", "qty": 1, "needs": ["steerer diameter"],
         "standard": False},
        {"id": "wheel", "name": "Wheel", "role": "rolls", "qty": 2, "needs": ["rim diameter"], "standard": False},
        {"id": "bolt_m6", "name": "M6 bolt", "role": "fastens", "qty": 8, "needs": [], "standard": True},
    ],
    "interfaces": [
        {"between": ["frame", "fork"], "kind": "press", "params": {"steerer_d_mm": 44.5},
         "source": "standard: ISO 5775 BSD 622 mm"},
    ],
    "frames": [{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "fork", "origin_mm": [400, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "wheel", "origin_mm": [400, 0, -350], "axis": [1, 0, 0]},
               {"part_id": "bolt_m6", "origin_mm": [0, 40, 0], "axis": [0, 1, 0]}],
    "sources": {"steerer_d_mm": "standard: ISO 5775"},
    "questions": [],
}

# the same product, with one interface size that names no source at all
UNSOURCED = {**BIKE, "sources": {},
             "interfaces": [{"between": ["frame", "fork"], "kind": "press", "params": {"steerer_d_mm": 44.5}}]}


def _no_budget(monkeypatch):
    monkeypatch.setattr(ai_budget, "gate", lambda *a, **k: None)


def _fake_route(monkeypatch, plan=None, design=None):
    """The assembly routes without an AI call and without a job. -> the list of (base, payload) calls."""
    plan = plan if plan is not None else assembly_plan.validate(BIKE)[0]
    calls = []

    def fake(base, params):
        calls.append((base, params))
        if base == "/assembly/plan":
            return 200, {"success": True, "plan": plan, "errors": [],
                         "estimate": assembly_plan.estimate(plan), "questions": plan.get("questions") or []}
        if base == "/assembly/design":
            if design is not None:
                return design
            chosen = params.get("plan") if isinstance(params.get("plan"), dict) else plan
            return 200, {"success": True, "job_id": "job123", "status": "queued", "plan": chosen,
                         "estimate": assembly_plan.estimate(chosen), "poll": "/assembly/design/job123"}
        return None

    monkeypatch.setattr(assembly_ext, "handle_post", fake)
    return calls


# ------------------------------------------------------------------ E1: the first message never starts a job
def test_the_first_message_asks_and_starts_nothing(monkeypatch):
    _no_budget(monkeypatch)
    calls = _fake_route(monkeypatch)
    out = main.Handler._chat_assembly(None, "design whole bicycle 170 cm, 700c", None)
    assert out["success"] and out["intent"] == "assembly"
    assert [b for b, _ in calls] == ["/assembly/plan"]           # only the planner was asked
    a = out["assembly"]
    assert a["job_id"] is None and a["poll"] is None
    assert a["estimate"]["parts_to_design"] == 3 and a["estimate"]["standard_parts"] == 1
    assert "Start?" in out["answer"] and "minute" in out["answer"]
    assert a["not_checked"], "the assembly as one system is never claimed"
    assert a["plan"]["parts"][0]["id"] == "frame"


def test_the_answer_gives_the_counts_and_the_estimated_time(monkeypatch):
    _no_budget(monkeypatch)
    _fake_route(monkeypatch)
    out = main.Handler._chat_assembly(None, "whole bicycle", None)
    est = out["assembly"]["estimate"]
    assert f"{est['parts_total']} parts" in out["answer"]
    assert f"{est['parts_to_design']} to design" in out["answer"]
    assert f"{est['standard_parts']} standard parts" in out["answer"]


# ------------------------------------------------------------------ E2: the job needs the engineer's yes
def test_a_plan_sent_back_without_the_yes_still_starts_nothing(monkeypatch):
    _no_budget(monkeypatch)
    calls = _fake_route(monkeypatch)
    out = main.Handler._chat_assembly(None, "here is the plan",
                                      {"assembly_plan": BIKE, "assembly_request": "whole bicycle"})
    assert calls == []                                           # no AI call, no route call, no job
    assert out["assembly"]["job_id"] is None and "Start?" in out["answer"]
    assert out["assembly"]["plan"]["parts"][2]["qty"] == 2


def test_the_yes_starts_the_job_through_the_assembly_route(monkeypatch):
    _no_budget(monkeypatch)
    calls = _fake_route(monkeypatch)
    out = main.Handler._chat_assembly(None, "yes, start it",
                                      {"assembly_plan": BIKE, "assembly_confirm": True,
                                       "assembly_request": "whole bicycle 170 cm"})
    assert [b for b, _ in calls] == ["/assembly/design"]
    base, payload = calls[0]
    assert payload["request"] == "whole bicycle 170 cm", "the request, not the confirmation sentence"
    assert payload["plan"] == BIKE
    a = out["assembly"]
    assert a["job_id"] == "job123" and a["poll"] == "/assembly/design/job123"
    assert a["eta_s"] > 0 and "started" in out["answer"]


def test_the_yes_may_arrive_as_a_string(monkeypatch):
    _no_budget(monkeypatch)
    calls = _fake_route(monkeypatch)
    out = main.Handler._chat_assembly(None, "start", {"assembly_plan": BIKE, "assembly_confirm": "true"})
    assert out["assembly"]["job_id"] == "job123" and calls[0][0] == "/assembly/design"


def test_a_broken_plan_is_explained_and_starts_nothing():
    # the real route validates the plan (no AI, no job): a frame for a part that is not in the plan is refused
    bad = {**BIKE, "frames": [{"part_id": "saddle", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]}]}
    out = main.Handler._chat_assembly(None, "go", {"assembly_plan": bad, "assembly_confirm": True})
    assert out["assembly"]["job_id"] is None
    assert out["assembly"]["errors"] and "saddle" in out["assembly"]["errors"][0]
    assert "no job was started" in out["answer"]


# ------------------------------------------------------------------ E3: the guards the route already has
def test_a_size_without_a_source_holds_the_job_back(monkeypatch):
    _no_budget(monkeypatch)
    plan, errs = assembly_plan.validate(UNSOURCED)
    assert not errs and plan["unsourced"], "the route's own reading of that plan"
    need = (200, {"success": True, "need": "confirmation", "plan": plan, "unsourced": plan["unsourced"],
                  "questions": plan["questions"],
                  "how_to_continue": "give a source for each size, or confirm the ids you accept"})
    calls = _fake_route(monkeypatch, plan=plan, design=need)
    out = main.Handler._chat_assembly(None, "go", {"assembly_plan": UNSOURCED, "assembly_confirm": True})
    assert calls[0][0] == "/assembly/design"
    a = out["assembly"]
    assert a["job_id"] is None and a["unsourced"] == plan["unsourced"]
    assert "no source" in out["answer"] and "Confirm steerer_d_mm" in out["answer"]


def test_a_full_queue_is_answered_and_not_queued(monkeypatch):
    _no_budget(monkeypatch)
    busy = (503, {"success": False, "error": "The assembly queue is full. Please retry in a minute.",
                  "running": 1, "max_jobs": 1, "queued": 2})
    _fake_route(monkeypatch, design=busy)
    out = main.Handler._chat_assembly(None, "go", {"assembly_plan": BIKE, "assembly_confirm": True})
    a = out["assembly"]
    assert a["job_id"] is None and a["poll"] is None
    assert "Another assembly is being designed" in out["answer"] and "Try again in a few minutes" in out["answer"]
    assert a["busy"]["max_jobs"] == 1


def test_a_spent_ai_budget_writes_no_plan_and_starts_no_job(monkeypatch):
    monkeypatch.setattr(ai_budget, "gate",
                        lambda *a, **k: "Daily AI budget reached (300 of 300 AI calls today, IDRIS_AI_CALLS_DAILY_CAP).")
    calls = _fake_route(monkeypatch)
    out = main.Handler._chat_assembly(None, "design whole bicycle", None)
    assert calls == []                                           # not one AI call was made
    assert out["assembly"]["job_id"] is None and out["assembly"]["budget"]
    assert "Daily AI budget reached" in out["answer"]


def test_a_failed_plan_says_the_real_reason(monkeypatch):
    _no_budget(monkeypatch)
    _fake_route(monkeypatch, design=(422, {"success": False, "error": "the plan has no usable part",
                                           "errors": ["the plan has no usable part"]}))
    out = main.Handler._chat_assembly(None, "go", {"assembly_plan": BIKE, "assembly_confirm": True})
    assert out["assembly"]["job_id"] is None
    assert "the plan has no usable part" in out["answer"]


# ------------------------------------------------------------------ E4: the router and the order in /chat
def test_the_router_knows_the_assembly_route(monkeypatch):
    assert "assembly" in cad_router.ROUTES
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("IDRIS_AI_ROUTER", raising=False)
    seen = {}

    def fake(prov, system, user, max_tokens=500, timeout=None):
        seen["system"] = system
        return json.dumps({"route": "assembly", "reason": "whole product"}), "fake-model"

    monkeypatch.setattr(cad_router.llm, "call_provider", fake)
    r = cad_router.route("design whole bicycle 170 cm, 700c", provider="claude")
    assert r["route"] == "assembly"
    system = seen["system"]
    assert '"assembly"' in system and "whole" in system and "seluruh" in system
    assert "frame on its own is free_design" in system or "bicycle frame 540 mm" in system


def test_one_part_of_a_product_stays_a_design(monkeypatch):
    # the AI is faked: this pins the plumbing (a free_design answer is not turned into an assembly), not the
    # model's own reading, which no test here can prove
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("IDRIS_AI_ROUTER", raising=False)
    monkeypatch.setattr(cad_router.llm, "call_provider",
                        lambda *a, **k: (json.dumps({"route": "free_design", "reason": "one part"}), "fake-model"))
    assert cad_router.route("bicycle frame 540 mm", provider="claude")["route"] == "free_design"


def test_the_chat_reads_the_plan_before_it_routes_thesentence():
    src = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
    chat = src[src.index('if base == "/chat":'):]
    assert chat.index('ctx.get("assembly_plan")') < chat.index("cad_router.route(")
    assert chat.index('routed.get("route") == "assembly"') < chat.index("self._chat_cad(msg, ctx")
    assert "_assembly_confirming(" in chat, "a stale plan may only answer a short yes, not every sentence"


def test_a_short_yes_reads_a_pending_plan_and_a_new_sentence_does_not():
    assert main._assembly_confirming("yes, start it", BIKE) is True
    assert main._assembly_confirming("lanjut", BIKE) is True
    assert main._assembly_confirming("make the frame lighter and use 6061", BIKE) is False
    pending = {**BIKE, "questions": ["Which wheel size?"]}
    assert main._assembly_confirming("which wheel size? 700c", pending) is True


def test_the_assembly_routes_reach_the_chat_through_http_ext():
    import http_ext
    assert http_ext.known_post("/assembly/plan") and http_ext.known_post("/assembly/design")
    assert http_ext.calls_ai("/assembly/plan") and http_ext.calls_ai("/assembly/design")
