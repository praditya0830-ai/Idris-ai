"""Review of PR #5-#8, item 7: the AI money is counted per paid call, not per HTTP request.

No network and no key: `llm._http` (the one function that talks to a provider) is replaced, so these tests
watch the gate and the counter, never a model. A provider call that the gate refuses must not happen at all,
which is why the fake records every attempt.
"""
import json
import os

import pytest

import ai_budget
import assembly_ext
import assembly_plan
import design_job
import http_ext
import llm

OPENAI = "https://api.openai.com/v1/chat/completions"
DEEPSEEK = "https://api.deepseek.com/chat/completions"
ANT = "https://api.anthropic.com/v1/messages"


def _openai_reply(text="{}", usage=None):
    return {"choices": [{"message": {"content": text}}], "usage": usage or {}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ai_budget.reset()
    monkeypatch.setenv("IDRIS_LLM_RETRY_SLEEP", "0")       # no waiting between provider attempts in a test
    yield
    ai_budget.reset()


# ------------------------------------------------------------------ the counter
def test_every_paid_call_is_counted_and_priced(monkeypatch):
    monkeypatch.setattr(llm, "_http", lambda req, timeout: {
        "content": [{"type": "text", "text": "{}"}],
        "usage": {"input_tokens": 1_000_000, "output_tokens": 0}})
    llm._post(ANT, {}, {"model": "claude-haiku-5-5"})
    u = ai_budget.usage()
    assert u["calls_today"] == 1
    assert u["tokens_today"] == {"input": 1_000_000, "output": 0}
    assert u["estimated_cost_usd_today"] == pytest.approx(ai_budget.PRICES["claude"][0])


def test_a_gemini_reply_is_read_with_its_own_usage_shape(monkeypatch):
    monkeypatch.setattr(llm, "_http", lambda req, timeout: {
        "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
        "usageMetadata": {"promptTokenCount": 2000, "candidatesTokenCount": 500}})
    llm._post("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent", {},
              {})
    assert ai_budget.tokens({"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}) == (7, 3)
    u = ai_budget.usage()
    assert u["tokens_today"] == {"input": 2000, "output": 500}
    assert u["estimated_cost_usd_today"] == pytest.approx((2000 * 0.30 + 500 * 2.50) / 1e6, abs=5e-5)


# ------------------------------------------------------------------ the gate
def test_the_daily_call_cap_stops_the_provider(monkeypatch):
    monkeypatch.setenv("IDRIS_AI_CALLS_DAILY_CAP", "2")
    seen = []

    def http(req, timeout):
        seen.append(req.full_url)
        return _openai_reply()

    monkeypatch.setattr(llm, "_http", http)
    for _ in range(2):
        llm._post(OPENAI, {}, {"model": "gpt-4o-mini"})
    with pytest.raises(llm.LLMError) as e:
        llm._post(OPENAI, {}, {"model": "gpt-4o-mini"})
    assert e.value.status == 429
    assert "Daily AI budget reached" in str(e.value)
    assert len(seen) == 2, "no provider is called once the cap is reached"


def test_the_monthly_budget_stops_the_provider(monkeypatch):
    monkeypatch.setenv("IDRIS_AI_MONTHLY_BUDGET_USD", "0.01")
    monkeypatch.setattr(llm, "_http", lambda req, timeout: _openai_reply(
        usage={"prompt_tokens": 100_000, "completion_tokens": 0}))
    llm._post(DEEPSEEK, {}, {"model": "deepseek-flash"})
    assert ai_budget.usage()["estimated_cost_usd_month"] > 0.01
    with pytest.raises(llm.LLMError) as e:
        llm._post(DEEPSEEK, {}, {"model": "deepseek-flash"})
    assert "Monthly AI budget reached" in str(e.value)


def test_a_failed_call_is_still_counted(monkeypatch):
    def boom(req, timeout):
        raise TimeoutError("no route to host")

    monkeypatch.setattr(llm, "_http", boom)
    with pytest.raises(llm.LLMError):
        llm._post(OPENAI, {}, {"model": "gpt-4o-mini"})
    assert ai_budget.usage()["calls_today"] == 1, "a call that failed still cost a request"


# ------------------------------------------------------------------ the routes
def test_the_routes_that_can_spend_ai_are_the_ones_the_gate_names():
    assert http_ext.calls_ai("/assembly/plan") and http_ext.calls_ai("/assembly/design")
    assert http_ext.calls_ai("/shape/families")
    assert not http_ext.calls_ai("/report/part"), "a report copies numbers; it calls no AI"
    assert not http_ext.calls_ai("/shape/rank"), "ranking measured values calls no AI"


def test_main_rate_limits_those_routes_before_it_starts_the_work():
    src = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
    post = src[src.index("def do_POST"):]
    marker = 'base == "/design/job" or http_ext.calls_ai(base)'
    assert marker in post
    gate = post[post.index(marker):]
    assert "llm.check_rate(" in gate[:500]
    assert "429" in gate[:500]
    assert post.index(marker) < post.index("res = http_ext.handle_post(base, params)")


def test_the_usage_reply_carries_no_key_and_no_user_data(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-appear-123456")
    u = ai_budget.usage()
    assert "sk-should-never-appear" not in json.dumps(u)
    assert {"calls_today", "calls_today_left", "tokens_today", "estimated_cost_usd_today",
            "estimated_cost_usd_month", "monthly_budget_left_usd", "blocked"} <= set(u)
    assert u["price_source"] and u["price_assumed"]
    assert u["blocked"] is None


# ------------------------------------------------------------------ the jobs
PLAN = {
    "parts": [{"id": "frame", "name": "Frame", "role": "carries", "qty": 1, "needs": [], "standard": False},
              {"id": "fork", "name": "Fork", "role": "holds", "qty": 1, "needs": [], "standard": False}],
    "interfaces": [{"between": ["frame", "fork"], "kind": "press", "params": {"d_mm": 44.5},
                    "source": "standard: ISO 5775"}],
    "frames": [{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]},
               {"part_id": "fork", "origin_mm": [1, 0, 0], "axis": [0, 0, 1]}],
    "sources": {"d_mm": "standard: ISO 5775"},
    "questions": [],
}


def test_an_assembly_job_stops_when_the_budget_is_used_up(monkeypatch):
    monkeypatch.setenv("IDRIS_AI_CALLS_DAILY_CAP", "0")
    plan, errs = assembly_plan.validate(PLAN)
    assert not errs
    asked = []
    parts, _asm = assembly_ext.design_parts(plan, designer=lambda prompt, prov=None: asked.append(prompt) or {})
    assert asked == [], "no part is sent to the AI once the day is used up"
    assert all(p["status"] == "failed" and "Daily AI budget reached" in p["error"] for p in parts)


def test_a_design_job_stops_with_the_budget_message(monkeypatch):
    monkeypatch.setenv("IDRIS_AI_CALLS_DAILY_CAP", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")

    def cad_fn(msg, ctx, prov):
        """cad_codegen turns an AI error into ok=False plus the reason; the budget is such an error."""
        try:
            llm.call_provider("claude", "system", "user", max_tokens=8)
        except llm.LLMError as e:
            return {"ok": False, "answer": str(e), "attempts": []}
        return {"ok": True, "cad": {"title": "plate", "code": "return { body };", "params": []}}

    res = design_job.run("plate 100 mm", cad_fn=cad_fn, job_id=None)
    assert res["label"] == "Not validated"
    assert "Daily AI budget reached" in res["answer"]
    assert [e for e in res["events"] if e["step"] == "code" and e["status"] == "fail"]
