"""/chat discussion mode (UI v19 `context`): actions are validated, invented numbers are flagged."""
import json
import os
import llm

CTX = {"advice": {"topic": "thermal", "status": "FAIL", "oil_temp_c": 237.8, "limit_c": 90,
                  "options": [{"label": "Oil cooler 400 W", "change": {"oil_cooler_w": 400}, "oil_temp_c": 60.3}]},
       "pending_question": "envBasis"}


def _with_reply(reply):
    os.environ["GEMINI_API_KEY"] = "test"
    os.environ["IDRIS_LLM_ORDER"] = "gemini"
    llm.call_provider = lambda prov, system, user, max_tokens=500: (json.dumps(reply), "fake")


def test_actions_are_sanitized_and_numbers_checked():
    _with_reply({"intent": "discuss", "language": "id",
                 "answer": "Oli 237,8 °C di atas 90 °C; oil cooler 400 W memberi 60,3 °C. Mungkin 120 °C dengan sirip.",
                 "actions": [{"type": "run_analysis", "analysis": "thermal", "options": {"oil_cooler_w": 400, "x": 1}},
                             {"type": "delete_everything"},
                             {"type": "run_analysis", "analysis": "thermal", "options": {"air_speed_ms": 999}}]})
    r = llm.interpret(None, "coba oil cooler", None, [], "B", CTX)
    assert r["mode"] == "discussion" and r["intent"] == "discuss"
    assert r["actions"][0] == {"type": "run_analysis", "analysis": "thermal", "options": {"oil_cooler_w": 400.0}}
    assert all(a["type"] != "delete_everything" for a in r["actions"])
    assert "run_analysis.options.air_speed_ms" in r["rejected"]
    assert r["unverified_numbers"] == [120.0] and not r["numbers_verified"] and "120" in r["answer"]


def test_envelope_answer_works_without_ai():
    def down(*a, **k):
        raise llm.LLMError("down", 503)
    os.environ["GEMINI_API_KEY"] = "test"
    llm.call_provider = down
    r = llm.interpret(None, "hitung sampai ujung poros", None, [], None, {"pending_question": "envBasis"})
    assert r["provider"] == "rules" and r["actions"] == [{"type": "set_envelope_basis", "basis": "overall"}]
    r = llm.interpret(None, "housing saja", None, [], None, {"pending_question": "envBasis"})
    assert r["actions"][0]["basis"] == "housing"


def test_old_mode_unchanged():
    _with_reply({"intent": "set_params", "params": {"power_kw": 11.19, "gear_ratio": 99}, "answer": "", "language": "en"})
    r = llm.interpret(None, "15 hp ratio 99", None, None, None)
    assert r["params"] == {"power_kw": 11.19} and "gear_ratio" in r["rejected"] and "actions" not in r
