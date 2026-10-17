"""/chat discussion mode (UI v19 `context`): actions are validated, invented numbers are flagged."""
import json
import os
import llm

ORIG_CALL_PROVIDER = llm.call_provider      # captured before the tests below swap in fakes

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


def test_q021_quantities_below_ten_are_checked():
    ctx = {"advice": {"oil_temp_c": 237.8, "options": [{"change": {"air_speed_ms": 5}}]}}
    bad = llm.unverified_numbers("SF 1,8 dan 3,4 kg, kipas 7 m/s; ada 2 opsi di tahap 1", ctx)
    assert bad == [1.8, 3.4, 7.0]
    assert llm.unverified_numbers("kipas 5 m/s, oli 237,8 °C", ctx) == []
    assert llm.unverified_numbers("biaya Rp 5 juta", ctx) == [5e6]


BAIT = [  # QA kerangka 3.11 — (question, invented reply, honest reply)
    ("Kalau sirip 20 mm, oli jadi berapa °C?", "Dengan sirip 20 mm oli sekitar 140 °C.",
     "Sirip setinggi itu belum dihitung solver; aku bisa jalankan termal dengan sirip itu."),
    ("Berapa beratnya kalau pakai oil cooler?", "Oil cooler menambah sekitar 1,2 kg.",
     "Massa oil cooler tidak ada di model solver, jadi aku tidak punya angkanya."),
    ("SF-nya berapa kalau pakai baja 4140?", "Dengan baja 4140 SF naik ke 2,3.",
     "Itu perlu dihitung ulang; aku bisa jalankan pencarian desain dengan material itu."),
    ("Kipas 15 m/s cukup tidak?", "Kipas 15 m/s membuat oli sekitar 130 °C, masih FAIL.",
     "Kecepatan itu belum dihitung; mau aku jalankan termal dengan kipas itu?"),
    ("Berapa biaya produksinya?", "Biayanya kira-kira Rp 4,5 juta per unit.",
     "Idris belum punya model biaya, jadi aku tidak bisa memberi angka biaya."),
]


def test_q018_bait_questions_flag_invented_numbers():
    for q, invented, honest in BAIT:
        _with_reply({"intent": "discuss", "language": "id", "answer": invented, "actions": []})
        r = llm.interpret(None, q, None, [], "B", CTX)
        assert r["numbers_verified"] is False and r["unverified_numbers"], (q, r)
        assert "belum berasal dari solver" in r["answer"]
        _with_reply({"intent": "discuss", "language": "id", "answer": honest, "actions": []})
        r = llm.interpret(None, q, None, [], "B", CTX)
        assert r["numbers_verified"] is True, (q, r)


def test_goals_in_set_params_are_validated():
    _with_reply({"intent": "set_params", "language": "id", "answer": "",
                 "params": {"goals": {"oil_temp_max_c": 90, "mass_max_kg": -3,
                                      "cooling_allowed": ["natural", "fins", "teleport"]}}})
    r = llm.interpret(None, "oli maks 90 C, tanpa kipas", None, None, None)
    assert r["params"]["goals"] == {"oil_temp_max_c": 90.0, "cooling_allowed": ["natural", "fins"]}
    assert "goals.mass_max_kg" in r["rejected"] and "goals.cooling_allowed" in r["rejected"]


def test_nvidia_backup_used_when_gemini_quota_is_out():
    os.environ.update(GEMINI_API_KEY="g", NVIDIA_API_KEY="n", IDRIS_LLM_ORDER="gemini,nvidia",
                      IDRIS_LLM_RETRY_SLEEP="0")
    calls = []

    def fake_one(provider, model, system, user, max_tokens=500):
        calls.append((provider, model))
        if provider == "gemini":
            raise llm.LLMError("provider returned HTTP 429: quota", http=429)
        return json.dumps({"intent": "set_params", "params": {"power_kw": 11.19}, "answer": "", "language": "en"})
    real_one = llm._call_one
    llm._call_one, llm.call_provider = fake_one, ORIG_CALL_PROVIDER
    try:
        r = llm.interpret(None, "15 hp", None, None, None)
        assert r["provider"] == "nvidia" and r["params"] == {"power_kw": 11.19}
        assert calls[-1] == ("nvidia", "nvidia/nemotron-3-super-120b-a12b")
        assert llm.models_for("nvidia") == ["nvidia/nemotron-3-super-120b-a12b", "openai/gpt-oss-20b"]
    finally:
        llm._call_one = real_one
        os.environ.pop("NVIDIA_API_KEY", None)


def test_plain_text_answer_is_kept_and_checked():
    os.environ.update(GEMINI_API_KEY="g", IDRIS_LLM_ORDER="gemini")
    llm.call_provider = lambda prov, system, user, max_tokens=500: (
        "Oil cooler tidak punya data massa di solver; mungkin sekitar 1,5 kg.", "fake")
    r = llm.interpret(None, "Berapa beratnya kalau pakai oil cooler?", None, [], "B", CTX)
    assert r["mode"] == "discussion" and r["actions"] == []
    assert r["unverified_numbers"] == [1.5] and "belum berasal dari solver" in r["answer"]
