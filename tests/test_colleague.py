"""The colleague layer (v0.41): a message that asks no solver is answered by an engineer (doc: chat rekan engineer).

Every test here is offline: the AI is a fake that returns exactly the JSON a model would, so the rules that matter
(language, stage, at most three questions, no number the solver never gave, no gearbox template) are tested, not the
model.
"""
import json

import pytest

import brief
import colleague
import llm


SNOW_PLOW = {"brief": {"object": "snow plow", "purpose": "clear snow from a pickup", "key_sizes": "2.4 m blade",
                       "material": "S355", "ready": True, "assumptions": [], "decided": []},
             "references": {"mode": "solver"}}
ID_HISTORY = [{"role": "user", "text": "aku mau bikin snow plow buat pikap"},
              {"role": "assistant", "text": "Oke, bilahnya selebar apa dan untuk kendaraan apa?"}]


def _ai(monkeypatch, payload):
    """Point the layer at one fake provider whose answer is `payload` (dict or list of dicts, one per call)."""
    monkeypatch.setenv("IDRIS_LLM_ORDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def fake(provider, system, user, max_tokens=500, timeout=None):
        calls.append({"system": system, "user": user})
        item = payload[min(len(calls) - 1, len(payload) - 1)] if isinstance(payload, list) else payload
        if isinstance(item, Exception):
            raise item
        return json.dumps(item, ensure_ascii=False), "fake-model"

    monkeypatch.setattr(llm, "call_provider", fake)
    return calls


# ------------------------------------------------------------------ A7: the snow plow case
def test_cuy_on_a_snow_plow_project_talks_about_the_snow_plow(monkeypatch):
    _ai(monkeypatch, {"reply": "Snow plow-nya masih di tahap konsep. Dudukan ke pikap belum kita tentukan.",
                      "language": "id", "suggestions": [{"label": "Sesuaikan dudukan ke kendaraanmu"},
                                                        {"label": "Tambah mekanisme angkat"},
                                                        {"label": "Cek sambungan las"}],
                      "questions": [], "action": None, "stage": 3})
    r = colleague.ask(None, "cuy", ID_HISTORY, SNOW_PLOW)
    text = json.dumps(r, ensure_ascii=False).lower()
    assert "snow plow" in text
    assert r["language"] == "id"
    assert len(r["suggestions"]) >= 2
    assert "gear ratio" not in text and "rpm" not in text


def test_the_session_is_sent_to_the_model_with_the_project_status(monkeypatch):
    calls = _ai(monkeypatch, {"reply": "Masih di snow plow.", "language": "id", "suggestions": [], "questions": []})
    colleague.ask(None, "cuy", ID_HISTORY, SNOW_PLOW)
    user = calls[0]["user"]
    assert "snow plow" in user and "buat pikap" in user
    assert "PROJECT" in user and "SESSION" in user and "MESSAGE TO ANSWER" in user


def test_the_history_keeps_twenty_messages_and_summarises_the_rest():
    rows = [{"role": "user", "text": f"pesan {i}"} for i in range(30)]
    text, summary = colleague.history_block(rows)
    assert "pesan 29" in text and "pesan 10" in text and "pesan 9" not in text
    assert "10 older message" in summary


def test_when_the_ai_is_away_the_answer_is_still_the_project_not_a_template():
    # no provider configured at all: the rules write the status line and the next steps
    r = colleague.ask(None, "cuy", ID_HISTORY, SNOW_PLOW, reply_language="id")
    assert "snow plow" in r["reply"].lower()
    assert len(r["suggestions"]) >= 2
    low = json.dumps(r, ensure_ascii=False).lower()
    assert "gear ratio" not in low and "rpm" not in low
    assert r["provider"].startswith("IDRIS colleague")


def test_a_greeting_with_no_project_offers_a_start_and_never_a_gearbox():
    r = colleague.ask(None, "hi", [], None)
    low = r["reply"].lower()
    assert "no project is open" in low
    assert "gearbox" not in low and "gear ratio" not in low
    assert r["language"] == "en" and len(r["suggestions"]) >= 2


def test_lanjut_on_an_open_gearbox_offers_gearbox_steps(monkeypatch):
    ctx = {"current": {"power_kw": 15, "input_rpm": 1450, "gear_ratio": 8}}
    variants = [{"name": "Variant A", "mass_kg": 41.2, "stress": {"sf_bending": 1.4}}]
    _ai(monkeypatch, {"reply": "Gearbox 15 kW masih di Variant A; cek kekuatan sudah jalan.",
                      "language": "id", "suggestions": [{"label": "Bandingkan varian dari faktor aman"}],
                      "questions": [], "action": None, "stage": 7})
    r = colleague.ask(None, "lanjut", [], ctx, current=ctx["current"], variants=variants)
    assert r["stage"] >= 4
    assert len(r["suggestions"]) >= 2
    assert r["language"] == "id"


# ------------------------------------------------------------------ A3: language
@pytest.mark.parametrize("message,expected", [("ok", None), ("cuy", None), ("123", None), ("😀", None),
                                              ("buatkan aku dudukan", "id"), ("make it lighter", "en"),
                                              ("中文 请继续", "zh"), ("продолжай", "ru")])
def test_the_language_of_a_message(message, expected):
    assert colleague.detect_language(message) == expected


def test_an_unclear_message_takes_the_previous_clear_language():
    assert colleague.language(ID_HISTORY, "ok") == "id"
    assert colleague.language([], "ok") == "en"
    assert colleague.language([{"role": "user", "text": "hello there, how are you"}], "cuy") == "en"


def test_a_chinese_message_gets_a_chinese_reply(monkeypatch):
    _ai(monkeypatch, {"reply": "我们还在扫雪铲的概念阶段。要不要先确定挂架？", "language": "zh",
                      "suggestions": [{"label": "确定挂架"}, {"label": "检查焊接"}], "questions": []})
    r = colleague.ask(None, "接着做", ID_HISTORY, SNOW_PLOW)
    assert r["language"] == "zh"
    assert "扫雪铲" in r["reply"]


def test_an_english_message_gets_an_english_reply(monkeypatch):
    _ai(monkeypatch, {"reply": "The snow plow is still at the concept stage.", "language": "en",
                      "suggestions": [{"label": "Fit the mount to your truck"}, {"label": "Check the welds"}]})
    r = colleague.ask(None, "hey, continue", [], SNOW_PLOW)
    assert r["language"] == "en"


# ------------------------------------------------------------------ A4b: questions per stage
def _q(qid, stage, n=2):
    return {"id": qid, "text": f"pertanyaan {qid}", "stage": stage,
            "options": [{"label": f"opsi {i}", "note": "", "recommended": i == 0} for i in range(n)]}


def test_only_the_current_stage_is_asked_and_three_max_on_stage_one(monkeypatch):
    questions = [_q("q1", 1), _q("q2", 1), _q("q3", 1), _q("q4", 1), _q("q5", 5), _q("q6", 7)]
    _ai(monkeypatch, {"reply": "Oke.", "language": "id", "suggestions": [], "questions": questions})
    ctx = {"brief": {"object": "sepeda gunung", "ready": False}}
    r = colleague.ask(None, "sepeda gunung", [], ctx, reply_language="id")
    assert r["stage"] == 1
    assert len(r["questions"]) == 3
    assert all(q["stage"] == 1 for q in r["questions"])
    assert [q["id"] for q in r["questions"]] == ["q1", "q2", "q3"]


def test_a_later_stage_gets_at_most_two_questions(monkeypatch):
    ctx = {"brief": {"object": "snow plow", "ready": True}, "references": {"mode": "solver"},
           "cad": {"title": "plow mount"}}
    _ai(monkeypatch, {"reply": "Cek rakitan dulu.", "language": "id",
                      "questions": [_q("a", 5), _q("b", 5), _q("c", 5)], "suggestions": []})
    r = colleague.ask(None, "lanjut", [], ctx, reply_language="id")
    assert r["stage"] == 5
    assert len(r["questions"]) == 2


def test_every_question_offers_a_recommended_option_and_not_sure(monkeypatch):
    _ai(monkeypatch, {"reply": "Oke.", "language": "en",
                      "questions": [{"id": "q1", "text": "Which process?", "stage": 1,
                                     "options": [{"label": "Laser cut"}, {"label": "Plasma cut"}]}],
                      "suggestions": []})
    r = colleague.ask(None, "bike frame", [], {"brief": {"object": "bike frame"}})
    opts = r["questions"][0]["options"]
    assert any(o["recommended"] for o in opts)
    assert any("Not sure" in o["label"] for o in opts)


def test_a_question_already_answered_is_not_asked_again(monkeypatch):
    _ai(monkeypatch, {"reply": "Oke.", "language": "id",
                      "questions": [_q("q1", 1), _q("q2", 1)], "suggestions": []})
    ctx = {"brief": {"object": "sepeda gunung", "purpose": "trail"}, "answered": [{"id": "q1"}]}
    r = colleague.ask(None, "lanjut", [], ctx, reply_language="id")
    assert [q["id"] for q in r["questions"]] == ["q2"]


def test_the_brief_questions_carry_stage_one_and_never_more_than_three():
    b = brief.blank("part")
    q = brief.fallback_questions(b, ["object", "purpose", "sizes", "loads"])
    assert len(q["questions"]) == 3
    assert all(x.get("stage") == 1 for x in q["questions"])


# ------------------------------------------------------------------ A5: numbers
def test_a_number_the_solver_never_gave_is_called_out(monkeypatch):
    _ai(monkeypatch, {"reply": "Dudukannya aman dengan SF 3.4 kg dan las 6 mm.", "language": "id",
                      "suggestions": [], "questions": []})
    r = colleague.ask(None, "cuy", ID_HISTORY, SNOW_PLOW)
    assert r["numbers_verified"] is False
    assert "belum berasal dari solver" in r["reply"]


def test_a_number_the_solver_gave_is_accepted(monkeypatch):
    _ai(monkeypatch, {"reply": "Bilahnya 2.4 m, seperti yang kamu tulis.", "language": "id",
                      "suggestions": [], "questions": []})
    r = colleague.ask(None, "cuy", ID_HISTORY, SNOW_PLOW)
    assert r["numbers_verified"] is True


# ------------------------------------------------------------------ A6: the response shape
def test_the_response_has_the_agreed_keys(monkeypatch):
    _ai(monkeypatch, {"reply": "Oke.", "language": "id", "suggestions": [], "questions": []})
    out = colleague.answer(None, "cuy", ID_HISTORY, SNOW_PLOW)
    for key in ("reply", "language", "suggestions", "questions", "action", "stage", "answer"):
        assert key in out
    assert out["stage_name"] == colleague.STAGES[out["stage"] - 1]
    assert out["answer"] == out["reply"]


def test_an_unusable_ai_answer_falls_back_to_the_rules_without_raising(monkeypatch):
    _ai(monkeypatch, RuntimeError("402 payment required"))
    r = colleague.ask(None, "cuy", ID_HISTORY, SNOW_PLOW, reply_language="id")
    assert "snow plow" in r["reply"].lower()
    assert r["provider"].startswith("IDRIS colleague")
