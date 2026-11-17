"/chat has to hand a computable question to the bench, not to the CAD/library reader.

Found in a live end-to-end test: "berapa torsi baut M10 kelas 8.8?" was answered by the standard-parts
library (a screw with a length question) and the bench never ran, and a part question whose evidence block
was attached came back as HTTP 502 "the AI returned malformed data", because the structured requirements
reader was asked for JSON while the right answer was prose.
"""
import pathlib

import llm


def test_engineering_evidence_goes_to_the_discussion_path(monkeypatch):
    seen = {}

    def fake_call(provider, system, user, max_tokens=500, timeout=None):
        seen["system"] = system
        seen["user"] = user
        return '{"intent": "discuss", "answer": "SF 1,49 pada pitting.", "actions": []}', "stub-model"

    monkeypatch.setattr(llm, "call_provider", fake_call)
    res = llm.interpret("claude", "kenapa pelat ini diperingatkan?",
                        context={"engineering": "ENGINEERING EVIDENCE ...\nREVIEW WALL_THIN ..."})
    assert seen["system"] == llm.DISCUSS_SYSTEM, "evidence must be discussed, not parsed as requirements"
    assert "ENGINEERING EVIDENCE" in seen["user"]
    assert res["answer"].startswith("SF 1,49")


def test_a_message_without_evidence_still_uses_the_requirements_reader(monkeypatch):
    seen = {}

    def fake_call(provider, system, user, max_tokens=500, timeout=None):
        seen["system"] = system
        return '{"intent": "other", "answer": "ok", "actions": []}', "stub-model"

    monkeypatch.setattr(llm, "call_provider", fake_call)
    llm.interpret("claude", "15 kW gearbox ratio 8")
    assert seen["system"] == llm.SYSTEM


def test_the_bench_route_wins_over_the_library_and_cad_paths():
    src = pathlib.Path("main.py").read_text()
    assert 'eng_ctx = bool(eng and (eng.get("claims") or eng.get("missing")))' in src
    # every early return that could answer instead of the bench must be gated by eng_ctx
    assert "if not (gear_ctx or cad_ctx or eng_ctx):" in src
    assert "elif not (gear_ctx or eng_ctx or (ctx and isinstance(ctx.get(\"cad_error\"), dict))):" in src
    assert "if cad_ctx or (not gear_ctx and not eng_ctx and not is_prim and is_obj):" in src
    assert "if not gear_ctx and not eng_ctx and is_prim:" in src


def test_the_chat_response_carries_the_review_the_model_saw():
    src = pathlib.Path("main.py").read_text()
    for key in ('"advice": eng.get("advice")', '"review": eng.get("review")',
                '"review_summary": eng.get("review_summary")'):
        assert key in src
