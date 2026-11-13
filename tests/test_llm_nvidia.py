import llm


def test_nvidia_turns_off_thinking_and_reports_empty(monkeypatch):
    sent = {}

    def fake_post(url, headers, body, timeout=25.0):
        sent.update(body)
        return {"choices": [{"message": {"content": None, "reasoning_content": "thinking..."}, "finish_reason": "length"}]}
    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    monkeypatch.setattr(llm, "_post", fake_post)
    try:
        llm._call_one("nvidia", "nvidia/nemotron-3-super-120b-a12b", "s", "u", 500)
        raised = False
    except llm.LLMError as e:
        raised = "finish_reason=length" in str(e)
    assert raised
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["max_tokens"] >= 1024


def test_ping_reports_missing_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    r = llm.ping("nvidia")
    assert r["ok"] is False and "NVIDIA_API_KEY" in r["error"]


def test_ping_ok(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    monkeypatch.setattr(llm, "_post", lambda *a, **k: {"choices": [{"message": {"content": '{"ok": true}'}}]})
    r = llm.ping("nvidia")
    assert r["ok"] and all(m["ok"] for m in r["models"])


def test_model_that_refuses_temperature_is_called_again_without_it(monkeypatch):
    """v0.31.2: claude-haiku-5-5 answers HTTP 400 '`temperature` is deprecated for this model'."""
    import llm
    sent = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    llm._NO_TEMPERATURE.discard("claude-test")

    def post(url, headers, body, timeout=25.0):
        sent.append(dict(body))
        if "temperature" in body:
            raise llm.LLMError("provider returned HTTP 400: `temperature` is deprecated for this model.", http=400)
        return {"content": [{"type": "text", "text": "{}"}]}
    monkeypatch.setattr(llm, "_post", post)
    assert llm._call_one("claude", "claude-test", "s", "u", 100) == "{}"
    assert ["temperature" in b for b in sent] == [True, False]
    llm._call_one("claude", "claude-test", "s", "u", 100)            # remembered: no wasted first call
    assert ["temperature" in b for b in sent] == [True, False, False]
    llm._NO_TEMPERATURE.discard("claude-test")
