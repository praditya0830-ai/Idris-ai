"""DeepSeek, Qwen and Kimi go through one OpenAI-compatible adapter (local mock, no keys, no network)."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import llm

SEEN = []


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, dict(self.headers), body))
        out = {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_all_three_are_registered_and_ordered():
    for p in ("deepseek", "qwen", "kimi"):
        assert p in llm.PROVIDERS and p in llm.KEY_ENV and p in llm.DEFAULT_MODELS
    assert llm.provider_order.__doc__ and "IDRIS_LLM_ORDER" in llm.provider_order.__doc__


def test_each_provider_hits_its_own_endpoint(monkeypatch):
    srv, base = _serve()
    try:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k-deep")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k-qwen")
        monkeypatch.setenv("MOONSHOT_API_KEY", "k-kimi")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base)
        monkeypatch.setenv("DASHSCOPE_BASE_URL", base + "/compatible-mode/v1")
        monkeypatch.setenv("MOONSHOT_BASE_URL", base + "/v1")
        for prov, path, key in (("deepseek", "/chat/completions", "k-deep"),
                                ("qwen", "/compatible-mode/v1/chat/completions", "k-qwen"),
                                ("kimi", "/v1/chat/completions", "k-kimi")):
            SEEN.clear()
            out = llm._call_one(prov, llm.model_for(prov), "sys", "hi", 100)
            assert json.loads(out) == {"ok": True}
            assert SEEN[0][0] == path, (prov, SEEN[0][0])
            assert SEEN[0][1]["Authorization"] == "Bearer " + key
            body = SEEN[0][2]
            assert body["messages"][0] == {"role": "system", "content": "sys"}
            assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    finally:
        srv.shutdown()


def test_order_and_cad_order_are_env_driven(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "deepseek,claude")
    assert [p for p in llm.provider_order() if p == "deepseek"] == ["deepseek"]
    assert llm.configured()["deepseek"] is True
    assert llm.configured()["qwen"] is False          # no key -> not used


def test_unknown_provider_still_rejected(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    try:
        llm.interpret("nope", "x")
        raise AssertionError("expected LLMError")
    except llm.LLMError as e:
        assert e.status == 400


def test_model_override_per_provider(monkeypatch):
    monkeypatch.setenv("IDRIS_MODEL_DEEPSEEK", "my-model,other-model")
    assert llm.models_for("deepseek") == ["my-model", "other-model"]
    assert llm.model_for("deepseek") == "my-model"


def test_thinking_is_switched_off_where_the_api_allows_it(monkeypatch):
    """DeepSeek V4, Qwen3.8 and Kimi answer with reasoning tokens by default; that empties a short answer."""
    srv, base = _serve()
    try:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
        monkeypatch.setenv("MOONSHOT_API_KEY", "k")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base)
        monkeypatch.setenv("DASHSCOPE_BASE_URL", base)
        monkeypatch.setenv("MOONSHOT_BASE_URL", base)
        for prov, model in (("deepseek", "deepseek-flash"), ("qwen", "qwen3.8-max"), ("kimi", "kimi-k2.6")):
            SEEN.clear()
            llm._call_one(prov, model, "sys", "hi", 500)
            body = SEEN[0][2]
            if prov == "deepseek":
                assert body["thinking"] == {"type": "disabled"}
            else:
                assert body["enable_thinking"] is False
            assert body["max_tokens"] == 500          # a normal model keeps the caller's limit
    finally:
        srv.shutdown()


def test_a_thinking_only_model_gets_room_and_no_switch(monkeypatch):
    """kimi-k3.x cannot turn thinking off, so the switch is not sent and the token budget is raised."""
    srv, base = _serve()
    try:
        monkeypatch.setenv("MOONSHOT_API_KEY", "k")
        monkeypatch.setenv("MOONSHOT_BASE_URL", base)
        SEEN.clear()
        llm._call_one("kimi", "kimi-k3.1-pro", "sys", "hi", 300)
        body = SEEN[0][2]
        assert "enable_thinking" not in body
        assert body["max_tokens"] == llm.THINKING_ONLY_MIN_TOKENS
    finally:
        srv.shutdown()


def test_an_empty_completion_says_which_model_and_why(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setattr(llm, "_post_t", lambda *a, **k: {"choices": [{"message": {"content": ""},
                                                                     "finish_reason": "length"}]})
    try:
        llm._call_one("deepseek", "deepseek-v4-pro", "sys", "hi", 500)
        raise AssertionError("expected LLMError")
    except llm.LLMError as e:
        assert "deepseek-v4-pro" in str(e) and "finish_reason=length" in str(e)
