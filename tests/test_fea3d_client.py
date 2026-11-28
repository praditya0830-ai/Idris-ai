"""The solver side of the worker: never a 500 because the worker is not there.

Two states matter to the Studio: "not deployed" (env unset, host unreachable, token
rejected) must come back as `{"deployed": false, "reason": ...}`, and a healthy
worker must be forwarded as-is.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import fea3d


class _Fake(BaseHTTPRequestHandler):
    payload = {"worker": "idris-fea", "version": "0.1.0", "analyses": {"static_linear": {"available": True}}}
    status = 200
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Fake.seen.append((self.path, self.headers.get("Authorization")))
        data = json.dumps(_Fake.payload).encode()
        self.send_response(_Fake.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_worker():
    _Fake.seen = []
    _Fake.status = 200
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_no_env_reports_not_deployed(monkeypatch):
    monkeypatch.delenv("FEA_WORKER_URL", raising=False)
    monkeypatch.delenv("FEA_WORKER_TOKEN", raising=False)
    out = fea3d.capabilities()
    assert out == {"deployed": False, "reason": fea3d.NOT_DEPLOYED}


def test_route_answers_200_not_500_when_undeployed(monkeypatch):
    monkeypatch.delenv("FEA_WORKER_URL", raising=False)
    monkeypatch.delenv("FEA_WORKER_TOKEN", raising=False)
    code, body = fea3d.handle_get("/fea3d/capabilities")
    assert code == 200
    assert body["deployed"] is False
    assert "not deployed yet" in body["reason"]


def test_unknown_path_is_not_claimed():
    assert fea3d.handle_get("/fea3d/something-else") is None


def test_forwards_capabilities_and_sends_the_token(monkeypatch, fake_worker):
    monkeypatch.setenv("FEA_WORKER_URL", fake_worker)
    monkeypatch.setenv("FEA_WORKER_TOKEN", "secret-token")
    out = fea3d.capabilities()
    assert out["deployed"] is True
    assert out["worker"] == "idris-fea"
    assert out["analyses"]["static_linear"]["available"] is True
    assert _Fake.seen[-1][0] == "/capabilities"
    assert _Fake.seen[-1][1] == "Bearer secret-token"


def test_rejected_token_is_a_reason_not_an_error(monkeypatch, fake_worker):
    monkeypatch.setenv("FEA_WORKER_URL", fake_worker)
    monkeypatch.setenv("FEA_WORKER_TOKEN", "wrong")
    _Fake.status = 401
    _Fake.payload = {"error": "unauthorized"}
    out = fea3d.capabilities()
    assert out["deployed"] is False
    assert "401" in out["reason"]


def test_unreachable_worker_is_a_reason(monkeypatch):
    monkeypatch.setenv("FEA_WORKER_URL", "http://127.0.0.1:9")     # discard port
    monkeypatch.setenv("FEA_WORKER_TOKEN", "x")
    out = fea3d.capabilities()
    assert out["deployed"] is False
    assert "not reachable" in out["reason"]
