"""The worker's public surface: health, auth and the limits around a request.

No CalculiX on the test machine, so these tests only touch what does not need a
solver: the auth gate, `/health`, `/capabilities`' shape, the body and element
limits, and the honest 501 for an analysis this build does not contain.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from fea_worker import app as worker_app
from fea_worker import capabilities
from fea_worker import limits
from fea_worker.config import Config

TOKEN = "t0ken-for-tests"


@pytest.fixture()
def worker():
    cfg = Config({
        "FEA_WORKER_TOKEN": TOKEN,
        "PORT": "0",
        "FEA_MAX_BODY": "4096",
        "FEA_MAX_ELEMENTS": "1000",
        "FEA_RUN_BENCHMARKS": "0",
    })
    httpd, state = worker_app.create_server(cfg, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], state
    finally:
        httpd.shutdown()
        httpd.server_close()


def call(port, path, body=None, token=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 method="POST" if data is not None else "GET")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as ex:
        return ex.code, json.loads(ex.read().decode() or "{}")


def test_health_needs_no_token(worker):
    port, _ = worker
    code, body = call(port, "/health")
    assert code == 200 and body == {"ok": True}


def test_capabilities_without_token_is_401(worker):
    port, _ = worker
    code, body = call(port, "/capabilities")
    assert code == 401 and "unauthorized" in body["error"]


def test_capabilities_with_wrong_token_is_401(worker):
    port, _ = worker
    code, _ = call(port, "/capabilities", token="not-the-token")
    assert code == 401


def test_capabilities_shape_with_token(worker):
    port, _ = worker
    code, body = call(port, "/capabilities", token=TOKEN)
    assert code == 200
    assert body["worker"] == "idris-fea"
    assert body["version"] == capabilities.VERSION
    assert "ccx_version" in body and "gmsh_version" in body
    for name in capabilities.ANALYSES:
        assert name in body["analyses"], name
        entry = body["analyses"][name]
        assert entry["available"] is False, f"{name} must not claim availability without its module"
        assert entry.get("reason"), f"{name} needs a reason"
    assert body["analyses"]["cfd_3d"]["reason"] == "not CalculiX; needs OpenFOAM"
    assert body["limits"]["max_elements"] == 1000
    assert body["limits"]["max_body_bytes"] == 4096


def test_verify_lists_every_solver_it_has(worker):
    port, state = worker
    code, body = call(port, "/verify", token=TOKEN)
    assert code == 200
    assert set(body["analyses"]) == set(state.solvers)


def test_unknown_analysis_is_501_with_reason(worker):
    port, _ = worker
    code, body = call(port, "/fea/nonsense", body={"a": 1}, token=TOKEN)
    assert code == 501
    assert body["available"] is False
    assert "not built" in body["reason"] or "not built" in body["error"]


def test_body_over_limit_is_413(worker):
    port, _ = worker
    big = b'{"x":"' + b"a" * 5000 + b'"}'
    code, body = call(port, "/fea/static", raw=big, token=TOKEN)
    assert code == 413 and "limit is" in body["error"]


def test_invalid_json_is_400(worker):
    port, _ = worker
    code, body = call(port, "/fea/static", raw=b"{nope", token=TOKEN)
    assert code == 400 and "invalid JSON" in body["error"]


def test_empty_token_means_every_protected_route_is_closed():
    httpd, _ = worker_app.create_server(Config({"PORT": "0", "FEA_WORKER_TOKEN": ""}), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        code, _ = call(port, "/capabilities", token="")
        assert code == 401
        code, _ = call(port, "/capabilities", token="anything")
        assert code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- limits, as units ------------------------------------------------------

def test_body_length_limit():
    cfg = Config({"FEA_MAX_BODY": "1000"})
    assert limits.check_body_length(1000, cfg) == 1000
    with pytest.raises(limits.PayloadTooLarge):
        limits.check_body_length(1001, cfg)
    with pytest.raises(ValueError):
        limits.check_body_length("abc", cfg)


def test_element_limit():
    cfg = Config({"FEA_MAX_ELEMENTS": "10"})
    assert limits.check_elements(10, cfg) == 10
    with pytest.raises(limits.TooManyElements) as ex:
        limits.check_elements(11, cfg)
    assert "mesh.size_mm" in str(ex.value)


def test_count_elements_reads_what_the_caller_sent():
    assert limits.count_elements({"mesh": {"elements": 12}}) == 12
    assert limits.count_elements({"elements": 7}) == 7
    assert limits.count_elements({"mesh": {"size_mm": 2}}) == 0
    assert limits.count_elements([]) == 0


def test_missing_binaries_report_null_versions(monkeypatch):
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: None)
    body = capabilities.build(Config({"FEA_RUN_BENCHMARKS": "0"}))
    assert body["ccx_version"] is None and body["gmsh_version"] is None
    assert body["binaries"] == {"ccx": False, "gmsh": False}
