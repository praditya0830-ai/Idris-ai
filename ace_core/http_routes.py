"""One dispatcher for ALL solvers — server.py needs only one branch for GET and one for POST.

    GET  /solvers              -> catalog (name, fidelity, schema, benchmarks, limitations)
    GET  /solvers/tools        -> LLM function declarations (Gemini format)
    GET  /verify               -> cached benchmark report (runs in background; ?refresh=1 re-runs)
    GET  /verify/<name>        -> run one solver's benchmarks
    POST /solve/<name>         -> run a solver; body = JSON inputs
    POST /fea                  -> housing FEA in the UI team's job format (see analysis.py)
    GET  /fea/<job_id>         -> job status / result
    POST /modal, /cfd, /review -> same job format (GET /<endpoint>/<job_id>)

Returns (status_code, dict) or None if the path is not an ACE path.
"""
import threading
import time
from . import registry, analysis
import solvers  # noqa: F401  (registers all solvers on import)

# /verify runs every benchmark (several FEA solves). On a slow free-tier CPU that can take
# minutes, so the first visit starts it in the background and later visits show the cached report.
_verify = {"status": "idle", "report": None, "started": None, "finished": None}
_verify_lock = threading.Lock()


def _verify_worker():
    try:
        rep = registry.verify()
        with _verify_lock:
            _verify.update(status="done", report=rep, finished=time.time())
    except Exception as ex:  # pragma: no cover
        with _verify_lock:
            _verify.update(status="failed", report={"error": f"{type(ex).__name__}: {ex}"},
                           finished=time.time())


def start_verify(force=False):
    with _verify_lock:
        if _verify["status"] == "running" or (_verify["status"] == "done" and not force):
            return
        _verify.update(status="running", started=time.time(), finished=None)
    threading.Thread(target=_verify_worker, daemon=True, name="ace-verify").start()


def verify_page(query=""):
    start_verify(force="refresh=1" in query)
    with _verify_lock:
        v = dict(_verify)
    if v["status"] == "done":
        took = round(v["finished"] - v["started"], 1)
        return 200, {"status": "done", "computed_in_s": took,
                     "computed_at_unix": round(v["finished"]), **v["report"]}
    if v["status"] == "failed":
        return 500, {"status": "failed", **v["report"]}
    return 200, {"status": "running",
                 "running_for_s": round(time.time() - (v["started"] or time.time()), 1),
                 "message": "Benchmarks are running in the background (FEA solves take a while on "
                            "this server). Refresh this page in about a minute."}



def handle_get(path):
    res = analysis.handle_get(path)            # GET /fea/{job_id}
    if res is not None:
        return res
    query = path.partition("?")[2]
    path = path.split("?")[0].rstrip("/")
    if path == "/solvers":
        return 200, {"solvers": registry.catalog()}
    if path == "/solvers/tools":
        return 200, {"function_declarations": registry.llm_tools()}
    if path == "/verify":
        return verify_page(query)
    if path.startswith("/verify/"):
        name = path[len("/verify/"):]
        if registry.get(name) is None:
            return 404, {"ok": False, "errors": [f"unknown solver '{name}'"]}
        return 200, registry.verify(name)
    return None


def handle_post(path, payload):
    res = analysis.handle_post(path, payload)  # POST /fea (UI job format)
    if res is not None:
        return res
    path = path.split("?")[0].rstrip("/")
    if path.startswith("/solve/"):
        name = path[len("/solve/"):]
        res = registry.run(name, payload)
        if res.get("errors") and "available" in res:
            return 404, res
        return (200 if res.get("ok") else 400), res
    return None


def encode_json(body, accept_encoding=""):
    """Serialize a response compactly; gzip it when the browser accepts it (FEA surfaces shrink ~4x).

    Returns (bytes, extra_headers). In server.py:
        data, headers = encode_json(body, self.headers.get("Accept-Encoding", ""))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in headers.items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    """
    import gzip
    import json
    data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > 2048 and "gzip" in (accept_encoding or ""):
        return gzip.compress(data, 6), {"Content-Encoding": "gzip", "Vary": "Accept-Encoding"}
    return data, {}
