"""One dispatcher for ALL solvers — server.py needs only one branch for GET and one for POST.

    GET  /solvers              -> catalog (name, fidelity, schema, benchmarks, limitations)
    GET  /solvers/tools        -> LLM function declarations (Gemini format)
    GET  /verify               -> run every benchmark of every solver (credibility table)
    GET  /verify/<name>        -> run one solver's benchmarks
    POST /solve/<name>         -> run a solver; body = JSON inputs
    POST /fea                  -> housing FEA in the UI team's job format (see analysis.py)
    GET  /fea/<job_id>         -> job status / result

Returns (status_code, dict) or None if the path is not an ACE path.
"""
from . import registry, analysis
import solvers  # noqa: F401  (registers all solvers on import)


def handle_get(path):
    res = analysis.handle_get(path)            # GET /fea/{job_id}
    if res is not None:
        return res
    path = path.split("?")[0].rstrip("/")
    if path == "/solvers":
        return 200, {"solvers": registry.catalog()}
    if path == "/solvers/tools":
        return 200, {"function_declarations": registry.llm_tools()}
    if path == "/verify":
        return 200, registry.verify()
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
