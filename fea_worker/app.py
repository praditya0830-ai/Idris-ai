"""The worker's HTTP surface: /health, /capabilities, /verify, /fea/<analysis>.

Only /health is public — Cloud Run's health probe needs it. Everything else needs
`Authorization: Bearer <FEA_WORKER_TOKEN>`; an empty or wrong token is a 401, and the
token is never written to a log line.

Solver modules register themselves in `fea_worker/solvers/<analysis>.py` with
`probe(cfg)`, `verify(cfg)` and `run(payload, cfg)`. A build without that module
answers 501 for the analysis and reports `available: false` in /capabilities, so
nothing here has to claim a capability it does not have.
"""
from __future__ import annotations

import hmac
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import capabilities
from .config import Config
from .limits import PayloadTooLarge, TooManyElements, check_body_length, check_elements, count_elements

VERSION = capabilities.VERSION


class State:
    """What this process can do, resolved once at start."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.solvers = capabilities.solver_modules()
        self.caps = capabilities.build(cfg)


class Handler(BaseHTTPRequestHandler):
    state = None
    server_version = "idris-fea/" + VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[fea] " + (fmt % args) + "\n")

    # -- plumbing ---------------------------------------------------------
    def _send(self, code, body):
        data = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        token = (self.state.cfg.token or "")
        if not token:
            return False
        return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

    def _require_auth(self):
        if self._authorized():
            return True
        self._send(401, {"error": "unauthorized",
                         "detail": "send Authorization: Bearer $FEA_WORKER_TOKEN"})
        return False

    def _path(self):
        return (self.path.split("?")[0] or "/").rstrip("/") or "/"

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        path = self._path()
        if path == "/health":
            self._send(200, {"ok": True})
            return
        if not self._require_auth():
            return
        if path == "/capabilities":
            self._send(200, self.state.caps)
            return
        if path == "/verify":
            self._send(200, self._verify())
            return
        self._send(404, {"error": "unknown endpoint", "path": path})

    def do_POST(self):
        if not self._require_auth():
            return
        path = self._path()
        try:
            length = check_body_length(self.headers.get("Content-Length"), self.state.cfg)
        except PayloadTooLarge as ex:
            self._send(413, {"error": str(ex)})
            return
        except ValueError as ex:
            self._send(400, {"error": str(ex)})
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception as ex:
            self._send(400, {"error": f"invalid JSON body: {ex}"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": "body must be a JSON object"})
            return
        if not path.startswith("/fea/"):
            self._send(404, {"error": "unknown endpoint", "path": path})
            return
        name = path[len("/fea/"):]
        mod = self.state.solvers.get(name)
        if mod is None:
            self._send(501, {"error": f"{name} is not built in this worker",
                             "available": False,
                             "reason": (self.state.caps["analyses"].get(name) or {}).get("reason",
                                                                                        "not built yet")})
            return
        try:
            check_elements(count_elements(payload), self.state.cfg)
            result = mod.run(payload, self.state.cfg)
        except TooManyElements as ex:
            self._send(413, {"error": str(ex)})
            return
        except ValueError as ex:
            self._send(400, {"error": str(ex)})
            return
        except Exception as ex:                       # pragma: no cover - defensive
            traceback.print_exc()
            self._send(500, {"error": f"{type(ex).__name__}: {ex}"})
            return
        code = 200 if result.get("ok", True) else 400
        self._send(code, result)

    def _verify(self):
        out = {"worker": "idris-fea", "version": VERSION, "analyses": {}}
        for name, mod in self.state.solvers.items():
            try:
                out["analyses"][name] = mod.verify(self.state.cfg)
            except Exception as ex:                   # a broken gate must not take the worker down
                out["analyses"][name] = {"available": False, "validated": False,
                                         "error": f"{type(ex).__name__}: {ex}"}
        if not self.state.solvers:
            out["note"] = "no solver is built into this worker yet"
        return out


def create_server(cfg=None, port=None):
    """Start the server object; the caller decides whether to serve_forever()."""
    cfg = cfg or Config()
    state = State(cfg)
    handler = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("0.0.0.0", cfg.port if port is None else int(port)), handler)
    httpd.daemon_threads = True
    return httpd, state


def main():
    httpd, state = create_server()
    caps = state.caps
    sys.stderr.write("[fea] idris-fea %s listening on :%d  ccx=%s  gmsh=%s\n" % (
        VERSION, httpd.server_address[1], caps.get("ccx_version"), caps.get("gmsh_version")))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:                          # pragma: no cover
        pass


if __name__ == "__main__":
    main()
