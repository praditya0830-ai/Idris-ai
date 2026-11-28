"""3D FEA through the Cloud Run worker (CalculiX + gmsh).

The worker is a separate service, not part of this process: it scales to zero
between requests so it costs nothing while idle, which means the first call after a
quiet period pays a cold start of up to ~30 s. This module therefore:

* reports `{"deployed": false, "reason": ...}` — never a 500 — when
  `FEA_WORKER_URL` / `FEA_WORKER_TOKEN` are unset, when the worker cannot be
  reached, or when it rejects the token;
* uses a short timeout for `/capabilities` (a page load) and the long one for a
  solve;
* keeps the token out of URLs and out of logs.

The Studio reads these answers to decide whether a button is enabled, and shows the
reason string next to it when it is not.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

CAP_TIMEOUT_S = 8.0
RUN_TIMEOUT_S = 300.0        # the worker's own ccx limit is 240 s
NOT_DEPLOYED = "the FEA worker is not deployed yet"


def worker_config():
    url = (os.environ.get("FEA_WORKER_URL") or "").strip().rstrip("/")
    token = (os.environ.get("FEA_WORKER_TOKEN") or "").strip()
    return url, token


def _not_deployed(reason):
    return {"deployed": False, "reason": reason}


def _call(path, payload=None, timeout=CAP_TIMEOUT_S):
    """(status_code, body_dict). Never raises: every failure becomes a reason string."""
    url, token = worker_config()
    if not url or not token:
        return 200, _not_deployed(NOT_DEPLOYED)
    req = urllib.request.Request(url + path, method="POST" if payload is not None else "GET")
    req.add_header("Authorization", "Bearer " + token)
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as ex:
        detail = ""
        try:
            detail = json.loads(ex.read().decode() or "{}").get("error", "")
        except Exception:
            pass
        if ex.code == 401:
            return 200, _not_deployed("the FEA worker rejected the token (401) - check FEA_WORKER_TOKEN")
        return ex.code, {"deployed": True, "error": detail or f"the FEA worker answered {ex.code}"}
    except Exception as ex:
        return 200, _not_deployed(f"the FEA worker is not reachable: {type(ex).__name__}")
    finally:
        del data


def capabilities():
    """What the deployed worker can actually run, or why it cannot be used."""
    code, body = _call("/capabilities")
    if body.get("deployed") is False:
        return body
    if code != 200:
        return _not_deployed(body.get("error") or f"the FEA worker answered {code}")
    return {"deployed": True, **body}


def verify():
    code, body = _call("/verify")
    if body.get("deployed") is False:
        return body
    return body if code == 200 else _not_deployed(body.get("error") or f"the FEA worker answered {code}")


def handle_get(path):
    """Routes served under /fea3d/* (GET)."""
    base = path.split("?")[0].rstrip("/")
    if base == "/fea3d/capabilities":
        return 200, capabilities()
    if base == "/fea3d/verify":
        return 200, verify()
    return None
