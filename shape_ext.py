"""Bagian H routes: goals -> measurable objectives, shape families, and a ranking from measured values.

POST /shape/objectives  {"request": "motor listrik tidak bising"}
     -> goals, quantities with the solver that measures each one, scoreable, not_checked
POST /shape/families    {"request": ..., "count": 3 | 5 | 7 | 10}
     -> shape families with their parametric design space (the engineer's sizes locked)
POST /shape/rank        {"measured": <from /shape/objectives>, "results": [...], "weights": {...}}
     -> ranking, only from values that were really measured
GET  /shape/objectives?q=...            same as the POST, for a quick look
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from ace_core import objectives as obj

POST_ROUTES = ("/shape/objectives", "/shape/families", "/shape/rank")


def handle_get(path: str):
    if path.split("?")[0].rstrip("/") != "/shape/objectives":
        return None
    q = (parse_qs(urlparse(path).query).get("q") or [""])[0].strip()[:600]
    if not q:
        return 400, {"success": False, "error": "add ?q=your goal sentence, e.g. /shape/objectives?q=motor senyap"}
    return 200, {"success": True, **obj.measurable(q)}


def handle_post(base: str, params: dict):
    if base == "/shape/objectives":
        text = str(params.get("request") or "").strip()
        if not text:
            return 400, {"success": False, "error": "request is required (one sentence with the goal)"}
        return 200, {"success": True, **obj.measurable(text)}

    if base == "/shape/families":
        text = str(params.get("request") or "").strip()
        if not text:
            return 400, {"success": False, "error": "request is required (one sentence with the goal)"}
        try:
            count = int(params.get("count") or 3)
        except (TypeError, ValueError):
            return 400, {"success": False, "error": "count must be one of 3, 5, 7, 10"}
        if count not in obj.COUNTS:
            return 422, {"success": False, "error": "count must be one of 3, 5, 7, 10",
                         "allowed": list(obj.COUNTS)}
        out = obj.families(text, count, provider=params.get("provider"))
        return (200 if out["ok"] else 422), {"success": out["ok"], **out,
                                            "objective_map": obj.measurable(text)}

    if base == "/shape/rank":
        results = params.get("results") if isinstance(params.get("results"), list) else None
        if not results:
            return 400, {"success": False, "error": "results is required: one entry per evaluated variant"}
        measured = params.get("measured") if isinstance(params.get("measured"), dict) else None
        if measured is None:
            text = str(params.get("request") or "").strip()
            if not text:
                return 400, {"success": False,
                             "error": "send measured (from POST /shape/objectives) or the request sentence"}
            measured = {q["key"]: q for q in obj.measurable(text)["quantities"]}
        weights = params.get("weights") if isinstance(params.get("weights"), dict) else None
        out = obj.rank(results, weights, measured)
        return 200, {"success": True, **out}

    return None
