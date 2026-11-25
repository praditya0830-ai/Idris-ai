"""Failed design reports that are kept even when no job exists (doc: laporan gagal selalu tersimpan, 25 Nov 2026).

A free design that fails on /chat carries `design_report` and no job id, so the old feedback route (which needs a
running job) could not store it. This module keeps that report in the same place the job feedback lives
(`rounds_data/`, overridable with IDRIS_REPORT_DIR), without any personal data: the report itself, the message the
engineer sent, the topic and the solver version. No AI call is made, so it does not spend ai_budget.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time

STORE_ENV = "IDRIS_REPORT_DIR"
FILE_NAME = "design_reports.jsonl"
MAX_REPORT_BYTES = 24_000
MAX_MESSAGE_CHARS = 2000
MAX_TOPIC_CHARS = 120
MAX_VERSION_CHARS = 40
MAX_LIST = 200
DEFAULT_LIST = 50
POST_ROUTES = ("/design/report",)


def store_dir() -> str:
    return os.environ.get(STORE_ENV) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "rounds_data")


def store_path() -> str:
    return os.path.join(store_dir(), FILE_NAME)


_LOCK = threading.Lock()


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else time.time()))


def _new_id() -> str:
    return "rep-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)


def _clean(params: dict) -> tuple:
    """-> (row, error). Only the four documented fields are kept, each capped; nothing is invented."""
    if not isinstance(params, dict):
        return None, "body must be a JSON object"
    report = params.get("report")
    if not isinstance(report, dict) or not report:
        return None, "send `report`: the design_report object the answer carried"
    try:
        size = len(json.dumps(report, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return None, "report must be JSON-serialisable"
    if size > MAX_REPORT_BYTES:
        return None, f"report is {size} bytes; keep it under {MAX_REPORT_BYTES}"
    message = params.get("message")
    topic = params.get("topic")
    version = params.get("solver_version")
    row = {"id": _new_id(), "saved_at": _iso(),
           "report": report,
           "message": (str(message)[:MAX_MESSAGE_CHARS] if message is not None else None),
           "topic": (str(topic)[:MAX_TOPIC_CHARS] if topic is not None else None),
           "solver_version": (str(version)[:MAX_VERSION_CHARS] if version is not None else None)}
    return row, None


def _append(row: dict) -> None:
    path = store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
        # a store that grows without bound is a disk problem, not a feature: keep the newest 500 rows
        try:
            if os.path.getsize(path) > 8_000_000:
                _rewrite_last(500)
        except OSError:
            pass


def _read_all() -> list:
    try:
        with open(store_path(), encoding="utf-8") as fh:
            rows = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("id"):
                    rows.append(row)
            return rows
    except FileNotFoundError:
        return []
    except OSError:
        return []


def _rewrite_last(keep: int) -> None:
    rows = _read_all()[-keep:]
    path = store_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def save(params: dict) -> tuple:
    """POST /design/report -> (status, {ok, id} or {error})."""
    row, error = _clean(params)
    if error:
        return 400, {"success": False, "ok": False, "error": error}
    try:
        _append(row)
    except OSError as e:
        return 500, {"success": False, "ok": False, "error": f"the report could not be stored: {e}"}
    return 200, {"success": True, "ok": True, "id": row["id"], "saved_at": row["saved_at"]}


def listing(limit=None) -> list:
    """The newest reports first, at most `limit` (default 50, hard cap 200)."""
    try:
        want = int(limit) if limit is not None else DEFAULT_LIST
    except (TypeError, ValueError):
        want = DEFAULT_LIST
    want = max(1, min(MAX_LIST, want))
    return list(reversed(_read_all()))[:want]


def handle_post(base: str, params: dict):
    if str(base or "").rstrip("/") != "/design/report":
        return None
    return save(params if isinstance(params, dict) else {})


def handle_get(path: str):
    """/design/reports?limit=50 -> the reports that were kept."""
    raw = str(path or "")
    if raw.split("?")[0].rstrip("/") != "/design/reports":
        return None
    limit = None
    if "?" in raw:
        for part in raw.split("?", 1)[1].split("&"):
            key, _, value = part.partition("=")
            if key == "limit" and value:
                limit = value
    rows = listing(limit)
    return 200, {"success": True, "count": len(rows), "reports": rows}
