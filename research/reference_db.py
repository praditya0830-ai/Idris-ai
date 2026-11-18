"""Calculation-reference database, stored on its own branch of the Idris-ai repository.

Why a separate branch: Render deploys automatically from `main`, so every commit to main restarts
the live server. References live on `reference-db` (env IDRIS_REFERENCE_BRANCH), written through
the GitHub Contents API with a fine-grained token (env GITHUB_TOKEN: Contents read/write on this
one repository).

The rule this module exists to protect: a reference is a CANDIDATE until a human verifies it.
Nothing here changes `status`. A new file is always written as "candidate", an existing file is
never overwritten, so no code path can promote a reference (nor demote a verified one). Promotion
happens only when Arya edits the JSON by hand after a worked example, solver code and a passing
`/verify` benchmark exist.

Every failure is reported, never raised at the chat: without a token, or with GitHub unreachable,
the chat keeps answering and says the reference could not be stored.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DIR = "references"                      # references/<slug>.json on the reference branch
DEFAULT_BRANCH = "reference-db"
PROTECTED_BRANCHES = ("main", "master")  # Render deploys from main: a write here restarts the server
TTL_S = 600                             # re-read the list at most every 10 minutes
DEFAULT_REPO = "praditya0830-ai/Idris-ai"
UNREADABLE = "could not read the formula reliably"
SOURCE_TYPES = ("standard", "textbook", "datasheet", "other")
STATUSES = ("candidate", "verified")
FIELDS = ("slug", "topic", "title", "source_type", "source_name", "url", "formula", "variables", "units",
          "valid_range", "assumptions", "worked_example", "status", "found_for", "found_by", "date",
          "verified_by")

_cache = {"at": 0.0, "items": [], "error": None, "branch": None}
_by_sha: dict[str, dict] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ configuration
def token() -> str:
    return (os.environ.get("GITHUB_TOKEN") or "").strip()


def branch_name() -> str:
    return (os.environ.get("IDRIS_REFERENCE_BRANCH") or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH


def repo() -> str:
    return (os.environ.get("IDRIS_GITHUB_REPO") or DEFAULT_REPO).strip().strip("/")


def available() -> bool:
    """True when the server can read and write the database (a token is configured)."""
    return bool(token())


def _api() -> str:
    return (os.environ.get("IDRIS_GITHUB_API") or "https://api.github.com").rstrip("/")


def _headers() -> dict:
    return {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token()}",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "IdrisReferenceDB/1.0"}


def _http(method: str, path: str, body: dict | None = None, timeout: float = 12.0):
    """One GitHub API call -> (status, payload). Network failures raise (callers report them)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(_api() + path, data=data, method=method, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw or "{}")
        except ValueError:
            return e.code, {"message": raw[:200]}


def refuse_protected(branch: str) -> str | None:
    """-> the reason a branch may not be written, or None. The database never writes to main."""
    if (branch or "").strip().lower() in PROTECTED_BRANCHES:
        return (f"refusing to write to '{branch}': Render deploys from main, so a commit there would "
                f"restart the live server")
    return None


# ------------------------------------------------------------------ record shape
def slugify(topic, title="") -> str:
    raw = f"{topic or 'reference'} {title or ''}".lower()
    keep = "".join(c if (c.isalnum() or c in " -_") else " " for c in raw).split()
    return "-".join(keep)[:80].strip("-") or "reference"


def _shape(rec: dict) -> dict:
    """Fill every field of the agreed JSON, keeping a status that was already set on disk."""
    r = rec if isinstance(rec, dict) else {}
    st = str(r.get("source_type") or "other").strip().lower()
    status = str(r.get("status") or "candidate").strip().lower()
    variables = r.get("variables")
    example = r.get("worked_example")
    return {
        "slug": str(r.get("slug") or slugify(r.get("topic"), r.get("title")))[:80],
        "topic": str(r.get("topic") or "general")[:80],
        "title": str(r.get("title") or "")[:200],
        "source_type": st if st in SOURCE_TYPES else "other",
        "source_name": str(r.get("source_name") or "")[:200],
        "url": str(r.get("url") or "")[:600],
        "formula": str(r.get("formula") or UNREADABLE)[:2000],
        "variables": ({str(k)[:32]: str(v)[:200] for k, v in variables.items()} if isinstance(variables, dict) else {}),
        "units": str(r.get("units") or "")[:200],
        "valid_range": str(r.get("valid_range") or "")[:600],
        "assumptions": str(r.get("assumptions") or "")[:600],
        "worked_example": example if isinstance(example, dict) else None,
        "status": status if status in STATUSES else "candidate",
        "found_for": str(r.get("found_for") or "")[:400],
        "found_by": str(r.get("found_by") or "")[:80],
        "date": str(r.get("date") or time.strftime("%Y-%m-%d"))[:10],
        "verified_by": r.get("verified_by") if isinstance(r.get("verified_by"), (str, type(None))) else None,
    }


def normalize(rec: dict) -> dict:
    """The record a write is allowed to create: always a candidate, never verified by the code."""
    out = _shape(rec)
    out["status"] = "candidate"
    out["verified_by"] = None
    return out


# ------------------------------------------------------------------ reading (cached 10 minutes)
def reset_cache() -> None:
    """Forget what was read (tests, or after a configuration change)."""
    with _lock:
        _cache.update(at=0.0, items=[], error=None, branch=None)
        _by_sha.clear()


def warm() -> None:
    """Read the list once at server start, in the background: startup must not wait for GitHub."""
    if not available():
        return

    def _run():
        try:
            list_all()
        except Exception:               # pragma: no cover - the chat never depends on this
            pass

    threading.Thread(target=_run, daemon=True, name="idris-reference-warm").start()


def _snapshot() -> dict:
    return {"items": list(_cache["items"]), "error": _cache["error"], "branch": _cache["branch"] or branch_name(),
            "cached_at": _cache["at"] or None, "ttl_s": TTL_S}


def list_all(force: bool = False) -> dict:
    """All stored references -> {items, error, branch, cached_at}. An outage keeps the last read list."""
    b = branch_name()
    if not available():
        return {"items": [], "error": "GITHUB_TOKEN is not set on the server", "branch": b,
                "cached_at": None, "ttl_s": TTL_S}
    if not force and _cache["at"] and time.time() - _cache["at"] < TTL_S:
        return _snapshot()
    with _lock:
        if not force and _cache["at"] and time.time() - _cache["at"] < TTL_S:
            return _snapshot()
        try:
            _cache["items"] = _fetch_all(b)
            _cache["error"] = None
            _cache["at"] = time.time()
            _cache["branch"] = b
        except Exception as e:
            # keep the items already read: an outage must not empty the database in memory
            _cache["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return _snapshot()


def _fetch_all(b: str) -> list:
    status, payload = _http("GET", f"/repos/{repo()}/contents/{DIR}?ref={urllib.parse.quote(b)}")
    if status == 404:
        return []                       # the branch or the folder is not created yet: an empty database
    if status != 200:
        raise RuntimeError(f"GitHub answered {status}: {str((payload or {}).get('message'))[:120]}")
    rows = payload if isinstance(payload, list) else []
    out = []
    for row in rows:
        name = str((row or {}).get("name") or "")
        if not name.endswith(".json") or "/" in name:
            continue
        sha = str((row or {}).get("sha") or "")
        if sha and sha in _by_sha:
            out.append(_by_sha[sha])
            continue
        rec = _read_one(name, b)
        if rec:
            out.append(rec)
            if sha:
                _by_sha[sha] = rec
    return sorted(out, key=lambda r: (r.get("date") or "", r.get("slug") or ""), reverse=True)


def _read_one(name: str, b: str) -> dict | None:
    status, payload = _http("GET", f"/repos/{repo()}/contents/{DIR}/{urllib.parse.quote(name)}"
                                   f"?ref={urllib.parse.quote(b)}")
    if status != 200 or not isinstance(payload, dict):
        return None
    try:
        raw = base64.b64decode(str(payload.get("content") or ""), validate=False).decode("utf-8", "replace")
        rec = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return _shape(rec) if isinstance(rec, dict) else None


def get(slug: str) -> dict | None:
    """One full reference by slug (cache first, then GitHub)."""
    slug = _clean_slug(slug)
    if not slug:
        return None
    for rec in list_all().get("items") or []:
        if rec.get("slug") == slug:
            return rec
    if not available():
        return None
    try:
        return _read_one(f"{slug}.json", branch_name())
    except Exception:
        return None


def find(topic: str) -> list:
    """Stored references for one topic (the same request is not searched again)."""
    key = (topic or "").strip().lower()
    if not key or key == "general":
        return []
    hits = []
    for rec in list_all().get("items") or []:
        t = str(rec.get("topic") or "").lower()
        if t == key or key in t or (t and t in key):
            hits.append(rec)
    return hits


def find_duplicate(rec: dict) -> dict | None:
    """Same slug or same url -> already stored. Nothing is written twice."""
    slug = _clean_slug(rec.get("slug"))
    url = _norm_url(rec.get("url"))
    for old in list_all().get("items") or []:
        if slug and old.get("slug") == slug:
            return old
        if url and _norm_url(old.get("url")) == url:
            return old
    return None


# ------------------------------------------------------------------ writing (candidate only)
def save(record: dict, branch: str | None = None) -> dict:
    """Store one reference as a CANDIDATE. -> {saved, slug, branch, reason, existing?}

    Refuses: main (Render deploys from it), a missing token, a record without a url, and anything
    already stored under the same slug or url. Never raises: the chat reports the reason instead.
    """
    b = (branch or branch_name()).strip()
    rec = normalize(record)
    out = {"saved": False, "slug": rec["slug"], "branch": b, "reason": None, "existing": False}
    if not rec.get("url"):
        return {**out, "reason": "a reference needs a url: the link is the evidence"}
    why = refuse_protected(b)
    if why:
        return {**out, "reason": why}
    if not available():
        return {**out, "reason": "GITHUB_TOKEN is not set on the server, so the reference was not stored"}
    dup = find_duplicate(rec)
    if dup:
        return {**out, "reason": f"already stored as '{dup.get('slug')}'", "existing": True,
                "slug": dup.get("slug") or rec["slug"]}
    body = {"message": f"reference(candidate): {rec['topic']} - {rec['source_name'] or rec['url'][:60]}",
            "content": base64.b64encode(json.dumps(rec, indent=2, ensure_ascii=False).encode("utf-8")).decode(),
            "branch": b}
    try:
        status, payload = _http("PUT", f"/repos/{repo()}/contents/{DIR}/{rec['slug']}.json", body)
    except Exception as e:
        return {**out, "reason": f"GitHub is unreachable: {type(e).__name__}: {str(e)[:120]}"}
    if status in (200, 201):
        with _lock:
            _cache["items"] = [i for i in _cache["items"] if i.get("slug") != rec["slug"]] + [rec]
            _cache["at"] = _cache["at"] or time.time()
            _cache["branch"] = b
        return {**out, "saved": True}
    msg = str((payload or {}).get("message") or "")[:160]
    if status in (409, 422) and ("sha" in msg.lower() or "already exists" in msg.lower()):
        return {**out, "reason": "already stored on the branch", "existing": True}
    return {**out, "reason": f"GitHub rejected the write ({status}): {msg}"}


# ------------------------------------------------------------------ HTTP payloads
def public_list() -> dict:
    """GET /references/db: slug, topic, title, status, url for every stored reference."""
    snap = list_all()
    rows = [{"slug": r.get("slug"), "topic": r.get("topic"), "title": r.get("title"), "status": r.get("status"),
             "url": r.get("url"), "source_type": r.get("source_type"), "source_name": r.get("source_name"),
             "date": r.get("date")} for r in snap.get("items") or []]
    return {"count": len(rows), "references": rows, "branch": snap.get("branch"),
            "stored": available(), "error": snap.get("error"),
            "cached_at": snap.get("cached_at"), "ttl_s": snap.get("ttl_s")}


def public_one(slug: str) -> dict | None:
    rec = get(slug)
    if not rec:
        return None
    return {**rec, "label": f"**Unverified formula from {rec.get('source_name') or rec.get('url')}: "
                            f"not used for any IDRIS result yet**"} if rec.get("status") == "candidate" else rec


def _clean_slug(slug) -> str:
    s = str(slug or "").strip().lower()
    return s if s and all(c.isalnum() or c in "-_" for c in s) and len(s) <= 80 else ""


def _norm_url(url) -> str:
    return str(url or "").strip().lower().rstrip("/")
