"""Calculation-reference database: solver first, then the stored candidates, then the web.

No network: GitHub is a fake Contents API, the AI is a fake reader, the pages are fixed text.
The rules under test are the ones that keep a guessed formula out of a design result.
"""
import base64
import hashlib
import json
import os
import urllib.parse

import pytest

import llm
from ai import engineer as ENG
from research import fetch, reference_db as db, reference_flow as flow, search

PAGE = ("ISO 6336-2:2006 contact stress. The contact stress is sigma_H = Z_H * Z_E * Z_eps * "
        "sqrt(F_t / (d_1 b) * (u + 1) / u) for spur and helical gears. It assumes steel gears, "
        "full load sharing and the ISO 6336-1 factors. Valid for 1.0 <= u <= 8.")
READ = {"found": True, "title": "ISO 6336-2 contact stress", "source_name": "ISO 6336-2:2006",
        "source_type": "standard", "formula": "sigma_H = Z_H · Z_E · Z_eps · sqrt(F_t/(d_1·b) · (u+1)/u)",
        "variables": {"sigma_H": "contact stress, MPa", "F_t": "tangential force, N"},
        "units": "N, mm, MPa", "valid_range": "spur and helical gears, 1.0 <= u <= 8",
        "assumptions": "steel gears, load sharing per ISO 6336-1", "worked_example": None,
        "quote": "The contact stress is sigma_H = Z_H * Z_E * Z_eps * sqrt(F_t / (d_1 b) * (u + 1) / u)"}
URL = "https://www.iso.org/standard/80324.html"


class FakeGitHub:
    """GitHub Contents API double: one dict of files per branch, no network at all."""

    def __init__(self, files=None, fail=False):
        self.files = {"reference-db": dict(files or {})}
        self.fail = fail
        self.calls = []

    def __call__(self, method, path, body=None, timeout=12.0):
        self.calls.append({"method": method, "path": path, "body": body})
        if self.fail:
            raise OSError("github is unreachable")
        parts = urllib.parse.urlparse(path)
        seg = [p for p in parts.path.split("/") if p]
        if "contents" not in seg:
            return 404, {"message": "Not Found"}
        tail = seg[seg.index("contents") + 1:]
        if not tail or tail[0] != "references":
            return 404, {"message": "Not Found"}
        ref = (urllib.parse.parse_qs(parts.query).get("ref") or ["reference-db"])[0]
        store = self.files.setdefault(ref, {})
        if method == "GET" and len(tail) == 1:
            return 200, [{"name": n, "sha": hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest(),
                          "type": "file"} for n, r in store.items()]
        if method == "GET" and len(tail) == 2:
            rec = store.get(tail[1])
            if rec is None:
                return 404, {"message": "Not Found"}
            return 200, {"name": tail[1], "sha": "x",
                         "content": base64.b64encode(json.dumps(rec).encode()).decode()}
        if method == "PUT" and len(tail) == 2:
            self.files.setdefault(body["branch"], {})[tail[1]] = json.loads(base64.b64decode(body["content"]))
            return 201, {"content": {"path": f"references/{tail[1]}"}}
        return 404, {"message": "Not Found"}

    def writes(self):
        return [c for c in self.calls if c["method"] == "PUT"]


@pytest.fixture
def gh(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("IDRIS_REFERENCE_BRANCH", "reference-db")
    monkeypatch.setenv("IDRIS_GITHUB_REPO", "o/r")
    monkeypatch.setenv("IDRIS_GITHUB_API", "http://github.test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test")
    fake = FakeGitHub()
    monkeypatch.setattr(db, "_http", fake)
    db.reset_cache()
    yield fake
    db.reset_cache()


def _wire_web(monkeypatch, page=PAGE, hit_url=URL, read=READ):
    monkeypatch.setattr(search, "search", lambda q, count=5: [{"title": "ISO 6336-2", "url": hit_url,
                                                               "snippet": ""}])
    monkeypatch.setattr(fetch, "fetch", lambda url, **k: page)
    monkeypatch.setattr(llm, "call_provider", lambda *a, **k: (json.dumps(read), "fake-model"))


# ------------------------------------------------------------------ 1a. the solver owns a known calculation
def test_a_bench_question_names_the_solver_and_never_browses(gh, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a calculation the bench already does must not be searched on the web")
    monkeypatch.setattr(search, "search", boom)
    monkeypatch.setattr(fetch, "fetch", boom)

    eng = ENG.solve("torsi baut M10 kelas 8.8, preload 75%")
    assert eng["claims"], "the bench must compute this before the reference step is tested"
    res = flow.for_request("torsi baut M10 kelas 8.8", eng, "claude")

    assert res["mode"] == "solver" and res["searched"] is False
    names = {s["name"] for s in res["solvers"]}
    assert "bolt_proof_load" in names
    assert any(b["id"] for s in res["solvers"] for b in s["benchmarks"]), res["solvers"]
    assert "do NOT search the web" in res["block"]
    assert "Unverified" not in res["block"]
    assert gh.writes() == [] and gh.calls == [], "a stored calculation must not even read the database"
    note = flow.reply_note(res)
    assert "IDRIS solver" in note and "benchmark:" in note
    assert any(f"`{n}`" in note for n in names)


# ------------------------------------------------------------------ 1b-1d. web -> candidate, stored once
def test_an_unknown_formula_is_stored_as_a_candidate_on_its_own_branch(gh, monkeypatch):
    _wire_web(monkeypatch)
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")

    assert res["mode"] == "candidate_new" and res["searched"] is True
    assert len(gh.writes()) == 1
    put = gh.writes()[0]
    assert put["body"]["branch"] == "reference-db"                 # never main
    assert put["path"].startswith("/repos/o/r/contents/references/") and put["path"].endswith(".json")
    assert res["saved"][0]["saved"] is True

    label = "**Unverified formula from ISO 6336-2:2006: not used for any IDRIS result yet**"
    assert label in res["block"]
    assert "Unverified formula from ISO 6336-2:2006: not used for any IDRIS result yet" in flow.reply_note(res)
    assert "ISO 6336" in flow.reply_note(res) and URL in flow.reply_note(res)

    stored = gh.files["reference-db"][list(gh.files["reference-db"])[0]]
    assert stored["status"] == "candidate" and stored["verified_by"] is None
    assert stored["formula"].startswith("sigma_H")
    assert stored["found_by"] == "fake-model"
    assert stored["topic"] == "gear" and URL in stored["url"]


def test_an_unreadable_formula_keeps_only_the_link(gh, monkeypatch):
    _wire_web(monkeypatch, read={**READ, "found": False, "formula": ""})
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    rec = res["items"][0]
    assert rec["formula"] == "could not read the formula reliably"
    assert rec["url"] == URL
    assert res["saved"][0]["saved"] is True


def test_an_invented_quote_does_not_become_a_formula(gh, monkeypatch):
    _wire_web(monkeypatch, read={**READ, "quote": "The contact stress is sigma_H = 42 MPa for all gears"})
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    assert res["items"][0]["formula"] == "could not read the formula reliably"


def test_a_forum_or_blog_is_never_a_source(gh, monkeypatch):
    _wire_web(monkeypatch, hit_url="https://www.eng-tips.com/viewthread.cfm?qid=1")
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    assert res["mode"] == "none" and res["items"] == []
    assert gh.writes() == []


def test_the_same_reference_is_not_searched_or_stored_twice(gh, monkeypatch):
    _wire_web(monkeypatch)
    first = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    assert first["saved"][0]["saved"] is True and len(gh.writes()) == 1

    def no_search(*a, **k):
        raise AssertionError("an existing candidate must not be searched again")
    monkeypatch.setattr(search, "search", no_search)
    second = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    assert second["mode"] == "candidate" and second["searched"] is False
    assert len(second["items"]) == 1 and len(gh.writes()) == 1

    again = db.save(second["items"][0])
    assert again["saved"] is False and again["existing"] is True
    assert len(gh.writes()) == 1


# ------------------------------------------------------------------ 2. a candidate is never a result
def test_a_candidate_carries_no_result_field_and_its_numbers_stay_unverified(gh, monkeypatch):
    _wire_web(monkeypatch)
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    for forbidden in ("cad", "params", "requirements", "variants", "ranking", "fea", "mass_kg", "safety_factor"):
        assert forbidden not in res
    assert res["items"][0]["status"] == "candidate"

    # the gate still treats a candidate's number as a guess: it can never enter an accepted result
    eng = ENG.solve("torsi baut M10 kelas 8.8, preload 75%")
    v = ENG.verify("faktor 3.1416 dipakai untuk semua hasil", eng)
    assert v["ok"] is False and v["unverified"]


# ------------------------------------------------------------------ GitHub down / no token
def test_github_down_still_answers_and_says_the_reference_was_not_stored(gh, monkeypatch):
    _wire_web(monkeypatch)
    gh.fail = True
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    assert res["mode"] == "candidate_new" and res["items"]
    assert res["saved"][0]["saved"] is False
    assert "unreachable" in res["saved"][0]["reason"]
    assert "could not be stored" in flow.reply_note(res)
    assert "Unverified formula" in res["block"]


def test_without_a_token_the_chat_answers_and_nothing_is_written(gh, monkeypatch):
    _wire_web(monkeypatch)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    db.reset_cache()
    res = flow.for_request("rumus tegangan kontak gigi menurut ISO 6336", None, "claude")
    assert res["items"] and res["mode"] == "candidate_new"
    assert res["saved"][0]["saved"] is False and "GITHUB_TOKEN" in res["saved"][0]["reason"]
    assert gh.writes() == []
    assert "GITHUB_TOKEN" in flow.reply_note(res)


# ------------------------------------------------------------------ main is protected in code
def test_a_write_to_main_is_refused_by_the_code(gh):
    rec = {"topic": "weld", "title": "Fillet weld stress", "url": "https://www.iso.org/x",
           "formula": "tau = F / (a * l)"}
    out = db.save(rec, branch="main")
    assert out["saved"] is False and "main" in out["reason"]
    assert db.refuse_protected("main") and db.refuse_protected("MASTER")
    assert db.refuse_protected("reference-db") is None
    assert gh.writes() == []


def test_the_branch_can_be_configured_but_main_is_still_refused(gh, monkeypatch):
    monkeypatch.setenv("IDRIS_REFERENCE_BRANCH", "main")
    db.reset_cache()
    out = db.save({"topic": "weld", "title": "x", "url": "https://www.iso.org/y", "formula": "tau = F / A"})
    assert out["saved"] is False and "main" in out["reason"]
    assert gh.writes() == []


# ------------------------------------------------------------------ the endpoints
def test_list_endpoint_shows_slug_topic_title_status_url(gh):
    db.save({"topic": "weld", "title": "Fillet weld", "url": "https://www.iso.org/w", "formula": "tau = F / (a l)",
             "source_name": "ISO 2553"})
    body = db.public_list()
    assert body["count"] == 1 and body["stored"] is True and body["error"] is None
    row = body["references"][0]
    assert {"slug", "topic", "title", "status", "url"} <= set(row)
    assert row["status"] == "candidate" and row["slug"] == "weld-fillet-weld"


def test_one_reference_endpoint_returns_every_field_and_the_label(gh):
    slug = db.save({"topic": "weld", "title": "Fillet weld", "url": "https://www.iso.org/w",
                    "formula": "tau = F / (a l)", "source_name": "ISO 2553"})["slug"]
    rec = db.public_one(slug)
    assert set(db.FIELDS) <= set(rec)
    assert rec["formula"] == "tau = F / (a l)" and rec["status"] == "candidate"
    assert rec["label"].startswith("**Unverified formula from ISO 2553")
    assert db.public_one("../" * 2 + "etc/passwd") is None
    assert db.public_one("nope") is None


def test_the_code_cannot_promote_a_reference(gh):
    """Only a human may verify: a record sent with status 'verified' is written as a candidate."""
    out = db.save({"topic": "weld", "title": "x", "url": "https://www.iso.org/z",
                   "formula": "tau = F / A", "status": "verified", "verified_by": "gpt"})
    assert out["saved"] is True
    stored = gh.files["reference-db"][f"{out['slug']}.json"]
    assert stored["status"] == "candidate" and stored["verified_by"] is None


# ------------------------------------------------------------------ wiring in main.py
def test_main_py_exposes_the_reference_routes_and_runs_the_lookup():
    src = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
    assert '"GET /references/db"' in src and '"GET /references/db/<slug>"' in src
    assert 'self.path.split("?")[0].rstrip("/") == "/references/db"' in src
    assert 'self.path.split("?")[0].startswith("/references/db/")' in src
    chat = src[src.index('if base == "/chat"'):src.index('if base == "/catalog/bolt-check"')]
    assert "reference_flow as _REF" in chat and "_REF.for_request(msg, eng" in chat
    assert 'ctx["references"] = refs["block"]' in chat
    assert 'res["references"] = {k: refs.get(k)' in chat
