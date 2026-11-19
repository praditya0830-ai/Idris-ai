"""Route hook: routes that live in their own small module register themselves here.

Why: main.py is large and every section of the task document that adds a route would otherwise edit the same
lines (the /health list and the POST_ROUTES tuple), so two open pull requests would always conflict. main.py
asks this module first, and a section adds one small file instead:

    shape_ext.py      Bagian H  (goals -> measurable objectives, shape families)
    assembly_ext.py   Bagian E  (POST /assembly/design)
    report_ext.py     Bagian D  (POST /report/part, 2D drawings, STEP in)

A module may define `POST_ROUTES` (a tuple of paths it answers), `handle_get(path)` and
`handle_post(base, params)`, each returning `(status, payload)` or None. A missing module is simply skipped, so
the order in which the pull requests are merged does not matter.
"""
from __future__ import annotations

import importlib

MODULES = ("design_ext", "shape_ext", "assembly_ext", "report_ext")

# Review of PR #5-#8, item 7a: module routes that can call a paid provider. main.py rate limits these per IP
# before it hands the request over, exactly like /chat. A module may also name its own with AI_POST_ROUTES.
AI_POSTS = ("/assembly/plan", "/assembly/design", "/shape/families")


def _mods():
    """Yield the modules that exist in this checkout (a section that is not merged yet is skipped)."""
    for name in MODULES:
        try:
            yield importlib.import_module(name)
        except ImportError:
            continue


def calls_ai(base: str) -> bool:
    """True when a module route can spend AI money: the caller rate limits it per IP first."""
    if base in AI_POSTS:
        return True
    return any(base in tuple(getattr(m, "AI_POST_ROUTES", ())) for m in _mods())


def known_post(base: str) -> bool:
    """True when one of the modules answers this POST path: main.py's route gate asks this first."""
    for m in _mods():
        if base in tuple(getattr(m, "POST_ROUTES", ())):
            return True
    return False


def handle_get(path: str):
    for m in _mods():
        fn = getattr(m, "handle_get", None)
        if fn is None:
            continue
        res = fn(path)
        if res is not None:
            return res
    return None


def handle_post(base: str, params: dict):
    for m in _mods():
        fn = getattr(m, "handle_post", None)
        if fn is None:
            continue
        res = fn(base, params)
        if res is not None:
            return res
    return None
