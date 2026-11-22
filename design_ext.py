"""Bagian F route: the target check on its own, so the workspace can ask before it designs.

GET /design/target-check?q=sayap +50% kecepatan dan -50% bahan bakar
  -> target_check: {targets, verdict, reason, what_is_checkable, note}

No AI, no number invented: the verdict comes from the laws written in cad_fallback.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import cad_fallback

POST_ROUTES: tuple = ("/join/plan",)


def handle_get(path: str):
    p = path.split("?")[0].rstrip("/")
    if p != "/design/target-check":
        return None
    q = (parse_qs(urlparse(path).query).get("q") or [""])[0].strip()[:600]
    if not q:
        return 400, {"success": False, "error": "add ?q=your sentence, e.g. /design/target-check?q=ringan 20%"}
    tc = cad_fallback.target_check(q)
    if tc is None:
        return 200, {"success": True, "target_check": None,
                     "note": "no performance target was found in that sentence"}
    return 200, {"success": True, "target_check": tc}


def handle_post(base: str, params: dict):
    """POST /join/plan {parts, interfaces, bolt_size, fix} -> joints, geometry check, BOM/cut list/welds/bolts (v0.42).

    No `bolt_size` means the joint says M8 is assumed; no `fix` means the overlap shift is a proposal, not an edit.
    """
    if str(base or "").split("?")[0].rstrip("/") != "/join/plan":
        return None
    import joining
    parts = params.get("parts") if isinstance(params.get("parts"), list) else []
    return 200, {"success": True, **joining.plan(parts, params.get("interfaces"),
                                                  params.get("bolt_size") or None,
                                                  bool(params.get("fix", False)))}
