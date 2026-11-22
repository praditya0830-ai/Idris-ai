"""Bagian F route: the target check on its own, so the workspace can ask before it designs.

GET /design/target-check?q=sayap +50% kecepatan dan -50% bahan bakar
  -> target_check: {targets, verdict, reason, what_is_checkable, note}

No AI, no number invented: the verdict comes from the laws written in cad_fallback.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import cad_fallback

POST_ROUTES: tuple = ("/references/design",)


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
    """POST /references/design {brief, project_id} -> 3-6 reference cards for stage 2 (v0.41, doc A8).

    No AI call: the cards are cut from the search snippets, so the step spends nothing from ai_budget. A missing
    provider, a 402 or an empty answer comes back as one honest line with `ok: false`, never as a silent success.
    """
    if str(base or "").split("?")[0].rstrip("/") != "/references/design":
        return None
    from research import design_refs
    brief_in = params.get("brief") if isinstance(params.get("brief"), dict) else {}
    if not brief_in and isinstance(params.get("message"), str):
        brief_in = {"object": params["message"][:120]}
    try:
        r = design_refs.run(brief_in, params.get("project_id"))
    except Exception as e:                      # a search outage must never break the design path
        return 200, {"success": False, "ok": False, "cards": [], "queries": [], "patterns": [],
                     "proposal": "", "steps": [], "mode": "none",
                     "error": f"reference search failed: {type(e).__name__}: {e}"[:200]}
    return 200, {"success": True, **r}
