"""AI reads the request first (v0.34). One short AI call decides WHO in IDRIS handles a new sentence:

    library      one standard part from the IDRIS library (bolt, bearing, spur gear, plow blade, ...)
    primitive    a plain box / cylinder / tube / plate, nothing else
    gearbox      the gearbox workflow (ACE variants, ISO 6336)
    assembly     a whole product of several separate parts: planned as parts, then designed part by part
    free_design  any other part: the AI CAD designer, then kernel checks, FEA, variants
    other        a question, or an edit of the open design

Before v0.34 this was decided by keyword rules, which kept failing on real sentences ("pillow block" -> block,
"square tube" table frame -> tube, "door key" -> machine key). The AI only chooses the route and copies values the
sentence states. It computes nothing: every value it passes on is checked against the sentence, and the library /
solver validates it again. If the AI is unreachable or answers badly, the keyword rules decide as before.
"""
from __future__ import annotations

import os
import re

import llm
from physics import parts_library as plib

ROUTES = ("library", "primitive", "gearbox", "assembly", "free_design", "other")


def enabled() -> bool:
    return os.environ.get("IDRIS_AI_ROUTER", "1").strip().lower() not in ("0", "false", "off", "no")


def _library_lines() -> str:
    out = []
    for tid, t in plib.TEMPLATES.items():
        keys = ", ".join(f"{i['k']} = {str(i.get('label') or i['k']).split(' (')[0]}" for i in plib.inputs_for(tid))
        out.append(f"- {tid}: {t['name']}" + (f" ({t['standard']})" if t.get("standard") else "") + f" | inputs: {keys}")
    return "\n".join(out)


def _system() -> str:
    return ("""You are the first reader in IDRIS, an engineering design tool. Read ONE sentence from an engineer and decide
who handles it. Output ONE JSON object only, no prose:
{"route": "library"|"primitive"|"gearbox"|"assembly"|"free_design"|"other", "library_id": string|null, "query": {},
 "concepts": [], "reason": "max 12 words"}

Routes:
- "library": the request IS one standard part from the list below, as a part on its own. NOT a part designed for or
  around a standard part ("bushing for bearing 6205", "holder for an M6 nut", "pillow block for a 6205" are free_design).
- "primitive": ONLY a plain box/cube, solid cylinder, tube/ring, or flat plate with a simple hole pattern, and
  nothing else in the sentence. Any frame, weldment, slot, step, thread grid, pocket, rib or named product is free_design.
- "gearbox": design, size or change a gearbox / reducer / gear pair by power, ratio, speed or torque.
- "assembly": a whole product built from several separate parts that have to fit each other (a bicycle, a cart, a
  robot arm, a test stand, a machine frame with wheels and drive). Use this for "whole ..." / "complete ..." /
  "entire ..." / "seluruh ..." and for any product name that is clearly more than one part. NOT for one part of a
  product: "bicycle frame 540 mm" is free_design, and a gearbox is gearbox.
- "free_design": any other single physical part to design (bracket, housing, frame, fixture, jaw, hook, stand,
  enclosure, impeller, ...), in any language.
- "other": a question, a greeting, or a change to the design that is already open ("make it lighter").

"concepts" (free_design only): 3 design concepts that an experienced engineer would compare for this part, clearly
different in SHAPE or construction, not only in size (e.g. for a bracket: "L-bracket with a gusset", "U-channel",
"lightened plate with ribs"). Each: {"name": "max 4 words", "idea": "one short sentence, no numbers"}. Every concept
must meet the request; never change what the engineer fixed (sizes, material, process, hole pattern).

"query" (library only): keys from that part's inputs. Copy values exactly as the sentence states them; convert cm, m
and inch to mm. NEVER add a value the sentence does not give: missing inputs are asked from the engineer later.

IDRIS library:
""" + _library_lines())


def _clean_concepts(raw) -> list:
    """-> up to 3 {name, idea}; numbers are removed so no size can travel from the AI's idea into the design."""
    out, seen = [], set()
    for c in (raw if isinstance(raw, list) else [])[:5]:
        if not isinstance(c, dict):
            continue
        name = re.sub(r"\s+", " ", re.sub(r"\d+(?:[.,]\d+)?", "", str(c.get("name") or ""))).strip(" -:,.")[:40]
        idea = re.sub(r"\s+", " ", re.sub(r"\d+(?:[.,]\d+)?", "", str(c.get("idea") or ""))).strip()[:160]
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "idea": idea})
    return out[:3]


def _clean_query(tid: str, q, message: str) -> dict:
    """Keep only inputs of that part whose value the sentence really states."""
    if not isinstance(q, dict):
        return {}
    keys = {i["k"] for i in plib.inputs_for(tid)} | {"keyway", "key_width"}
    low = (message or "").lower()
    nums = set()                                   # numbers as written, plus cm / m / inch written WITH that unit
    for tok, unit in re.findall(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(cm|mm|m|inch(?:es)?|in|\")?(?![a-z])", low):
        n = float(tok.replace(",", "."))
        nums.add(round(n, 3))
        f = {"cm": 10.0, "m": 1000.0, "inch": 25.4, "inches": 25.4, "in": 25.4, '"': 25.4}.get(unit)
        if f:
            nums.add(round(n * f, 3))
    out = {}
    for k, v in q.items():
        if k not in keys or v in (None, ""):
            continue
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
            continue
        s = str(v).strip()
        try:
            x = float(s.replace(",", "."))
        except ValueError:
            if s.lower() in low or s.lower().replace(" ", "") in low.replace(" ", ""):
                out[k] = s                        # "M6", "6205", "S355": text that is in the sentence
            continue
        if round(x, 3) in nums:
            out[k] = f"{x:g}"
    return out


def route(message: str, has_gearbox: bool = False, provider=None):
    """-> {"route", "library_id", "query", "reason", "read_by"} or None (AI off, unreachable or unusable answer)."""
    if not enabled() or not (message or "").strip():
        return None
    order = [provider] if provider and provider != "auto" else llm.provider_order()
    order = [p for p in order if p in llm.KEY_ENV and os.environ.get(llm.KEY_ENV[p])]
    if not order:
        return None
    user = (f"Sentence: {message.strip()[:800]}\nA gearbox design is open in the workspace: {'yes' if has_gearbox else 'no'}")
    try:
        text, used = llm.call_provider(order[0], _system(), user, 450,
                                       timeout=float(os.environ.get("IDRIS_ROUTER_TIMEOUT_S", "12")))
        raw = llm.extract_json(text)
    except Exception:
        return None
    if not isinstance(raw, dict) or raw.get("route") not in ROUTES:
        return None
    out = {"route": raw["route"], "library_id": None, "query": {}, "reason": str(raw.get("reason") or "")[:120],
           "read_by": f"{order[0]} ({used})", "concepts": []}
    if out["route"] == "free_design":
        out["concepts"] = _clean_concepts(raw.get("concepts"))
    if out["route"] == "library":
        tid = raw.get("library_id")
        if tid not in plib.TEMPLATES:
            out["route"] = "free_design"          # a "standard part" IDRIS does not have: design it
        else:
            q = _clean_query(tid, raw.get("query"), message)
            rule = plib.match_sentence(message)            # where the built-in reader agrees on the part, its values win
            if rule and rule[0] == tid:
                q = {**q, **rule[1]}
            out["library_id"], out["query"] = tid, q
    return out
