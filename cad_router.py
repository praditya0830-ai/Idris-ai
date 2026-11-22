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

ROUTES = ("library", "primitive", "gearbox", "assembly", "free_design", "other", "brief")


def enabled() -> bool:
    return os.environ.get("IDRIS_AI_ROUTER", "1").strip().lower() not in ("0", "false", "off", "no")


def _library_lines() -> str:
    out = []
    for tid, t in plib.TEMPLATES.items():
        keys = ", ".join(f"{i['k']} = {str(i.get('label') or i['k']).split(' (')[0]}" for i in plib.inputs_for(tid))
        out.append(f"- {tid}: {t['name']}" + (f" ({t['standard']})" if t.get("standard") else "") + f" | inputs: {keys}")
    return "\n".join(out)


def _system(concept_count: int | None = None) -> str:
    n_concepts = max(1, min(10, int(concept_count or 3)))
    text = ("""You are the first reader in IDRIS, an engineering design tool. Read ONE sentence from an engineer and decide
who handles it. Output ONE JSON object only, no prose:
{"route": "brief"|"library"|"primitive"|"gearbox"|"assembly"|"free_design"|"other", "library_id": string|null,
 "query": {}, "concepts": [], "kind": "part"|"assembly", "brief": {}, "reason": "max 12 words"}

Routes:
- "brief": a new design (a part OR a whole product) that does not yet say enough to start. Judge this BEFORE the design
  routes: (a) what the object is AND what it is for, (b) its key sizes or the space/envelope it must fit, (c) when
  strength matters, the load or service condition. If any of these is missing, return "brief" — never free_design or
  assembly. Add "kind" ("part" or "assembly") and a "brief" object holding only what the sentence itself says:
  object, purpose, envelope, key_sizes, loads, material, process, quantity, standards, constraints, features.
  NEVER put a number in "brief" that the sentence does not contain: a missing size stays out, never a guess.
  A standard part, a gearbox, a primitive, a question or a change to the open design is never brief.
- "library": the request IS one standard part from the list below, as a part on its own. NOT a part designed for or
  around a standard part ("bushing for bearing 6205", "holder for an M6 nut", "pillow block for a 6205" are free_design).
- "primitive": ONLY a plain box/cube, solid cylinder, tube/ring, or flat plate with a simple hole pattern, and
  nothing else in the sentence. Any frame, weldment, slot, step, thread grid, pocket, rib or named product is free_design.
- "gearbox": design, size or change a gearbox / reducer / gear pair by power, ratio, speed or torque.
- "assembly": a whole product (enough information to start) built from several separate parts that have to fit each other (a bicycle, a cart, a
  robot arm, a test stand, a machine frame with wheels and drive). Use this for "whole ..." / "complete ..." /
  "entire ..." / "seluruh ..." and for any product name that is clearly more than one part. NOT for one part of a
  product: "bicycle frame 540 mm" is free_design, and a gearbox is gearbox.
- "free_design": any other single physical part to design (bracket, housing, frame, fixture, jaw, hook, stand,
  enclosure, impeller, ...), in any language, when the sentence already says what it is for and its key size.
- "other": a question, a greeting, or a change to the design that is already open ("make it lighter").

"concepts" (free_design only): @N_CONCEPTS@ design concepts that an experienced engineer would compare for this
part, clearly different in SHAPE or construction, not only in size (e.g. for a bracket: "L-bracket with a gusset",
"U-channel", "lightened plate with ribs"). Each: {"name": "max 4 words", "idea": "one short sentence, no
numbers"}. The concepts must be distinct from each other: no two may differ only by a size. Every concept must meet
the request; never change what the engineer fixed (sizes, material, process, hole pattern).

"query" (library only): keys from that part's inputs. Copy values exactly as the sentence states them; convert cm, m
and inch to mm. NEVER add a value the sentence does not give: missing inputs are asked from the engineer later.

IDRIS library:
""" + _library_lines())
    return text.replace("@N_CONCEPTS@", f"{n_concepts} design concepts")


def _clean_concepts(raw, limit: int = 3) -> list:
    """-> up to `limit` {name, idea}; numbers are removed so no size can travel from the AI's idea into the design."""
    out, seen = [], set()
    limit = max(1, min(10, int(limit or 3)))
    for c in (raw if isinstance(raw, list) else [])[:limit * 2]:
        if not isinstance(c, dict):
            continue
        name = re.sub(r"\s+", " ", re.sub(r"\d+(?:[.,]\d+)?", "", str(c.get("name") or ""))).strip(" -:,.")[:40]
        idea = re.sub(r"\s+", " ", re.sub(r"\d+(?:[.,]\d+)?", "", str(c.get("idea") or ""))).strip()[:160]
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "idea": idea})
    return out[:limit]


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


# --------------------------------------------------------------------------- whole products, not one solid
# A bicycle, a drone or a plane is a product of several parts. The router must never send one to the single-solid
# free_design path, and a new sentence about a different product must not read the previous design's context.
# Order matters: "sepeda motor" is a motorcycle, and the loop returns the first product that matches. "motor" alone
# is never a product: an electric motor, a motor mount and "dudukan motor" are parts of a machine, not a motorbike.
_PRODUCTS = (
    ("motorcycle", r"motorcycle|motorbike|sepeda motor|motor bebek|motor sport"),
    ("bicycle", r"\bsepeda\b|\bbicycle\b|\bbike\b|pushbike"),
    ("drone", r"\bdrone\b|quadcopter|\buav\b"),
    ("airplane", r"airplane|aircraft|aeroplane|aero plane|\bpesawat\b"),
    ("car", r"\bmobil\b|\bcar\b|passenger car|kendaraan penumpang"),
    ("chair", r"\bkursi\b|\bchair\b"),
    ("table", r"\bmeja\b"),
    ("robot", r"\brobot\b|robot arm|lengan robot"),
    ("cart", r"\bcart\b|trolley|gerobak|kereta dorong"),
    ("boat", r"\bboat\b|\bkapal\b|perahu"),
    ("scooter", r"scooter|skuter"),
)
# "plane" and "table" are ordinary engineering words (top plane, ISO table) and a product only when the sentence
# asks for one: "design a plane", "make a table". A bare "table" after a standard or data word is never a product.
_DESIGN_VERB = (r"(?:design|make|build|create|draw|model|construct|buat(?:kan|lah)?|desain(?:kan|lah)?|"
                r"rancang(?:lah)?|bikin(?:kan)?)")
_PLANE_PRODUCT = re.compile(r"\b" + _DESIGN_VERB + r"\b[^.\n]{0,24}?\bplane\b", re.I)
_TABLE_PRODUCT = re.compile(r"\b" + _DESIGN_VERB + r"\b[^.\n]{0,24}?\btable\b", re.I)
# One part OF a product is a part, not the product: "cart wheel bracket", "drone arm", "chair leg", "motor mount"
# and the other word order too: "mount plate for a drone". "arm" is left out for a robot, because a robot arm IS the
# product (robot arm 600 mm).
_PART_WORDS = (r"mount(?:ing)?|mount ?plate|bracket|plate|arm|leg|legs|wheel|wheels|housing|cover|frame|handlebar|"
               r"saddle|fork|chain|brake|pedal|stem|seat ?post|basket|rack|mudguard|blade|axle|hub|panel|dudukan|"
               r"braket|pelat|plat|lengan|kaki|roda|rangka|setang|sadel|garpu|rantai|rem|keranjang")
_PART_WORDS_OTHER = (r"mount(?:ing)?|mount ?plate|bracket|plate|leg|legs|wheel|wheels|housing|cover|frame|handlebar|"
                     r"saddle|fork|chain|brake|pedal|stem|seat ?post|basket|rack|mudguard|blade|axle|hub|panel|"
                     r"dudukan|braket|pelat|plat|lengan|kaki|roda|rangka|setang|sadel|garpu|rantai|rem|keranjang")
# A sentence that asks for a gearbox is a new topic even though a gearbox is not a "product": it has its own route
# with its own requirements, so it must not be answered as a fact inside another object's brief.
_GEARBOX_REQ = re.compile(r"\b(?:gearbox|gear ?box|reducer|reduser|reduction gear|gear pair)\b", re.I)
_ENOUGH = re.compile(r"\b(?:complete|whole|entire|full|seluruh|lengkap|utuh)\b", re.I)
_SIZE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?\s*(mm|cm|\bm\b|inch|in|\"|kg|n\b|km/h)", re.I)


def _is_a_part(low: str, name: str, pat: str) -> bool:
    """Is the product word here the whole product, or one part of it? Both word orders, one or two words apart."""
    parts = _PART_WORDS_OTHER if name == "robot" else _PART_WORDS
    # "cart wheel bracket", "chair leg", "bicycle frame", "drone arm 200 mm"
    if re.search(r"(?:" + pat + r")\s+(?:[\w-]+\s+){0,2}?(?:" + parts + r")\b", low):
        return True
    # "wheel bracket for a cart", "mount plate on a drone"
    if re.search(r"\b(?:" + parts + r")\s+(?:[\w-]+\s+){0,1}?(?:" + pat + r")", low):
        return True
    return False


def product_of(message: str) -> str | None:
    """The whole product the sentence names (bicycle, drone, airplane…), or None when it names no product.

    Deliberately narrow: it is the guard that stops a part being designed as one solid product, so a false product is
    worse than no product. "motor" (electric motor), a CAD "top plane" and an "ISO table" are not products.
    """
    low = str(message or "").lower()
    if not low.strip():
        return None
    for name, pat in _PRODUCTS:
        if not re.search(pat, low):
            continue
        if _is_a_part(low, name, pat):
            continue
        return name
    if _PLANE_PRODUCT.search(low):                   # "design a plane" -- but never "sketch on the top plane"
        return "airplane"
    if _TABLE_PRODUCT.search(low) and not _is_a_part(low, "table", r"\btable\b"):
        return "table"                                # "design a table" -- but never "ISO table", "table top"
    return None


def enough_info(message: str) -> bool:
    """Does the sentence already say enough to plan a product? It must name the product and give a size, a load or an
    envelope, or say the product is complete/whole. A bare "design a bicycle" is not enough: it is a brief."""
    low = str(message or "")
    return bool(_ENOUGH.search(low) or _SIZE.search(low))


def product_route(message: str, route_name: str | None) -> dict | None:
    """A whole product never goes to free_design/primitive/other: -> route dict for brief or assembly, else None."""
    if route_name not in (None, "free_design", "primitive", "other"):
        return None
    name = product_of(message)
    if not name:
        return None
    enough = enough_info(message)
    return {"route": "assembly" if enough else "brief", "kind": "assembly", "library_id": None, "query": {},
            "concepts": [], "brief": {"object": name, "kind": "assembly"},
            "reason": f"whole product ({name})" + ("" if enough else ", details missing"),
            "read_by": "product rule (whole product, not one solid)"}


def _saved_project(ctx: dict, key: str) -> dict:
    """The open project, small enough to travel back in `context_reset` so it is kept, not thrown away."""
    blk = ctx.get(key)
    if not isinstance(blk, dict):
        return {}
    out = {k: blk[k] for k in ("object", "purpose", "kind", "request", "title", "ready") if blk.get(k)}
    out["from"] = key
    if isinstance(blk.get("features"), list):
        out["features"] = blk["features"][:8]
    if isinstance(blk.get("missing"), list):
        out["missing"] = [str((m or {}).get("k") or m) for m in blk["missing"][:3]]
    return out


def topic_reset(message: str, ctx) -> dict | None:
    """A NEW sentence about a different product must not inherit the open design.

    -> {"reason", "from", "to", "previous"} when the context must be dropped, else None. Only a sentence that names a
    product can reset: "make it lighter" and "80 mm" are about the design that is open. The open project travels back
    in `previous` (and the caller keeps gearbox/engineering context), so a reset is a saved project, never a silent
    deletion. A context that cannot be read as a product is never dropped: a gearbox housing is not "replaced" by a
    sentence that mentions a part of an electric motor.
    """
    if not isinstance(ctx, dict):
        return None
    new = product_of(message)
    if not new and _GEARBOX_REQ.search(str(message or "")):
        new = "gearbox"
    if not new:
        return None
    old, old_kind = None, None
    for key, field in (("pending_cad", "request"), ("current_cad", "request"), ("brief", "object"),
                       ("assembly_plan", "object")):
        blk = ctx.get(key)
        if not isinstance(blk, dict):
            continue
        old = str(blk.get(field) or blk.get("object") or blk.get("title") or "") or None
        if old:
            old_kind = key
            break
    if not old_kind:
        return None                               # nothing open that could be replaced
    old_p = product_of(old or "")
    if not old_p and new != "gearbox":
        return None                               # the open context is not a product: keep it, never drop it
    if not old_p and _GEARBOX_REQ.search(old or ""):
        return None                               # a gearbox is open and stays open
    if old_p == new:
        return None
    return {"reason": f"new request is a different product ({new}, the open context was {old_p})",
            "from": old, "to": new, "previous": _saved_project(ctx, old_kind)}


def route(message: str, has_gearbox: bool = False, provider=None, concepts_wanted: int | None = None):
    """-> {"route", "library_id", "query", "reason", "read_by"} or None (AI off, unreachable or unusable answer).

    `concepts_wanted` (UI v66 B): how many concepts the engineer asked to compare (1-10). The same reader proposes
    them, so asking for seven is one question and seven genuinely different shapes, not three repeated.
    """
    if not enabled() or not (message or "").strip():
        return None
    order = [provider] if provider and provider != "auto" else llm.provider_order()
    order = [p for p in order if p in llm.KEY_ENV and os.environ.get(llm.KEY_ENV[p])]
    if not order:
        return None
    user = (f"Sentence: {message.strip()[:800]}\nA gearbox design is open in the workspace: {'yes' if has_gearbox else 'no'}"
            + (f"\nConcepts wanted: {max(1, min(10, int(concepts_wanted)))}" if concepts_wanted else ""))
    try:
        text, used = llm.call_provider(order[0], _system(concepts_wanted), user, 450,
                                       timeout=float(os.environ.get("IDRIS_ROUTER_TIMEOUT_S", "12")))
        raw = llm.extract_json(text)
    except Exception:
        return None
    if not isinstance(raw, dict) or raw.get("route") not in ROUTES:
        return None
    out = {"route": raw["route"], "library_id": None, "query": {}, "reason": str(raw.get("reason") or "")[:120],
           "read_by": f"{order[0]} ({used})", "concepts": [], "kind": None, "brief": {}}
    # a whole product is never designed as one solid, whatever the reader said (bicycle, drone, airplane, car…)
    prod = product_route(message, out["route"])
    if prod:
        prod["read_by"] = prod["read_by"] + " · " + out["read_by"]
        return prod
    if out["route"] == "brief":
        out["kind"] = "assembly" if raw.get("kind") == "assembly" else "part"
        out["brief"] = raw.get("brief") if isinstance(raw.get("brief"), dict) else {}
        return out
    if out["route"] == "free_design":
        out["concepts"] = _clean_concepts(raw.get("concepts"), concepts_wanted or 3)
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


def concepts_for(message: str, count: int = 3, provider=None) -> list:
    """n concept names for one sentence, from the same reader that routes it (UI v66 B).

    -> [{name, idea}] with at most `count` entries; [] when the reader cannot be reached, so the caller says
    "no concept could be proposed" instead of designing n copies of one shape. Nothing is padded: a request for
    seven that the reader answers with three gives three, and the reply reports the number that was really made.
    """
    n = max(1, min(10, int(count or 3)))
    out = route(message, False, provider, concepts_wanted=n) or {}
    concepts = [c for c in (out.get("concepts") or []) if isinstance(c, dict) and c.get("name")][:n]
    for c in concepts:
        c.setdefault("read_by", out.get("read_by"))
    return concepts
