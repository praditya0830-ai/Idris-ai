"""The design brief: IDRIS talks like an engineer before it writes any code (v0.40).

Arya, 21 Nov: "IDRIS belum bertindak seperti engineer asisten yang interaktif, selalu terpaku dengan prompt."
The router already decides WHO handles a sentence. This module decides whether the sentence is enough to start at all:
what the object is and what it is for, the key sizes (or the space it must fit), and — when strength matters — the
load or service condition. When one of those is missing the route is `brief`, and the reply is a short engineering
conversation (at most three questions with real options, one to three feature suggestions), never a form and never a
number the engineer did not give.

Two rules run through every line here:
  * a number exists only when the engineer wrote it (or it is already in the brief they confirmed). The reader may
    fill words ("camera drone", "quiet"); a digit it invents is dropped by `merge`.
  * nothing is designed until the brief is ready AND the engineer says start. The brief lives in `context.brief`, so
    the Studio can show it, edit it field by field, and send it back every turn.

The AI is one small call per brief turn (IDRIS_BRIEF_MAX_TOKENS, 1800 by default, 30 s) through `ai_budget`.
When the AI is away, times out or answers with prose, a cut-off object or nothing at all, the caller does NOT slide
back into the old "design it anyway" path: `fallback_questions` writes the same three questions locally from
`missing()`, with common options and no invented number, and marks the brief `source: "fallback questions (AI
unavailable)"`. The failure itself travels in `brief_error` (provider, model, stop reason, length, first 300 chars).
"""
from __future__ import annotations

import json
import os
import re

import ai_budget
import llm

FIELDS = ("object", "purpose", "kind", "envelope", "key_sizes", "loads", "material", "process", "quantity",
          "standards", "features", "constraints", "assumptions", "decided", "open", "needs")
SCALARS = ("object", "purpose", "kind", "envelope", "key_sizes", "loads", "material", "process", "quantity",
           "standards", "constraints")
LISTS = ("features", "assumptions", "decided", "open", "needs")
CONCEPT_OPTIONS = (3, 5, 7, 10)

# "just design it with your assumptions" in the languages engineers actually type
PROCEED = re.compile(r"(langsung( aja| saja)?( desain| design| buat| kerjakan)?|desain(kan)? (aja|saja)"
                     r"|buat(kan)? (aja|saja)|just (design|build|do) it|go ahead|use your (own )?assumptions?"
                     r"|with your assumptions|lanjut( aja| saja)?|kerjakan( aja| saja)?)", re.I)
START = re.compile(r"^(ya|yes|yep|ok|oke|okay|siap|go|start|mulai|lanjut|gas|kerjakan|desain(kan)?|buat(kan)?"
                   r"|start design|plan the assembly|rencanakan rakitan)\b", re.I)
_LOAD_WORDS = re.compile(r"\b(load|loads|force|beban|gaya|menahan|menopang|strength|kekuatan|safety factor"
                         r"|faktor aman)\b|\d+(?:[.,]\d+)?\s*(?:n|kn)\b", re.I)
_ID_WORDS = re.compile(r"\b(buat|buatkan|aku|saya|yang|dan|untuk|dengan|desain|konsep|mau|bisa|tolong|dari|ini|itu"
                       r"|gimana|kenapa|jadi|pengin|butuh|juga|aja|saja|banget|lebih)\b", re.I)
_UNIT = {"cm": 10.0, "m": 1000.0, "inch": 25.4, "inches": 25.4, "in": 25.4, '"': 25.4}
# a performance number is never IDRIS's to guess: time, distance, speed, energy, sound, share
_PERF = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:-|–|to )?\s*(?:min|mins|minute|minutes|menit|h|hr|hrs|hour|hours|jam"
                   r"|km|kilometer|kilometre|kilometers|mph|km/h|m/s|wh|kwh|db|dba|percent)\b\.?", re.I)


# --------------------------------------------------------------------------- the brief itself
def blank(kind: str = "part") -> dict:
    b = {k: "" for k in SCALARS}
    b.update({k: [] for k in LISTS})
    b["kind"] = kind if kind in ("part", "assembly") else "part"
    b["ready"] = False
    b["asked"] = False
    b["assumed"] = False
    b["source"] = ""            # "" / "AI" / "fallback questions (AI unavailable)"
    return b


def from_context(ctx) -> dict | None:
    """The brief the Studio sends back, cleaned; None when there is none."""
    if not isinstance(ctx, dict):
        return None
    b = ctx.get("brief")
    if not isinstance(b, dict) or not any(b.get(k) for k in SCALARS + LISTS):
        return None
    out = blank(b.get("kind"))
    for k in SCALARS:
        out[k] = str(b.get(k) or "")[:400]
    for k in LISTS:
        out[k] = [x for x in (b.get(k) or []) if isinstance(x, (dict, str))][:20]
    out["ready"] = bool(b.get("ready"))
    out["asked"] = bool(b.get("asked"))
    out["assumed"] = bool(b.get("assumed"))
    out["source"] = str(b.get("source") or "")[:120]
    return out


def _stated(text: str) -> set:
    """Every number the sentence itself states (mm, plus cm / m / inch written with that unit)."""
    out = set()
    low = str(text or "").lower()
    for tok, unit in re.findall(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(cm|mm|m|inch(?:es)?|in|\")?(?![a-z])", low):
        n = float(tok.replace(",", "."))
        out.add(round(n, 3))
        f = _UNIT.get(unit or "")
        if f:
            out.add(round(n * f, 3))
    return out


def _numbers(value) -> set:
    if isinstance(value, str):
        return {round(float(x.replace(",", ".")), 3) for x in re.findall(r"\d+(?:[.,]\d+)?", value)}
    return set()


def _grounded(value, allowed: set) -> bool:
    """A value with digits is kept only when every digit was stated. Words are always allowed."""
    return _numbers(value) <= allowed


def merge(prev, new, message: str = "") -> dict:
    """Add what the reader/engineer said to the brief. Never invents a digit, never drops a decided field."""
    b = blank((prev or {}).get("kind"))
    b.update({k: v for k, v in (prev or {}).items() if k in b})
    allowed = _stated(message)
    for k in SCALARS:
        allowed |= _numbers(b.get(k))
    for item in b.get("assumptions") or []:
        if isinstance(item, dict):
            allowed |= _numbers(item.get("value"))
    incoming = new if isinstance(new, dict) else {}
    for k in SCALARS:
        v = incoming.get(k)
        if v in (None, "", []):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = f"{v:g}"
        v = re.sub(r"\s+", " ", str(v)).strip()[:400]
        if not v or not _grounded(v, allowed):
            continue
        b[k] = v
    for k in LISTS:
        add = incoming.get(k)
        if not isinstance(add, list):
            continue
        for x in add:
            if isinstance(x, str):
                x = {"text": x}
            if not isinstance(x, dict):
                continue
            x = {kk: vv for kk, vv in x.items() if isinstance(vv, (str, int, float, bool))}
            if k != "assumptions" and not _grounded(json.dumps(x, ensure_ascii=False), allowed):
                continue
            if x and x not in b[k]:
                b[k].append(x)
    return b


def needs_load(b: dict) -> bool:
    if b.get("loads") or "loads" in (b.get("needs") or []):
        return True
    text = " ".join(str(b.get(k) or "") for k in ("object", "purpose", "constraints", "key_sizes"))
    text += " " + " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in (b.get("features") or []))
    return bool(_LOAD_WORDS.search(text))


def missing(b: dict) -> list:
    """What still stops IDRIS from starting, in the order the brief asks for it."""
    out = []
    if not b.get("object"):
        out.append("object")
    if not b.get("purpose"):
        out.append("purpose")
    if not (b.get("key_sizes") or b.get("envelope")):
        out.append("sizes")
    if needs_load(b) and not b.get("loads"):
        out.append("loads")
    return out


def with_assumptions(b: dict) -> dict:
    """'just design it': keep going, and write down every gap as an assumption the engineer can see."""
    b = dict(b)
    names = {"object": "what the part is", "purpose": "what it is for", "sizes": "key sizes / envelope",
             "loads": "load or service condition"}
    have = {a.get("what") for a in b.get("assumptions") or [] if isinstance(a, dict)}
    for k in missing(b):
        if k in have:
            continue
        b.setdefault("assumptions", []).append({
            "what": k, "value": names.get(k, k),
            "why": "the engineer asked IDRIS to design with its own assumptions", "assumed": True})
    b["ready"] = True
    b["assumed"] = True
    b["asked"] = False
    return b


# --------------------------------------------------------------------------- language
def language(message: str, forced=None) -> str:
    if str(forced or "").lower().startswith("id"):
        return "id"
    if str(forced or "").lower().startswith("en"):
        return "en"
    return "id" if len(_ID_WORDS.findall(str(message or ""))) >= 2 else "en"


_T = {
    "en": {
        "ask": "Shall I start? How many concepts: 3 / 5 / 7 / 10?",
        "or": 'Or say "just design it with your assumptions" and every number I pick is marked assumed.',
        "ideas": "Worth adding",
        "want": "What decides the design",
        "assumed": "Assumed (not given)",
        "start": "Start design",
        "plan": "Plan the assembly",
    },
    "id": {
        "ask": "Mulai? Berapa konsep: 3 / 5 / 7 / 10?",
        "or": 'Atau bilang "langsung desain dengan asumsimu": setiap angka yang aku pilih ditandai assumed.',
        "ideas": "Saran yang relevan",
        "want": "Yang paling menentukan",
        "assumed": "Asumsi (ditandai assumed)",
        "start": "Mulai desain",
        "plan": "Rencanakan rakitan",
    },
}


def _clean_text(s, limit=400) -> str:
    """No performance number survives: IDRIS may suggest a feature, never a claim like 'flies 25 minutes'."""
    s = _PERF.sub("", str(s or ""))
    s = re.sub(r"\s{2,}", " ", s).strip(" .,-")
    return s[:limit]


def _clean_questions(raw, lang: str) -> list:
    out = []
    for q in (raw if isinstance(raw, list) else [])[:3]:
        if not isinstance(q, dict):
            continue
        text = _clean_text(q.get("text"), 180)
        opts = []
        for o in (q.get("options") if isinstance(q.get("options"), list) else [])[:4]:
            if not isinstance(o, dict):
                continue
            label = _clean_text(o.get("label"), 80)
            note = _clean_text(o.get("note"), 120)
            if label:
                opts.append({"label": label, "note": note})
        if text and len(opts) >= 2:
            # A4b: every question carries the stage it belongs to, so the Studio shows it with the stage and the
            # colleague layer never asks a stage-5 question during the brief
            out.append({"id": str(q.get("id") or f"q{len(out) + 1}")[:24], "text": text, "options": opts,
                        "allow_free_text": True, "stage": 1})
    return out


def _clean_suggestions(raw) -> list:
    out, seen = [], set()
    for s in (raw if isinstance(raw, list) else [])[:3]:
        if isinstance(s, str):
            s = {"text": s}
        if not isinstance(s, dict):
            continue
        text = _clean_text(s.get("text"), 120)
        why = _clean_text(s.get("why"), 160)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append({"id": str(s.get("id") or f"s{len(out) + 1}")[:24], "text": text, "why": why})
    return out


# --------------------------------------------------------------------------- the AI call
def _order(provider=None) -> list:
    order = [provider] if provider and provider != "auto" else llm.provider_order()
    return [p for p in order if p in llm.KEY_ENV and os.environ.get(llm.KEY_ENV[p])]


def _system(lang: str) -> str:
    return f"""You are a senior mechanical engineer at IDRIS, talking with another engineer. Read the request and fill
the design brief. Output ONE JSON object only, no prose:
{{"understanding": "one or two sentences that show you understood the request",
 "brief": {{"object": "", "purpose": "", "envelope": "", "key_sizes": "", "loads": "", "material": "", "process": "",
            "quantity": "", "standards": "", "constraints": "", "features": [], "needs": []}},
 "questions": [{{"id": "q1", "text": "", "options": [{{"label": "", "note": ""}}]}}],
 "suggestions": [{{"id": "s1", "text": "", "why": ""}}]}}

Rules:
- Write everything in {"Indonesian" if lang == "id" else "English"}, the language the engineer used. Technical terms
  may stay in English.
- At most THREE questions, the ones that change the design most. Each has 2-4 real options an experienced engineer
  would actually choose between, and each option's note is a short qualitative trade-off (fast/stable/cheap/stiff...).
  Never a form, never a numbered checklist, never a question whose answer you can assume.
- One to three feature suggestions for THIS object, each with why it matters. No performance claims and no
  performance numbers (never "flies 25 minutes", "lasts 8 hours", "reaches 40 km/h").
- Numbers: only a size or class the engineer wrote, or a common size class that names the option itself
  ("5-inch", "450 mm", "750 W"). Never invent an envelope, a load, a weight or a strength value.
- Fill `brief` only with what the engineer stated in this turn or earlier. Leave everything else "" or [].
- `needs` lists extra facts this object really needs ("loads" when strength matters, "standards" for a thread).
- Never write CAD code, never promise a design, never say the design has started.
"""


def _user(message: str, b: dict, miss: list) -> str:
    return ("Engineer says: " + str(message or "").strip()[:800]
            + "\n\nBrief so far: " + json.dumps({k: b.get(k) for k in SCALARS if b.get(k)}, ensure_ascii=False)
            + "\nAlready decided: " + json.dumps(b.get("decided") or [], ensure_ascii=False)
            + "\nStill missing: " + (", ".join(miss) or "nothing")
            + ("\nThe engineer already chose to continue with assumptions." if b.get("assumed") else ""))


DEFAULT_MAX_TOKENS = 1800
DEFAULT_TIMEOUT_S = 30.0
# one retry, with an instruction the model cannot read as "write a friendly paragraph"
RETRY_NOTE = "\n\nReturn ONLY one JSON object, no prose and no markdown fence."


def max_tokens() -> int:
    try:
        return max(400, int(os.environ.get("IDRIS_BRIEF_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get("IDRIS_BRIEF_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _read_json(text: str):
    """The answer as a dict: strict JSON first, then the same repair the CAD writer uses, so a cut-off object
    (max_tokens) still yields its questions instead of losing the whole turn."""
    try:
        raw = llm.extract_json(text)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    try:
        raw = llm.salvage_json(text)
    except Exception:
        raw = None
    return raw if isinstance(raw, dict) else None


def _meta() -> dict:
    try:
        return llm.last_meta()
    except Exception:
        return {}


def write_detail(message: str, b: dict, miss: list, provider=None, lang: str = "en") -> tuple:
    """One small AI call (plus one stricter retry) that writes the engineer's next question.

    -> (answer, error). `answer` is None when the AI is away, timed out, or answered with something that is not a
    usable brief object; `error` then says provider, model, stop reason, length and the first 300 characters, plus
    every attempt, so the server log and the JSON can show the real cause instead of "the AI did not return
    structured data".
    """
    order = _order(provider)
    gate = ai_budget.gate() if hasattr(ai_budget, "gate") else None
    if not order or gate:
        why = str(gate or "no AI provider is configured on the server")
        return None, {"provider": (order[0] if order else None), "model": None, "stop_reason": None,
                      "chars": 0, "head": "", "error": why[:300], "attempts": []}
    prov = order[0]
    system, user = _system(lang), _user(message, b, miss)
    attempts = []
    for attempt in (1, 2):
        text, used, stop = "", None, None
        try:
            text, used = llm.call_provider(prov, system, user + (RETRY_NOTE if attempt == 2 else ""),
                                           max_tokens(), timeout=timeout_s())
            stop = _meta().get("stop_reason")
        except Exception as e:
            meta = _meta()
            attempts.append({"provider": prov, "model": meta.get("model"), "stop_reason": meta.get("stop_reason"),
                             "chars": 0, "head": "", "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        raw = _read_json(text)
        if isinstance(raw, dict) and any(raw.get(k) for k in ("understanding", "brief", "questions", "suggestions")):
            qs = _clean_questions(raw.get("questions"), lang)
            # an answer with no usable question while the brief is open is a failure too: say why the questions
            # were dropped (usually one option, or a cut-off list), instead of showing an empty turn
            err = None if qs else {"provider": prov, "model": _meta().get("model") or used, "stop_reason": stop,
                                   "chars": len(text or ""), "head": (text or "")[:300],
                                   "error": "the AI answer carried no usable question (each needs a text and "
                                            "2-4 options)"}
            return {"understanding": _clean_text(raw.get("understanding"), 400),
                    "brief": raw.get("brief") if isinstance(raw.get("brief"), dict) else {},
                    "questions": qs, "suggestions": _clean_suggestions(raw.get("suggestions")),
                    "read_by": f"{prov} ({used})"}, err
        attempts.append({"provider": prov, "model": _meta().get("model") or used, "stop_reason": stop,
                         "chars": len(text or ""), "head": (text or "")[:300],
                         "error": "the answer was not a JSON object with a brief, questions or suggestions"})
    last = attempts[-1] if attempts else {}
    return None, {**last, "attempts": attempts}


def write(message: str, b: dict, miss: list, provider=None, lang: str = "en") -> dict | None:
    """One small AI call that writes the engineer's next question. None when the AI is away or unusable;
    `write_detail` carries the reason. Zero questions is allowed: the answer may simply complete the brief."""
    w, _err = write_detail(message, b, miss, provider, lang)
    return w


# --------------------------------------------------------------------------- when the AI cannot write the questions
# Fixed questions, one per missing field, with options an engineer would recognise. No option carries a digit: the
# fallback may ask for a size or a load, it may never suggest one.
_FALLBACK = {
    "en": {
        "object": ("What should IDRIS design?", [
            ("Bracket or mount", "holds another part in place"),
            ("Enclosure or housing", "protects what is inside"),
            ("Shaft or axle", "carries a rotating load"),
            ("Frame", "carries the main structure")]),
        "purpose": ("What will it be used for?", [
            ("Indoor / bench use", "light service, no weather"),
            ("Outdoor / mobile", "weather and vibration"),
            ("Machine or vehicle", "continuous load"),
            ("Display or prototype", "form first, light load")]),
        "sizes": ("Roughly how big?", [
            ("Palm-sized", "fits in one hand, or give a size"),
            ("Desk-sized", "sits on a desk, or give a size"),
            ("Person-sized", "about as tall as a person, or give a size"),
            ("Larger than a person", "or give a size")]),
        "loads": ("What load or weight must it carry?", [
            ("Hand load only", "nothing heavier than a hand"),
            ("A person's weight", "steps, sits or leans on it"),
            ("A vehicle load", "moves with a vehicle"),
            ("Give a number", "type the load with its unit")]),
    },
    "id": {
        "object": ("IDRIS harus mendesain apa?", [
            ("Bracket atau dudukan", "menahan part lain"),
            ("Enclosure atau housing", "melindungi isinya"),
            ("Poros atau axle", "membawa beban berputar"),
            ("Rangka", "membawa struktur utama")]),
        "purpose": ("Untuk apa?", [
            ("Dipakai di dalam / meja", "ringan, tanpa cuaca"),
            ("Luar ruangan / bergerak", "cuaca dan getaran"),
            ("Mesin atau kendaraan", "beban terus-menerus"),
            ("Pajangan atau prototipe", "bentuk dulu, beban ringan")]),
        "sizes": ("Kira-kira sebesar apa?", [
            ("Seukuran telapak", "muat di satu tangan, atau sebutkan ukurannya"),
            ("Seukuran meja", "muat di atas meja, atau sebutkan ukurannya"),
            ("Seukuran orang", "setinggi orang, atau sebutkan ukurannya"),
            ("Lebih besar dari orang", "atau sebutkan ukurannya")]),
        "loads": ("Beban atau berat apa yang harus ditahan?", [
            ("Beban tangan saja", "tidak lebih berat dari tangan"),
            ("Berat orang", "diinjak, diduduki, atau disandari"),
            ("Beban kendaraan", "ikut bergerak dengan kendaraan"),
            ("Sebutkan angkanya", "ketik beban beserta satuannya")]),
    },
}


def fallback_questions(b: dict, miss: list = None, lang: str = "en") -> dict:
    """The same three questions, written here instead of by the AI: never a number, never a silent design."""
    table = _FALLBACK["id" if lang == "id" else "en"]
    out = []
    for k in list(miss if miss is not None else missing(b))[:3]:
        text, opts = table.get(k, table["purpose"])
        out.append({"id": f"fallback_{k}", "text": text, "options": [{"label": a, "note": n} for a, n in opts],
                    "allow_free_text": True, "fallback": True, "stage": 1})
    return {"understanding": "", "brief": {}, "questions": out, "suggestions": [],
            "read_by": "fallback questions (no AI)"}


# --------------------------------------------------------------------------- the answers
def _route(kind: str, read_by=None) -> dict:
    r = {"route": "brief", "of": "assembly" if kind == "assembly" else "free_design"}
    if read_by:
        r["read_by"] = read_by
    return r


def _out(kind: str, answer: str, b: dict, read_by=None, lang: str = "en") -> dict:
    return {"success": True, "intent": "discuss", "answer": answer, "actions": [], "params": {}, "rejected": [],
            "select": None, "compare": None, "cad": None, "concepts": [], "provider": read_by or "IDRIS brief",
            "language": lang, "route": _route(kind), "brief": b, "questions": [], "suggestions": [],
            "ready": False}


def brief_answer(b: dict, w: dict, kind: str, lang: str) -> dict:
    """A short engineer reply: what IDRIS understood, the questions with their options, then the suggestions."""
    t = _T["id" if lang == "id" else "en"]
    parts = []
    if w.get("understanding"):
        parts.append(w["understanding"])
    if w.get("questions"):
        parts.append(t["want"] + ":")
        for i, q in enumerate(w["questions"], 1):
            opts = " / ".join(o["label"] + (f" ({o['note']})" if o.get("note") else "") for o in q["options"])
            parts.append(f"{i}. {q['text']}\n   {opts}")
    if w.get("suggestions"):
        parts.append(t["ideas"] + ": " + "; ".join(
            s["text"] + (f" — {s['why']}" if s.get("why") else "") for s in w["suggestions"]))
    parts.append(t["or"])
    out = _out(kind, "\n\n".join(parts)[:1800], b, w.get("read_by"), lang)
    out.update({"questions": w["questions"], "suggestions": w["suggestions"], "ready": False})
    return out


def _summary(b: dict, lang: str) -> str:
    lines = []
    head = b.get("object") or ("assembly" if b.get("kind") == "assembly" else "part")
    if b.get("purpose"):
        head += (" untuk " if lang == "id" else " for ") + b["purpose"]
    lines.append(head[:200])
    labels = [("key_sizes", "Ukuran" if lang == "id" else "Sizes"), ("envelope", "Ruang" if lang == "id" else "Space"),
              ("loads", "Beban" if lang == "id" else "Loads"), ("material", "Material" if lang == "id" else "Material"),
              ("process", "Proses" if lang == "id" else "Process"), ("quantity", "Jumlah" if lang == "id" else "Quantity")]
    got = [f"{lab}: {b[k]}" for k, lab in labels if b.get(k)]
    if got:
        lines.append(" · ".join(got))
    if b.get("assumptions"):
        lines.append((_T["id" if lang == "id" else "en"])["assumed"] + ": " + "; ".join(
            f"{a.get('what')} = {a.get('value')}" for a in b["assumptions"] if isinstance(a, dict))[:300])
    return "\n".join(lines)


def ready_answer(b: dict, kind: str, lang: str) -> dict:
    """The brief is enough: show it back and ask the one closing question (how many concepts / plan the assembly)."""
    t = _T["id" if lang == "id" else "en"]
    answer = _summary(b, lang) + "\n\n" + t["ask"]
    out = _out(kind, answer, b, lang=lang)
    out["ready"] = True
    if kind == "part":
        out["ask"] = {"k": "concept_count", "options": list(CONCEPT_OPTIONS), "default": 3}
    else:
        out["ask"] = {"k": "assembly_start", "options": [1], "default": 1}
    out["brief"] = {**b, "ready": True, "asked": True}
    return out


def as_request(b: dict) -> str:
    """The brief as one design request, so the code writer designs what the engineer decided (and nothing else)."""
    bits = [f"{b.get('object') or 'part'}" + (f" for {b['purpose']}" if b.get("purpose") else "")]
    for k, lab in (("key_sizes", "key sizes"), ("envelope", "envelope"), ("loads", "load"),
                   ("material", "material"), ("process", "process"), ("quantity", "quantity"),
                   ("standards", "standard"), ("constraints", "constraint")):
        if b.get(k):
            bits.append(f"{lab}: {b[k]}")
    feats = [f.get("text") if isinstance(f, dict) else str(f) for f in (b.get("features") or [])]
    feats = [x for x in feats if x]
    if feats:
        bits.append("wanted features: " + "; ".join(feats))
    text = "Design " + "; ".join(bits) + "."
    if b.get("assumptions"):
        text += ("\nAssumptions (each one is assumed, not given by the engineer; say so in the answer): "
                 + "; ".join(f"{a.get('what')} = {a.get('value')}" for a in b["assumptions"] if isinstance(a, dict)))
    return text[:1200]


def wants_assumptions(message: str) -> bool:
    return bool(PROCEED.search(str(message or "")))


def resume(ctx, message: str, concept_count=None) -> dict | None:
    """A ready brief + an explicit start: hand the design path the brief instead of asking again.

    Only an explicit start resumes (a count, 'start', 'ya', 'lanjut', 'plan the assembly'), so a free engineering
    question in the middle of the conversation still goes to the engineer chat and the brief stays in memory.
    """
    b = from_context(ctx)
    if not b or not b.get("ready"):
        return None
    n = concept_count
    if n is None:
        m = re.fullmatch(r"\s*(\d{1,2})\s*", str(message or ""))
        if m and 1 <= int(m.group(1)) <= 10:
            n = int(m.group(1))
    said = str(message or "").strip()
    if not (n or START.match(said)):
        return None
    b = {**b, "asked": False, "started": True}
    out = {"brief": b, "message": as_request(b), "kind": b.get("kind") or "part"}
    if n:
        out["concept_count"] = n
    return out
