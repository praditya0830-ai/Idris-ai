"""Bagian F: a design never comes home empty.

Arya's rule: the design has to succeed, it may not keep failing; other tools always show geometry even when
it is not valid. IDRIS does the same, and stays honest about what that geometry is: shapes may be simplified,
labels and numbers may not lie.

The ladder, for a free-form design in /chat (and for each concept):
  1. the full design (the existing path, `cad_codegen.generate`);
  2. one call that asks for the simplest valid version: main shape only, no fillets, no patterns, every size
     the engineer gave kept — only while there is time left in the budget;
  3. a local envelope built here, with no AI, from the sizes the engineer actually said (box/plate or
     cylinder/tube), so the workspace still shows a part.
A `fidelity` field says which rung produced the geometry: full, simplified or placeholder. `validated` is
always False in chat, because the chat path runs no strength or thermal analysis, and the answer carries a
"Not checked by IDRIS: ..." line instead of a pass.

F2 is here too: `target_check` reads a performance target in the sentence and says whether physics allows
it, from a stated law, and never quotes a saving percentage IDRIS cannot compute.
"""
from __future__ import annotations

import re
import time

import llm

FIDELITIES = ("full", "simplified", "placeholder")

# F2 verdicts (review of PR #5-#8, item 2): "doubtful" is kept for a target that a law of physics argues
# against. A quantity IDRIS cannot measure at all is "not_checkable_yet" — a quieter motor is not doubtful just
# because this build has no acoustic solver.
VERDICTS = ("plausible", "not_achievable_by_this_part", "not_checkable_yet", "doubtful")

# Digit-free on purpose: it is appended to the request, and a stray number there would be read as a size the
# engineer gave (cad_codegen collects the user's numbers from the request text).
SIMPLIFY_NOTE = ("\n\nSimplest valid version, only this once: the main shape only, no fillets, no patterns, "
                 "no small details, a few lines of code, ONE solid. Keep every size the engineer gave, exactly.")

TARGET_NOTE = ("IDRIS cannot quantify this yet: a saving percentage is only quoted here when a solver or a "
               "stored reference produces it.")

_NUM_MM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(?:mm|milimeter|milimiter)\b", re.I)
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w.])")
_PCT = re.compile(r"([+\-−–]?\s*\d+(?:[.,]\d+)?)\s*%")

_BOX_WORDS = ("kubus", "kotak", "balok", "plat", "pelat", "plate", "box", "block", "blok", "bracket", "braket",
              "housing", "rumah", "panel", "cover", "tutup", "dudukan", "mount", "base", "flange", "flens")
_CYL_WORDS = ("silinder", "tabung", "tube", "pipa", "pipe", "poros", "shaft", "disc", "disk", "cakram",
              "cylinder", "bushing", "bos", "cincin", "pulley", "roller", "sleeve", "spacer")

# what the chat path does not check, kept in one place so no answer can quietly drop it
CHAT_NOT_CHECKED = (
    "strength (FEA): the chat path runs no analysis; send a load and a support to /design/job",
    "manufacturability (DFM): the rule table runs in the design job, not in the chat path",
    "thermal and flow of a free-form part: IDRIS has no solver for it yet",
)

_LAWS = {
    "speed": "drag rises with the square of the speed and the power to overcome it with the cube, so a faster "
             "body needs more energy, not less",
    "fuel": "fuel burn follows the energy the part needs; without an engine and a duty cycle it is not a "
            "property of this part",
    "mass": "mass is the volume the geometry encloses times the density of its material, so it is known once "
            "the kernel has measured the solid",
    "strength": "strength is a stress compared with a material limit, and that needs a load and a support",
    "thermal": "temperature needs a heat input and a cooling path; IDRIS has no thermal solver for a free part",
    "noise": "noise needs an acoustic model (source spectrum and radiation); IDRIS has no acoustic solver",
}

_TARGET_WORDS = (
    ("speed", ("lebih cepat", "kecepatan", "speed", "faster", "top speed")),
    ("fuel", ("bahan bakar", "konsumsi", "fuel", "consumption", "boros", "hemat energi")),
    ("mass", ("lebih ringan", "ringan", "lighter", "light", "massa", "mass", "berat")),
    ("strength", ("lebih kuat", "kuat", "kekuatan", "strength", "stronger", "kaku", "stiff")),
    ("thermal", ("lebih dingin", "dingin", "suhu", "panas", "cooler", "thermal", "temperature")),
    ("noise", ("tidak bising", "bising", "senyap", "suara", "quiet", "noise", "akustik", "acoustic")),
)


# ------------------------------------------------------------------ F2: is the target possible at all
def target_check(message: str) -> dict | None:
    """-> {targets, verdict, reason, what_is_checkable} or None when the sentence states no target.

    The verdict comes from a law of physics written down here, never from a guessed percentage. A number
    IDRIS cannot compute is not invented: `TARGET_NOTE` says so in so many words.
    """
    text = (message or "").lower()
    if not text.strip():
        return None
    pcts = [p.replace(" ", "").replace(",", ".") for p in _PCT.findall(text)]
    found = [(kind, word) for kind, words in _TARGET_WORDS for word in words if word in text]
    if not found:
        return None
    kinds, targets = [], []
    for kind, word in found:
        if kind in kinds:
            continue
        kinds.append(kind)
        targets.append({"quantity": kind, "stated": word, "percent": pcts[0] if len(pcts) == 1 else
                        (pcts[kinds.index(kind)] if kinds.index(kind) < len(pcts) else None)})
    if "speed" in kinds and "fuel" in kinds:
        verdict = "not_achievable_by_this_part"
        reason = ("speed and fuel cannot both improve by reshaping a single part: " + _LAWS["speed"] +
                  "; and " + _LAWS["fuel"])
        checkable = []
    elif "noise" in kinds:
        verdict = "not_checkable_yet"
        reason = (_LAWS["noise"] + ". IDRIS has no acoustic solver in this build, so this target is neither "
                  "achievable nor doubtful here: it is simply not judged")
        checkable = ["mass and stiffness of the shape, from the measured solid"]
    elif "thermal" in kinds:
        verdict = "not_checkable_yet"
        reason = (_LAWS["thermal"] + ". IDRIS has no thermal solver for a free part in this build, so this "
                  "target is not judged here")
        checkable = ["mass and surface area of the shape, from the measured solid"]
    elif "strength" in kinds:
        verdict = "plausible"
        reason = _LAWS["strength"]
        checkable = ["stress and safety factor, once a load and a support are given"]
    elif "mass" in kinds:
        verdict = "plausible"
        reason = _LAWS["mass"]
        checkable = ["mass = measured volume x material density"]
    else:
        verdict = "plausible"
        reason = _LAWS[kinds[0]]
        checkable = []
    return {"targets": targets, "verdict": verdict, "reason": reason, "what_is_checkable": checkable,
            "note": TARGET_NOTE if pcts else None}


# ------------------------------------------------------------------ F3: what the answer must admit
def not_checked(cad: dict, fidelity: str = "full") -> list:
    """The checks this path did not run. Short, concrete, and never a pass."""
    out = list(CHAT_NOT_CHECKED)
    if fidelity == "placeholder":
        out.insert(0, "shape: this is an envelope made from the sizes in the request, not the part you "
                      "described; features, holes and fillets are missing")
    elif fidelity == "simplified":
        out.insert(0, "details: the main shape only; fillets, patterns and small features were left out")
    if (cad or {}).get("missing"):
        out.insert(0, "sizes: " + ", ".join(str(m.get("question") or m.get("k") or m) for m in cad["missing"][:3]))
    return out


# ------------------------------------------------------------------ F1 rung 3: the local envelope
def _sizes(message: str) -> list:
    text = message or ""
    vals = [float(x.replace(",", ".")) for x in _NUM_MM.findall(text)]
    if not vals:
        vals = [float(x.replace(",", ".")) for x in _NUM.findall(text)]
    return [v for v in vals if 0.5 <= v <= 5000][:3]


def _shape(message: str) -> str:
    m = (message or "").lower()
    return "cylinder" if any(w in m for w in _CYL_WORDS) else "box"


def _param(k: str, label: str, label_id: str, value: float) -> dict:
    return {"k": k, "label": label, "label_id": label_id, "unit": "mm", "value": value,
            "min": round(max(0.5, value * 0.2), 2), "max": round(value * 4, 2), "source": "user"}


def placeholder_cad(message: str) -> dict | None:
    """A part built here, with no AI, from the sizes the engineer said -> `cad` the browser can build.

    Returns None when the sentence carries no usable size: then IDRIS asks for one to three main sizes
    instead of drawing something nobody asked for.
    """
    sizes = _sizes(message)
    if not sizes:
        return None
    shape = _shape(message)
    if shape == "cylinder":
        d = sizes[0]
        h = sizes[1] if len(sizes) > 1 else sizes[0]
        code = ("const { d, h } = P;\n"
                "// step: envelope\n"
                "const body = makeCylinder(d / 2, h, [0, 0, 0], [0, 0, 1]);\n"
                "return { body };")
        params = [_param("d", "Diameter", "Diameter", d), _param("h", "Height", "Tinggi", h)]
        expect = [{"label": "Diameter", "axis": "x", "value": d, "tol": 0.5},
                  {"label": "Height", "axis": "z", "value": h, "tol": 0.5}]
        title = "Cylinder envelope"
        omitted = "the bore, holes, fillets and any feature on the outside"
    else:
        length = sizes[0]
        width = sizes[1] if len(sizes) > 1 else sizes[0]
        height = sizes[2] if len(sizes) > 2 else min(length, width)
        code = ("const { L, W, H } = P;\n"
                "// step: envelope\n"
                "const body = makeBox([0, 0, 0], [L, W, H]);\n"
                "return { body };")
        params = [_param("L", "Length", "Panjang", length), _param("W", "Width", "Lebar", width),
                  _param("H", "Height", "Tinggi", height)]
        expect = [{"label": "Length", "axis": "x", "value": length, "tol": 0.5},
                  {"label": "Width", "axis": "y", "value": width, "tol": 0.5},
                  {"label": "Height", "axis": "z", "value": height, "tol": 0.5}]
        title = "Block envelope"
        omitted = "holes, pockets, fillets, ribs and any feature on the outside"
    return {"title": title, "kind": "interface", "process": None, "material": "steel",
            "note": "envelope from the sizes in the request", "code": code, "params": params,
            "expect": expect, "holes": [], "variants": [], "missing": [], "requirements": None,
            "read_by": "IDRIS envelope template (no AI)", "fidelity": "placeholder",
            "placeholder": True, "omitted": omitted}


# ------------------------------------------------------------------ F1: the ladder
def _default_generate():
    import cad_codegen
    return cad_codegen.generate


def _reason(failed: dict) -> str:
    """One short line for the engineer: why the rung above did not produce geometry."""
    text = str((failed or {}).get("answer") or "").strip()
    if not text:
        return "the AI did not return code the server could accept"
    return text.split("\n")[0][:240]


def _headline(fidelity: str, reason: str) -> str:
    if fidelity == "simplified":
        return (f"Full design failed ({reason}). Showing a simplified version: the main shape only, so "
                "fillets, patterns and small details are left out.")
    return (f"Full design failed ({reason}). Showing a placeholder envelope built from the sizes in your "
            "request, so you can see the size and the space it takes.")


def _placeholder_answer(cad: dict) -> str:
    sizes = ", ".join(f"{p['label']} {p['value']:g} mm" for p in cad.get("params") or [])
    return (f"{cad['title']}: {sizes}. Built by IDRIS with no AI, from the sizes you gave. "
            f"Not in it: {cad.get('omitted')}.")


def ask_sizes(reason: str) -> str:
    """F4: a failure that is left over is a step for the engineer, not a stack trace."""
    return ("IDRIS could not build this part (" + reason + "). To try again, send one to three main sizes "
            "(for example the length, width and height in millimetres), or ask for one part at a time. "
            "No geometry is shown because no usable size was given; IDRIS will not invent one.")


def design(message: str, context: dict | None = None, provider: str | None = None,
           reply_language: str | None = None, generate_fn=None, deadline: float | None = None) -> dict:
    """The ladder. -> {ok, cad, answer, fidelity, validated, target_check, not_checked, read_by, attempts}.

    `generate_fn(message, context, provider, reply_language)` defaults to `cad_codegen.generate`;
    `deadline` is a monotonic time after which no second AI call is started (Bagian G's budget).
    """
    generate = generate_fn or _default_generate()
    ctx = dict(context or {})
    tc = target_check(message)
    try:
        full = generate(message, ctx, provider, reply_language)
    except llm.LLMError as e:                      # no AI at all: rung 3 still gives geometry
        full = {"ok": False, "cad": None, "answer": str(e), "attempts": [], "read_by": None}
    except Exception as e:                         # a fault is not a reason to return nothing
        full = {"ok": False, "cad": None, "answer": f"{type(e).__name__}: {e}", "attempts": [], "read_by": None}
    if full.get("ok") and full.get("cad"):
        return _finish(full, "full", tc)
    reason = _reason(full)

    left = None if deadline is None else deadline - time.time()
    if deadline is None or left > 20:
        try:
            simple = generate(str(message) + SIMPLIFY_NOTE, ctx, provider, reply_language)
        except llm.LLMError:
            simple = {"ok": False, "cad": None, "answer": "", "attempts": []}
        except Exception:
            simple = {"ok": False, "cad": None, "answer": "", "attempts": []}
        if simple.get("ok") and simple.get("cad"):
            out = _finish(simple, "simplified", tc)
            out["answer"] = _headline("simplified", reason) + "\n" + (out.get("answer") or "")
            return out

    cad = placeholder_cad(message)
    if cad is not None:
        out = {"ok": True, "cad": cad, "answer": "", "read_by": cad["read_by"],
               "attempts": (full or {}).get("attempts") or [], "fidelity": "placeholder",
               "validated": False, "target_check": tc, "detail": reason,
               "not_checked": not_checked(cad, "placeholder")}
        out["answer"] = _headline("placeholder", reason) + "\n" + _placeholder_answer(cad)
        return out

    return {"ok": False, "cad": None, "answer": ask_sizes(reason), "read_by": None,
            "attempts": (full or {}).get("attempts") or [], "fidelity": None, "validated": False,
            "target_check": tc, "detail": reason, "not_checked": [], "missing_sizes": True}


def _finish(result: dict, fidelity: str, tc: dict | None) -> dict:
    cad = dict(result.get("cad") or {})
    cad["fidelity"] = fidelity
    cad["validated"] = False                       # the chat path runs no analysis, so this is never a pass
    out = {"ok": True, "cad": cad, "answer": result.get("answer") or "", "read_by": result.get("read_by"),
           "attempts": result.get("attempts") or [], "fidelity": fidelity, "validated": False,
           "target_check": tc, "not_checked": not_checked(cad, fidelity)}
    if tc and tc.get("verdict") == "not_achievable_by_this_part":
        out["answer"] = ("Target check: " + str(tc.get("reason"))[:300] + ". The geometry below is a concept "
                         "for the shape you described, not a result.\n" + out["answer"])
    return out


def answer_notes(result: dict) -> str:
    """The two lines every answer carries: the target check and what IDRIS did not check."""
    lines = []
    tc = (result or {}).get("target_check")
    if tc:
        lines.append(f"Target check: {tc['verdict']} — {str(tc.get('reason'))[:240]}")
    missing = (result or {}).get("not_checked") or []
    if missing:
        lines.append("Not checked by IDRIS: " + "; ".join(missing[:5]))
    return "\n".join(lines)
