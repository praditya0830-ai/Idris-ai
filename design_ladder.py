"""The design ladder: features -> code -> an honest failure report (v0.43).

Target, written down because it decides every choice here: NOT "never fails" (impossible, and forcing it bends the
numbers until a part that is wrong looks finished — worse than an honest failure). The target is: the highest
success rate, the smartest route, and a failure that says exactly what went wrong and what would fix it.

  rung 1 "features"  a validated feature recipe (feature_recipe.py), built by the browser kernel, editable step by step
  rung 2 "code"      the AI's CAD code (the existing `full`/`simplified` path)
  rung 3 "failed"    NO geometry as a design: {ok: False, fidelity: "failed", report: {...}}. The old envelope box is
                     carried as `preview` and labelled "failed preview — not a design"; it is never a result.

The shape is read BEFORE the rung is chosen (1b): a prismatic/mechanical part starts with features, a free-form or
double-curved part starts with code, because forcing sketch-extrude on a moldboard only produces a worse failure.

Numbers (1c): every number says where it came from — `you`, `computed` (with `method` and `from`), or
`table:<standard>` (assumed until the engineer confirms). A number with no source, or with source "guess", is not a
number: it becomes a question. Nothing here estimates silently.
"""
from __future__ import annotations

import json
import os

import feature_recipe

SHAPE_CLASSES = ("prismatic", "freeform", "revolve", "sheet", "frame")
FIDELITIES = ("features", "code", "failed")

# 1b: words that name the shape, so a message without an explicit shape_class still routes sensibly.
_WORDS = (
    ("freeform", ("moldboard", "aerodynamic", "airfoil", "fairing", "curved surface", "double curve", "freeform",
                  "free-form", "organic", "sculpted", "bodyshell", "body shell", "spoiler", "propeller", "impeller",
                  "turbine blade", "helmet", "saddle", "nose cone", "shell curve")),
    ("sheet", ("sheet metal", "sheet-metal", "bent plate", "flange plate", "bracket from sheet", "press brake",
               "bending", "flat pattern")),
    ("frame", ("frame", "rangka", "truss", "chassis", "tube frame", "weldment", "weld frame")),
    ("revolve", ("shaft", "poros", "bushing", "bos", "pulley", "sleeve", "spacer", "collar", "flange disc",
                 "turned part", "cylinder", "silinder", "tube", "tabung")),
)
_NOT_CHECKED = ("strength (FEA): this path runs no analysis", "manufacturability (DFM): not run here",
                "the fit of this part in its assembly: checked in the assembly step")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def shape_class_of(part_spec: dict | None = None, message: str = "") -> dict:
    """-> {shape_class, reason, source}. Explicit plan value wins; then the words; then prismatic (safest rung first)."""
    spec = part_spec if isinstance(part_spec, dict) else {}
    given = str(spec.get("shape_class") or "").strip().lower()
    if given in SHAPE_CLASSES:
        return {"shape_class": given, "source": "plan",
                "reason": str(spec.get("shape_reason") or "the plan names this shape")[:200]}
    text = " ".join(str(x or "") for x in (message, spec.get("intent"), spec.get("name"), spec.get("role"))).lower()
    for cls, words in _WORDS:
        hit = next((w for w in words if w in text), None)
        if hit:
            return {"shape_class": cls, "source": "words", "reason": f"the request says '{hit}'"}
    return {"shape_class": "prismatic", "source": "default",
            "reason": "no shape was named, so the ladder starts with the safest rung and reads the numbers"}


def normalise_numbers(part_spec: dict | None) -> dict:
    """Every number carries a source, or it is not used. -> {numbers, questions, dropped, flat}.

    Accepted sources:
      "you"            the engineer gave it
      "computed"       derived here, with `method` (the law or standard) and `from` (the inputs it used)
      "table:<std>"    a printed table, `assumed` true until the engineer confirms it
    Anything else — including "guess" — is dropped and asked back as a question.
    """
    spec = part_spec if isinstance(part_spec, dict) else {}
    numbers, questions, dropped, flat = [], [], [], {}
    for n in spec.get("numbers") or []:
        if not isinstance(n, dict):
            continue
        k = str(n.get("k") or n.get("key") or "").strip()[:40]
        v = _num(n.get("value", n.get("v")))
        if not k or v is None:
            continue
        src = str(n.get("source") or "").strip()
        row = {"k": k, "value": v, "unit": str(n.get("unit") or "")[:10]}
        if src == "computed":
            if not str(n.get("method") or "").strip() or not (n.get("from") or []):
                dropped.append({"k": k, "why": "computed without a method or without the inputs it used"})
                questions.append({"k": k, "question": f"What value should I use for {k.replace('_', ' ')}? (a "
                                                       f"computed number needs the formula and the inputs)"})
                continue
            row.update(source="computed", method=str(n.get("method"))[:200],
                       from_=[str(x)[:40] for x in (n.get("from") or [])][:8])
        elif src.startswith("table:"):
            row.update(source=src[:120], assumed=bool(n.get("assumed", True)))
        elif src == "you":
            row.update(source="you")
        else:
            dropped.append({"k": k, "why": f"source '{src or 'none'}' is not a source"})
            questions.append({"k": k, "question": f"What value should I use for {k.replace('_', ' ')}? "
                                                   f"(give it with its unit)"})
            continue
        numbers.append(row)
        flat[k] = v
    return {"numbers": numbers, "questions": questions, "dropped": dropped, "flat": flat}


def _source_rows(numbers: list) -> dict:
    return {str(n.get("k")): {"value": n.get("value"), "unit": n.get("unit"), "source": n.get("source"),
                              "method": n.get("method"), "assumed": n.get("assumed")} for n in numbers}


def _features_result(spec: dict, cls: dict, recipe: dict, numbers: dict, message: str) -> dict:
    title = str(spec.get("name") or recipe.get("title") or "Part")[:80]
    origin = str(recipe.get("origin") or "template")
    tpl = recipe.get("template")
    how = ("the known-good template" if origin == "template" else "an AI-written recipe")
    cad = {"title": title, "kind": str(spec.get("kind") or "functional"), "process": spec.get("process"),
           "material": str(spec.get("material") or "steel"), "params": [], "code": None,
           "feature_tree": recipe["steps"], "read_by": "IDRIS feature recipe (validated)",
           "numbers": numbers["numbers"], "sources": _source_rows(numbers["numbers"]), "missing": [],
           "shape_class": cls["shape_class"], "feature_template": tpl, "feature_origin": origin,
           "feature_title": str(recipe.get("title") or title)[:80],
           "recipe": {"title": str(recipe.get("title") or title)[:80], "origin": origin, "template": tpl,
                      "steps": len(recipe["steps"]), "attempts": recipe.get("attempts")}}
    return {"ok": True, "rung": "features", "fidelity": "features", "cad": cad, "read_by": cad["read_by"],
            "numbers": numbers["numbers"], "questions": numbers["questions"], "report": None,
            "validated": False, "not_checked": list(_NOT_CHECKED),
            "shape_class": cls, "recipe": cad["recipe"],
            "answer": (f"{len(recipe['steps'])} feature steps for {title}, from the sizes you gave ({how}). Every "
                       f"step is editable; the feature tree shows where each number came from.")}


def _recipe_ai_allowed() -> bool:
    """The AI recipe rung runs only when a provider is configured, and can be turned off with IDRIS_RECIPE_AI=0."""
    if os.environ.get("IDRIS_RECIPE_AI", "1").strip() == "0":
        return False
    try:
        import llm
        return bool(llm.provider_order())
    except Exception:
        return False


def _report(spec: dict, cls: dict, message: str, tried: list, numbers: dict, suggestion: str) -> dict:
    return {"part_id": str(spec.get("id") or "")[:60] or None,
            "part_name": str(spec.get("name") or spec.get("id") or "part")[:80],
            "shape_class": cls["shape_class"],
            "requested": {"dims": {str(n.get("k")): n.get("value") for n in numbers["numbers"]},
                          "intent": str(message or "")[:240]},
            "rungs_tried": tried,
            "missing_numbers": numbers["questions"][:6],
            "suggestion": suggestion[:400],
            "not_checked": list(_NOT_CHECKED)}


def design(message: str, context: dict | None = None, provider=None, reply_language=None, deadline=None,
           recipe_fn=None, code_fn=None) -> dict:
    """The ladder. -> {ok, rung, fidelity, cad|preview, answer, report, numbers, shape_class, ...}.

    `recipe_fn(spec, numbers, cls) -> recipe|None` and `code_fn(message, ctx, provider, reply_language, deadline=)`
    are injectable so the tests never call an AI or a kernel.
    """
    ctx = dict(context or {})
    spec = ctx.get("part_spec") if isinstance(ctx.get("part_spec"), dict) else {}
    cls = shape_class_of(spec, message)
    numbers = normalise_numbers(spec)
    tried = []

    # 1b: freeform starts at rung 2 on purpose; forcing a sketch-extrude on a double curve only fails later and worse.
    if cls["shape_class"] != "freeform":
        rec = None
        if recipe_fn:
            rec = recipe_fn(spec, numbers, cls)
        else:
            rec = feature_recipe.template_for(cls["shape_class"], numbers["flat"], spec.get("holes"),
                                              spec.get("fillet_mm"), hint=spec.get("template"))
        if rec:
            ok, errs = feature_recipe.validate(rec, numbers["flat"])
            if ok:
                return _features_result(spec, cls, rec, numbers, message)
            tried.append({"rung": "features", "error": "; ".join(errs)[:500]})
        else:
            tried.append({"rung": "features", "error": "no known-good template fits this shape"})
        # 1c (A2): no template, or the one that fits was refused -> the AI writes a recipe in the same vocabulary,
        # checked here before any kernel call. Code stays the next rung, not the default.
        if not recipe_fn and _recipe_ai_allowed():
            got = feature_recipe.ai_recipe(message, numbers["flat"], spec, cls["shape_class"], provider=provider,
                                           deadline=deadline)
            if got.get("steps"):
                ok, errs = feature_recipe.validate(got, numbers["flat"])
                if ok:
                    return _features_result(spec, cls, got, numbers, message)
                tried.append({"rung": "features", "error": ("the AI recipe was refused: "
                                                             + "; ".join(errs))[:500]})
            else:
                tried.append({"rung": "features", "error": ("the AI recipe was refused: "
                                                             + str(got.get("error") or "no steps"))[:300]})
    else:
        tried.append({"rung": "features", "error": "free-form shape: the ladder starts with code on purpose"})

    # rung 2: code
    code = None
    if code_fn:
        try:
            code = code_fn(message, ctx, provider, reply_language, deadline=deadline)
        except TypeError:                                   # an injected fn without the deadline keyword
            code = code_fn(message, ctx, provider, reply_language)
        except Exception as e:                              # a fault is not a reason to return nothing
            code = {"ok": False, "cad": None, "answer": f"{type(e).__name__}: {e}"}
    else:
        import cad_fallback
        code = cad_fallback.design(message, ctx, provider, reply_language, deadline=deadline)
    cad = (code or {}).get("cad") if isinstance(code, dict) else None
    if isinstance(cad, dict) and cad.get("code"):
        return {**code, "rung": "code", "fidelity": "code", "cad": cad, "report": None,
                "numbers": numbers["numbers"], "questions": numbers["questions"], "shape_class": cls,
                "features_error": (tried[0]["error"] if tried else None)}
    tried.append({"rung": "code", "error": str((code or {}).get("detail") or (code or {}).get("answer")
                                               or "the code did not build")[:500]})

    # rung 3: an honest report. The envelope (when the old ladder made one) is only a labelled preview.
    preview = None
    if isinstance(cad, dict):
        preview = {"cad": cad, "label": "failed preview — not a design"}
    else:
        try:
            import cad_fallback
            pv = cad_fallback.placeholder_cad(message)
            if pv:
                preview = {"cad": pv, "label": "failed preview — not a design"}
        except Exception:
            preview = None
    missing = numbers["questions"]
    suggestion = ("Give the missing size(s): " + ", ".join(m["k"] for m in missing[:4]) + "."
                  if missing else
                  "Describe the main shape with its key sizes, or ask for a free-form surface if the part is "
                  "double-curved (the feature rung cannot draw that).")
    report = _report(spec, cls, message, tried, numbers, suggestion)
    answer = (f"I could not build this part, and I am not showing a box as if it were the design. "
              f"Tried: {'; '.join(t['rung'] + ' — ' + t['error'] for t in tried)[:600]}. "
              f"What would help: {suggestion}")
    return {"ok": False, "rung": "failed", "fidelity": "failed", "cad": None, "preview": preview,
            "answer": answer, "report": report, "numbers": numbers["numbers"], "questions": missing,
            "shape_class": cls, "validated": False, "not_checked": list(_NOT_CHECKED), "read_by": None}


def report_json(report: dict | None) -> str:
    """A stable string for logs and feedback, never a Python repr."""
    return json.dumps(report or {}, ensure_ascii=False, sort_keys=True)[:4000]
