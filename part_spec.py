"""The engineer's sizes, read out of the request (Tahap A1).

The design ladder builds a part from `part_spec`, and until now nothing filled it: `plate_steps({})` was always
empty, so a plate with four holes fell through to AI code even though a plate template exists. This module fills it
in every door that designs one part: `/chat` (and `/cad/try`), and each part of an assembly plan.

How it reads, in order of honesty:
  * a deterministic parser first - mm sizes, `120x80x8`, `\u00d840 x 120`, `t=5`, `width 30`, `4 holes M8`, `PCD 120`,
    `R3` - because a number the engineer wrote must never depend on an AI reading it;
  * no AI is used here at all. If a size is not in the sentence it is not invented: the spec leaves it out, the
    template does not fill, and the reply asks for it.
  * a bolt size becomes a clearance hole from ISO 273 (physics/catalog.py, the table this repo already ships),
    labelled `table:ISO 273 medium` and `assumed` until the engineer confirms it.

Every number carries its source: `you`, `table:<standard>`, or `computed` with the method and the inputs it used.
Hole positions are computed from the plate size with a stated rule (2\u00b7d inset), never picked silently.
"""
from __future__ import annotations

import math
import re

import sheet_features
from physics import catalog

WORDS_N = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10}

_DIA2 = re.compile(r"(?:\u00d8|\u2300|\bdia\.?\s*|\bdiameter\s*)(\d+(?:\.\d+)?)\s*[x\u00d7]\s*(\d+(?:\.\d+)?)")
_DIA = re.compile(r"(?:\u00d8|\u2300|\bdia\.?\s*|\bdiameter\s*)(\d+(?:\.\d+)?)")
_STEP = re.compile(r"(?:\u00d8|\u2300|\bdia\.?\s*)(\d+(?:\.\d+)?)\s*[x\u00d7]\s*(\d+(?:\.\d+)?)\s*step")
_T = re.compile(r"(?:\bt\s*=\s*|\bthick(?:ness)?\s*|\btebal\s*|\bwall\s*)(\d+(?:\.\d+)?)")
_WIDTH = re.compile(r"(?:\bwidth\s*|\blebar\s*)(\d+(?:\.\d+)?)")
_LENGTH = re.compile(r"(?:\blength\s*|\bpanjang\s*|\blong\s*)(\d+(?:\.\d+)?)")
_PCD = re.compile(r"\bpcd\s*=?\s*(\d+(?:\.\d+)?)")
_FILLET = re.compile(r"(?:\bfillet\s*(?:r|radius)?\s*=?\s*|\br\s*=?\s*)(\d+(?:\.\d+)?)")
_POCKET = re.compile(r"\bpocket\s*(?:depth\s*)?(?:of\s*)?(\d+(?:\.\d+)?)")
_HOLES = re.compile(r"(\d+|[a-z]+)\s*(?:x\s*)?(?:holes?|lubang)\s*[\u00d8\u00f8]?\s*m\s*(\d+(?:\.\d+)?)")
_M_HOLES = re.compile(r"(\d+|[a-z]+)\s*m\s*(\d+(?:\.\d+)?)\s*(?:holes?|lubang)")
_M_TIMES = re.compile(r"\b(\d+)\s*[x\u00d7]\s*m\s*(\d+(?:\.\d+)?)")
_TRIPLE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?![\d.])")
_PAIR = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?![\d.])")


def _f(v) -> float:
    return float(str(v).replace(",", "."))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:40] or "part"


TEMPLATE_WORDS = (
    ("bracket_l", ("l bracket", "l-bracket", "angle bracket", "braket l", "siku")),
    ("channel", ("channel", "u bracket", "u-bracket", "kanal", "c channel")),
    ("gusset", ("gusset", "rusuk", "rib plate")),
    ("flange", ("flange", "flensa")),
    ("shaft_step", ("shaft", "poros", "stepped", "bertingkat")),
    ("tube", ("tube", "tabung", "pipe", "pipa", "sleeve", "bushing", "bos")),
    ("housing", ("housing", "enclosure", "box shell", "kotak")),
    ("plate_pocket", ("pocket", "kantong", "slot")),
)


def template_hint(text: str) -> str:
    """Which template the words name; 'plate' when they name none (the ladder's safest shape).

    Words are matched whole: 'bos' (bushing) must not fire on 'emboss', and 'siku' must not fire on a longer word,
    or a sheet-metal part is read as a turned one and the wrong template is chosen.
    """
    t = " " + re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip() + " "
    for name, words in TEMPLATE_WORDS:
        if any(" " + re.sub(r"[^a-z0-9]+", " ", w).strip() + " " in t for w in words):
            return name
    return "plate"


NEEDS = {"plate": ("length_mm", "width_mm", "thickness_mm"),
         "plate_pocket": ("length_mm", "width_mm", "thickness_mm", "pocket_depth_mm"),
         "bracket_l": ("leg_a_mm", "leg_b_mm", "thickness_mm", "width_mm"),
         "channel": ("width_mm", "height_mm", "thickness_mm", "length_mm"),
         "gusset": ("leg_a_mm", "leg_b_mm", "thickness_mm"),
         "flange": ("diameter_mm", "thickness_mm"),
         "shaft_step": ("diameter_mm", "length_mm", "step_diameter_mm", "step_length_mm"),
         "tube": ("diameter_mm", "length_mm", "thickness_mm"),
         "housing": ("length_mm", "width_mm", "height_mm", "thickness_mm")}


def template_ready(tpl: str, dims: dict) -> bool:
    need = NEEDS.get(tpl, ())
    return bool(need) and all(k in dims for k in need)


def _shape_class(tpl: str, text: str) -> str:
    if tpl in ("shaft_step", "tube"):
        return "revolve"
    if any(w in " " + text + " " for w in ("sheet metal", "bent plate", "press brake", "plate metal")):
        return "sheet"
    return "prismatic"


def _hole_count(text: str):
    for rx in (_HOLES, _M_HOLES):
        m = rx.search(text)
        if m:
            n = m.group(1)
            n = int(n) if n.isdigit() else WORDS_N.get(n, 0)
            if n:
                return n, m.group(2)
    m = _M_TIMES.search(text)
    if m:
        return int(m.group(1)), m.group(2)
    return 0, None


def _hole_d(m_size: str):
    """ISO 273 medium clearance for an M size, from the table this repo ships. None when the table has no row."""
    f = catalog.fastener("M" + str(m_size))
    return (float(f["clearance"]["medium"]), str(f["sources"]["clearance"])) if f else (None, None)


def _corner_holes(n: int, d: float, L: float, W: float) -> list:
    """Hole centres for a rectangular plate: 2\u00b7d inset from each edge, spread as evenly as the count allows.

    The rule is stated, not hidden: it is the usual bolt edge-distance practice, not a standard, so it is labelled
    computed with the inputs it used and the engineer can move any hole in the feature tree afterwards.
    """
    inset = 2.0 * d
    if n == 4:
        pts = [(L / 2 - inset, W / 2 - inset), (L / 2 - inset, -(W / 2 - inset)),
               (-(L / 2 - inset), -(W / 2 - inset)), (-(L / 2 - inset), W / 2 - inset)]
        return [{"x_mm": x, "y_mm": y} for x, y in pts]
    if n == 2:
        return [{"x_mm": -(L / 2 - inset), "y_mm": 0.0}, {"x_mm": L / 2 - inset, "y_mm": 0.0}]
    if n == 1:
        return [{"x_mm": 0.0, "y_mm": 0.0}]
    span = max(0.0, L - 2 * inset)
    return [{"x_mm": (-span / 2 + span * i / (n - 1)), "y_mm": 0.0} for i in range(n)]


def _pcd_holes(n: int, pcd: float, start_deg: float = 45.0) -> list:
    r = pcd / 2
    return [{"x_mm": r * math.cos(math.radians(start_deg + 360 * i / n)),
             "y_mm": r * math.sin(math.radians(start_deg + 360 * i / n))} for i in range(n)]


def from_message(message: str, context: dict | None = None) -> dict:
    """-> a part_spec {id, name, template, shape_class, numbers, holes, fillet_mm}, or {} when no size is stated.

    A size that is not in the sentence is left out on purpose: the ladder then has no template to fill and replies
    with the question, instead of drawing a part from a number nobody gave.
    """
    raw = str(message or "")[:2000]
    text = raw.lower().replace("\u00d7", "x").replace("\u2013", "-")
    # lowercasing turns Ø into ø; the patterns read the capital symbol, so put it back
    text = text.replace("\u00f8", "\u00d8").replace("\u2300", "\u00d8")
    hint = template_hint(text)
    dims, numbers = {}, []

    def take(k: str, v, unit: str = "mm", source: str = "you"):
        if v is None or k in dims:
            return
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return
        if not (fv > 0):
            return
        dims[k] = fv
        numbers.append({"k": k, "value": fv, "unit": unit, "source": source})

    rest = text
    mstep = _STEP.search(rest)
    if mstep:
        take("step_diameter_mm", mstep.group(1))
        take("step_length_mm", mstep.group(2))
        rest = rest[:mstep.start()] + " " * (mstep.end() - mstep.start()) + rest[mstep.end():]

    # a forming tool's D x depth is the punch, not the part: it is masked so the pair rule cannot read it as a size,
    # but neither number becomes the part's diameter or length
    forming = any(w in text for w in ("emboss", "lance", "louvre", "louver", "forming tool", "timbul"))
    # Ø first: `Ø120 x 10` must not be read as a plain `120 x 10` by the pair rule below
    dia_pairs = list(_DIA2.finditer(rest))
    for i, m in enumerate(dia_pairs):
        if forming:
            pass
        elif i == 0:
            take("diameter_mm", m.group(1))
            if hint == "flange":
                take("thickness_mm", m.group(2))
            else:
                take("length_mm", m.group(2))
        else:
            take("step_diameter_mm", m.group(1))
            take("step_length_mm", m.group(2))
        rest = rest[:m.start()] + " " * (m.end() - m.start()) + rest[m.end():]

    # the explicit names win over a bare `a x b x c`
    named = {}
    for rx, k in ((_T, "thickness_mm"), (_WIDTH, "width_mm"), (_LENGTH, "length_mm"), (_PCD, "pcd_mm")):
        m = rx.search(rest)
        if m:
            named[k] = _f(m.group(1))
            rest = rest[:m.start()] + " " * (m.end() - m.start()) + rest[m.end():]

    mp = _POCKET.search(rest)
    if mp and hint in ("plate_pocket", "housing"):
        named["pocket_depth_mm"] = _f(mp.group(1))
        rest = rest[:mp.start()] + " " * (mp.end() - mp.start()) + rest[mp.end():]

    m3 = _TRIPLE.search(rest)
    if m3:
        a, b, c = (_f(m3.group(1)), _f(m3.group(2)), _f(m3.group(3)))
        if hint == "bracket_l":
            take("leg_a_mm", a), take("leg_b_mm", b), take("thickness_mm", c)
        elif hint == "gusset":
            take("leg_a_mm", a), take("leg_b_mm", b), take("thickness_mm", c)
        elif hint == "channel":
            take("width_mm", a), take("height_mm", b), take("thickness_mm", c)
        elif hint == "housing":
            take("length_mm", a), take("width_mm", b), take("height_mm", c)
        else:
            take("length_mm", a), take("width_mm", b), take("thickness_mm", c)
        rest = rest[:m3.start()] + " " * (m3.end() - m3.start()) + rest[m3.end():]
    else:
        m2 = _PAIR.search(rest)
        if m2:
            a, b = _f(m2.group(1)), _f(m2.group(2))
            if hint in ("bracket_l", "gusset"):
                take("leg_a_mm", a), take("leg_b_mm", b)
            else:
                take("length_mm", a), take("width_mm", b)
            rest = rest[:m2.start()] + " " * (m2.end() - m2.start()) + rest[m2.end():]

    if "diameter_mm" not in dims:
        m = _DIA.search(rest)
        if m:
            take("diameter_mm", m.group(1))

    for k, v in named.items():
        take(k, v)

    # a flange's \u00d8 and thickness, read as `\u00d8120 x 10`, keep their own names
    if hint == "flange" and "thickness_mm" not in dims and "length_mm" in dims:
        take("thickness_mm", dims.pop("length_mm"))
        numbers[:] = [n for n in numbers if n["k"] != "length_mm"]

    count, msize = _hole_count(text)
    holes = []
    if count and msize:
        d, src = _hole_d(msize)
        if d:
            on_circle = "pcd_mm" in dims and hint in ("flange", "tube", "shaft_step")
            rows = (_pcd_holes(count, dims["pcd_mm"]) if on_circle
                    else _corner_holes(count, d, dims.get("length_mm") or d * 6, dims.get("width_mm") or d * 6))
            method = ("evenly spaced on the bolt circle (PCD given by the engineer)" if on_circle
                      else "2\u00b7d inset from each edge, spread evenly (bolt edge-distance rule of thumb, "
                           "not a standard)")
            for row in rows:
                holes.append({"d_mm": d, "x_mm": row["x_mm"], "y_mm": row["y_mm"], "m": "M" + str(msize),
                              "source": "table:ISO 273 medium", "x_source": "computed", "y_source": "computed",
                              "method": method})
            numbers.append({"k": "hole_d_mm", "value": d, "unit": "mm", "source": "table:ISO 273 medium",
                            "assumed": True})
            dims["hole_d_mm"] = d

    mf = _FILLET.search(text)
    fillet = _f(mf.group(1)) if mf else None
    if fillet is not None and fillet > 0:
        numbers.append({"k": "fillet_mm", "value": fillet, "unit": "mm", "source": "you"})

    # folds and punches: read here as well, because a hem or a jog is a size the engineer typed and a formula on it
    sheet = sheet_features.parse(raw, dims) if dims else {}
    for row in sheet.get("numbers") or []:
        if not any(n.get("k") == row.get("k") for n in numbers):
            numbers.append(row)

    if not dims:
        return {}
    tpl = hint
    if not template_ready(tpl, dims) and template_ready("plate", dims) and tpl not in ("plate", "plate_pocket"):
        tpl = "plate"
    name = ("Sheet metal part" if sheet.get("features") and tpl == "plate" else
            {"plate": "Plate", "plate_pocket": "Plate with pocket", "bracket_l": "L bracket",
            "channel": "Channel bracket", "gusset": "Gusset", "flange": "Flange", "shaft_step": "Stepped shaft",
             "tube": "Tube", "housing": "Housing"}.get(tpl, "Part"))
    return {"id": _slug(name + " " + raw[:24]), "name": name, "template": tpl,
            "shape_class": _shape_class(tpl, text), "numbers": numbers, "holes": holes,
            "fillet_mm": fillet, "dims": dims, "read_by": "part_spec (deterministic parser)",
            "sheet_features": sheet if (sheet.get("features") or sheet.get("questions")) else None,
            "questions": sheet.get("questions") or []}


def from_plan(part: dict, plan: dict | None = None) -> dict:
    """One part of an assembly plan -> a part_spec, using the plan's own sizes and their sources.

    A size the plan carries with a source keeps that source. A size with no source is not used: it comes back in
    `questions`, so the part is asked about instead of drawn from a number nobody stands behind.
    """
    part = part if isinstance(part, dict) else {}
    dims = part.get("dims") if isinstance(part.get("dims"), dict) else {}
    src_map = part.get("sources") if isinstance(part.get("sources"), dict) else {}
    numbers, questions = [], []
    for k, v in dims.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        src = str(src_map.get(k) or part.get("source") or "").strip()
        if src == "computed":
            numbers.append({"k": str(k), "value": fv, "unit": "mm", "source": src,
                            "method": str(part.get("method") or "computed in the assembly plan"),
                            "from_": [str(x) for x in (part.get("from") or [])][:6]})
        elif src == "you" or src.startswith("table:"):
            numbers.append({"k": str(k), "value": fv, "unit": "mm", "source": src})
        else:
            questions.append({"k": str(k), "question": (f"What value should I use for {str(k).replace('_', ' ')} of "
                                                       f"{part.get('name') or 'this part'}? The plan has no source "
                                                       f"for it.")})
    if not numbers:
        text = " ".join(str(x or "") for x in (part.get("name"), part.get("role"),
                                               " ".join(part.get("needs") or [])))
        spec = from_message(text)
        return {**spec, "id": str(part.get("id") or spec.get("id") or "part"),
                "name": str(part.get("name") or "Part"), "questions": questions,
                "read_by": "part_spec (assembly plan)"}
    flat = {n["k"]: n["value"] for n in numbers}
    tpl = template_hint(str(part.get("name") or "") + " " + str(part.get("role") or ""))
    return {"id": str(part.get("id") or _slug(str(part.get("name")))), "name": str(part.get("name") or "Part"),
            "template": tpl if template_ready(tpl, flat) else None,
            "shape_class": _shape_class(tpl, str(part.get("name") or "")),
            "numbers": numbers, "holes": part.get("holes") if isinstance(part.get("holes"), list) else [],
            "fillet_mm": part.get("fillet_mm"), "dims": flat, "questions": questions,
            "read_by": "part_spec (assembly plan)"}
