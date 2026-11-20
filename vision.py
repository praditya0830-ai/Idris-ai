"""Sketch to CAD, Photo to CAD, and the missing part from a photo (vision task V1-V3).

One rule decides everything here: **an image gives shape and proportion, never a measured size.** A photo
has no scale bar, and perspective bends what a ruler would read. So the only millimetres this module lets
through are
  * numbers the engineer WROTE on the sketch, read by the AI (`dims_written`) and confirmed in the Studio, and
  * one reference length the engineer states (`reference: {what, mm}`).
Everything else stays a ratio. Without a reference the answer says "Scale not set - proportions from image",
the design job is told to use a provisional scale, and FEA is refused (a safety factor on an invented scale
is a made-up number).

The AI is used for what it is good at - naming what it sees, listing features, guessing which side is front -
and its answer is validated here: every size that did not come from `dims_written` or `reference` is dropped
and counted, so the Studio can show "3 sizes invented by the AI were removed".

Shape match (V3) is measured, not claimed: two silhouettes are rasterized and compared by IoU.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import struct

import llm

MODES = ("sketch", "photo", "missing_part")
FEATURE_KINDS = ("plate", "boss", "hole", "slot", "rib", "fillet", "tube", "revolve", "pocket", "step",
                 "flange", "web", "other")
VIEWS = ("front", "top", "side", "iso", "perspective")
SYMMETRY = ("none", "mirror-x", "mirror-y", "radial")

MAX_IMAGE_BYTES = 10 * 1024 * 1024        # V1: 10 MB, the limit the task document names
MAX_SIDE = 1600                            # V1: longest side after preparing the image
NO_SCALE_LABEL = "Scale not set - proportions from image"
MAX_QUESTIONS = 3

# What the AI may read out of an image. It is asked for JSON; anything else is a failed answer, not a crash.
_SYSTEM = """You read ONE image for IDRIS, an engineering CAD assistant, and you describe only what is visible.

Absolute rules:
- Do NOT invent millimetres. A number in millimetres may appear ONLY in `dims_written`, and only when that
  number is literally written in the image (a dimension text, a handwritten size, a callout). Quote the text
  you saw in `text_seen`.
- Everywhere else, use proportions: a ratio to a named reference feature, between 0 and 1, with a confidence
  between 0 and 1. "the hole is about 0.4 of the overall width, confidence 0.7" is useful; "the hole is 8 mm"
  is a lie unless 8 mm is written on the drawing.
- Perspective, lens distortion and a photographed rule make pixel measurements wrong. Never claim a measured
  size, a tolerance, a material or a load you cannot see written.
- If something is unclear (a hidden feature, an ambiguous line, a part behind another), put it in `unclear`
  instead of guessing. At most 3 questions, the ones that actually block the design.
Answer ONE JSON object, in the language of the note (English if there is no note), with exactly these keys:
{"object": "<what it is>", "views": ["front"|"top"|"side"|"iso"|"perspective", ...],
 "features": [{"name": "...", "kind": "plate|boss|hole|slot|rib|fillet|tube|revolve|pocket|step|flange|web|other",
               "count": 1, "relation": "<how it relates to the rest>"}],
 "proportions": [{"name": "...", "ratio_to": "<the feature it is a ratio of>", "value": 0.0, "confidence": 0.0}],
 "dims_written": [{"text_seen": "<exactly what is written>", "value_mm": 0.0, "applies_to": "...", "confidence": 0.0}],
 "symmetry": "none|mirror-x|mirror-y|radial",
 "unclear": ["..."],
 "questions": ["..."]}
For mode "missing_part" also include:
{"missing_part": {"location": "<where the gap is>", "neighbours": [{"name": "...", "interface_kind": "..."}],
                  "function": "<what that part does, from its neighbours>", "loads_guess": "<words, no numbers>",
                  "interfaces_needed": [{"between": "...", "what_to_measure": "..."}]}}
`interfaces_needed` is where the engineer must measure: bolt hole spacing, shaft diameter, seat width. A photo
cannot give those."""


# ------------------------------------------------------------------ the image
def _decode(image_base64) -> tuple:
    """base64 (optionally a data: URL) -> (bytes, error)."""
    s = str(image_base64 or "")
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    s = re.sub(r"\s", "", s)
    if not s:
        return None, "image_base64 is required (a base64 PNG or JPEG)"
    try:
        return base64.b64decode(s, validate=True), ""
    except Exception:
        return None, "image_base64 is not valid base64"


def _mime_of(raw: bytes) -> str | None:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return None


def png_size(raw: bytes) -> tuple:
    """(width, height) from the IHDR chunk, without decoding the pixels."""
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return (0, 0)
    return struct.unpack(">II", raw[16:24])


def jpeg_size(raw: bytes) -> tuple:
    """(width, height) from the first SOFn marker. 0,0 when the file is not readable that way."""
    i, n = 2, len(raw)
    while i + 9 < n:
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = struct.unpack(">H", raw[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", raw[i + 5:i + 9])
            return (w, h)
        i += 2 + seg
    return (0, 0)


def strip_png_metadata(raw: bytes) -> bytes:
    """Drop every ancillary text/time/exif chunk (tEXt, iTXt, zTXt, eXIf, tIME), keep the picture intact.

    The chunks are copied byte for byte with their own CRC, so the PNG stays valid.
    """
    drop = (b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME")
    out, i = [raw[:8]], 8
    while i + 8 <= len(raw):
        length = struct.unpack(">I", raw[i:i + 4])[0]
        kind = raw[i + 4:i + 8]
        end = i + 12 + length
        if end > len(raw):
            break
        if kind not in drop:
            out.append(raw[i:end])
        i = end
    return b"".join(out) or raw


def strip_jpeg_metadata(raw: bytes) -> bytes:
    """Drop APP1..APP15 (EXIF with the GPS tag, XMP) and COM comments, keep the image data."""
    out, i, n = [raw[:2]], 2, len(raw)
    while i + 4 <= n:
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        if marker == 0xDA:                       # start of scan: everything after it is pixel data
            out.append(raw[i:])
            break
        seg = struct.unpack(">H", raw[i + 2:i + 4])[0]
        end = i + 2 + seg
        if end > n:
            break
        if not (0xE1 <= marker <= 0xEF) and marker != 0xFE:
            out.append(raw[i:end])
        i = end
    return b"".join(out) or raw


def _pillow(raw: bytes, mime: str) -> tuple | None:
    """Resize to MAX_SIDE and re-encode with Pillow, which also drops every metadata block. None if unusable."""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        w, h = im.size
        scale = min(1.0, MAX_SIDE / float(max(w, h) or 1))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        out = io.BytesIO()
        if mime == "image/jpeg":
            im.convert("RGB").save(out, "JPEG", quality=88, optimize=True)
            return out.getvalue(), "image/jpeg", im.size, scale < 1.0
        im.save(out, "PNG", optimize=True)
        return out.getvalue(), "image/png", im.size, scale < 1.0
    except Exception:
        return None


def prepare(image_base64, mime: str | None = None) -> tuple:
    """-> (clean, error, status). `clean` = {mime, data(base64), width, height, original, resized, metadata}.

    Order: size limit, real type, metadata out, then downscale when Pillow is on this server. Without Pillow an
    image that is too large is refused with the reason instead of being sent to the AI as it is.
    """
    raw, err = _decode(image_base64)
    if err:
        return None, err, 400
    if len(raw) > MAX_IMAGE_BYTES:
        return None, (f"the image is {len(raw) / 1048576:.1f} MB; {MAX_IMAGE_BYTES // 1048576} MB is the limit "
                      f"(photograph it again at a lower resolution)"), 413
    real = _mime_of(raw)
    if real is None:
        return None, "the image must be a PNG or a JPEG", 415
    w, h = png_size(raw) if real == "image/png" else jpeg_size(raw)
    out, stripped = raw, False
    resized = False
    got = _pillow(raw, real)
    if got is not None:
        out, real, (w, h), resized = got
        stripped = True                            # re-encoding leaves no EXIF, GPS or comment behind
    else:
        cleaned = strip_png_metadata(raw) if real == "image/png" else strip_jpeg_metadata(raw)
        stripped = len(cleaned) != len(raw)
        out = cleaned
        if max(w, h) > MAX_SIDE:
            return None, (f"this server cannot shrink an image (Pillow is not installed) and it is {w}x{h}; "
                          f"send it with the longest side at most {MAX_SIDE} px"), 415
    return {"mime": real, "data": base64.b64encode(out).decode("ascii"), "width": w, "height": h,
            "original": {"width": w, "height": h} if not resized else None,
            "resized": resized, "metadata_stripped": stripped}, "", 200


# ------------------------------------------------------------------ the AI answer, made safe
def _num(v, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    if lo is not None and x < lo:
        return None
    if hi is not None and x > hi:
        return None
    return round(x, 4)


def _txt(v, limit: int = 160) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:limit]


_MM_KEYS = ("mm", "value_mm", "size_mm", "dimension_mm", "length_mm", "width_mm", "height_mm", "dia_mm")


def _strip_mm(obj, counter: list) -> None:
    """Remove any key that claims millimetres outside dims_written (the AI likes to invent them)."""
    if isinstance(obj, dict):
        for k in list(obj):
            if k in _MM_KEYS:
                obj.pop(k, None)
                counter[0] += 1
            else:
                _strip_mm(obj[k], counter)
    elif isinstance(obj, list):
        for v in obj:
            _strip_mm(v, counter)


def sanitize(raw, mode: str, reference: dict | None) -> dict:
    """The AI answer -> the contract of V1, with every millimetre that was not written or referenced removed."""
    raw = raw if isinstance(raw, dict) else {}
    invented = [0]
    _strip_mm({k: v for k, v in raw.items() if k not in ("dims_written", "missing_part")}, invented)

    views = [str(v).lower() for v in (raw.get("views") or []) if str(v).lower() in VIEWS]
    feats = []
    for f in (raw.get("features") or [])[:24]:
        if not isinstance(f, dict):
            continue
        kind = str(f.get("kind") or "other").lower()
        feats.append({"name": _txt(f.get("name"), 60) or kind,
                      "kind": kind if kind in FEATURE_KINDS else "other",
                      "count": max(1, int(_num(f.get("count"), 1) or 1)),
                      "relation": _txt(f.get("relation"), 120)})
    props = []
    for p in (raw.get("proportions") or [])[:24]:
        if not isinstance(p, dict):
            continue
        val = _num(p.get("value"), 0.0)
        if val is None:
            continue
        if val > 1.0:                    # a ratio bigger than the whole cannot be a ratio: it was a size
            invented[0] += 1
            continue
        props.append({"name": _txt(p.get("name"), 60), "ratio_to": _txt(p.get("ratio_to"), 60),
                      "value": val, "confidence": _num(p.get("confidence"), 0.0, 1.0) or 0.0})
    dims = []
    for d in (raw.get("dims_written") or [])[:16]:
        if not isinstance(d, dict):
            continue
        seen = _txt(d.get("text_seen"), 60)
        mm = _num(d.get("value_mm"), 0.0)
        if not seen or mm is None:       # no text seen = the AI made the number up
            invented[0] += 1 if mm is not None else 0
            continue
        dims.append({"text_seen": seen, "value_mm": mm, "applies_to": _txt(d.get("applies_to"), 60),
                     "confidence": _num(d.get("confidence"), 0.0, 1.0) or 0.0})
    sym = str(raw.get("symmetry") or "none").lower()
    interp = {"object": _txt(raw.get("object"), 80) or "unclear object",
              "views": views or ["perspective"], "features": feats, "proportions": props,
              "dims_written": dims, "symmetry": sym if sym in SYMMETRY else "none",
              "unclear": [_txt(u, 120) for u in (raw.get("unclear") or [])[:8] if _txt(u)]}

    missing = None
    if mode == "missing_part" and isinstance(raw.get("missing_part"), dict):
        m = raw["missing_part"]
        loads = _txt(m.get("loads_guess"), 200)
        loads_clean = re.sub(r"\d+(?:[.,]\d+)?", "", loads).strip(" ,;:-")     # V1: words, no numbers
        loads_removed = loads_clean != loads
        missing = {"location": _txt(m.get("location"), 160),
                   "neighbours": [{"name": _txt(n.get("name"), 60), "interface_kind": _txt(n.get("interface_kind"), 40)}
                                  for n in (m.get("neighbours") or [])[:8] if isinstance(n, dict)],
                   "function": _txt(m.get("function"), 240),
                   "loads_guess": loads_clean if loads_removed else loads,
                   "loads_numbers_removed": loads_removed,
                   "interfaces_needed": [{"between": _txt(i.get("between"), 80),
                                          "what_to_measure": _txt(i.get("what_to_measure"), 120)}
                                         for i in (m.get("interfaces_needed") or [])[:10] if isinstance(i, dict)]}
    questions = [_txt(q, 160) for q in (raw.get("questions") or [])[:MAX_QUESTIONS] if _txt(q)]
    return {"interpretation": interp, "missing_part": missing, "questions": questions,
            "invented_removed": invented[0], "scale": scale_of(interp, reference)}


def scale_of(interp: dict, reference: dict | None) -> dict:
    """Where the millimetres come from - or the honest statement that there are none."""
    ref = reference if isinstance(reference, dict) else None
    mm = _num((ref or {}).get("mm"), 0.0)
    dims = interp.get("dims_written") or []
    if mm:
        return {"set": True, "source": "reference", "mm": mm, "what": _txt(ref.get("what"), 60) or "reference size",
                "mm_per_unit": None,
                "label": f"Scale from: {_txt(ref.get('what'), 60) or 'reference size'} {mm:g} mm (engineer)",
                "note": "one length the engineer stated; every other size is a proportion of it"}
    if dims:
        return {"set": True, "source": "written dims", "mm": None, "what": ", ".join(d["text_seen"] for d in dims[:4]),
                "mm_per_unit": None,
                "label": f"Scale from written dimensions: {len(dims)} size(s) read off the drawing and confirmed",
                "note": "only the sizes that are written and confirmed are millimetres"}
    return {"set": False, "source": None, "mm": None, "what": None, "mm_per_unit": None,
            "label": NO_SCALE_LABEL,
            "note": "a photo or a sketch has no scale bar; give one reference length (for example 'overall width "
                    "120 mm') and every proportion becomes a size"}


# ------------------------------------------------------------------ V1: read the image
def providers() -> list:
    """Providers that can see an image, in the order the server wants them tried."""
    raw = os.environ.get("IDRIS_VISION_ORDER", "claude,openai,gemini")
    return [p for p in (x.strip() for x in raw.split(",")) if p in llm.VISION_PROVIDERS]


def _ask_text(params: dict) -> str:
    note = _txt(params.get("note"), 600)
    ref = params.get("reference") if isinstance(params.get("reference"), dict) else None
    region = params.get("region") if isinstance(params.get("region"), dict) else None
    lines = [f"Mode: {params.get('mode')}."]
    if params.get("mode") == "sketch":
        lines.append("This is a hand sketch or a drawing. Read the views and the features; read a size only "
                     "where a number is written.")
    elif params.get("mode") == "photo":
        lines.append("This is a photograph of a real object. Read its shape and proportions; the camera "
                     "distorts size, so no measured millimetres.")
    else:
        lines.append("This is a photo of an assembly with a part missing. Describe where the gap is, which "
                     "parts it sits between, what the missing part must do, and where the engineer must "
                     "measure to fit it.")
    if note:
        lines.append("Engineer's note: " + note)
    if ref and _num(ref.get("mm"), 0.0):
        lines.append(f"The engineer states one reference size: {_txt(ref.get('what'), 60) or 'a size'} = "
                     f"{_num(ref.get('mm'), 0.0):g} mm. Use it only to make the proportions usable; do not "
                     f"invent any other millimetre.")
    if region:
        lines.append("The engineer circled the missing part at " + json.dumps(
            {k: region.get(k) for k in ("x", "y", "w", "h", "unit") if region.get(k) is not None})[:200]
            + " (fraction of the image). Look only there for the missing part.")
    return "\n".join(lines)


def read(image_base64, mime: str | None = None, mode: str = "sketch", note=None, reference=None,
         region=None, provider: str | None = None) -> tuple:
    """-> (status, payload). The payload is the V1 contract; a failure names what was wrong, never a guess."""
    if mode not in MODES:
        return 400, {"success": False, "error": f"mode must be one of {', '.join(MODES)}"}
    clean, err, status = prepare(image_base64, mime)
    if err:
        return status, {"success": False, "error": err}
    if not providers():
        return 503, {"success": False,
                     "error": "no AI with image input is configured on the server (a Claude, OpenAI or Gemini "
                              "key is needed)"}
    params = {"mode": mode, "note": note, "reference": reference, "region": region}
    text = _ask_text(params)
    cands = [provider] if provider in llm.VISION_PROVIDERS else providers()
    attempts = []
    for prov in cands:
        try:
            out, model = llm.call_vision(prov, _SYSTEM, text,
                                         [{"mime": clean["mime"], "data": clean["data"]}], 2000, 90.0)
            raw = llm.extract_json(out)
        except llm.LLMError as e:
            attempts.append({"provider": prov, "error": str(e)[:200]})
            continue
        except Exception as e:                    # an answer IDRIS cannot read is a rejected answer
            attempts.append({"provider": prov, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        safe = sanitize(raw, mode, reference)
        safe["readings"] = readings(safe)
        payload = {"success": True, "mode": mode, **safe,
                   "read_by": f"{prov} ({model})", "attempts": attempts,
                   "image": {"mime": clean["mime"], "width": clean["width"], "height": clean["height"],
                             "resized": clean["resized"], "metadata_stripped": clean["metadata_stripped"],
                             "max_side": MAX_SIDE},
                   "rules": {
                       "sizes": "only dims_written (confirmed by the engineer) and the reference are millimetres",
                       "proportions": "ratios to a named feature, with a confidence",
                       "shape_match": "POST /vision/similarity compares silhouettes; it is not a 3D claim"}}
        if mode == "missing_part":
            payload["questions"] = (payload["questions"] or [])[:MAX_QUESTIONS]
        return 200, payload
    return 422, {"success": False, "error": "the AI could not read this image", "attempts": attempts}


# ------------------------------------------------------------------ V2: the design that follows
def confirmed_sizes(params: dict) -> dict:
    """The millimetres the engineer confirmed, plus the reference. Anything else is not a size."""
    out = {}
    dims = params.get("confirmed_dims")
    if isinstance(dims, dict):
        for k, v in list(dims.items())[:24]:
            mm = _num(v, 0.0)
            if mm is not None:
                out[_txt(k, 40)] = mm
    elif isinstance(dims, list):
        for d in dims[:24]:
            if isinstance(d, dict):
                mm = _num(d.get("value_mm"), 0.0)
                if mm is not None:
                    out[_txt(d.get("applies_to") or d.get("text_seen"), 40)] = mm
    ref = params.get("reference") if isinstance(params.get("reference"), dict) else None
    if ref and _num(ref.get("mm"), 0.0):
        out.setdefault(_txt(ref.get("what"), 40) or "reference", _num(ref.get("mm"), 0.0))
    return out


def readings(payload: dict, max_n: int = 3) -> list:
    """V2: when the sketch is ambiguous, the readings the engineer chooses between - stated, not invented.

    The AI does not get to pick silently: an `unclear` item becomes two named readings, the engineer accepts one
    (or none), and the job is told which. No size is affected by the choice.
    """
    interp = (payload or {}).get("interpretation") or {}
    out = []
    for i, item in enumerate((interp.get("unclear") or [])[:max_n]):
        out.append({"n": i + 1, "unclear": item,
                    "reading_a": f"as drawn: take \"{item}\" as the shape shows it",
                    "reading_b": f"simplest: leave \"{item}\" out of this first design",
                    "effect": "changes the shape only; every confirmed size stays what it is"})
    return out


def missing_interfaces(payload: dict, confirmed: dict) -> list:
    """Which interface sizes the engineer still has to measure before a missing part may be designed."""
    missing = (payload or {}).get("missing_part") or {}
    needed = []
    for i in (missing.get("interfaces_needed") or []):
        what = str(i.get("what_to_measure") or "").strip()
        if not what:
            continue
        if not any(what.lower() in k.lower() or k.lower() in what.lower() for k in confirmed):
            needed.append(i)
    return needed


def job_request(payload: dict, note=None, confirmed: dict | None = None) -> str:
    """The sentence the design job starts from: the engineer's words, never a size the image did not carry."""
    interp = payload.get("interpretation") or {}
    scale = payload.get("scale") or scale_of(interp, None)
    bits = [str(note or "").strip() or f"Design a {interp.get('object') or 'part'}"]
    feats = interp.get("features") or []
    if feats:
        bits.append("Features: " + "; ".join(
            f"{f['count']}x {f['name']}" + (f" ({f['relation']})" if f.get("relation") else "") for f in feats[:10]))
    if confirmed:
        bits.append("Confirmed sizes (engineer): " + ", ".join(f"{k} = {v:g} mm" for k, v in confirmed.items()))
    missing = payload.get("missing_part") or {}
    if missing:
        bits.append("Missing part: " + str(missing.get("location") or "in the assembly")
                    + ". It must: " + str(missing.get("function") or "fit its neighbours"))
    reading = str((payload or {}).get("reading") or "").strip()
    if reading:
        bits.append("The engineer chose this reading of the unclear part: " + reading[:200])
    # cad_codegen reads this sentence as the user request, so the rules travel with it: the design job is the
    # one place that has to know that a proportion is not a millimetre.
    bits.append("Shape from the picture: " + json.dumps(
        {"views": interp.get("views"), "symmetry": interp.get("symmetry"),
         "proportions": interp.get("proportions"), "unclear": interp.get("unclear")}, ensure_ascii=False)[:700])
    bits.append("Scale: " + str(scale.get("label") or "not set") + ". " + str(scale.get("note") or ""))
    bits.append("Rules: the picture gives shape and proportion, not a measured size; use only the confirmed "
                "sizes above, and turn a proportion into millimetres only through the scale"
                + ("" if scale.get("set") else
                   " (no scale is set, so pick ONE provisional overall size, keep every other size a proportion "
                   "of it, and say in the note that the scale is provisional)")
                + ". Do not claim strength, stiffness, tolerance or a measured size"
                + ("" if scale.get("set") else "; no strength check may run on a provisional scale") + ".")
    return ". ".join(b for b in bits if b)[:1800]


def job_context(payload: dict, confirmed: dict | None = None) -> dict:
    """`context.vision` for cad_codegen: the interpretation, the confirmed sizes, and the scale it may use."""
    interp = payload.get("interpretation") or {}
    scale = payload.get("scale") or scale_of(interp, None)
    return {"vision": {"mode": payload.get("mode"), "object": interp.get("object"),
                       "fea_allowed": bool(scale.get("set")),
                       "fea_refused_because": None if scale.get("set") else
                       "a provisional scale makes a safety factor a made-up number",
                       "views": interp.get("views"), "features": interp.get("features"),
                       "proportions": interp.get("proportions"), "symmetry": interp.get("symmetry"),
                       "unclear": interp.get("unclear"),
                       "confirmed": confirmed or {}, "scale": scale,
                       "missing_part": payload.get("missing_part"),
                       "reading": payload.get("reading"),
                       "label": scale.get("label")}}


# ------------------------------------------------------------------ V3: a measured shape match
def _polys(sil) -> list:
    polys = []
    for poly in (sil or [])[:64]:
        pts = []
        for p in (poly or [])[:2000]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = _num(p[0]), _num(p[1])
                if x is not None and y is not None:
                    pts.append((x, y))
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def _is_bitmap(sil) -> bool:
    """True for [[0,1,1],[1,1,0]] (a mask the browser already rasterized) rather than [[[x,y],...]] polygons."""
    if not isinstance(sil, list) or not sil or not isinstance(sil[0], list) or not sil[0]:
        return False
    first = sil[0][0]
    return isinstance(first, (int, float)) or isinstance(first, bool)


def raster(sil, size: int = 256):
    """Silhouettes -> a boolean grid.

    Two shapes are accepted: polygons in 0..1 coordinates (filled with the even-odd rule, so a hole stays a hole)
    and a bitmap of rows of 0/1 that the browser already produced. A bitmap is resampled nearest-neighbour, so
    the caller's resolution does not change the answer much - and the resolution used is reported back.
    """
    import numpy as np
    if _is_bitmap(sil):
        rows = [r for r in sil if isinstance(r, list) and r]
        h = len(rows)
        w = min(len(r) for r in rows)
        if not h or not w:
            return np.zeros((size, size), dtype=bool)
        src = np.array([[1 if v else 0 for v in r[:w]] for r in rows], dtype=bool)
        yi = (np.arange(size) * h // size).clip(0, h - 1)
        xi = (np.arange(size) * w // size).clip(0, w - 1)
        return src[yi][:, xi]
    img = np.zeros((size, size), dtype=bool)
    polys = _polys(sil)
    if not polys:
        return img
    ys = np.arange(size, dtype=float) + 0.5
    for poly in polys:
        pts = [(x * size, y * size) for x, y in poly]
        n = len(pts)
        for row, y in enumerate(ys):
            edges = []
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                if (y1 <= y < y2) or (y2 <= y < y1):
                    edges.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            edges.sort()
            for a, b in zip(edges[0::2], edges[1::2]):        # even-odd: inside between pairs of crossings
                lo, hi = max(0, int(round(a))), min(size, int(round(b)))
                if hi > lo:
                    img[row, lo:hi] ^= True
    return img


def similarity(model_sil, image_sil, size: int = 256) -> dict:
    """Silhouette IoU in percent, plus the biggest area where the two disagree.

    The numbers are measured from the two outlines the caller sends; nothing is inferred about 3D shape.
    """
    a, b = raster(model_sil, size), raster(image_sil, size)
    area_a, area_b = int(a.sum()), int(b.sum())
    if not area_a or not area_b:
        return {"ok": False, "error": "both silhouettes need a closed outline with an area",
                "area_model": area_a, "area_image": area_b}
    inter = int((a & b).sum())
    union = area_a + area_b - inter
    iou = 100.0 * inter / union if union else 0.0
    diff = a ^ b
    return {"ok": True, "iou_percent": round(iou, 1), "mismatch_percent": round(100.0 - iou, 1),
            "area_model": round(area_a / float(size * size), 4),
            "area_image": round(area_b / float(size * size), 4),
            "only_model_percent": round(100.0 * int((a & ~b).sum()) / max(1, union), 1),
            "only_image_percent": round(100.0 * int((b & ~a).sum()) / max(1, union), 1),
            "biggest_region": _biggest_region(diff, size),
            "view": "silhouette comparison; ask for one view at a time, never a 3D claim",
            "resolution": size}


def _biggest_region(diff, size: int) -> dict | None:
    """The largest block of disagreement, as a bounding box in percent (a coarse 32x32 grid, no scipy)."""
    if not diff.any():
        return None
    cell = max(1, size // 32)
    rows, cols = size // cell, size // cell
    blocks = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            blocks[r][c] = bool(diff[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].any())
    best, seen = None, set()
    for r in range(rows):
        for c in range(cols):
            if not blocks[r][c] or (r, c) in seen:
                continue
            stack, cells = [(r, c)], []
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):      # 4-connected: a diagonal touch is not one area
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and blocks[nr][nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            if best is None or len(cells) > best[0]:
                best = (len(cells), cells)
    if not best:
        return None
    rs = [c[0] for c in best[1]]
    cs = [c[1] for c in best[1]]
    return {"x_percent": round(100.0 * min(cs) / cols, 1), "y_percent": round(100.0 * min(rs) / rows, 1),
            "w_percent": round(100.0 * (max(cs) - min(cs) + 1) / cols, 1),
            "h_percent": round(100.0 * (max(rs) - min(rs) + 1) / rows, 1),
            "share_of_image_percent": round(100.0 * len(best[1]) * cell * cell / float(size * size), 1)}
