"""AI writes CAD code (replicad) for any non-gearbox object; the browser (UI v55) builds and checks it.

Contract: ui/perintah-solver-ai-menulis-kode-cad-v55.md and ui/pesan-arya-untuk-solver-mark1-desain-apa-saja.md.

What the server does:
  * builds the prompt (replicad API summary, look rules, process rules, standard interface tables, examples);
  * asks the strongest configured AI (order IDRIS_CAD_ORDER, default claude,openai,gemini,nvidia);
  * validates the answer: forbidden words, `return { body }`, params in range, `source: user` only for numbers
    that really are in the user's sentence, `expect` only for sizes the user gave; sizes from the user's world
    that were not given -> `missing` (no code is sent then);
  * one automatic retry when the static check fails; repair loop with context.cad_error (max 2 per design).
What the server does NOT do: run the code. Geometry, volume and checks come from the kernel in the browser.
No strength number is produced here: kind/labels say "not checked" until FEA (Mark 1 #7) runs.
"""
from __future__ import annotations

import json
import os
import re

import llm

KINDS = ("functional", "interface", "styling")
PROCESSES = ("fdm", "cnc", "sheet", "weld")
MATERIALS = ("aluminium", "steel", "stainless", "titanium", "brass", "pla", "abs", "nylon")
FORBIDDEN = ("import", "eval", "Function", "constructor", "prototype", "globalThis", "self", "window", "document",
             "fetch", "XMLHttpRequest", "WebSocket", "setTimeout", "setInterval", "WebAssembly", "arguments",
             "require", "process", "__proto__")
MAX_CODE = 20000
MAX_REPAIRS = 2

# ------------------------------------------------------------------ standard interfaces (tables, not guesses)
from physics import catalog as _cat
ISO273_MEDIUM = {k: v[3] for k, v in _cat.METRIC.items() if float(k[1:]) <= 16}   # ISO 273 medium series (mm)
ISO4762_HEAD = {"M3": (5.5, 3), "M4": (7, 4), "M5": (8.5, 5), "M6": (10, 6), "M8": (13, 8), "M10": (16, 10)}
# socket head cap screw nominal head diameter dk, head height k (ISO 4762, mm)
INTERFACES = {
    "handlebar_clamp_mm": [22.2, 25.4, 28.6, 31.8],       # bar diameters at the clamp (7/8", 1", 1-1/8", 31.8 mm)
    "handlebar_grip_mm": [22.2, 25.4],                    # grip section diameter
    "model_rocket_motor_mm": [13, 18, 24, 29, 38],        # NAR/TRA motor diameters
    "iso273_clearance_mm": {k: {"fine": v[2], "medium": v[3], "coarse": v[4]} for k, v in _cat.METRIC.items()
                            if 2 <= float(k[1:]) <= 16 and k not in ("M3.5", "M7")},
    "tap_drill_mm": {k: v[1] for k, v in _cat.METRIC.items() if 2 <= float(k[1:]) <= 16 and k not in ("M3.5", "M7")},
    "iso4762_head_dk_k_mm": ISO4762_HEAD,
    "deep_groove_bearings_d_D_B_mm": {"608": [8, 22, 7], "625": [5, 16, 5], "6000": [10, 26, 8],
                                      "6001": [12, 28, 8], "6200": [10, 30, 9], "6201": [12, 32, 10],
                                      "6202": [15, 35, 11], "6204": [20, 47, 14]},
    "raspberry_pi_4_board_mm": {"size": [85, 56], "holes_spacing": [58, 49], "hole_d": 2.7},
    "fdm_min_wall_mm": 1.2, "fdm_max_overhang_deg": 45,
}

API = """replicad API you may use (all are in scope; parameters come from object P; units mm; Z up; for FDM the
part sits on z = 0):
- makeBox([x0,y0,z0],[x1,y1,z1]), makeCylinder(r, h, [x,y,z]?, [dx,dy,dz]?), makeSphere(r)
- draw([x,y]?) pen: .hLine(dx) .vLine(dy) .line(dx,dy) .lineTo([x,y]) .polarLine(dist,deg)
  .cubicBezierCurveTo([x,y], [c1x,c1y], [c2x,c2y]) .threePointsArcTo([x,y],[mx,my]) .tangentArcTo([x,y])
  .customCorner(r) (rounds the corner just drawn) .close() -> Drawing
  Drawing: .sketchOnPlane("XY"|"XZ"|"YZ", offset?) -> Sketch; .fillet? no: use customCorner while drawing
- sketchCircle(r, {plane, origin}?), sketchRectangle(w, h, {plane, origin}?), sketchRoundedRectangle(w, h, r, {plane, origin}?)
- Sketch: .extrude(d, {extrusionDirection?, twistAngle?}) .revolve() (about the Z axis through the origin; draw the
  half-profile on "XZ" with x = radius >= 0, y = height) .loftWith([otherSketches]) .sweepSketch(fn)
- Shape: .fuse(s) .cut(s) .intersect(s) .fillet(r, (e) => e.inDirection("Z")) .chamfer(r, filter?) .shell(t, (f) => f.inPlane("XY", z))
  .translate([x,y,z]) .translateZ(z) .rotate(deg, [cx,cy,cz], [ax,ay,az]) .mirror("YZ"|"XZ"|"XY") .clone()
Plane "XZ" faces -Y: a sketch on "XZ" extruded by t goes from y = 0 to y = -t."""

RULES = """Rules:
1. Output ONE JSON object only (no prose, no markdown). Schema:
{"title": str, "kind": "functional"|"interface"|"styling", "process": "fdm"|"cnc"|"sheet"|"weld"|null,
 "material": one of aluminium,steel,stainless,titanium,brass,pla,abs,nylon,
 "params": [{"k": "L", "label": "Length", "label_id": "Panjang", "unit": "mm", "value": 40, "min": 10, "max": 200,
             "source": "user"|"default", "wall": true?}],
 "code": "const { L, t } = P; ... return { body };",
 "expect": [{"label": str, "axis": "x"|"y"|"z", "value": number, "tol": 0.5}],
 "missing": [{"k": str, "question": str}],
 "variants": [{"name": str, "params": {"k": value, ...}, "code": str?}],
 "holes": [{"x": mm, "y": mm, "z_top": mm, "axis": "z"|"x"|"y", "d": mm, "kind": "clearance"|"tapped"|"simple",
            "M": "M6"?, "thru": true|false, "depth": mm?}],
 "note": str}
2. NEVER invent sizes from the user's world (phone thickness, bar/clamp diameter, mounting space, motor size, board
   size). If the sentence does not give them and they are not in the standard tables below, put them in "missing"
   (question in the user's language) and send "code": "" . Design choices (radius, wall, angles, styling) may be
   "default" values inside sensible min/max.
3. Units: always mm in params. Convert inch (×25.4) and cm/m; a converted user number stays "source": "user".
   "source": "user" ONLY for numbers literally in the sentence. "expect" ONLY for overall sizes the user gave
   (x = length, y = width, z = height of the bounding box) - never for your own defaults.
4. Code is the body of a JavaScript function: read params with const { ... } = P; end with return { body } or
   return { body, probes: { name: solid } } (probes = reference objects not printed, e.g. the phone on a stand).
   Forbidden words: import eval Function constructor prototype globalThis self window document fetch
   XMLHttpRequest WebSocket setTimeout setInterval WebAssembly arguments require process; no obj['x'] access.
5. Make it look designed, not a stack of blocks: build from 2D profiles with rounded corners (customCorner) then
   extrude/revolve; curves with lines + cubicBezierCurveTo with explicit control points (NEVER
   drawPointsInterpolation); complex shapes as the intersection of side/top/front silhouettes; put small fillets in
   try { ... } catch (e) {} so a failed fillet never breaks the part; add functional details (cable notch,
   lightening holes, screw bosses, ribs) when they make sense.
6. Process rules: fdm -> walls >= 1.2 mm, no overhang beyond 45 deg (horizontal holes with a 45 deg pointed top,
   chamfer instead of fillet on the bed edge), fits a 220 x 220 x 250 mm bed unless the user says otherwise;
   cnc -> inside corner radius >= 3 mm (tool radius), no undercuts; sheet -> one thickness everywhere, bends with
   inside radius >= thickness.
7. Standard interfaces come from these tables, never guessed: """ + json.dumps(INTERFACES, separators=(",", ":")) + """
8. "variants": give 2 extra styles (e.g. minimal / sporty / sturdy) as different param values of the SAME code
   ("params" object of overrides); include "code" only if the style needs different geometry.
9. Geometry only. Do not claim strength, stability or aerodynamics; the solver checks those separately.
10. Every hole your code makes MUST be listed in "holes" (part coordinates, mm, for the default parameter values;
   x, y = hole axis position, z_top = where the hole starts on the top face for axis "z"). Bolt holes take their
   diameter from the tables: clearance = iso273_clearance (fine/medium/coarse) for "M", tapped = tap drill.
   Auto Assembly uses this list to bolt parts together; holes not listed cannot be bolted."""

EXAMPLES = r"""Example A (user: "bracket L 40 mm tebal 3 mm untuk baut M4, PLA"):
{"title":"L bracket","kind":"functional","process":"fdm","material":"pla",
 "params":[{"k":"L","label":"Leg length","label_id":"Panjang kaki","unit":"mm","value":40,"min":15,"max":200,"source":"user"},
  {"k":"W","label":"Width","label_id":"Lebar","unit":"mm","value":20,"min":10,"max":100,"source":"default"},
  {"k":"t","label":"Thickness","label_id":"Tebal","unit":"mm","value":3,"min":1.2,"max":12,"source":"user","wall":true},
  {"k":"h","label":"Hole (M4, ISO 273)","label_id":"Lubang (M4, ISO 273)","unit":"mm","value":4.5,"min":4.5,"max":4.5,"source":"default"}],
 "code":"const { L, W, t, h } = P;\nconst prof = draw([0,0]).hLine(L).vLine(t).hLine(-(L-t-3)).customCorner(3).line(-3,3).vLine(L-t-3).hLine(-t).close();\nlet body = prof.sketchOnPlane('XZ').extrude(-W);\nconst holeA = makeCylinder(h/2, t*3, [L*0.65, W/2, -t]);\nconst holeB = makeCylinder(h/2, t*3, [-t, W/2, L*0.65], [1,0,0]);\nbody = body.cut(holeA).cut(holeB);\ntry { body = body.fillet(1.5, (e) => e.inDirection('Y')); } catch (e) {}\nreturn { body };",
 "expect":[{"label":"Leg X","axis":"x","value":40,"tol":0.5},{"label":"Leg Z","axis":"z","value":40,"tol":0.5}],
 "missing":[],"variants":[{"name":"Sturdy","params":{"t":5,"W":25}},{"name":"Minimal","params":{"W":15}}],
 "holes":[{"x":26,"y":10,"z_top":3,"axis":"z","d":4.5,"kind":"clearance","M":"M4","thru":true},
  {"x":0,"y":10,"z_top":26,"axis":"x","d":4.5,"kind":"clearance","M":"M4","thru":true}],
 "note":"Horizontal hole in the vertical leg is round; for FDM print the bracket lying on its side."}

Example C (user: "holder hp"): sizes from the user's world are unknown ->
{"title":"Phone stand","kind":"functional","process":"fdm","material":"pla","params":[],"code":"",
 "expect":[],"missing":[{"k":"T","question":"How thick is the phone including its case (mm)?"},{"k":"Hp","question":"How tall is the phone (mm)?"}],
 "variants":[],"note":""}"""


# ------------------------------------------------------------------ routing
OBJECT_WORDS = (
    "holder", "dudukan", "stand", "mount", "bracket", "braket", "roket", "rocket", "drone", "frame", "rangka",
    "stang", "handlebar", "body mobil", "car body", "mobil", "kotak", "box elektronik", "enclosure", "case", "casing",
    "hook", "gantungan", "kait", "pot", "gagang", "handle", "knob", "pulley", "puli", "spacer", "flange", "klem",
    "clamp", "raspberry", "kamera", "camera", "tutup", "lid", "cover", "engsel", "hinge", "nozzle", "duct", "corong",
    "rak", "shelf", "gear knob", "pedal", "sirip", "fin", "propeller", "baling", "wheel", "roda", "vas", "vase",
    "organizer", "tray", "nampan", "cup", "gelas", "botol", "bottle", "adapter", "adaptor", "coupler", "kopling",
    "bushing", "standoff", "keychain", "gantungan kunci", "logo", "plakat", "sign", "mainan", "toy", "figure",
    "helm", "helmet", "kursi", "chair", "meja", "table", "lampu", "lamp", "speaker", "joystick", "remote",
    "circlip", "snap ring", "pin", "dowel", "spring", "pegas", "shaft", "poros", "gear", "spur gear",
    "sprocket", "rantai", "chain", "belt", "sabuk", "cam", "lever", "tuas", "hinge", "nozzle", "impeller",
    "propeller", "rivet", "paku keling", "threaded rod", "stud", "insert", "grommet", "seal", "gasket", "packing",
    "profil", "profile", "pipa", "tube", "beam", "balok i", "channel", "siku", "angle bar",
    "kunci", "wrench", "spanner", "sepeda", "bicycle", "bike", "motor", "sepeda motor", "pesawat", "aircraft",
    "airplane", "wing", "sayap", "kapal", "boat", "hull", "gitar", "guitar", "tas", "jam", "watch", "sendok",
    "spoon", "garpu", "pisau", "knife", "palu", "hammer", "obeng", "screwdriver", "tang", "pliers", "engsel",
    "rumah", "house", "bangunan", "building", "jembatan", "bridge", "tower", "menara",
    # German
    "halter", "handyhalter", "halterung", "winkel", "rakete", "gehäuse", "deckel", "haken", "griff", "lenker",
    "karosserie", "blumentopf", "abstandshalter", "flansch", "riemenscheibe", "knauf", "ständer", "kasten",
    # Polish
    "uchwyt", "wspornik", "rakieta", "obudowa", "pokrywa", "hak", "kierownica", "nadwozie", "doniczka",
    "dystans", "kołnierz", "koło pasowe", "gałka", "stojak", "pudełko", "rama")
DESIGN_VERBS = ("desain", "design", "buat", "bikin", "buatkan", "bikinin", "rancang", "rancangkan", "modelkan",
                "make", "create", "generate", "draw", "gambar", "entwirf", "entwerfen", "konstruiere", "zaprojektuj",
                "zrób", "stwórz")
GEAR_WORDS = ("gearbox", "roda gigi", "reducer", "gear ratio", "rasio", "ratio", " kw", "rpm", "torsi", "torque",
              "varian", "variant", "getriebe", "przekładnia", "drehmoment", "moment obrotowy")


def looks_like_object_request(message: str, has_gearbox: bool = False) -> bool:
    m = " " + (message or "").lower() + " "
    if any(w in m for w in GEAR_WORDS):
        return False
    edit = re.search(r"(?<!\w)(lebih|kurang|more|less|change|ubah|ganti|naikkan|turunkan|tambah|kurangi|it|itu|ini|"
                     r"the|thicker|thinner|tebal|tipis)(?!\w)", m)
    verb = any(re.search(rf"(?<!\w){v}(?!\w)", m) for v in DESIGN_VERBS)
    # long words may sit inside compounds (German "Handyhalter"); short ones must be whole words ("pot" != "potong")
    noun = any(re.search(re.escape(w) if len(w) >= 6 else rf"(?<!\w){re.escape(w)}(?!\w)", m) for w in OBJECT_WORDS)
    if has_gearbox:
        # a gearbox is open: "make the cover thicker" / "ganti seal" talk about it; only "design a new X" is new
        return bool(noun and verb and not edit)
    if noun:
        return True
    # "buatkan aku design X": a design verb with no gearbox words is a free-form object request,
    # but "buat lebih ringan" / "make it stronger" edits the current design
    if re.search(r"(?<!\w)(lebih|kurang|more|less|change|ubah|ganti|naikkan|turunkan|tambah|kurangi|it|itu|ini)(?!\w)", m):
        return False
    return any(re.search(rf"(?<!\w){v}(?!\w)", m) for v in DESIGN_VERBS)


def cad_order() -> list:
    raw = os.environ.get("IDRIS_CAD_ORDER", "claude,openai,gemini,nvidia")
    order = [p.strip() for p in raw.split(",") if p.strip() in llm.PROVIDERS]
    return [p for p in order if os.environ.get(llm.KEY_ENV[p])]


# ------------------------------------------------------------------ validation
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)")


def message_numbers(text: str) -> set:
    out = set()
    for tok in _NUM.findall(text or ""):
        v = float(tok.replace(",", "."))
        out.update({round(v, 3), round(v * 10, 3), round(v * 1000, 3), round(v * 25.4, 3), round(v * 25.4, 1),
                    round(v * 304.8, 1)})                                    # mm, cm, m, inch, ft
    return out


def check_code(code: str) -> list:
    errs = []
    if not isinstance(code, str) or not code.strip():
        return ["empty code"]
    if len(code) > MAX_CODE:
        errs.append(f"code longer than {MAX_CODE} characters")
    for w in FORBIDDEN:
        if re.search(rf"(?<![\w$]){re.escape(w)}(?![\w$])", code):
            errs.append(f"forbidden word: {w}")
    if re.search(r"\[\s*['\"`]", code):
        errs.append("property access with a string (obj['x']) is not allowed")
    if "drawPointsInterpolation" in code:
        errs.append("drawPointsInterpolation is not allowed (closed interpolated curves overshoot); use cubicBezierCurveTo")
    if not re.search(r"\breturn\s*\{[^}]*\bbody\b", code):
        errs.append("code must end with return { body } (optionally with probes)")
    return errs


def _num(v):
    try:
        f = float(v)
        return f if f == f and abs(f) < 1e7 else None
    except (TypeError, ValueError):
        return None


def validate(raw: dict, message: str = "") -> tuple:
    """-> (cad, errors, warnings). errors = reasons to ask the AI again; warnings = fixed silently and reported."""
    errs, warns = [], []
    if not isinstance(raw, dict):
        return None, ["answer is not a JSON object"], warns
    nums = message_numbers(message)
    cad = {"title": str(raw.get("title") or "Part")[:60],
           "kind": raw.get("kind") if raw.get("kind") in KINDS else "functional",
           "process": raw.get("process") if raw.get("process") in PROCESSES else None,
           "material": raw.get("material") if raw.get("material") in MATERIALS else "pla",
           "note": str(raw.get("note") or "")[:400]}
    missing = []
    for m in (raw.get("missing") or [])[:8]:
        if isinstance(m, dict) and m.get("question"):
            missing.append({"k": re.sub(r"\W", "", str(m.get("k") or "q"))[:16] or "q", "question": str(m["question"])[:200]})
    cad["missing"] = missing

    params, keys = [], set()
    for p in (raw.get("params") or [])[:24]:
        if not isinstance(p, dict):
            continue
        k = str(p.get("k") or "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,15}", k) or k in keys or k in FORBIDDEN:
            errs.append(f"bad or duplicate parameter name: {k!r}")
            continue
        v, lo, hi = _num(p.get("value")), _num(p.get("min")), _num(p.get("max"))
        if v is None:
            errs.append(f"parameter {k} has no numeric value")
            continue
        lo = v if lo is None else lo
        hi = v if hi is None else hi
        if lo > hi:
            lo, hi = hi, lo
        if not lo - 1e-9 <= v <= hi + 1e-9:
            errs.append(f"parameter {k} = {v:g} is outside its own range {lo:g}-{hi:g}")
        src = "user" if p.get("source") == "user" else "default"
        if src == "user" and round(v, 3) not in nums:
            src = "default"
            warns.append(f"{k} = {v:g} is not in the sentence, marked as a default (not from you)")
        q = {"k": k, "label": str(p.get("label") or k)[:40], "label_id": str(p.get("label_id") or p.get("label") or k)[:40],
             "unit": str(p.get("unit") or "mm")[:8], "value": v, "min": lo, "max": hi, "source": src}
        if p.get("wall"):
            q["wall"] = True
        params.append(q)
        keys.add(k)
    cad["params"] = params

    expect = []
    for e in (raw.get("expect") or [])[:6]:
        if not isinstance(e, dict) or e.get("axis") not in ("x", "y", "z"):
            continue
        v = _num(e.get("value"))
        if v is None or v <= 0:
            continue
        if round(v, 3) not in nums:
            warns.append(f"expected size {v:g} mm dropped: it is not in the sentence")
            continue
        expect.append({"label": str(e.get("label") or e["axis"])[:30], "axis": e["axis"], "value": v,
                       "tol": min(max(_num(e.get("tol")) or 0.5, 0.05), 5.0)})
    cad["expect"] = expect

    code = raw.get("code") or ""
    if missing:
        cad["code"] = None
    else:
        errs += check_code(code)
        used = set(re.findall(r"\bconst\s*\{([^}]*)\}\s*=\s*P\b", code or ""))
        names = {n.strip().split(":")[0].strip() for g in used for n in g.split(",") if n.strip()}
        unknown = sorted(n for n in names if n and n not in keys)
        if unknown:
            errs.append(f"code reads parameters that are not in params: {unknown}")
        cad["code"] = code

    holes = []
    for i, h in enumerate((raw.get("holes") or [])[:64]):
        if not isinstance(h, dict):
            continue
        x, y, zt, dd = (_num(h.get(k)) for k in ("x", "y", "z_top", "d"))
        if None in (x, y, zt) or dd is None or dd <= 0:
            warns.append(f"hole {i + 1} dropped: needs x, y, z_top and d")
            continue
        kind = h.get("kind") if h.get("kind") in ("clearance", "tapped", "simple") else "simple"
        q = {"x": x, "y": y, "z_top": zt, "axis": h.get("axis") if h.get("axis") in ("x", "y", "z") else "z",
             "d": dd, "kind": kind, "thru": bool(h.get("thru", True))}
        if _num(h.get("depth")):
            q["depth"] = _num(h.get("depth"))
        size = _cat.norm_size(h.get("M")) if h.get("M") else None
        if size:
            q["M"] = size
            row = _cat.METRIC[size]
            ok = (abs(dd - row[1]) <= 0.05) if kind == "tapped" else any(abs(dd - v) <= 0.05 for v in row[2:])
            q["catalog_ok"] = ok if kind in ("clearance", "tapped") else None
            if kind in ("clearance", "tapped") and not ok:
                warns.append(f"hole {i + 1}: Ø{dd:g} is not the ISO {'tap drill' if kind == 'tapped' else '273 clearance'} "
                             f"for {size}")
        holes.append(q)
    cad["holes"] = holes if not missing else []

    variants = []
    for v in (raw.get("variants") or [])[:4]:
        if not isinstance(v, dict):
            continue
        over = v.get("params") if isinstance(v.get("params"), dict) else {}
        clean = {}
        for k, val in over.items():
            p = next((x for x in params if x["k"] == k), None)
            f = _num(val)
            if p and f is not None and p["min"] <= f <= p["max"]:
                clean[k] = f
        vc = v.get("code")
        if vc:
            ve = check_code(vc)
            if ve:
                warns.append(f"variant {v.get('name')!r} dropped: {ve[0]}")
                continue
        if clean or vc:
            variants.append({"name": str(v.get("name") or f"Variant {len(variants) + 1}")[:30], "params": clean,
                             **({"code": vc} if vc else {})})
    cad["variants"] = variants if not missing else []
    return cad, errs, warns


# ------------------------------------------------------------------ AI call
def lang_rule(reply_language: str | None) -> str:
    if (reply_language or default_language()) == "en":
        return ("LANGUAGE: the user may write in any language (Indonesian, German, Polish, ...). Read it fully, but write "
                "EVERY user-facing text in ENGLISH: title, label, missing questions, note, critique, variant names. "
                "label_id may stay in Indonesian.")
    return "LANGUAGE: write user-facing texts in the user's language."


def default_language() -> str:
    return "auto" if os.environ.get("IDRIS_REPLY_LANGUAGE", "en").lower() == "auto" else "en"


def _system(reply_language: str | None = None, message: str | None = None) -> str:
    import cad_recipes
    return ("You are the CAD code writer of IDRIS. You turn a request into parametric replicad code that a CAD "
            "kernel in the browser builds and checks.\n\n" + API + "\n\n" + RULES + "\n\n"
            + lang_rule(reply_language) + "\n\n" + EXAMPLES + "\n\n" + cad_recipes.prompt_block(message))


def _user_block(message: str, context: dict | None) -> str:
    ctx = context or {}
    parts = [f"User request: {message}"]
    pend = ctx.get("pending_cad") if isinstance(ctx.get("pending_cad"), dict) else None
    if pend:
        parts.append("This answers your earlier questions for: " + json.dumps(
            {"title": pend.get("title"), "request": pend.get("request"), "missing": pend.get("missing")},
            ensure_ascii=False)[:1500])
    err = ctx.get("cad_error") if isinstance(ctx.get("cad_error"), dict) else None
    if err:
        parts.append("Your previous code FAILED in the CAD kernel. Fix it and answer with the full JSON again "
                     "(same params unless the fix needs a change). Error: " + str(err.get("error"))[:800]
                     + "\nPrevious code:\n" + str(err.get("code"))[:MAX_CODE]
                     + "\nPrevious params: " + json.dumps(err.get("params"), ensure_ascii=False)[:1500])
    return "\n\n".join(parts)


def generate(message: str, context: dict | None = None, provider: str | None = None,
             reply_language: str | None = None) -> dict:
    """-> {ok, cad?, answer, attempts, read_by}. Raises llm.LLMError only when no AI is configured."""
    ctx = context or {}
    err = ctx.get("cad_error") if isinstance(ctx.get("cad_error"), dict) else None
    attempt = int(_num((err or {}).get("attempt")) or 1) if err else 0
    if err and attempt > MAX_REPAIRS:
        return {"ok": False, "attempts": [], "read_by": None,
                "answer": (f"IDRIS tried {MAX_REPAIRS} automatic fixes and the code still does not build "
                           f"({str(err.get('error'))[:160]}). Try a simpler description, or fewer details at once.")}
    cands = [provider] if provider in llm.PROVIDERS else cad_order()
    if not cands:
        raise llm.LLMError("no AI provider is configured on the server", 503)
    full_msg = " ".join(str(x) for x in (message, (ctx.get("pending_cad") or {}).get("request") if isinstance(
        ctx.get("pending_cad"), dict) else "") if x)
    lang = "en" if (reply_language or default_language()) == "en" else "auto"
    system, user = _system(lang, full_msg), _user_block(message, ctx)
    attempts = []
    max_tokens = int(os.environ.get("IDRIS_CAD_MAX_TOKENS", "6000"))
    import time
    t_start = time.time()
    deadline = t_start + float(os.environ.get("IDRIS_CAD_BUDGET_S", "85"))
    per_call = float(os.environ.get("IDRIS_CAD_CALL_S", "45"))
    for prov in cands:
        retry_note = ""
        for round_ in (0, 1):            # one automatic retry when the static check fails
            left = deadline - time.time()
            if left < 12:
                attempts.append({"provider": prov, "error": "skipped: time budget used up"})
                break
            t0 = time.time()
            try:
                text, model = llm.call_provider(prov, system, user + retry_note, max_tokens,
                                                timeout=min(left - 2, per_call))
            except llm.LLMError as e:
                attempts.append({"provider": prov, "error": str(e), "seconds": round(time.time() - t0, 1)})
                break
            try:
                raw = llm.extract_json(text)
            except llm.LLMError:
                raw = llm.salvage_json(text) if hasattr(llm, "salvage_json") else None
            cad, errs, warns = validate(raw if isinstance(raw, dict) else None, full_msg)
            if not errs and cad:
                cad["read_by"] = f"{prov} ({model})"
                if warns:
                    cad["warnings"] = warns
                return {"ok": True, "cad": cad, "attempts": attempts, "read_by": cad["read_by"],
                        "seconds": round(time.time() - t_start, 1),
                        "answer": answer_text(cad, repaired=bool(err), lang=lang, message=full_msg)}
            attempts.append({"provider": prov, "model": model, "error": "; ".join(errs)[:300],
                             "seconds": round(time.time() - t0, 1), "answer_chars": len(text or "")})
            retry_note = ("\n\nYour last answer was rejected by the server check: " + "; ".join(errs)[:600]
                          + ". Answer again with the corrected full JSON.")
    real = [a for a in attempts if not a["error"].startswith("skipped")]
    why = "; ".join(f"{a['provider']}: {a['error'][:140]}" + (f" ({a['seconds']:g} s)" if a.get("seconds") else "")
                    for a in (real or attempts)[:4])
    return {"ok": False, "attempts": attempts, "read_by": None, "seconds": round(time.time() - t_start, 1),
            "answer": "IDRIS could not get valid CAD code from the AI this time. " + (why or "no answer")
                      + ". Please try again, or describe the part more simply."}


def answer_text(cad: dict, repaired: bool = False, lang: str = "en", message: str = "") -> str:
    idn = lang != "en" and llm._looks_indonesian(message)
    if cad.get("missing"):
        head = (f"Sebelum mendesain {cad['title']}, IDRIS perlu tahu:" if idn
                else f"Before designing the {cad['title']}, IDRIS needs to know:")
        return head + "\n" + "\n".join("- " + m["question"] for m in cad["missing"])
    if idn:
        label = {"styling": "Konsep (bentuk, bukan hasil analisis)", "interface": "Bagian antarmuka: kekuatan belum dicek",
                 "functional": "Kekuatan belum dicek"}[cad["kind"]]
        lines = [("Kode diperbaiki: " if repaired else "") + f"{cad['title']}: kode CAD ditulis oleh {cad['read_by']}. "
                 "Browser membangun dan mengecek geometrinya (solid, ukuran, aturan proses).", f"Status: {label}."]
        user_params = [p for p in cad["params"] if p["source"] == "user"]
        if user_params:
            lines.append("Dari kamu: " + ", ".join(f"{p['label_id']} {p['value']:g} {p['unit']}".strip() for p in user_params))
        if cad.get("variants"):
            lines.append("Varian: " + ", ".join(v["name"] for v in cad["variants"]))
        if cad.get("warnings"):
            lines.append("Catatan: " + "; ".join(cad["warnings"]))
        return "\n".join(lines)
    label = {"styling": "Concept (shape, not an analysis result)", "interface": "Interface part: strength not checked",
             "functional": "Strength not checked"}[cad["kind"]]
    lines = [("Code fixed: " if repaired else "") + f"{cad['title']}: CAD code written by {cad['read_by']}. "
             "The browser builds it and checks the geometry (solid, sizes, process rules).", f"Status: {label}."]
    user_params = [p for p in cad["params"] if p["source"] == "user"]
    if user_params:
        lines.append("From you: " + ", ".join(f"{p['label']} {p['value']:g} {p['unit']}".strip() for p in user_params))
    if cad.get("variants"):
        lines.append("Variants: " + ", ".join(v["name"] for v in cad["variants"]))
    if cad.get("warnings"):
        lines.append("Note: " + "; ".join(cad["warnings"]))
    return "\n".join(lines)


# ------------------------------------------------------------------ "eyes": AI looks at renders and improves
MAX_REVIEW_ROUNDS = 2
MAX_IMAGES = 4
MAX_IMAGE_B64 = 1_500_000          # characters of base64 per image (~1.1 MB)
VIEW_NAMES = ("iso", "front", "side", "top", "back", "bottom")


def vision_order() -> list:
    raw = os.environ.get("IDRIS_VISION_ORDER", "claude,openai,gemini")
    order = [p.strip() for p in raw.split(",") if p.strip() in llm.VISION_PROVIDERS]
    return [p for p in order if os.environ.get(llm.KEY_ENV[p])]


def clean_images(images) -> tuple:
    out, errs = [], []
    for i, im in enumerate((images or [])[:MAX_IMAGES]):
        if not isinstance(im, dict):
            continue
        data = str(im.get("png") or im.get("data") or "")
        mime = "image/jpeg" if str(im.get("mime", "")).endswith(("jpeg", "jpg")) else "image/png"
        if data.startswith("data:"):
            mime = data[5:data.find(";")] if ";" in data else mime
            data = data.split(",", 1)[-1]
        if mime not in ("image/png", "image/jpeg") or not re.fullmatch(r"[A-Za-z0-9+/=\s]+", data[:2000] or "!"):
            errs.append(f"image {i + 1}: must be base64 PNG or JPEG")
            continue
        if len(data) > MAX_IMAGE_B64:
            errs.append(f"image {i + 1}: larger than {MAX_IMAGE_B64 // 1000} kB of base64; render at 768 px")
            continue
        view = str(im.get("view") or "").lower()
        out.append({"mime": mime, "data": re.sub(r"\s", "", data), "view": view if view in VIEW_NAMES else f"view {i + 1}"})
    return out, errs


REVIEW_SYSTEM = """You are the design reviewer of IDRIS. You see renders of a part that the browser CAD kernel built
from the code below, plus the kernel's measurements. Judge it like a senior product designer and a mechanical
engineer:
- Does it look like what the user asked for, oriented sensibly (Z up, sits on z = 0 for FDM)?
- Does it look designed (good proportions, rounded profiles, smooth transitions, functional details) rather than a
  stack of boxes?
- Visible defects: floating or disconnected pieces, parts poking through each other, wrong side, missing feature,
  holes in the wrong place, fins/legs not touching the body.
Answer ONE JSON object only:
{"verdict": "ok", "critique": "<2-4 short sentences>"}  when it is good enough, or
{"verdict": "revise", "critique": "<what is wrong and what you change>", ...the FULL cad JSON (same schema as before:
title, kind, process, material, params, code, expect, missing, variants, holes, note)}.
When revising: keep every size the user gave (source "user", expect) unchanged; change only design choices and
code. Same code rules as before (forbidden words, return { body }, no drawPointsInterpolation).
Do not claim strength, stability or aerodynamics."""


def review(request: str, cad: dict, images: list, measurements: dict | None = None, round_: int = 1,
           provider: str | None = None, reply_language: str | None = None) -> dict:
    """-> {ok, verdict: ok|revise|stop, critique, cad?, read_by, attempts}."""
    if round_ > MAX_REVIEW_ROUNDS:
        return {"ok": True, "verdict": "stop", "critique": f"Reviewed {MAX_REVIEW_ROUNDS} times; keeping this version.",
                "attempts": [], "read_by": None}
    if not isinstance(cad, dict) or not cad.get("code"):
        return {"ok": False, "verdict": "stop", "critique": "no code to review", "attempts": [], "read_by": None}
    imgs, img_errs = clean_images(images)
    if not imgs:
        return {"ok": False, "verdict": "stop", "critique": "; ".join(img_errs) or "no images sent",
                "attempts": [], "read_by": None}
    cands = [provider] if provider in llm.VISION_PROVIDERS else vision_order()
    if not cands:
        return {"ok": False, "verdict": "stop", "attempts": [], "read_by": None,
                "critique": "No AI with image input is configured (Claude, OpenAI or Gemini key needed)."}
    text = ("User request: " + str(request or cad.get("title"))[:llm.MAX_MSG]
            + "\nViews in order: " + ", ".join(im["view"] for im in imgs)
            + "\nKernel measurements: " + json.dumps(measurements or {}, ensure_ascii=False)[:2000]
            + "\nCurrent cad JSON:\n" + json.dumps({k: cad.get(k) for k in (
                "title", "kind", "process", "material", "params", "code", "expect", "missing", "variants", "holes",
                "note")},
                ensure_ascii=False)[:MAX_CODE + 6000]
            + "\n\n" + lang_rule(reply_language)
            + "\n\nCode rules and API reminder:\n" + API + "\nStandard interfaces: " + json.dumps(INTERFACES))
    attempts = []
    for prov in cands:
        try:
            out, model = llm.call_vision(prov, REVIEW_SYSTEM, text, imgs,
                                         int(os.environ.get("IDRIS_CAD_MAX_TOKENS", "6000")), 90.0)
            raw = llm.extract_json(out)
        except llm.LLMError as e:
            attempts.append({"provider": prov, "error": str(e)})
            continue
        read_by = f"{prov} ({model})"
        critique = str(raw.get("critique") or "")[:800]
        if raw.get("verdict") != "revise":
            return {"ok": True, "verdict": "ok", "critique": critique, "read_by": read_by, "attempts": attempts}
        user_text = str(request or "") + " " + " ".join(f"{p['value']:g}" for p in cad.get("params", [])
                                                       if p.get("source") == "user")
        new, errs, warns = validate(raw, user_text)
        if errs or not new:
            attempts.append({"provider": prov, "model": model, "error": "revision rejected: " + "; ".join(errs)[:300]})
            continue
        lost = [e for e in cad.get("expect", []) if not any(
            n["axis"] == e["axis"] and abs(n["value"] - e["value"]) < 1e-6 for n in new["expect"])]
        if lost:
            attempts.append({"provider": prov, "model": model,
                             "error": "revision changed sizes the user gave: " + ", ".join(e["label"] for e in lost)})
            continue
        new["read_by"] = read_by
        if warns:
            new["warnings"] = warns
        return {"ok": True, "verdict": "revise", "critique": critique, "cad": new, "read_by": read_by,
                "attempts": attempts}
    return {"ok": False, "verdict": "stop", "attempts": attempts, "read_by": None,
            "critique": "Review failed: " + (attempts[-1]["error"] if attempts else "no answer")}
