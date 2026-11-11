"""IDRIS mechanical parts library (like the SOLIDWORKS Toolbox).

Each template = a standard part whose dimensions come from a table (ISO/DIN) and whose 3D shape is replicad code
that the browser kernel (UI v55 engine) builds. The engineer picks the size (and length where free); everything
else is looked up, never guessed. Output uses the same `cad` contract as AI-written parts, so Studio builds,
measures (expect) and exports it the same way.

Shapes are simplified where noted (threads are not modelled; bearing internals and circlip lugs are simplified).
Dimensions that matter for fit (d, D, B, head, across flats, key width/height, groove) are the table values.
"""
from __future__ import annotations

import math
import re

from physics import catalog as cat

# ------------------------------------------------------------------ extra tables (sources in SOURCES below)
HEX_ISO4017 = {   # k max, s, length range (ISO 4017 full thread hex screw)
    "M3": (2.2, 5.5, 5, 30), "M4": (3.0, 7, 6, 40), "M5": (3.74, 8, 8, 50), "M6": (4.24, 10, 12, 60),
    "M8": (5.54, 13, 16, 80), "M10": (6.69, 16, 20, 100), "M12": (7.79, 18, 25, 120), "M14": (9.09, 21, 25, 140),
    "M16": (10.29, 24, 30, 150), "M20": (12.85, 30, 40, 160)}
ISO4017_LENGTHS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120,
                   130, 140, 150, 160, 180, 200)
SHCS_SOCKET_T = {"M2": 1, "M3": 1.3, "M4": 2, "M5": 2.5, "M6": 3, "M8": 4, "M10": 5, "M12": 6, "M14": 7, "M16": 8,
                 "M20": 10, "M24": 12, "M30": 15.5}            # ISO 4762 socket depth t (min)
KEY_DIN6885 = [   # shaft d up to (mm): b, h, t1 shaft depth, t2 hub depth
    (8, 2, 2, 1.2, 1.0), (10, 3, 3, 1.8, 1.4), (12, 4, 4, 2.5, 1.8), (17, 5, 5, 3.0, 2.3), (22, 6, 6, 3.5, 2.8),
    (30, 8, 7, 4.0, 3.3), (38, 10, 8, 5.0, 3.3), (44, 12, 8, 5.0, 3.3)]
KEY_LENGTHS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 70, 80, 90, 100)
SOURCES = {
    "iso4762": cat.SOURCES["shcs"] + "; socket depth t = ISO 4762 min",
    "iso4017": "ISO 4017 via bd_warehouse (Apache-2.0); k = max, s nominal (M10 s = 16, ISO not DIN 933)",
    "iso4032": cat.SOURCES["nut"], "iso7089": cat.SOURCES["washer"], "bearing": cat.SOURCES["bearing"],
    "din6885": "DIN 6885-1 form A parallel key via bd_warehouse (Apache-2.0): b, h, t1 (shaft), t2 (hub)",
    "iso3601": "ISO 3601-1 O-ring inner diameter and cross-section via bd_warehouse (Apache-2.0)",
}


def _load_csv(name):
    import csv
    import os
    p = os.path.join(os.path.dirname(__file__), "data", name)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ shapes (replicad code, P = resolved params)
HEX = "const hex = (sf, z, hgt) => { const r = sf / Math.sqrt(3); let p = draw([r, 0]); " \
      "for (let i = 1; i < 6; i++) p = p.lineTo([r * Math.cos(i * Math.PI / 3), r * Math.sin(i * Math.PI / 3)]); " \
      "return p.close().sketchOnPlane('XY', z).extrude(hgt); };\n"

CODE = {
    "socket_head_cap_screw": HEX + """const { d, L, dk, k, s, t } = P;
// z = 0 is the underside of the head; the shank points down (-Z). Thread not modelled.
let head = makeCylinder(dk / 2, k, [0, 0, 0], [0, 0, 1]);
try { head = head.fillet(Math.min(0.08 * d, 1), (e) => e.inPlane('XY', k)); } catch (e) {}
head = head.cut(hex(s, k - t, t + 1));
let shank = makeCylinder(d / 2, L, [0, 0, -L], [0, 0, 1]);
try { shank = shank.chamfer(Math.min(0.1 * d, 1), (e) => e.inPlane('XY', -L)); } catch (e) {}
const body = head.fuse(shank);
return { body };""",
    "hex_bolt": HEX + """const { d, L, k, s } = P;
// z = 0 is the underside of the head; full thread (ISO 4017), thread not modelled.
let head = hex(s, 0, k);
try { head = head.chamfer(Math.min(0.08 * s, 1), (e) => e.inPlane('XY', k)); } catch (e) {}
let shank = makeCylinder(d / 2, L, [0, 0, -L], [0, 0, 1]);
try { shank = shank.chamfer(Math.min(0.1 * d, 1), (e) => e.inPlane('XY', -L)); } catch (e) {}
return { body: head.fuse(shank) };""",
    "hex_nut": HEX + """const { m, s, dh } = P;
// hole = tap drill diameter (thread not modelled)
let body = hex(s, 0, m).cut(makeCylinder(dh / 2, m + 2, [0, 0, -1], [0, 0, 1]));
try { body = body.chamfer(Math.min(0.08 * s, 0.8), (e) => e.inPlane('XY', m)); } catch (e) {}
return { body };""",
    "washer": """const { d1, d2, h } = P;
const body = makeCylinder(d2 / 2, h, [0, 0, 0], [0, 0, 1]).cut(makeCylinder(d1 / 2, h + 2, [0, 0, -1], [0, 0, 1]));
return { body };""",
    "ball_bearing": """const { d, D, B } = P;
// simplified: inner + outer ring with a shield recess on both faces (balls not modelled)
const w = (D - d) / 2, ri = d / 2 + 0.3 * w, ro = D / 2 - 0.3 * w, rec = Math.min(0.3, B * 0.05);
let body = makeCylinder(D / 2, B, [0, 0, 0], [0, 0, 1]).cut(makeCylinder(d / 2, B + 2, [0, 0, -1], [0, 0, 1]));
const recess = (z) => makeCylinder(ro, rec + 0.01, [0, 0, z], [0, 0, 1]).cut(makeCylinder(ri, rec + 2, [0, 0, z - 1], [0, 0, 1]));
body = body.cut(recess(B - rec)).cut(recess(-0.01));
try { body = body.fillet(Math.min(0.3, w * 0.1)); } catch (e) {}
return { body };""",
    "parallel_key": """const { b, h, L } = P;
// DIN 6885 form A: rounded ends (radius b/2), lying along X, z = 0 at the bottom
const body = draw([b / 2, -b / 2]).hLineTo(L - b / 2).threePointsArcTo([L - b / 2, b / 2], [L, 0])
  .hLineTo(b / 2).threePointsArcTo([b / 2, -b / 2], [0, 0]).close().sketchOnPlane('XY').extrude(h);
return { body };""",
    "o_ring": """const { id, w } = P;
// torus: circular cross-section w on inner diameter id
const R = id / 2 + w / 2;
const body = draw([R + w / 2, 0]).threePointsArcTo([R - w / 2, 0], [R, w / 2]).threePointsArcTo([R + w / 2, 0], [R, -w / 2])
  .close().sketchOnPlane('XZ').revolve();
return { body };""",
    "spacer": """const { OD, ID, L } = P;
let body = makeCylinder(OD / 2, L, [0, 0, 0], [0, 0, 1]);
if (ID > 0) body = body.cut(makeCylinder(ID / 2, L + 2, [0, 0, -1], [0, 0, 1]));
try { body = body.chamfer(Math.min(0.3, (OD - ID) / 8)); } catch (e) {}
return { body };""",
}


# ------------------------------------------------------------------ templates
def _sizes(table, lo=3, hi=20):
    return [k for k in table if lo <= float(k[1:]) <= hi]


def _o_rings():
    rows = _load_csv("o-ring_parameters.csv")
    return {r["Size"]: (float(r["iso3601:id"]), float(r["iso3601:w"])) for r in rows
            if r.get("iso3601:id") and r.get("iso3601:w")}


O_RINGS = _o_rings()

TEMPLATES = {
    "socket_head_cap_screw": {"name": "Socket head cap screw", "name_id": "Baut L (socket head)", "standard": "ISO 4762",
                              "category": "fasteners", "words": ("socket", "cap screw", "baut l", "baut l ", "allen", "inbus", "shcs")},
    "hex_bolt": {"name": "Hex head screw", "name_id": "Baut kepala segi enam", "standard": "ISO 4017",
                 "category": "fasteners", "words": ("hex bolt", "hex screw", "hex head", "baut segi enam", "baut hex", "baut")},
    "hex_nut": {"name": "Hex nut", "name_id": "Mur", "standard": "ISO 4032", "category": "fasteners",
                "words": ("nut", "mur")},
    "washer": {"name": "Plain washer", "name_id": "Ring plat", "standard": "ISO 7089", "category": "fasteners",
               "words": ("washer", "ring", "cincin", "scheibe", "podkładka")},
    "ball_bearing": {"name": "Deep groove ball bearing", "name_id": "Bearing", "standard": "ISO 15",
                     "category": "bearings", "words": ("bearing", "laher", "lager", "łożysko")},
    "parallel_key": {"name": "Parallel key (form A)", "name_id": "Pasak", "standard": "DIN 6885",
                     "category": "shaft", "words": ("key", "pasak", "spie", "wpust")},
    "o_ring": {"name": "O-ring", "name_id": "O-ring", "standard": "ISO 3601-1", "category": "seals",
               "words": ("o-ring", "oring", "o ring", "seal ring")},
    "spacer": {"name": "Spacer / sleeve", "name_id": "Spacer / bus", "standard": None, "category": "general",
               "words": ("spacer", "sleeve", "bushing", "bus ", "standoff")},
}


def catalog_index() -> list:
    out = []
    for tid, t in TEMPLATES.items():
        out.append({"id": tid, "name": t["name"], "name_id": t["name_id"], "standard": t["standard"],
                    "category": t["category"], "inputs": inputs_for(tid)})
    return out


def inputs_for(tid: str) -> list:
    if tid == "socket_head_cap_screw":
        return [{"k": "size", "label": "Size", "type": "choice", "options": _sizes(cat.SHCS_ISO4762, 3, 16)},
                {"k": "length", "label": "Length under head (mm)", "type": "choice_by_size",
                 "options": {s: cat.fastener(s)["socket_head"]["lengths"] for s in _sizes(cat.SHCS_ISO4762, 3, 16)}}]
    if tid == "hex_bolt":
        return [{"k": "size", "label": "Size", "type": "choice", "options": list(HEX_ISO4017)},
                {"k": "length", "label": "Length under head (mm)", "type": "choice_by_size",
                 "options": {s: [x for x in ISO4017_LENGTHS if r[2] <= x <= r[3]] for s, r in HEX_ISO4017.items()}}]
    if tid in ("hex_nut", "washer"):
        tab = cat.NUT_ISO4032 if tid == "hex_nut" else cat.WASHER_ISO7089
        return [{"k": "size", "label": "Size", "type": "choice", "options": list(tab)}]
    if tid == "ball_bearing":
        return [{"k": "designation", "label": "Bearing", "type": "choice", "options": list(cat.BEARINGS)}]
    if tid == "parallel_key":
        return [{"k": "shaft_d", "label": "Shaft diameter (mm)", "type": "number", "min": 6.01, "max": 44},
                {"k": "length", "label": "Key length (mm)", "type": "choice", "options": list(KEY_LENGTHS)}]
    if tid == "o_ring":
        return [{"k": "code", "label": "ISO 3601 size code", "type": "choice", "options": list(O_RINGS)[:400]},
                {"k": "id", "label": "or: inner diameter (mm) - nearest size", "type": "number", "min": 0.5, "max": 700},
                {"k": "w", "label": "cross-section (mm), optional", "type": "number", "min": 0.5, "max": 10}]
    return [{"k": "OD", "label": "Outer diameter (mm)", "type": "number", "min": 1, "max": 500},
            {"k": "ID", "label": "Inner diameter (mm, 0 = solid)", "type": "number", "min": 0, "max": 490},
            {"k": "L", "label": "Length (mm)", "type": "number", "min": 0.5, "max": 1000}]


def _p(k, label, value, unit="mm", source="table"):
    v = round(float(value), 4)
    return {"k": k, "label": label, "label_id": label, "unit": unit, "value": v, "min": v, "max": v, "source": source}


def resolve(tid: str, q: dict) -> dict:
    """q = engineer's choices -> {ok, cad | missing | errors}. Returns the `cad` contract (UI v55)."""
    if tid not in TEMPLATES:
        return {"ok": False, "errors": [f"unknown part {tid!r}; see GET /library"]}
    t = TEMPLATES[tid]
    errs, missing, params, expect, holes, extra = [], [], [], [], [], {}
    src = {}
    num = lambda k: (float(str(q[k]).replace(",", ".")) if q.get(k) not in (None, "") else None)
    if tid in ("socket_head_cap_screw", "hex_bolt", "hex_nut", "washer"):
        size = cat.norm_size(q.get("size"))
        if not size:
            missing.append({"k": "size", "question": "Which size (e.g. M6)?"})
        elif tid == "socket_head_cap_screw" and size not in cat.SHCS_ISO4762:
            errs.append(f"{size} is not in the ISO 4762 table")
        elif tid == "hex_bolt" and size not in HEX_ISO4017:
            errs.append(f"{size} is not in the ISO 4017 table")
        elif tid == "hex_nut" and size not in cat.NUT_ISO4032:
            errs.append(f"{size} is not in the ISO 4032 table")
        elif tid == "washer" and size not in cat.WASHER_ISO7089:
            errs.append(f"{size} is not in the ISO 7089 table")
        if size and not errs and tid in ("socket_head_cap_screw", "hex_bolt"):
            f = cat.fastener(size)
            lengths = (f["socket_head"]["lengths"] if tid == "socket_head_cap_screw"
                       else [x for x in ISO4017_LENGTHS if HEX_ISO4017[size][2] <= x <= HEX_ISO4017[size][3]])
            L = num("length")
            if L is None:
                missing.append({"k": "length", "question": f"Length under the head for {size} (standard: "
                                                           f"{', '.join(f'{x:g}' for x in lengths)} mm)?"})
            elif not any(abs(L - x) < 1e-6 for x in lengths):
                nxt = next((x for x in lengths if x >= L), None)
                errs.append(f"{L:g} mm is not a standard length for {size}" + (f"; next standard length is {nxt:g} mm" if nxt else ""))
                extra["suggest_length"] = nxt
            if not missing and not errs:
                params += [_p("d", "Nominal diameter", f["d"]), _p("L", "Length under head", L, source="user")]
                if tid == "socket_head_cap_screw":
                    dk, k, s = cat.SHCS_ISO4762[size]
                    params += [_p("dk", "Head diameter (max)", dk), _p("k", "Head height", k),
                               _p("s", "Hex key size", s), _p("t", "Socket depth (min)", SHCS_SOCKET_T[size])]
                    src["dims"] = SOURCES["iso4762"]
                    expect = [{"label": "Total length", "axis": "z", "value": L + k, "tol": 0.05},
                              {"label": "Head diameter", "axis": "x", "value": dk, "tol": 0.05}]
                else:
                    k, s = HEX_ISO4017[size][:2]
                    params += [_p("k", "Head height (max)", k), _p("s", "Across flats", s)]
                    src["dims"] = SOURCES["iso4017"]
                    expect = [{"label": "Total length", "axis": "z", "value": L + k, "tol": 0.05},
                              {"label": "Across corners", "axis": "x", "value": round(2 * s / math.sqrt(3), 3), "tol": 0.05}]
                extra["mate"] = {"axis": "z", "underside_of_head_z": 0, "d": f["d"]}
                extra["name"] = f"{t['standard']} {size}×{L:g}"
        elif size and not errs and tid == "hex_nut":
            m, s = cat.NUT_ISO4032[size]
            f = cat.fastener(size)
            params += [_p("m", "Nut height", m), _p("s", "Across flats", s), _p("dh", "Bore (tap drill)", f["tap_drill"])]
            src["dims"] = SOURCES["iso4032"]
            expect = [{"label": "Height", "axis": "z", "value": m, "tol": 0.05}]
            extra["name"] = f"ISO 4032 {size}"
        elif size and not errs and tid == "washer":
            f = cat.fastener(size)
            w = f["washer"]
            params += [_p("d1", "Inner diameter", w["d1"]), _p("d2", "Outer diameter", w["d2"]), _p("h", "Thickness", w["h"])]
            src["dims"] = SOURCES["iso7089"]
            expect = [{"label": "Outer diameter", "axis": "x", "value": w["d2"], "tol": 0.05},
                      {"label": "Thickness", "axis": "z", "value": w["h"], "tol": 0.05}]
            extra["name"] = f"ISO 7089 {size}"
    elif tid == "ball_bearing":
        des = str(q.get("designation") or "").strip()
        if not des:
            missing.append({"k": "designation", "question": f"Which bearing ({', '.join(cat.BEARINGS)})?"})
        elif des not in cat.BEARINGS:
            errs.append(f"bearing {des} is not in the library yet ({', '.join(cat.BEARINGS)})")
        else:
            d, D, B, C, C0, bsrc, ok = cat.BEARINGS[des]
            params += [_p("d", "Bore", d), _p("D", "Outer diameter", D), _p("B", "Width", B)]
            src["dims"] = "ISO 15 boundary dimensions"
            src["ratings"] = bsrc or SOURCES["bearing"]
            expect = [{"label": "Outer diameter", "axis": "x", "value": D, "tol": 0.05},
                      {"label": "Width", "axis": "z", "value": B, "tol": 0.05}]
            extra["ratings"] = {"C_kN": C, "C0_kN": C0, "verified": ok}
            extra["name"] = f"Bearing {des} ({d}×{D}×{B})"
    elif tid == "parallel_key":
        sd, L = num("shaft_d"), num("length")
        if sd is None:
            missing.append({"k": "shaft_d", "question": "Shaft diameter (mm)?"})
        if L is None:
            missing.append({"k": "length", "question": f"Key length (standard: {', '.join(map(str, KEY_LENGTHS))} mm)?"})
        if sd is not None:
            row = next((r for r in KEY_DIN6885 if sd <= r[0]), None)
            if sd <= 6 or row is None:
                errs.append("shaft diameter must be over 6 and up to 44 mm for this table")
            elif L is not None:
                _, b, h, t1, t2 = row
                if not any(abs(L - x) < 1e-6 for x in KEY_LENGTHS):
                    errs.append(f"{L:g} mm is not a standard key length")
                elif L < 2 * b:
                    errs.append(f"key length {L:g} mm is shorter than 2·b = {2 * b:g} mm")
                else:
                    params += [_p("b", "Key width", b), _p("h", "Key height", h), _p("L", "Key length", L, source="user")]
                    extra["keyway"] = {"shaft_depth_t1": t1, "hub_depth_t2": t2, "width": b}
                    src["dims"] = SOURCES["din6885"]
                    expect = [{"label": "Length", "axis": "x", "value": L, "tol": 0.05},
                              {"label": "Height", "axis": "z", "value": h, "tol": 0.05}]
                    extra["name"] = f"DIN 6885 A {b}×{h}×{L:g} (shaft Ø{sd:g})"
    elif tid == "o_ring":
        code = str(q.get("code") or "").strip()
        idq = num("id")
        if code and code in O_RINGS:
            iid, w = O_RINGS[code]
        elif idq is not None and O_RINGS:
            wq = num("w")
            pool = {k: v for k, v in O_RINGS.items() if wq is None or abs(v[1] - wq) < 0.06} or O_RINGS
            if wq is not None and len(pool) == len(O_RINGS):
                extra["note"] = f"no ISO 3601 cross-section {wq:g} mm; nearest size used"
            code, (iid, w) = min(pool.items(), key=lambda kv: (abs(kv[1][0] - idq), kv[1][1]))
            if abs(iid - idq) > 0.05 * idq + 0.2:
                extra["note"] = f"nearest standard size to Ø{idq:g} is {code} (ID {iid:g} mm)"
        else:
            missing.append({"k": "id", "question": "O-ring inner diameter (mm) or ISO 3601 size code?"})
            iid = w = None
        if iid:
            params += [_p("id", "Inner diameter", iid), _p("w", "Cross-section", w)]
            src["dims"] = SOURCES["iso3601"]
            expect = [{"label": "Outer diameter", "axis": "x", "value": round(iid + 2 * w, 4), "tol": 0.05},
                      {"label": "Cross-section", "axis": "z", "value": w, "tol": 0.05}]
            extra["name"] = f"O-ring ISO 3601 {code} ({iid:g}×{w:g})"
    else:
        OD, ID, L = num("OD"), num("ID") or 0.0, num("L")
        for k, v in (("OD", OD), ("L", L)):
            if v is None:
                missing.append({"k": k, "question": f"{'Outer diameter' if k == 'OD' else 'Length'} (mm)?"})
        if OD is not None and ID >= OD - 0.5:
            errs.append("inner diameter must be at least 0.5 mm smaller than the outer diameter")
        if not missing and not errs:
            params += [_p("OD", "Outer diameter", OD, source="user"), _p("ID", "Inner diameter", ID, source="user"),
                       _p("L", "Length", L, source="user")]
            expect = [{"label": "Outer diameter", "axis": "x", "value": OD, "tol": 0.05},
                      {"label": "Length", "axis": "z", "value": L, "tol": 0.05}]
            extra["name"] = f"Spacer Ø{OD:g}/Ø{ID:g}×{L:g}"
            src["dims"] = "engineer's input"
    if missing or errs:
        return {"ok": False, "missing": missing, "errors": errs, **({"suggest": extra} if extra else {})}
    cad = {"title": extra.pop("name"), "kind": "interface", "process": None,
           "material": "steel" if tid not in ("o_ring",) else "nylon", "params": params, "code": CODE[tid],
           "expect": expect, "missing": [], "variants": [], "holes": holes, "read_by": "IDRIS library (standard table)",
           "library": {"id": tid, "standard": t["standard"], "sources": src, **extra},
           "note": "Threads are not modelled." if tid in ("socket_head_cap_screw", "hex_bolt", "hex_nut") else
                   ("Simplified shape (balls and seals not modelled)." if tid == "ball_bearing" else "")}
    if tid == "o_ring":
        cad["note"] = "Material shown as nylon only for the mass estimate; real O-rings are elastomers (NBR, FKM)."
    return {"ok": True, "cad": cad}


# ------------------------------------------------------------------ sentence -> library part (no AI)
def match_sentence(msg: str):
    """'baut M6 panjang 20', 'socket head M5x16', 'mur M8', 'bearing 6205', 'pasak poros 20 panjang 32',
    'o-ring id 20' -> (template id, query) or None."""
    m = " " + (msg or "").lower().replace("×", "x") + " "
    # a part designed AROUND a standard part ("bushing for bearing 608", "holder for M6 nut") is not a library part
    if re.search(r"(?<![a-z])(holder|dudukan|bracket|braket|housing|cover|tutup|adapter|adaptor|mount|bushing|standoff|"
                 r"klem|clamp|enclosure|kotak|untuk|for|für|dla)(?![a-z])", m):
        return None
    tid = None
    hit = lambda w: re.search(re.escape(w) if len(w) >= 6 else rf"(?<![a-z]){re.escape(w.strip())}(?![a-z])", m)
    for cand in ("socket_head_cap_screw", "hex_nut", "o_ring", "washer", "ball_bearing", "parallel_key", "hex_bolt"):
        if any(hit(w) for w in TEMPLATES[cand]["words"]):
            tid = cand
            break
    if tid == "parallel_key" and not any(hit(w) for w in TEMPLATES[tid]["words"] if w != "key"):
        # "key" alone is also a door key, a hex key, a keychain: a machine key needs a shaft-side word next to it
        if re.search(r"door|pintu|house|rumah|car |mobil|motor|hex key|allen|keychain|key chain|gantungan|keyboard|keycap|"
                     r"key holder|key ring|keyring|piano", m):
            tid = None
        elif not re.search(r"parallel|feather|shaft|keyway|machine key|din\s?6885|poros|welle|hub|b\s?=|\d+\s?x\s?\d+", m):
            tid = None
    if not tid:
        return None
    q = {}
    sz = re.search(r"\bm\s?(\d+(?:[.,]\d+)?)(?:\s*x\s*(\d+(?:[.,]\d+)?))?", m)
    if sz:
        q["size"] = "M" + sz.group(1).replace(",", ".")
        if sz.group(2):
            q["length"] = sz.group(2).replace(",", ".")
    ln = re.search(r"(?:panjang|length|long|lang|l\s?=?)\s*(\d+(?:[.,]\d+)?)", m)
    if ln and "length" not in q:
        q["length"] = ln.group(1).replace(",", ".")
    if tid == "ball_bearing":
        des = re.search(r"\b(6\d{2,3}|6\d/\d|16\d{3}|62\d{2}|63\d{2})\b", m)
        if des:
            q["designation"] = des.group(1)
    if tid == "parallel_key":
        sd = re.search(r"(?:poros|shaft|welle|wał)\D{0,8}(\d+(?:[.,]\d+)?)", m)
        if sd:
            q["shaft_d"] = sd.group(1).replace(",", ".")
    if tid == "o_ring":
        idm = re.search(r"(?:id|inner|dalam|diameter)\D{0,4}(\d+(?:[.,]\d+)?)", m)
        if idm:
            q["id"] = idm.group(1).replace(",", ".")
        xs = re.search(r"(\d+(?:[.,]\d+)?)\s*x\s*(\d+(?:[.,]\d+)?)", m)
        if xs:
            q.setdefault("id", xs.group(1).replace(",", "."))
            q["w"] = xs.group(2).replace(",", ".")
        cs = re.search(r"(?:cs|cross|penampang|tebal)\D{0,4}(\d+(?:[.,]\d+)?)", m)
        if cs:
            q["w"] = cs.group(1).replace(",", ".")
    return tid, q


def chat_answer(tid: str, r: dict) -> str:
    t = TEMPLATES[tid]
    if r.get("ok"):
        c = r["cad"]
        lib = c["library"]
        lines = [f"{c['title']} from the IDRIS library ({t['standard'] or 'custom'}). Dimensions come from the table, "
                 "the browser builds the model.",
                 "Sizes: " + ", ".join(f"{p['label']} {p['value']:g} {p['unit']}".strip() for p in c["params"])]
        if lib.get("keyway"):
            k = lib["keyway"]
            lines.append(f"Keyway: width {k['width']:g} mm, shaft depth t1 {k['shaft_depth_t1']:g} mm, "
                         f"hub depth t2 {k['hub_depth_t2']:g} mm.")
        if lib.get("ratings"):
            rt = lib["ratings"]
            lines.append(f"Load ratings: C {rt['C_kN']:g} kN" + (f", C0 {rt['C0_kN']:g} kN" if rt['C0_kN'] else "")
                         + ("" if rt["verified"] else " (not yet checked against the manufacturer catalogue)") + ".")
        if lib.get("note"):
            lines.append(lib["note"] + ".")
        if c.get("note"):
            lines.append(c["note"])
        lines.append("Source: " + "; ".join(lib["sources"].values()))
        return "\n".join(lines)
    parts = []
    if r.get("missing"):
        parts.append(f"For the {t['name']} IDRIS needs:\n" + "\n".join("- " + m["question"] for m in r["missing"]))
    if r.get("errors"):
        parts.append("Not possible: " + "; ".join(r["errors"]))
    return "\n".join(parts)


def follow_up(request: str, answer: str, pending_missing=None):
    """The engineer answers a library question ('16', 'M6', '6205'). -> (tid, query) or None."""
    hit = match_sentence(request)
    if not hit:
        return None
    tid, q = hit
    extra = match_sentence((request or "") + " " + (answer or ""))
    if extra and extra[0] == tid:
        q.update({k: v for k, v in extra[1].items() if k not in q})
    keys = [m.get("k") for m in (pending_missing or []) if isinstance(m, dict)] or [k for k in ("size", "length") if k not in q]
    a = (answer or "").lower()
    sz = re.search(r"\bm\s?(\d+(?:[.,]\d+)?)", a)
    if sz and "size" in keys and "size" not in q:
        q["size"] = "M" + sz.group(1).replace(",", ".")
        a = a.replace(sz.group(0), " ")
    nums = [x.replace(",", ".") for x in re.findall(r"(\d+(?:[.,]\d+)?)", a)]
    for k in keys:
        if k in q or not nums:
            continue
        if k == "size":
            continue
        q[k] = nums.pop(0)
    return tid, q
