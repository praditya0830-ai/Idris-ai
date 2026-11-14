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
    "plow_blade": """const { W, H, c, t, dr, tr, ce_h, ce_t } = P;
// Welded snow plow blade, side profile drawn in XZ (x = travel direction, z = up), width along Y.
// Moldboard = rolled sheet (circular arc, concave face forward); ribs = plates on the back; cutting edge = flat bar.
const R = (H * H / 4 + c * c) / (2 * c);            // mid-surface radius from chord H and sagitta c
const a = Math.asin(H / (2 * R));                   // half angle of the arc
const xc = R - c, zc = H / 2;                       // arc centre
const pt = (r, f) => [xc - r * Math.cos(f), zc + r * Math.sin(f)];
const sector = (r1, r2, y, len) => draw(pt(r1, -a)).threePointsArcTo(pt(r1, a), pt(r1, 0)).lineTo(pt(r2, a))
  .threePointsArcTo(pt(r2, -a), pt(r2, 0)).close().sketchOnPlane('XZ', y).extrude(len);
const Ri = R - t / 2, Ro = R + t / 2;
let body = sector(Ri, Ro, 0, W);                    // moldboard
const n = Math.max(2, Math.round(W / 600) + 1);     // ribs about every 600 mm
for (let i = 0; i < n; i++) body = body.fuse(sector(R, Ro + dr, i * (W - tr) / (n - 1), tr));
const b = pt(Ri, -a), drop = ce_h / 3;              // cutting edge: on the front of the bottom edge, 1/3 below it
const edge = draw([b[0] - t / 2, -drop]).lineTo([b[0] - t / 2 + ce_t, -drop]).lineTo([b[0] - t / 2 + ce_t, ce_h - drop])
  .lineTo([b[0] - t / 2, ce_h - drop]).close().sketchOnPlane('XZ', 0).extrude(W);
body = body.fuse(edge);
return { body };""",
    "spur_gear": """const { m, z, alpha, b, bore, kw, kt, hub_d, hub_l } = P;
// Involute spur gear, ISO 53 basic rack (addendum 1 m, dedendum 1.25 m). Flanks are true involutes drawn as
// short straight segments. Gear axis = Z, tooth 0 centred on +X, gear face on z = 0, hub on top.
const a = alpha * Math.PI / 180;
const rp = m * z / 2, rb = rp * Math.cos(a), ra = rp + m, rf = rp - 1.25 * m;
const inv = (x) => Math.tan(x) - x;
const half = Math.PI / (2 * z) + inv(a);
const r0 = Math.max(rb, rf), n = 6;
const flank = [];
if (rf < rb) flank.push([rf, half]);
for (let i = 0; i <= n; i++) {
  const r = r0 + (ra - r0) * i / n;
  flank.push([r, half - inv(Math.acos(Math.min(1, rb / r)))]);
}
const back = flank.slice().reverse();
const pts = [];
for (let k = 0; k < z; k++) {
  const c = 2 * Math.PI * k / z;
  for (const f of flank) pts.push([f[0] * Math.cos(c - f[1]), f[0] * Math.sin(c - f[1])]);
  for (const f of back) pts.push([f[0] * Math.cos(c + f[1]), f[0] * Math.sin(c + f[1])]);
}
let pen = draw(pts[0]);
for (let i = 1; i < pts.length; i++) pen = pen.lineTo(pts[i]);
let body = pen.close().sketchOnPlane('XY').extrude(b);
if (hub_l > 0) body = body.fuse(makeCylinder(hub_d / 2, hub_l, [0, 0, b]));
if (bore > 0) body = body.cut(makeCylinder(bore / 2, b + hub_l + 2, [0, 0, -1]));
if (kw > 0) body = body.cut(makeBox([-kw / 2, 0, -1], [kw / 2, bore / 2 + kt, b + hub_l + 1]));
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
    "plow_blade": {"name": "Snow plow blade (welded)", "name_id": "Bilah bajak salju (las)", "standard": None,
                   "category": "weldments", "words": ("snow plow", "snowplow", "snow plough", "plow blade", "plough blade",
                                                     "bajak salju", "schneepflug", "pług")},
    "spur_gear": {"name": "Spur gear (involute)", "name_id": "Roda gigi lurus (involut)", "standard": "ISO 53 / ISO 54",
                  "category": "gears", "words": ("spur gear", "involute gear", "roda gigi lurus", "stirnrad")},
    "o_ring": {"name": "O-ring", "name_id": "O-ring", "standard": "ISO 3601-1", "category": "seals",
               "words": ("o-ring", "oring", "o ring", "seal ring")},
    "spacer": {"name": "Spacer / sleeve", "name_id": "Spacer / bus", "standard": None, "category": "general",
               "words": ("spacer", "sleeve", "bushing", "bus ", "standoff")},
}


# Involute spur gear. ISO 54 gives the preferred modules; ISO 53 the basic rack (alpha 20 deg, ha = m, hf = 1.25 m).
ISO54_MODULES = (0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
GEAR_INPUTS = {   # k: (label, min, max, default)
    "m": ("Module (mm)", 0.5, 10, None),
    "z": ("Number of teeth", 9, 150, None),
    "alpha": ("Pressure angle (deg)", 14.5, 25, 20),
    "b": ("Face width (mm)", 1, 200, None),
    "bore": ("Bore diameter (mm), 0 = solid", 0, 150, 0),
    "hub_d": ("Hub diameter (mm)", 0, 300, 0),
    "hub_l": ("Hub length beyond the face (mm), 0 = no hub", 0, 200, 0),
}


def gear_geometry(m: float, z: int, alpha_deg: float) -> dict:
    """Exact involute gear geometry (no profile shift)."""
    a = math.radians(alpha_deg)
    d, db = m * z, m * z * math.cos(a)
    da, df = d + 2 * m, d - 2.5 * m
    inv = lambda x: math.tan(x) - x
    half = math.pi / (2 * z) + inv(a)
    tip_half = half - inv(math.acos(db / da))
    return {"pitch_d": d, "base_d": db, "tip_d": da, "root_d": df, "circular_pitch": math.pi * m,
            "tooth_thickness_pitch": math.pi * m / 2, "tip_thickness": da * tip_half,
            "z_min_no_undercut": 2 / math.sin(a) ** 2, "tip_half_angle": tip_half}


def _resolve_gear(q: dict) -> dict:
    errs, vals, given, missing = [], {}, set(), []
    for k, (lab, lo, hi, dv) in GEAR_INPUTS.items():
        raw = q.get(k)
        if raw in (None, ""):
            vals[k] = dv
            if dv is None:
                missing.append({"k": k, "question": f"{lab.split(' (')[0]}?" + (
                    " Preferred (ISO 54): " + ", ".join(f"{x:g}" for x in ISO54_MODULES) if k == "m" else "")})
            continue
        try:
            x = float(str(raw).replace(",", "."))
        except ValueError:
            errs.append(f"{lab}: {raw!r} is not a number")
            continue
        if not lo <= x <= hi:
            errs.append(f"{lab.split(' (')[0]} must be between {lo:g} and {hi:g} (got {x:g})")
        vals[k] = x
        given.add(k)
    if errs:
        return {"ok": False, "missing": [], "errors": errs}
    if missing:
        return {"ok": False, "errors": [], "missing": missing}
    if vals["z"] != int(vals["z"]):
        return {"ok": False, "missing": [], "errors": ["the number of teeth must be a whole number"]}
    m, z, al, b, bore, hub_d, hub_l = (vals[k] for k in ("m", "z", "alpha", "b", "bore", "hub_d", "hub_l"))
    z = int(z)
    g = gear_geometry(m, z, al)
    kw = kt = 0.0
    key_txt = None
    want_key = str(q.get("keyway") or "").lower() in ("1", "true", "yes", "on") or q.get("keyway") is True
    if want_key:
        if bore <= 0:
            errs.append("a keyway needs a bore")
        else:
            row = next((r for r in KEY_DIN6885 if bore <= r[0]), None)
            if not row or bore <= 6:
                errs.append(f"DIN 6885 has no key for a bore of {bore:g} mm (table covers over 6 up to {KEY_DIN6885[-1][0]:g} mm)")
            else:
                kw, kt = float(row[1]), float(row[4])
                key_txt = f"keyway {kw:g} wide, {kt:g} deep in the hub (DIN 6885-1 for a shaft of {bore:g} mm)"
                asked = q.get("key_width")
                if asked not in (None, "") and abs(float(asked) - kw) > 1e-6:
                    errs.append(f"DIN 6885 gives a {kw:g} mm key for a {bore:g} mm shaft, not {float(asked):g} mm")
    if g["tip_thickness"] <= 0.2 * m:
        errs.append(f"with {z} teeth at {al:g} deg the tooth tip is pointed ({g['tip_thickness']:.2f} mm): use more teeth")
    if bore > 0 and bore / 2 + kt + m > g["root_d"] / 2:
        errs.append(f"bore {bore:g} mm{' with its keyway' if kt else ''} leaves less than one module of rim under the "
                    f"teeth (root diameter {g['root_d']:.2f} mm)")
    if hub_l > 0 and hub_d <= bore + 2 * kt + 2:
        errs.append(f"hub diameter {hub_d:g} mm is too small for the bore{' and keyway' if kt else ''}: needs more than "
                    f"{bore + 2 * kt + 2:g} mm")
    if hub_l > 0 and hub_d <= 0:
        errs.append("a hub length needs a hub diameter")
    if errs:
        return {"ok": False, "missing": [], "errors": errs}
    notes = []
    if abs(min(ISO54_MODULES, key=lambda x: abs(x - m)) - m) > 1e-9:
        notes.append(f"module {m:g} is not an ISO 54 preferred module ({', '.join(f'{x:g}' for x in ISO54_MODULES)}): "
                     "cutters may be hard to find")
    if z < g["z_min_no_undercut"]:
        notes.append(f"{z} teeth is below {g['z_min_no_undercut']:.0f}, the limit for undercut at {al:g} deg without "
                     "profile shift: a cut gear would have undercut roots, which this model does not show")
    full = {"m": m, "z": z, "alpha": al, "b": b, "bore": bore, "kw": kw, "kt": kt, "hub_d": hub_d if hub_l > 0 else 0.0,
            "hub_l": hub_l}
    lab = {"m": "Module", "z": "Teeth", "alpha": "Pressure angle", "b": "Face width", "bore": "Bore", "kw": "Keyway width",
           "kt": "Keyway depth in hub", "hub_d": "Hub diameter", "hub_l": "Hub length"}
    params = [{"k": k, "label": lab[k], "label_id": lab[k], "unit": "deg" if k == "alpha" else ("" if k == "z" else "mm"),
               "value": round(float(v), 4), "min": round(float(v), 4), "max": round(float(v), 4),
               "source": "table" if k in ("kw", "kt") and kw else ("user" if k in given else "default")}
              for k, v in full.items()]
    expect = [{"label": "Total length (face + hub)", "axis": "z", "value": round(b + hub_l, 3), "tol": 0.05}]
    if z % 2 == 0:
        expect.append({"label": "Tip diameter", "axis": "x", "value": round(g["tip_d"] * math.cos(g["tip_half_angle"]), 3), "tol": 0.05})
    cad = {"title": f"Spur gear m{m:g} z{z}", "kind": "functional", "process": "cnc", "material": "steel",
           "params": params, "code": CODE["spur_gear"], "expect": expect, "missing": [], "variants": [], "holes": [],
           "read_by": "IDRIS library (ISO 53 involute)",
           "library": {"id": "spur_gear", "standard": "ISO 53:1998 basic rack, ISO 54:1996 modules" + (", DIN 6885-1 keyway" if kw else ""),
                       "sources": {"geometry": "d = m z, da = d + 2 m, df = d - 2.5 m, db = d cos(alpha); involute flanks",
                                   **({"keyway": SOURCES["din6885"]} if kw else {})},
                       "pitch_d": round(g["pitch_d"], 4), "tip_d": round(g["tip_d"], 4), "root_d": round(g["root_d"], 4),
                       "base_d": round(g["base_d"], 4), "circular_pitch": round(g["circular_pitch"], 4),
                       "tooth_thickness_pitch": round(g["tooth_thickness_pitch"], 4),
                       "tip_thickness": round(g["tip_thickness"], 3), "keyway": key_txt, "notes": notes},
           "note": ("Geometry only: no profile shift, no tip relief, no root fillet (the root is a sharp corner here; a "
                    "cut gear has a fillet). Tooth strength is NOT checked for a single gear: IDRIS rates a gear PAIR "
                    "to ISO 6336 in the gearbox workflow. Flanks are involutes drawn with 6 straight segments each.")}
    return {"ok": True, "cad": cad}


# Welded snow plow blade: one parametric family. W (working width) is the only input IDRIS must be given.
PLOW_INPUTS = {   # k: (label, min, max, default)
    "W": ("Working width (mm)", 1000, 4500, None),
    "H": ("Blade height, chord (mm)", 400, 1500, 900),
    "c": ("Curl depth of the moldboard (mm)", 40, 400, None),      # default 0.18 H
    "t": ("Moldboard sheet thickness (mm)", 3, 12, 6),
    "dr": ("Rib depth (mm)", 40, 300, 120),
    "tr": ("Rib plate thickness (mm)", 4, 20, 8),
    "ce_h": ("Cutting edge height (mm)", 80, 250, 150),
    "ce_t": ("Cutting edge thickness (mm)", 8, 30, 16),
}
STEEL_RHO = 7850.0      # kg/m3, structural steel (EN 10025 S235/S355)


def plow_geometry(v: dict) -> dict:
    """Exact geometry of the plow blade family from its 8 dimensions (mm). No strength numbers."""
    W, H, c, t, dr, tr, ce_h, ce_t = (v[k] for k in ("W", "H", "c", "t", "dr", "tr", "ce_h", "ce_t"))
    R = (H * H / 4 + c * c) / (2 * c)
    a = math.asin(H / (2 * R))
    Ri, Ro = R - t / 2, R + t / 2
    n = max(2, round(W / 600) + 1)
    xb = (R - c) - Ri * math.cos(a)                       # front face at the bottom end
    drop = ce_h / 3
    x_min, x_max = -(c + t / 2 + dr), xb - t / 2 + ce_t
    z_min, z_max = -drop, H / 2 + (Ro + dr) * math.sin(a)
    dev = 2 * a * R                                       # developed length of the sheet (neutral axis at mid-thickness)
    v_sheet = 2 * a * R * t * W                           # = a (Ro^2 - Ri^2) W
    v_rib = a * ((Ro + dr) ** 2 - Ro ** 2) * tr           # part of one rib outside the sheet
    v_edge = ce_h * ce_t * W
    mass = STEEL_RHO * (v_sheet + n * v_rib + v_edge) * 1e-9
    return {"R": R, "half_angle_deg": math.degrees(a), "ribs": n, "rib_pitch": (W - tr) / (n - 1),
            "bbox": {"x": x_max - x_min, "y": W, "z": z_max - z_min}, "developed_length": dev,
            "mass_kg": mass,
            "cut_list": [
                {"item": "Moldboard sheet", "qty": 1, "size": f"{W:g} x {dev:.0f} x {t:g} mm (flat blank, rolled to R {R:.0f})",
                 "mass_kg": round(STEEL_RHO * v_sheet * 1e-9, 2)},
                {"item": "Rib plate", "qty": n, "size": f"arc segment {dr:g} deep x {tr:g} mm, inner R {Ro:.0f}",
                 "mass_kg": round(STEEL_RHO * v_rib * 1e-9, 2)},
                {"item": "Cutting edge flat bar", "qty": 1, "size": f"{W:g} x {ce_h:g} x {ce_t:g} mm",
                 "mass_kg": round(STEEL_RHO * v_edge * 1e-9, 2)}]}


def _resolve_plow(q: dict) -> dict:
    errs, vals, given = [], {}, set()
    for k, (lab, lo, hi, dv) in PLOW_INPUTS.items():
        raw = q.get(k)
        if raw in (None, ""):
            vals[k] = dv
            continue
        try:
            x = float(str(raw).replace(",", "."))
        except ValueError:
            errs.append(f"{lab}: {raw!r} is not a number")
            continue
        if not lo <= x <= hi:
            errs.append(f"{lab} must be between {lo:g} and {hi:g} (got {x:g})")
        vals[k] = x
        given.add(k)
    if errs:
        return {"ok": False, "missing": [], "errors": errs}
    if vals["W"] is None:
        return {"ok": False, "errors": [],
                "missing": [{"k": "W", "question": "Working width of the blade in mm (1000 to 4500)?"}]}
    if vals["c"] is None:
        vals["c"] = round(0.18 * vals["H"], 1)
    if vals["c"] > 0.45 * vals["H"]:
        return {"ok": False, "missing": [], "errors": [f"curl depth {vals['c']:g} mm is more than 45 % of the height "
                                                       f"{vals['H']:g} mm: the sheet would be bent past a half circle"]}
    g = plow_geometry(vals)
    params = [{"k": k, "label": PLOW_INPUTS[k][0].split(" (")[0], "label_id": PLOW_INPUTS[k][0].split(" (")[0],
               "unit": "mm", "value": round(vals[k], 4), "min": PLOW_INPUTS[k][1], "max": PLOW_INPUTS[k][2],
               "source": "user" if k in given else "default"} for k in PLOW_INPUTS]
    b = g["bbox"]
    cad = {"title": f"Snow plow blade {vals['W']:g} mm", "kind": "functional", "process": "weld", "material": "steel",
           "params": params, "code": CODE["plow_blade"],
           "expect": [{"label": "Working width", "axis": "y", "value": round(b["y"], 3), "tol": 0.5, "from_param": "W"},
                      {"label": "Overall height", "axis": "z", "value": round(b["z"], 3), "tol": 1.0},
                      {"label": "Overall depth", "axis": "x", "value": round(b["x"], 3), "tol": 1.0}],
           "missing": [], "variants": [], "holes": [], "read_by": "IDRIS library (parametric weldment)",
           "library": {"id": "plow_blade", "standard": None, "parametric": True,
                       "sources": {"dims": "engineer's input and IDRIS defaults (shown per parameter)",
                                   "geometry": "circular arc from chord H and sagitta c: R = (H²/4 + c²) / (2c)",
                                   "mass": f"steel {STEEL_RHO:g} kg/m³ x volume of sheet, ribs and cutting edge"},
                       "ribs": g["ribs"], "rib_pitch": round(g["rib_pitch"], 1), "radius": round(g["R"], 1),
                       "developed_length": round(g["developed_length"], 1), "mass_kg": round(g["mass_kg"], 1),
                       "cut_list": g["cut_list"]},
           "note": ("Strength is NOT checked yet: run the weldment check (shell FEA) with your own loads, a pressure "
                    "on the blade and/or a force on the cutting edge. IDRIS does not guess snow loads. Mounting frame, "
                    "trip springs and weld sizes are not modelled.")}
    cad["library"]["check"] = {"endpoint": "POST /weldment-check", "loads": ["pressure_kPa", "edge_force_N", "edge_zone_mm"],
                               "options": ["steel (S235 | S355)", "mount_ribs", "mount_z_mm", "design_factor",
                                           "max_deflection_mm", "element_size_mm", "convergence_study"]}
    return {"ok": True, "cad": cad}


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
    if tid == "spur_gear":
        return [{"k": k, "label": lab, "type": "number", "min": lo, "max": hi, **({"default": dv} if dv is not None else {})}
                for k, (lab, lo, hi, dv) in GEAR_INPUTS.items()] + [
            {"k": "keyway", "label": "Keyway to DIN 6885 (needs a bore)", "type": "boolean", "default": False}]
    if tid == "plow_blade":
        return [{"k": k, "label": lab, "type": "number", "min": lo, "max": hi, **({"default": dv} if dv is not None else {})}
                for k, (lab, lo, hi, dv) in PLOW_INPUTS.items()]
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
    if tid == "plow_blade":
        return _resolve_plow(q)
    if tid == "spur_gear":
        return _resolve_gear(q)
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
    if any(w in m for w in TEMPLATES["plow_blade"]["words"]):
        return "plow_blade", _plow_query(m)
    if (any(w in m for w in TEMPLATES["spur_gear"]["words"]) or (re.search(r"\bgear\b|roda gigi", m) and re.search(
            r"\bmodul", m) and re.search(r"teeth|tooth|gigi\b|z\s*=", m))) and not re.search(
            r"gearbox|reducer|reduksi|ratio|rasio|stage|planet|worm|bevel|helical|rack\b|pump", m):
        return "spur_gear", _gear_query(m)
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


_LEN = r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?(?![a-z])"


def _mm(num: str, unit: str | None, bare_small_is_m: bool = False) -> float:
    x = float(num.replace(",", "."))
    if unit == "m" or (unit is None and bare_small_is_m and x < 10):
        return x * 1000
    return x * 10 if unit == "cm" else x


def _plow_query(m: str) -> dict:
    """'snow plow blade 2.4 m wide, 900 high, sheet 6 mm' -> {W, H, t}. Only what the sentence says."""
    q = {}
    keys = (("W", r"(?:working width|width|wide|lebar|breite|szeroko\w*)", True),
            ("H", r"(?:height|high|tall|tinggi|höhe|wysoko\w*)", True),
            ("t", r"(?:thickness|thick|tebal|sheet|plat|blech)", False))
    for k, words, small_m in keys:
        hit = (re.search(words + r"\D{0,12}?" + _LEN, m) or re.search(_LEN + r"\s*(?:of\s+)?" + words, m))
        if hit:
            q[k] = f"{_mm(hit.group(1), hit.group(2), small_m):g}"
    if "W" not in q:                       # "snow plow blade 3000 mm": the one large length is the width
        big = [_mm(a, u, True) for a, u in re.findall(_LEN, m)]
        big = [x for x in big if x >= 1000 and f"{x:g}" not in q.values()]
        if len(big) == 1:
            q["W"] = f"{big[0]:g}"
    return q


def _gear_query(m: str) -> dict:
    """'spur gear, module 2, 32 teeth, 20 degree pressure angle, face width 15 mm, bore 12 mm with a 4 mm keyway and
    a hub 30 mm diameter, 10 mm long' -> the query for _resolve_gear. Only what the sentence says."""
    n = r"(\d+(?:[.,]\d+)?)"
    q = {}

    def grab(k, *pats, text=m):
        for pat in pats:
            hit = re.search(pat, text)
            if hit:
                q[k] = next(g for g in hit.groups() if g).replace(",", ".")
                return True
        return False
    grab("m", r"modul\w*\s*(?:of\s*|=\s*)?" + n, r"\bm\s*=?\s*" + n + r"\b(?!\s*m)")
    grab("z", r"(\d+)\s*(?:teeth|tooth|gigi\b|zähne)", r"\bz\s*=?\s*(\d+)")
    grab("alpha", n + r"\s*(?:°|deg\w*|derajat)", r"pressure angle\s*(?:of\s*)?" + n)
    grab("b", r"(?:face ?width|lebar gigi|lebar muka|zahnbreite|thickness|tebal)\s*(?:of\s*|=\s*)?" + n,
         n + r"\s*mm\s*(?:face ?width|thick|wide)")
    grab("bore", r"(?:bore|lubang poros|lubang|bohrung)\s*(?:of\s*|ø\s*|diameter\s*|=\s*)?" + n, n + r"\s*mm\s*bore")
    hub = re.search(r"\bhub\b|naf|nabe", m)
    if hub:
        tail = m[hub.start():]
        grab("hub_d", n + r"\s*mm\s*(?:diameter|dia\b|ø)", r"(?:diameter|dia|ø)\s*" + n, text=tail)
        grab("hub_l", n + r"\s*mm\s*(?:long|length|panjang)", r"(?:long|length|panjang)\s*" + n, text=tail)
    if re.search(r"key ?way|keyseat|alur pasak|passfedernut", m):
        q["keyway"] = "true"
        kw = re.search(n + r"\s*mm\s*(?:key ?way|keyseat)|(?:key ?way|alur pasak)\s*(?:of\s*)?" + n, m)
        if kw:
            q["key_width"] = next(g for g in kw.groups() if g).replace(",", ".")
    return q


def chat_answer(tid: str, r: dict) -> str:
    t = TEMPLATES[tid]
    if tid == "spur_gear" and r.get("ok"):
        c = r["cad"]
        lib = c["library"]
        dflt = [p for p in c["params"] if p["source"] == "default" and p["value"]]
        lines = [f"{c['title']} from the IDRIS library: involute geometry to ISO 53, built by the browser's CAD kernel.",
                 "From you: " + ", ".join(f"{p['label']} {p['value']:g}{(' ' + p['unit']) if p['unit'] else ''}"
                                          for p in c["params"] if p["source"] == "user"),
                 ("IDRIS defaults: " + ", ".join(f"{p['label']} {p['value']:g} {p['unit']}" for p in dflt)) if dflt else "",
                 f"Gear data: pitch diameter {lib['pitch_d']:g} mm, tip {lib['tip_d']:g}, root {lib['root_d']:g}, base "
                 f"{lib['base_d']:g}; tooth thickness at the pitch circle {lib['tooth_thickness_pitch']:g} mm.",
                 ("Standard table: " + lib["keyway"] + ".") if lib["keyway"] else "",
                 ("Note: " + "; ".join(lib["notes"]) + ".") if lib["notes"] else "", c["note"]]
        return "\n".join(x for x in lines if x)
    if tid == "plow_blade" and r.get("ok"):
        c = r["cad"]
        lib = c["library"]
        dflt = [p for p in c["params"] if p["source"] == "default"]
        lines = [f"{c['title']} from the IDRIS library (parametric welded part). The browser builds the model; "
                 "change any size, for example the working width, and it rebuilds.",
                 "From you: " + (", ".join(f"{p['label']} {p['value']:g} mm" for p in c["params"] if p["source"] == "user")),
                 ("IDRIS defaults (change them freely): " + ", ".join(f"{p['label']} {p['value']:g} mm" for p in dflt))
                 if dflt else "",
                 f"Result: {lib['ribs']} ribs every {lib['rib_pitch']:g} mm, moldboard radius {lib['radius']:g} mm, "
                 f"mass about {lib['mass_kg']:g} kg (steel).",
                 "Cut list: " + "; ".join(f"{x['qty']} x {x['item']} {x['size']}" for x in lib["cut_list"]) + ".",
                 c["note"]]
        return "\n".join(x for x in lines if x)
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
