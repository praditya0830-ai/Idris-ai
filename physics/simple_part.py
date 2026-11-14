"""Simple parts from a sentence: box/cube (with through-holes), cylinder, tube, sphere.

Pipeline (same principle as the gearbox): the AI only turns the sentence into a structured spec;
this module validates the spec, builds the geometry and computes volume and mass. No number in the
result comes from the AI except the dimensions the engineer asked for.

Outputs: STL (preview / 3D print), STEP (exact B-rep through OpenCascade when the server has it,
otherwise a faceted STEP), volume, mass, bounding box, and a self-check of mesh volume vs exact volume.
"""
from __future__ import annotations

import math
import os
import re
import struct

import numpy as np

from assembly3d import build_box, build_plate_with_holes, build_ring, build_shaft, mesh_volume_mm3, _merge

SHAPES = ("box", "cylinder", "sphere")
MATERIALS = {   # density kg/m^3 (typical handbook values)
    "aluminium": 2700.0, "steel": 7850.0, "stainless": 8000.0, "titanium": 4430.0, "brass": 8500.0,
    "pla": 1240.0, "abs": 1050.0, "nylon": 1140.0,
}
MAT_ALIASES = {"aluminum": "aluminium", "alu": "aluminium", "al": "aluminium", "baja": "steel", "besi": "steel",
               "stainless steel": "stainless", "sus": "stainless", "titanium": "titanium", "titan": "titanium",
               "kuningan": "brass", "plastik": "pla", "plastic": "pla"}
LIM = (0.5, 2000.0)     # mm, any dimension

SPEC_HELP = (
    'One JSON object. shape = "box" | "cylinder" | "sphere". Units mm.\n'
    'box: {"shape":"box","length":L,"width":W,"height":H,"holes":[{"diameter":d,"x":cx,"y":cy}]} '
    '(a cube has L = W = H; holes go straight through the height, x/y measured from the corner, optional)\n'
    'cylinder: {"shape":"cylinder","diameter":D,"height":H,"bore":d (optional, makes a tube)}\n'
    'sphere: {"shape":"sphere","diameter":D}\n'
    'Optional for all: "material": aluminium|steel|stainless|titanium|brass|pla|abs|nylon, "name": short label.\n'
    'Features not in this list (fillets, chamfers, threads, pockets) -> put them in "unsupported": ["..."]; '
    'never invent dimensions the user did not give: if a size is missing put the field name in "missing": [...].'
)


# ------------------------------------------------------------------ validation
def validate(spec: dict) -> tuple[dict, list, list]:
    """-> (clean spec, errors, notes)."""
    errs, notes = [], []
    if not isinstance(spec, dict):
        return {}, ["spec must be an object"], notes
    shape = str(spec.get("shape", "")).lower().strip()
    shape = {"cube": "box", "kubus": "box", "block": "box", "balok": "box", "plate": "box", "plat": "box",
             "tube": "cylinder", "tabung": "cylinder", "silinder": "cylinder", "pipe": "cylinder",
             "ball": "sphere", "bola": "sphere"}.get(shape, shape)
    if shape not in SHAPES:
        return {}, [f"shape '{spec.get('shape')}' is not supported yet (supported: box/cube, cylinder/tube, sphere)"], notes
    out = {"shape": shape}
    told_missing = bool(spec.get("missing"))

    def num(key, required=True):
        v = spec.get(key)
        if v is None:
            if required and not told_missing:
                errs.append(f"missing {key}")
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            errs.append(f"{key} must be a number")
            return None
        if not (LIM[0] <= v <= LIM[1]):
            errs.append(f"{key} = {v:g} mm is outside {LIM[0]:g}-{LIM[1]:g} mm")
            return None
        out[key] = v
        return v

    if shape == "box":
        L, W, H = num("length"), num("width"), num("height")
        holes = spec.get("holes") or []
        if not isinstance(holes, list):
            errs.append("holes must be a list")
            holes = []
        clean_h = []
        for i, h in enumerate(holes[:12]):
            try:
                d, x, y = float(h["diameter"]), float(h.get("x", (L or 0) / 2)), float(h.get("y", (W or 0) / 2))
            except (TypeError, ValueError, KeyError):
                errs.append(f"hole {i + 1}: needs diameter (and x, y)")
                continue
            if None in (L, W):
                continue
            wall = min(x - d / 2, L - x - d / 2, y - d / 2, W - y - d / 2)
            if d <= 0 or wall < 0.5:
                errs.append(f"hole {i + 1} (Ø{d:g} at x={x:g}, y={y:g}) does not fit inside the "
                            f"{L:g} × {W:g} face with ≥ 0.5 mm wall")
                continue
            clean_h.append({"diameter": d, "x": x, "y": y})
        for i in range(len(clean_h)):
            for j in range(i + 1, len(clean_h)):
                a, b = clean_h[i], clean_h[j]
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < (a["diameter"] + b["diameter"]) / 2 + 0.5:
                    errs.append(f"holes {i + 1} and {j + 1} overlap")
        if len(holes) > 12:
            notes.append("only the first 12 holes are used")
        out["holes"] = clean_h
    elif shape == "cylinder":
        D, H = num("diameter"), num("height")
        bore = num("bore", required=False)
        if bore is not None and D is not None and bore > D - 1.0:
            errs.append(f"bore Ø{bore:g} leaves less than 0.5 mm wall in Ø{D:g}")
    else:
        num("diameter")

    mat = str(spec.get("material") or "aluminium").lower().strip()
    mat = MAT_ALIASES.get(mat, mat)
    if mat not in MATERIALS:
        notes.append(f"material '{spec.get('material')}' unknown, aluminium used for the mass")
        mat = "aluminium"
    out["material"] = mat
    out["name"] = re.sub(r"[^\w \-]", "", str(spec.get("name") or shape))[:40] or shape
    for k in ("unsupported", "missing"):
        if spec.get(k):
            out[k] = [str(x)[:60] for x in spec[k]][:6]
    if out.get("missing"):
        errs.append("missing size: " + ", ".join(out["missing"]))
    if out.get("unsupported"):
        notes.append("not built (not supported yet): " + ", ".join(out["unsupported"]))
    return out, errs, notes


# ------------------------------------------------------------------ geometry
def _sphere_mesh(r, n_lat=32, n_lon=64):
    verts = [[0.0, 0.0, -r]]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat - math.pi / 2
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            verts.append([r * math.cos(th) * math.cos(ph), r * math.cos(th) * math.sin(ph), r * math.sin(th)])
    verts.append([0.0, 0.0, r])
    top = len(verts) - 1
    f = []
    ring = lambda i, j: 1 + (i - 1) * n_lon + (j % n_lon)
    for j in range(n_lon):
        f.append([0, ring(1, j + 1), ring(1, j)])
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j)
            f += [[a, b, c], [a, c, d]]
    for j in range(n_lon):
        f.append([top, ring(n_lat - 1, j), ring(n_lat - 1, j + 1)])
    v = np.array(verts) + r
    return v, np.array(f, np.int64)


def build_mesh(spec: dict):
    s = spec["shape"]
    if s == "box":
        L, W, H = spec["length"], spec["width"], spec["height"]
        holes = [(h["x"], h["y"], h["diameter"]) for h in spec.get("holes", [])]
        return build_plate_with_holes(0, L, 0, W, 0, H, holes, segs=96) if holes else build_box(0, L, 0, W, 0, H)
    if s == "cylinder":
        D, H, bore = spec["diameter"], spec["height"], spec.get("bore")
        if bore:
            return build_ring(bore, D, D / 2, D / 2, 0, H, n=128)
        return build_shaft(D, D / 2, D / 2, 0, H, n=128)
    return _sphere_mesh(spec["diameter"] / 2)


def exact_volume_mm3(spec: dict) -> float:
    s = spec["shape"]
    if s == "box":
        v = spec["length"] * spec["width"] * spec["height"]
        return v - sum(math.pi * h["diameter"] ** 2 / 4 * spec["height"] for h in spec.get("holes", []))
    if s == "cylinder":
        return math.pi / 4 * (spec["diameter"] ** 2 - (spec.get("bore") or 0) ** 2) * spec["height"]
    return math.pi / 6 * spec["diameter"] ** 3


def bbox_mm(spec):
    s = spec["shape"]
    if s == "box":
        return [spec["length"], spec["width"], spec["height"]]
    if s == "cylinder":
        return [spec["diameter"], spec["diameter"], spec["height"]]
    return [spec["diameter"]] * 3


def write_stl(verts, faces, path, name="idris part"):
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        for n_, t in zip(nrm, tri):
            fh.write(struct.pack("<12fH", *n_, *t[0], *t[1], *t[2], 0))


def write_exact_step(spec: dict, path: str) -> tuple[bool, str]:
    """Exact B-rep STEP through OpenCascade (only primitives + boolean cut, the operations the server
    selftest proved). Returns (ok, message)."""
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
    except Exception as e:                                     # noqa: BLE001
        return False, f"OpenCascade not available ({type(e).__name__})"
    try:
        s = spec["shape"]
        z = gp_Dir(0, 0, 1)
        if s == "box":
            shp = BRepPrimAPI_MakeBox(spec["length"], spec["width"], spec["height"]).Shape()
            for h in spec.get("holes", []):
                cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(h["x"], h["y"], -1.0), z), h["diameter"] / 2,
                                               spec["height"] + 2.0).Shape()
                shp = BRepAlgoAPI_Cut(shp, cyl).Shape()
        elif s == "cylinder":
            D, H = spec["diameter"], spec["height"]
            shp = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(D / 2, D / 2, 0), z), D / 2, H).Shape()
            if spec.get("bore"):
                hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(D / 2, D / 2, -1.0), z), spec["bore"] / 2, H + 2).Shape()
                shp = BRepAlgoAPI_Cut(shp, hole).Shape()
        else:
            r = spec["diameter"] / 2
            shp = BRepPrimAPI_MakeSphere(gp_Pnt(r, r, r), r).Shape()
        w = STEPControl_Writer()
        w.Transfer(shp, STEPControl_AsIs)
        if w.Write(path) != IFSelect_RetDone:
            return False, "STEP writer failed"
        return True, "exact B-rep (OpenCascade)"
    except Exception as e:                                     # noqa: BLE001
        return False, f"OpenCascade build failed: {type(e).__name__}: {e}"


def make_part(spec: dict, out_dir: str, tag: str) -> dict:
    clean, errs, notes = validate(spec)
    if errs:
        return {"ok": False, "spec": clean, "errors": errs, "notes": notes}
    v, f = build_mesh(clean)
    vol_mesh = mesh_volume_mm3(v, f)
    vol = exact_volume_mm3(clean)
    rho = MATERIALS[clean["material"]]
    stl_name = f"part_{tag}.stl"
    write_stl(v, f, os.path.join(out_dir, stl_name), clean["name"])
    step_name = f"part_{tag}.step"
    ok, how = write_exact_step(clean, os.path.join(out_dir, step_name))
    if not ok:
        from physics.step_export import write_step
        txt, _ = write_step([{"name": clean["name"], "verts": v, "faces": f}], title=f"Idris part: {clean['name']}")
        with open(os.path.join(out_dir, step_name), "w", encoding="ascii") as fh:
            fh.write(txt)
        how = f"faceted (triangles), because {how}"
    err = abs(vol_mesh / vol - 1)
    return {
        "ok": True, "spec": clean, "notes": notes,
        "results": {"volume_mm3": round(vol, 2), "volume_cm3": round(vol / 1000, 3),
                    "mass_g": round(vol * 1e-9 * rho * 1000, 2), "material": clean["material"],
                    "density_kg_m3": rho, "bounding_box_mm": bbox_mm(clean),
                    "triangles": int(len(f))},
        "self_check": {"mesh_volume_mm3": round(vol_mesh, 2), "exact_volume_mm3": round(vol, 2),
                       "difference_pct": round(100 * err, 3),
                       "ok": err < (0.02 if clean["shape"] != "box" or clean.get("holes") else 1e-9),
                       "note": "STL is a triangle approximation of curved faces; the exact volume is analytic."},
        "files": {"stl": stl_name, "step": step_name, "step_kind": how},
    }


# ------------------------------------------------------------------ built-in parser (no AI needed)
_NUM = r"(\d+(?:[.,]\d+)?)"
PART_WORDS = ("kubus", "cube", "box", "balok", "block", "silinder", "cylinder", "tabung", "tube", "pipa", "pipe",
              "bola", "sphere", "ball", "plat", "plate")
GEAR_WORDS = ("gear", "gearbox", "roda gigi", "reducer", "rasio", "ratio", "kw", "rpm", "torsi", "torque")


# A primitive word inside a real design request is not a primitive: "pillow block bearing housing", "engine block",
# "ball bearing holder", "pipe clamp". Those go to the CAD designer, not to the box/cylinder builder.
NOT_PRIMITIVE = re.compile(
    r"pillow|plummer|engine block|terminal block|block bearing|bearing|bantalan|laher|housing|rumah|bracket|braket|"
    r"holder|dudukan|clamp|klem|mount|slot|flange|flens|\brib|rusuk|fillet|chamfer|thread|ulir|hinge|engsel|"
    r"cent(?:re|er) height|\bfor an? \b|\buntuk\b|frame|rangka|table|meja|weld|\blas\b|shelf|\brak\b|feet|\bkaki\b|"
    r"\bjaw|chuck|fixture|\bgrid\b|dowel|square tube|hollow section|profil|stepped|bertingkat")
MAX_PRIMITIVE_WORDS = 14          # "plate 100x50x5 with 4 holes" is short; a real design brief is a long sentence


def looks_like_part_request(msg: str) -> bool:
    m = (msg or "").lower()
    if NOT_PRIMITIVE.search(m) or len(re.findall(r"[^\W\d_]{2,}", m)) > MAX_PRIMITIVE_WORDS:
        return False
    return any(re.search(rf"\b{w}\b", m) for w in PART_WORDS) and not any(w in m for w in GEAR_WORDS)


def _f(s):
    return float(s.replace(",", "."))


def builtin_parse(msg: str) -> dict:
    """Rule-based fallback for short requests: 'kubus 50 mm', 'balok 100x60x20 aluminium',
    'silinder diameter 30 tinggi 80', 'tabung d40 lubang 20 tinggi 60', 'bola 25'."""
    m = msg.lower()
    mat = next((k for k in list(MATERIALS) + list(MAT_ALIASES) if re.search(rf"\b{k}\b", m)), None)
    scale = 10.0 if re.search(r"\d\s*cm\b", m) else (1000.0 if re.search(r"\d\s*m\b", m) else 1.0)
    nums = [_f(x) * scale for x in re.findall(_NUM, m)]
    spec = {"material": mat} if mat else {}
    if re.search(r"\b(kubus|cube)\b", m):
        if nums:
            return {**spec, "shape": "box", "length": nums[0], "width": nums[0], "height": nums[0], "name": "cube"}
        return {**spec, "shape": "box", "missing": ["side length"]}
    if re.search(r"\b(bola|sphere|ball)\b", m):
        return {**spec, "shape": "sphere", "diameter": nums[0]} if nums else {**spec, "shape": "sphere", "missing": ["diameter"]}
    if re.search(r"\b(silinder|cylinder|tabung|tube|pipa|pipe)\b", m):
        def after(words):
            g = re.search(rf"(?:{words})\s*(?:=|:)?\s*{_NUM}", m)
            return _f(g.group(1)) * scale if g else None
        D = after(r"diameter|dia|d|ø|od")
        H = after(r"tinggi|panjang|height|length|h|l")
        bore = after(r"lubang|bore|id|dalam")
        rest = [x for x in nums if x not in (D, H, bore)]
        D = D or (rest.pop(0) if rest else None)
        H = H or (rest.pop(0) if rest else None)
        out = {**spec, "shape": "cylinder", "diameter": D, "height": H}
        if bore:
            out["bore"] = bore
        miss = [k for k in ("diameter", "height") if out[k] is None]
        return {**out, "missing": miss} if miss else out
    if re.search(r"\b(balok|box|block|plat|plate)\b", m):
        if len(nums) >= 3:
            return {**spec, "shape": "box", "length": nums[0], "width": nums[1], "height": nums[2]}
        return {**spec, "shape": "box", "missing": ["length, width, height"]}
    return {"shape": "unknown"}


# ------------------------------------------------------------------ sentence -> part
def design_from_message(message: str, out_dir: str, tag: str, ai_parse=None, base_url: str = "") -> dict:
    """ai_parse(message) -> {spec, provider, model, attempts} or raises. Falls back to the built-in parser.
    Returns the make_part result plus who read the sentence, absolute download links and a chat answer."""
    parsed_by, attempts = "built-in parser (no AI)", []
    spec = None
    if ai_parse is not None:
        try:
            r = ai_parse(message)
            spec, parsed_by = r["spec"], f"{r['provider']} ({r['model']})"
            attempts = r.get("attempts") or []
        except Exception as e:                                 # noqa: BLE001
            attempts = list(getattr(e, "attempts", None) or [{"error": str(e)}])
    if spec is None or str(spec.get("shape", "")).lower() in ("", "unknown", "none"):
        b = builtin_parse(message)
        if spec is None or b.get("shape") != "unknown":
            spec = b
            parsed_by = "built-in parser (no AI)" if ai_parse is None or not attempts else \
                "built-in parser (AI failed: " + "; ".join(a.get("error", "")[:80] for a in attempts) + ")"
    res = make_part(spec, out_dir, tag)
    res["read_by"] = parsed_by
    res["ai_attempts"] = attempts
    if res["ok"]:
        for k in ("stl", "step"):
            res["files"][k + "_url"] = f"{base_url}/download/{res['files'][k]}"
    res["answer"] = chat_answer(res)
    return res


def chat_answer(res: dict) -> str:
    sp = res.get("spec") or {}
    if not res.get("ok"):
        why = "; ".join(res.get("errors") or ["I could not read a part from this sentence"])
        return (f"I could not build this part yet: {why}. Try e.g. 'kubus 50 mm', 'balok 100x60x20 aluminium', "
                f"'tabung d40 lubang 20 tinggi 60' or 'bola 25'. (Read by: {res.get('read_by')})")
    r, f = res["results"], res["files"]
    if sp["shape"] == "box":
        size = f"{sp['length']:g} × {sp['width']:g} × {sp['height']:g} mm"
        if sp.get("holes"):
            size += f" with {len(sp['holes'])} through-hole(s) " + ", ".join(
                f"Ø{h['diameter']:g} at ({h['x']:g}, {h['y']:g})" for h in sp["holes"])
    elif sp["shape"] == "cylinder":
        size = f"Ø{sp['diameter']:g} × {sp['height']:g} mm" + (f", bore Ø{sp['bore']:g}" if sp.get("bore") else "")
    else:
        size = f"Ø{sp['diameter']:g} mm"
    lines = [f"Built: {sp['name']} ({sp['shape']}), {size}, {sp['material']}.",
             f"Volume {r['volume_cm3']:g} cm³, mass {r['mass_g']:g} g (density {r['density_kg_m3']:g} kg/m³).",
             f"Self-check: mesh volume vs exact {res['self_check']['difference_pct']:g} %.",
             f"STEP ({f['step_kind']}): {f.get('step_url', f['step'])}",
             f"STL: {f.get('stl_url', f['stl'])}",
             f"Sentence read by: {res['read_by']}. Dimensions are yours; volume and mass are computed, not from the AI."]
    if res.get("notes"):
        lines.append("Note: " + "; ".join(res["notes"]))
    return "\n".join(lines)
