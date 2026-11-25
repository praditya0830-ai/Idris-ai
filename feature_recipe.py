"""Feature recipes: an ordered list of feature steps the browser kernel builds, checked BEFORE the kernel runs.

Why a recipe and not only code (Arya, 22 Nov): a recipe is a feature tree the engineer can open, edit and rebuild
one step at a time, and a broken step can be repaired without throwing the whole part away. Code stays the second
rung, for shapes a recipe cannot describe.

Vocabulary: the same feature names the Studio already edits (`index.html` FEAT), so a recipe is not a second
language:

  sketch   {prof: rect|circle|slot|poly, W, Hh, D, n, plane, at, off, u, v}
  push     {sketch, dist, op}
  revolve  {sketch, axis, angle, op}
  hole     {D, x, y, depthMode, depth}
  edge     {mode: fillet|chamfer, r, which}
  repeat   {kind, count, dx, dy, cx, cy, plane}

Order is fixed: datum/base sketch -> additive (push/revolve) -> holes -> dress (fillet/chamfer) last. A recipe that
breaks that order, points at an id that does not exist yet, opens a profile that is not closed, puts a hole outside
the face, or asks for a fillet bigger than the material is refused HERE, with a sentence the engineer can act on —
before a kernel call is spent.

Every number in the recipe comes from the caller's `numbers` (each with its source); this module never invents one.
"""
from __future__ import annotations

import math

import sheet_features

# The stage a step belongs to. Sheet-metal steps join the same order: the base flange is the additive step, the folds
# and the punch are dress-up, and the analysis steps (motion, sizes, curvature) run last and change no geometry.
STEP_ORDER = {"sketch": 0, "push": 1, "revolve": 1, "smbase": 1, "surfbound": 1,
              "hole": 2, "edge": 3, "repeat": 3, "wall": 3,
              "smedge": 3, "smmiter": 3, "hem": 3, "jog": 3, "form": 3, "weldtrim": 3,
              "motion": 4, "sizes": 4, "curv": 4}
ALLOWED = tuple(STEP_ORDER)

# feature names the Studio's buildPart understands for these steps; kept in one place so the two sides can be compared
STUDIO_TYPES = {"sketch": "sketch", "push": "push", "revolve": "revolve", "hole": "hole", "edge": "edge",
                "repeat": "repeat", "wall": "wall", "smbase": "smbase", "smedge": "smedge",
                "smmiter": "smmiter", "hem": "hem", "jog": "jog", "form": "form", "weldtrim": "weldtrim",
                "motion": "motion", "sizes": "sizes", "curv": "curv", "surfbound": "surfbound"}

# values that describe the recipe rather than the part (an angle, a side count); any other number must be one the
# engineer gave, or the recipe is refused with the number named
STRUCTURAL = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 30.0, 45.0, 60.0, 90.0, 180.0, 270.0, 360.0)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _first(d: dict, *keys):
    """The first present value, so a legitimate 0 is not lost to `or`."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _step(sid, kind, **params):
    return {"id": str(sid), "type": kind, "p": {k: v for k, v in params.items() if v is not None}}


# ------------------------------------------------------------------ templates (known-good, filled with the numbers)
def plate_steps(numbers: dict, holes: list, fillet=None, plane="XY") -> list:
    """A plate: one closed rectangle, one push, the holes, and the fillet last (never before the holes)."""
    L, W, T = numbers.get("length_mm"), numbers.get("width_mm"), numbers.get("thickness_mm")
    if not (L and W and T and L > 0 and W > 0 and T > 0):
        return []
    steps = [_step("base", "sketch", prof="rect", W=L, Hh=W, plane=plane, at="value", off=0, u=0, v=0),
             _step("body", "push", sketch="base", dist=T, op="add")]
    for i, h in enumerate(holes or [], 1):
        d, x, y = _num(_first(h, "d_mm", "D")), _num(_first(h, "x_mm", "x")), _num(_first(h, "y_mm", "y"))
        if not (d and d > 0 and x is not None and y is not None):
            return []
        steps.append(_step(f"hole{i}", "hole", D=d, x=x, y=y, depthMode="thru"))
    if fillet:
        steps.append(_step("dress", "edge", mode="fillet", r=float(fillet), which="vertical"))
    return steps


TITLES = {"plate": "Plate with holes", "plate_pocket": "Plate with a pocket", "bracket_l": "L bracket",
          "channel": "Channel bracket", "gusset": "Gusset", "flange": "Bolt flange", "shaft_step": "Stepped shaft",
          "tube": "Tube", "housing": "Housing"}


def _ents(pts: list) -> list:
    """A closed profile from corner points: the kernel draws real lines and closes the loop."""
    return [{"t": "line", "a": [float(pts[i][0]), float(pts[i][1])],
             "b": [float(pts[(i + 1) % len(pts)][0]), float(pts[(i + 1) % len(pts)][1])]}
            for i in range(len(pts))]


def _prism(sid: str, pts: list, plane: str, dist: float, holes=None) -> list:
    steps = [_step(sid, "sketch", prof="ents", ents=_ents(pts), plane=plane, at="value", off=0, u=0, v=0),
             _step(sid + "_b", "push", sketch=sid, dist=dist, op="add")]
    for i, h in enumerate(holes or [], 1):
        d, x, y = _num(_first(h, "d_mm", "D")), _num(_first(h, "x_mm", "x")), _num(_first(h, "y_mm", "y"))
        if not (d and d > 0 and x is not None and y is not None):
            return []
        steps.append(_step(f"hole{i}", "hole", D=d, x=x, y=y, depthMode="thru", type="simple"))
    return steps


def bracket_l_steps(n: dict, fillet=None) -> list:
    """L bracket: the L profile in XZ, extruded across its width. legs and thickness must leave room for the corner."""
    A, B, T, W = n.get("leg_a_mm"), n.get("leg_b_mm"), n.get("thickness_mm"), n.get("width_mm")
    if not (A and B and T and W and T < A and T < B):
        return []
    steps = _prism("base", [(0, 0), (A, 0), (A, T), (T, T), (T, B), (0, B)], "XZ", W)
    if steps and fillet:
        steps.append(_step("dress", "edge", mode="fillet", r=float(fillet), which="vertical"))
    return steps


def channel_steps(n: dict, fillet=None) -> list:
    """U / channel: two legs and a base, extruded along its length."""
    W, H, T, L = n.get("width_mm"), n.get("height_mm"), n.get("thickness_mm"), n.get("length_mm")
    if not (W and H and T and L and 2 * T < W and T < H):
        return []
    pts = [(0, 0), (W, 0), (W, H), (W - T, H), (W - T, T), (T, T), (T, H), (0, H)]
    return _prism("base", pts, "XZ", L)


def gusset_steps(n: dict, fillet=None) -> list:
    A, B, T = n.get("leg_a_mm"), n.get("leg_b_mm"), n.get("thickness_mm")
    if not (A and B and T):
        return []
    return _prism("base", [(0, 0), (A, 0), (0, B)], "XZ", T)


def shaft_steps(n: dict, fillet=None) -> list:
    """Stepped shaft: one profile with both diameters, revolved about Z (the axis is the left edge)."""
    D, L, d, l = n.get("diameter_mm"), n.get("length_mm"), n.get("step_diameter_mm"), n.get("step_length_mm")
    if not (D and L and d and l and d < D and l < L):
        return []
    R, r = D / 2, d / 2
    pts = [(0, 0), (R, 0), (R, L - l), (r, L - l), (r, L), (0, L)]
    return [_step("base", "sketch", prof="ents", ents=_ents(pts), plane="XZ", at="value", off=0, u=0, v=0),
            _step("body", "revolve", sketch="base", axis="Z", angle=360, op="add")]


def tube_steps(n: dict) -> list:
    D, L, T = n.get("diameter_mm"), n.get("length_mm"), n.get("thickness_mm")
    if not (D and L and T and 2 * T < D):
        return []
    R, r = D / 2, D / 2 - T
    pts = [(r, 0), (R, 0), (R, L), (r, L)]
    return [_step("base", "sketch", prof="ents", ents=_ents(pts), plane="XZ", at="value", off=0, u=0, v=0),
            _step("body", "revolve", sketch="base", axis="Z", angle=360, op="add")]


def flange_steps(n: dict, holes: list, fillet=None) -> list:
    D, T = n.get("diameter_mm"), n.get("thickness_mm")
    if not (D and T):
        return []
    steps = [_step("base", "sketch", prof="circle", D=D, plane="XY", at="value", off=0, u=0, v=0),
             _step("body", "push", sketch="base", dist=T, op="add")]
    for i, h in enumerate(holes or [], 1):
        d, x, y = _num(_first(h, "d_mm", "D")), _num(_first(h, "x_mm", "x")), _num(_first(h, "y_mm", "y"))
        if not (d and d > 0 and x is not None and y is not None):
            return []
        steps.append(_step(f"hole{i}", "hole", D=d, x=x, y=y, depthMode="thru", type="simple"))
    return steps


def housing_steps(n: dict, holes: list, fillet=None) -> list:
    """Box housing: a solid box, its bolt holes, then the wall that opens the top."""
    L, W, H, T = n.get("length_mm"), n.get("width_mm"), n.get("height_mm"), n.get("thickness_mm")
    if not (L and W and H and T and 2 * T < min(L, W, H)):
        return []
    steps = [_step("base", "sketch", prof="rect", W=L, Hh=W, plane="XY", at="value", off=0, u=0, v=0),
             _step("body", "push", sketch="base", dist=H, op="add")]
    for i, h in enumerate(holes or [], 1):
        d, x, y = _num(_first(h, "d_mm", "D")), _num(_first(h, "x_mm", "x")), _num(_first(h, "y_mm", "y"))
        if not (d and d > 0 and x is not None and y is not None):
            return []
        steps.append(_step(f"hole{i}", "hole", D=d, x=x, y=y, depthMode="thru", type="simple"))
    steps.append(_step("wall", "wall", t=T, open="top"))
    return steps


def pocket_steps(n: dict, holes: list, fillet=None) -> list:
    """A plate with a pocket: the second sketch sits on the top face and cuts down by the depth the engineer gave."""
    L, W, T, PD = n.get("length_mm"), n.get("width_mm"), n.get("thickness_mm"), n.get("pocket_depth_mm")
    if not (L and W and T and PD and PD < T and 2 * T < min(L, W)):
        return []
    steps = [_step("base", "sketch", prof="rect", W=L, Hh=W, plane="XY", at="value", off=0, u=0, v=0),
             _step("body", "push", sketch="base", dist=T, op="add"),
             _step("pocket", "sketch", prof="rect", W=L - 2 * T, Hh=W - 2 * T, plane="XY", at="top", off=0, u=0, v=0),
             _step("pocket_b", "push", sketch="pocket", dist=-PD, op="cut")]
    for i, h in enumerate(holes or [], 1):
        d, x, y = _num(_first(h, "d_mm", "D")), _num(_first(h, "x_mm", "x")), _num(_first(h, "y_mm", "y"))
        if not (d and d > 0 and x is not None and y is not None):
            return []
        steps.append(_step(f"hole{i}", "hole", D=d, x=x, y=y, depthMode="thru", type="simple"))
    return steps


def sheet_steps(n: dict, feats: list, material: str = "steel", R: float | None = None) -> list:
    """A sheet part: the base flange, then the folds and punches the engineer asked for, in tab order.

    Thickness and radius live on the base flange (the Studio's sheet-metal steps read them from there), so a hem or a
    jog never has to guess them and every later step stands on the same plate.
    """
    L, W, T = n.get("length_mm"), n.get("width_mm"), n.get("thickness_mm")
    if not (L and W and T and L > 0 and W > 0 and T > 0):
        return []
    steps = [_step("base", "smbase", L=float(L), W=float(W), t=float(T),
                   R=float(R if R is not None else T), material=material, k=0)]
    for i, ft in enumerate(feats or [], 1):
        if not isinstance(ft, dict) or not ft.get("type"):
            return []
        # built directly: a forming tool's own `kind` field would collide with _step's positional argument
        steps.append({"id": str(ft.get("id") or f"{ft['type']}{i}"), "type": str(ft["type"]),
                      "p": {k: v for k, v in (ft.get("p") or {}).items() if v is not None}})
    if len(steps) < 2:
        return []
    return steps


def boundary_steps(n: dict, sketch_pts: list | None = None) -> list:
    """A boundary surface: one closed sketch of exactly four curves, then the surface through all four."""
    pts = sketch_pts if sketch_pts and len(sketch_pts) == 4 else None
    if not pts:
        return []
    return [_step("edges", "sketch", prof="ents", ents=_ents(pts), plane="XZ", at="value", off=0, u=0, v=0),
            _step("patch", "surfbound", sketch="edges", t=0, solid="sheet")]


def template_for(shape_class: str, numbers: dict, holes: list | None = None, fillet=None,
                 hint: str | None = None, sheet: dict | None = None) -> dict | None:
    """-> a recipe {title, shape_class, steps, template} or None when no known-good template fits this shape.

    `hint` names the template the engineer's words asked for (part_spec.template_hint); the shape class stays the
    safety net when the words name nothing.
    """
    cls = str(shape_class or "prismatic").lower()
    n = numbers or {}
    tpl = str(hint or "").lower()
    # a sheet part with folds or a punch: the base flange carries t and R, and the features follow it
    if isinstance(sheet, dict) and sheet.get("features") and cls in ("sheet", "prismatic", "revolve"):
        feats = [x for x in sheet["features"] if isinstance(x, dict) and x.get("ok", True)]
        steps = sheet_steps(n, feats, str(sheet.get("material") or "steel"), sheet.get("R"))
        if steps:
            return {"title": "Sheet metal part", "shape_class": cls, "steps": steps,
                    "template": "sheet_features", "origin": "template"}
        return None                              # a fold with a size missing: ask, never draw it with a guess
    built = {"bracket_l": lambda: bracket_l_steps(n, fillet),
             "channel": lambda: channel_steps(n, fillet),
             "gusset": lambda: gusset_steps(n, fillet),
             "shaft_step": lambda: shaft_steps(n, fillet),
             "tube": lambda: tube_steps(n),
             "flange": lambda: flange_steps(n, holes or [], fillet),
             "housing": lambda: housing_steps(n, holes or [], fillet),
             "plate_pocket": lambda: pocket_steps(n, holes or [], fillet)}
    if tpl in built:
        steps = built[tpl]()
        if steps:
            return {"title": TITLES[tpl], "shape_class": cls, "steps": steps, "template": tpl,
                    "origin": "template"}
        if tpl != "plate":
            return None                      # the named shape does not have its sizes: ask, never swap silently
    if cls in ("prismatic", "sheet", "frame"):
        steps = plate_steps(n, holes or [], fillet)
        return {"title": "Plate with holes", "shape_class": cls, "steps": steps, "template": "plate",
                "origin": "template"} if steps else None
    if cls == "revolve":
        D, T = n.get("diameter_mm"), n.get("length_mm")
        if not (D and T and D > 0 and T > 0):
            return None
        return {"title": "Revolved boss", "shape_class": cls, "template": "boss", "origin": "template",
                "steps": [_step("base", "sketch", prof="circle", D=D, plane="XY", at="value", off=0, u=0, v=0),
                          _step("body", "revolve", sketch="base", axis="Z", angle=360, op="add")]}
    return None                                  # freeform and anything unknown: the caller goes to the code rung


# ------------------------------------------------------------------ pre-build validation
def validate(recipe: dict | None, numbers: dict | None = None) -> tuple:
    """-> (ok, errors). Every error is a sentence about this recipe, not a stack trace."""
    errors = []
    steps = (recipe or {}).get("steps") if isinstance(recipe, dict) else recipe
    if not isinstance(steps, list) or not steps:
        return False, ["the recipe has no steps"]
    seen, solid = {}, False
    order = -1
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or not s.get("type"):
            errors.append(f"step {i + 1} has no feature type")
            continue
        kind, sid = str(s.get("type")), str(s.get("id") or f"s{i + 1}")
        p = s.get("p") if isinstance(s.get("p"), dict) else {}
        if kind not in allowed_steps():
            errors.append(f"step '{sid}': '{kind}' is not a feature this build validates")
            continue
        if sid in seen:
            errors.append(f"step '{sid}' appears twice")
            continue
        # a sketch does not advance the stage: a pocket needs a second sketch after the first push
        if kind != "sketch" and STEP_ORDER[kind] < order:
            errors.append(f"step '{sid}': {kind} comes after a later stage; the order is sketch, additive, holes, dress")
        if kind != "sketch":
            order = max(order, STEP_ORDER[kind])
        if kind == "sketch":
            closed, why = _closed_profile(p)
            if not closed:
                errors.append(f"sketch '{sid}': {why}")
        elif kind in ("push", "revolve"):
            ref = str(p.get("sketch") or "")
            if ref and ref not in seen:
                errors.append(f"step '{sid}' points at sketch '{ref}', which does not exist before it")
            if not ref and not solid:
                errors.append(f"step '{sid}' has no sketch to build on")
            if kind == "push" and not (_num(p.get("dist")) and _num(p.get("dist")) != 0):
                errors.append(f"step '{sid}' needs a distance")
            solid = True
        elif kind == "hole":
            if not solid:
                errors.append(f"hole '{sid}' has no solid before it")
            d = _num(p.get("D"))
            if not (d and d > 0):
                errors.append(f"hole '{sid}' needs a diameter")
            elif not _hole_inside(p, steps, numbers or {}):
                errors.append(f"hole '{sid}' at ({p.get('x')}, {p.get('y')}) falls outside the face")
        elif kind == "wall":
            if not solid:
                errors.append(f"wall '{sid}' has no solid to hollow out")
            t = _num(p.get("t"))
            if not (t and t > 0):
                errors.append(f"wall '{sid}' needs a thickness")
        elif kind == "smbase":
            L, W, T = _num(p.get("L")), _num(p.get("W")), _num(p.get("t"))
            if not (L and W and T and L > 0 and W > 0 and T > 0):
                errors.append(f"base flange '{sid}' needs a length, a width and a thickness")
            elif _num(p.get("R")) is not None and _num(p.get("R")) < 0:
                errors.append(f"base flange '{sid}': the inside radius cannot be negative")
            solid = True
        elif kind in ("hem", "jog", "form", "smedge", "smmiter"):
            sh = _sheet_before(steps, sid)
            if not solid or not sh:
                errors.append(f"{kind} '{sid}' needs a Base Flange before it")
                continue
            t, r, L, W = sh
            if kind == "hem":
                f_leg = _num(p.get("F"))
                if not (f_leg and f_leg > 0):
                    errors.append(f"hem '{sid}' needs a leg length")
                else:
                    ok, why = sheet_features.check_hem(t, r, f_leg, L, W)
                    if not ok:
                        errors.append(f"hem '{sid}': {why}")
            elif kind == "jog":
                d, h = _num(p.get("d")), _num(p.get("h"))
                if not (d and h):
                    errors.append(f"jog '{sid}' needs the jog line and the rise")
                else:
                    ok, why = sheet_features.check_jog(t, r, d, h, L)
                    if not ok:
                        errors.append(f"jog '{sid}': {why}")
            elif kind == "form":
                D, dep = _num(p.get("D")), _num(p.get("depth"))
                if not (D and dep):
                    errors.append(f"forming tool '{sid}' needs a diameter and a depth")
                else:
                    ok, why = sheet_features.check_form(D, dep, _num(p.get("x")) or 0.0,
                                                        _num(p.get("y")) or 0.0, L, W)
                    if not ok:
                        errors.append(f"forming tool '{sid}': {why}")
            else:
                if not (_num(p.get("F")) and _num(p.get("F")) > 0):
                    errors.append(f"{kind} '{sid}' needs a flange length")
        elif kind == "surfbound":
            ref = str(p.get("sketch") or "")
            if ref and ref not in seen:
                errors.append(f"boundary surface '{sid}' points at sketch '{ref}', which does not exist before it")
            elif ref:
                sk = (seen.get(ref) or {}).get("p") or {}
                ents = sk.get("ents") or []
                if str(sk.get("prof")) != "ents" or len(ents) != 4:
                    errors.append(f"boundary surface '{sid}' takes four boundary curves: sketch '{ref}' has "
                                  f"{len(ents) if str(sk.get('prof')) == 'ents' else 0}")
        elif kind == "weldtrim":
            if not solid:
                errors.append(f"Trim/Extend '{sid}' has no body to cut")
            ref = str(p.get("sketch") or "")
            if not ref or ref not in seen:
                errors.append(f"Trim/Extend '{sid}' needs the sketch whose line runs the cutter")
            if not str(p.get("profile") or "").strip():
                errors.append(f"Trim/Extend '{sid}' needs the cutter profile")
        elif kind == "motion":
            ang, steps_n = _num(p.get("angle")), _num(p.get("steps"))
            if not (ang and 0 < ang <= 360):
                errors.append(f"motion study '{sid}' needs a travel angle above 0 and at most 360 degrees")
            if not (steps_n and 2 <= steps_n <= 24):
                errors.append(f"motion study '{sid}' needs 2 to 24 steps")
            if not solid:
                errors.append(f"motion study '{sid}' has no part to move")
        elif kind == "sizes":
            ref = str(p.get("step") or "")
            if not ref or ref not in seen:
                errors.append(f"size edit '{sid}' points at a step that does not exist: '{ref}'")
            if not str(p.get("param") or "").strip():
                errors.append(f"size edit '{sid}' needs the size to change")
            if not (_num(p.get("value")) and _num(p.get("value")) > 0):
                errors.append(f"size edit '{sid}' needs a value above 0")
        elif kind == "curv":
            if not solid:
                errors.append(f"curvature '{sid}' has no surface to measure")
        elif kind == "edge":
            r = _num(p.get("r"))
            if not (r and r > 0):
                errors.append(f"step '{sid}' needs a size")
            elif not _fillet_fits(r, steps, numbers or {}):
                errors.append(f"fillet R{r:g} mm is too big for this part (must be under half the thinnest wall)")
        seen[sid] = s
    return (not errors), errors


def allowed_steps() -> tuple:
    """The types this build validates. A caller can compare it with the Studio's feature list."""
    return ALLOWED


def _sheet_before(steps, sid: str):
    """-> (t, R, L, W) from the base flange the step stands on, or None."""
    for s in steps:
        if isinstance(s, dict) and s.get("id") == sid:
            break
        if isinstance(s, dict) and s.get("type") == "smbase":
            p = s.get("p") or {}
            t, L, W = _num(p.get("t")), _num(p.get("L")), _num(p.get("W"))
            r = _num(p.get("R"))
            if t and L and W:
                return (t, r if r is not None else t, L, W)
    return None


def _closed_profile(p: dict) -> tuple:
    prof = str(p.get("prof") or "")
    if prof in ("rect", "circle", "slot"):
        dims = [_num(p.get(k)) for k in ("W", "Hh", "D")]
        if not any(d and d > 0 for d in dims):
            return False, "the profile has no size"
        return True, ""
    if prof == "poly":
        n = _num(p.get("n"))
        return (bool(n and n >= 3), "a polygon needs at least three sides")
    if prof == "ents":
        ents = p.get("ents") or []
        if len(ents) < 3:
            return False, "a sketch from lines needs at least three entities"
        pts = []
        for e in ents:
            if not isinstance(e, dict) or not e.get("a"):
                return False, "a sketch entity has no start point"
            pts.append(tuple(e["a"]))
            if e.get("b"):
                pts.append(tuple(e["b"]))
        if len(pts) >= 4 and _is_closed(pts):
            return True, ""
        return False, "the lines and arcs do not close: the profile must end where it started"
    return False, "the profile type is not one this build validates"


def _is_closed(pts) -> bool:
    return math.dist(pts[0], pts[-1]) < 1e-6


def _face_of(steps) -> dict | None:
    """The first solid-making profile in the recipe: {kind: rect, W, Hh} or {kind: circle, D}."""
    for s in steps:
        if isinstance(s, dict) and s.get("type") == "sketch":
            p = s.get("p") or {}
            if str(p.get("prof")) == "rect":
                W, H = _num(p.get("W")), _num(p.get("Hh"))
                if W and H:
                    return {"kind": "rect", "W": W, "Hh": H}
            if str(p.get("prof")) == "circle":
                D = _num(p.get("D"))
                if D:
                    return {"kind": "circle", "D": D}
    return None


def _hole_inside(p: dict, steps, numbers: dict) -> bool:
    face = _face_of(steps)
    d = _num(p.get("D")) or 0
    x, y = _num(p.get("x")) or 0, _num(p.get("y")) or 0
    if not face:
        W, H = _num(numbers.get("length_mm")), _num(numbers.get("width_mm"))
        face = {"kind": "rect", "W": W, "Hh": H} if (W and H) else None
    if not face:
        return True                              # nothing to compare against: the kernel is the next check
    if face["kind"] == "circle":
        return (x * x + y * y) ** 0.5 + d / 2 <= face["D"] / 2 + 1e-9
    return abs(x) + d / 2 <= face["W"] / 2 + 1e-9 and abs(y) + d / 2 <= face["Hh"] / 2 + 1e-9


def _fillet_fits(r: float, steps, numbers: dict) -> bool:
    thin = _num(numbers.get("thickness_mm"))
    for s in steps:
        if isinstance(s, dict) and s.get("type") in ("push", "revolve"):
            p = s.get("p") or {}
            thin = thin or _num(p.get("dist"))
    face = _face_of(steps)
    limits = [thin] if thin else []
    if face:
        limits.append(min(face["W"], face["Hh"]) if face["kind"] == "rect" else face["D"])
    if not limits:
        return True
    return all(r < lim / 2 for lim in limits)


# ------------------------------------------------------------------ sources for the UI
def step_sources(recipe: dict, numbers: dict) -> list:
    """One row per step that carries a number, with where that number came from (the UI shows this)."""
    by_key = {k: v for k, v in (numbers or {}).items() if isinstance(v, dict)}
    rows = []
    for s in (recipe or {}).get("steps") or []:
        if not isinstance(s, dict):
            continue
        used = {}
        for k, v in (s.get("p") or {}).items():
            if isinstance(v, (int, float)):
                used[k] = v
        if used:
            srcs = {k: by_key.get({"W": "length_mm", "Hh": "width_mm", "dist": "thickness_mm", "D": "hole_d_mm",
                                   "r": "fillet_mm"}.get(k, k), {}) for k in used}
            rows.append({"step": s.get("id"), "feature": s.get("type"), "values": used,
                         "sources": {k: (v or {}).get("source") for k, v in srcs.items()}})
    return rows


# ------------------------------------------------------------------ an AI-written recipe (Tahap A2)
# Arya asked for IDRIS to use CAD tools, not to write code. So when no known-good template fits, the AI is asked for
# a RECIPE in this vocabulary - never for code - and every number in it must be one the engineer gave, a table value,
# or structural (an angle, a side count). A recipe that invents a size is refused with the number named, and the AI
# is asked to repair exactly those steps, at most twice, before the ladder falls to code.
RECIPE_SYSTEM = """You write a CAD feature recipe as JSON for a browser kernel. Answer with the JSON object only.

Shape: {"title": "short name", "steps": [{"id": "base", "type": "sketch", "p": {...}}, ...]}

Step types and their parameters:
  sketch   p: {prof: rect|circle|slot|poly|ents, W, Hh, D, n, plane: XY|XZ|YZ, at: value|top, off, u, v, ents}
           ents = [{"t": "line", "a": [x, y], "b": [x, y]}] and must close (last b = first a)
  push     p: {sketch: <sketch id>, dist, op: add|cut}
  revolve  p: {sketch: <sketch id>, axis: X|Y|Z, angle, op: add|cut}
  hole     p: {D, x, y, depthMode: thru|value, depth}
  wall     p: {t, open: top|bottom}
  edge     p: {mode: fillet|chamfer, r, which: vertical|top|bottom|all}
  repeat   p: {kind: line|circle|mirror, count, dx, dy, cx, cy, plane}
  smbase   p: {L, W, t, R, material, k}                      sheet-metal base flange (the plate)
  hem      p: {edge: x+|x-|y+|y-, F}                        180 degree fold at an edge; needs an smbase
  jog      p: {d, h}                                       two 90 degree bends at the line x = d; needs an smbase
  form     p: {kind: emboss|lance, D, depth, x, y}         forming tool (punch) on the sheet; needs an smbase
  surfbound p: {sketch, t, solid: sheet|solid}             boundary surface through a sketch of exactly four curves
  weldtrim p: {sketch, profile, extend}                     cut a member with another member's profile
  motion   p: {axis: X|Y|Z, angle, steps, px, py, pz}      swept envelope of the travel (changes no geometry)
  sizes    p: {step, param, value}                          edit one size of an earlier step and rebuild
  curv     p: {kind: curv|zebra}                            curvature map / isophote bands of the body

Order: sketch -> push/revolve/smbase -> hole -> wall/edge/repeat/hem/jog/form, then motion/sizes/curv last.
A sketch may appear again later (a pocket needs one).
Every number must be one of the numbers the engineer gave; do not add sizes of your own and do not round them.
Only 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 30, 45, 60, 90, 180, 270 and 360 may appear without being a given size.
Use millimetres. Keep the sketch profile closed and keep holes inside the face."""


def _numbers_for_prompt(numbers: dict) -> str:
    rows = []
    for k, v in sorted((numbers or {}).items()):
        if isinstance(v, (int, float)):
            rows.append(f"  {k} = {v:g}")
    return "\n".join(rows) or "  (none)"


def unsourced_numbers(recipe: dict | None, numbers: dict | None = None) -> list:
    """Every number in the recipe must be a given size or structural; -> the ones that are not, by step."""
    flat = [float(v) for v in (numbers or {}).values() if isinstance(v, (int, float))]
    bad = []
    for s in (recipe or {}).get("steps") or []:
        if not isinstance(s, dict):
            continue
        for k, v in (s.get("p") or {}).items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if any(abs(float(v) - f) <= 1e-9 for f in flat) or float(v) in STRUCTURAL:
                continue
            bad.append(f"step '{s.get('id')}': {k} = {v:g} is not a size the engineer gave")
    return bad


def ai_recipe(message: str, numbers: dict | None = None, spec: dict | None = None,
              shape_class: str = "prismatic", provider=None, deadline=None, call=None,
              repairs: int = 2) -> dict:
    """Ask the AI for a recipe, validate it here, and let it repair only the refused steps (at most `repairs`).

    -> {title, shape_class, steps, origin: "ai", attempts} or {steps: [], error, attempts} when it never passed.
    `call(prompt) -> text` is injectable, so a test never reaches a paid provider.
    """
    import llm
    flat = {k: v for k, v in (numbers or {}).items() if isinstance(v, (int, float))}
    name = str((spec or {}).get("name") or "Part")
    prompt = (f"The engineer asked for: {message}\n\nPart: {name} ({shape_class}).\n"
              f"The sizes the engineer gave (use these and nothing else):\n{_numbers_for_prompt(flat)}\n\n"
              "Write the recipe for this part.")
    if call is None:
        provider = provider or (llm.provider_order() or [None])[0]
        if not provider:
            return {"steps": [], "error": "no AI provider is configured, so no recipe was written", "attempts": 0}

        def call(p):
            text, _model = llm.call_provider(provider, RECIPE_SYSTEM, p, 1400,
                                             timeout=max(5.0, (deadline - __import__("time").time()))
                                             if deadline else None)
            return text
    errors, attempts = [], 0
    for _ in range(max(1, repairs + 1)):
        attempts += 1
        ask = prompt if not errors else (prompt + "\n\nThe previous recipe was refused for these reasons:\n"
                                         + "\n".join(f"- {e}" for e in errors[:8])
                                         + "\nReturn the whole recipe again with exactly those steps fixed.")
        try:
            text = call(ask)
        except Exception as e:                                  # a provider fault is a reason, not a crash
            return {"steps": [], "error": f"{type(e).__name__}: {e}"[:200], "attempts": attempts}
        try:
            raw = llm.extract_json(text)
        except Exception:
            raw = llm.salvage_json(text) if hasattr(llm, "salvage_json") else None
        if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
            errors = ["the answer was not a JSON object with a steps list"]
            continue
        rec = {"title": str(raw.get("title") or name)[:80], "shape_class": shape_class,
               "steps": raw["steps"], "origin": "ai", "attempts": attempts}
        ok, errs = validate(rec, flat)
        bad = unsourced_numbers(rec, flat)
        if ok and not bad:
            return rec
        errors = errs + bad
    return {"steps": [], "error": "; ".join(errors[:6])[:600], "attempts": attempts}
