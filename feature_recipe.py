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

STEP_ORDER = {"sketch": 0, "push": 1, "revolve": 1, "hole": 2, "edge": 3, "repeat": 3}
ALLOWED = tuple(STEP_ORDER)

# feature names the Studio's buildPart understands for these steps; kept in one place so the two sides can be compared
STUDIO_TYPES = {"sketch": "sketch", "push": "push", "revolve": "revolve", "hole": "hole", "edge": "edge", "repeat": "repeat"}


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


def template_for(shape_class: str, numbers: dict, holes: list | None = None, fillet=None) -> dict | None:
    """-> a recipe {title, shape_class, steps} or None when no known-good template fits this shape."""
    cls = str(shape_class or "prismatic").lower()
    if cls in ("prismatic", "sheet", "frame"):
        steps = plate_steps(numbers or {}, holes or [], fillet)
        return {"title": "Plate with holes", "shape_class": cls, "steps": steps} if steps else None
    if cls == "revolve":
        D, T = (numbers or {}).get("diameter_mm"), (numbers or {}).get("length_mm")
        if not (D and T and D > 0 and T > 0):
            return None
        return {"title": "Revolved boss", "shape_class": cls,
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
        if STEP_ORDER[kind] < order:
            errors.append(f"step '{sid}': {kind} comes after a later stage; the order is sketch, additive, holes, dress")
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
            if isinstance(e, dict) and e.get("a"):
                pts.append(tuple(e["a"]))
        if len(pts) >= 3 and _is_closed(pts):
            return True, ""
        return False, "the lines and arcs do not close: the profile must end where it started"
    return False, "the profile type is not one this build validates"


def _is_closed(pts) -> bool:
    return math.dist(pts[0], pts[-1]) < 1e-6


def _face_of(steps) -> tuple:
    """The first rectangle in the recipe: (W, Hh) or None."""
    for s in steps:
        if isinstance(s, dict) and s.get("type") == "sketch":
            p = s.get("p") or {}
            if str(p.get("prof")) == "rect":
                W, H = _num(p.get("W")), _num(p.get("Hh"))
                if W and H:
                    return W, H
    return None


def _hole_inside(p: dict, steps, numbers: dict) -> bool:
    face = _face_of(steps)
    d = _num(p.get("D")) or 0
    x, y = _num(p.get("x")) or 0, _num(p.get("y")) or 0
    if not face:
        W, H = _num(numbers.get("length_mm")), _num(numbers.get("width_mm"))
        face = (W, H) if (W and H) else None
    if not face:
        return True                              # nothing to compare against: the kernel is the next check
    W, H = face
    return abs(x) + d / 2 <= W / 2 + 1e-9 and abs(y) + d / 2 <= H / 2 + 1e-9


def _fillet_fits(r: float, steps, numbers: dict) -> bool:
    thin = _num(numbers.get("thickness_mm"))
    for s in steps:
        if isinstance(s, dict) and s.get("type") in ("push", "revolve"):
            p = s.get("p") or {}
            thin = thin or _num(p.get("dist"))
    face = _face_of(steps)
    limits = [thin] if thin else []
    if face:
        limits.append(min(face))
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
