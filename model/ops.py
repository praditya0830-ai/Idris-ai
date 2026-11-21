"""Feature operations and their schemas (Fase 1, pilar B).

Same policy as the solver API (ace_core/schema.py): a field that is not in the schema is an error, so
an AI inventing a parameter fails loudly instead of producing a shape nobody asked for. Dimensions
may be numbers or expressions over `dims`; an expression is parsed by model.expr, never eval().

`validate_params` returns the parameters with every expression already resolved to a number, plus the
original text under "exprs" so the report can show what the engineer actually typed.
"""
from __future__ import annotations

from ace_core.schema import validate as _validate

from . import expr as E

PLANES = ["XY", "XZ", "YZ"]


def _e(required=False, default=None, desc=""):
    rule = {"expr": True, "desc": desc}
    if required:
        rule["required"] = True
    if default is not None:
        rule["default"] = default
    return rule


SCHEMAS = {
    "sketch": {
        "plane": {"type": "string", "enum": PLANES, "default": "XY", "desc": "sketch plane"},
        "profile": {"type": "string", "default": "rectangle",
                    "enum": ["rectangle", "circle", "polygon", "triangle", "lshape"],
                    "desc": "rectangle/lshape sit in (0,0)..(w,h); circle/polygon/triangle are centred on the origin"},
        "w": _e(True, desc="width / diameter / circumscribed diameter"),
        "h": _e(False, desc="height (rectangle/lshape) or width of the leg"),
        "n": {"type": "integer", "min": 3, "max": 64, "default": 6,
              "desc": "sides of a polygon (triangle ignores it)"},
        "thickness": _e(False, desc="leg thickness of an lshape (default min(w,h)/3)"),
        "offset": _e(False, desc="distance from the plane along its normal: lets a sketch sit inside the model"),
        "cx": _e(False, desc="profile centre along the plane's first axis (X2); absent = on the plane origin"),
        "cy": _e(False, desc="profile centre along the plane's second axis (X2)"),
    },
    "extrude": {
        "distance": _e(True),
        "direction": {"type": "string", "enum": ["+normal", "-normal", "symmetric"], "default": "+normal"},
        "draft_deg": _e(False, 0.0),
        "op": {"type": "string", "enum": ["add", "cut"], "default": "add",
               "desc": "add = boss, cut = remove material (X2: a FeatureCut4, not only a boss)"},
        "through": {"type": "boolean", "default": False,
                    "desc": "a cut that goes through the whole body, instead of `distance` deep (X2)"},
    },
    "revolve": {
        "angle_deg": _e(False, 360.0),
        "axis": {"type": "string", "enum": ["x", "y", "z"], "default": "z"},
    },
    "hole": {
        "d": _e(True, desc="hole diameter"),
        "x": _e(True), "y": _e(True),
        "through": {"type": "boolean", "default": True},
        "depth": _e(False),
        "from": {"type": "string", "enum": ["sketch", "top"], "default": "sketch",
                 "desc": "where the bore starts: the sketch plane, or the top face at `z_top` (X2)"},
        "z_top": _e(False, desc="height of the face a hole from 'top' starts on (X2, needed by from: 'top')"),
        "cbore_d": _e(False, desc="counterbore diameter (X2); needs cbore_depth as well"),
        "cbore_depth": _e(False, desc="counterbore depth (X2)"),
    },
    "fillet": {
        "radius": _e(True),
        "edges": {"type": "string", "enum": ["all", "vertical", "horizontal", "top", "bottom"], "default": "all"},
    },
    "pattern": {
        "mode": {"type": "string", "enum": ["linear", "circular"], "default": "linear"},
        "count": {"type": "integer", "min": 1, "max": 500, "default": 2},
        "dx": _e(False, 0.0), "dy": _e(False, 0.0), "angle_deg": _e(False, 0.0),
    },
    # X2: the two features the SolidWorks macro could not rebuild before
    "chamfer": {
        "distance": _e(True, desc="equal-distance chamfer leg"),
        "edges": {"type": "string", "enum": ["all", "vertical", "horizontal", "top", "bottom"], "default": "all"},
    },
    "mirror": {
        "plane": {"type": "string", "enum": PLANES, "default": "XY", "desc": "mirror plane"},
        "feature": {"type": "string", "desc": "id of the feature to mirror; the one before it by default"},
    },
    # a note the Studio sends for a feature it knows the macro cannot rebuild: validation must accept it
    "not_translated": {
        "what": {"type": "string", "desc": "the feature as the engineer sees it"},
        "why": {"type": "string", "desc": "why the macro cannot rebuild it"},
    },
    "import": {
        "path": {"type": "string", "required": True},
        "format": {"type": "string", "enum": ["step", "iges", "brep"], "default": "step"},
        "recognise": {"type": "boolean", "default": True},
    },
    # Migration door: AI-generated CAD code is ONE node with ONE hash and no partial regeneration.
    "opaque": {
        "code": {"type": "string", "required": True},
        "kernel": {"type": "string", "enum": ["replicad", "occt"], "default": "replicad"},
        "hash": {"type": "string"},
    },
}

OPS = tuple(SCHEMAS)


def validate_params(op: str, params, dims: dict):
    """-> (clean_params, errors). Resolved numbers for every expression field."""
    if op not in SCHEMAS:
        return {}, [f"unknown op {op!r}; known: {list(OPS)}"]
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {}, [f"{op}: params must be an object"]

    schema = SCHEMAS[op]
    structural = {k: v for k, v in schema.items() if not v.get("expr")}
    expr_fields = {k: v for k, v in schema.items() if v.get("expr")}

    # only the structural fields go to the shared validator; expression fields are resolved below
    structural_data = {k: params[k] for k in structural if k in params}
    clean, errors = _validate(structural, structural_data) if structural else ({}, [])
    errors = list(errors)
    unknown = [k for k in params if k not in schema]
    if unknown:
        errors.append(f"{op}: unknown field(s) {unknown}; allowed: {sorted(schema)}")

    resolved, texts = {}, {}
    for name, rule in expr_fields.items():
        if name not in params or params[name] is None:
            if rule.get("required"):
                errors.append(f"{op}.{name}: required")
            elif "default" in rule:
                resolved[name] = float(rule["default"])
            continue
        raw = params[name]
        try:
            value = E.evaluate(raw, dims)
        except E.ExprError as exc:
            errors.append(f"{op}.{name}: {exc}")
            continue
        if value <= 0 and name not in ("draft_deg", "dx", "dy", "angle_deg", "offset", "cx", "cy", "z_top"):
            errors.append(f"{op}.{name}: must be > 0, got {value}")
            continue
        resolved[name] = value
        if isinstance(raw, str):
            texts[name] = raw

    merged = dict(clean)
    merged.update(resolved)
    # X2: a field that only makes sense with another one is an error, not a silent default
    if op == "hole" and merged.get("from") == "top" and "z_top" not in merged:
        errors.append("hole.z_top: a hole from 'top' needs z_top, the height of the face it starts on")
    if op == "hole" and (("cbore_d" in merged) != ("cbore_depth" in merged)):
        errors.append("hole: a counterbore needs both cbore_d and cbore_depth")
    if texts:
        merged["exprs"] = texts
    if "path" in schema and params.get("path") is not None:
        errors += _path_errors(str(params["path"]))
    return merged, errors


def _path_errors(path: str) -> list:
    """No absolute paths, no traversal: an import may only read inside the parts directory."""
    import os
    if os.path.isabs(path) or ".." in path.replace("\\", "/").split("/"):
        return [f"import.path: {path!r} must be a relative path inside the parts directory"]
    return []


def default_params(op: str) -> dict:
    if op not in SCHEMAS:
        raise KeyError(op)
    return {k: v["default"] for k, v in SCHEMAS[op].items() if "default" in v}
