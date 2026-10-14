"""HTTP glue for the shaft + bearing solver (stdlib only).

In server.py, inside do_POST, add a branch:

    from shaft_api import handle_design_shaft
    ...
    elif self.path == "/design-shaft":
        result = handle_design_shaft(payload)   # payload = parsed JSON body
        # send `result` as JSON with status 200 (or 400 if not result["ok"])

Body option 1 (straight from a gear-stage result):
    {"gear_stage": {"input_torque_Nm": 20, "input_rpm": 6000,
                    "pitch_dia_pinion_mm": 28, "pitch_dia_gear_mm": 112,
                    "face_width_mm": 24.5, "module_mm": 1.75},
     "options": {"material": "4140_QT650", "life_hours": 5000}}

Body option 2 (full manual spec): see physics/shaft_design.design_shaft.
"""
from physics.shaft_design import design_shaft, shafts_for_gear_stage, ENGINE
from physics.shaft import SHAFT_MATERIALS

ALLOWED_OPTIONS = {"material", "finish", "reliability", "design_factor", "criterion",
                   "life_hours", "bearing_reliability", "application_factor",
                   "bearing_series", "torque_alternating_fraction", "axial_load_N"}


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) < 1e9


def handle_design_shaft(payload):
    if not isinstance(payload, dict):
        return {"ok": False, "engine": ENGINE, "errors": ["body must be a JSON object"]}
    opts = payload.get("options") or {}
    bad = [k for k in opts if k not in ALLOWED_OPTIONS]
    if bad:
        return {"ok": False, "engine": ENGINE, "errors": [f"unknown options {bad}"]}
    if opts.get("material") and opts["material"] not in SHAFT_MATERIALS:
        return {"ok": False, "engine": ENGINE,
                "errors": [f"material must be one of {list(SHAFT_MATERIALS)}"]}
    try:
        if "gear_stage" in payload:
            st = payload["gear_stage"]
            for k in ("input_torque_Nm", "input_rpm", "pitch_dia_pinion_mm", "pitch_dia_gear_mm"):
                if not _num(st.get(k)) or st[k] <= 0:
                    return {"ok": False, "engine": ENGINE, "errors": [f"gear_stage.{k} must be a positive number"]}
            return shafts_for_gear_stage(st, **opts)
        return design_shaft({**payload, **opts})
    except Exception as ex:  # never leak a stack trace to the UI
        return {"ok": False, "engine": ENGINE, "errors": [f"solver error: {type(ex).__name__}: {ex}"]}


def materials():
    return {"engine": ENGINE, "materials": SHAFT_MATERIALS}
