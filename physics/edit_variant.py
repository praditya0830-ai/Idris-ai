"""POST /edit-variant: change one parameter of a variant and get a NEW, fully re-checked variant (UI drag handles).

ui/rencana-ruang-kerja-designer-dan-edit-interaktif.md, Bagian A + C#1. A shape changed only on screen is never a
result: the solver rebuilds the design with the new value and re-runs every check that /generate-variants runs
(gear strength, shafts and bearings, manufacturable housing, envelope, oil, thermal screen). The original variant
is untouched; the answer is "Variant A′" with a before/after diff.

Editable now: wall_thickness_mm, face_width_mm, housing_material, gear_material.
Not editable yet (said honestly in `allowed_range.not_editable`): outer housing size (it is derived from the
gears, bearings and wall), ribs (the housing has none yet), fillets/chamfers (need the OpenCascade kernel).
"""
from dataclasses import replace

from physics.gears import GEAR_MATERIALS, HOUSING_MATERIALS
from physics import variants as V

WALL_RANGE = (3.0, 12.0, 0.5)     # mm; 3 mm = practical minimum for machined/cast aluminium (as /fea options)
EDITABLE = ("wall_thickness_mm", "face_width_mm", "housing_material", "gear_material")


def allowed_range(variant, req):
    d = variant["design"]
    m, zp = float(d["module_mm"]), int(d["pinion_teeth"])
    lo = round(min(V.FACE_WIDTH_FACTORS) * m)
    hi = int(min(max(V.FACE_WIDTH_FACTORS) * m, V.MAX_WIDTH_TO_DIAMETER * m * zp))
    return {
        "wall_thickness_mm": {"min": WALL_RANGE[0], "max": WALL_RANGE[1], "step": WALL_RANGE[2],
                              "current": req.wall_thickness, "unit": "mm"},
        "face_width_mm": {"min": float(lo), "max": float(hi), "step": 1.0, "current": float(d["face_width_mm"]),
                          "unit": "mm", "basis": f"b/m between {min(V.FACE_WIDTH_FACTORS)} and "
                                                 f"{max(V.FACE_WIDTH_FACTORS)}, b ≤ {V.MAX_WIDTH_TO_DIAMETER}·d1"},
        "housing_material": {"options": list(HOUSING_MATERIALS), "current": req.housing_material},
        "gear_material": {"options": list(GEAR_MATERIALS), "current": req.gear_material},
        "not_editable": {
            "outer_housing_size": "derived from gears, bearings and wall; change the wall or face width instead",
            "ribs": "the housing has no ribs yet (Tahap 4: housing shape optimisation)",
            "fillet_chamfer": "needs the OpenCascade kernel on the server (Tahap 1)",
        },
    }


def _summary(v):
    if not v:
        return None
    ef = v.get("envelope_fit") or {}
    ts = v.get("thermal_screen") or {}
    st = v.get("stress") or {}
    return {"mass_total_kg": v["mass_kg"]["total"], "mass_housing_kg": v["mass_kg"].get("housing"),
            "sf_bending": st.get("sf_bending"), "sf_contact": st.get("sf_contact"),
            "physics_pct": v["scores"]["physics_pct"], "design_pct": v["scores"]["design_pct"],
            "housing_box_mm": ef.get("housing_box_mm"), "overall_box_mm": ef.get("overall_box_mm"),
            "fits_envelope": ef.get("fits"), "oil_iso_vg": (v.get("lubricant") or {}).get("iso_vg"),
            "oil_temp_c": ts.get("oil_temp_c"), "face_width_mm": v["design"]["face_width_mm"],
            "wall_thickness_mm": (v.get("housing_params") or {}).get("wall_thickness")}


def _diff(a, b):
    out = []
    for k in a:
        x, y = a[k], b.get(k)
        if x == y:
            continue
        row = {"quantity": k, "before": x, "after": y}
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            row["change"] = round(y - x, 4)
        out.append(row)
    return out


def _one(req, m, zp, b, basis, goals, ambient_c, pool):
    out = V._generate_basis(req, 1, basis, goals, ambient_c, only=(m, zp, b), score_pool=pool)
    vs = out.get("variants") or []
    return (vs[0] if vs else None), out


def edit_variant(req, variant, change, basis="housing", goals=None, ambient_c=25.0):
    """req: the Requirements the variant was generated with. Returns (http status, body)."""
    if not isinstance(change, dict) or len(change) != 1:
        return 400, {"success": False, "error": f"change must hold exactly one of {list(EDITABLE)}"}
    (key, val), = change.items()
    if key not in EDITABLE:
        return 400, {"success": False, "error": f"'{key}' cannot be edited; editable: {list(EDITABLE)}",
                     "allowed_range": allowed_range(variant, req)}
    rng = allowed_range(variant, req)
    d = variant["design"]
    m, zp, b = float(d["module_mm"]), int(d["pinion_teeth"]), float(d["face_width_mm"])
    req2, b2 = req, b
    try:
        if key in ("wall_thickness_mm", "face_width_mm"):
            val = float(val)
            r = rng[key]
            if not r["min"] - 1e-9 <= val <= r["max"] + 1e-9:
                return 422, {"success": False, "error": f"{key} {val:g} is outside {r['min']:g}–{r['max']:g} mm",
                             "allowed_range": rng}
            if key == "wall_thickness_mm":
                req2 = replace(req, wall_thickness=val)
            else:
                b2 = float(round(val))
        else:
            if val not in rng[key]["options"]:
                return 422, {"success": False, "error": f"unknown {key} '{val}'", "allowed_range": rng}
            req2 = replace(req, **{key.replace("_material", "") + "_material": val})
    except (TypeError, ValueError):
        return 400, {"success": False, "error": f"{key} must be a number"}

    pool_before = V.sweep(req)[0]
    before, _ = _one(req, m, zp, b, basis, goals, ambient_c, pool_before)
    pool_after = pool_before if req2 is req else V.sweep(req2)[0]
    after, out = _one(req2, m, zp, b2, basis, goals, ambient_c, pool_after)

    name = variant.get("name") or "Variant"
    old = rng.get(key, {}).get("current")
    label = f"{key.replace('_mm', '').replace('_', ' ')} {old} → {val}" + (" mm" if key.endswith("_mm") else "")
    body = {"success": True, "output_level": "screening", "change_applied": {key: val}, "label": label,
            "allowed_range": allowed_range(after, req2) if after else rng, "feasible": after is not None,
            "before": _summary(before), "after": _summary(after)}
    if after:
        after["name"] = name + "′"
        after["tag"] = f"Edited: {label}"
        after["edited_from"] = name
        body["variant"] = after
        body["diff"] = _diff(body["before"], body["after"]) if before else []
    else:
        body["variant"] = None
        body["reasons"] = out.get("suggestions") or ["the edited design fails a check"]
        body["diagnostics"] = out.get("diagnostics")
        body["repairs"] = out.get("repairs", [])
    return 200, body
