"""Shaft + bearing check for every gearbox variant (MVP priority 1).

Uses the same layout as /generate-assembly (physics.housing.build_layout) and runs the ACE
shaft solver (physics.shaft_design.design_shaft: Shigley ch.6-7 fatigue, deflection, Rayleigh
critical speed, ISO 281 bearing life) on each shaft. The torsion-only diameters that
gears.shaft_diameter_mm gives (used by the 3D model) are compared with what combined bending
+ torsion fatigue and standard bearings actually need.
"""
import math
from .housing import build_layout, shaft_supports_z
from .gears import ETA_STAGE
from .shaft_design import design_shaft

SHAFT_MATERIAL = "4140_QT650"
LIFE_H = 5000
BEARING_RELIABILITY = 0.90          # basic L10 rating


def _shaft_spec(L, si, rpm_in):
    s = L["shafts"][si]
    za, zb = shaft_supports_z(L, si)
    b = L["face_width_mm"]
    m = L["module_mm"]
    zp, zg = L["teeth"]
    u = zg / zp
    T_in = L["input_torque_Nm"]
    gears = []
    zs = [za, zb, s["z0"], s["z1"]]
    for me in L["meshes"]:
        T_p = T_in * (u * ETA_STAGE) ** me["stage"]
        for role, mate_role in (("pinion", "gear"), ("gear", "pinion")):
            g = L["gears"][me[role]]
            if g["shaft"] != si:
                continue
            mate = L["gears"][me[mate_role]]
            torque = -T_p if role == "pinion" else T_p * u * ETA_STAGE
            gears.append({"z": g["z0"] + b / 2, "pitch_dia_mm": g["d"], "torque_Nm": torque,
                          "face_width_mm": b, "module_mm": m,
                          "mesh_angle_deg": 0.0 if mate["x"] > g["x"] else 180.0})
    origin = min(zs)
    length = max(zs) - origin
    coupling = None
    net = sum(g["torque_Nm"] for g in gears)
    if abs(net) > 1e-6 * max(abs(g["torque_Nm"]) for g in gears):
        z_c = -11.0 if s["z0"] < 0 else L["D"] + 11.0
        coupling = z_c - origin
    spec = {"rpm": rpm_in * s["rpm_factor"], "xA_mm": za - origin, "xB_mm": zb - origin,
            "length_mm": length, "coupling_x_mm": coupling, "material": SHAFT_MATERIAL,
            "life_hours": LIFE_H, "bearing_reliability": BEARING_RELIABILITY,
            "gears": [{**{k: v for k, v in g.items() if k != "z"}, "x_mm": g["z"] - origin} for g in gears]}
    return spec


def check_variant_shafts(variant, input_rpm):
    """Returns (shafts_summary list, warnings list) for one variant dict from generate_variants."""
    L = build_layout(variant["design"], variant["kinematics"], variant.get("loads") or {},
                     variant.get("shaft") or {}, variant.get("housing_params") or {})
    out, warns = [], []
    for si, s in enumerate(L["shafts"]):
        spec = _shaft_spec(L, si, input_rpm)
        res = design_shaft(spec)
        entry = {"name": s["name"], "rpm": round(spec["rpm"], 1), "torque_nm": round(s["torque"], 2),
                 "diameter_in_model_mm": s["d"],
                 "bearing_positions_z_mm": [round(z, 2) for z in shaft_supports_z(L, si)]}
        if not res.get("ok"):
            entry.update({"ok": False, "error": " ".join(map(str, res.get("errors", [])))})
            warns.append(f"{s['name']}: shaft/bearing check failed — {entry['error']}")
            out.append(entry)
            continue
        g = res["geometry"]
        brg = res["bearings"]
        n_fat = min(sec["n_fatigue"] for sec in res["sections"])
        entry.update({
            "ok": True,
            "journal_diameter_mm": g["journal_dia_mm"], "seat_diameter_mm": g["seat_dia_mm"],
            "bearings": [{"position": k, "designation": v["designation"], "bore_mm": v["d_mm"],
                          "outer_mm": v["D_mm"], "width_mm": v["B_mm"], "radial_load_n": round(v["Fr_N"], 1),
                          "life_h_L10": round(v["life_h_at_reliability"])} for k, v in brg.items()],
            "min_fatigue_safety_factor": round(n_fat, 2),
            "max_deflection_mm": round(max(c["value"] for k, c in res["checks"].items() if k.endswith("deflection")), 5),
            "critical_speed_rpm": round(res["checks"]["critical_speed"]["value"]),
            "all_checks_pass": all(c["pass"] for c in res["checks"].values()),
            "method": "Shigley DE-Goodman fatigue + Euler-Bernoulli deflection + Rayleigh critical speed + ISO 281 L10",
        })
        seat_need = max((sec["d_required_mm"] for sec in res["sections"] if sec["name"] == "gear seat"), default=0)
        entry["seat_diameter_required_by_strength_mm"] = round(seat_need, 2)
        # QA Q-015: why this diameter — each requirement as data, and which one governs
        cand = {"torsion_only": float(s["d"]), "fatigue_at_gear_seat": float(seat_need),
                "bearing_journal": float(g["journal_dia_mm"])}
        gov = max(cand, key=cand.get)
        entry["diameter_basis_mm"] = {**{k: round(v, 2) for k, v in cand.items()}, "governing": gov,
                                      "notes": {"torsion_only": "τ_allow sizing used by the 3D model",
                                                "fatigue_at_gear_seat": "Shigley DE-Goodman, bending + torsion, Kf",
                                                "bearing_journal": f"bore of {brg['A']['designation']} chosen for "
                                                                   f"L10 ≥ {LIFE_H} h; journal also sized for "
                                                                   "fatigue, slope and deflection"}}
        # since /generate-assembly v4 the 3D model draws THIS designed stepped shaft (not the torsion-only size)
        entry["torsion_only_diameter_mm"] = s["d"]
        entry["diameter_in_model_mm"] = g["seat_dia_mm"]
        entry["model_segments_mm"] = [[g_["x0_mm"], g_["x1_mm"], g_["dia_mm"]] for g_ in g["segments"]]
        if max(g["journal_dia_mm"], seat_need) > s["d"] * 1.01:
            warns.append(f"{s['name']}: drawn as a stepped shaft — Ø{g['journal_dia_mm']:.0f} mm journals for "
                         f"{brg['A']['designation']}/{brg['B']['designation']} bearings (L10 ≥ {LIFE_H} h at "
                         f"{spec['rpm']:.0f} rpm) and a Ø{g['seat_dia_mm']:.0f} mm gear seat (bending + torsion fatigue); "
                         f"the Ø{s['d']:.0f} mm torsion-only size is only the starting estimate.")
        out.append(entry)
    if L["stages"] == 2:
        warns.append("Shaft check assumes a pilot bearing between the coaxial input and output shafts "
                     "(reverted layout) and bearings in both housing walls for the intermediate shaft.")
    return out, warns


def input_rpm_of(variant, default=6000.0):
    k = variant.get("kinematics") or {}
    try:
        return float(k["output_rpm"]) * float(k["ratio_actual"])
    except (KeyError, TypeError, ValueError):
        return default


def shaft_plan(variant, input_rpm=None):
    """Designed shaft geometry for the 3D model and the analyses (QA Q-015 / Q-001).

    Returns {"shafts": [{name, x, segments: [(z0, z1, d)], bearings: [{z, designation, d, D, B}],
             journal_mm, seat_mm}], "front_bore_mm", "back_bore_mm"} in the housing frame, or None when
    the shaft solver cannot design a shaft (the callers then keep the torsion-only model).
    Bores in the end walls are sized to the OD of the bearing that sits there."""
    rpm = input_rpm or input_rpm_of(variant)
    L = build_layout(variant["design"], variant["kinematics"], variant.get("loads") or {},
                     variant.get("shaft") or {}, variant.get("housing_params") or {})
    out = []
    for si, s in enumerate(L["shafts"]):
        spec = _shaft_spec(L, si, rpm)
        res = design_shaft(spec)
        if not res.get("ok"):
            return None
        origin = min(shaft_supports_z(L, si) + (s["z0"], s["z1"]))
        segs = [(origin + g["x0_mm"], origin + g["x1_mm"], float(g["dia_mm"])) for g in res["geometry"]["segments"]]
        za, zb = shaft_supports_z(L, si)
        brg = [{"z": z, "position": k, "designation": v["designation"], "d": v["d_mm"], "D": v["D_mm"],
                "B": v["B_mm"]} for z, (k, v) in zip((za, zb), sorted(res["bearings"].items()))]
        out.append({"name": s["name"], "x": s["x"], "segments": segs, "bearings": brg,
                    "journal_mm": res["geometry"]["journal_dia_mm"], "seat_mm": res["geometry"]["seat_dia_mm"]})
    t, D = L["wall_mm"], L["D"]
    front = None
    back = None
    # through-bores: the input shaft in the front wall, the output shaft in the back wall
    for sh in out:
        for b in sh["bearings"]:
            if sh["name"] == "Input shaft" and abs(b["z"] - t / 2) < 1e-6:
                front = b["D"]
            if sh["name"] == "Output shaft" and abs(b["z"] - (D - t / 2)) < 1e-6:
                back = b["D"]
    # a bearing seat must fit between the walls; otherwise keep the clearance hole and say so
    notes = []
    W, H = L["W"], L["H"]

    def fits(x, dia):
        r = dia / 2.0
        return x - r >= t and x + r <= W - t and L["yc"] - r >= t and L["yc"] + r <= H - t
    x_front = L["bores"][0]["x"]
    x_back = L["bores"][1]["x"]
    if front and not fits(x_front, front):
        notes.append(f"Front bearing (OD {front:g} mm) does not fit beside the wall at this axis: the housing must "
                     "grow there or the bearing needs an external flange housing — drawn as a clearance hole.")
        front = None
    if back and not fits(x_back, back):
        notes.append(f"Back bearing (OD {back:g} mm) does not fit beside the wall at this axis: the housing must "
                     "grow there or the bearing needs an external flange housing — drawn as a clearance hole.")
        back = None
    return {"shafts": out, "front_bore_mm": front, "back_bore_mm": back, "input_rpm": rpm, "notes": notes}


def designed_housing_params(variant, hp=None, input_rpm=None):
    """housing_params + the bores (bearing OD) and end cover of /generate-assembly v4, so every
    analysis meshes the same housing that the 3D model shows. Unchanged when no shaft design exists."""
    hp = dict(hp if hp is not None else (variant.get("housing_params") or {}))
    try:
        plan = shaft_plan({**{k: variant.get(k) for k in ("design", "kinematics", "loads", "shaft")},
                           "housing_params": hp}, input_rpm)
    except Exception:
        plan = None
    if not plan:
        return hp
    hp["split_cover"] = True
    if plan.get("front_bore_mm"):
        hp["front_bore_diameter"] = plan["front_bore_mm"]
    if plan.get("back_bore_mm"):
        hp["back_bore_diameter"] = plan["back_bore_mm"]
    return hp
