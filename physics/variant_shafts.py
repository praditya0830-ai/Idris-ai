"""Shaft + bearing check for every gearbox variant (MVP priority 1).

Uses the same layout as /generate-assembly (physics.housing.build_layout) and runs the ACE
shaft solver (physics.shaft_design.design_shaft: Shigley ch.6-7 fatigue, deflection, Rayleigh
critical speed, ISO 281 bearing life) on each shaft. The torsion-only diameters that
gears.shaft_diameter_mm gives (used by the 3D model) are compared with what combined bending
+ torsion fatigue and standard bearings actually need.
"""
import math
from .gears import web_hub_diameter
from .housing import build_layout, shaft_supports_z, support_seat
from .gears import WEB_SEPTUM
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
        z_c = s["z0"] + 7.5 if s["z0"] < 0 else s["z1"] - 7.5        # middle of the 15 mm coupling seat
        coupling = z_c - origin
    spec = {"rpm": rpm_in * s["rpm_factor"], "xA_mm": za - origin, "xB_mm": zb - origin,
            "length_mm": length, "coupling_x_mm": coupling, "material": SHAFT_MATERIAL,
            "life_hours": LIFE_H, "bearing_reliability": BEARING_RELIABILITY,
            "gears": [{**{k: v for k, v in g.items() if k != "z"}, "x_mm": g["z"] - origin} for g in gears]}
    return spec


def input_rpm_of(variant, default=6000.0):
    k = variant.get("kinematics") or {}
    try:
        return float(k["output_rpm"]) * float(k["ratio_actual"])
    except (KeyError, TypeError, ValueError):
        return default


def _layout_for(variant, hp):
    return build_layout(variant["design"], variant["kinematics"], variant.get("loads") or {},
                        variant.get("shaft") or {}, hp)


def shaft_plan(variant, input_rpm=None):
    """Design every shaft + bearing on the manufacturable layout v4 and size the housing around them.

    Iterates: layout (bearing widths, web) -> shaft solver per shaft -> real bearing widths -> layout …
    until the bearings stop changing. Returns None when a shaft cannot be designed. Result:
      shafts[]: name, x, segments [(z0, z1, d)] (global z), bearings [{z, position, seat, designation,
                d, D, B}], journal_mm, seat_mm, res (solver output), spec
      hp:       housing_params of the designed housing (envelope_depth, bearing_widths, web_mm, bores …)
      L:        the final layout; notes[]: honest remarks (e.g. a bearing that does not fit)
    """
    rpm = input_rpm or input_rpm_of(variant)
    hp0 = dict(variant.get("housing_params") or {})
    depth0 = hp0.get("envelope_depth")
    widths, web = None, None
    for _ in range(5):
        hp = {k: v for k, v in hp0.items() if k != "envelope_depth"}
        if widths:
            hp["bearing_widths"] = widths
        if web:
            hp["web_mm"] = web
        L = _layout_for(variant, hp)
        results = []
        for si in range(len(L["shafts"])):
            spec = _shaft_spec(L, si, rpm)
            res = design_shaft(spec)
            if not res.get("ok"):
                return None
            results.append((spec, res))
        new_widths = [[float(res["bearings"]["A"]["B_mm"]), float(res["bearings"]["B"]["B_mm"])]
                      for _, res in results]
        new_web = (new_widths[0][1] + WEB_SEPTUM + new_widths[2][0]) if L["stages"] == 2 else None
        if new_widths == widths and new_web == web:
            break
        widths, web = new_widths, new_web
    need_D = L["D"]
    D = max(float(depth0 or 0), float(need_D))
    hp_final = dict(hp, envelope_depth=D, bearing_widths=widths)
    if web:
        hp_final["web_mm"] = web
    if D != need_D:
        L = _layout_for(variant, hp_final)
        results = [(_shaft_spec(L, si, rpm), None) for si in range(len(L["shafts"]))]
        results = [(spec, design_shaft(spec)) for spec, _ in results]
        if not all(r.get("ok") for _, r in results):
            return None
    out = []
    for si, (spec, res) in enumerate(results):
        s = L["shafts"][si]
        origin = min(shaft_supports_z(L, si) + (s["z0"], s["z1"]))
        segs = [(origin + g["x0_mm"], origin + g["x1_mm"], float(g["dia_mm"])) for g in res["geometry"]["segments"]]
        zz = shaft_supports_z(L, si)
        brg = [{"z": zz[k], "position": key, "seat": support_seat(L, si, k), "designation": v["designation"],
                "d": v["d_mm"], "D": v["D_mm"], "B": v["B_mm"]}
               for k, (key, v) in enumerate(sorted(res["bearings"].items()))]
        out.append({"name": s["name"], "x": s["x"], "segments": segs, "bearings": brg,
                    "journal_mm": res["geometry"]["journal_dia_mm"], "seat_mm": res["geometry"]["seat_dia_mm"],
                    "res": res, "spec": spec})
    notes = []
    t, W, H = L["wall_mm"], L["W"], L["H"]

    def fits(x, dia):
        r = dia / 2.0
        return x - r >= t and x + r <= W - t and L["yc"] - r >= t and L["yc"] + r <= H - t
    front = next((b["D"] for b in out[0]["bearings"] if b["seat"] == "front"), None)
    back = next((b["D"] for b in out[-1]["bearings"] if b["seat"] == "back"), None)
    if front and not fits(L["bores"][0]["x"], front):
        notes.append(f"Front bearing (OD {front:g} mm) does not fit beside the side wall at this axis.")
        front = None
    if back and not fits(L["bores"][1]["x"], back):
        notes.append(f"Back bearing (OD {back:g} mm) does not fit beside the side wall at this axis.")
        back = None
    if front:
        hp_final["front_bore_diameter"] = front
    if back:
        hp_final["back_bore_diameter"] = back
    if L["stages"] == 2:
        mid = out[1]
        z_w = 0.5 * (L["web"]["z0"] + L["web"]["z1"])
        d_at_web = max((sg[2] for sg in mid["segments"] if sg[0] <= z_w <= sg[1]), default=mid["seat_mm"])
        hp_final["web_hole_diameter"] = float(d_at_web) + 4.0
        od = max([b["D"] for b in out[0]["bearings"] if b["seat"] == "web"] +
                 [b["D"] for b in out[2]["bearings"] if b["seat"] == "web"])
        hub = web_hub_diameter(float(od))
        # the hub must clear the intermediate shaft through the web
        if abs(L["x_b"] - L["x_a"]) - hub / 2.0 - (float(d_at_web) + 4.0) / 2.0 < 2.0:
            notes.append(f"Web bearing hub Ø{hub:.0f} mm collides with the intermediate shaft.")
        hp_final["web_hub_diameter"] = round(hub, 1)
    hp_final["split_cover"] = True
    L = _layout_for(variant, hp_final)
    manufacturable = not notes
    hp_final["stickout_mm"] = list(L["stickout"])
    return {"shafts": out, "front_bore_mm": front, "back_bore_mm": back, "input_rpm": rpm, "notes": notes,
            "hp": hp_final, "L": L, "manufacturable": manufacturable, "depth_mm": D, "depth_needed_mm": need_D}


def check_variant_shafts(variant, input_rpm):
    """Returns (shafts_summary list, warnings list) for one variant dict from generate_variants."""
    plan = shaft_plan(variant, input_rpm)
    if plan is None:
        return [], ["Shaft/bearing design failed for this variant: no standard shaft + deep-groove bearing "
                    "combination meets fatigue, stiffness and L10 life."]
    L = plan["L"]
    out, warns = [], list(plan["notes"])
    for si, ps in enumerate(plan["shafts"]):
        s = L["shafts"][si]
        res, spec = ps["res"], ps["spec"]
        g = res["geometry"]
        n_fat = min(sec["n_fatigue"] for sec in res["sections"])
        seat_need = max((sec["d_required_mm"] for sec in res["sections"] if sec["name"] == "gear seat"), default=0)
        cand = {"torsion_only": float(s["d"]), "fatigue_at_gear_seat": float(seat_need),
                "bearing_journal": float(g["journal_dia_mm"])}
        gov = max(cand, key=cand.get)
        out.append({
            "name": s["name"], "rpm": round(spec["rpm"], 1), "torque_nm": round(s["torque"], 2), "ok": True,
            "torsion_only_diameter_mm": s["d"], "diameter_in_model_mm": g["seat_dia_mm"],
            "journal_diameter_mm": g["journal_dia_mm"], "seat_diameter_mm": g["seat_dia_mm"],
            "model_segments_mm": [[round(a, 2), round(b, 2), d] for a, b, d in ps["segments"]],
            "bearing_positions_z_mm": [round(b["z"], 2) for b in ps["bearings"]],
            "bearings": [{"position": b["position"], "seat": b["seat"], "designation": b["designation"],
                          "bore_mm": b["d"], "outer_mm": b["D"], "width_mm": b["B"],
                          "radial_load_n": round(res["bearings"][b["position"]]["Fr_N"], 1),
                          "life_h_L10": round(res["bearings"][b["position"]]["life_h_at_reliability"])}
                         for b in ps["bearings"]],
            "min_fatigue_safety_factor": round(n_fat, 2),
            "max_deflection_mm": round(max(c["value"] for k, c in res["checks"].items() if k.endswith("deflection")), 5),
            "critical_speed_rpm": round(res["checks"]["critical_speed"]["value"]),
            "all_checks_pass": all(c["pass"] for c in res["checks"].values()),
            "seat_diameter_required_by_strength_mm": round(seat_need, 2),
            "diameter_basis_mm": {**{k: round(v, 2) for k, v in cand.items()}, "governing": gov,
                                  "notes": {"torsion_only": "τ_allow starting estimate",
                                            "fatigue_at_gear_seat": "Shigley DE-Goodman, bending + torsion, Kf",
                                            "bearing_journal": f"bore of the bearings chosen for L10 ≥ {LIFE_H} h; "
                                                               "journal also sized for fatigue, slope, deflection"}},
            "method": "Shigley DE-Goodman fatigue + Euler-Bernoulli deflection + Rayleigh critical speed + ISO 281 L10",
        })
        if not out[-1]["all_checks_pass"]:
            warns.append(f"{s['name']}: not every shaft check passes — see shafts[].")
    return out, warns


def designed_housing_params(variant, hp=None, input_rpm=None):
    """housing_params of the manufacturable housing (layout v4) for this variant: real bearing widths,
    central web, bearing-OD bores, body + covers. Unchanged when no shaft design exists."""
    v = dict(variant)
    if hp is not None:
        v["housing_params"] = hp
    try:
        plan = shaft_plan(v, input_rpm)
    except Exception:
        plan = None
    if not plan:
        return dict(v.get("housing_params") or {})
    return dict(plan["hp"])
