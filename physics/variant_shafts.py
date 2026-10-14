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
        if seat_need > s["d"] * 1.01:
            warns.append(f"{s['name']}: fatigue under combined bending + torsion needs Ø{seat_need:.1f} mm at the gear "
                         f"seat, more than the Ø{s['d']:.0f} mm torsion-only diameter in the model.")
        need = max(g["journal_dia_mm"], 0)
        if need > s["d"] * 1.01:
            warns.append(f"{s['name']}: the Ø{s['d']:.0f} mm torsion-only diameter is too small at the bearings — "
                         f"combined bending + torsion fatigue and a {brg['A']['designation']} bearing "
                         f"(L10 ≥ {LIFE_H} h at {spec['rpm']:.0f} rpm) need Ø{need:.0f} mm journals.")
        out.append(entry)
    if L["stages"] == 2:
        warns.append("Shaft check assumes a pilot bearing between the coaxial input and output shafts "
                     "(reverted layout) and bearings in both housing walls for the intermediate shaft.")
    return out, warns
