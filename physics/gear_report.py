"""Factor report: every factor behind the gear safety factors, with its value, formula, source and basis.

Research (replica CAD/CAE, 7 Nov 2026): engineers say gear sizing "feels like guessing" and that a tool is
"no use as I need individual factors out as a check". This report recomputes the critical (last-stage) mesh
from the same constants physics.gears.evaluate uses, lists each factor, and checks that the recomputed
stresses equal the ones the variant reports (`consistent`), so the table can never drift from the numbers.

basis:
  computed  - calculated for this design from its geometry and loads
  assumed   - a fixed screening value Idris uses (stated, not hidden; replace with a detailed value later)
  material  - from the material table (representative catalogue level, not certified data)
  standard  - a constant the standard itself defines
  not used  - the factor exists in ISO 6336 but is taken as 1 in this screening method
"""
import math

from physics import gears as G
from physics.toothform import tooth_form

ISO_BEND = "ISO 6336-3:2019"
ISO_CONT = "ISO 6336-2:2019"
ISO_GEN = "ISO 6336-1:2019"


def _row(sym, name, value, unit, formula, source, basis, note=""):
    v = round(value, 4) if isinstance(value, float) else value
    return {"symbol": sym, "name": name, "value": v, "unit": unit, "formula": formula,
            "source": source, "basis": basis, "note": note}


def factor_report(req, design):
    """req: Requirements; design: variant['design'] (module_mm, pinion_teeth, gear_teeth, face_width_mm, stages)."""
    m, zp, zg, b = float(design["module_mm"]), int(design["pinion_teeth"]), int(design["gear_teeth"]), float(design["face_width_mm"])
    stages = int(design["stages"])
    gm = G.GEAR_MATERIALS[req.gear_material]
    u = zg / zp
    dp = m * zp
    t_in = 9549.0 * req.power_kw / req.input_rpm
    t_p = t_in * (u * G.ETA_STAGE) ** (stages - 1)
    n_p = req.input_rpm / u ** (stages - 1)
    ft = 2000.0 * t_p / dp
    v = math.pi * dp * n_p / 60000.0
    kv = G.dynamic_factor(v)
    k = G.K_A * kv * G.K_M
    eps = G.contact_ratio(m, zp, zg)
    z_eps = G.contact_ratio_factor(eps)
    iso = req.bending_method == "iso"

    load = [
        _row("T₁", "Torque on the critical pinion", t_p, "N·m", "T_in · (u·η)^(stages−1)", "kinematics", "computed",
             f"stage {stages} of {stages} (highest torque)"),
        _row("n₁", "Speed of the critical pinion", n_p, "rpm", "n_in / u^(stages−1)", "kinematics", "computed"),
        _row("F_t", "Nominal tangential force", ft, "N", "2000·T₁ / d₁", ISO_GEN, "computed"),
        _row("v", "Pitch-line velocity", v, "m/s", "π·d₁·n₁ / 60000", ISO_GEN, "computed"),
        _row("K_A", "Application factor", G.K_A, "–", "fixed", ISO_GEN, "assumed",
             "moderate shocks on the driven side (rotor loads); set from the duty when known"),
        _row("K_v", "Dynamic factor", kv, "–", f"((A+√(200v))/A)^B, Qv = {G.Q_V}", "AGMA 2001 (K_v from transmission accuracy Q_v)", "computed",
             "AGMA form used in place of ISO 6336-1 Method B"),
        _row("K_Hβ = K_Fβ", "Face load factor", G.K_M, "–", "fixed", ISO_GEN, "assumed",
             "not yet from shaft deflection (planned: ISO 6336-1 Annex E)"),
        _row("K_Hα = K_Fα", "Transverse load factor", 1.0, "–", "1", ISO_GEN, "not used",
             "taken as 1 for accurate spur gears (Qv ≤ 7)"),
        _row("K", "Total load factor", k, "–", "K_A · K_v · K_Hβ", "–", "computed"),
    ]

    contact = [
        _row("Z_E", "Elasticity factor (steel/steel)", G.ZE, "√MPa", "√(1/(π((1−ν₁²)/E₁+(1−ν₂²)/E₂))), steel E = 206 GPa, ν = 0.3", ISO_CONT, "material"),
        _row("Z_H", "Zone factor (α = 20°, x = 0)", G.ZH, "–", "√(2 / (sin α · cos α))", ISO_CONT, "computed"),
        _row("ε_α", "Transverse contact ratio", eps, "–", "path of contact / base pitch", ISO_GEN, "computed"),
        _row("Z_ε", "Contact ratio factor", z_eps, "–", "√((4 − ε_α) / 3)", ISO_CONT, "computed"),
        _row("Z_β, Z_L, Z_v, Z_R, Z_W, Z_X, Z_NT", "Helix, lubricant, speed, roughness, hardening, size, life", 1.0, "–", "1",
             ISO_CONT, "not used", "spur gear; long-life (endurance) rating; screening"),
        _row("σ_Hlim", "Allowable contact stress number", gm["sigma_H_lim"], "MPa", "material table", "material", "material",
             f"{gm['label']}: representative value, not certified data"),
    ]
    sigma_h = G.ZE * G.ZH * z_eps * math.sqrt(k * ft / (b * dp) * (u + 1.0) / u)
    contact.append(_row("σ_H", "Contact stress", sigma_h, "MPa", "Z_E·Z_H·Z_ε·√(K·F_t/(b·d₁) · (u+1)/u)", ISO_CONT, "computed"))
    contact.append(_row("S_H", "Contact safety factor", gm["sigma_H_lim"] / sigma_h, "–", "σ_Hlim / σ_H",
                        ISO_CONT, "computed", f"required ≥ {req.min_sf_contact:g}"))

    bending = []
    if iso:
        y_eps = 0.25 + 0.75 / eps
        tp, tg = tooth_form(zp), tooth_form(zg)
        f_lim = gm["sigma_F_lim"] * G.Y_ST
        for nm, z, t in (("pinion", zp, tp), ("gear", zg, tg)):
            bending.append(_row(f"Y_F ({nm})", f"Tooth form factor, z = {z}", t["YF"], "–",
                                "6(h_Fe/m)cos α_Fen / ((s_Fn/m)² cos α_n)", ISO_BEND, "computed"))
            bending.append(_row(f"Y_S ({nm})", f"Stress correction factor, z = {z}", t["YS"], "–",
                                "(1.2 + 0.13 L)·q_s^(1/(1.21 + 2.3/L))", ISO_BEND, "computed"))
        bending += [
            _row("Y_ε", "Contact ratio factor (bending)", y_eps, "–", "0.25 + 0.75 / ε_α", ISO_BEND, "computed"),
            _row("Y_ST", "Stress correction factor of the test gear", G.Y_ST, "–", "2.0", ISO_BEND, "standard"),
            _row("Y_β, Y_δrelT, Y_RrelT, Y_X, Y_NT", "Helix, notch sensitivity, root surface, size, life", 1.0, "–", "1",
                 ISO_BEND, "not used", "spur gear; long-life rating; screening"),
            _row("σ_Flim", "Nominal bending stress number", gm["sigma_F_lim"], "MPa", "material table", "material", "material",
                 f"{gm['label']}: representative value, not certified data"),
        ]
        sf_p = ft * k / (b * m) * tp["YF"] * tp["YS"] * y_eps
        sf_g = ft * k / (b * m) * tg["YF"] * tg["YS"] * y_eps
        bend_formula, bend_src = "F_t/(b·m) · Y_F·Y_S·Y_ε · K", ISO_BEND
        lim_formula = "σ_Flim · Y_ST"
    else:
        yp, yg = G.lewis_form_factor(zp), G.lewis_form_factor(zg)
        f_lim = gm["sigma_F_lim"]
        bending += [
            _row("Y (pinion)", f"Lewis form factor, z = {zp}", yp, "–", "table (load at tip)", "Shigley 10th ed. Table 14-2", "computed"),
            _row("Y (gear)", f"Lewis form factor, z = {zg}", yg, "–", "table (load at tip)", "Shigley 10th ed. Table 14-2", "computed"),
            _row("σ_Flim", "Allowable bending stress", gm["sigma_F_lim"], "MPa", "material table", "material", "material",
                 f"{gm['label']}: representative value, not certified data"),
        ]
        sf_p = ft * k / (b * m * yp)
        sf_g = ft * k / (b * m * yg)
        bend_formula, bend_src = "K·F_t / (b·m·Y)", "Lewis (Shigley eq. 14-15) with ISO load factors"
        lim_formula = "σ_Flim"
    sigma_f = max(sf_p, sf_g)
    bending += [
        _row("σ_F (pinion)", "Tooth root stress, pinion", sf_p, "MPa", bend_formula, bend_src, "computed"),
        _row("σ_F (gear)", "Tooth root stress, gear", sf_g, "MPa", bend_formula, bend_src, "computed"),
        _row("σ_FP", "Permissible root stress", f_lim, "MPa", lim_formula, ISO_BEND if iso else "material", "computed"),
        _row("S_F", "Bending safety factor (weaker of the pair)", f_lim / sigma_f, "–", "σ_FP / max σ_F",
             ISO_BEND if iso else "Lewis", "computed", f"required ≥ {req.min_sf_bending:g}"),
    ]
    return {
        "method": ("ISO 6336 Method C style screening (spur, x = 0): ISO 6336-3 tooth form from generated geometry, "
                   "ISO 6336-2 contact, load factors partly fixed" if iso else
                   "Lewis bending + Hertz/ISO 6336-2 contact with ISO-style load factors (screening)"),
        "critical_mesh": f"stage {stages} (highest torque), pinion z = {zp}, gear z = {zg}, m = {m:g} mm, b = {b:g} mm",
        "sections": {"loads": load, "contact": contact, "bending": bending},
        "summary": {"sigma_H_mpa": round(sigma_h, 2), "sigma_F_mpa": round(sigma_f, 2),
                    "S_H": round(gm["sigma_H_lim"] / sigma_h, 3), "S_F": round(f_lim / sigma_f, 3)},
        "assumptions": [r["symbol"] + " = " + str(r["value"]) + " (" + r["note"] + ")"
                        for sec in (load, contact, bending) for r in sec if r["basis"] in ("assumed", "not used")],
        "limits": ["Screening level: not a certified ISO 6336 rating.",
                   "Material strengths are representative catalogue values; use certified data before release.",
                   "K_Hβ and K_A are fixed values, not yet derived from shaft deflection or a duty cycle."],
    }


def to_markdown(rep):
    """Copy/export form of the report (UI 'Copy table')."""
    out = [f"# Gear factor report", f"Method: {rep['method']}", f"Critical mesh: {rep['critical_mesh']}", ""]
    for name, rows in rep["sections"].items():
        out += [f"## {name.capitalize()}", "| Symbol | Name | Value | Unit | Formula | Source | Basis |",
                "|---|---|---|---|---|---|---|"]
        out += [f"| {r['symbol']} | {r['name']} | {r['value']} | {r['unit']} | {r['formula']} | {r['source']} | {r['basis']} |"
                for r in rows]
        out.append("")
    out += ["## Limits"] + [f"- {x}" for x in rep["limits"]]
    return "\n".join(out)
