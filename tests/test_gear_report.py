"""Gear factor report (research 7 Nov 2026): every factor visible, and the table equals the variant's numbers."""
from dataclasses import replace
from physics.gears import Requirements
from physics.variants import generate_variants

S1 = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150,
                  wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)
S2 = Requirements(power_kw=11.0, input_rpm=6000, gear_ratio=6, env_w=250, env_h=200, env_d=180,
                  wall_thickness=4.0, gear_material="17CrNiMo6", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)


def test_report_matches_variant_numbers_for_both_methods():
    for base in (S1, S2):
        for meth in ("lewis", "iso"):
            req = replace(base, bending_method=meth)
            for v in generate_variants(req, compare_basis=False)["variants"]:
                rep = v["factor_report"]
                s = rep["summary"]
                assert abs(s["S_F"] - v["stress"]["sf_bending"]) < 2e-3, (meth, s, v["stress"])
                assert abs(s["S_H"] - v["stress"]["sf_contact"]) < 2e-3
                assert abs(s["sigma_F_mpa"] - v["stress"]["bending_mpa"]) < 0.02
                assert abs(s["sigma_H_mpa"] - v["stress"]["contact_mpa"]) < 0.02


def test_every_row_has_formula_source_and_basis():
    v = generate_variants(replace(S1, bending_method="iso"), compare_basis=False)["variants"][0]
    rep = v["factor_report"]
    rows = [r for sec in rep["sections"].values() for r in sec]
    syms = {r["symbol"] for r in rows}
    for must in ("K_A", "K_v", "K_Hβ = K_Fβ", "Z_E", "Z_H", "Z_ε", "Y_ε", "Y_F (pinion)", "Y_S (gear)", "S_F", "S_H"):
        assert must in syms, must
    for r in rows:
        assert r["formula"] and r["source"] and r["basis"] in ("computed", "assumed", "material", "not used", "standard")
    # fixed values are declared, not hidden
    assert any(a.startswith("K_A") for a in rep["assumptions"]) and any(a.startswith("K_Hβ") for a in rep["assumptions"])
    md = rep["export_markdown"]
    assert md.startswith("# Gear factor report") and "| K_v |" in md and "## Limits" in md
