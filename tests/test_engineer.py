"""The bench may only produce numbers a formula produced, and the gate must catch the rest.

Each expectation below is an independent closed-form value (or a table value quoted with its source),
not a copy of what the module computes.
"""
import math

import pytest

from ai import engineer as ENG


def test_rectangle_section_matches_bh3_over_12():
    s = ENG.section("rect", b_mm=20, h_mm=10)
    assert s["A_mm2"] == pytest.approx(200)
    assert s["I_mm4"] == pytest.approx(20 * 10 ** 3 / 12)
    assert s["c_mm"] == pytest.approx(5)
    assert s["r_mm"] == pytest.approx(math.sqrt(20 * 1000 / 12 / 200))


def test_tube_section_uses_od_and_id():
    s = ENG.section("tube", od_mm=20, id_mm=16)
    assert s["A_mm2"] == pytest.approx(math.pi * (400 - 256) / 4)
    assert s["I_mm4"] == pytest.approx(math.pi * (20 ** 4 - 16 ** 4) / 64)


def test_solid_round_section():
    s = ENG.section("circle", d_mm=10)
    assert s["I_mm4"] == pytest.approx(math.pi * 10 ** 4 / 64)


def test_tube_rejects_inner_larger_than_outer():
    with pytest.raises(ValueError):
        ENG.section("tube", od_mm=10, id_mm=12)


def test_cantilever_point_load_matches_fl3_over_3ei():
    r = ENG.beam("cantilever_point", L_mm=500, E_mpa=200_000, I_mm4=10_000, c_mm=10, F_n=100)
    assert r["M_nmm"] == pytest.approx(100 * 500)
    assert r["delta_mm"] == pytest.approx(100 * 500 ** 3 / (3 * 200_000 * 10_000))
    assert r["sigma_mpa"] == pytest.approx(100 * 500 / 10_000 * 10)


def test_simply_supported_udl_matches_5wl4_over_384ei():
    r = ENG.beam("simple_udl", L_mm=1000, E_mpa=200_000, I_mm4=10_000, c_mm=10, w_n_per_mm=1.0)
    assert r["M_nmm"] == pytest.approx(1.0 * 1000 ** 2 / 8)
    assert r["delta_mm"] == pytest.approx(5 * 1.0 * 1000 ** 4 / (384 * 200_000 * 10_000))


def test_beam_deflection_check_uses_the_stated_limit():
    r = ENG.beam("simple_point_center", L_mm=1000, E_mpa=200_000, I_mm4=10_000, c_mm=10,
                 F_n=1000, allow_delta_mm=2.0)
    assert r["delta_ok"] is (abs(r["delta_mm"]) <= 2.0)


def test_beam_without_load_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        ENG.beam("cantilever_point", L_mm=500, E_mpa=200_000, I_mm4=10_000, c_mm=10)


def test_euler_and_johnson_agree_at_the_transition_slenderness():
    E, sy, A, I = 200_000.0, 300.0, 100.0, 10_000.0
    r_gyr = math.sqrt(I / A)
    cc = math.sqrt(2 * math.pi ** 2 * E / sy)
    L = cc * r_gyr
    euler = math.pi ** 2 * E * I / L ** 2
    johnson = sy * A * (1 - sy * L ** 2 / (4 * math.pi ** 2 * E * r_gyr ** 2))
    assert euler == pytest.approx(johnson, rel=1e-9)


def test_long_column_reports_euler_and_short_column_johnson():
    E, sy, A, I = 200_000.0, 300.0, 100.0, 10_000.0
    long_ = ENG.column(L_mm=5000, E_mpa=E, I_mm4=I, A_mm2=A, sy_mpa=sy)
    short = ENG.column(L_mm=100, E_mpa=E, I_mm4=I, A_mm2=A, sy_mpa=sy)
    assert long_["regime"] == "euler" and short["regime"] == "johnson"
    assert long_["Pcr_n"] == pytest.approx(math.pi ** 2 * E * I / 5000 ** 2)


def test_m10_88_tensile_area_and_preload():
    r = ENG.bolt(10, 1.5, "8.8")
    at = math.pi / 4 * (10 - 0.9382 * 1.5) ** 2
    assert r["At_mm2"] == pytest.approx(at)
    assert r["At_mm2"] == pytest.approx(58.0, abs=0.2)          # Shigley table 8-1: At = 58.0 mm2
    assert r["Fi_n"] == pytest.approx(0.75 * at * 600)
    assert r["torque_nm"] == pytest.approx(0.2 * r["Fi_n"] * 10 / 1000)


def test_bolt_separation_load_is_preload_over_stiffness_ratio():
    r = ENG.bolt(10, 1.5, "8.8", external_n=5000, joint_stiffness_ratio=0.2)
    assert r["separation_n"] == pytest.approx(r["Fi_n"] / 0.2)
    assert r["Fb_n"] == pytest.approx(r["Fi_n"] + 0.2 * 5000)
    assert r["stays_clamped"] is True


def test_unknown_property_class_is_refused():
    with pytest.raises(ValueError):
        ENG.bolt(10, 1.5, "9.9")


def test_steel_expands_0585_mm_per_metre_at_50k():
    r = ENG.thermal(L_mm=1000, delta_t_k=50, alpha_1_k=11.7e-6)
    assert r["delta_L_mm"] == pytest.approx(0.585, abs=1e-9)


def test_restrained_thermal_stress_is_e_alpha_dt():
    r = ENG.thermal(L_mm=1000, delta_t_k=50, alpha_1_k=11.7e-6, E_mpa=205_000, constrained=True)
    assert r["sigma_mpa"] == pytest.approx(205_000 * 11.7e-6 * 50)


@pytest.mark.parametrize("formula,expected", [
    ("H2O", 18.015), ("CO2", 44.009), ("Ca(OH)2", 74.093),
    ("CuSO4·5H2O", 249.68), ("C6H12O6", 180.156),
])
def test_molar_mass_against_standard_atomic_weights(formula, expected):
    assert ENG.molar_mass(formula)["molar_mass_g_mol"] == pytest.approx(expected, abs=0.02)


def test_unknown_element_is_refused():
    with pytest.raises(ValueError):
        ENG.molar_mass("Xx2O")


def test_ph_of_strong_acid_and_base():
    assert ENG.solution_ph(0.01)["pH"] == pytest.approx(2.0)
    assert ENG.solution_ph(0.01, base=True)["pH"] == pytest.approx(12.0)


def test_ideal_gas_one_mole_at_stp_is_22414_litres():
    r = ENG.ideal_gas(n_mol=1, T_k=273.15, p_pa=101_325)
    assert r["v_m3"] == pytest.approx(0.022414, abs=1e-6)


def test_arithmetic_respects_precedence_and_caret():
    assert ENG.evaluate("2+3*4") == 14
    assert ENG.evaluate("2^10") == 1024
    assert ENG.evaluate("sqrt(16)+pi") == pytest.approx(4 + math.pi)


def test_expression_with_a_name_is_refused():
    with pytest.raises((ValueError, SyntaxError)):
        ENG.evaluate("__import__('os').system('ls')")


def test_bisection_finds_the_known_root():
    r = ENG.root("x**2-4", 0, 10)
    assert r["x"] == pytest.approx(2.0, abs=1e-6)


def test_beam_solve_names_the_missing_load_instead_of_assuming_one():
    out = ENG.solve("tegangan balok pipa", given={"section": "tube", "od_mm": 20, "id_mm": 16,
                                                  "L_mm": 500, "material": ENG.material("6061")})
    assert out["claims"] == []
    assert any(m["input"] == "F_n" for m in out["missing"])


def test_beam_solve_with_all_inputs_produces_the_closed_form_numbers():
    mat = ENG.material("steel_1018")
    out = ENG.solve("", given={"topic": "beam", "section": "rect", "b_mm": 20, "h_mm": 10,
                               "L_mm": 500, "F_n": 100, "material": mat, "case": "cantilever_point"})
    got = {c["id"]: c["value"] for c in out["claims"]}
    I = 20 * 10 ** 3 / 12                       # b·h³/12 for a 20 x 10 section
    assert got["beam.M"] == pytest.approx(100 * 500)
    assert got["beam.sigma"] == pytest.approx(100 * 500 * 5 / I)      # 150 MPa
    assert got["beam.delta"] == pytest.approx(100 * 500 ** 3 / (3 * mat["E_mpa"] * I), rel=1e-9)
    assert all(c["formula"] and c["source"] for c in out["claims"])


def test_bolt_solve_reads_the_size_from_the_sentence():
    out = ENG.solve("berapa torsi baut M10 kelas 8.8")
    got = {c["id"]: c["value"] for c in out["claims"]}
    assert got["bolt.At"] == pytest.approx(58.0, abs=0.2)
    assert got["bolt.torque"] > 0


def test_chemistry_solve_reads_the_formula_from_the_sentence():
    out = ENG.solve("massa molar Ca(OH)2 berapa")
    got = {c["id"]: c["value"] for c in out["claims"]}
    assert got["chem.M"] == pytest.approx(74.093, abs=0.02)


def test_empty_request_yields_no_claims():
    out = ENG.solve("")
    assert out["claims"] == [] and out["notes"]


def test_gate_accepts_a_number_that_came_from_the_bench():
    eng = {"claims": [{"id": "beam.sigma", "value": 50.0}]}
    assert ENG.verify("Tegangannya 50 MPa, masih aman.", eng)["ok"] is True


def test_gate_flags_an_invented_number_with_its_phrase():
    eng = {"claims": [{"id": "beam.sigma", "value": 50.0}]}
    v = ENG.verify("Tegangannya 62 MPa, jadi aman.", eng)
    assert v["ok"] is False
    assert v["unverified"][0]["value"] == 62
    assert "62" in v["unverified"][0]["phrase"]


def test_gate_ignores_a_bare_count_but_not_a_quantity():
    eng = {"claims": []}
    assert ENG.verify("Ada 3 opsi di sini.", eng)["ok"] is True
    assert ENG.verify("Ada 3 mm di sini.", eng)["ok"] is False


def test_chat_block_lists_claims_and_missing_inputs():
    eng = ENG.solve("", given={"topic": "beam", "section": "rect", "b_mm": 20, "h_mm": 10,
                               "L_mm": 500, "material": ENG.material("steel_1018")})
    block = ENG.chat_block(eng)
    assert "ENGINEERING EVIDENCE" in block
    assert "MISSING INPUTS" in block and "F_n" in block
    assert "never assume" in block


def test_lesson_carries_steps_and_a_check():
    lesson = ENG.teach("assembly")
    assert lesson["found"] is True
    assert lesson["steps"] and lesson["check"]
    assert ENG.teach("nonsense")["found"] is False


def test_material_table_is_internally_consistent():
    for key, m in ENG.MATERIALS.items():
        assert m["sy_mpa"] <= m["sut_mpa"], key
        assert m["E_mpa"] > 0 and m["rho_kg_m3"] > 0
        assert m["source"], key
