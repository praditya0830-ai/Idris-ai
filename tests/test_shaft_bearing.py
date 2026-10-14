"""Validation of the shaft + bearing solver against textbook results and closed-form theory."""
import math
from physics.shaft import (marin_ka, marin_kb, marin_ke, endurance_limit, neuber_sqrt_a,
                           fatigue_kf, von_mises_stresses, fatigue_safety, beam_response,
                           beam_at, rayleigh_critical_speed)
from physics.bearings import required_C10, life_hours, select_bearing, equivalent_load
from physics.shaft_design import design_shaft, shafts_for_gear_stage

IN, LBF, KPSI = 25.4, 4.448222, 6.894757


def test_shigley_7_1_marin_factors():
    # Shigley Ex. 7-1: Sut=105 kpsi machined, d=1.1 in, R=0.99 -> ka=0.787, kb=0.870, ke=0.814, Se=29.3 kpsi
    Sut = 105 * KPSI
    assert abs(marin_ka(Sut, "machined") - 0.787) < 0.003
    assert abs(marin_kb(1.1 * IN) - 0.870) < 0.004
    assert abs(marin_ke(0.99) - 0.814) < 0.001
    Se = endurance_limit(Sut, 1.1 * IN, "machined", 0.99)["Se"]
    assert abs(Se / KPSI - 29.3) < 0.3


def test_shigley_7_1_notch_sensitivity():
    # Sut=105 kpsi, r=0.11 in, Kt=1.68 -> q=0.85, Kf=1.58
    q = 1 / (1 + neuber_sqrt_a(105 * KPSI) / math.sqrt(0.11 * IN))
    assert abs(q - 0.85) < 0.01
    assert abs(fatigue_kf(1.68, 0.11 * IN, 105 * KPSI) - 1.58) < 0.01


def test_shigley_7_1_safety_factors():
    # M=1260 lbf*in alternating, T=1100 lbf*in steady, Kf=1.58, Kfs=1.39 -> Goodman 1.62, Gerber 1.87
    d = 1.1 * IN
    M, T = 1260 * LBF * IN, 1100 * LBF * IN
    sa, sm = von_mises_stresses(d, M, 0, 0, T, 1.58, 1.39)
    Se, Sut, Sy = 29.3 * KPSI, 105 * KPSI, 82 * KPSI
    assert abs(fatigue_safety(sa, sm, Se, Sut, Sy, "goodman") - 1.62) < 0.02
    assert abs(fatigue_safety(sa, sm, Se, Sut, Sy, "gerber") - 1.87) < 0.02


def test_beam_central_load_closed_form():
    # Simply supported, uniform: y_max = F L^3/(48 E I), end slope = F L^2/(16 E I)
    L, d, E, F = 300.0, 25.0, 207000.0, 1000.0
    I = math.pi * d**4 / 64
    b = beam_response([(0, L, d)], E, 0, L, [(L / 2, -F, 0)])
    assert abs(beam_at(b, "defl", L / 2) / (F * L**3 / (48 * E * I)) - 1) < 0.005
    assert abs(beam_at(b, "slope", 0) / (F * L**2 / (16 * E * I)) - 1) < 0.005
    assert abs(beam_at(b, "M", L / 2) - F * L / 4) < 1e-6
    assert abs(b["y"]["RA"] - F / 2) < 1e-9


def test_beam_offset_load_reactions():
    b = beam_response([(0, 400, 30)], 207000, 0, 400, [(100, 0, -800)])
    assert abs(b["z"]["RA"] - 600) < 1e-9 and abs(b["z"]["RB"] - 200) < 1e-9


def test_rayleigh_uniform_shaft():
    # exact first critical of a uniform simply supported shaft: w = (pi/L)^2 sqrt(EI/(rho A))
    L, d, E, rho = 600.0, 20.0, 207000.0, 7850.0
    I, A = math.pi * d**4 / 64, math.pi * d**2 / 4
    w = (math.pi / L) ** 2 * math.sqrt(E * 1e6 * I * 1e-12 / (rho * A * 1e-6)) * 1e3 ** 2 / 1e3
    exact_rpm = (math.pi / (L / 1e3)) ** 2 * math.sqrt(E * 1e6 * (I * 1e-12) / (rho * A * 1e-6)) * 60 / (2 * math.pi)
    got = rayleigh_critical_speed([(0, L, d)], E, rho, 0, L)
    assert abs(got / exact_rpm - 1) < 0.02


def test_shigley_11_1_catalog_rating():
    # 2 kN, 5000 h at 1725 rpm, R=0.9 -> C10 ~ 16.0 kN (Shigley Ex. 11-1, ignoring Weibull ~1%)
    C = required_C10(2000, 1725, 5000, reliability=0.9, af=1.0)
    assert abs(C / 1000 - 16.0) < 0.3


def test_bearing_life_roundtrip():
    C = required_C10(3000, 3000, 4000, reliability=0.99, af=1.2)
    assert abs(life_hours(C, 3000, 3000, 0.99, 1.2) - 4000) < 1e-6


def test_equivalent_load_axial():
    eq = equivalent_load(1000, 600, 6550)        # Fa/C0 = 0.092 -> X=0.56, Y~1.52
    assert eq["X"] == 0.56 and 1.45 < eq["Y"] < 1.55
    assert equivalent_load(1000, 0, 6550)["P"] == 1000


def test_select_bearing_meets_requirement():
    b = select_bearing(20, 2500, 0, 3000, 5000, 0.99, 1.2)
    assert b["d_mm"] >= 20 and b["C_kN"] >= b["C10_required_kN"]


def test_design_shaft_converges_and_passes():
    res = design_shaft({"rpm": 6000, "xA_mm": 38, "xB_mm": 102, "length_mm": 114,
                        "coupling_x_mm": 15,
                        "gears": [{"x_mm": 70, "pitch_dia_mm": 28, "torque_Nm": -20,
                                   "face_width_mm": 24, "module_mm": 1.75}]})
    assert res["ok"], res
    assert all(s["pass"] for s in res["sections"])
    assert all(c["pass"] for c in res["checks"].values())
    for b in res["bearings"].values():
        assert b["d_mm"] == res["geometry"]["journal_dia_mm"]


def test_gear_stage_two_shafts():
    res = shafts_for_gear_stage({"input_torque_Nm": 20, "input_rpm": 6000,
                                 "pitch_dia_pinion_mm": 28, "pitch_dia_gear_mm": 112,
                                 "face_width_mm": 24.5, "module_mm": 1.75})
    assert res["ok"], res
    inp, out = res["input_shaft"], res["output_shaft"]
    # output shaft carries ~4x torque -> its strength-required seat diameter is larger
    req = lambda s: max(x["d_required_mm"] for x in s["sections"] if x["name"] == "gear seat")
    assert req(out) > req(inp)
    # input shaft runs 4x faster -> needs the larger bearing for the same life
    assert inp["bearings"]["A"]["C_kN"] >= out["bearings"]["A"]["C_kN"]


def test_rejects_bad_input():
    assert not design_shaft({"rpm": 0, "gears": []})["ok"]


def test_stiffness_loop_enlarges_long_shaft():
    # long, slender span: strength alone is fine, deflection/slope must drive the size up
    res = design_shaft({"rpm": 3000, "xA_mm": 20, "xB_mm": 620, "length_mm": 640,
                        "coupling_x_mm": 10,
                        "gears": [{"x_mm": 320, "pitch_dia_mm": 60, "torque_Nm": -15,
                                   "face_width_mm": 20, "module_mm": 1.0}]})
    assert res["ok"], res
    assert any("stiffen" in it["action"] for it in res["iterations"])
    assert all(c["pass"] for c in res["checks"].values())
