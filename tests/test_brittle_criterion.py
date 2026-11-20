"""Brittle materials in /part-check: Rankine (maximum principal stress) against tensile Rm, with its label.

A grey cast iron has no proof stress, so "von Mises vs yield" is not a number about that material. These tests pin
the three things that make the answer honest: the principal stresses are right, the rule and the strength it uses
travel with the number, and a load case with no tension says so instead of printing an infinite factor.
"""
import math

import numpy as np

import solvers  # noqa: F401  (registers fea_part_3d)
from ace_core import analysis
from physics import part_fea as pf
from solvers.part_fea import BOX_EX

# the cantilever box with the hand-entered steel values removed, so the material row decides E, nu and strength
BRITTLE_BOX = {k: v for k, v in BOX_EX.items() if k not in ("E_MPa", "nu", "yield_MPa")}
BRITTLE_BOX["material"] = "cast_iron"


def test_principal_stresses_are_the_eigenvalues_in_descending_order():
    uni = pf.principal_stresses([[100.0, 0, 0, 0, 0, 0]])[0]
    assert np.allclose(uni, [100.0, 0.0, 0.0], atol=1e-9)
    shear = pf.principal_stresses([[0, 0, 0, 100.0, 0, 0]])[0]      # pure shear: +100, 0, -100
    assert np.allclose(shear, [100.0, 0.0, -100.0], atol=1e-9)
    hydro = pf.principal_stresses([[-50.0, -50.0, -50.0, 0, 0, 0]])[0]
    assert np.allclose(hydro, [-50.0, -50.0, -50.0], atol=1e-9)


def test_the_criterion_follows_the_material_and_says_which_strength_it_uses():
    brittle = pf.criterion_for(pf.MATERIALS["cast_iron"])
    assert brittle["id"] == "rankine" and "Rankine" in brittle["label"]
    assert "max principal stress vs Rm (brittle" in brittle["label"]
    assert brittle["strength_symbol"] == "Rm" and "does not yield" in brittle["strength_basis"]
    ductile = pf.criterion_for(pf.MATERIALS["steel"])
    assert ductile["id"] == "von_mises" and "Rp0.2" in ductile["label"]


def test_rankine_uses_the_tensile_principal_stress_and_not_von_mises():
    iron = pf.safety(pf.MATERIALS["cast_iron"], vm_max=300.0, s1_max=100.0)
    assert iron["criterion"]["id"] == "rankine"
    assert iron["critical_stress_MPa"] == 100.0                 # not the 300 MPa von Mises number
    assert math.isclose(iron["min_safety_factor"], 250.0 / 100.0)   # Rm / sigma_1
    steel = pf.safety(pf.MATERIALS["steel"], vm_max=100.0, s1_max=999.0)
    assert steel["criterion"]["id"] == "von_mises"
    assert math.isclose(steel["min_safety_factor"], 235.0 / 100.0)  # Rp0.2 / von Mises, s1 ignored


def test_no_tension_means_no_rankine_factor_instead_of_a_huge_one():
    out = pf.safety(pf.MATERIALS["cast_iron"], vm_max=40.0, s1_max=-5.0, s3_min=-60.0)
    assert out["min_safety_factor"] is None
    assert "no tensile principal stress" in out["safety_note"]


def test_a_compressed_part_says_compression_dominates_and_that_strength_is_not_used():
    out = pf.safety(pf.MATERIALS["cast_iron"], vm_max=12.0, s1_max=1.0, s3_min=-30.0)
    assert out["min_safety_factor"] is not None
    assert "compression dominates" in out["safety_note"]
    assert "stronger in compression" in out["safety_note"] and "not used" in out["safety_note"]


def _loaded(force):
    """The 100 x 10 x 10 bar, clamped at one end, pulled (+x) or pushed (-x) along its axis."""
    return {**BRITTLE_BOX, "element_size_mm": 2.5,
            "supports": [{"where": {"plane": "x", "side": "min"}}],
            "loads": [{"force_N": list(force), "where": {"plane": "x", "side": "max"}}]}


def test_pure_tension_is_judged_by_rankine_and_the_factor_is_rm_over_sigma_1():
    o = pf.run(_loaded((1000.0, 0, 0)))          # F/A = 1000/100 = 10 MPa in tension (the clamped end adds some)
    assert o["criterion"]["id"] == "rankine"
    assert 5.0 < o["max_principal_MPa"] < 40.0
    assert math.isclose(o["min_safety_factor"], o["strength_MPa"] / o["max_principal_MPa"], rel_tol=1e-6)
    assert o["safety_note"] is None or "compression dominates" not in o["safety_note"]
    assert len(o["surface"]["principal_max"]) == len(o["surface"]["von_mises"])


def test_bending_puts_one_side_in_tension_and_the_other_in_compression():
    o = pf.run(dict(BRITTLE_BOX, element_size_mm=2.5))       # the cantilever tip load: bending
    assert o["max_principal_MPa"] > 0 > o["min_principal_MPa"]
    assert o["criterion"]["id"] == "rankine"


def test_an_axial_compression_run_never_reports_the_compressive_strength_as_used():
    o = pf.run(_loaded((-1000.0, 0, 0)))
    assert o["min_principal_MPa"] < 0
    note = str(o["safety_note"] or "")
    assert (o["min_safety_factor"] is None and "no tensile principal stress" in note) or \
           "compression dominates" in note


def test_part_check_on_cast_iron_reports_rankine_with_its_label():
    o = pf.run(dict(BRITTLE_BOX, element_size_mm=2.5))
    crit = o["criterion"]
    assert crit["id"] == "rankine" and "Rankine" in crit["label"]
    assert o["material"]["brittle"] is True and o["strength_MPa"] == 250.0
    assert o["max_principal_MPa"] > 0
    assert math.isclose(o["critical_stress_MPa"], o["max_principal_MPa"], rel_tol=1e-6)
    assert math.isclose(o["min_safety_factor"], o["strength_MPa"] / o["critical_stress_MPa"], rel_tol=1e-6)
    assert o["safety_note"] is None
    assert len(o["surface"]["principal_max"]) == len(o["surface"]["von_mises"])


def test_a_ductile_part_keeps_von_mises_and_its_own_label():
    o = pf.run(dict(BOX_EX, element_size_mm=2.5))
    assert o["criterion"]["id"] == "von_mises" and "Rp0.2" in o["criterion"]["label"]
    assert math.isclose(o["critical_stress_MPa"], o["max_von_mises_MPa"], rel_tol=1e-3)   # rounded to 4 decimals
    assert math.isclose(o["min_safety_factor"], 250.0 / o["max_von_mises_MPa"], rel_tol=1e-9)


def test_the_route_carries_the_criterion_the_label_and_the_check_name():
    code, body = analysis.handle_post("/part-check?sync=1", dict(BRITTLE_BOX, element_size_mm=2.5))
    assert code == 200 and body["status"] == "done"
    r = body["result"]
    assert r["summary"]["criterion"] == "rankine" and "Rankine" in r["summary"]["criterion_label"]
    assert r["summary"]["strength_mpa"] == 250.0
    assert r["summary"]["max_principal_mpa"] > 0
    names = [c["name"] if isinstance(c, dict) else c.name for c in r["checks"]]
    assert any("Rankine" in str(n) for n in names)
    assert any("Strength criterion" in str(w) for w in r["warnings"])
