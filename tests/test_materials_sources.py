"""UI v65 item 1: the five materials the Studio offers are in the solver table, each with a source.

The point of these tests is not that a number is pretty: it is that every density and every strength can say where
it comes from, and that a material which does not yield (cast iron) says so instead of quietly looking like a
ductile metal with a yield point.
"""
import ai.fea as FEA
from physics.part_fea import MATERIALS, MATERIAL_NOTE, material_catalog

WANTED = ("stainless", "brass", "abs", "nylon", "cast_iron")


def test_the_five_ui_materials_are_in_the_solver_table():
    assert set(WANTED) <= set(MATERIALS)
    for key in WANTED:
        row = MATERIALS[key]
        assert row["E"] > 0 and row["Sy"] > 0 and row["rho"] > 0, key
        src = row.get("source") or {}
        assert src.get("density") and "g/cm^3" in src["density"], key
        assert src.get("strength") and "MPa" in src["strength"], key
        assert src.get("stiffness") and "GPa" in src["stiffness"], key


def test_every_material_in_the_table_has_a_source_not_only_the_new_five():
    for key, row in MATERIALS.items():
        src = row.get("source") or {}
        assert {"density", "strength", "stiffness"} <= set(src), key


def test_stainless_is_304_at_the_standard_minimum_not_a_typical_value():
    row = MATERIALS["stainless"]
    assert "304" in row["name"]
    assert row["Sy"] == 205.0 and "ASTM A240" in row["source"]["strength"]   # Rp0.2 minimum, annealed
    assert row["rho"] == 8000.0


def test_nylon_is_pa12_and_brass_admits_its_temper_dependence():
    assert "PA12" in MATERIALS["nylon"]["name"]
    assert "EOS PA 2200" in MATERIALS["nylon"]["source"]["density"]
    assert "temper" in MATERIALS["brass"]["source"]["strength"]   # Rp0.2 200 MPa is the softest temper


def test_cast_iron_is_brittle_and_says_which_strength_is_used():
    row = MATERIALS["cast_iron"]
    assert row["rho"] == 7200.0 and row["Sy"] == 250.0
    assert row.get("brittle") is True
    assert "no yield point" in row["source"]["strength"] and "Rm" in row["source"]["strength"]
    assert "Cast iron" in MATERIAL_NOTE and "brittle" in MATERIAL_NOTE.lower()


def test_the_catalog_serves_the_sources_and_part_check_accepts_cast_iron():
    row = next(m for m in material_catalog()["materials"] if m["id"] == "cast_iron")
    assert row["rho"] == 7200.0 and row["source"]["strength"]
    assert "cast_iron" in FEA.SOLVER_MATERIALS          # POST /part-check will not refuse it as unknown
    assert material_catalog()["note"] == MATERIAL_NOTE
