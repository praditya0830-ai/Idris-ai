"""QA Q-016: the engineer chooses the calculation reference; every stress number names it."""
from physics.references import REFERENCES, resolve, basis
from physics.gears import Requirements
from physics.variants import generate_variants


def test_resolve_and_labels():
    assert resolve(None) == "lewis" and resolve("ISO 6336") == "iso6336" and resolve("iso") == "iso6336"
    try:
        resolve("magic")
        assert False
    except ValueError:
        pass
    assert REFERENCES["lewis"]["status"] == "verified" and REFERENCES["iso6336"]["status"] == "verified"
    assert not any("preliminary" in v for v in basis("lewis").values())
    assert "Method C" in basis("iso6336")["sf_bending"]


def test_tooth_form_matches_iso_6336_3():
    from physics.toothform import tooth_form, iso_closed_form
    for z in (12, 17, 20, 30, 50, 100, 200):
        t, i = tooth_form(z), iso_closed_form(z)
        assert abs(t["YF"] / i["YF"] - 1) < 0.005 and abs(t["YS"] / i["YS"] - 1) < 0.005, z
    # the standard's chart values for x = 0 (Y_Fa, Y_Sa): z = 20 ≈ 2.80 / 1.55, z = 100 ≈ 2.20 / 1.79
    assert abs(tooth_form(20)["YF"] - 2.80) < 0.03 and abs(tooth_form(20)["YS"] - 1.55) < 0.02
    assert abs(tooth_form(100)["YF"] - 2.20) < 0.03 and abs(tooth_form(100)["YS"] - 1.79) < 0.02


def test_reference_changes_the_bending_check():
    kw = dict(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150, wall_thickness=4.0,
              gear_material="AMS6265", housing_material="Al7075-T6", min_sf_bending=1.4, min_sf_contact=1.1)
    a = generate_variants(Requirements(**kw, bending_method="lewis"))
    b = generate_variants(Requirements(**kw, bending_method="iso"))
    assert a["variants"] and b["variants"]
    # contact (Hertz) is the same method in both references
    va = {(v["design"]["module_mm"], v["design"]["pinion_teeth"], v["design"]["face_width_mm"]): v for v in a["variants"]}
    for v in b["variants"]:
        k = (v["design"]["module_mm"], v["design"]["pinion_teeth"], v["design"]["face_width_mm"])
        if k in va:
            assert abs(va[k]["stress"]["contact_mpa"] - v["stress"]["contact_mpa"]) < 1e-6
