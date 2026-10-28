"""POST /edit-variant (UI drag handles): a change is never only on screen; the solver re-checks everything."""
from dataclasses import replace
from physics.gears import Requirements
from physics.variants import generate_variants
from physics.edit_variant import edit_variant

S1 = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150,
                  wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)


def _v():
    return generate_variants(S1, compare_basis=False)["variants"][0]


def test_wall_edit_gives_new_rechecked_variant_with_diff():
    v = _v()
    code, b = edit_variant(S1, v, {"wall_thickness_mm": 5.0})
    assert code == 200 and b["feasible"] and b["output_level"] == "screening"
    nv = b["variant"]
    assert nv["name"] == v["name"] + "′" and nv["edited_from"] == v["name"]
    assert nv["housing_params"]["wall_thickness"] == 5.0 and nv["design"] == v["design"]
    # before = the original variant, recomputed the same way
    assert b["before"]["mass_total_kg"] == v["mass_kg"]["total"] and b["before"]["physics_pct"] == v["scores"]["physics_pct"]
    q = {d["quantity"]: d for d in b["diff"]}
    assert q["mass_housing_kg"]["change"] > 0 and q["wall_thickness_mm"]["after"] == 5.0
    assert nv["warnings"] == [] and nv["shafts"]


def test_face_width_edit_changes_strength_and_respects_range():
    v = _v()
    rng = edit_variant(S1, v, {"face_width_mm": 1e6})
    assert rng[0] == 422 and "outside" in rng[1]["error"]
    lo = rng[1]["allowed_range"]["face_width_mm"]["min"]
    code, b = edit_variant(S1, v, {"face_width_mm": lo})
    assert code == 200
    if b["feasible"]:
        assert b["after"]["sf_bending"] < b["before"]["sf_bending"]
    else:
        assert b["reasons"]


def test_material_edit_and_not_editable():
    v = _v()
    code, b = edit_variant(S1, v, {"housing_material": "Al6061-T6"})
    assert code == 200 and (b["variant"] is None or b["variant"]["materials"]["housing"] == "Al6061-T6")
    code, b = edit_variant(S1, v, {"fillet_mm": 3})
    assert code == 400 and "fillet_chamfer" in b["allowed_range"]["not_editable"]
    assert edit_variant(S1, v, {"wall_thickness_mm": 5, "face_width_mm": 30})[0] == 400


def test_generate_unchanged_by_hooks():
    a = generate_variants(S1, compare_basis=False)["variants"]
    assert [x["design"] for x in a] == [x["design"] for x in generate_variants(S1, compare_basis=False)["variants"]]
