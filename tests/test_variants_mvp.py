"""MVP framework priorities: shafts+bearings in every variant, solver-proposed repairs."""
from physics.gears import Requirements
from physics.variants import generate_variants

DEMO = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150,
                    gear_material="AMS6265")


def test_every_variant_has_shaft_and_bearing_check():
    out = generate_variants(DEMO)
    assert out["variants"]
    for v in out["variants"]:
        assert len(v["shafts"]) == v["design"]["stages"] + 1
        for s in v["shafts"]:
            assert s["ok"], s
            assert s["bearings"] and all(b["life_h_L10"] >= 5000 for b in s["bearings"])
            assert s["min_fatigue_safety_factor"] >= 1.5
        assert isinstance(v["warnings"], list)


def test_repairs_are_verified_by_the_solver():
    req = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=120, env_h=100, env_d=80)
    out = generate_variants(req)
    assert not out["variants"] and out["repairs"]
    best = out["repairs"][0]
    fixed = Requirements(**{**req.__dict__, **{
        {"envelope_width": "env_w", "envelope_height": "env_h", "envelope_depth": "env_d"}.get(k, k): v
        for k, v in best["change"].items()}})
    assert len(generate_variants(fixed)["variants"]) > 0
    assert out["suggestions"][0].startswith("Solver check:")


def test_no_repair_message_when_out_of_scope():
    out = generate_variants(Requirements(power_kw=200, input_rpm=1500, gear_ratio=30, env_w=200, env_h=200,
                                         env_d=150, gear_material="AMS6265"))
    assert not out["repairs"] and "design space" in out["suggestions"][0]
