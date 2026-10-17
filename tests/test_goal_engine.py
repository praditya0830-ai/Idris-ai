"""ACE Goal Engine G1: thermal options in /cfd, fin model, thermal screen, goals in the generator."""
import math
from ace_core.analysis import handle_post
from solvers._example import EXAMPLE_VARIANT as VARIANT
from physics.thermal_lumped import oil_temp_lumped, h_channel_natural, fin_conv_factor, fin_layout
from physics.gears import Requirements
from physics.variants import generate_variants

BASE = {**VARIANT, "power_kw": 11.19, "input_rpm": 6000}
REQ = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150,
                   wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                   min_sf_bending=1.4, min_sf_contact=1.1)


def _cfd(extra=None):
    c, b = handle_post("/cfd?sync=1", {**BASE, **(extra or {})})
    assert c == 200, b
    return b["result"]


def test_lumped_screen_matches_3d_fem():
    r = _cfd()
    W, H, D = r["housing_box_mm"]
    q = oil_temp_lumped(r["heat_to_housing_w"], W, H, D, 4, r["conductivity_w_mk"], r["h_oil_w_m2k"])
    assert abs(q["oil_temp_c"] - r["oil_temp_c"]) / (r["oil_temp_c"] - 25) < 0.02
    assert r["surface_area_m2"] == r["area_outer_m2"] > 0


def test_cfd_options_are_reproducible_with_try():
    r = _cfd()
    assert r["verdict"] == "FAIL" and r["options"]
    kinds = {o["kind"] for o in r["options"]}
    assert {"fan", "fins", "oil_cooler", "efficiency", "larger_housing", "combined"} <= kinds
    assert any(o["pass"] for o in r["options"] if o["kind"] != "oil_cooler")
    for o in r["options"]:
        if o["kind"] == "larger_housing":
            continue                     # same physics, checked in the test below (slower)
        r2 = _cfd(o["change"])
        assert abs(r2["oil_temp_c"] - o["oil_temp_c"]) < 0.6, o
        assert (r2["verdict"] == "PASS") == o["pass"], o


def test_larger_housing_and_fins_cool_down():
    r0 = _cfd()
    r1 = _cfd({"housing_scale": 1.3})
    assert r1["oil_temp_c"] < r0["oil_temp_c"] and r1["surface_area_m2"] > r0["surface_area_m2"]
    r2 = _cfd({"fins": {"count": 16, "height_mm": 25}})
    assert r2["oil_temp_c"] < r0["oil_temp_c"] and r2["fins"]["mass_kg"] > 0
    c, b = handle_post("/cfd?sync=1", {**BASE, "fins": {"count": 60, "height_mm": 20}})
    assert c == 400 and "do not fit" in b["error"]


def test_oil_cooler_removes_heat():
    r = _cfd({"oil_cooler_w": 400})
    assert r["heat_to_housing_w"] < r["heat_generated_w"] and r["oil_temp_c"] < 90


def test_cfd_goals_filter_options():
    r = _cfd({"goals": {"oil_temp_max_c": 100, "added_mass_max_kg": 0.3, "allowed": ["fan", "fins", "efficiency", "combined"]}})
    assert all(o["kind"] in ("fan", "fins", "efficiency", "combined") for o in r["options"])
    assert all(o["added_mass_kg"] <= 0.3 for o in r["options"] if o["pass"])
    assert r["goal_report"]["oil_temp_max_c"] == 100 and r["checks"][0]["limit"] == 100


def test_fin_physics_limits():
    # very short fin -> efficiency -> 1 (Incropera eq. 3.89)
    fl = fin_layout({"count": 10, "height_mm": 0.5}, 100, 100)
    _, eta = fin_conv_factor(fl, 10.0, 130, 0.1, "forced", 80, 25)
    assert eta > 0.999
    # Bar-Cohen & Rohsenow: wide channel -> isolated plate Nu_s ≈ 0.59·El^¼
    s, L, Ts, Ta = 60.0, 100.0, 80.0, 25.0
    h = h_channel_natural(s, L, Ts, Ta)
    beta = 1 / (0.5 * (Ts + Ta) + 273.15)
    Ra = 9.81 * beta * (Ts - Ta) * (s * 1e-3) ** 3 / (1.57e-5 * 2.22e-5)
    El = Ra * s / L
    assert abs(h / (0.0263 / (s * 1e-3)) / (El ** 0.5 / 2.873) ** 0.5 - 1) < 0.05


def test_generator_overall_basis_and_bbox():
    # layout v4: real shaft ends (boss + seal + coupling) do not fit a 150 mm overall depth -> honest
    # "no variant" with a re-checked repair; at the repaired depth every assembly box fits
    o = generate_variants(REQ, envelope_basis="overall")
    assert o["envelope_basis"] == "overall" and not o["variants"]
    rep = o["repairs"][0]
    assert rep["feasible_variants"] > 0 and "re-checked" in rep["reason"]
    d = rep["change"]["envelope_depth"]
    o = generate_variants(Requirements(**{**REQ.__dict__, "env_d": d}), envelope_basis="overall")
    assert len(o["variants"]) == rep["feasible_variants"]
    for v in o["variants"]:
        bb = v["assembly_bbox_mm"]
        assert bb["max"][2] - bb["min"][2] <= d + 1e-6
        assert bb["min"][2] <= -22 and v["thermal_screen"]["oil_temp_natural_c"] > 90


def test_generator_goals_steer_selection():
    o = generate_variants(REQ, goals={"oil_temp_max_c": 90, "mass_max_kg": 5.0,
                                      "cooling_allowed": ["natural", "fan", "fins", "efficiency", "oil_cooler"]})
    assert o["diagnostics"]["meeting_goals"] > 0
    for v in o["variants"]:
        assert all(x["pass"] for x in v["goal_check"].values())
        assert v["goal_check"]["mass_max_kg"]["value"] <= 5.0
    o2 = generate_variants(REQ, goals={"cooling_allowed": ["natural"]})
    assert o2["goal_summary"]["variants_meeting_all"] == []
    assert "oil_temp_max_c" in o2["goal_summary"]["blocking_goals"]
    assert all(0 < v["thermal_screen"]["max_power_natural_kw"] < 11.19 for v in o2["variants"])


def test_efficiency_assumption_never_alone_makes_pass():
    # QA Q-020: without an oil cooler, S1 only "passes" with the 99 % efficiency assumption -> goal not met
    o = generate_variants(REQ, goals={"oil_temp_max_c": 90})
    for v in o["variants"]:
        ts = v["thermal_screen"]
        if ts["measure"] and ts["measure"]["basis"] == "assumption":
            assert "(assumption)" in ts["measure"]["label"] and ts["best_without_assumption"]
            assert v["goal_check"]["oil_temp_max_c"]["pass"] is False
            assert v["goal_check"]["oil_temp_max_c"]["passes_only_with_assumption"] is True
    r = _cfd()
    kinds = [o["basis"] for o in r["options"]]
    assert kinds == sorted(kinds, key=lambda b: b != "computed")       # computed first
    assert all(o["assumption"] for o in r["options"] if "efficiency_per_stage" in o["change"])
    assert r["best_without_assumption"]["basis"] == "computed" and r["best_without_assumption"]["pass"]


def test_screen_vs_fem_on_demo_variants():
    # QA Q-019: L1 screen vs 3D FEM on demo variants (layout v4 with the central web: max 1.2 % over 15 variants)
    import json, os
    path = os.path.join(os.path.dirname(__file__), "..", "qa-skenario-demo.json")
    if not os.path.exists(path):
        return
    from physics.goal_engine import h_oil
    from physics.variant_shafts import designed_housing_params
    D = json.load(open(path))
    for sk in D:
        v = D[sk]["variants"][1]
        req = D[sk]["requirements"]
        body = {k: v[k] for k in ("design", "kinematics", "loads", "shaft", "housing_params") if k in v}
        body.update(power_kw=req["power_kw"], input_rpm=req["input_rpm"])
        r = _cfd_body(body)
        W, H, Dd = r["housing_box_mm"]
        web = designed_housing_params(body, body["housing_params"], req["input_rpm"]).get("web_mm", 0.0)
        q = oil_temp_lumped(r["heat_to_housing_w"], W, H, Dd, 4, r["conductivity_w_mk"], h_oil(v, req["input_rpm"]),
                            web_mm=web)
        assert abs(q["oil_temp_c"] - r["oil_temp_c"]) / (r["oil_temp_c"] - 25) < 0.01, (sk, q["oil_temp_c"], r["oil_temp_c"])


def _cfd_body(body):
    c, b = handle_post("/cfd?sync=1", body)
    assert c == 200, b
    return b["result"]


def test_q022_options_respect_envelope():
    env = {"envelope_mm": {"width": 200, "height": 200, "depth": 150}, "envelope_basis": "overall"}
    r = _cfd(env)
    for o in r["options"]:
        if o["fits_envelope"] is False:
            assert not o["pass"] and "exceeds envelope" in o["label"] and o["envelope_violation"]
        w, h, d = o["outer_size_mm"]
        assert o["fits_envelope"] == (w <= 200.5 and h <= 200.5 and d <= 150.5)
    b = r["best_without_assumption"]
    assert b is None or b["fits_envelope"] is True
    # generator: every computed measure stays inside the envelope (fins included)
    o = generate_variants(REQ, goals={"oil_temp_max_c": 90})
    for v in o["variants"]:
        m = v["thermal_screen"]["measure"]
        hp = v["housing_params"]
        f = (m or {}).get("change", {}).get("fins")
        if f:
            if f.get("location") == "top":
                assert hp["envelope_height"] + f["height_mm"] <= 200.5
            else:
                assert hp["envelope_width"] + 2 * f["height_mm"] <= 200.5


def test_allowed_filters_every_measure_of_combined_options():
    r = _cfd({"goals": {"oil_temp_max_c": 90, "allowed": ["fins", "oil_cooler", "larger_housing"]}})
    for o in r["options"]:
        ch = o["change"]
        assert ch.get("cooling") != "forced" and "efficiency_per_stage" not in ch, o["label"]
