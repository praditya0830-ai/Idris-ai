"""Layout v4 (v0.11): every variant Idris generates must be buildable as delivered (Arya, 16 Oct 2026).

Checks on a 2-stage and a 1-stage scenario: every part is a closed solid, every shaft runs in two
designed bearings with a real seat, the review finds no critical or warning, and the housing still
fits the engineer's envelope.
"""
import numpy as np
from ace_core.analysis import handle_post
from assembly3d import build_assembly
from physics.gears import Requirements
from physics.variants import generate_variants

S1 = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150,
                  wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)
S2 = Requirements(power_kw=11.0, input_rpm=6000, gear_ratio=6, env_w=250, env_h=200, env_d=180,
                  wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)


def _closed(part):
    V = np.reshape(part["verts"], (-1, 3))
    F = np.reshape(part["faces"], (-1, 3))
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    return set(c) == {2} and vol > 0


def _check(req, stages):
    o = generate_variants(req)
    assert o["variants"], o
    for v in o["variants"][:2]:
        assert v["design"]["stages"] == stages
        hp = v["housing_params"]
        assert hp["envelope_width"] <= req.env_w and hp["envelope_height"] <= req.env_h
        assert hp["envelope_depth"] <= req.env_d
        assert v["warnings"] == [] and v["notes"]
        asm = build_assembly(v["design"], v["kinematics"], v["shaft"], hp)
        bad = [p["name"] for p in asm["parts"] if not _closed(p)]
        assert not bad, bad
        kinds = {p["kind"] for p in asm["parts"]}
        assert {"housing", "bearing", "shaft", "gear", "seal", "fastener", "plug"} <= kinds
        n_shafts = stages + 1
        assert sum(p["kind"] == "bearing" for p in asm["parts"]) == 2 * n_shafts
        assert asm["layout"]["manufacturable"] is True and asm["warnings"] == []
        if stages == 2:
            assert any(p["name"] == "Housing front cover" for p in asm["parts"]) and asm["layout"]["web_mm"] > 0
        c, b = handle_post("/review?sync=1", {**v, "checks": ["interference", "fits", "dfm", "assembly"]})
        assert c == 200
        cnt = b["result"]["counts"]
        assert cnt["critical"] == 0 and cnt["warning"] == 0, b["result"]["findings"]


def test_two_stage_variants_are_buildable():
    _check(S1, 2)


def test_one_stage_variants_are_buildable():
    _check(S2, 1)


def test_web_is_ribbed_not_solid():
    # the central web is a plate + bearing hub, so its mass is far below a solid block of the web length
    from physics.gears import web_volume_mm3
    solid = 180 * 110 * 28
    assert web_volume_mm3(180, 110, 28, 4.0, 58.0, [39.0]) < 0.4 * solid


def test_envelope_fit_is_reported_by_the_solver():
    # v0.11.1: the UI shows the solver's verdict instead of recomputing (UI v26 showed a false mismatch)
    v = generate_variants(S1)["variants"][0]
    ef = v["envelope_fit"]
    hp = v["housing_params"]
    assert ef["basis"] == "housing" and ef["fits"] is True and ef["fits_housing"] is True
    assert ef["housing_box_mm"] == [hp["envelope_width"], hp["envelope_height"], hp["envelope_depth"]]
    asm = build_assembly(v["design"], v["kinematics"], v["shaft"], hp)
    assert asm["layout"]["overall_box_mm"] == ef["overall_box_mm"]
    assert asm["layout"]["housing_box_mm"] == ef["housing_box_mm"]


def test_envelope_choice_offers_both_bases():
    # Arya 17 Oct: let the engineer choose in the chat — housing basis vs the whole unit incl. shaft ends
    o = generate_variants(S1)
    c = o["envelope_choice"]
    assert c and len(c["options"]) == 2
    h, w = c["options"]
    assert h["basis"] == "housing" and h["selected"] and h["variant_count"] == len(o["variants"]) == 5
    assert w["basis"] == "overall" and not w["selected"] and w["variant_count"] == len(w["variants"]) >= 1
    d = w["request_change"]["envelope_depth"]
    for v in w["variants"]:
        assert v["envelope_fit"]["fits_overall"] and v["envelope_fit"]["overall_box_mm"][2] <= d + 0.5
    # applying the change reproduces the offered variants
    o2 = generate_variants(Requirements(**{**S1.__dict__, "env_d": d}), envelope_basis="overall", compare_basis=False)
    assert [v["design"] for v in o2["variants"]] == [v["design"] for v in w["variants"]]
    # when the whole unit already fits, there is nothing to ask
    assert generate_variants(S2)["envelope_choice"] is None
