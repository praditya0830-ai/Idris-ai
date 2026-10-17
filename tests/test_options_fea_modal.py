"""UI request #2 (second half): computed `options` for /fea and /modal, reproducible with Try."""
from ace_core.analysis import handle_post
from solvers._example import EXAMPLE_VARIANT as V

ENV = {"envelope_mm": {"width": 200, "height": 200, "depth": 150}}


def test_fea_options_try_roundtrip():
    c, b = handle_post("/fea?sync=1", {**V, "mesh": "coarse", **ENV})
    r = b["result"]
    assert c == 200 and r["options"] and r["fits_envelope"] is True
    wall = [o for o in r["options"] if o["kind"] == "wall"]
    assert wall, r["options"]
    o = wall[0]
    c2, b2 = handle_post("/fea?sync=1", {**V, "mesh": "coarse", **ENV, **o["change"]})
    sf2 = b2["result"]["summary"]["min_safety_factor"]
    assert abs(sf2 - o["safety_factor"]) / o["safety_factor"] < 0.01
    for o in r["options"]:
        assert o["basis"] == "computed" and (o["fits_envelope"] is not False or not o["pass"])


def test_modal_speed_map_and_wall_option():
    c, b = handle_post("/modal?sync=1", {**V, "mesh": "coarse", "modes": 4})
    assert c == 200
    for sm in b["result"]["speed_map"]:           # flag == operating speed inside the band to avoid
        assert sm["includes_operating_speed"] == (sm["avoid_input_rpm"][0] <= 6000 <= sm["avoid_input_rpm"][1])
    # run the same housing at 5200 rpm input: stage-1 mesh (≈1560 Hz) now sits on the first mode
    kin = dict(V["kinematics"])
    kin["output_rpm"] = 5200 / kin["ratio_actual"]
    c, b = handle_post("/modal?sync=1", {**V, "kinematics": kin, "mesh": "coarse", "modes": 4})
    r = b["result"]
    assert c == 200
    hot = [sm for sm in r["speed_map"] if sm["includes_operating_speed"]]
    assert hot and all(sm["avoid_input_rpm"][0] <= 5200 <= sm["avoid_input_rpm"][1] for sm in hot)
    kinds = {o["kind"] for o in r["options"]}
    assert "speed" in kinds and "wall" in kinds
    w = [o for o in r["options"] if o["kind"] == "wall"][0]
    assert w["first_mode_hz"] > r["frequencies_hz"][0]          # thicker wall -> stiffer housing
