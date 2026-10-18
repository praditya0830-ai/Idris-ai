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


def test_cover_joints_are_sized_and_sealed():
    # Tahap 1.1: screws + dowels from the bolted-joint solver; covers thick enough for the O-ring
    for req, stages in ((S1, 2), (S2, 1)):
        v = generate_variants(req, compare_basis=False)["variants"][0]
        asm = build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"])
        joints = asm["layout"]["joints"]
        assert set(joints) == ({"front", "back"} if stages == 2 else {"back"})
        for wall, j in joints.items():
            assert j["pass"] and j["size"] in ("M5", "M6", "M8") and j["count"] >= 3
            assert all(sf is None or sf >= j["targets"][k] for k, sf in j["safety_factors"].items())
            assert j["cover"]["oring_lift_mm"] <= j["cover"]["allowed_lift_mm"] + 1e-9
            assert j["cover"]["thickness_mm"] == asm["layout"]["cover_mm"] >= v["housing_params"]["wall_thickness"]
        names = {p["name"] for p in asm["parts"]}
        assert {"Cover O-rings", "Cover bolts", "Housing corner columns"} <= names
        spec = next(p for p in asm["parts"] if p["name"] == "Cover bolts")["spec"]
        assert "ISO 4762" in spec and "ISO 8734" in spec and "DIN 7603" in spec


def test_bom_lists_every_part():
    # Tahap 1.2: BOM from the assembly; made parts carry a mass from the closed STL volume
    v = generate_variants(S1, compare_basis=False)["variants"][0]
    asm = build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"], v.get("materials"))
    bom = asm["bom"]
    names = " | ".join(r["name"] for r in bom)
    for must in ("Housing", "Input shaft", "Gear (stage 2)", "Deep-groove ball bearing", "Radial lip seal",
                 "Socket head cap screw", "Dowel pin", "O-ring", "Magnetic drain plug"):
        assert must in names, must
    assert sum(r["qty"] for r in bom if r["name"].startswith("Deep-groove")) == 6
    for r in bom:
        if r["kind"] == "made":
            assert r["mass_kg_each"] and r["mass_kg_each"] > 0
    from physics.bom import bom_csv
    assert bom_csv(bom).splitlines()[0].startswith("item,name,qty")


def test_step_export_reproduces_the_solids():
    # Tahap 1.3: faceted STEP AP214; an independent re-read gives back every solid's volume
    from physics.step_export import write_step, check_step, step_solid_volumes
    from physics.bom import _volume_mm3
    v = generate_variants(S2, compare_basis=False)["variants"][0]
    asm = build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"], v.get("materials"))
    txt, stats = write_step(asm["parts"])
    assert txt.startswith("ISO-10303-21;") and "AUTOMOTIVE_DESIGN" in txt and txt.rstrip().endswith("END-ISO-10303-21;")
    ok, msg = check_step(txt)
    assert ok, msg
    vols = step_solid_volumes(txt)
    assert len(vols) == len(stats) and min(vols.values()) > 0
    for p in asm["parts"]:
        sv = sum(val for k, val in vols.items() if k == p["name"] or k.startswith(p["name"] + " #"))
        assert abs(sv / _volume_mm3(p) - 1) < 1e-6, p["name"]


def test_drawings_are_valid_svg_with_fits():
    # Tahap 1.4: housing / covers / shafts sheets, outlines cut from the same meshes, ISO 286 fits
    import xml.etree.ElementTree as ET
    from physics.drawing import make_drawings
    from physics.variant_shafts import shaft_plan
    v = generate_variants(S1, compare_basis=False)["variants"][0]
    asm = build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"], v.get("materials"))
    plan = shaft_plan(v)
    d = make_drawings(asm, plan, v.get("materials"))
    assert set(d) == {"housing", "covers", "shafts"}
    for name, svg in d.items():
        root = ET.fromstring(svg.split("?>", 1)[1])
        assert root.tag.endswith("svg") and len(root) > 20, name
    assert "N7 (-0.0" in d["housing"] and "k5 (+0.0" in d["shafts"] and "ISO 4762" in d["covers"]
