"""Stage 5 joining and manufacturing (v0.42, doc tahap sambungan produksi): a design a workshop can build."""
import pytest

import design_ext
import joining
from physics import catalog, dfm_rules


BLADE = {"name": "blade", "material": "S355", "process": "laser_cut",
         "shape": {"type": "box", "size": [2400, 100, 6], "at": [0, 0, 0]}}
PLATE = {"name": "mount plate", "material": "S355", "process": "laser_cut",
         "shape": {"type": "box", "size": [120, 100, 6], "at": [-600, 0, 6]}}
TUBE = {"name": "push tube", "material": "S355", "process": "tube_cut",
        "shape": {"type": "cylinder", "d": 40, "l": 800, "wall": 3, "axis": "z", "at": [-600, 0, 409]}}


def test_a_stack_of_plate_and_tube_touches_instead_of_passing_through():
    r = joining.plan([BLADE, PLATE, TUBE])
    assert r["ok"] is True, r["failed"]
    keys = {(frozenset(j["between"]), j["kind"]) for j in r["joints"]}
    assert (frozenset(["blade", "mount plate"]), "fillet_weld") in keys
    assert (frozenset(["mount plate", "push tube"]), "fillet_weld") in keys
    overlap = [c for c in r["checks"] if c["check"] == "overlap"]
    assert overlap and all(c["status"] == "pass" and c["value_mm3"] == 0 for c in overlap)


def test_a_tube_that_passes_through_the_plate_is_proposed_not_moved():
    """Found in review (22 Nov): the shift was applied automatically, which can pull a tube off the blade."""
    through = {**TUBE, "shape": {**TUBE["shape"], "at": [-600, 0, 6]}}
    r = joining.plan([BLADE, PLATE, through])
    assert r["ok"] is False, "the overlap is still there, so the stage cannot pass"
    assert r["fixes"] == []
    assert r["issues"] and all(i["level"] == "proposed" for i in r["issues"])
    prop = [p for p in r["proposed_fixes"] if p["part"] == "push tube"][0]
    assert prop["accepted"] is False and prop["axis"] == "z" and prop["shift_mm"] > 0
    assert "if the engineer accepts" in prop["why"]
    tube = [p for p in r["parts"] if p["name"] == "push tube"][0]
    assert tube["process"] == "tube_cut"


def test_the_engineer_accepting_the_proposal_moves_the_part_and_passes():
    through = {**TUBE, "shape": {**TUBE["shape"], "at": [-600, 0, 6]}}
    r = joining.plan([BLADE, PLATE, through], fix=True)
    assert r["ok"] is True, r["failed"]
    assert any("push tube moved" in f for f in r["fixes"])
    assert all(c["status"] == "pass" for c in r["checks"] if c["check"] == "overlap")


def test_a_part_with_no_joint_is_reported_as_floating():
    lonely = {"name": "spare plate", "material": "S355", "shape": {"type": "box", "size": [50, 50, 6], "at": [2000, 0, 0]}}
    r = joining.plan([BLADE, PLATE, TUBE, lonely])
    fl = [c for c in r["checks"] if c["check"] == "every part is joined"][0]
    assert fl["status"] == "fail" and "spare plate" in fl["parts"]
    assert r["ok"] is False


def test_the_weld_leg_comes_from_the_sourced_table():
    r = joining.plan([BLADE, PLATE, TUBE])
    w = r["welds"][0]
    assert w["leg_mm"] == dfm_rules.weld_fillet_min_mm(max(joining._thickness(BLADE), joining._thickness(PLATE)))
    assert "AWS D1.1 Table 7.7" in w["source"]


def test_a_bolted_joint_uses_the_iso_273_clearance_hole_and_a_catalogue_length():
    a = {"name": "bracket", "material": "S355", "shape": {"type": "box", "size": [100, 100, 6], "at": [0, 0, 0]},
         "holes": [{"at": [0, 0, 3], "axis": "z", "d": 9}]}
    b = {"name": "frame plate", "material": "S355", "shape": {"type": "box", "size": [100, 100, 6], "at": [0, 0, 6]},
         "holes": [{"at": [0, 0, 3], "axis": "z", "d": 9}]}
    r = joining.plan([a, b], bolt_size="M8")
    j = [x for x in r["joints"] if x["kind"] == "bolted"][0]
    med = (catalog.fastener("M8")["clearance"])["medium"]
    assert j["size"]["value"] == med
    assert "ISO 273" in j["size"]["source"]
    assert r["bolts"] and r["bolts"][0]["size"] == "M8"
    assert all(c["status"] == "pass" for c in r["checks"] if c["check"] == "bolt holes on one axis")


def test_the_workshop_documents_are_built():
    r = joining.plan([BLADE, PLATE, TUBE])
    assert {b["part"] for b in r["bom"]} == {"blade", "mount plate", "push tube"}
    assert all(b["mass_kg"] > 0 for b in r["bom"])
    assert r["mass_total_kg"] > 0
    plate_cut = [c for c in r["cut_list"] if c["part"] == "mount plate"][0]
    assert plate_cut["stock"] == "plate 6.0 mm" and plate_cut["length_mm"] == 120.0
    tube_cut = [c for c in r["cut_list"] if c["part"] == "push tube"][0]
    assert tube_cut["length_mm"] == 800.0
    assert r["welds"] and r["assembly_order"]
    assert r["assembly_order"][0]["what"] == "weld the sub-assemblies"


def test_a_joint_strength_is_not_claimed_without_a_load():
    r = joining.plan([BLADE, PLATE, TUBE])
    assert r["joint_strength"] and all(j["status"] == "not_checked" for j in r["joint_strength"])


def test_a_part_without_a_process_says_so_instead_of_guessing():
    part = {"name": "web", "material": "S355", "shape": {"type": "box", "size": [200, 100, 8], "at": [0, 0, 0]}}
    st = joining.stock_for(part)
    assert st["process"] == "laser_cut" and "process was not stated" in st["needs"]
    assert st["stock"]["thickness_mm"] == 8.0 and "plate stock list" in st["source"]


def test_an_overlap_is_labelled_an_estimate_not_kernel_geometry():
    """An AABB intersection is not the kernel's intersection; on a curved part it is a bound, so it says so."""
    through = {**TUBE, "shape": {**TUBE["shape"], "at": [-600, 0, 6]}}
    r = joining.plan([BLADE, PLATE, through])
    ov = [c for c in r["checks"] if c["check"] == "overlap"]
    assert ov and all(c["source"] == "bounding-box estimate (not kernel geometry)" for c in ov)
    assert all(c["fidelity"] == "estimate" for c in ov)
    assert not any("kernel geometry" in str(c["source"]).lower().replace("not kernel geometry", "") for c in r["checks"])


def _holed(name, z):
    return {"name": name, "material": "S355", "shape": {"type": "box", "size": [100, 100, 6], "at": [0, 0, z]},
            "holes": [{"at": [0, 0, 3], "axis": "z", "d": 9}]}


def test_a_bolt_size_nobody_gave_is_an_assumption_and_a_stage_five_question():
    r = joining.plan([_holed("bracket", 0), _holed("frame plate", 6)])
    j = [x for x in r["joints"] if x["kind"] == "bolted"][0]
    assert j["bolt_selection"]["source"] == "assumed M8 — no load given, change anytime"
    assert "ISO 273" not in j["bolt_selection"]["source"], "ISO 273 is the clearance hole, not the M8 choice"
    assert "ISO 273" in j["size"]["source"]
    assert any("bolt size M8" in a["what"] for a in r["assumptions"])
    q = r["questions"][0]
    assert q["text"] == "Bolt size?" and q["stage"] == 5
    assert any(o.get("recommended") for o in q["options"])
    assert any("Not sure" in o["label"] for o in q["options"])


def test_a_bolt_size_the_engineer_gave_is_not_an_assumption():
    r = joining.plan([_holed("bracket", 0), _holed("frame plate", 6)], bolt_size="M10")
    j = [x for x in r["joints"] if x["kind"] == "bolted"][0]
    assert j["bolt"]["size"] == "M10"
    assert j["bolt_selection"] == {"size": "M10", "source": "given by the engineer"}
    assert not any("bolt size" in a["what"] for a in r["assumptions"])
    assert r["questions"] == []


def test_a_machined_block_is_listed_as_an_assumption():
    block = {"name": "housing", "material": "Al6061-T6",
             "shape": {"type": "box", "size": [120, 120, 120], "at": [0, 0, 0]}}
    r = joining.plan([block, {**_holed("lid", 63), "material": "Al6061-T6"}])
    assert any(a["about"] == "housing" and "machined block is assumed" in a["what"] for a in r["assumptions"])


def test_the_endpoint_answers_with_the_whole_stage():
    code, body = design_ext.handle_post("/join/plan", {"parts": [BLADE, PLATE, TUBE]})
    assert code == 200 and body["success"] is True
    assert body["ok"] is True and body["welds"] and body["bom"]
    assert body["notes"] and "conservative bound" in body["notes"][0]
