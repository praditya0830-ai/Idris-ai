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


def test_a_tube_that_passes_through_the_plate_is_detected_and_moved_out():
    through = {**TUBE, "shape": {**TUBE["shape"], "at": [-600, 0, 6]}}
    r = joining.plan([BLADE, PLATE, through])
    assert any(i["code"] == "overlap" for i in r["issues"])
    assert any("overlap" in f for f in r["fixes"])
    assert all(c["status"] == "pass" for c in r["checks"] if c["check"] == "overlap")
    assert r["ok"] is True
    tube = [p for p in r["parts"] if p["name"] == "push tube"][0]
    assert tube["process"] == "tube_cut"


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


def test_the_endpoint_answers_with_the_whole_stage():
    code, body = design_ext.handle_post("/join/plan", {"parts": [BLADE, PLATE, TUBE]})
    assert code == 200 and body["success"] is True
    assert body["ok"] is True and body["welds"] and body["bom"]
    assert body["notes"] and "conservative bound" in body["notes"][0]
