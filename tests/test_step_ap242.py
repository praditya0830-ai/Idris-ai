"""STEP AP242 export (P5a): named products, colours, units, assembly tree -- and the AP214 export untouched."""
import numpy as np

from assembly3d import build_box
from engine_routes import handle_post
from physics.step_ap242 import read_back, step_references_ok, write_step_ap242
from physics.step_export import check_step, step_solid_volumes, write_step


def _mesh(x0, x1, y0, y1, z0, z1):
    v, f = build_box(x0, x1, y0, y1, z0, z1)
    return {"verts": v, "faces": f, "volume_mm3": (x1 - x0) * (y1 - y0) * (z1 - z0)}


PLATE = {"name": "plate", "colour": [0.8, 0.2, 0.2], **_mesh(0, 40, 0, 20, 0, 10)}
BLOCK = {"name": "block", "colour": [0.2, 0.4, 0.8], "translation": [50, 0, 0], **_mesh(0, 10, 0, 10, 0, 10)}


def test_the_header_is_ap242():
    text, stats = write_step_ap242([PLATE], title="bracket")
    assert "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF" in text
    assert stats["format"] == "STEP AP242" and stats["units"] == "mm, rad"
    assert "SI_UNIT(.MILLI.,.METRE.)" in text


def test_every_product_is_named_and_carries_its_colour():
    text, stats = write_step_ap242([PLATE, BLOCK], title="assembly")
    back = read_back(text)
    assert "plate" in back["products"] and "block" in back["products"]
    assert back["styled_items"] == 2 and stats["styled_items"] == 2
    assert "COLOUR_RGB('',0.8,0.2,0.2)" in text


def test_the_solids_read_back_with_the_same_volume_as_the_mesh():
    text, stats = write_step_ap242([PLATE, BLOCK], title="assembly")
    vols = step_solid_volumes(text)
    assert stats["solids"] == 2 and len(vols) == 2
    got = sorted(abs(v) for v in vols.values())
    want = sorted([PLATE["volume_mm3"], BLOCK["volume_mm3"]])
    assert np.allclose(got, want, rtol=1e-6), (got, want)


def test_the_assembly_tree_has_one_instance_per_part_with_a_placement():
    text, stats = write_step_ap242([PLATE, BLOCK], title="assembly")
    assert text.count("NEXT_ASSEMBLY_USAGE_OCCURRENCE(") == 2
    assert stats["instances"] == 2
    assert "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(" in text
    assert "CARTESIAN_POINT('',(50.,0.,0.))" in text


def test_every_reference_resolves_and_every_edge_is_used_twice():
    text, _ = write_step_ap242([PLATE, BLOCK], title="assembly")
    ok, detail = step_references_ok(text)
    assert ok, detail
    assert detail["dangling_references"] == []
    assert check_step(text)[0]


def test_a_rotated_part_keeps_its_orientation_in_the_placement():
    rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    text, _ = write_step_ap242([{**PLATE, "rotation": rot, "translation": [5, 6, 7]}], title="rotated")
    assert "CARTESIAN_POINT('',(5.,6.,7.))" in text
    assert "DIRECTION('',(0.,0.,1.))" in text          # z axis is unchanged by a rotation about z
    assert "DIRECTION('',(0.,1.,0.))" in text          # x axis of the part now points along +y


def test_the_route_answers_with_the_file_and_its_self_checks():
    code, body = handle_post("/exchange/step-ap242", {"title": "bracket", "parts": [PLATE, BLOCK]})
    assert code == 200 and body["success"] is True
    assert body["step"].startswith("ISO-10303-21;") and "AP242" in body["format"]
    assert all(c["ok"] for c in body["checks"])
    assert body["read_back"]["instances"] == 2


def test_the_route_refuses_an_empty_part_list():
    code, body = handle_post("/exchange/step-ap242", {"parts": []})
    assert code == 400 and body["success"] is False
    code, body = handle_post("/exchange/step-ap242", {"parts": [{"name": "x"}]})
    assert code == 400 and "verts" in body["error"]


def test_the_ap214_export_is_unchanged():
    text, stats = write_step(write_parts := [{"name": "plate", "verts": PLATE["verts"], "faces": PLATE["faces"]}])
    assert "10303 214" in text and "AP242" not in text
    assert stats[0]["faces"] == 6                    # a box merges to six planar faces
    assert abs(list(step_solid_volumes(text).values())[0]) == 8000
