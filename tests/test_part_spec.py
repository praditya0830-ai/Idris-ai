"""A1: the engineer's sizes are read from the sentence, with their source. No AI, no invented number."""
import part_spec


def test_plate_and_iso_273_holes():
    s = part_spec.from_message("design a mounting bracket plate 120x80x8 with 4 holes M8")
    d = s["dims"]
    assert (d["length_mm"], d["width_mm"], d["thickness_mm"]) == (120, 80, 8)
    assert s["template"] == "plate" and s["shape_class"] == "prismatic"
    # ISO 273 medium for M8 is 9.0 mm, from the catalog this repo ships
    assert len(s["holes"]) == 4 and all(h["d_mm"] == 9.0 for h in s["holes"])
    src = {n["k"]: n["source"] for n in s["numbers"]}
    assert src["length_mm"] == "you" and src["hole_d_mm"] == "table:ISO 273 medium"
    assert all(h["x_source"] == "computed" and h["method"] for h in s["holes"])


def test_l_bracket_width_is_the_extrusion_not_the_second_leg():
    s = part_spec.from_message("L bracket 60x40x5 width 30")
    d = s["dims"]
    assert s["template"] == "bracket_l"
    assert (d["leg_a_mm"], d["leg_b_mm"], d["thickness_mm"], d["width_mm"]) == (60, 40, 5, 30)


def test_stepped_shaft_reads_both_diameters():
    s = part_spec.from_message("shaft \u00d830 length 120 with \u00d820 x 30 step")
    d = s["dims"]
    assert s["template"] == "shaft_step" and s["shape_class"] == "revolve"
    assert (d["diameter_mm"], d["length_mm"], d["step_diameter_mm"], d["step_length_mm"]) == (30, 120, 20, 30)


def test_flange_holes_sit_on_the_pcd():
    s = part_spec.from_message("flange \u00d8120 x 10 with 6 holes M8 on PCD 100")
    assert s["template"] == "flange" and s["dims"]["thickness_mm"] == 10
    radii = [round((h["x_mm"] ** 2 + h["y_mm"] ** 2) ** 0.5, 6) for h in s["holes"]]
    assert len(radii) == 6 and all(abs(r - 50.0) < 1e-6 for r in radii)


def test_a_message_without_sizes_makes_no_spec():
    assert part_spec.from_message("a nice bracket for the workshop") == {}


def test_plan_part_without_a_source_asks_instead_of_using_it():
    part = {"id": "plate-1", "name": "Mount plate", "role": "holds the motor",
            "dims": {"length_mm": 200, "width_mm": 120, "thickness_mm": 8},
            "sources": {"length_mm": "you", "width_mm": "you"}}
    s = part_spec.from_plan(part, {})
    assert [n["k"] for n in s["numbers"]] == ["length_mm", "width_mm"]
    assert s["questions"] and s["questions"][0]["k"] == "thickness_mm"
