"""UI v65 item 2: the assembly plan's orientation contract -- a part is turned, not only moved.

`origin_mm` says where a part goes; `axis` alone still leaves it free to spin about that axis, so the contract now
carries `x_axis` (the part's own +X in assembly coordinates) and the server reports the full right-handed basis
(`axis`, `x_axis`, `y_axis`) that Studio builds the rotation from. A plan written before this contract still
places, and says that its x_axis was derived rather than chosen.
"""
import math

import assembly_ext
import assembly_plan

PARTS = [{"id": "frame", "name": "Frame", "role": "carries the load", "qty": 1, "needs": [], "standard": False},
         {"id": "bracket", "name": "Bracket", "role": "holds the frame", "qty": 1, "needs": [], "standard": False}]


def _plan(frames):
    return {"parts": PARTS, "interfaces": [], "frames": frames, "sources": {}, "questions": []}


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_a_planned_x_axis_is_kept_and_made_perpendicular():
    b = assembly_plan.basis([0, 0, 1], [1, 0, 1])       # sloppy x_axis: 45 degrees out of the plane
    assert b["x_axis_source"] == "plan"
    assert _close(b["axis"], [0, 0, 1]) and _close(b["x_axis"], [1, 0, 0])
    assert _close(b["y_axis"], [0, 1, 0])


def test_a_missing_x_axis_is_derived_deterministically_and_marked():
    b1 = assembly_plan.basis([0, 0, 1])
    b2 = assembly_plan.basis([0, 0, 2])                # same direction, not a unit vector
    assert b1["x_axis_source"] == "derived" and b2["x_axis_source"] == "derived"
    assert _close(b1["x_axis"], [1, 0, 0]) and _close(b1["axis"], b2["axis"])
    assert b1["x_axis"] == b2["x_axis"]


def test_an_x_axis_parallel_to_the_axis_is_refused_with_a_reason():
    plan, errs = assembly_plan.validate(_plan([{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1],
                                               "x_axis": [0, 0, 1]}]))
    assert any("parallel to axis" in e and "frame" in e for e in errs)
    assert plan["frames"] == []                     # a frame that cannot be read is not half placed


def test_a_zero_axis_is_refused_instead_of_placing_the_part_anywhere():
    plan, errs = assembly_plan.validate(_plan([{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 0]}]))
    assert any("zero vector" in e for e in errs) and plan["frames"] == []


def test_an_x_axis_that_is_not_three_numbers_is_refused():
    plan, errs = assembly_plan.validate(_plan([{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1],
                                               "x_axis": [1, 0]}]))
    assert any("x_axis that is not three numbers" in e for e in errs) and plan["frames"] == []


def test_a_validated_plan_and_the_placed_parts_carry_the_whole_basis():
    raw = _plan([{"part_id": "frame", "origin_mm": [0, 0, 0], "axis": [0, 0, 1], "x_axis": [1, 0, 0]},
                 {"part_id": "bracket", "origin_mm": [10, 0, 5], "axis": [0, 1, 1]}])
    plan, errs = assembly_plan.validate(raw)
    assert not errs
    assert [f["x_axis_source"] for f in plan["frames"]] == ["plan", "derived"]
    parts = [{"id": "frame", "name": "Frame", "status": "done", "fidelity": "full"},
             {"id": "bracket", "name": "Bracket", "status": "queued", "fidelity": None}]
    asm = assembly_ext.assembly_of(plan, parts)
    assert asm["frame_contract"] and asm["frames_complete"]
    for p in asm["parts"]:
        z, x, y = p["axis"], p["x_axis"], p["y_axis"]
        for v in (z, x, y):
            assert abs(math.sqrt(_dot(v, v)) - 1.0) < 1e-6                 # unit vectors
        assert abs(_dot(z, x)) < 1e-6 and abs(_dot(z, y)) < 1e-6 and abs(_dot(x, y)) < 1e-6
        assert _close(_cross(z, x), y)                                     # right-handed
    assert asm["parts"][1]["origin_mm"] == [10.0, 0.0, 5.0] and asm["parts"][1]["x_axis_source"] == "derived"


def test_a_frame_written_before_this_contract_still_places_and_says_so():
    fr = assembly_plan.placed_frame({"origin_mm": [0, 0, 0], "axis": [0, 0, 0]})
    assert fr["x_axis_source"] == "default" and fr["frame_error"]
    assert _close(fr["axis"], [0, 0, 1]) and _close(fr["x_axis"], [1, 0, 0])
