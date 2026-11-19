"""V3: the shape match is computed from the outlines, so the numbers below are checks of geometry.

The expected values are analytic, not copied from a previous run: a circle inside the square that binds it
covers pi/4 = 78.5% of it, a triangle covers half of its bounding box, and two identical outlines match 100%.
"""
import math

import http_ext
import vision


def square(x0=0.0, y0=0.0, x1=1.0, y1=1.0):
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


def circle(cx=0.5, cy=0.5, r=0.5, n=180):
    return [[(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]]


def triangle():
    return [[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]


def test_two_identical_outlines_match_completely():
    out = vision.similarity(square(), square())
    assert out["ok"] and out["iou_percent"] == 100.0
    assert out["mismatch_percent"] == 0.0 and out["biggest_region"] is None


def test_a_circle_in_its_bounding_square_is_pi_over_four():
    out = vision.similarity(circle(), square())
    assert abs(out["iou_percent"] - 78.5) < 1.0, out["iou_percent"]          # pi/4 = 78.54 %
    assert abs(out["only_image_percent"] - 21.5) < 1.0
    assert out["only_model_percent"] < 1.0


def test_a_triangle_is_half_of_its_bounding_box():
    out = vision.similarity(triangle(), square())
    assert abs(out["iou_percent"] - 50.0) < 1.0, out["iou_percent"]
    assert abs(out["only_image_percent"] - 50.0) < 1.0


def test_outlines_that_do_not_touch_match_zero_and_say_where():
    out = vision.similarity(square(0, 0, 0.4, 0.4), square(0.6, 0.6, 1.0, 1.0))
    assert out["ok"] and out["iou_percent"] == 0.0 and out["mismatch_percent"] == 100.0
    region = out["biggest_region"]
    # the largest block of disagreement is one of the two squares (0.4 x 0.4 = 16 % of the image), not both
    assert region and 10.0 < region["share_of_image_percent"] < 25.0, region
    assert abs(region["w_percent"] - 40.0) < 5 and abs(region["h_percent"] - 40.0) < 5


def test_a_shifted_copy_shows_a_region_of_disagreement():
    out = vision.similarity(square(0.0, 0.0, 0.6, 1.0), square(0.2, 0.0, 0.8, 1.0))
    assert abs(out["iou_percent"] - 50.0) < 2.0, out["iou_percent"]         # overlap 0.4 of a 0.8 union = 50 %
    region = out["biggest_region"]
    assert region["w_percent"] > 10 and region["h_percent"] > 10
    assert abs(region["h_percent"] - 100.0) < 6, "the disagreement runs the full height"


def test_several_polygons_make_one_shape():
    two = [[(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)], [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)]]
    out = vision.similarity(two, two)
    assert out["iou_percent"] == 100.0


def test_a_bitmap_from_the_browser_is_accepted_and_resampled():
    mask = [[1] * 64 + [0] * 64 for _ in range(128)]              # left half of a 128 x 128 mask
    assert vision.similarity(mask, mask)["iou_percent"] == 100.0
    half = [[1] * 64 + [0] * 64 for _ in range(64)] + [[0] * 128 for _ in range(64)]   # left, top quarter
    out = vision.similarity(mask, half)
    assert abs(out["iou_percent"] - 50.0) < 2.0, out["iou_percent"]


def test_a_mask_and_a_polygon_of_the_same_shape_agree():
    mask = [[1] * 64 + [0] * 64 for _ in range(128)]
    out = vision.similarity(mask, square(0.0, 0.0, 0.5, 1.0))
    assert abs(out["iou_percent"] - 100.0) < 2.0, out["iou_percent"]


def test_an_empty_outline_is_refused_instead_of_scored():
    out = vision.similarity([], square())
    assert out["ok"] is False and "closed outline" in out["error"]


def test_the_route_returns_the_label_the_studio_shows():
    status, body = http_ext.handle_post("/vision/similarity", {
        "model_silhouette": circle(), "image_silhouette": square(), "size": 192})
    assert status == 200 and body["success"]
    assert body["label"] == f"Shape match: {body['iou_percent']}% (silhouette)"
    assert body["resolution"] == 192, "the resolution used is reported, so a number can be reproduced"


def test_a_missing_silhouette_is_a_clear_400():
    status, body = http_ext.handle_post("/vision/similarity", {"model_silhouette": square()})
    assert status == 400 and "image_silhouette" in body["error"]
