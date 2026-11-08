"""Layout v5 (Arya 28 Oct): cover screws all round the joint, nothing overlapping on the covers."""
import math
import numpy as np
from physics.gears import Requirements
from physics.variants import generate_variants
from physics.variant_shafts import shaft_plan
from assembly3d import build_assembly
from physics.bolted_joint import METRIC

S1 = Requirements(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150,
                  wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)
S2 = Requirements(power_kw=11.0, input_rpm=6000, gear_ratio=6, env_w=250, env_h=200, env_d=180,
                  wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                  min_sf_bending=1.4, min_sf_contact=1.1)


def _circle_rect(cx, cy, r, rect):
    x0, y0, w, h = rect
    nx, ny = min(max(cx, x0), x0 + w), min(max(cy, y0), y0 + h)
    return math.hypot(nx - cx, ny - cy) < r


def test_more_than_corner_screws_and_envelope_kept():
    for req in (S1, S2):
        for v in generate_variants(req, compare_basis=False)["variants"]:
            p = shaft_plan(v)
            for wall, j in p["joints"].items():
                assert j["pass"] and j["count"] > 4, (wall, j["count"])
                assert "pitch" in j and j["pitch"]["max_gap_mm"] > 0
                if not j["pitch"]["ok"]:
                    assert j["pitch"]["note"]
            ef = v["envelope_fit"]
            assert ef["fits"] and ef["housing_box_mm"][0] <= req.env_w + 0.5 and ef["housing_box_mm"][1] <= req.env_h + 0.5


def test_screws_plugs_and_dowels_do_not_overlap_on_the_back_cover():
    for req in (S1, S2):
        v = generate_variants(req, compare_basis=False)["variants"][1]
        asm = build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"], v.get("materials"))
        j = asm["layout"]["joints"]["back"]
        head_r = METRIC[j["size"]][2] / 2
        screws = [tuple(p) for p in j["positions"]]
        dowels = [tuple(p) for p in j["dowels"]["positions"]]
        # every pair of screws / dowels is apart
        pts = [(x, y, head_r) for x, y in screws] + [(x, y, j["dowels"]["diameter_mm"] / 2) for x, y in dowels]
        for i in range(len(pts)):
            for k in range(i + 1, len(pts)):
                assert math.hypot(pts[i][0] - pts[k][0], pts[i][1] - pts[k][1]) > pts[i][2] + pts[k][2], (pts[i], pts[k])
        # plugs on the back cover (cylinders along z behind the cover) are clear of screws, dowels and columns
        plug = next(p for p in asm["parts"] if p["kind"] == "plug")
        V = np.reshape(plug["verts"], (-1, 3))
        D = asm["layout"]["housing_box_mm"][2]
        back = V[V[:, 2] > D - 1e-6]
        # cluster plug rims by x
        xs = sorted(set(np.round(back[:, 0] / 4) * 4))
        rects = (j.get("column_rects") or []) + (j.get("pads") or [])
        for cx in {round(float(back[:, 0][np.abs(back[:, 0] - x) < 10].mean()), 1) for x in xs}:
            sel = back[np.abs(back[:, 0] - cx) < 10]
            cy = float(sel[:, 1].mean())
            r = float(np.max(np.hypot(sel[:, 0] - sel[:, 0].mean(), sel[:, 1] - cy)))
            for x, y, rr in pts:
                assert math.hypot(cx - x, cy - y) > r + rr, ("plug on a screw/dowel", cx, cy, x, y)
            for rc in rects:
                assert not _circle_rect(cx, cy, r, rc), ("plug on a screw column", cx, cy, rc)
