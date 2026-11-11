from physics import catalog as c


def test_iso273_values_not_bd_warehouse():
    assert c.fastener("M5")["clearance"] == {"fine": 5.3, "medium": 5.5, "coarse": 5.8}
    assert c.fastener("M6")["clearance"] == {"fine": 6.4, "medium": 6.6, "coarse": 7.0}
    assert c.fastener("M12")["clearance"]["medium"] == 13.5        # bd_warehouse says 14 (not ISO 273)
    assert c.fastener("M16")["clearance"]["coarse"] == 18.5        # bd_warehouse says 19
    assert c.fastener("m8x1.25")["tap_drill"] == 6.8 and c.norm_size("M7.5") is None


def test_auto_assembly_catches_deliberate_errors():
    r = c.bolt_check({"size": "M6", "hole_d": 5.5})               # M5 hole, M6 bolt
    assert not r["ok"] and "clearance hole of M5" in r["findings"][0]["message"]
    r = c.bolt_check({"size": "M5", "hole_d": 5.5, "clamp_mm": 10, "bolt_length_mm": 12, "thread_material": "aluminium"})
    assert not r["ok"] and "engagement" in r["findings"][0]["message"] and "20 mm" in r["findings"][0]["fix"]
    r = c.bolt_check({"size": "M8", "hole_d": 6.8, "hole_kind": "tapped"})
    assert r["ok"]
    r = c.bolt_check({"size": "M8", "hole_d": 8.5, "hole_kind": "tapped"})
    assert not r["ok"] and "M10" in r["findings"][0]["message"]
    r = c.bolt_check({"size": "M6", "hole_d": 6.6, "clamp_mm": 12, "bolt_length_mm": 20, "nut": True, "washers": 1})
    assert not r["ok"] and "25 mm" in r["findings"][0]["fix"]      # 12 + 1.8 + 5.2 + 2 = 21 -> 25


def test_correct_assembly_has_no_false_warning():
    for p in ({"size": "M5", "hole_d": 5.5, "clamp_mm": 6, "bolt_length_mm": 16, "thread_material": "aluminium"},
              {"size": "M6", "hole_d": 6.6, "clamp_mm": 10, "bolt_length_mm": 25, "nut": True},
              {"size": "M4", "hole_d": 4.3, "clamp_mm": 3, "bolt_length_mm": 8, "thread_material": "steel"}):
        r = c.bolt_check(p)
        assert r["ok"] and not r["findings"], (p, r["findings"])


def test_bearings_labelled_unverified():
    b = c.bearings(8, 10)
    assert {x["designation"] for x in b} == {"608", "629", "6000", "6200"} and not any(x["verified"] for x in b)


def test_ui_json_keys_for_m3_to_m16():
    for size in ("M3", "M4", "M5", "M6", "M8", "M10", "M12", "M14", "M16"):
        f = c.fastener(size)
        assert {"size", "pitch", "tap_drill", "clearance", "socket_head", "nut", "washer"} <= set(f), size
        assert set(f["clearance"]) == {"fine", "medium", "coarse"}
        assert {"dk", "k", "s", "lengths"} <= set(f["socket_head"]) and f["socket_head"]["lengths"]
        assert {"m", "s"} <= set(f["nut"]) and {"d1", "d2", "h"} <= set(f["washer"])
    assert c.fastener("M6")["washer"]["h"] == 1.6 and c.fastener("M6")["socket_head"]["lengths"][0] == 8
    assert c.size_for_hole(5.5)[0] == {"size": "M5", "kind": "clearance", "series": "medium"}


def test_finding_codes_and_lower_hole():
    r = c.bolt_check({"size": "M8", "hole_d": 9, "lower_hole_d": 8.5, "lower_hole_kind": "tapped"})
    assert [x["code"] for x in r["findings"]] == ["lower_tap_mismatch"] and "M10" in r["findings"][0]["message"]
    r = c.bolt_check({"size": "M6", "hole_d": 5.5})
    assert r["findings"][0]["code"] == "clearance_small" and r["findings"][0]["fix_value"] == 6.6
    r = c.bolt_check({"size": "M5", "hole_d": 5.5, "clamp_mm": 10, "bolt_length_mm": 12, "thread_material": "aluminium"})
    assert r["findings"][0]["code"] == "engagement_short" and r["findings"][0]["fix_value"] == 20
    r = c.bolt_check({"size": "M5", "clamp_mm": 6, "bolt_length_mm": 25, "thread_material": "steel", "thread_depth_mm": 10})
    assert r["findings"][0]["code"] == "too_long_for_thread" and r["findings"][0]["fix_value"] == 16
    r = c.bolt_check({"size": "M6", "hole_d": 6.6, "lower_hole_d": 6.6, "lower_hole_kind": "clearance",
                      "clamp_mm": 12, "bolt_length_mm": 20, "washers": 1})
    assert r["findings"][0]["code"] == "nut_too_short" and r["findings"][0]["fix_value"] == 25   # 12+1.6+5.2+2
    assert all(x["code"] in c.CODES for x in r["findings"])


def test_joint_strength_matches_gearbox_joint_constant():
    # M8 into steel, 12 mm cover: C = 0.2049 is the existing /verify benchmark (shigley-8-22-joint-constant-m8)
    r = c.bolt_check({"size": "M8", "hole_d": 9, "clamp_mm": 12, "bolt_length_mm": 30, "thread_material": "steel",
                      "load_n": 5000})
    s = r["strength"]
    assert s["label"] == "screening" and abs(s["joint_constant"] - 0.2049) < 1e-4
    assert abs(s["preload_n"] - 0.75 * 580 * s["tensile_area_mm2"]) < 1
    r = c.bolt_check({"size": "M4", "hole_d": 4.5, "clamp_mm": 5, "bolt_length_mm": 10, "thread_material": "steel",
                      "load_n": 8000})
    assert not r["ok"] and {"strength_yield", "strength_separation"} <= {x["code"] for x in r["findings"]}
    r = c.bolt_check({"size": "M5", "clamp_mm": 6, "bolt_length_mm": 16, "thread_material": "pla", "load_n": 100})
    assert r["strength"]["label"] == "not computed" and r["findings"][-1]["code"] == "strength_not_valid"


def test_bigger_bearings_have_sources():
    b = {x["designation"]: x for x in c.bearings(12, 30)}
    assert b["6205"]["d"] == 25 and b["6205"]["C_kN"] == 15.4 and b["6205"]["verified"]
    assert all(x["source"] for x in b.values())
