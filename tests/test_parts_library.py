import math

import cad_codegen as cg
from physics import catalog as cat
from physics import parts_library as L


def test_every_template_resolves_with_clean_code():
    q = {"size": "M6", "length": "20", "designation": "6205", "shaft_d": "20", "id": "20", "OD": "10", "ID": "5", "L": "15", "W": "2400", "m": "2", "z": "32", "b": "15"}
    for tid in L.TEMPLATES:
        r = L.resolve(tid, q)
        assert r["ok"], (tid, r)
        c = r["cad"]
        assert cg.check_code(c["code"]) == [], tid
        names = {n.strip() for n in c["code"].split("= P;")[0].split("{")[-1].split("}")[0].split(",")}
        assert names == {p["k"] for p in c["params"]}, tid       # code reads exactly the resolved params
        assert c["expect"] and c["library"]["sources"]


def test_sizes_come_from_tables():
    c = L.resolve("socket_head_cap_screw", {"size": "M6", "length": 20})["cad"]
    p = {x["k"]: x["value"] for x in c["params"]}
    assert (p["dk"], p["k"], p["s"]) == cat.SHCS_ISO4762["M6"] and p["L"] == 20
    assert c["expect"][0]["value"] == 26                                   # 20 + head 6
    c = L.resolve("hex_bolt", {"size": "M10", "length": 40})["cad"]
    assert {x["k"]: x["value"] for x in c["params"]}["s"] == 16             # ISO 4017, not DIN 933 (17)
    c = L.resolve("parallel_key", {"shaft_d": 20, "length": 32})["cad"]
    assert [x["value"] for x in c["params"]] == [6, 6, 32] and c["library"]["keyway"]["shaft_depth_t1"] == 3.5
    c = L.resolve("o_ring", {"id": 20, "w": 2.62})["cad"]
    assert math.isclose(c["expect"][0]["value"], 20.29 + 2 * 2.62)


def test_refuses_non_standard_or_missing():
    r = L.resolve("hex_bolt", {"size": "M6", "length": 21})
    assert not r["ok"] and "next standard length is 25" in r["errors"][0] and r["suggest"]["suggest_length"] == 25
    r = L.resolve("socket_head_cap_screw", {"size": "M6"})
    assert r["missing"][0]["k"] == "length" and "8, 10, 12" in r["missing"][0]["question"]
    assert not L.resolve("ball_bearing", {"designation": "6310"})["ok"]
    assert "2·b" in L.resolve("parallel_key", {"shaft_d": 40, "length": 16})["errors"][0]


def test_sentences():
    assert L.match_sentence("baut M6x20") == ("hex_bolt", {"size": "M6", "length": "20"})
    assert L.match_sentence("socket head M5 panjang 16")[0] == "socket_head_cap_screw"
    assert L.match_sentence("ring M6")[0] == "washer" and L.match_sentence("o-ring 20x2.62")[0] == "o_ring"
    assert L.match_sentence("harga murah") is None                           # 'mur' only as a whole word
    assert L.match_sentence("bushing for bearing 608") is None               # designed around a bearing -> AI
    assert L.match_sentence("holder hp") is None


def test_follow_up_answers():
    assert L.follow_up("baut", "M8 x 30", [{"k": "size"}]) == ("hex_bolt", {"size": "M8", "length": "30"})
    assert L.follow_up("socket head M5", "16", [{"k": "length"}]) == ("socket_head_cap_screw", {"size": "M5", "length": "16"})
    assert L.follow_up("holder hp", "9", [{"k": "T"}]) is None


def test_plow_blade_geometry_is_exact_and_width_is_a_parameter():
    """Weldment family: numbers are closed-form geometry, checked here against an independent calculation."""
    r = L.resolve("plow_blade", {"W": 2400})
    assert r["ok"]
    c = r["cad"]
    lib = c["library"]
    assert c["process"] == "weld" and c["kind"] == "functional" and "NOT checked" in c["note"]
    # H 900, c 162 -> R = (900^2/4 + 162^2) / (2*162) = 706 mm exactly
    assert math.isclose(lib["radius"], 706.0, abs_tol=0.05) and lib["ribs"] == 5
    a = math.asin(900 / (2 * 706.0))
    assert math.isclose(lib["developed_length"], 2 * a * 706.0, abs_tol=0.06)
    # independent mass: annulus-sector areas by the r2^2 - r1^2 form
    sheet = a * (709.0 ** 2 - 703.0 ** 2) * 2400
    rib = a * (829.0 ** 2 - 709.0 ** 2) * 8
    edge = 150 * 16 * 2400
    assert math.isclose(lib["mass_kg"], 7850e-9 * (sheet + 5 * rib + edge), abs_tol=0.06)
    assert math.isclose(sum(x["qty"] * x["mass_kg"] for x in lib["cut_list"]), lib["mass_kg"], abs_tol=0.1)
    exp = {e["axis"]: e["value"] for e in c["expect"]}
    assert exp["y"] == 2400 and math.isclose(exp["z"], 50 + 450 + 829.0 * 900 / 1412.0, abs_tol=0.01)
    # a wider blade: same family, more ribs, heavier; only W changed
    c2 = L.resolve("plow_blade", {"W": 3000})["cad"]
    assert c2["code"] == c["code"] and c2["library"]["ribs"] == 6 and c2["library"]["mass_kg"] > lib["mass_kg"]
    w = [p for p in c2["params"] if p["k"] == "W"][0]
    assert (w["value"], w["min"], w["max"], w["source"]) == (3000, 1000, 4500, "user")
    assert [p["source"] for p in c2["params"] if p["k"] == "H"] == ["default"]


def test_plow_blade_limits_and_questions():
    assert L.resolve("plow_blade", {})["missing"][0]["k"] == "W"
    assert "between 1000 and 4500" in L.resolve("plow_blade", {"W": 9000})["errors"][0]
    assert "half circle" in L.resolve("plow_blade", {"W": 2400, "H": 600, "c": 300})["errors"][0]


def test_plow_blade_from_a_sentence():
    assert L.match_sentence("snow plow blade 2.4 m wide for a pickup truck") == ("plow_blade", {"W": "2400"})
    assert L.match_sentence("bilah bajak salju lebar 3000 mm tinggi 800 tebal 5 mm") == (
        "plow_blade", {"W": "3000", "H": "800", "t": "5"})
    assert L.match_sentence("Design a snow plow blade 3200 mm") == ("plow_blade", {"W": "3200"})
    assert L.match_sentence("a snow plow blade")[1] == {}
    tid, q = L.follow_up("a snow plow blade", "2800", [{"k": "W"}])
    assert (tid, q) == ("plow_blade", {"W": "2800"})
    assert "rebuilds" in L.chat_answer("plow_blade", L.resolve(tid, q)) and "Cut list" in L.chat_answer("plow_blade", L.resolve(tid, q))


def test_spur_gear_from_the_sentence_that_failed_on_live():
    """13 Nov: the AI could not write this gear (answer cut off). A standard machine element belongs in the library."""
    msg = ("Design a spur gear, module 2, 32 teeth, 20 degree pressure angle, face width 15 mm, bore 12 mm with a "
           "4 mm keyway and a hub 30 mm diameter, 10 mm long, steel")
    tid, q = L.match_sentence(msg)
    assert tid == "spur_gear" and q == {"m": "2", "z": "32", "alpha": "20", "b": "15", "bore": "12", "hub_d": "30",
                                        "hub_l": "10", "keyway": "true", "key_width": "4"}
    r = L.resolve(tid, q)
    assert r["ok"], r
    c, lib = r["cad"], r["cad"]["library"]
    assert (lib["pitch_d"], lib["tip_d"], lib["root_d"]) == (64, 68, 59)             # d = m z, da = d + 2m, df = d - 2.5m
    assert math.isclose(lib["base_d"], 64 * math.cos(math.radians(20)), abs_tol=1e-4)
    assert math.isclose(lib["tooth_thickness_pitch"], math.pi, abs_tol=1e-4)          # pi m / 2
    p = {x["k"]: x for x in c["params"]}
    assert (p["kw"]["value"], p["kt"]["value"], p["kw"]["source"]) == (4, 1.8, "table")   # DIN 6885 for a 12 mm shaft
    assert p["m"]["source"] == "user" and "NOT checked" in c["note"]
    assert {e["axis"]: e["value"] for e in c["expect"]}["z"] == 25
    text = L.chat_answer(tid, r)
    assert "pitch diameter 64 mm" in text and "DIN 6885-1" in text


def test_spur_gear_profile_is_a_true_involute():
    """Rebuild the profile the way the CAD code does and measure it."""
    m, z, al, n = 2.0, 32, 20.0, 6
    a = math.radians(al)
    rp, rb, ra, rf = m * z / 2, m * z / 2 * math.cos(a), m * z / 2 + m, m * z / 2 - 1.25 * m
    inv = lambda x: math.tan(x) - x
    half = math.pi / (2 * z) + inv(a)
    for i in range(n + 1):                                   # every flank point lies on the involute of the base circle
        r = max(rb, rf) + (ra - max(rb, rf)) * i / n
        t = half - inv(math.acos(min(1, rb / r)))
        x, y = r * math.cos(t), r * math.sin(t)
        # involute property: the tangent length from the point to the base circle equals the unrolled arc
        phi = math.acos(rb / r)
        roll = phi + (half - t)                               # angle of the tangent point from the flank origin
        assert math.isclose(math.sqrt(max(x * x + y * y - rb * rb, 0.0)), rb * math.tan(phi), abs_tol=1e-6)
        assert math.isclose(math.tan(phi), roll, abs_tol=1e-9)
    assert math.isclose(2 * rp * (half - inv(a)), math.pi * m / 2, abs_tol=1e-12)        # thickness at the pitch circle
    g = L.gear_geometry(m, z, al)
    assert g["tip_thickness"] > 0.25 * m and math.isclose(g["z_min_no_undercut"], 17.1, abs_tol=0.01)


def test_spur_gear_refusals_and_routing():
    assert "between 9 and 150" in L.resolve("spur_gear", {"m": 2, "z": 8, "b": 10})["errors"][0]
    assert [x["k"] for x in L.resolve("spur_gear", {"m": 2})["missing"]] == ["z", "b"]
    assert "not 5 mm" in L.resolve("spur_gear", {"m": 2, "z": 32, "b": 15, "bore": 12, "keyway": "true", "key_width": 5})["errors"][0]
    assert "keyway needs a bore" in L.resolve("spur_gear", {"m": 2, "z": 32, "b": 15, "keyway": "true"})["errors"][0]
    assert "rim" in L.resolve("spur_gear", {"m": 2, "z": 12, "b": 15, "bore": 18})["errors"][0]
    assert "undercut" in L.resolve("spur_gear", {"m": 2, "z": 12, "b": 15})["cad"]["library"]["notes"][0]
    assert "ISO 54" in L.resolve("spur_gear", {"m": 2.2, "z": 30, "b": 15})["cad"]["library"]["notes"][0]
    assert L.match_sentence("gearbox 15 hp ratio 8 with spur gears") is None          # the gearbox workflow keeps these
    assert L.match_sentence("gear pump housing 60 mm") is None
    assert L.match_sentence("roda gigi lurus modul 3 20 gigi tebal 25 lubang 20") == (
        "spur_gear", {"m": "3", "z": "20", "b": "25", "bore": "20"})
