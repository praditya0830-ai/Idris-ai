import math

import cad_codegen as cg
from physics import catalog as cat
from physics import parts_library as L


def test_every_template_resolves_with_clean_code():
    q = {"size": "M6", "length": "20", "designation": "6205", "shaft_d": "20", "id": "20", "OD": "10", "ID": "5", "L": "15", "W": "2400"}
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
