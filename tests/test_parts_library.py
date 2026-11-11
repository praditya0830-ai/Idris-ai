import math

import cad_codegen as cg
from physics import catalog as cat
from physics import parts_library as L


def test_every_template_resolves_with_clean_code():
    q = {"size": "M6", "length": "20", "designation": "6205", "shaft_d": "20", "id": "20", "OD": "10", "ID": "5", "L": "15"}
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
