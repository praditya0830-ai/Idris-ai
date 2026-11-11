import json

import cad_codegen as cg
import llm

GOOD = {"title": "L bracket", "kind": "functional", "process": "fdm", "material": "pla",
        "params": [{"k": "L", "value": 40, "min": 10, "max": 200, "source": "user"},
                   {"k": "t", "value": 3, "min": 1.2, "max": 10, "source": "user", "wall": True},
                   {"k": "W", "value": 20, "min": 10, "max": 100, "source": "user"}],
        "code": "const { L, t, W } = P;\nconst body = makeBox([0,0,0],[L,W,t]).fuse(makeBox([0,0,0],[t,W,L]));\nreturn { body };",
        "expect": [{"label": "Leg X", "axis": "x", "value": 40}, {"label": "Width", "axis": "y", "value": 20}],
        "missing": [], "variants": [{"name": "Sturdy", "params": {"t": 5, "W": 999}}]}
MSG = "bracket L 40 mm tebal 3 mm"


def _fake(monkeypatch, answers):
    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("IDRIS_CAD_ORDER", "gemini")

    def fake(prov, system, user, max_tokens=500, timeout=None):
        calls.append(user)
        a = answers[min(len(calls) - 1, len(answers) - 1)]
        return (a if isinstance(a, str) else json.dumps(a)), "gemini-test"
    monkeypatch.setattr(llm, "call_provider", fake)
    return calls


def test_validate_marks_invented_numbers_and_drops_expect():
    cad, errs, warns = cg.validate(GOOD, MSG)
    assert not errs
    src = {p["k"]: p["source"] for p in cad["params"]}
    assert src == {"L": "user", "t": "user", "W": "default"}         # 20 is not in the sentence
    assert [e["axis"] for e in cad["expect"]] == ["x"]               # width 20 was never asked
    assert cad["variants"][0]["params"] == {"t": 5.0}                 # W = 999 is out of range -> dropped
    assert any("20" in w for w in warns)


def test_forbidden_words_and_missing_return_are_errors():
    for code, word in (("const x = fetch('a'); return { body };", "fetch"),
                       ("const b = P['L']; return { body };", "string"),
                       ("const body = makeBox([0,0,0],[1,1,1]);", "return"),
                       ("const s = drawPointsInterpolation([[0,0],[1,1]]); return { body };", "drawPointsInterpolation")):
        assert any(word in e for e in cg.check_code(code)), code


def test_missing_sizes_send_no_code():
    raw = {"title": "Phone stand", "params": [], "code": "return { body };",
           "missing": [{"k": "T", "question": "Berapa tebal HP?"}]}
    cad, errs, _ = cg.validate(raw, "holder hp")
    assert not errs and cad["code"] is None and cad["missing"][0]["k"] == "T"
    assert "tebal HP" in cg.answer_text({**cad, "read_by": "x"})


def test_generate_retries_once_after_bad_code(monkeypatch):
    bad = {**GOOD, "code": "return { body: fetch('x') };"}
    calls = _fake(monkeypatch, [bad, GOOD])
    r = cg.generate(MSG)
    assert r["ok"] and len(calls) == 2 and "rejected by the server check" in calls[1]
    assert r["cad"]["read_by"].startswith("gemini")


def test_repair_loop_sends_error_and_stops_after_two(monkeypatch):
    calls = _fake(monkeypatch, [GOOD])
    ctx = {"cad_error": {"title": "L bracket", "code": "old code", "error": "BRep_API: command not done",
                         "params": {"L": 40}, "attempt": 1}}
    r = cg.generate(MSG, ctx)
    assert r["ok"] and "BRep_API" in calls[0] and "old code" in calls[0]
    r = cg.generate(MSG, {"cad_error": {**ctx["cad_error"], "attempt": 3}})
    assert not r["ok"] and len(calls) == 1                           # no AI call after 2 repairs


def test_pending_cad_answer_counts_as_user_numbers(monkeypatch):
    stand = {"title": "Phone stand", "params": [{"k": "T", "value": 9, "min": 5, "max": 20, "source": "user"}],
             "code": "const { T } = P; const body = makeBox([0,0,0],[T,T,T]); return { body };", "missing": []}
    calls = _fake(monkeypatch, [stand])
    r = cg.generate("9 mm", {"pending_cad": {"title": "Phone stand", "request": "holder hp",
                                             "missing": [{"k": "T", "question": "tebal?"}]}})
    assert r["ok"] and r["cad"]["params"][0]["source"] == "user" and "holder hp" in calls[0]


def test_routing_words():
    assert cg.looks_like_object_request("tolong desain roket model 600 mm")
    assert cg.looks_like_object_request("holder hp")
    assert not cg.looks_like_object_request("gearbox 5 kW rasio 8")
    assert not cg.looks_like_object_request("buat lebih ringan", has_gearbox=True)


def test_acceptance_list_is_well_formed():
    data = json.load(open("tests/cad_acceptance.json", encoding="utf-8"))
    assert len(data["cases"]) == 30
    for c in data["cases"]:
        assert c["q"] and isinstance(c["expect_mm"], dict) and isinstance(c["must_ask"], list)
        assert cg.looks_like_object_request(c["q"]), c["q"]


PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _cad():
    cad, errs, _ = cg.validate(GOOD, MSG)
    assert not errs
    return {**cad, "read_by": "x"}


def _vision(monkeypatch, answer):
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("IDRIS_VISION_ORDER", "claude")

    def fake(prov, system, text, images, max_tokens=4000, timeout=90.0):
        seen.update(images=images, text=text)
        return json.dumps(answer), "claude-test"
    monkeypatch.setattr(llm, "call_vision", fake)
    return seen


def test_review_ok_verdict(monkeypatch):
    seen = _vision(monkeypatch, {"verdict": "ok", "critique": "Clean bracket."})
    r = cg.review(MSG, _cad(), [{"view": "iso", "png": "data:image/png;base64," + PNG}], {"bbox": [40, 20, 40]})
    assert r["ok"] and r["verdict"] == "ok" and seen["images"][0]["mime"] == "image/png"
    assert "bbox" in seen["text"]


def test_review_revision_must_keep_user_sizes(monkeypatch):
    changed = {**GOOD, "verdict": "revise", "critique": "longer",
               "expect": [{"label": "Leg X", "axis": "x", "value": 50}]}
    _vision(monkeypatch, changed)
    r = cg.review(MSG, _cad(), [{"view": "iso", "png": PNG}])
    assert not r["ok"] and "changed sizes" in r["attempts"][0]["error"]
    better = {**GOOD, "verdict": "revise", "critique": "Rounded the corner."}
    _vision(monkeypatch, better)
    r = cg.review(MSG, _cad(), [{"view": "iso", "png": PNG}])
    assert r["ok"] and r["verdict"] == "revise" and r["cad"]["code"]


def test_review_limits():
    assert cg.review(MSG, _cad(), [{"png": PNG}], round_=3)["verdict"] == "stop"
    imgs, errs = cg.clean_images([{"png": "not base64 !!"}, {"png": "A" * (cg.MAX_IMAGE_B64 + 4)}])
    assert not imgs and len(errs) == 2


def test_recipes_pass_server_check_and_are_in_prompt():
    import cad_recipes
    for r in cad_recipes.RECIPES:
        assert cg.check_code(r["code"]) == [], r["id"]
    sys_ = cg._system("en")
    assert "tangent ogive" in sys_ and "ENGLISH" in sys_


def test_german_sentence_gives_english_answer(monkeypatch):
    stand = {"title": "Phone stand", "kind": "functional", "process": "fdm", "material": "pla",
             "params": [{"k": "T", "label": "Phone thickness", "value": 9, "min": 5, "max": 20, "source": "user"},
                        {"k": "Hp", "label": "Phone height", "value": 160, "min": 100, "max": 250, "source": "user"}],
             "code": "const { T, Hp } = P; const body = makeBox([0,0,0],[T,Hp,T]); return { body };", "missing": []}
    calls = _fake(monkeypatch, [stand])
    msg = "Handyhalter, 9 mm dick, 160 mm hoch"
    assert cg.looks_like_object_request(msg)
    r = cg.generate(msg, reply_language="en")
    assert r["ok"] and [p["source"] for p in r["cad"]["params"]] == ["user", "user"]
    assert r["answer"].startswith("Phone stand: CAD code written by") and "From you: Phone thickness 9 mm" in r["answer"]
    assert "ENGLISH" not in calls[0]           # language rule lives in the system prompt, not in the user block


def test_missing_question_heading_english():
    cad, _, _ = cg.validate({"title": "Phone stand", "params": [], "code": "",
                             "missing": [{"k": "T", "question": "How thick is the phone?"}]}, "uchwyt na telefon")
    assert cg.answer_text({**cad, "read_by": "x"}, lang="en", message="uchwyt na telefon").startswith(
        "Before designing the Phone stand")


def test_gearbox_reader_gets_english_rule(monkeypatch):
    seen = {}
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "gemini")

    def fake(prov, system, user, max_tokens=500, timeout=None):
        seen["user"] = user
        return json.dumps({"intent": "set_params", "params": {"gear_ratio": 8}, "answer": "Ratio 8.", "language": "id"}), "g"
    monkeypatch.setattr(llm, "call_provider", fake)
    r = llm.interpret(None, "gearbox rasio 8", reply_language="en")
    assert "REPLY LANGUAGE: English" in seen["user"] and r["language"] == "en"


def test_holes_are_validated_against_catalog():
    raw = {**GOOD, "holes": [{"x": 26, "y": 10, "z_top": 3, "d": 4.5, "kind": "clearance", "M": "M4", "thru": True},
                             {"x": 10, "y": 10, "z_top": 3, "d": 5.0, "kind": "clearance", "M": "M4"},
                             {"x": 5, "y": 5, "z_top": 3, "d": 3.3, "kind": "tapped", "M": "M4", "thru": False, "depth": 8},
                             {"y": 1}]}
    cad, errs, warns = cg.validate(raw, MSG)
    assert not errs and len(cad["holes"]) == 3
    assert [h["catalog_ok"] for h in cad["holes"]] == [True, False, True]
    assert any("not the ISO 273 clearance for M4" in w for w in warns) and any("hole 4 dropped" in w for w in warns)


def test_free_form_design_requests_route_to_cad():
    for q in ("Buatkan aku design kunci", "Buatkan aku design frame sepeda", "design a bottle opener", "bikin desain asbak"):
        assert cg.looks_like_object_request(q), q
    for q in ("buat lebih ringan", "make it stronger", "buat gearbox 5 kW", "design 3 kW 3000 rpm"):
        assert not cg.looks_like_object_request(q), q
    nums = cg.message_numbers("29 inch, 20 inch, 135 mm")
    assert {736.6, 508.0, 135.0} <= nums


def test_parts_not_in_library_go_to_cad_and_open_gearbox_is_safe():
    for q in ("circlip 10 mm", "dowel pin 6x20", "spur gear module 2 20 teeth", "shaft 20 mm panjang 100"):
        assert cg.looks_like_object_request(q), q
    assert not cg.looks_like_object_request("make the cover thicker", has_gearbox=True)
    assert not cg.looks_like_object_request("ganti seal", has_gearbox=True)
    assert cg.looks_like_object_request("design a phone holder", has_gearbox=True)
