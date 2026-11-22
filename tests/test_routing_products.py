"""Whole products are never designed as one solid, and a new product does not inherit the old design's context.

The three findings from the 21 Nov evening test:
  * "buatkan saya design konsep sepeda" must go to brief/assembly, never free_design;
  * a missing size is asked in engineer words, never as `body_dimension` / `table_source`;
  * a question carries no `fidelity`, because no geometry exists yet;
  * "meja 120×60" followed by "design an airplane concept" starts a new design.
"""
import os

import pytest

import ace_core
import cad_fallback
import cad_router


# --------------------------------------------------------------------------------------------- products
@pytest.mark.parametrize("sentence,product", [
    ("buatkan saya design konsep sepeda", "bicycle"),
    ("buatkan saya design konsep sepeda untuk pengendara 170 cm", "bicycle"),
    ("design a bicycle concept", "bicycle"),
    ("design an airplane concept", "airplane"),
    ("buatkan pesawat", "airplane"),
    ("design a plane", "airplane"),
    ("desain mobil listrik", "car"),
    ("buat drone untuk inspeksi", "drone"),
    ("meja 120x60", "table"),
    ("kursi kerja", "chair"),
    ("robot arm 600 mm", "robot"),
])
def test_product_of_names_the_product(sentence, product):
    assert cad_router.product_of(sentence) == product


@pytest.mark.parametrize("sentence", [
    "bracket for a 6205 bearing",
    "bicycle frame 540 mm",
    "make it lighter",
    "80 mm",
    "reference plane at 20 mm",
    "plane strain block",
    "table frame 900 mm",
    "",
])
def test_product_of_ignores_parts_and_follow_ups(sentence):
    assert cad_router.product_of(sentence) is None


def test_product_route_sends_a_bare_product_to_brief():
    r = cad_router.product_route("buatkan saya design konsep sepeda", "free_design")
    assert r["route"] == "brief" and r["kind"] == "assembly"
    assert r["brief"]["object"] == "bicycle"
    assert "product" in r["read_by"]


def test_product_route_sends_a_sized_product_to_assembly():
    r = cad_router.product_route("whole bicycle for a 170 cm rider, 26 inch wheels", "free_design")
    assert r["route"] == "assembly" and r["kind"] == "assembly"


def test_product_route_never_touches_library_gearbox_or_assembly():
    for route_name in ("library", "gearbox", "assembly", "brief"):
        assert cad_router.product_route("design a bicycle", route_name) is None


def test_route_without_a_reader_still_protects_products(monkeypatch):
    monkeypatch.setenv("IDRIS_AI_ROUTER", "0")
    assert cad_router.route("buatkan saya design konsep sepeda") is None
    # and the rule main.py applies when the reader is off:
    r = cad_router.product_route("buatkan saya design konsep sepeda", None)
    assert r and r["route"] == "brief"


# --------------------------------------------------------------------------------------------- context reset
def test_a_new_product_drops_the_old_context():
    ctx = {"pending_cad": {"request": "meja 120x60", "missing": [{"k": "H"}]}}
    r = cad_router.topic_reset("design an airplane concept", ctx)
    assert r and r["to"] == "airplane"
    assert "different product" in r["reason"]


def test_the_same_product_keeps_the_context():
    ctx = {"pending_cad": {"request": "sepeda untuk pengendara 170 cm", "missing": [{"k": "wheel"}]}}
    assert cad_router.topic_reset("buatkan sepeda lengkap", ctx) is None


def test_a_size_answer_is_not_a_new_topic():
    ctx = {"pending_cad": {"request": "bracket 80x40", "missing": [{"k": "T"}]}}
    assert cad_router.topic_reset("80 mm", ctx) is None
    assert cad_router.topic_reset("everything", ctx) is None


def test_a_product_after_a_part_keeps_the_open_part():
    # Review of PR #31, item 5: a context that cannot be read as a product is never dropped. The part stays open and
    # the sentence is answered on top of it.
    ctx = {"current_cad": {"request": "bracket untuk bearing 6205"}}
    assert cad_router.topic_reset("design a plane concept", ctx) is None


# ---------------------------------------------------------------- review of PR #31: the product reader is narrow
def test_a_gearbox_is_not_dropped_by_a_sentence_about_a_motor_part():
    ctx = {"current_cad": {"request": "gearbox housing 200 mm"}}
    assert cad_router.topic_reset("tambah dudukan motor listrik", ctx) is None


def test_a_reset_keeps_the_old_project_instead_of_discarding_it():
    ctx = {"pending_cad": {"request": "sepeda untuk pengendara 170 cm", "missing": [{"k": "wheel"}]}}
    r = cad_router.topic_reset("buat meja", ctx)
    assert r and r["to"] == "table"
    assert r["previous"]["from"] == "pending_cad"
    assert r["previous"]["request"] == "sepeda untuk pengendara 170 cm"
    assert r["previous"]["missing"] == ["wheel"]


@pytest.mark.parametrize("sentence", [
    "gearbox untuk motor listrik 5 kW",
    "pasang dudukan motor",
    "motor mount plate 100 mm",
    "motor listrik 3 kW untuk pompa",
    "ambil ukuran dari ISO table",
    "ISO table",
    "standard table for M8 bolts",
    "tabel ISO untuk baut",
    "from the table",
    "table source",
    "sketch on the top plane",
    "xy plane sketch",
    "sketch on front plane",
    "top plane",
    "cart wheel bracket",
    "drone arm 200 mm",
    "chair leg",
    "bicycle frame 540 mm",
    "drone mount plate 100 mm",
])
def test_a_part_a_table_or_a_plane_is_never_a_product(sentence):
    assert cad_router.product_of(sentence) is None, sentence


@pytest.mark.parametrize("sentence,product", [
    ("design sepeda gunung", "bicycle"),
    ("buat pesawat", "airplane"),
    ("design a drone", "drone"),
    ("sepeda motor 150 cc", "motorcycle"),
    ("motor bebek 110 cc", "motorcycle"),
    ("design a plane concept", "airplane"),
    ("design a table 1200 mm", "table"),
    ("buat meja kerja 1200 mm", "table"),
    ("robot arm 600 mm", "robot"),
])
def test_the_product_the_sentence_really_asks_for(sentence, product):
    assert cad_router.product_of(sentence) == product


def test_a_missing_size_from_the_simplified_attempt_is_asked(monkeypatch):
    """Review of PR #31, item 6: the questions were computed before `simple` existed, so they never travelled."""
    def fake_generate(message, context, provider, reply_language):
        if "Simplest valid version" in message:
            return {"ok": True, "cad": {"title": "Bicycle", "code": None, "params": [],
                                        "missing": [{"k": "wheel"}]}, "answer": "Sizes?", "read_by": "fake"}
        return {"ok": False, "cad": None, "answer": "the answer was cut off", "attempts": []}
    monkeypatch.setattr(cad_fallback, "placeholder_cad", lambda msg: None)
    out = cad_fallback.design("buatkan design konsep sepeda", {}, None, None, generate_fn=fake_generate)
    assert out["cad"] is None and out["fidelity"] is None
    assert [m["k"] for m in out["missing"]] == ["wheel"]
    assert "wheel" in out["missing"][0]["question"].lower()


# --------------------------------------------------------------------------------------------- no geometry, no fidelity
_KEY = "body_dimension"


def test_a_cad_with_missing_sizes_is_not_a_design(monkeypatch):
    def fake_generate(message, context, provider, reply_language):
        return {"ok": True, "cad": {"title": "Bicycle", "code": None, "params": [],
                                    "missing": [{"k": _KEY}, {"k": "table_source"}, {"k": "percentile"}]},
                "answer": "I need sizes.", "read_by": "fake"}
    monkeypatch.setattr(cad_fallback, "placeholder_cad", lambda msg: None)
    out = cad_fallback.design("buatkan design konsep sepeda", {}, None, None, generate_fn=fake_generate)
    assert out["cad"] is None
    assert out["fidelity"] is None
    assert out["missing_sizes"] is True
    assert [m["k"] for m in out["missing"]] == [_KEY, "table_source", "percentile"]
    for m in out["missing"]:
        assert "_" not in m["question"], m
        assert m["question"].endswith((")", "?", "?")), m


def test_a_question_for_a_bare_field_name_is_written_out():
    assert cad_fallback.question_for({"k": "body_dimension"}) == \
        "What are the main outside dimensions — length, width and height? (mm)"
    assert "percentile" in cad_fallback.question_for({"k": "percentile"})
    assert cad_fallback.question_for({"k": "frame_angle", "question": "What seat angle do you want? (deg)"}) == \
        "What seat angle do you want? (deg)"
    # a bare field name repeated as the question is replaced too
    assert cad_fallback.question_for({"k": "wall", "question": "wall"}).startswith("How thick is the wall")


def test_a_real_cad_with_code_keeps_its_fidelity(monkeypatch):
    def fake_generate(message, context, provider, reply_language):
        return {"ok": True, "cad": {"title": "Plate", "code": "return box(80,50,10)", "params": [],
                                    "missing": []}, "answer": "Here it is.", "read_by": "fake"}
    out = cad_fallback.design("plate 80x50x10", {}, None, None, generate_fn=fake_generate)
    assert out["fidelity"] == "full" and out["cad"]["code"]


def test_version_is_0_40_3():
    assert ace_core.SOLVER_VERSION == "0.40.3"
