"""Bagian F: a design never comes home empty.

No AI, no kernel here: the code writer is a fake, and the rules under test are the ones that keep the answer
honest — the ladder, the fidelity label, the engineer's sizes in the placeholder, and the target check that
never quotes a saving percentage it cannot compute.
"""
import cad_fallback as fb
import llm


def _cad(title="L bracket", code="const body = makeBox([0, 0, 0], [40, 20, 5]);\nreturn { body };"):
    return {"title": title, "kind": "functional", "process": "cnc", "material": "aluminium", "note": "",
            "code": code,
            "params": [{"k": "L", "label": "Length", "label_id": "Panjang", "unit": "mm", "value": 40.0,
                        "min": 10, "max": 200, "source": "user"}],
            "expect": [{"label": "Length", "axis": "x", "value": 40.0, "tol": 0.5}],
            "holes": [], "variants": [], "missing": [], "requirements": None, "read_by": "fake (model)"}


def _pass(*_a):
    return {"ok": True, "cad": _cad(), "answer": "L bracket: code written by fake.", "read_by": "fake (model)",
            "attempts": []}


def _fail(*_a):
    return {"ok": False, "cad": None, "answer": "IDRIS could not get valid CAD code from the AI this time.",
            "attempts": [], "read_by": None}


# ------------------------------------------------------------------ F1: the ladder
def test_the_full_design_is_used_when_it_works():
    calls = []

    def gen(msg, ctx, prov, lang):
        calls.append(msg)
        return _pass()

    res = fb.design("bracket 40 mm", generate_fn=gen)
    assert res["fidelity"] == "full" and len(calls) == 1
    assert res["cad"]["fidelity"] == "full"
    assert res["validated"] is False, "the chat path runs no analysis, so it is never validated"


def test_a_failed_design_falls_to_one_simplified_call():
    calls = []

    def gen(msg, ctx, prov, lang):
        calls.append(msg)
        return _fail() if len(calls) == 1 else _pass()

    res = fb.design("bracket 40 mm", generate_fn=gen)
    assert len(calls) == 2, "the ladder may ask for the simplest version exactly once"
    assert "simplest valid version" in calls[1].lower()
    assert res["fidelity"] == "simplified"
    assert res["answer"].startswith("Full design failed")
    assert "fillets" in res["answer"], "the answer says what was left out"


def test_a_simplified_failure_gives_a_placeholder_with_the_engineer_sizes():
    res = fb.design("plat 120 mm x 80 mm x 6 mm", generate_fn=_fail)
    assert res["fidelity"] == "placeholder" and res["ok"] is True
    assert res["cad"] is not None
    assert [p["value"] for p in res["cad"]["params"]] == [120.0, 80.0, 6.0]
    assert all(p["source"] == "user" for p in res["cad"]["params"])
    assert "makeBox" in res["cad"]["code"]
    assert res["cad"]["read_by"].startswith("IDRIS envelope template")
    assert res["validated"] is False


def test_a_cylinder_request_gets_a_cylinder_envelope():
    res = fb.design("silinder diameter 30 mm tinggi 80 mm", generate_fn=_fail)
    assert res["fidelity"] == "placeholder"
    assert "makeCylinder" in res["cad"]["code"]
    assert [p["value"] for p in res["cad"]["params"]] == [30.0, 80.0]


def test_without_one_size_idris_asks_for_sizes_and_draws_nothing():
    res = fb.design("bracket yang bagus", generate_fn=_fail)
    assert res["ok"] is False and res["cad"] is None
    assert "size" in res["answer"].lower()
    assert "No geometry is shown" in res["answer"]


def test_no_ai_at_all_still_gives_geometry():
    def gen(msg, ctx, prov, lang):
        raise llm.LLMError("no AI provider is configured on the server", 503)

    res = fb.design("plat 100 mm x 50 mm", generate_fn=gen)
    assert res["fidelity"] == "placeholder" and res["cad"] is not None


def test_the_placeholder_says_what_it_is_not():
    res = fb.design("plat 100 mm x 50 mm", generate_fn=_fail)
    joined = " ".join(res["not_checked"]).lower()
    assert "envelope" in joined and "not the part" in joined
    assert res["cad"]["omitted"]


# ------------------------------------------------------------------ F2: the target check
def test_a_speed_and_fuel_target_is_not_achievable_and_the_shape_still_exists():
    msg = "sayap: +50% kecepatan dan -50% bahan bakar"
    tc = fb.target_check(msg)
    assert tc["verdict"] == "not_achievable_by_this_part"
    assert "square" in tc["reason"] and "cube" in tc["reason"]
    res = fb.design(msg + " 300 mm", generate_fn=_pass)
    assert res["cad"] is not None
    assert res["validated"] is False
    assert res["target_check"]["verdict"] == "not_achievable_by_this_part"
    assert res["answer"].startswith("Target check:")


def test_no_percentage_idris_cannot_compute_is_quoted():
    tc = fb.target_check("motor listrik lebih ringan 20%")
    assert tc["verdict"] == "plausible"
    assert tc["note"] == fb.TARGET_NOTE
    res = fb.design("plat 100 mm", generate_fn=_fail)
    assert "%" not in res["answer"], res["answer"]


def test_a_sentence_without_a_target_has_no_target_check():
    assert fb.target_check("bracket 40 mm aluminium") is None
    assert fb.target_check("") is None


# ------------------------------------------------------------------ F3/F4: what is left out, and how it is said
def test_every_answer_carries_the_not_checked_line():
    res = fb.design("bracket 40 mm", generate_fn=_pass)
    note = fb.answer_notes(res)
    assert "Not checked by IDRIS:" in note
    assert "strength (FEA)" in note
