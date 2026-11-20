"""UI v65 item 3: `context.shape_family = {name, principle, design_space}` on /chat.

The contract is read before a prompt is written (three fields, capped lengths, a design space of plain values), the
family reaches the CAD writer's user block, and the answer says what happened to it -- including that IDRIS does not
verify the code stays inside the design space.
"""
import cad_codegen
import main
import shape_family


def _sf(**over):
    raw = {"name": "l_bracket", "principle": "two legs at 90 degrees carry the load into the wall",
           "design_space": {"L": [20, 80], "leg_t": 3, "hole_d": [4, 5, 6]}}
    raw.update(over)
    return raw


# ------------------------------------------------------------------ the contract
def test_a_complete_family_is_kept_with_its_design_space():
    clean, errs = shape_family.parse(_sf())
    assert not errs and clean["name"] == "l_bracket"
    assert clean["design_space"] == {"L": [20, 80], "leg_t": 3, "hole_d": [4, 5, 6]}


def test_no_family_at_all_is_not_an_error():
    assert shape_family.parse(None) == (None, [])


def test_all_three_fields_are_required_and_named_in_the_reason():
    for missing in ("name", "principle", "design_space"):
        raw = _sf()
        raw.pop(missing)
        clean, errs = shape_family.parse(raw)
        assert clean is None and any(missing in e for e in errs), missing


def test_a_design_space_cannot_smuggle_an_object_or_a_crowd_into_the_prompt():
    clean, errs = shape_family.parse(_sf(design_space={"L": {"min": 1, "max": 2}}))
    assert clean is None and "object" in errs[0]
    clean, errs = shape_family.parse(_sf(design_space={f"p{i}": i for i in range(30)}))
    assert clean is None and "limit is 24" in errs[0]
    clean, errs = shape_family.parse(_sf(design_space={"L": [1, 2, 3, 4, 5, 6, 7, 8, 9]}))
    assert clean is None and "more than 8 items" in errs[0]


def test_a_sentence_is_a_valid_design_space_and_a_long_name_is_cut():
    clean, errs = shape_family.parse(_sf(design_space="stays inside a 60 x 40 x 3 mm envelope", name="x" * 200))
    assert not errs and clean["design_space"].startswith("stays inside")
    assert len(clean["name"]) == shape_family.NAME_MAX


def test_the_prompt_states_the_family_and_that_it_is_not_a_standard():
    block = shape_family.prompt_block(shape_family.parse(_sf())[0])
    assert "l_bracket" in block and "two legs at 90 degrees" in block
    assert "L = [20, 80]" in block and "not a standard" in block and "note" in block


def test_the_reply_says_accepted_and_what_is_not_checked():
    ok = shape_family.reply(shape_family.parse(_sf())[0])
    assert ok["used"] is True and ok["applied_to"] == "the CAD prompt"
    assert "does not verify" in ok["not_checked"]
    bad = shape_family.reply(None, "shape_family.name is required")
    assert bad["used"] is False and "name" in bad["why"]


# ------------------------------------------------------------------ /chat plumbing
def test_the_cad_writer_receives_the_family_in_its_user_block():
    clean = shape_family.parse(_sf())[0]
    block = cad_codegen._user_block("bracket for a 40 mm tube", {"shape_family": clean})
    assert "User request: bracket for a 40 mm tube" in block
    assert "The Studio's own shape family" in block and "l_bracket" in block


def test_chat_checks_the_family_before_it_reaches_a_prompt(monkeypatch):
    import cad_fallback
    seen = {}

    def fake_design(message, context=None, provider=None, reply_language=None, deadline=None):
        seen["context"] = context
        return {"answer": "ok", "cad": {"title": "Bracket"}, "cad_attempts": [], "fidelity": "full",
                "validated": True, "not_checked": []}

    monkeypatch.setattr(cad_fallback, "design", fake_design)
    out = main.Handler._chat_cad(None, "bracket", {"shape_family": _sf()}, None)
    assert out["shape_family"]["used"] is True and out["shape_family"]["name"] == "l_bracket"
    assert seen["context"]["shape_family"]["design_space"]["leg_t"] == 3


def test_a_family_that_cannot_be_read_is_dropped_and_the_answer_says_why(monkeypatch):
    import cad_fallback
    seen = {}

    def fake_design(message, context=None, provider=None, reply_language=None, deadline=None):
        seen["context"] = context
        return {"answer": "ok", "cad": {"title": "Bracket"}, "cad_attempts": [], "fidelity": "full",
                "validated": True, "not_checked": []}

    monkeypatch.setattr(cad_fallback, "design", fake_design)
    out = main.Handler._chat_cad(None, "bracket", {"shape_family": {"name": "x"}}, None)
    assert out["shape_family"]["used"] is False and "principle" in out["shape_family"]["why"]
    assert "shape_family" not in seen["context"]        # nothing half accepted reached the writer
    assert "shape_family_error" in seen["context"]      # and the answer can still say why


def test_an_assembly_answer_does_not_pretend_the_family_was_applied():
    out = main.Handler._chat_assembly(None, "design whole bicycle", {"shape_family": _sf()}, None)
    assert out["shape_family"]["used"] is False
    assert "part by part" in out["shape_family"]["why"]
