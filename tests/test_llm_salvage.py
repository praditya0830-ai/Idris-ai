"""Chat (Arya 28 Oct): a reply cut off by the token limit was shown as raw JSON in the Studio chat."""
import llm

CUT = ('{"intent":"set_params|ask","answer":"Berikut beberapa hal yang perlu dipertimbangkan: 1) aliran udara. Saya sudah mengatur '
       'parameter.","actions":[{"type":"set_params","params":{"power_kw":15,"input_rpm":6000,"gear_ratio":8,"envelope')


def test_cut_off_reply_is_salvaged_not_shown_raw(monkeypatch):
    monkeypatch.setattr(llm, "call_provider", lambda *a, **k: (CUT, "test-model"))
    res = llm.discuss("nvidia", "gearbox 15 kW, 6000 rpm input, ratio 8:1", {}, [], None, {}, "auto")
    assert not res["answer"].lstrip().startswith("{") and '"intent"' not in res["answer"]
    assert res["answer"].startswith("Berikut beberapa hal")
    assert res["intent"] == "discuss" and res["language"] == "id"


def test_plain_prose_still_works(monkeypatch):
    monkeypatch.setattr(llm, "call_provider", lambda *a, **k: ("Varian B paling seimbang.", "m"))
    res = llm.discuss("nvidia", "varian mana yang bagus?", {}, [], None, {})
    assert res["answer"].startswith("Varian B")
