import json
import math
import os
import tempfile

import llm
from physics.simple_part import (builtin_parse, design_from_message, looks_like_part_request, make_part, validate)

OUT = tempfile.mkdtemp()


def test_cube_exact_volume_and_mass():
    r = make_part({"shape": "box", "length": 50, "width": 50, "height": 50, "material": "steel"}, OUT, "a")
    assert r["ok"] and r["results"]["volume_mm3"] == 125000.0
    assert math.isclose(r["results"]["mass_g"], 981.25, rel_tol=1e-6)
    assert r["self_check"]["ok"] and os.path.getsize(os.path.join(OUT, r["files"]["stl"])) == 84 + 50 * 12


def test_curved_parts_self_check_within_1_percent():
    for sp in ({"shape": "cylinder", "diameter": 40, "height": 60, "bore": 20}, {"shape": "sphere", "diameter": 25},
               {"shape": "box", "length": 80, "width": 50, "height": 10,
                "holes": [{"diameter": 6, "x": 8, "y": 8}, {"diameter": 6, "x": 72, "y": 42}]}):
        r = make_part(sp, OUT, "b")
        assert r["ok"] and r["self_check"]["difference_pct"] < 1.0, (sp, r["self_check"])


def test_bad_specs_are_refused_not_guessed():
    for sp, word in (({"shape": "box", "length": 20, "width": 20, "height": 5,
                       "holes": [{"diameter": 25, "x": 10, "y": 10}]}, "does not fit"),
                     ({"shape": "cylinder", "diameter": 20, "height": 10, "bore": 20}, "wall"),
                     ({"shape": "torus", "diameter": 20}, "not supported"),
                     ({"shape": "box", "missing": ["side length"]}, "missing size")):
        _, errs, _ = validate(sp)
        assert any(word in e for e in errs), (sp, errs)


def test_builtin_parser_indonesian():
    assert builtin_parse("buat kubus 4 cm dari baja") == {"material": "baja", "shape": "box", "length": 40.0,
                                                          "width": 40.0, "height": 40.0, "name": "cube"}
    assert builtin_parse("tabung d40 lubang 20 tinggi 60")["bore"] == 20.0
    assert looks_like_part_request("tolong design kubus") and not looks_like_part_request("gearbox 5 kW ratio 8 kubus")


def test_ai_reads_sentence_solver_builds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "gemini")
    spec = {"shape": "box", "length": 80, "width": 50, "height": 10, "material": "aluminium",
            "holes": [{"diameter": 6, "x": 8, "y": 8}, {"diameter": 6, "x": 72, "y": 8},
                      {"diameter": 6, "x": 8, "y": 42}, {"diameter": 6, "x": 72, "y": 42}]}
    monkeypatch.setattr(llm, "call_provider", lambda *a, **k: (json.dumps(spec), "gemini-test"))
    r = design_from_message("plat 80x50x10 dengan 4 lubang 6 mm di pojok", OUT, "c",
                            lambda m: llm.parse_part(m), "https://x")
    assert r["ok"] and r["read_by"].startswith("gemini") and len(r["spec"]["holes"]) == 4
    assert r["files"]["step_url"].startswith("https://x/download/")
    assert math.isclose(r["results"]["volume_mm3"], 80 * 50 * 10 - 4 * math.pi * 9 * 10, rel_tol=1e-6)


def test_ai_failure_falls_back_to_builtin(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("provider returned HTTP 429: quota")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("IDRIS_LLM_ORDER", "gemini")
    monkeypatch.setattr(llm, "call_provider", boom)
    r = design_from_message("kubus 30 mm", OUT, "d", lambda m: llm.parse_part(m))
    assert r["ok"] and "AI failed" in r["read_by"] and r["results"]["volume_mm3"] == 27000.0


def test_pillow_block_is_a_design_request_not_a_block():
    """v0.31.1: a primitive word inside a real design sentence must not start the box builder."""
    from physics.simple_part import looks_like_part_request as lp
    assert not lp("Design a pillow block bearing housing for a 6205 bearing, base 130 × 38 mm, cast iron")
    assert not lp("engine block 300 mm") and not lp("ball bearing holder 40 mm") and not lp("pipe clamp 30 mm")
    for plain in ("kubus 50 mm", "block 100 × 60 × 20 mm", "tabung d40 lubang 20 tinggi 60", "plate 100x50x5 with 4 holes"):
        assert lp(plain), plain


def test_long_design_briefs_are_never_primitives():
    """14 Nov, live: a welded table frame 'from 40 x 40 x 2 mm square tube' was read as a tube and IDRIS asked for a diameter."""
    from physics.simple_part import looks_like_part_request as lp
    assert not lp("Design a welded steel table frame 1200 × 600 mm, height 750 mm, from 40 × 40 × 2 mm square tube, "
                  "with a lower shelf frame 150 mm above the floor and levelling feet M10")
    assert not lp("Design a modular fixture plate 250 × 250 × 20 mm, M6 threaded holes on a 25 mm grid, two 8 mm dowel "
                  "holes in opposite corners, aluminium 6061")
    assert not lp("a plate that carries the motor and the two bearings and the belt tensioner and also the sensor and "
                  "the cable chain and the cover and the label")
    for plain in ("kubus 50 mm", "block 100 × 60 × 20 mm", "tabung d40 lubang 20 tinggi 60", "plate 100x50x5 with 4 holes",
                  "pipa diameter luar 60 diameter dalam 50 panjang 200 baja"):
        assert lp(plain), plain
