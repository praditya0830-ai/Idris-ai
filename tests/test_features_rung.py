"""The feature rung in practice: the AI recipe and its repair loop, an assembly part's own sizes, and a 20-request
summary of how often a part is built from features instead of code."""
import design_ladder
import feature_recipe
import part_spec


def _code_ok(*a, **k):
    return {"ok": True, "cad": {"title": "x", "code": "return { body: makeBox([0,0,0],[1,1,1]) };", "params": []},
            "answer": "code", "read_by": "test"}


def _spec(msg):
    return part_spec.from_message(msg)


def test_ai_recipe_repairs_only_the_refused_step():
    """An invented 77 mm is refused by name, and the second answer passes; the repair prompt carries the reason."""
    bad = {"title": "Angle plate", "steps": [
        {"id": "base", "type": "sketch", "p": {"prof": "rect", "W": 80, "Hh": 77, "plane": "XY", "at": "value"}},
        {"id": "body", "type": "push", "p": {"sketch": "base", "dist": 6}}]}
    good = {"title": "Angle plate", "steps": [
        {"id": "base", "type": "sketch", "p": {"prof": "rect", "W": 80, "Hh": 50, "plane": "XY", "at": "value"}},
        {"id": "body", "type": "push", "p": {"sketch": "base", "dist": 6}}]}
    seen = []

    def call(prompt):
        seen.append(prompt)
        import json
        return json.dumps(bad if len(seen) == 1 else good)

    r = feature_recipe.ai_recipe("angle plate 80x50x6", {"length_mm": 80, "width_mm": 50, "thickness_mm": 6},
                                 {"name": "Angle plate"}, "prismatic", call=call)
    assert r["steps"] and r["origin"] == "ai" and r["attempts"] == 2
    assert "77" in seen[1] and "not a size the engineer gave" in seen[1]


def test_ai_recipe_gives_up_after_two_repairs_and_says_why():
    invented = '{"title": "x", "steps": [{"id": "b", "type": "sketch", "p": {"prof": "rect", "W": 99, "Hh": 99}}]}'
    calls = []

    def call(prompt):
        calls.append(prompt)
        return invented

    r = feature_recipe.ai_recipe("plate 80x50x6", {"length_mm": 80, "width_mm": 50}, {}, "prismatic", call=call)
    assert r["steps"] == [] and r["attempts"] == 3 and "99" in r["error"] and len(calls) == 3


def test_an_assembly_part_is_built_from_the_plans_own_numbers():
    import assembly_ext
    plan = {"parts": [{"id": "mount", "name": "Mount plate", "role": "holds the motor", "qty": 1,
                       "dims": {"length_mm": 200, "width_mm": 120, "thickness_mm": 8},
                       "sources": {"length_mm": "you", "width_mm": "you", "thickness_mm": "you"}}],
            "interfaces": []}
    got = {}

    def designer(prompt, prov=None, ctx=None):
        got["spec"] = (ctx or {}).get("part_spec")
        return _code_ok()

    assembly_ext.design_parts(plan, designer=designer)
    assert got["spec"]["dims"]["length_mm"] == 200
    assert all(n["source"] == "you" for n in got["spec"]["numbers"])


REQUESTS = [
    "design a mounting bracket plate 120x80x8 with 4 holes M8",
    "plate 200 x 100 x 6 with 4 holes M10",
    "bracket plate 150x90x10 4 holes M6",
    "L bracket 60x40x5 width 30",
    "L bracket 80x60x6 width 40",
    "channel 80x40x5 length 200",
    "channel 100x50x6 length 300",
    "gusset 60x40x6",
    "shaft \u00d830 length 120 with \u00d820 x 30 step",
    "shaft \u00d850 length 200 with \u00d835 x 60 step",
    "tube \u00d860 length 200 wall 5",
    "tube \u00d840 length 150 wall 4",
    "flange \u00d8120 x 10 with 6 holes M8 on PCD 100",
    "flange \u00d8160 x 12 with 4 holes M10 on PCD 130",
    "plate 120x80x8 with a pocket 5 mm deep",
    "housing 100x60x40 t=2",
    "housing 160x120x60 t=3",
    "motor plate 120x80x8 with 4 holes M8",
    "a bracket for the workshop",
    "curved aerodynamic fairing for the nose 400 mm",
]


def test_twenty_requests_features_versus_code():
    counts = {"features": 0, "code": 0, "failed": 0}
    rows = []
    for msg in REQUESTS:
        spec = _spec(msg)
        r = design_ladder.design(msg, {"part_spec": spec} if spec else {}, code_fn=_code_ok)
        counts[r["rung"]] += 1
        rows.append((msg, r["rung"], (r.get("cad") or {}).get("feature_template")))
    print("\n%-52s %-9s %s" % ("request", "rung", "template"))
    for msg, rung, tpl in rows:
        print("%-52s %-9s %s" % (msg[:52], rung, tpl or "-"))
    print("summary: %s of %d" % (counts, len(REQUESTS)))
    # every request that states sizes must come out as features; the two without sizes must not pretend to
    assert counts["features"] >= 18, counts
    assert counts["code"] == 2, counts            # the sizeless bracket and the free-form fairing
    assert counts["failed"] == 0, counts
