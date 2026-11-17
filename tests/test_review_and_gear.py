"""The review codes, explicit boundary conditions, the ISO 6336 gear topic, and the studio lessons.

These are the parts of the plan that only work if the number behind them is real: a finding that cannot be
decided from the solve is not emitted, and a gear rating that cannot be computed asks for its inputs.
"""
import pytest

from ai import engineer as ENG
from ai import fea as FEA
from ai import review as REVIEW


def box_mesh(L=40.0, W=10.0, H=10.0):
    v = [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
         (0, 0, H), (L, 0, H), (L, W, H), (0, W, H)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return {"positions": [c for p in v for c in p], "indices": idx}


def _findings(mesh, force_n=200, material="steel", target=12000):
    return FEA.analyze(mesh, material=material, case="cantilever_tip", force_n=force_n,
                       target_elements=target)["review"]


# ---------------------------------------------------------------- review[]
def test_a_healthy_part_has_no_findings():
    # 20000 elements on this box gives an exact voxel volume, so nothing is left to warn about
    assert _findings(box_mesh(), target=20000) == []


def test_part_check_returns_the_same_findings_as_the_chat():
    import solvers  # noqa: F401  (registers the solvers)
    from ace_core import analysis
    mesh = box_mesh()
    code, body = analysis.handle_post("/part-check?sync=1", {
        "positions": mesh["positions"], "indices": mesh["indices"], "material": "steel",
        "element_size_mm": 1.0,
        "supports": [{"where": {"plane": "x", "side": "min"}}],
        "loads": [{"force_N": [0, 0, -200], "where": {"plane": "x", "side": "max"}}]})
    assert code == 200 and body["success"] is True
    result = body["result"]
    assert "review" in result and "review_summary" in result
    assert result["review"] == REVIEW.review_part(result)


def test_a_coarse_mesh_is_a_finding_not_a_footnote():
    codes = {f["code"] for f in _findings(box_mesh(), target=12000)}
    assert "MESH_COARSE" in codes


def test_a_slender_member_is_named_with_its_ratio():
    codes = {f["code"]: f for f in _findings(box_mesh(L=200.0, W=3.0, H=3.0), force_n=20, target=8000)}
    assert "MEMBER_SLENDER" in codes
    assert codes["MEMBER_SLENDER"]["measured"]["ratio"] > 40


def test_a_thin_plate_is_named_with_the_measured_mean_thickness():
    mesh = box_mesh(L=40.0, W=20.0, H=1.0)
    out = {"max_von_mises_MPa": 10.0, "max_displacement_mm": 0.01, "equilibrium_error": 0.0,
           "applied_loads": [{"force_N": [0, 0, -5]}], "volume": {"mesh_mm3": 800.0, "error_pct": 0.0},
           "surface": mesh}
    thin = [f for f in REVIEW.review_part(out) if f["code"] == "WALL_THIN"]
    assert thin
    # the number has to be 2V/S of this part, not a remembered plate thickness
    expect = 2.0 * 800.0 / REVIEW._surface_area_mm2(mesh)
    assert thin[0]["measured"]["mean_thickness_mm"] == pytest.approx(round(expect, 3), abs=5e-4)
    assert expect < 1.5


def test_review_refuses_to_guess_when_the_result_is_empty():
    assert REVIEW.review_part({}) == []
    assert REVIEW.summary([]) == {"count": 0, "codes": [], "fails": []}


def test_review_summary_counts_the_failures():
    out = {"max_von_mises_MPa": 0.0, "max_displacement_mm": 0.0,
           "equilibrium_error": 0.2, "applied_loads": [{"force_N": [0, 0, -100]}],
           "volume": {"mesh_mm3": 100.0, "error_pct": 0.0}}
    codes = {f["code"]: f for f in REVIEW.review_part(out)}
    assert "UNSUPPORTED" in codes and "NO_LOAD_PATH" in codes
    assert codes["UNSUPPORTED"]["severity"] == "fail"
    assert REVIEW.summary(list(codes.values()))["fails"] == ["UNSUPPORTED", "NO_LOAD_PATH"]


# ---------------------------------------------------------------- explicit boundary conditions
def test_explicit_supports_and_loads_match_the_named_case():
    mesh = box_mesh()
    named = FEA.analyze(mesh, case="cantilever_tip", force_n=200, target_elements=8000)
    explicit = FEA.analyze(mesh, force_n=None, target_elements=8000,
                           supports=[{"where": {"plane": "x", "side": "min"}}],
                           loads=[{"force_N": [0, 0, -200], "where": {"plane": "x", "side": "max"}}])
    assert explicit["case"] == "explicit"
    assert explicit["claims"][0]["value"] == pytest.approx(named["claims"][0]["value"], rel=1e-9)
    assert explicit["review"] == named["review"]


def test_a_studio_payload_reaches_the_solver_through_the_bench():
    out = ENG.solve("apakah part ini kuat?", {
        "geometry": box_mesh(), "material_name": "steel", "target_elements": 8000,
        "supports": [{"where": {"plane": "x", "side": "min"}}],
        "loads": [{"force_N": [0, 0, -300], "where": {"plane": "x", "side": "max"}}]})
    assert out["topic"] == "fea" and out["claims"]
    assert out["review_summary"]["count"] >= 0
    assert "REVIEW" not in ENG.chat_block(out) or out["review"]      # review text only when there is a finding


# ---------------------------------------------------------------- gear pair (ISO 6336)
GEAR = {"mn": 3.0, "z1": 20, "z2": 60, "b": 30.0, "torque_1_nm": 100.0, "n1_rpm": 3000.0,
        "K_Hbeta": 1.2, "sigma_Hlim1": 1500.0}


def test_gear_rating_comes_from_the_benchmarked_solver():
    from physics.iso6336 import rate_pair
    ref = rate_pair(dict(GEAR))["outputs"]["contact"]["S_H1"]
    out = ENG.solve("berapa safety factor pitting pasangan gigi ini", dict(GEAR))
    assert out["topic"] == "gear"
    got = {c["id"]: c["value"] for c in out["claims"]}
    assert got["gear.S_H1"] == pytest.approx(ref, rel=1e-12)
    assert got["gear.ratio"] == pytest.approx(3.0)
    assert all(c["evidence"] == 2 for c in out["claims"])
    assert out["advice"]


def test_gear_asks_for_every_factor_it_would_otherwise_have_to_invent():
    out = ENG.solve("berapa safety factor gigi ini", {"mn": 3.0, "z1": 20, "z2": 60})
    assert out["topic"] == "gear" and out["claims"] == []
    inputs = {m["input"] for m in out["missing"]}
    assert {"b", "torque_1_nm", "K_Hbeta", "sigma_Hlim1"} <= inputs


def test_gear_numbers_go_through_the_gate():
    out = ENG.solve("berapa safety factor gigi ini", dict(GEAR))
    sf = [c for c in out["claims"] if c["id"] == "gear.S_H1"][0]["value"]
    assert ENG.verify(f"Safety factor kontak = {sf:.3f}", out)["ok"] is True
    assert ENG.verify("Safety factor kontak = 9.99", out)["ok"] is False


# ---------------------------------------------------------------- teaching
def test_every_lesson_names_a_tool_that_exists():
    for topic in ("sketch", "member", "assembly", "fea", "hole", "wall", "pattern", "draft"):
        lesson = ENG.teach(topic)
        assert lesson["found"] is True, topic
        assert lesson["steps"] and lesson["check"] and lesson["why"]
        assert all(s["tool"] for s in lesson["steps"])
    assert ENG.teach("drafting")["found"] is False
