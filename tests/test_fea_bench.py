"""A strength answer is a solve, and it has to survive a closed-form cross-check.

The box below is 40 x 10 x 10 mm. For a cantilever with F at the free end:
    I = b·h³/12 = 833.33 mm⁴,  M = F·L,  σ = M·c/I,  δ = F·L³/(3EI)
where L is the span the solver sees (the voxelized bbox).
"""
import pytest

from ai import engineer as ENG
from ai import fea as FEA


def box_mesh(L=40.0, W=10.0, H=10.0):
    v = [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
         (0, 0, H), (L, 0, H), (L, W, H), (0, W, H)]
    pos = [c for p in v for c in p]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return {"positions": pos, "indices": idx}


def test_cantilever_agrees_with_the_closed_form():
    r = FEA.analyze(box_mesh(), material="steel", case="cantilever_tip", force_n=200, target_elements=20000)
    sigma_fe = r["claims"][0]["value"]
    sigma_cf = r["cross_check"]["sigma_cf"]
    assert sigma_cf == pytest.approx(200 * r["cross_check"]["span_mm"] * 5 / (10 * 10 ** 3 / 12), rel=1e-6)
    assert sigma_fe == pytest.approx(sigma_cf, rel=0.15)


def test_a_cantilever_across_the_other_axis_is_checked_against_its_own_span():
    # 10 wide in x, 40 long in y: the span is the y extent and the depth is the x extent
    r = FEA.analyze(box_mesh(L=10.0, W=40.0, H=10.0), material="steel", case="cantilever_tip_x",
                    force_n=200, target_elements=20000)
    assert r["cross_check"]["span_mm"] == pytest.approx(40.0, abs=1e-6)
    assert r["cross_check"]["ratio_sigma"] == pytest.approx(1.0, abs=0.15)


def test_the_snapped_element_size_voxelizes_the_box_exactly():
    r = FEA.analyze(box_mesh(), material="steel", case="cantilever_tip", force_n=100, target_elements=20000)
    assert abs(r["volume"]["error_pct"]) <= 1.0, r["volume"]
    assert r["checks"][0]["ok"] is True


def test_equilibrium_check_passes_and_sf_is_sy_over_sigma():
    r = FEA.analyze(box_mesh(), material="steel", case="cantilever_tip", force_n=100, target_elements=12000)
    ids = {c["id"]: c for c in r["claims"]}
    assert ids["fea.sf_yield"]["value"] == pytest.approx(235.0 / ids["fea.sigma_max"]["value"], rel=1e-9)
    eq = [c for c in r["checks"] if c["id"] == "fea.equilibrium"][0]
    assert eq["ok"] is True


def test_a_load_above_yield_is_reported_as_fail_not_softened():
    r = FEA.analyze(box_mesh(), material="steel", case="cantilever_tip", force_n=4000, target_elements=8000)
    assert any(a.startswith("FAIL") for a in r["advice"]), r["advice"]
    # stiffness is not "the limit" when the part has already yielded
    assert not any(a.startswith("FLEXIBLE") for a in r["advice"]), r["advice"]


def test_the_solver_refuses_a_case_it_does_not_have():
    with pytest.raises(ValueError):
        FEA.analyze(box_mesh(), case="four_point", force_n=100)


def test_no_force_means_no_answer():
    with pytest.raises(ValueError):
        FEA.analyze(box_mesh(), case="cantilever_tip")


def test_bench_routes_a_mesh_to_the_solver_and_keeps_the_advice():
    out = ENG.solve("apakah part ini kuat?", {"geometry": box_mesh(), "force_n": 200,
                                             "material_name": "steel", "case": "cantilever_tip",
                                             "target_elements": 12000})
    assert out["topic"] == "fea"
    assert {c["id"] for c in out["claims"]} >= {"fea.sigma_max", "fea.delta_max", "fea.sf_yield"}
    assert out["advice"] and out["checks"]
    levels = {c["id"]: c["evidence"] for c in out["claims"]}
    assert levels["fea.sigma_max"] == 3                          # numerical solver, not an estimate
    assert levels["fea.sigma_closed_form"] == 2                  # the cross-check is closed form


def test_bench_asks_for_the_mesh_instead_of_inventing_a_strength():
    out = ENG.solve("apakah part ini kuat?")
    assert out["topic"] == "fea" and out["claims"] == []
    assert any(m["input"] == "geometry" for m in out["missing"])


def test_chat_block_carries_the_engineer_read_and_the_checks():
    out = ENG.solve("apakah part ini kuat?", {"geometry": box_mesh(), "force_n": 200,
                                             "target_elements": 12000})
    block = ENG.chat_block(out)
    assert "ENGINEERING EVIDENCE" in block
    assert "CHECK fea.equilibrium: pass" in block
    assert "ENGINEER'S READ" in block


def test_gate_accepts_a_number_that_only_appears_in_the_advice():
    out = ENG.solve("apakah part ini kuat?", {"geometry": box_mesh(), "force_n": 200,
                                             "target_elements": 8000})
    sf = [c for c in out["claims"] if c["id"] == "fea.sf_yield"][0]["value"]
    answer = next(a for a in out["advice"] if a.startswith("OK") or a.startswith("THIN") or a.startswith("FAIL"))
    assert ENG.verify(answer, out)["ok"] is True, ENG.verify(answer, out)
    assert sf > 0
