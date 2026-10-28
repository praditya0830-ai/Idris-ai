"""OCCT stage 1 (optional kernel, fallback must work without it) and evidence-level labels."""
import importlib.util
from physics.occt_export import status, selftest


def test_occt_status_and_selftest_never_raise():
    st = status()
    assert set(st) >= {"available", "version", "error", "step_mode_default"}
    assert st["step_mode_default"] == "faceted"
    rep = selftest()
    assert "steps" in rep and "pass" in rep and "rss_mb_before_import" in rep
    if importlib.util.find_spec("OCP") is None:
        assert rep["pass"] is False and rep["steps"][0]["ok"] is False      # fallback path tested
    else:
        assert rep["pass"], rep


def test_results_carry_evidence_level():
    import solvers  # noqa: F401
    from ace_core.analysis import handle_post
    from solvers._example import EXAMPLE_VARIANT as V
    c, b = handle_post("/fea?sync=1", {**V, "mesh": "coarse"})
    ev = b["result"]["evidence"]
    assert ev["level"] in (3, 4) and ev["screening"] is True
    assert any("level 5" in x for x in ev["not_yet"])
    from physics.variants import generate_variants
    from tests.test_manufacturable import S2
    v = generate_variants(S2, compare_basis=False)["variants"][0]
    assert v["evidence"]["level"] == 4 and v["evidence"]["verified_on"]
