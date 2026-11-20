"""What IDRIS does not do is listed as `planned`, never as a solver (P2c: drop test, P2d topology, plate buckling)."""
import solvers  # noqa: F401
from ace_core.http_routes import handle_get
from ace_core.planned import PLANNED
from ace_core.registry import get


def test_every_planned_entry_says_why_and_what_it_needs():
    assert PLANNED
    for item in PLANNED:
        assert item["name"] and item["domain"] and item["status"]
        assert len(item["reason"]) > 60 and item["needs"] and item["would_show"]
        assert get(item["name"]) is None, "a planned solver must not be registered"


def test_solvers_endpoint_lists_planned_next_to_the_real_solvers():
    code, body = handle_get("/solvers")
    assert code == 200
    names = {s["name"] for s in body["solvers"]}
    planned = {p["name"] for p in body["planned"]}
    assert {"fea_part_3d", "modal_part_3d", "buckling_beam", "pipe_flow_darcy"} <= names
    assert "drop_test" in planned and "topology_simp_3d" in planned and "cfd_3d" in planned
    assert not (names & planned)


def test_the_drop_decision_is_documented_with_its_reason():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "P2-drop-test-decision.md")
    text = open(path, encoding="utf-8").read()
    assert "not implemented" in text.lower() and "IEC 60068-2-27" in text
