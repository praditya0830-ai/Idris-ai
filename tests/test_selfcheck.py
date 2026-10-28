"""GET /selfcheck: end-to-end pipeline check used to verify the live server without manual clicking."""
from ace_core import selfcheck
from ace_core.http_routes import handle_get


def test_one_scenario_end_to_end():
    r = selfcheck._scenario("S2", selfcheck.SCENARIOS["S2"])
    assert r["pass"], r["problems"]
    assert set(r["steps"]) == {"generate_variants", "assembly", "review", "fea", "modal", "thermal"}


def test_route_answers_immediately():
    code, body = handle_get("/selfcheck")
    assert code == 200 and body["status"] in ("running", "done")
