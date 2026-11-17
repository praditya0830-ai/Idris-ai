"""The bench must be reachable over HTTP and must never answer a quantitative question without inputs."""
from engine_routes import GET_PATHS, POST_PATHS, handle_get, handle_post


def test_bench_paths_are_declared():
    assert {"/engineer/solve", "/engineer/verify"} <= set(POST_PATHS)
    assert "/engineer/materials" in GET_PATHS


def test_solve_reads_the_size_from_the_sentence_over_http():
    status, body = handle_post("/engineer/solve", {"message": "berapa torsi baut M10 kelas 8.8"})
    assert status == 200 and body["success"] is True
    ids = [c["id"] for c in body["claims"]]
    assert "bolt.At" in ids and "bolt.torque" in ids
    assert all(c["formula"] and c["source"] for c in body["claims"])


def test_solve_answers_with_missing_inputs_instead_of_a_guess():
    status, body = handle_post("/engineer/solve", {
        "message": "tegangan balok pipa 20x16 panjang 500",
        "given": {"section": "tube", "od_mm": 20, "id_mm": 16, "L_mm": 500}})
    assert status == 200
    assert body["claims"] == []
    assert any(m["input"] == "F_n" for m in body["missing"])


def test_solve_verifies_a_supplied_answer():
    status, body = handle_post("/engineer/solve", {
        "message": "baut M10 kelas 8.8", "answer": "Preload 26,1 kN, jadi aman."})
    assert status == 200
    assert body["verification"]["ok"] is True, body["verification"]

    status, body = handle_post("/engineer/solve", {
        "message": "baut M10 kelas 8.8", "answer": "Preload 31 kN, jadi aman."})
    assert body["verification"]["ok"] is False
    assert body["verification"]["unverified"]


def test_verify_endpoint_flags_a_number_the_bench_did_not_compute():
    status, body = handle_post("/engineer/verify", {
        "answer": "Tegangan 62 MPa", "engineering": {"claims": [{"id": "beam.sigma", "value": 50.0}]}})
    assert status == 200 and body["ok"] is False
    assert body["unverified"][0]["value"] == 62


def test_materials_get_is_the_table_the_bench_uses():
    status, body = handle_get("/engineer/materials")
    assert status == 200 and body["success"] is True
    assert "al_6061_t6" in body["materials"]
    assert body["materials"]["al_6061_t6"]["source"]
