"""Tests for the 3D FEA engine, the housing solver and the /fea job endpoint."""
import time
import numpy as np
from ace_core.http_routes import handle_get, handle_post, encode_json
from physics.fem3d import box_model
from physics.housing import build_layout, build_model, bearing_loads, fixture_nodes

REQ = {"analysis": "static",
       "design": {"module_mm": 2, "pinion_teeth": 18, "gear_teeth": 51, "face_width_mm": 48,
                  "stages": 2, "center_distance_mm": 69},
       "kinematics": {"input_torque_nm": 23.9, "output_torque_nm": 184.1, "output_rpm": 747,
                      "ratio_actual": 8.028, "efficiency": 0.96},
       "loads": {"tangential_force_n": 3683, "pitch_line_velocity_ms": 4.0},
       "shaft": {"output_diameter_mm": 30}, "housing_params": {},
       "housing_material": "Al7075-T6", "fixture": "base", "mesh": "coarse"}


def test_3d_engine_benchmarks():
    rep = handle_get("/verify/fea_solid_block")[1]
    assert rep["all_pass"], rep


def test_surface_is_closed_and_outward():
    m = box_model(20, 10, 6, 2)
    ids, tris = m.surface()
    P = m.nodes[ids]
    vol = np.einsum("ij,ij->i", P[tris[:, 0]], np.cross(P[tris[:, 1]], P[tris[:, 2]])).sum() / 6
    assert abs(vol - 20 * 10 * 6) < 1e-6
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts) == {2}          # watertight


def test_housing_equilibrium_and_loads():
    L = build_layout(REQ["design"], REQ["kinematics"], {}, REQ["shaft"], {})
    m = build_model(L, L["wall_mm"])
    loads, applied = bearing_loads(m, L)
    assert len(applied) == 6
    # internal gear forces: net force on the housing is zero, net moment is not
    assert np.allclose(loads.sum(0), 0, atol=1e-6)
    # stage-2 tangential force matches the variant's loads.tangential_force_n (3683 N) within 1%
    T2 = L["shaft_torque_Nm"][1]
    assert abs(2e3 * T2 / (2 * 18) - 3683) / 3683 < 0.01
    fx = fixture_nodes(m, L, "base")
    r = m.solve(71700, 0.33, fx, loads)
    assert np.allclose(r["reactions"][fx].sum(0), 0, atol=1e-3 * np.abs(loads).sum())


def test_fea_job_flow():
    code, b = handle_post("/fea", REQ)
    assert code == 202 and b["status"] == "queued" and b["job_id"]
    t0 = time.time()
    while True:
        s = handle_get(f"/fea/{b['job_id']}")[1]
        if s["status"] in ("done", "failed") or time.time() - t0 > 120:
            break
        assert s["stage"] in ("Queued", "Meshing", "Applying loads", "Solving", "Post-processing")
        time.sleep(0.2)
    assert s["status"] == "done", s
    r = s["result"]
    sf = r["surface"]
    n = len(sf["von_mises"])
    assert len(sf["positions"]) == 3 * n and len(sf["displacement"]) == 3 * n
    assert len(sf["indices"]) % 3 == 0 and max(sf["indices"]) < n
    assert r["yield_mpa"] == 503 and r["mesh"]["elements"] > 0 and r["applied_loads"]


def test_fea_errors_are_readable():
    c, b = handle_post("/fea?sync=1", dict(REQ, housing_material="Unobtainium"))
    assert c == 400 and not b["success"] and "Available" in b["error"]
    c, b = handle_post("/fea?sync=1", dict(REQ, mesh="fine", design=dict(REQ["design"], face_width_mm=200)))
    assert c == 400 and "elements" in b["error"]
    assert handle_get("/fea/doesnotexist")[0] == 404


def test_flange_fixture():
    c, b = handle_post("/fea?sync=1", dict(REQ, fixture="flange"))
    assert c == 200 and b["result"]["summary"]["max_von_mises_mpa"] > 0


def test_gzip_encoding():
    data, h = encode_json({"a": list(range(5000))}, "gzip, deflate")
    assert h.get("Content-Encoding") == "gzip"
    assert encode_json({"a": 1}, "")[1] == {}
