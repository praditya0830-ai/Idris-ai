"""Tests for /modal, /cfd (thermal) and /review in the UI job format."""
import time
from ace_core.http_routes import handle_get, handle_post
from ace_core import registry

VARIANT = {"name": "V1", "tag": "balanced", "scores": {"physics": 80},
           "design": {"module_mm": 2, "pinion_teeth": 18, "gear_teeth": 51, "face_width_mm": 48,
                      "stages": 2, "center_distance_mm": 69},
           "kinematics": {"input_torque_nm": 23.9, "output_torque_nm": 184.1, "output_rpm": 747,
                          "ratio_actual": 8.028, "efficiency": 0.96},
           "loads": {"tangential_force_n": 3683, "pitch_line_velocity_ms": 4.0},
           "shaft": {"output_diameter_mm": 30}, "housing_params": {}, "housing_material": "Al7075-T6"}


def _job(path, body, timeout=180):
    code, b = handle_post(path, body)
    assert code == 202 and b["status"] == "queued", b
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = handle_get(f"{path}/{b['job_id']}")[1]
        if s["status"] in ("done", "failed"):
            return s
        assert s["stage"] in ("Queued", "Meshing", "Applying loads", "Solving", "Post-processing")
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_engine_benchmarks():
    for name in ("modal_solid_block", "thermal_slab"):
        assert registry.verify(name)["all_pass"], name


def test_modal_job():
    s = _job("/modal", {**VARIANT, "analysis": "modal", "modes": 6, "fixture": "base", "mesh": "standard"})
    assert s["status"] == "done", s
    r = s["result"]
    n = len(r["surface"]["positions"]) // 3
    assert len(r["surface"]["modes"]) == 6
    for m in r["surface"]["modes"]:
        assert m["frequency_hz"] > 0 and len(m["displacement"]) == 3 * n
    f = [m["frequency_hz"] for m in r["surface"]["modes"]]
    assert f == sorted(f)
    assert any("coarser mesh" in w for w in r["warnings"])          # auto-downgrade reported
    assert any("Fields not used" in w for w in r["warnings"])       # name/tag/scores ignored


def test_thermal_job_heat_balance():
    s = _job("/cfd", {**VARIANT, "analysis": "thermal", "power_kw": 15, "input_rpm": 6000,
                      "ambient_c": 25, "cooling": "forced", "air_speed_ms": 5, "oil": "ISO VG 100"})
    assert s["status"] == "done", s
    r = s["result"]
    assert abs(r["heat_dissipated_w"] - 600) < 6
    n = len(r["surface"]["positions"]) // 3
    assert len(r["surface"]["temperature"]) == n
    assert r["oil_temp_c"] > 25 and min(r["surface"]["temperature"]) >= 25 - 1e-6


def test_thermal_more_air_is_cooler():
    a = handle_post("/cfd?sync=1", {**VARIANT, "power_kw": 5, "cooling": "natural", "air_speed_ms": 0})[1]
    b = handle_post("/cfd?sync=1", {**VARIANT, "power_kw": 5, "cooling": "forced", "air_speed_ms": 10})[1]
    assert b["result"]["oil_temp_c"] < a["result"]["oil_temp_c"]


def test_review_job():
    parts = [{"name": "housing", "kind": "housing"}, {"name": "pinion_s1", "kind": "pinion"},
             {"name": "gear_s1", "kind": "gear"}, {"name": "pinion_s2", "kind": "pinion"},
             {"name": "gear_s2", "kind": "gear"}]
    s = _job("/review", {**VARIANT, "analysis": "review", "parts": parts,
                         "checks": ["interference", "fits", "dfm", "assembly"]})
    assert s["status"] == "done", s
    fs = s["result"]["findings"]
    assert fs and all(set(f) >= {"severity", "part", "message", "fix", "at"} for f in fs)
    assert all(len(f["at"]) == 3 for f in fs)
    assert any(f["severity"] == "critical" and f["part"] == "housing" for f in fs)
    assert {f["part"] for f in fs} <= {p["name"] for p in parts} | {"Input shaft", "Intermediate shaft", "Output shaft"}


def test_review_subset_and_errors():
    r = handle_post("/review?sync=1", {**VARIANT, "checks": ["dfm"]})[1]
    assert all(f["check"] == "dfm" for f in r["result"]["findings"])
    c, b = handle_post("/review?sync=1", {**VARIANT, "checks": ["magic"]})
    assert c == 400 and "unknown checks" in b["error"]
