import math
import solvers  # noqa: F401
from ace_core.registry import verify, get
from ace_core.schema import validate
from physics.iso6336 import rate_pair, life_factor_contact, tol_1328
from solvers.gear_pair_iso import EX1_INPUTS, EX1_EXPECTED, SCHEMA


def _get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def test_example1_all_quantities_within_0_1_percent():
    out = rate_pair(EX1_INPUTS)
    for path, ref in EX1_EXPECTED.items():
        got = _get(out, path)
        assert math.isclose(got, ref, rel_tol=1e-3, abs_tol=1e-9), (path, got, ref)


def test_benchmark_registered_and_passes():
    r = verify("gear_pair_iso6336")
    assert r["all_pass"] and r["passed"] == 1


def test_life_factor_points():
    assert life_factor_contact(1e4) == 1.6
    assert math.isclose(life_factor_contact(5e7), 1.0, rel_tol=1e-9)
    assert math.isclose(life_factor_contact(1e10), 0.85, rel_tol=1e-9)
    assert math.isclose(life_factor_contact(1e10, 1.0), 1.0, rel_tol=1e-9)


def test_tolerance_rounding():
    f_pt, f_fa = tol_1328(856.35, 8.0, 5)
    assert f_pt == 9.1 and f_fa == 13.0


def test_spur_pair_sane():
    o = rate_pair(dict(mn=3, z1=20, z2=60, b=30, torque_1_nm=100, n1_rpm=3000, K_Hbeta=1.2,
                       sigma_Hlim1=1500, accuracy_grade=6))["outputs"]
    f = o["factors"]
    assert o["geometry"]["eps_beta"] == 0
    assert f["Z_B"] >= 1.0 and f["Z_D"] >= 1.0
    assert 0.85 < f["Z_eps"] < 1.0
    assert f["K_v"] > 1.0 and f["K_Halpha"] >= 1.0
    assert o["contact"]["sigma_H1"] >= o["contact"]["sigma_H0"]


def test_more_torque_lowers_safety():
    a = rate_pair(dict(EX1_INPUTS))["outputs"]["contact"]["S_H1"]
    b = rate_pair(dict(EX1_INPUTS, torque_1_nm=12000))["outputs"]["contact"]["S_H1"]
    assert b < a


def test_k_hbeta_is_required():
    clean, errs = validate(SCHEMA, {k: v for k, v in EX1_INPUTS.items() if k != "K_Hbeta"})
    assert any("K_Hbeta" in e for e in errs)


def test_assumptions_are_reported():
    r = rate_pair(dict(EX1_INPUTS))
    text = " ".join(r["assumptions"])
    assert "K_Hβ" in text and "1328" in text
