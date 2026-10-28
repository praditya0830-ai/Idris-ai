"""Tahap 2: ISO 281:2007 modified rating life and the oil grade it drives."""
from physics.bearing_life import a_iso, nu1, fatigue_load_limit, modified_life


def test_aiso_benchmark_6206():
    a, _ = a_iso(0.8, 0.2078, 509.0, 9700.0)
    assert abs(a / 0.331 - 1) < 0.01


def test_cu_and_limits():
    assert abs(fatigue_load_limit(11200.0, 46.0) - 509.09) < 0.1
    assert a_iso(0.05, 0.5, 500, 1000)[0] is None                 # κ < 0.1 not applicable
    assert a_iso(10.0, 1.0, 5000, 100)[0] == 50.0                 # capped
    assert abs(nu1(3000, 46) - 4500 * 3000 ** -0.5 * 46 ** -0.5) < 1e-12


def test_better_oil_gives_longer_life():
    lo = modified_life(20, 47, 13500, 6550, 1500, 750, 15.0)
    hi = modified_life(20, 47, 13500, 6550, 1500, 750, 45.0)
    assert hi["kappa"] > lo["kappa"] and hi["L_nm_h"] > lo["L_nm_h"]


def test_variants_carry_oil_grade_and_modified_life():
    from physics.variants import generate_variants
    from tests.test_manufacturable import S1
    v = generate_variants(S1, compare_basis=False)["variants"][0]
    vg = v["lubricant"]["iso_vg"]
    assert vg in (68, 100, 150, 220, 320)
    for s in v["shafts"]:
        for b in s["bearings"]:
            assert b["modified_life"]["L_nm_h"] >= 5000 and b["modified_life"]["oil_iso_vg"] == vg
    assert any(f"ISO VG {vg}" in n for n in v["notes"])
