"""Modified rating life of deep-groove ball bearings — ISO 281:2007 (L_nm = a1 · a_ISO · L10).

Sources and checks
- a_ISO for radial ball bearings (ISO 281:2007), in three viscosity-ratio ranges:
      a_ISO = 0.1 · [1 − (2.5671 − A / κ^B)^0.83 · (e_C · C_u / P)^(1/3)]^(−9.3),  a_ISO ≤ 50
      0.1 ≤ κ < 0.4: A = 2.2649, B = 0.054381
      0.4 ≤ κ < 1  : A = 1.9987, B = 0.19087
      1   ≤ κ ≤ 4  : A = 1.9987, B = 0.071739     (κ > 4 → κ = 4; κ < 0.1 → method not applicable)
  Constants: "Evolution of rolling bearing life rating through the standardization", MTM 2020 (Table 1).
  Benchmark: bearing 6206, κ = 0.8, e_C = 0.2078, C_u = 0.509 kN, P = 9.7 kN → a_ISO = 0.331 (same paper,
  Table 3). Limits a_ISO ≤ 50, κ clamp: NTN catalogue A-21.
- Viscosity ratio κ = ν / ν1 with ν1 = 45000 n^−0.83 D_pw^−0.5 (n < 1000 min⁻¹) or 4500 n^−0.5 D_pw^−0.5
  (n ≥ 1000 min⁻¹), D_pw ≈ (d + D)/2 (NTN catalogue formulas 3.10, 3.11).
- Fatigue load limit C_u = C0 / 22 for ball bearings with D_pw ≤ 100 mm, × (100/D_pw)^0.5 above
  (ISO 281 Annex; reproduces C_u = 0.509 kN of the 6206 example from C0 = 11.2 kN).
- Contamination factor e_C (NTN Table 3.4, D_pw < 100 mm): the lower end of each class is used.
"""
import math

A_B = ((0.1, 0.4, 2.2649, 0.054381), (0.4, 1.0, 1.9987, 0.19087), (1.0, 4.0001, 1.9987, 0.071739))
E_C = {  # D_pw < 100 mm, D_pw >= 100 mm (lower end of the range = conservative)
    "extreme_cleanliness": (1.0, 1.0), "high_cleanliness": (0.6, 0.8), "normal_cleanliness": (0.5, 0.6),
    "slight_contamination": (0.3, 0.4), "typical_contamination": (0.1, 0.2), "severe_contamination": (0.0, 0.0),
}


def nu1(n_rpm, dpw_mm):
    if n_rpm < 1000:
        return 45000.0 * n_rpm ** -0.83 * dpw_mm ** -0.5
    return 4500.0 * n_rpm ** -0.5 * dpw_mm ** -0.5


def fatigue_load_limit(C0_n, dpw_mm):
    cu = C0_n / 22.0
    return cu * (100.0 / dpw_mm) ** 0.5 if dpw_mm > 100 else cu


def a_iso(kappa, eC, Cu_n, P_n):
    """Returns (a_ISO, note). κ < 0.1 → (None, reason)."""
    if kappa < 0.1:
        return None, "κ < 0.1: lubricant film too thin for ISO 281 a_ISO (method not applicable)"
    k = min(kappa, 4.0)
    A, B = next((a, b) for lo, hi, a, b in A_B if lo <= k < hi)
    x = (2.5671 - A / k ** B)
    if x < 0:
        x = 0.0
    inner = 1.0 - x ** 0.83 * (max(eC, 0.0) * Cu_n / P_n) ** (1.0 / 3.0)
    if inner <= 0:
        return 50.0, "a_ISO capped at 50"
    a = 0.1 * inner ** -9.3
    return (min(a, 50.0), "a_ISO capped at 50" if a > 50 else None)


def modified_life(d, D, C_n, C0_n, Fr_n, n_rpm, nu_mm2s, contamination="normal_cleanliness"):
    dpw = 0.5 * (d + D)
    k = nu_mm2s / nu1(n_rpm, dpw)
    eC = E_C[contamination][0 if dpw < 100 else 1]
    Cu = fatigue_load_limit(C0_n, dpw)
    L10_h = (C_n / Fr_n) ** 3 * 1e6 / (60.0 * n_rpm) if Fr_n > 0 else math.inf
    a, note = a_iso(k, eC, Cu, Fr_n) if Fr_n > 0 else (50.0, None)
    return {"dpw_mm": dpw, "nu1_mm2_s": nu1(n_rpm, dpw), "kappa": k, "e_C": eC, "contamination": contamination,
            "Cu_n": Cu, "L10_h": L10_h, "a_ISO": a, "L_nm_h": (a * L10_h) if a is not None else None, "note": note}
