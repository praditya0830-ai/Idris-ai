"""ISO 6336 pitting rating of one gear pair, benchmarked on ISO/TR 6336-30 Example 1."""
from ace_core import SolverSpec, Benchmark, register
from physics.iso6336 import rate_pair

_N = {"type": "number"}
SCHEMA = {
    "mn": {"type": "number", "min": 0.3, "max": 50, "unit": "mm", "required": True},
    "alpha_n_deg": {"type": "number", "min": 14.5, "max": 30, "default": 20.0},
    "beta_deg": {"type": "number", "min": 0, "max": 40, "default": 0.0},
    "z1": {"type": "integer", "min": 7, "max": 400, "required": True},
    "z2": {"type": "integer", "min": 7, "max": 1000, "required": True},
    "x1": {"type": "number", "min": -1, "max": 1.5, "default": 0.0},
    "x2": {"type": "number", "min": -1, "max": 1.5, "default": 0.0},
    "b": {"type": "number", "min": 1, "max": 2000, "unit": "mm", "required": True},
    "a_w": {"type": "number", "min": 1, "max": 20000, "unit": "mm"},
    "da1": {"type": "number", "min": 1, "unit": "mm"},
    "da2": {"type": "number", "min": 1, "unit": "mm"},
    "haP": {"type": "number", "min": 0.8, "max": 1.4, "default": 1.0},
    "hfP": {"type": "number", "min": 1.0, "max": 1.6, "default": 1.25},
    "torque_1_nm": {"type": "number", "exclusive_min": 0, "max": 1e7, "unit": "N·m", "required": True},
    "n1_rpm": {"type": "number", "exclusive_min": 0, "max": 200000, "unit": "rpm", "required": True},
    "K_A": {"type": "number", "min": 1.0, "max": 3.0, "default": 1.0},
    "K_Hbeta": {"type": "number", "min": 1.0, "max": 3.0, "required": True},
    "accuracy_grade": {"type": "integer", "min": 1, "max": 11, "default": 6},
    "f_pb_um": {"type": "number", "exclusive_min": 0, "max": 200},
    "f_falpha_um": {"type": "number", "exclusive_min": 0, "max": 200},
    "tolerance_on": {"type": "string", "enum": ["wheel", "pinion"], "default": "wheel"},
    "C_a_um": {"type": "number", "min": 0, "max": 500, "default": 0.0},
    "bore_ratio1": {"type": "number", "min": 0, "max": 0.95, "default": 0.0},
    "bore_ratio2": {"type": "number", "min": 0, "max": 0.95, "default": 0.0},
    "material_type": {"type": "string", "enum": ["case_hardened", "through_hardened"], "default": "case_hardened"},
    "sigma_Hlim1": {"type": "number", "min": 200, "max": 2000, "unit": "MPa", "required": True},
    "sigma_Hlim2": {"type": "number", "min": 200, "max": 2000, "unit": "MPa"},
    "E": {"type": "number", "min": 50000, "max": 400000, "default": 206000.0},
    "nu": {"type": "number", "min": 0.1, "max": 0.45, "default": 0.3},
    "density": {"type": "number", "min": 1e-6, "max": 2e-5, "default": 7.83e-6, "unit": "kg/mm³"},
    "Ra_um": {"type": "number", "min": 0.05, "max": 10, "default": 0.8},
    "Ra1_um": {"type": "number", "min": 0.05, "max": 10},
    "Ra2_um": {"type": "number", "min": 0.05, "max": 10},
    "Rz1_um": {"type": "number", "min": 0.1, "max": 60},
    "Rz2_um": {"type": "number", "min": 0.1, "max": 60},
    "nu40": {"type": "number", "min": 10, "max": 3000, "default": 220.0, "unit": "mm²/s"},
    "life_h": {"type": "number", "min": 1, "max": 1e6, "default": 20000.0},
    "contacts_per_rev1": {"type": "integer", "min": 1, "max": 10, "default": 1},
    "contacts_per_rev2": {"type": "integer", "min": 1, "max": 10, "default": 1},
    "Z_NT_at_1e10": {"type": "number", "min": 0.85, "max": 1.0, "default": 0.85},
    "S_Hmin": {"type": "number", "min": 0.5, "max": 3.0, "default": 1.0},
}

EX1_INPUTS = {"mn": 8.0, "alpha_n_deg": 20.0, "beta_deg": 15.8, "z1": 17, "z2": 103, "x1": 0.145, "x2": 0.0,
              "b": 100.0, "a_w": 500.0, "hfP": 1.4, "torque_1_nm": 9000.0, "n1_rpm": 360.0, "K_A": 1.0,
              "K_Hbeta": 1.16, "accuracy_grade": 5, "Ra_um": 1.0, "nu40": 320.0, "sigma_Hlim1": 1500.0,
              "life_h": 50000.0, "C_a_um": 70.0, "bore_ratio1": 0.5, "bore_ratio2": 0.5}

# Reference values of ISO/TR 6336-30:2017 Example 1 (Annex A), as transcribed in the open-source
# Drivetrain project (github.com/gfsReboucas/Drivetrain, MATISO_6336.example_01). We have not read the
# TR itself; the values are the published results, used here as test data.
EX1_EXPECTED = {
    "outputs.loads.F_t": 127352.0, "outputs.loads.v": 2.664,
    "outputs.geometry.z_n1": 18.905, "outputs.geometry.z_n2": 114.543,
    "outputs.loads.N_L1": 1.080e9, "outputs.loads.N_L2": 1.783e8,
    "outputs.stiffness.c_th": 17.85584, "outputs.stiffness.c_prime": 12.37047,
    "outputs.stiffness.c_gamma_alpha": 17.46485, "outputs.stiffness.c_gamma_beta": 14.84512,
    "outputs.factors.K_v": 1.003, "outputs.factors.K_Halpha": 1.0, "outputs.factors.K_Falpha": 1.0,
    "outputs.factors.K_Fbeta": 1.12803,
    "outputs.factors.Z_H": 2.39533, "outputs.factors.Z_E": 189.81170, "outputs.factors.Z_eps": 0.803,
    "outputs.factors.Z_beta": 1.01944, "outputs.factors.Z_B": 1.0, "outputs.factors.Z_D": 1.0,
    "outputs.factors.Z_NT1": 0.91, "outputs.factors.Z_NT2": 0.962, "outputs.factors.Z_L": 1.04739,
    "outputs.factors.Z_v": 0.96911, "outputs.factors.Z_R": 0.96599, "outputs.factors.Z_X": 1.0,
    "outputs.factors.Z_W": 1.0,
    "outputs.contact.sigma_H0": 1206.58207, "outputs.contact.sigma_H1": 1301.35343,
    "outputs.contact.sigma_H2": 1301.35343, "outputs.contact.sigma_HP1": 1338.48050,
    "outputs.contact.sigma_HP2": 1414.52551, "outputs.contact.S_H1": 1.02853, "outputs.contact.S_H2": 1.08696,
}

register(SolverSpec(
    name="gear_pair_iso6336", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Re-rate an existing spur or helical gear pair for pitting (ISO 6336-2), showing every factor, "
            "so an engineer can compare it with their own tool.",
    method="ISO 6336-1:2019 Method B (c', c_γ, K_v all ranges, K_Hα), K_Fβ from K_Hβ; ISO 6336-2:2019 Method B "
           "(Z_H, Z_E, Z_ε, Z_β, Z_B/Z_D, Z_NT, Z_L, Z_v, Z_R, Z_X); ISO 1328-1:2013 tolerances when not measured",
    schema=SCHEMA,
    run=rate_pair,
    benchmarks=[Benchmark(
        "iso-tr-6336-30-example-1",
        "ISO/TR 6336-30:2017 Example 1 (single helical, case carburized), Annex A results - 33 quantities "
        "(values via the open-source Drivetrain project, MATISO_6336.example_01)",
        EX1_INPUTS, EX1_EXPECTED, 0.002)],
    limitations=["K_Hβ is an input (not yet computed from shaft deflection)",
                 "pitting only: tooth-root bending of this pair not in this solver",
                 "external spur/helical, steel/steel; no internal or double-helical gears",
                 "Z_W = 1; limited pitting not permitted; single load level (no load spectrum)",
                 "K_v outside the subcritical range not yet checked against a worked example"],
    example=EX1_INPUTS,
))
