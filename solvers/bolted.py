"""Bolted joint solver (cover bolts, dowels) — Shigley 10e ch. 8, ISO 898-1, VDI 2230-style checks."""
import math
from ace_core import SolverSpec, Benchmark, Check, register
from physics.bolted_joint import (METRIC, CLASSES, joint_check, member_stiffness, wileman_stiffness,
                                  proof_load, E_STEEL, TAN30)


_AT_M8 = math.pi / 4 * (8 - 0.9382 * 1.25) ** 2
_KB_M8 = _AT_M8 * E_STEEL / 16.0
_KM_M8 = 0.5774 * math.pi * E_STEEL * 8 / (2 * math.log(5 * (0.5774 * 16 + 0.5 * 8) / (0.5774 * 16 + 2.5 * 8)))
_C_M8 = _KB_M8 / (_KB_M8 + _KM_M8)


def _run(p):
    pos = p.get("bolt_positions_mm") or [[-30, -30], [30, -30], [-30, 30], [30, 30]]
    r = joint_check(p["size"], p["property_class"], p["cover_thickness_mm"], p["E_cover_mpa"],
                    p["E_body_mpa"], [tuple(x) for x in pos], p["shear_n"],
                    (p["moment_x_nmm"], p["moment_y_nmm"]), p["axial_n"], mu=p["friction"],
                    n_dowels=p["dowels"], d_dowel=p.get("dowel_diameter_mm"),
                    Su_body=p.get("Su_body_mpa"), Sy_cover=p.get("Sy_cover_mpa"))
    checks = [Check(f"{k} safety factor", v if v is not None else 1e9, r["targets"][k], "-", "min")
              for k, v in r["safety_factors"].items()]
    return {"outputs": r, "checks": checks}


def _run_stiffness(p):
    d, l, E = p["d_mm"], p["grip_mm"], p["E_mpa"]
    km = member_stiffness(d, [(l / 2, E), (l / 2, E)])
    # Shigley eq. 8-22: two identical members, 30° frusta, d_w = 1.5 d (closed form)
    closed = 0.5774 * math.pi * E * d / (2 * math.log(5 * (0.5774 * l + 0.5 * d) / (0.5774 * l + 2.5 * d)))
    return {"outputs": {"km_frustum_n_mm": km, "km_shigley_8_22_n_mm": closed,
                        "km_wileman_n_mm": wileman_stiffness(d, l, E, p["material"])}}


def _run_proof(p):
    return {"outputs": {"proof_load_n": proof_load(p["size"], p["property_class"])}}


SCHEMA = {
    "size": {"type": "string", "enum": list(METRIC), "default": "M5"},
    "property_class": {"type": "string", "enum": list(CLASSES), "default": "8.8"},
    "cover_thickness_mm": {"type": "number", "exclusive_min": 0, "max": 200, "default": 4.0},
    "E_cover_mpa": {"type": "number", "exclusive_min": 0, "default": 71700.0},
    "E_body_mpa": {"type": "number", "exclusive_min": 0, "default": 71700.0},
    "bolt_positions_mm": {"type": "array", "desc": "[[x, y], ...] in the joint plane"},
    "shear_n": {"type": "number", "min": 0, "default": 0.0},
    "moment_x_nmm": {"type": "number", "default": 0.0},
    "moment_y_nmm": {"type": "number", "default": 0.0},
    "axial_n": {"type": "number", "min": 0, "default": 0.0},
    "friction": {"type": "number", "min": 0.05, "max": 0.6, "default": 0.15},
    "dowels": {"type": "integer", "min": 0, "max": 8, "default": 2},
    "dowel_diameter_mm": {"type": "number", "exclusive_min": 0},
    "Su_body_mpa": {"type": "number", "exclusive_min": 0, "default": 572.0},
    "Sy_cover_mpa": {"type": "number", "exclusive_min": 0, "default": 503.0},
}

register(SolverSpec(
    name="bolted_joint", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Bolted cover joint: preload, joint constant, yield / load-factor / separation / slip safety, "
            "dowel shear, thread stripping in the tapped housing and head pressure.",
    method="Shigley 10e ch. 8 (eq. 8-17, 8-20, 8-24, 8-28…8-31), ISO 898-1 proof stress, VDI 2230-style slip, "
           "stripping and head-pressure checks; preload 0.75 F_p with ± 25 % torque scatter",
    schema=SCHEMA, run=_run,
    benchmarks=[Benchmark(
        "shigley-8-22-joint-constant-m8",
        "Independent hand calculation, Shigley 10e: tapped steel joint M8, cover 12 mm → l' = h + d/2 = 16 mm "
        "(Table 8-7), k_b = A_t E / l' (eq. 8-17), k_m by eq. 8-22 (identical members), C = k_b/(k_b + k_m) "
        "(eq. 8-24), F_i = 0.75 S_p A_t, separation n_0 = 0.75 F_i / (P (1 − C)) at P = 2 kN",
        {"size": "M8", "cover_thickness_mm": 12.0, "E_cover_mpa": E_STEEL, "E_body_mpa": E_STEEL,
         "bolt_positions_mm": [[0, 0]], "axial_n": 2000.0, "Su_body_mpa": 600.0, "Sy_cover_mpa": 600.0},
        {"outputs.joint_constant": round(_C_M8, 4),
         "outputs.safety_factors.separation": round(0.75 * 0.75 * 580 * _AT_M8 / (2000 * (1 - _C_M8)), 3)}, 0.005)],
    limitations=["static loads (no bolt fatigue check yet)", "metric coarse ISO 4762 screws into tapped holes",
                 "thread engagement 2·d and stripping area 0.5·π·d·L_e are guideline values, not VDI 2230 full",
                 "friction μ = 0.15 (dry machined aluminium) unless given"],
    example={"size": "M5", "shear_n": 1500.0, "moment_x_nmm": 15000.0},
))

register(SolverSpec(
    name="bolted_stiffness", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Member stiffness of a bolted joint from 30° pressure frusta (used by bolted_joint).",
    method="Frusta in series (Shigley eq. 8-20); compared with Shigley eq. 8-22 and Wileman's FE fit (eq. 8-23)",
    schema={"d_mm": {"type": "number", "exclusive_min": 0, "default": 10.0},
            "grip_mm": {"type": "number", "exclusive_min": 0, "default": 20.0},
            "E_mpa": {"type": "number", "exclusive_min": 0, "default": E_STEEL},
            "material": {"type": "string", "enum": ["steel", "aluminum"], "default": "steel"}},
    run=_run_stiffness,
    benchmarks=[
        Benchmark("shigley-8-22-closed-form", "Shigley 10e eq. 8-22: two identical members, 30° frusta, "
                  "d_w = 1.5 d (independent closed form), d = 10 mm, l = 20 mm, steel",
                  {"d_mm": 10.0, "grip_mm": 20.0, "E_mpa": E_STEEL},
                  {"outputs.km_frustum_n_mm": 0.5774 * math.pi * E_STEEL * 10.0 /
                   (2 * math.log(5 * (0.5774 * 20 + 5.0) / (0.5774 * 20 + 25.0)))}, 0.001),
        Benchmark("wileman-fe-fit-l-2d", "Wileman et al. (1991) FE fit, Shigley 10e eq. 8-23 / Table 8-8 (steel: "
                  "A = 0.78715, B = 0.62873); frusta agree within ±10 % for 0.5 ≤ l/d ≤ 4, checked here at l/d = 2",
                  {"d_mm": 10.0, "grip_mm": 20.0, "E_mpa": E_STEEL},
                  {"outputs.km_frustum_n_mm": E_STEEL * 10.0 * 0.78715 * math.exp(0.62873 * 10.0 / 20.0)}, 0.05),
    ],
    limitations=["axisymmetric pressure-cone model; valid when the members are wide compared with d_w"],
    example={"d_mm": 5.0, "grip_mm": 6.5, "E_mpa": 71700.0, "material": "aluminum"},
))

register(SolverSpec(
    name="bolt_proof_load", version="0.1.0", domain="machine_elements", fidelity="L0",
    summary="Proof load of a metric coarse screw: A_t = π/4 (d − 0.9382 P)² × S_p (ISO 898-1).",
    method="ISO 898-1 tensile stress area and proof stress",
    schema={"size": {"type": "string", "enum": list(METRIC), "default": "M5"},
            "property_class": {"type": "string", "enum": list(CLASSES), "default": "8.8"}},
    run=_run_proof,
    benchmarks=[Benchmark(f"iso-898-1-{s.lower()}-8.8", f"ISO 898-1 proof load table, {s} coarse, class 8.8: {v:,} N",
                          {"size": s, "property_class": "8.8"}, {"outputs.proof_load_n": float(v)}, 0.01)
                for s, v in (("M5", 8230), ("M8", 21200), ("M10", 33700))],
    limitations=["d ≤ 16 mm proof stresses"],
    example={"size": "M6", "property_class": "10.9"},
))
