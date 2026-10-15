"""Housing analyses for the UI's Analysis tab: modal (L2), thermal (L2), design review (L0)."""
import math
import os
import time
import numpy as np
from ace_core import SolverSpec, Benchmark, Check, register
from physics.fem3d import VoxelModel, box_model
from physics.housing import (HOUSING_MATERIALS, find_material, build_layout, build_model, fixture_nodes,
                             wetted_face_mask)
from physics.modal3d import natural_frequencies, natural_frequencies_fast
from physics.thermal3d import ThermalModel, h_natural_air, h_forced_air
from physics.thermal_lumped import (fin_layout, fin_conv_factor, fin_mass_kg, FIN_THICKNESS_MM,
                                    FIN_MIN_GAP_MM)
from physics.review import review
from .structural3d import VARIANT_DESIGN, VARIANT_KIN, MESH_DIVISIONS, FRAME_NOTE
from ._example import EXAMPLE_VARIANT

REG_F1 = 1347.39     # regression values, set after a validated run
REG_Q = 443.12
MAX_MODAL_ELEMENTS = int(os.environ.get("IDRIS_MODAL_MAX_ELEMENTS", "12000"))
MAX_THERMAL_ELEMENTS = int(os.environ.get("IDRIS_THERMAL_MAX_ELEMENTS", "40000"))
# thermal conductivity W/(m·K) of the housing materials
CONDUCTIVITY = {"Al7075-T6": 130, "Al6061-T6": 167, "Al2024-T3": 121, "A356-T6": 151, "Ti-6Al-4V": 6.7,
                "Steel-1045": 50, "Steel-4140": 42, "CastIron-GG25": 48, "Mg-AZ91D": 72}
OIL_GRADES = {"ISO VG 68": 68, "ISO VG 100": 100, "ISO VG 150": 150, "ISO VG 220": 220}
OIL_LIMIT_C = 90.0          # common continuous limit for mineral gear oils
TARGET_OIL_C = 85.0         # design target used by the repair options (5 °C margin)
EMISSIVITY = 0.85           # painted / anodised housing (bare machined Al ≈ 0.1–0.2)

VARIANT_SCHEMA = {
    "design": {"type": "object", "required": True, "fields": VARIANT_DESIGN, "allow_extra": True},
    "kinematics": {"type": "object", "required": True, "fields": VARIANT_KIN, "allow_extra": True},
    "loads": {"type": "object", "default": {}},
    "shaft": {"type": "object", "default": {}},
    "housing_params": {"type": "object", "default": {}},
    "housing_material": {"type": "string", "default": "Al7075-T6"},
}
def _material(p):
    mat = find_material(p["housing_material"])
    if mat is None:
        raise ValueError(f"Unknown housing material '{p['housing_material']}'. "
                         f"Available: {', '.join(HOUSING_MATERIALS)}")
    return mat


def _layout(p):
    try:
        return build_layout(p["design"], p["kinematics"], p.get("loads") or {}, p.get("shaft") or {},
                            p.get("housing_params") or {})
    except (KeyError, ValueError, TypeError) as ex:
        raise ValueError(f"Cannot build the housing model from this variant: {ex}")


def _surface(model, L):
    ids, tris = model.surface()
    return ids, tris, np.round(model.nodes[ids], 2).ravel()


# ------------------------------------------------------------------ modal
def _run_modal(p, progress):
    try:
        mat = _material(p)
        L = _layout(p)
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    E, nu, Sy, rho, _ = HOUSING_MATERIALS[mat]
    progress("Meshing", 0.05)
    pre_warn = []
    mesh_used = p["mesh"]
    while True:
        model = build_model(L, mesh_used)
        h = model.h
        ne = len(model.elems)
        if ne <= MAX_MODAL_ELEMENTS:
            break
        if mesh_used == "coarse":
            return {"ok": False, "errors": [
                f"This housing needs {ne:,} elements even on mesh 'coarse'; the server limit for modal analysis "
                f"is {MAX_MODAL_ELEMENTS:,}."]}
        pre_warn.append(f"Mesh '{mesh_used}' ({ne:,} elements) is above the server limit for eigenvalue problems "
                        f"({MAX_MODAL_ELEMENTS:,}); solved on a coarser mesh. For this housing type the first "
                        "frequencies change by less than 2 % between 'coarse' and 'standard'.")
        mesh_used = {"fine": "standard", "standard": "coarse"}[mesh_used]
    progress("Applying loads", 0.1)
    fixed = fixture_nodes(model, L, p["fixture"])
    progress("Solving", 0.15)
    t0 = time.perf_counter()

    def cb(it, res):
        progress("Solving", min(0.85, 0.15 + 0.7 * it / (it + 300)))

    f, shapes, info = natural_frequencies_fast(model, E, nu, rho, fixed, p["modes"], progress=cb)
    solve_s = time.perf_counter() - t0
    progress("Post-processing", 0.9)
    ids, tris, pos = _surface(model, L)
    modes = [{"frequency_hz": round(float(fk), 2), "displacement": np.round(sh[ids], 5).ravel(),
              "effective_mass_fraction_xyz": [round(v, 4) for v in info["effective_mass_fraction_xyz"][k]]}
             for k, (fk, sh) in enumerate(zip(f, shapes))]
    # excitation frequencies from the variant (shaft speeds and gear mesh 1x, 2x)
    kin = p["kinematics"]
    excit = []
    rpm_out = kin.get("output_rpm")
    ratio = kin.get("ratio_actual")
    if isinstance(rpm_out, (int, float)) and isinstance(ratio, (int, float)):
        rpm_in = rpm_out * ratio
        i_st = p["design"]["gear_teeth"] / p["design"]["pinion_teeth"]
        rpms = [rpm_in / i_st ** k for k in range(L["stages"] + 1)]
        for k, r in enumerate(rpms):
            excit.append({"source": f"shaft {k + 1} speed", "hz": r / 60})
        for s in range(L["stages"]):
            gm = rpms[s] / 60 * p["design"]["pinion_teeth"]
            excit += [{"source": f"stage {s + 1} gear mesh 1x", "hz": gm},
                      {"source": f"stage {s + 1} gear mesh 2x", "hz": 2 * gm}]
    warnings = pre_warn + [FRAME_NOTE, "Mode shapes are normalised to max |u| = 1 (arbitrary scale).",
                "Housing modelled empty: mass and stiffness of gears, shafts and oil are not included "
                "(real frequencies are usually somewhat lower)."]
    checks = []
    for e in excit:
        near = [m for m in modes if abs(m["frequency_hz"] - e["hz"]) <= 0.1 * e["hz"]]
        for m in near:
            warnings.append(f"Mode at {m['frequency_hz']:.0f} Hz is within ±10 % of {e['source']} ({e['hz']:.0f} Hz): resonance risk.")
    if excit:
        margin = min(abs(m["frequency_hz"] - e["hz"]) / e["hz"] for m in modes for e in excit)
        checks.append(Check("min separation mode ↔ excitation", margin, 0.1, "-", "min",
                            "±10 % rule; UI also flags this"))
    out = {"surface": {"positions": pos, "indices": tris.ravel(), "modes": modes},
           "mesh": {"elements": ne, "nodes": model.n_nodes, "type": "hex8-im (rectilinear brick)",
                    "element_size_mm": round(h, 3), "surface_triangles": int(len(tris))},
           "solve_time_s": round(solve_s, 2), "frequencies_hz": [m["frequency_hz"] for m in modes],
           "excitations_hz": excit, "eigen_method": info["method"], "material": mat,
           "housing_mass_kg": info["total_mass_kg"], "fixture": p["fixture"], "mesh_level": mesh_used}
    progress("Post-processing", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


register(SolverSpec(
    name="modal_housing_3d", version="0.1.0", domain="structural_dynamics", fidelity="L2", progress=True,
    summary="Natural frequencies and mode shapes of the gearbox housing, with resonance check against shaft and gear-mesh frequencies.",
    method=("3D FEM, 8-node incompatible-mode hexahedra on a voxel mesh, lumped mass; generalized eigenproblem by "
            "shift-invert Lanczos (scipy) or LOBPCG with block-Jacobi preconditioner (numpy)"),
    schema={**VARIANT_SCHEMA,
            "analysis": {"type": "string", "enum": ["modal"], "default": "modal"},
            "modes": {"type": "integer", "enum": [4, 6, 10], "default": 6},
            "fixture": {"type": "string", "enum": ["base", "flange"], "default": "base"},
            "mesh": {"type": "string", "enum": list(MESH_DIVISIONS), "default": "coarse"}},
    run=_run_modal,
    benchmarks=[Benchmark("regression-housing-modal-coarse",
                          "ACE regression v0.2 on a real variant; LOBPCG and scipy eigsh agree (self-consistency)",
                          {**EXAMPLE_VARIANT, "modes": 4, "mesh": "coarse"},
                          {"outputs.frequencies_hz.0": REG_F1}, 0.01)],
    limitations=["empty housing (no gears/shafts/oil mass)", "linear, undamped, small vibrations",
                 f"server cap {MAX_MODAL_ELEMENTS:,} elements (use mesh 'coarse')"],
    example={**EXAMPLE_VARIANT, "modes": 4, "mesh": "coarse"},
))


def _run_block_modal(p):
    m = box_model(p["length_mm"], p["width_mm"], p["height_mm"], p["element_size_mm"])
    if len(m.elems) > MAX_MODAL_ELEMENTS:
        return {"ok": False, "errors": ["too many elements; increase element_size_mm"]}
    fixed = np.where(m.nodes[:, 0] < 1e-9)[0]
    f, _, info = natural_frequencies_fast(m, p["E_MPa"], p["nu"], p["density_kg_m3"], fixed, 4)
    return {"outputs": {"frequencies_hz": [float(v) for v in f], "eigen_method": info["method"],
                        "elements": len(m.elems)}}


register(SolverSpec(
    name="modal_solid_block", version="0.1.0", domain="structural_dynamics", fidelity="L2",
    summary="Natural frequencies of a cantilevered solid block (engine verification).",
    method="Same eigen engine as modal_housing_3d",
    schema={"length_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "width_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "height_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "element_size_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "E_MPa": {"type": "number", "exclusive_min": 0, "default": 200000.0},
            "nu": {"type": "number", "min": 0, "max": 0.49, "default": 0.3},
            "density_kg_m3": {"type": "number", "exclusive_min": 0, "default": 7850.0}},
    run=_run_block_modal,
    benchmarks=[Benchmark("cantilever-euler-bernoulli-f1",
                          "Euler-Bernoulli cantilever: f₁ = (1.8751²/2π)·√(EI/ρAL⁴), L/H = 20 (Blevins, Formulas for Natural Frequency, Table 8-1)",
                          {"length_mm": 200, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5},
                          {"outputs.frequencies_hz.0": 203.8}, 0.02),
                Benchmark("cantilever-euler-bernoulli-f2",
                          "Euler-Bernoulli cantilever 2nd bending mode: β₂L = 4.6941",
                          {"length_mm": 200, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5},
                          {"outputs.frequencies_hz.2": 1277.5}, 0.02)],
    limitations=["voxel elements"],
    example={"length_mm": 200, "width_mm": 10, "height_mm": 10, "element_size_mm": 2.5},
))


# ------------------------------------------------------------------ thermal
OPTION_KINDS = ["fan", "fins", "oil_cooler", "efficiency", "larger_housing", "combined"]


def _scaled_layout(p, scale):
    """Layout with every outer housing dimension × scale (gears stay put, more air-side area)."""
    L0 = _layout(p)
    if scale <= 1.0 + 1e-9:
        return L0
    hp = dict(p.get("housing_params") or {})
    W, H, D = L0["W"], L0["H"], L0["D"]
    hp.update(envelope_width=round(W * scale), envelope_height=round(H * scale), envelope_depth=round(D * scale))
    return build_layout(p["design"], p["kinematics"], p.get("loads") or {}, p.get("shaft") or {}, hp)


def _shell_volume_mm3(L):
    W, H, D, t = L["W"], L["H"], L["D"], L["wall_mm"]
    return W * H * D - (W - 2 * t) * (H - 2 * t) * (D - 2 * t)


class _ThermalCase:
    """One housing geometry: mesh + oil node; `state()` answers 'oil temperature for heat Q' or
    'heat shed at oil ΔT' for any outer cooling (natural / forced, fins) by superposition."""

    def __init__(self, p, L, k, mesh):
        self.p, self.L, self.k = p, L, k
        self.model = build_model(L, mesh)
        self.ne = len(self.model.elems)
        if self.ne > MAX_THERMAL_ELEMENTS:
            raise ValueError(f"Mesh '{mesh}' needs {self.ne:,} elements but this server allows "
                             f"{MAX_THERMAL_ELEMENTS:,}. Please choose a coarser mesh.")
        th = self.th = ThermalModel(self.model, k)
        self.wet = wetted_face_mask(th, L)
        self.out_mask = ~self.wet
        self.A_in = float(th.face_area[self.wet].sum()) * 1e-6
        self.A_out = float(th.face_area[self.out_mask].sum()) * 1e-6
        v = (p.get("loads") or {}).get("pitch_line_velocity_ms")
        if not isinstance(v, (int, float)) or v <= 0:
            dp = p["design"]["module_mm"] * p["design"]["pinion_teeth"]
            v = 3.1416 * dp * p["input_rpm"] / 60000
        self.h_in = 200.0 * min(2.0, max(0.5, (v / 5.0) ** 0.5)) * (68 / OIL_GRADES[p["oil"]]) ** 0.2
        self.H_mm, self.D_mm, self.W_mm = L["H"], L["D"], L["W"]
        self.G_in = th.node_conductance(self.wet, self.h_in)
        self.S_in = float(self.G_in.sum())
        self.Ta = float(p["ambient_c"])
        self._b = None

    def state(self, dT_oil=None, Q_in=None, cooling="natural", speed=0.0, area_factor=1.0, fins=None):
        Ta = self.Ta
        fl = fin_layout(fins, self.H_mm, self.D_mm) if fins else None
        T_s = Ta + 20.0
        b = self._b
        f_fin, eta = 1.0, None
        for _ in range(12):
            if cooling == "forced":
                hc, hr, _ = h_forced_air(speed, self.W_mm * 1e-3, T_s, Ta, EMISSIVITY)
            else:
                hc, hr = h_natural_air(T_s - Ta, self.H_mm * 1e-3, T_s, Ta, EMISSIVITY)
            f_fin, eta = fin_conv_factor(fl, hc, self.k, self.A_out, cooling, T_s, Ta)
            G_out = self.th.node_conductance(self.out_mask, (hc * f_fin + hr) * area_factor)
            b, _it = self.th.solve_linear(self.G_in + G_out, self.G_in.copy(), x0=b)
            K_eff = self.S_in - float(self.G_in @ b)
            dT = dT_oil if dT_oil is not None else Q_in / K_eff
            sel = G_out > 0
            frac = float((G_out[sel] * b[sel]).sum() / G_out[sel].sum())
            T_new = Ta + frac * dT
            done = abs(T_new - T_s) < 0.05
            T_s = T_new
            if done:
                break
        self._b = b
        return {"T_oil": Ta + dT, "Q": K_eff * dT, "K_eff": K_eff, "b": b, "G_out": G_out, "hc": hc, "hr": hr,
                "T_s": T_s, "dT": dT, "fin_factor": f_fin, "fin_efficiency": eta,
                "fin_gap_mm": fl["gap_mm"] if fl else None}


def _fin_candidates(case):
    """A small grid of practical fin sets for this housing (count from a target gap, height)."""
    tf = FIN_THICKNESS_MM
    out = []
    for gap in (12.0, 8.0):
        per_side = int((case.D_mm) // (gap + tf))
        if per_side < 2:
            continue
        for hf in (15.0, 25.0, 35.0):
            out.append({"count": 2 * per_side, "height_mm": hf})
    return out


def _run_thermal(p, progress):
    try:
        mat = _material(p)
        L = _scaled_layout(p, p["housing_scale"])
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    warnings = [FRAME_NOTE]
    k = CONDUCTIVITY[mat]
    rho = HOUSING_MATERIALS[mat][3]
    n_st = L["stages"]
    if p.get("efficiency_per_stage") is not None:
        eta = float(p["efficiency_per_stage"]) ** n_st
    else:
        eta = float(p["kinematics"].get("efficiency", 0.97))
    Q_gen = p["power_kw"] * 1000 * (1 - eta)
    cooler = float(p["oil_cooler_w"])
    Q = max(0.0, Q_gen - cooler)
    goals = p.get("goals") or {}
    limit = float(goals.get("oil_temp_max_c", OIL_LIMIT_C))
    target = min(TARGET_OIL_C, limit - 5.0)
    allowed = goals.get("allowed") or OPTION_KINDS
    mass_cap = goals.get("added_mass_max_kg")
    fins = p.get("fins") or None
    if fins and (not fins.get("count") or not fins.get("height_mm")):
        fins = None
    if p["cooling"] == "forced" and p["air_speed_ms"] <= 0:
        return {"ok": False, "errors": ["cooling 'forced' needs air_speed_ms > 0 (2, 5 or 10)"]}
    progress("Meshing", 0.05)
    try:
        case = _ThermalCase(p, L, k, p["mesh"])
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    model, h, ne = case.model, case.model.h, case.ne
    fl_now = fin_layout(fins, case.H_mm, case.D_mm) if fins else None
    if fl_now and fl_now["gap_mm"] < FIN_MIN_GAP_MM:
        return {"ok": False, "errors": [f"{fins['count']} fins do not fit on {case.D_mm:.0f} mm depth: gap "
                                        f"{fl_now['gap_mm']:.1f} mm < {FIN_MIN_GAP_MM:.0f} mm. Use fewer fins."]}
    cool, spd = p["cooling"], p["air_speed_ms"]
    progress("Applying loads", 0.15)
    progress("Solving", 0.2)
    t0 = time.perf_counter()
    st = case.state(Q_in=Q, cooling=cool, speed=spd, fins=fins)
    T_oil, hc, hr, T_s, G_out = st["T_oil"], st["hc"], st["hr"], st["T_s"], st["G_out"]
    T = case.Ta + st["b"] * st["dT"]
    Ta = case.Ta
    fin_mass_now = fin_mass_kg(fl_now, rho)

    # ---- legacy `fixes` (kept for UI v18 compatibility): what brings the oil to TARGET_C
    fixes = []
    options = []
    if T_oil > limit:
        progress("Solving", 0.45)
        dT_t = target - Ta
        if dT_t <= 0:
            fixes.append({"fix": "ambient", "detail": f"Ambient {Ta:.0f} °C is above the {target:.0f} °C target."})
        else:
            base = dict(cooling=cool, speed=spd, fins=fins)
            q_allow = case.state(dT_oil=dT_t, **base)["Q"]
            Qg = Q_gen
            fixes.append({"fix": "duty_cycle", "title": "Limit continuous power (duty cycle)",
                          "detail": f"The housing sheds {q_allow:.0f} W at {target:.0f} °C oil: continuous power ≤ "
                                    f"{p['power_kw'] * min(1, (q_allow + cooler) / Qg):.1f} kW. Typical for actuators "
                                    "with short peak loads.",
                          "max_continuous_power_kw": round(p["power_kw"] * min(1, (q_allow + cooler) / Qg), 2),
                          "predicted_oil_temp_c": target})
            need_w = max(0.0, Q - q_allow)
            fixes.append({"fix": "oil_cooler", "title": "Oil cooler / circulating oil",
                          "detail": f"Remove {need_w:.0f} W with an external oil cooler (housing keeps {q_allow:.0f} W).",
                          "cooler_capacity_w": round(need_w), "predicted_oil_temp_c": target})
            Q99 = max(0.0, p["power_kw"] * 1000 * (1 - 0.99 ** n_st) - cooler)
            t99 = case.state(Q_in=Q99, **base)["T_oil"] if Q99 < Q else None
            if t99 is not None:
                fixes.append({"fix": "efficiency", "title": "Ground, profile-corrected gears + low-loss bearings",
                              "detail": f"η 99 %/stage instead of {100 * eta ** (1 / n_st):.1f} %: heat "
                                        f"{Q99:.0f} W → oil {t99:.0f} °C.", "predicted_oil_temp_c": round(t99, 1)})
            fan_t = {}
            for sp in (5.0, 10.0):
                fan_t[sp] = case.state(Q_in=Q, cooling="forced", speed=sp, fins=fins)["T_oil"]
                fixes.append({"fix": f"forced_air_{sp:g}", "title": f"Fan, {sp:g} m/s air over the housing",
                              "detail": f"Oil {fan_t[sp]:.0f} °C.", "predicted_oil_temp_c": round(fan_t[sp], 1)})
            Qc = min(Q, Q99)
            if case.state(dT_oil=dT_t, cooling="forced", speed=5.0, fins=fins)["Q"] >= Qc:
                fixes.append({"fix": "combined", "title": "Combined: 99 %/stage gears + 5 m/s fan",
                              "detail": f"Heat {Qc:.0f} W; reaches ≤ {target:.0f} °C oil without fins.",
                              "predicted_oil_temp_c": target})
            else:
                fixes.append({"fix": "combined", "title": "Combined: 99 %/stage gears + 5 m/s fan + fins",
                              "detail": "See `options` for the fin sets computed on this housing.",
                              "predicted_oil_temp_c": None})
            for f in fixes:
                f.setdefault("predicted_oil_temp_c", None)
                tv = f["predicted_oil_temp_c"]
                f["passes"] = tv is not None and tv <= limit

            # ---- `options` (UI v19 "Try" rows): each entry = a change to THIS request, computed here
            progress("Solving", 0.6)

            def opt(kind, label, change, t_oil, added_kg=0.0, note=""):
                ok_mass = mass_cap is None or added_kg <= float(mass_cap) + 1e-9
                options.append({"kind": kind, "label": label, "change": change, "oil_temp_c": round(t_oil, 1),
                                "pass": bool(t_oil <= limit and ok_mass), "added_mass_kg": round(added_kg, 3),
                                "note": note, "meets_mass_goal": ok_mass})

            if cool == "natural":
                for sp in (5.0, 10.0):
                    opt("fan", f"Fan, {sp:g} m/s over the housing", {"cooling": "forced", "air_speed_ms": sp},
                        fan_t[sp], note="needs an air flow (fan or propeller wash) along the housing")
            # fins under the current cooling, and under a 5 m/s fan if currently natural
            fin_res = []
            for fc in _fin_candidates(case):
                fl = fin_layout(fc, case.H_mm, case.D_mm)
                if fl["gap_mm"] < FIN_MIN_GAP_MM:
                    continue
                for c_, s_ in ([(cool, spd)] + ([("forced", 5.0)] if cool == "natural" else [])):
                    r_ = case.state(Q_in=Q, cooling=c_, speed=s_, fins=fc)
                    fin_res.append((fc, c_, s_, r_["T_oil"], fin_mass_kg(fl, rho), fl["gap_mm"], r_["fin_efficiency"]))
            for c_, s_ in ([(cool, spd)] + ([("forced", 5.0)] if cool == "natural" else [])):
                rows = [r for r in fin_res if r[1] == c_ and r[2] == s_]
                if not rows:
                    continue
                passing = sorted([r for r in rows if r[3] <= limit], key=lambda r: r[4])
                pick = passing[0] if passing else min(rows, key=lambda r: r[3])
                fc, _, _, t_, m_, gap_, eta_f = pick
                ch = {"fins": fc}
                lab = f"Cooling fins {fc['count']} × {fc['height_mm']:g} mm"
                if c_ == "forced" and cool == "natural":
                    ch.update(cooling="forced", air_speed_ms=s_)
                    lab += f" + fan {s_:g} m/s"
                opt("fins" if "cooling" not in ch else "combined", lab, ch, t_, m_,
                    f"on both side walls, {FIN_THICKNESS_MM:g} mm thick, gap {gap_:.0f} mm, fin efficiency "
                    f"{100 * eta_f:.0f} %; +{m_:.2f} kg")
            need_cool = max(0.0, Q - q_allow)
            cw = 50.0 * math.ceil(need_cool / 50.0)
            if cw > 0:
                t_c = case.state(Q_in=max(0.0, Q - cw), cooling=cool, speed=spd, fins=fins)["T_oil"]
                opt("oil_cooler", f"Oil cooler {cw + cooler:.0f} W", {"oil_cooler_w": cw + cooler}, t_c,
                    note="needs an oil pump/circuit and a heat sink (air or fuel)")
            if t99 is not None:
                opt("efficiency", "Gears + bearings at 99 %/stage", {"efficiency_per_stage": 0.99}, t99,
                    note="ground, profile-corrected gears and low-loss bearings: cost, not mass")
                t99f = case.state(Q_in=Q99, cooling="forced", speed=5.0, fins=fins)["T_oil"]
                if cool == "natural":
                    opt("combined", "99 %/stage gears + fan 5 m/s",
                        {"efficiency_per_stage": 0.99, "cooling": "forced", "air_speed_ms": 5.0}, t99f)
            # best combination: 99 %/stage + fan + the lightest fin set that reaches the limit
            if t99 is not None and not any(o["pass"] and o["kind"] != "oil_cooler" for o in options):
                found = None
                for sp in ((5.0, 10.0) if cool == "natural" else (max(spd, 5.0),)):
                    rows = []
                    for fc in _fin_candidates(case):
                        fl = fin_layout(fc, case.H_mm, case.D_mm)
                        if fl["gap_mm"] < FIN_MIN_GAP_MM:
                            continue
                        rows.append((fin_mass_kg(fl, rho), fc, fl["gap_mm"]))
                    for m_, fc, gap_ in sorted(rows, key=lambda r: r[0]):
                        t_ = case.state(Q_in=Q99, cooling="forced", speed=sp, fins=fc)["T_oil"]
                        if t_ <= limit:
                            found = (sp, fc, t_, m_, gap_)
                            break
                    if found:
                        break
                if found:
                    sp, fc, t_, m_, gap_ = found
                    opt("combined", f"99 %/stage gears + fan {sp:g} m/s + fins {fc['count']} × {fc['height_mm']:g} mm",
                        {"efficiency_per_stage": 0.99, "cooling": "forced", "air_speed_ms": sp, "fins": fc}, t_, m_,
                        f"fins on both side walls, gap {gap_:.0f} mm; +{m_:.2f} kg")
                    for f in fixes:
                        if f["fix"] == "combined" and f["predicted_oil_temp_c"] is None:
                            f.update(detail=f"99 %/stage gears + fan {sp:g} m/s + {fc['count']} fins × "
                                            f"{fc['height_mm']:g} mm: oil {t_:.0f} °C.",
                                     predicted_oil_temp_c=round(t_, 1), passes=True)
            # larger housing: a new mesh per size (more outer area, more oil)
            progress("Solving", 0.75)
            vol0 = _shell_volume_mm3(L)
            for sc in (1.3, 1.6):
                try:
                    Ls = _scaled_layout({**p, "housing_params": {**(p.get("housing_params") or {}),
                                                                 "envelope_width": L["W"],
                                                                 "envelope_height": L["H"],
                                                                 "envelope_depth": L["D"]}}, sc)
                    cs = _ThermalCase(p, Ls, k, "coarse")
                except ValueError:
                    continue
                t_s = cs.state(Q_in=Q, cooling=cool, speed=spd, fins=fins)["T_oil"]
                dm = (_shell_volume_mm3(Ls) - vol0) * 1e-9 * rho
                bx = Ls
                opt("larger_housing", f"Larger housing ×{sc:g} ({bx['W']:.0f}×{bx['H']:.0f}×{bx['D']:.0f} mm)",
                    {"housing_scale": round(p["housing_scale"] * sc, 3)}, t_s, dm,
                    f"+{dm:.2f} kg; check the envelope")
            options = [o for o in options if o["kind"] in allowed or
                       (o["kind"] == "combined" and all(x in allowed for x in ("fan",)))]
            options.sort(key=lambda o: (not o["pass"], o["added_mass_kg"] if o["pass"] else o["oil_temp_c"]))
    solve_s = time.perf_counter() - t0
    progress("Post-processing", 0.9)
    q_out = float((G_out * (T - Ta)).sum())
    ids, tris, pos = _surface(model, L)
    if Q < 1:
        warnings.append("Net heat load is below 1 W: temperatures stay at ambient.")
    warnings += [f"Oil-side heat transfer coefficient {case.h_in:.0f} W/m²K is an engineering estimate for splash "
                 "lubrication (scaled with pitch speed and viscosity), not a CFD result.",
                 f"Outer surface: {'forced' if cool == 'forced' else 'natural'} convection correlation "
                 f"(h_c = {hc:.1f} W/m²K) + radiation (ε = {EMISSIVITY}, h_r = {hr:.1f} W/m²K).",
                 "Heat source = P × (1 − η) delivered to a well-mixed oil volume; local hot spots at bearings "
                 "and the gear mesh are not resolved."]
    if fl_now:
        warnings.append(f"Fins: {fl_now['count']} × {fl_now['height_mm']:g} mm on the side walls (gap "
                        f"{fl_now['gap_mm']:.1f} mm, efficiency {100 * (st['fin_efficiency'] or 0):.0f} %) raise the outer "
                        f"convective conductance ×{st['fin_factor']:.2f}; applied uniformly to the outer skin "
                        "(fins are not meshed). Radiation is counted on the base envelope only.")
    if cooler > 0:
        warnings.append(f"Oil cooler removes {cooler:.0f} W directly from the oil (ideal sink).")
    if T_oil > limit:
        warnings.append(f"Oil temperature {T_oil:.0f} °C exceeds {limit:.0f} °C. See `options` for computed changes "
                        "(fan, fins, oil cooler, efficiency, larger housing).")
    checks = [Check("oil temperature", T_oil, limit, "°C", "max",
                    "goal from engineer" if "oil_temp_max_c" in goals else "mineral gear oil continuous limit"),
              Check("heat balance error", abs(q_out - Q) / max(Q, 1e-9) if Q >= 1 else 0.0, 0.01, "-", "max",
                    "solver self-check")]
    goal_report = None
    if goals:
        goal_report = {"oil_temp_max_c": limit, "added_mass_max_kg": mass_cap, "allowed": allowed,
                       "met_now": T_oil <= limit,
                       "met_by": [o["label"] for o in options if o["pass"]],
                       "closest": None if (T_oil <= limit or not options) else
                       min(options, key=lambda o: o["oil_temp_c"])["label"]}
    out = {"surface": {"positions": pos, "indices": tris.ravel(), "temperature": np.round(T[ids], 2)},
           "oil_temp_c": round(T_oil, 2), "heat_dissipated_w": round(q_out, 2), "heat_generated_w": round(Q_gen, 2),
           "heat_to_housing_w": round(Q, 2), "oil_cooler_w": cooler,
           "max_surface_temp_c": round(float(T.max()), 2), "mean_outer_surface_temp_c": round(T_s, 2),
           "h_oil_w_m2k": round(case.h_in, 1), "h_outer_convection_w_m2k": round(hc, 2),
           "h_outer_radiation_w_m2k": round(hr, 2), "area_outer_m2": round(case.A_out, 4),
           "surface_area_m2": round(case.A_out, 4), "area_wetted_m2": round(case.A_in, 4),
           "housing_box_mm": [L["W"], L["H"], L["D"]], "housing_scale": p["housing_scale"],
           "fins": ({**fl_now, "factor": round(st["fin_factor"], 3),
                     "efficiency": round(st["fin_efficiency"] or 0, 3), "mass_kg": round(fin_mass_now, 3)}
                    if fl_now else None),
           "efficiency_total": round(eta, 5), "oil_limit_c": limit,
           "material": mat, "conductivity_w_mk": k,
           "mesh": {"elements": ne, "nodes": model.n_nodes, "type": "hex8 (rectilinear brick)",
                    "element_size_mm": round(h, 3), "surface_triangles": int(len(tris))},
           "solve_time_s": round(solve_s, 2), "outer_iterations": 1,
           "target_oil_temp_c": target, "fixes": fixes, "options": options, "goal_report": goal_report}
    progress("Post-processing", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


register(SolverSpec(
    name="thermal_housing_3d", version="0.1.0", domain="thermal", fidelity="L2", progress=True,
    summary="Steady-state oil and housing temperatures of a gearbox: gear losses P·(1−η) into the oil, conduction through the walls, convection + radiation to ambient.",
    method=("3D conduction FEM (8-node hexahedra, voxel mesh) coupled to a lumped oil node; outer convection from "
            "McAdams (natural) / flat-plate Nusselt (forced) correlations + linearised radiation, iterated to convergence"),
    schema={**VARIANT_SCHEMA,
            "analysis": {"type": "string", "enum": ["thermal"], "default": "thermal"},
            "power_kw": {"type": "number", "min": 0, "max": 10000, "required": True},
            "input_rpm": {"type": "number", "exclusive_min": 0, "max": 1e5, "default": 6000.0},
            "ambient_c": {"type": "number", "min": -40, "max": 80, "default": 25.0},
            "cooling": {"type": "string", "enum": ["natural", "forced"], "default": "natural"},
            "air_speed_ms": {"type": "number", "min": 0, "max": 50, "default": 0.0},
            "oil": {"type": "string", "enum": list(OIL_GRADES), "default": "ISO VG 68"},
            "fins": {"type": "object", "default": {}, "allow_extra": False,
                     "fields": {"count": {"type": "integer", "min": 0, "max": 200},
                                "height_mm": {"type": "number", "min": 0, "max": 100},
                                "thickness_mm": {"type": "number", "min": 1, "max": 10}}},
            "oil_cooler_w": {"type": "number", "min": 0, "max": 1e6, "default": 0.0},
            "housing_scale": {"type": "number", "min": 1.0, "max": 3.0, "default": 1.0},
            "efficiency_per_stage": {"type": "number", "min": 0.8, "max": 0.999},
            "goals": {"type": "object", "default": {}, "allow_extra": False,
                      "fields": {"oil_temp_max_c": {"type": "number", "min": 30, "max": 200},
                                 "added_mass_max_kg": {"type": "number", "min": 0, "max": 1000},
                                 "allowed": {"type": "array", "items_type": "string"}}},
            "mesh": {"type": "string", "enum": list(MESH_DIVISIONS), "default": "coarse"}},
    run=_run_thermal,
    benchmarks=[Benchmark("regression-housing-thermal", "ACE regression v0.1 + heat-balance self-check",
                          {**EXAMPLE_VARIANT, "power_kw": 11.19, "input_rpm": 6000, "ambient_c": 25},
                          {"outputs.heat_dissipated_w": REG_Q}, 0.01)],
    limitations=["not CFD: convection from correlations, oil assumed well mixed",
                 "fins are not meshed: their effect is a fin-efficiency-weighted factor on outer convection",
                 "no sun load, housing emissivity assumed 0.85 (painted)"],
    example={**EXAMPLE_VARIANT, "power_kw": 11.19, "input_rpm": 6000, "ambient_c": 25, "cooling": "natural",
             "air_speed_ms": 0, "oil": "ISO VG 68"},
))


def _run_slab(p):
    n_t = max(2, round(p["thickness_mm"] / p["element_size_mm"]))
    h = p["thickness_mm"] / n_t
    m = VoxelModel(np.ones((n_t, 3, 3), bool), h)
    th = ThermalModel(m, p["k_W_mK"])
    left, right = th.normals[:, 0] < -0.5, th.normals[:, 0] > 0.5
    G1, G2 = th.node_conductance(left, p["h_hot"]), th.node_conductance(right, p["h_cold"])
    T, _ = th.solve_linear(G1 + G2, G1 * p["T_hot_c"] + G2 * p["T_cold_c"])
    area_m2 = 9 * h * h * 1e-6
    return {"outputs": {"heat_flux_W_m2": float((G2 * (T - p["T_cold_c"])).sum() / area_m2),
                        "hot_surface_temp_c": float(T[m.nodes[:, 0] < 1e-9].mean())}}


register(SolverSpec(
    name="thermal_slab", version="0.1.0", domain="thermal", fidelity="L2",
    summary="Plane wall between two fluids (engine verification of conduction + convection).",
    method="Same conduction/convection engine as thermal_housing_3d",
    schema={"thickness_mm": {"type": "number", "exclusive_min": 0, "required": True},
            "element_size_mm": {"type": "number", "exclusive_min": 0, "default": 1.0},
            "k_W_mK": {"type": "number", "exclusive_min": 0, "required": True},
            "h_hot": {"type": "number", "exclusive_min": 0, "required": True},
            "h_cold": {"type": "number", "exclusive_min": 0, "required": True},
            "T_hot_c": {"type": "number", "required": True},
            "T_cold_c": {"type": "number", "required": True}},
    run=_run_slab,
    benchmarks=[Benchmark("plane-wall-resistance", "Incropera, Fundamentals of Heat and Mass Transfer, eq. 3.19: "
                          "q = ΔT / (1/h₁ + L/k + 1/h₂) (exact)",
                          {"thickness_mm": 10, "k_W_mK": 50, "h_hot": 500, "h_cold": 20, "T_hot_c": 100, "T_cold_c": 20},
                          {"outputs.heat_flux_W_m2": 80 / (1 / 500 + 0.01 / 50 + 1 / 20),
                           "outputs.hot_surface_temp_c": 100 - 80 / (1 / 500 + 0.01 / 50 + 1 / 20) / 500}, 0.001)],
    limitations=["1D slab only"],
    example={"thickness_mm": 10, "k_W_mK": 50, "h_hot": 500, "h_cold": 20, "T_hot_c": 100, "T_cold_c": 20},
))


# ------------------------------------------------------------------ review
CHECKS = ["interference", "fits", "dfm", "assembly"]


def _run_review(p, progress):
    try:
        mat = _material(p)
        L = _layout(p)
    except ValueError as ex:
        return {"ok": False, "errors": [str(ex)]}
    progress("Solving", 0.3)
    t0 = time.perf_counter()
    bad = [c for c in p["checks"] if c not in CHECKS]
    if bad:
        return {"ok": False, "errors": [f"unknown checks {bad}; choose from {CHECKS}"]}
    findings = review(p["design"], p["kinematics"], L, mat, p["checks"], p.get("parts"), p.get("input_rpm"))
    progress("Post-processing", 1.0)
    counts = {s: sum(f["severity"] == s for f in findings) for s in ("critical", "warning", "info")}
    names = {pt.get("name") for pt in (p.get("parts") or []) if isinstance(pt, dict)}
    warnings = [FRAME_NOTE, "Rule-based review on the parametric layout of /generate-assembly "
                "(not a mesh clash detection)."]
    if not names:
        warnings.append("No parts list received: findings use generic part names.")
    return {"outputs": {"findings": findings, "counts": counts, "checks_run": p["checks"],
                        "checked_parts": len(names) or None, "solve_time_s": round(time.perf_counter() - t0, 3)},
            "checks": [Check("critical findings", counts["critical"], 0, "-", "max")],
            "warnings": warnings}


register(SolverSpec(
    name="review_gearbox", version="0.1.0", domain="design_review", fidelity="L0", progress=True,
    summary="Pre-manufacturing design review of a gearbox variant: interference, fits, DFM and assembly, each finding with a fix and a 3D location.",
    method="Rule base: AGMA/ISO gear proportions, SKF bearing fit practice, casting/machining DFM guidelines, assembly sequence logic",
    schema={**VARIANT_SCHEMA,
            "analysis": {"type": "string", "enum": ["review"], "default": "review"},
            "parts": {"type": "array", "default": [], "max_items": 5000},
            "input_rpm": {"type": "number", "exclusive_min": 0, "max": 1e5},
            "checks": {"type": "array", "items_type": "string", "default": CHECKS}},
    run=_run_review,
    benchmarks=[Benchmark("regression-review-2stage", "ACE regression v0.2: real variant must flag the closed-housing assembly problem",
                          {**EXAMPLE_VARIANT}, {"outputs.counts.critical": 1}, 0.0)],
    limitations=["rules on the parametric layout, not a geometric clash detection on the STL parts yet",
                 "part names matched heuristically from the parts list"],
    example={**EXAMPLE_VARIANT, "checks": CHECKS},
))
