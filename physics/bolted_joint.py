"""Bolted joint (tension + shear) for the housing covers — Shigley 10e ch. 8 with VDI 2230-style checks.

Units: mm, N, MPa. Screening level: static loads (spur gears give no axial force; the bearing reactions
have a fixed direction), metric coarse socket head cap screws ISO 4762 into tapped holes.

Sources
- Tensile stress area A_t = π/4 (d − 0.9382 P)²  (ISO 898-1; Shigley eq. 8-1 / Table 8-1).
- Proof stress S_p by property class (ISO 898-1: 8.8 → 580, 10.9 → 830, 12.9 → 970 MPa, d ≤ 16 mm).
- Bolt stiffness k_b = A_t E / l_t (fully threaded short screw in the grip; Shigley eq. 8-17).
- Member stiffness: 30° pressure frusta in series, each segment
      k = π E d tanα / ln[ (2 t tanα + D − d)(D + d) / ((2 t tanα + D + d)(D − d)) ]   (Shigley eq. 8-20),
  tapped hole: effective grip l' = h + d/2 (Shigley Table 8-7), washer face d_w = 1.5 d.
  Cross-checked against Wileman's FE fit k_m = E d A exp(B d / l) (Shigley eq. 8-23, Table 8-8).
- Joint constant C = k_b / (k_b + k_m); bolt load F_b = F_i + C P (Shigley eq. 8-24).
- Preload F_i = 0.75 F_p for a reusable joint (Shigley eq. 8-31); torque-wrench scatter ± 25 %.
- Checks (Shigley eq. 8-28…8-30): yield n_p = S_p A_t / (F_i,max + C P); load factor
  n_L = (S_p A_t − F_i,max) / (C P); separation n_0 = F_i,min / (P (1 − C)); slip
  n_slip = μ Σ(F_i,min − (1 − C) P_i) / V; dowel shear; thread stripping in the tapped housing
  (engagement 2 d, stripping area ≈ 0.5 π d L_e — guideline); head pressure on the cover.
"""
import math

TAN30 = math.tan(math.radians(30.0))

# d [mm], coarse pitch P [mm], socket head diameter d_k (ISO 4762), clearance hole medium d_h (ISO 273)
METRIC = {
    "M3": (3.0, 0.5, 5.5, 3.4), "M4": (4.0, 0.7, 7.0, 4.5), "M5": (5.0, 0.8, 8.5, 5.5),
    "M6": (6.0, 1.0, 10.0, 6.6), "M8": (8.0, 1.25, 13.0, 9.0), "M10": (10.0, 1.5, 16.0, 11.0),
    "M12": (12.0, 1.75, 18.0, 13.5),
}
# property class: proof S_p, yield S_y, ultimate S_u [MPa] (ISO 898-1, d ≤ 16 mm)
CLASSES = {"8.8": (580.0, 640.0, 800.0), "10.9": (830.0, 940.0, 1040.0), "12.9": (970.0, 1100.0, 1220.0)}
E_STEEL = 207000.0
WILEMAN = {"steel": (0.78715, 0.62873), "aluminum": (0.79670, 0.63816)}


def tensile_area(d, P):
    return math.pi / 4.0 * (d - 0.9382 * P) ** 2


def proof_load(size, cls="8.8"):
    d, P, _, _ = METRIC[size]
    return CLASSES[cls][0] * tensile_area(d, P)


def frustum_segment(E, d, D, t, alpha_tan=TAN30):
    """Stiffness of one frustum segment: narrow-end diameter D, thickness t (Shigley eq. 8-20)."""
    if t <= 0:
        return math.inf
    num = (2 * t * alpha_tan + D - d) * (D + d)
    den = (2 * t * alpha_tan + D + d) * (D - d)
    return math.pi * E * d * alpha_tan / math.log(num / den)


def member_stiffness(d, layers, dw=None):
    """layers = [(thickness, E), ...] from the head side down; two symmetric frusta over the grip."""
    dw = dw or 1.5 * d
    L = sum(t for t, _ in layers)
    half = L / 2.0
    inv = 0.0
    z = 0.0
    for t, E in layers:
        z0, z1 = z, z + t
        # top frustum covers [0, half] (narrow at the head), bottom frustum [half, L] (narrow at the end)
        for a, b, top in ((max(z0, 0.0), min(z1, half), True), (max(z0, half), min(z1, L), False)):
            if b - a <= 1e-12:
                continue
            s0 = a if top else L - b             # distance of the segment's narrow side from the frustum's narrow end
            D = dw + 2 * s0 * TAN30
            inv += 1.0 / frustum_segment(E, d, D, b - a)
        z = z1
    return 1.0 / inv


def wileman_stiffness(d, l, E, material="steel"):
    A, B = WILEMAN[material]
    return E * d * A * math.exp(B * d / l)


def joint_check(size, cls, h_cover, E_cover, E_body, bolt_xy, shear_v, moment, axial=0.0, mu=0.15,
                n_dowels=2, d_dowel=None, Sy_dowel=600.0, Sy_body=None, Su_body=None, Sy_cover=None,
                scatter=0.25, targets=None):
    """Check one bolted cover. bolt_xy: [(x, y)] bolt positions; shear_v: in-plane force [N] (magnitude);
    moment: (Mx, My) [N·mm] about axes in the joint plane through the bolt-pattern centroid; axial: force
    pulling the cover off [N]. Returns a dict with per-check values and pass flags."""
    t = targets or {"yield": 1.0, "load_factor": 2.0, "separation": 2.0, "slip": 1.5, "dowel": 2.0,
                    "stripping": 2.0, "head_pressure": 1.0}
    d, P, dk, dh = METRIC[size]
    Sp, Sy, Su = CLASSES[cls]
    At = tensile_area(d, P)
    Fp = Sp * At
    Fi = 0.75 * Fp
    Fi_max, Fi_min = Fi * (1 + scatter), Fi * (1 - scatter)
    l_eff = h_cover + d / 2.0                                     # tapped hole (Shigley Table 8-7)
    kb = At * E_STEEL / l_eff
    km = member_stiffness(d, [(h_cover, E_cover), (d / 2.0, E_body)])
    C = kb / (kb + km)
    n = len(bolt_xy)
    cx = sum(x for x, _ in bolt_xy) / n
    cy = sum(y for _, y in bolt_xy) / n
    Ixx = sum((y - cy) ** 2 for _, y in bolt_xy) or 1e-9
    Iyy = sum((x - cx) ** 2 for x, _ in bolt_xy) or 1e-9
    Mx, My = moment
    Pi = [axial / n + abs(Mx) * abs(y - cy) / Ixx + abs(My) * abs(x - cx) / Iyy for x, y in bolt_xy]
    Pmax = max(Pi)
    out = {"size": size, "class": cls, "count": n, "tensile_area_mm2": round(At, 2),
           "proof_load_n": round(Fp, 0), "preload_n": round(Fi, 0), "preload_min_n": round(Fi_min, 0),
           "preload_max_n": round(Fi_max, 0), "bolt_stiffness_n_mm": round(kb, 0),
           "member_stiffness_n_mm": round(km, 0), "joint_constant": round(C, 4),
           "external_tension_max_n": round(Pmax, 1), "shear_n": round(shear_v, 1),
           "tightening_torque_nm": round(0.2 * Fi * d / 1000.0, 2), "friction": mu}
    checks = {}
    checks["yield"] = Sp * At / (Fi_max + C * Pmax)
    checks["load_factor"] = (Sp * At - Fi_max) / (C * Pmax) if Pmax > 1e-9 else math.inf
    checks["separation"] = Fi_min / (Pmax * (1 - C)) if Pmax > 1e-9 else math.inf
    clamp = sum(max(0.0, Fi_min - (1 - C) * p) for p in Pi)
    checks["slip"] = mu * clamp / shear_v if shear_v > 1e-9 else math.inf
    dd = d_dowel or d
    tau = shear_v / (n_dowels * math.pi / 4.0 * dd ** 2) if n_dowels else math.inf
    checks["dowel"] = 0.577 * Sy_dowel / tau if shear_v > 1e-9 and n_dowels else math.inf
    if Su_body:
        Le = 2.0 * d
        tau_s = (Fi_max + C * Pmax) / (0.5 * math.pi * d * Le)
        checks["stripping"] = 0.6 * Su_body / tau_s
        out["thread_engagement_mm"] = Le
    if Sy_cover:
        p_head = Fi_max / (math.pi / 4.0 * (dk ** 2 - dh ** 2))
        checks["head_pressure"] = Sy_cover / p_head
        out["head_pressure_mpa"] = round(p_head, 1)
    out["safety_factors"] = {k: (round(float(v), 3) if math.isfinite(v) else None) for k, v in checks.items()}
    out["targets"] = t
    out["pass"] = all(v >= t[k] - 1e-9 for k, v in checks.items())
    out["governing"] = min(checks, key=lambda k: checks[k] / t[k])
    return out


def select_cover_bolts(h_cover, E_cover, E_body, positions_for, shear_v, moment, axial=0.0,
                       sizes=("M4", "M5", "M6", "M8"), classes=("8.8", "10.9", "12.9"), **kw):
    """Smallest screw (then lowest class) whose joint passes every check. positions_for(size) returns the
    bolt positions that fit for that size (column size grows with the screw). None if nothing passes."""
    tried = []
    for cls in classes:
        for size in sizes:
            pos = positions_for(size)
            if len(pos) < 3:
                tried.append({"size": size, "class": cls, "pass": False, "reason": "fewer than 3 bolt positions fit"})
                continue
            r = joint_check(size, cls, h_cover, E_cover, E_body, pos, shear_v, moment, axial, **kw)
            tried.append({"size": size, "class": cls, "pass": r["pass"], "governing": r["governing"]})
            if r["pass"]:
                r["positions"] = [list(map(lambda v: round(v, 2), p)) for p in pos]
                r["tried"] = tried
                return r
    return {"pass": False, "tried": tried}
