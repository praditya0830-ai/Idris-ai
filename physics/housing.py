"""Gearbox housing model for 3D FEA: geometry (voxels), bearing loads and fixtures.

Coordinate frame (MUST match /generate-assembly; see FRAME notes in fea_api.py):
  shafts run along +z, input shaft axis at x=0, y=0; stage-1 mesh at the front (low z).
  housing_params may override any dimension and add frame_offset_mm [dx, dy, dz].
"""
import math
import numpy as np
from .fem3d import VoxelModel
from .bearings import DEEP_GROOVE, STANDARD_BORES

HOUSING_MATERIALS = {
    # name: (E MPa, nu, yield MPa, density kg/m3, note)
    "Al7075-T6": (71700, 0.33, 503, 2810, ""),
    "Al6061-T6": (68900, 0.33, 276, 2700, ""),
    "Al2024-T3": (73100, 0.33, 345, 2780, ""),
    "A356-T6": (72400, 0.33, 165, 2670, "cast aluminium"),
    "Ti-6Al-4V": (113800, 0.342, 880, 4430, ""),
    "Steel-1045": (205000, 0.29, 530, 7850, ""),
    "Steel-4140": (205000, 0.29, 655, 7850, ""),
    "CastIron-GG25": (110000, 0.26, 165, 7200, "brittle: compare max principal stress, not von Mises"),
    "Mg-AZ91D": (45000, 0.35, 160, 1810, ""),
}


def find_material(name):
    if not isinstance(name, str):
        return None
    key = name.replace(" ", "").lower()
    for k in HOUSING_MATERIALS:
        if k.replace(" ", "").lower() == key:
            return k
    return None


def _bearing_for_bore(d):
    """Medium-series (62xx) deep-groove bearing for a bore >= d; returns (bore, OD, width)."""
    for b in STANDARD_BORES:
        if b >= d - 1e-9:
            for des, (bd, D, B, C, C0) in DEEP_GROOVE.items():
                if bd == b and des.startswith("62"):
                    return b, D, B
    return d, 1.8 * d, 0.4 * d          # beyond catalog: proportional estimate


def _shaft_dia_for_torque(T_Nm, tau_allow=40.0):
    return (16 * abs(T_Nm) * 1e3 / (math.pi * tau_allow)) ** (1 / 3)


def _num(d, k, default):
    v = d.get(k) if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else default


def build_layout(design, kinematics, loads, shaft, hp):
    """Shaft positions, gear forces and housing dimensions from a /generate-variants variant."""
    m = float(design["module_mm"])
    Np, Ng = int(design["pinion_teeth"]), int(design["gear_teeth"])
    F = float(design["face_width_mm"])
    stages = int(design.get("stages", 1))
    if stages not in (1, 2):
        raise ValueError("only 1- or 2-stage housings are supported")
    a = float(design.get("center_distance_mm") or m * (Np + Ng) / 2)
    eff = float(kinematics.get("efficiency", 0.97))
    T_in = float(kinematics["input_torque_nm"])
    i_stage = Ng / Np
    eff_stage = eff ** (1 / stages)
    dp, dg = m * Np, m * Ng
    dpo, dgo = dp + 2 * m, dg + 2 * m

    layout = (hp.get("layout") if isinstance(hp, dict) else None) or "inline"
    xs = [0.0, a] if stages == 1 else ([0.0, a, 2 * a] if layout == "inline" else [0.0, a, 0.0])
    if stages == 2 and layout != "inline":
        raise ValueError("only inline shaft layout supported for 2 stages (reverted: roadmap)")

    clr = _num(hp, "clearance_mm", max(3.0, 1.5 * m))
    t = _num(hp, "wall_thickness_mm", max(4.0, round(0.06 * dgo + 1)))
    gap = _num(hp, "axial_gap_mm", max(4.0, 0.25 * F))
    depth_in = stages * F + (stages + 1) * gap
    z_mesh = [gap + F / 2 + s * (F + gap) for s in range(stages)]   # interior z of each mesh

    # torques per shaft and shaft diameters
    T = [T_in]
    for s in range(stages):
        T.append(T[-1] * i_stage * eff_stage)
    d_out = _num(shaft, "output_diameter_mm", _shaft_dia_for_torque(T[-1]))
    d_sh = [max(10.0, _shaft_dia_for_torque(Ti)) for Ti in T[:-1]] + [d_out]
    brg = [_bearing_for_bore(d) for d in d_sh]

    # gear forces on shafts (input turns CCW about +z, pinions drive)
    forces = [[] for _ in xs]                      # per shaft: (z, Fx, Fy)
    for s in range(stages):
        p_idx, g_idx = s, s + 1
        e = np.array([xs[g_idx] - xs[p_idx], 0.0])
        e /= np.linalg.norm(e)
        tdir = np.array([-e[1], e[0]])             # z x e
        sign = 1 if s % 2 == 0 else -1             # each stage reverses rotation
        Wt = 2e3 * T[s] / dp
        Wr = Wt * math.tan(math.radians(20))
        f_gear = Wr * e + sign * Wt * tdir
        forces[g_idx].append((z_mesh[s], *f_gear))
        forces[p_idx].append((z_mesh[s], *(-f_gear)))

    half_y = dgo / 2 + clr
    x_min = min(xs[0] - dpo / 2, xs[1] - dgo / 2) - clr
    x_max = max(x + dgo / 2 for x in xs[1:]) + clr
    W_in = _num(hp, "inner_width_mm", x_max - x_min)
    H_in = _num(hp, "inner_height_mm", 2 * half_y)
    D_in = _num(hp, "inner_depth_mm", depth_in)
    x0 = x_min - (W_in - (x_max - x_min)) / 2
    return {
        "module_mm": m, "stages": stages, "center_distance_mm": a, "shaft_x_mm": xs,
        "shaft_torque_Nm": T, "shaft_dia_mm": d_sh, "bearings": brg, "forces": forces,
        "wall_mm": t, "boss_mm": _num(hp, "boss_height_mm", t),
        "inner": {"x0": x0, "x1": x0 + W_in, "y0": -H_in / 2, "y1": H_in / 2, "z0": 0.0, "z1": D_in},
        "offset": np.array(hp.get("frame_offset_mm", [0, 0, 0]) if isinstance(hp, dict) else [0, 0, 0], float),
    }


def voxelize(L, h):
    """Occupancy grid of the housing: closed box, bearing bores in front/back walls, bosses."""
    t, b, I = L["wall_mm"], L["boss_mm"], L["inner"]
    lo = np.array([I["x0"] - t, I["y0"] - t, I["z0"] - t - b])
    hi = np.array([I["x1"] + t, I["y1"] + t, I["z1"] + t + b])
    n = np.maximum(1, np.round((hi - lo) / h).astype(int))
    c = [lo[k] + h * (np.arange(n[k]) + 0.5) for k in range(3)]
    X, Y, Z = np.meshgrid(*c, indexing="ij")
    in_outer = ((X > I["x0"] - t) & (X < I["x1"] + t) & (Y > I["y0"] - t) & (Y < I["y1"] + t)
                & (Z > I["z0"] - t) & (Z < I["z1"] + t))
    in_cavity = ((X > I["x0"]) & (X < I["x1"]) & (Y > I["y0"]) & (Y < I["y1"])
                 & (Z > I["z0"]) & (Z < I["z1"]))
    solid = in_outer & ~in_cavity
    for x, (bore, OD, Bw) in zip(L["shaft_x_mm"], L["bearings"]):
        r2 = (X - x) ** 2 + Y**2
        boss = (r2 < (OD / 2 + t) ** 2) & (((Z > I["z0"] - t - b) & (Z <= I["z0"] - t))
                                           | ((Z >= I["z1"] + t) & (Z < I["z1"] + t + b)))
        solid |= boss
        solid &= ~(r2 < (OD / 2) ** 2)             # through-bore for the bearing outer ring
    return solid, lo


def build_model(L, h):
    solid, lo = voxelize(L, h)
    model = VoxelModel(solid, h, origin=lo)
    return model


def bearing_loads(model, L):
    """Distribute each shaft's bearing reactions over its two bore surfaces (cosine law)."""
    h, I, t, b = model.h, L["inner"], L["wall_mm"], L["boss_mm"]
    P = model.nodes
    loads = np.zeros((model.n_nodes, 3))
    applied = []
    zf, zb = I["z0"] - t / 2, I["z1"] + t / 2     # bearing planes (mid-wall)
    walls = {"front": (I["z0"] - t - b - 1e-6, I["z0"] + 1e-6),
             "back": (I["z1"] - 1e-6, I["z1"] + t + b + 1e-6)}
    names = ["Input", "Intermediate", "Output"] if L["stages"] == 2 else ["Input", "Output"]
    for si, (x, (bore, OD, Bw), flist) in enumerate(zip(L["shaft_x_mm"], L["bearings"], L["forces"])):
        R = {"front": np.zeros(2), "back": np.zeros(2)}
        for z, fx, fy in flist:
            zz = I["z0"] + z
            wb = (zz - zf) / (zb - zf)
            R["front"] += (1 - wb) * np.array([fx, fy])
            R["back"] += wb * np.array([fx, fy])
        for wall, (za, zb_) in walls.items():
            Fw = R[wall]
            mag = np.linalg.norm(Fw)
            if mag < 1e-9:
                continue
            rv = P[:, :2] - np.array([x, 0.0])
            r = np.linalg.norm(rv, axis=1)
            sel = (r < OD / 2 + 0.75 * h) & (P[:, 2] > za) & (P[:, 2] < zb_)
            idx = np.where(sel)[0]
            if len(idx) == 0:
                raise ValueError("could not find bearing bore nodes: mesh too coarse for this bore")
            cosang = (rv[idx] @ (Fw / mag)) / np.maximum(r[idx], 1e-9)
            w = np.clip(cosang, 0, None)
            if w.sum() == 0:
                w = np.ones(len(idx))
            loads[idx, :2] += np.outer(w / w.sum(), Fw)
            applied.append({"name": f"{names[si]} bearing ({wall})",
                            "force_n": [float(Fw[0]), float(Fw[1]), 0.0],
                            "bore_mm": bore, "bearing_od_mm": OD})
    return loads, applied


def fixture_nodes(model, L, fixture):
    P = model.nodes
    h = model.h
    if fixture == "base":
        return np.where(P[:, 1] < P[:, 1].min() + 1e-6)[0]
    if fixture == "flange":
        x_out = L["shaft_x_mm"][-1]
        OD = L["bearings"][-1][1]
        r = np.hypot(P[:, 0] - x_out, P[:, 1])
        zmax = P[:, 2].max()
        sel = (P[:, 2] > zmax - 1e-6) & (r > OD / 2 - 0.75 * h)
        return np.where(sel)[0]
    raise ValueError("fixture must be 'base' or 'flange'")


def wetted_face_mask(thermal_model, L):
    """True for faces that touch oil: cavity walls and bearing bores; False = outside air."""
    c, n = thermal_model.centres, thermal_model.normals
    h = thermal_model.m.h
    probe = c + 0.5 * h * n                       # just outside the face, in the fluid
    I = L["inner"]
    in_cav = ((probe[:, 0] > I["x0"]) & (probe[:, 0] < I["x1"]) & (probe[:, 1] > I["y0"])
              & (probe[:, 1] < I["y1"]) & (probe[:, 2] > I["z0"]) & (probe[:, 2] < I["z1"]))
    in_bore = np.zeros(len(c), bool)
    for x, (bore, OD, Bw) in zip(L["shaft_x_mm"], L["bearings"]):
        in_bore |= np.hypot(probe[:, 0] - x, probe[:, 1]) < OD / 2
    return in_cav | in_bore
