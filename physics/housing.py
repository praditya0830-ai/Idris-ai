"""Gearbox housing model for 3D analyses — SAME geometry and frame as assembly3d.py.

Frame (identical to /generate-assembly STL): housing occupies x∈[0,W], y∈[0,H], z∈[0,D] mm,
shafts parallel to +z at y = H/2. 1 stage: input axis A, output axis B. 2 stages ("reverted"):
input and output coaxial on axis A, intermediate shaft on axis B. Bores: front wall at axis A
(Ø input+2), back wall at the output axis (Ø output+2), exactly as assembly3d.build_housing.
"""
import math
import numpy as np
from .fem3d import VoxelModel
from .gears import ALPHA, CLEARANCE, ETA_STAGE, STAGE_GAP, shaft_diameter_mm

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
    """Accepts 'Al7075-T6', 'Steel4140', 'steel 4140', ... -> canonical key or None."""
    if not isinstance(name, str):
        return None
    norm = lambda s: s.replace(" ", "").replace("-", "").replace("_", "").lower()
    for k in HOUSING_MATERIALS:
        if norm(k) == norm(name):
            return k
    return None


def _num(d, k, default=None):
    v = d.get(k) if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else default


def build_layout(design, kinematics, loads, shaft, hp):
    """Positions, shaft sizes, gear forces and housing box — mirrors assembly3d.build_assembly."""
    m = float(design["module_mm"])
    zp, zg = int(design["pinion_teeth"]), int(design["gear_teeth"])
    b = float(design["face_width_mm"])
    stages = int(design.get("stages", 1))
    if stages not in (1, 2):
        raise ValueError("only 1- or 2-stage gearboxes are supported")
    a = m * (zp + zg) / 2.0
    dp, dg = m * zp, m * zg
    dap, dag = m * (zp + 2), m * (zg + 2)
    u = zg / zp
    T_in = float(kinematics["input_torque_nm"])
    d_in = shaft_diameter_mm(T_in)
    d_out = _num(shaft, "output_diameter_mm") or shaft_diameter_mm(T_in * u ** stages * ETA_STAGE ** stages)
    d_mid = shaft_diameter_mm(T_in * u * ETA_STAGE) if stages == 2 else None

    t = _num(hp, "wall_thickness", 4.0)
    if stages == 1:
        x_a = t + CLEARANCE + dap / 2.0
        x_b = x_a + a
        x_right = x_b + dag / 2.0 + CLEARANCE + t
        d_need = b + 2 * CLEARANCE + 2 * t
    else:
        x_a = t + CLEARANCE + dag / 2.0
        x_b = x_a + a
        x_right = x_b + dag / 2.0 + CLEARANCE + t
        d_need = 2 * b + STAGE_GAP + 2 * CLEARANCE + 2 * t
    h_need = max(dap, dag) + 2 * CLEARANCE + 2 * t
    W = _num(hp, "envelope_width", math.ceil(x_right))
    H = _num(hp, "envelope_height", math.ceil(h_need))
    D = _num(hp, "envelope_depth", math.ceil(d_need))
    if x_right > W + 1.0 or d_need > D + 1.0 or h_need > H + 1.0:
        raise ValueError("housing envelope is too small for this gear layout")
    shift = (W - x_right) / 2.0
    x_a += shift
    x_b += shift
    yc = H / 2.0
    z1 = t + CLEARANCE
    z2 = z1 + b + STAGE_GAP

    # shafts (z extents as in assembly3d)
    if stages == 1:
        shafts = [{"name": "Input shaft", "x": x_a, "d": d_in, "z0": -22.0, "z1": z1 + b, "torque": T_in,
                   "rpm_factor": 1.0},
                  {"name": "Output shaft", "x": x_b, "d": d_out, "z0": z1, "z1": D + 22.0,
                   "torque": T_in * u * ETA_STAGE, "rpm_factor": 1 / u}]
        gears = [{"name": "Pinion", "shaft": 0, "x": x_a, "z0": z1, "d": dp, "tip": dap, "teeth": zp, "stage": 0},
                 {"name": "Gear", "shaft": 1, "x": x_b, "z0": z1, "d": dg, "tip": dag, "teeth": zg, "stage": 0}]
        bores = [{"wall": "front", "x": x_a, "d": d_in + 2, "shaft": 0},
                 {"wall": "back", "x": x_b, "d": d_out + 2, "shaft": 1}]
    else:
        shafts = [{"name": "Input shaft", "x": x_a, "d": d_in, "z0": -22.0, "z1": z1 + b, "torque": T_in,
                   "rpm_factor": 1.0},
                  {"name": "Intermediate shaft", "x": x_b, "d": d_mid, "z0": 0.4 * t, "z1": D - 0.4 * t,
                   "torque": T_in * u * ETA_STAGE, "rpm_factor": 1 / u},
                  {"name": "Output shaft", "x": x_a, "d": d_out, "z0": z2, "z1": D + 22.0,
                   "torque": T_in * u * u * ETA_STAGE ** 2, "rpm_factor": 1 / u ** 2}]
        gears = [{"name": "Pinion (stage 1)", "shaft": 0, "x": x_a, "z0": z1, "d": dp, "tip": dap, "teeth": zp, "stage": 0},
                 {"name": "Gear (stage 1)", "shaft": 1, "x": x_b, "z0": z1, "d": dg, "tip": dag, "teeth": zg, "stage": 0},
                 {"name": "Pinion (stage 2)", "shaft": 1, "x": x_b, "z0": z2, "d": dp, "tip": dap, "teeth": zp, "stage": 1},
                 {"name": "Gear (stage 2)", "shaft": 2, "x": x_a, "z0": z2, "d": dg, "tip": dag, "teeth": zg, "stage": 1}]
        bores = [{"wall": "front", "x": x_a, "d": d_in + 2, "shaft": 0},
                 {"wall": "back", "x": x_a, "d": d_out + 2, "shaft": 2}]

    # gear mesh forces (on each gear, from its mate). Input turns CCW about +z.
    meshes = []
    for s in range(stages):
        pin = gears[2 * s]
        gear = gears[2 * s + 1]
        T_p = T_in * (u * ETA_STAGE) ** s
        Wt = 2000.0 * T_p / dp
        Wr = Wt * math.tan(ALPHA)
        e = np.array([gear["x"] - pin["x"], 0.0])
        e /= np.linalg.norm(e)
        tdir = np.array([-e[1], e[0]])
        sign = 1.0 if s % 2 == 0 else -1.0
        f_on_gear = Wr * e + sign * Wt * tdir
        zc = pin["z0"] + b / 2
        meshes.append({"stage": s, "z": zc, "Wt": Wt, "Wr": Wr, "pinion": 2 * s, "gear": 2 * s + 1,
                       "f_on_gear": f_on_gear, "f_on_pinion": -f_on_gear})
    return {"stages": stages, "module_mm": m, "face_width_mm": b, "teeth": (zp, zg),
            "center_distance_mm": a, "W": W, "H": H, "D": D, "wall_mm": t, "yc": yc, "z1": z1, "z2": z2,
            "x_a": x_a, "x_b": x_b, "shafts": shafts, "gears": gears, "bores": bores, "meshes": meshes,
            "input_torque_Nm": T_in,
            "inner": {"x0": t, "x1": W - t, "y0": t, "y1": H - t, "z0": t, "z1": D - t},
            "offset": np.zeros(3)}


MESH_LEVELS = {            # (layers through the wall, in-plane element size in wall thicknesses, min mm)
    "coarse": (1, 2.0, 6.0),
    "standard": (2, 1.0, 4.0),
    "fine": (3, 0.67, 3.0),
}


def _axis_planes(L_len, t, layers, s, extra=()):
    """Node planes along one axis: `layers` cells through each wall, ~s mm cells inside,
    plus forced planes (bore edges) so holes come out at their true size."""
    wall = list(np.linspace(0, t, layers + 1))
    n_in = max(1, int(round((L_len - 2 * t) / s)))
    inner = list(np.linspace(t, L_len - t, n_in + 1))
    pl = sorted(set(np.round(wall + inner + [L_len - t + w for w in np.linspace(0, t, layers + 1)], 6)))
    forced = sorted(e for e in extra if t < e < L_len - t)
    for e in forced:
        pl.append(e)
    pl = np.array(sorted(set(np.round(pl, 6))))
    # merge planes closer than 25 % of s (keep wall and forced planes)
    keep = set(np.round(wall, 6)) | {round(L_len - t + w, 6) for w in np.linspace(0, t, layers + 1)} | set(np.round(forced, 6))
    out = [pl[0]]
    for x in pl[1:]:
        if x - out[-1] < 0.25 * s and round(x, 6) not in keep and round(out[-1], 6) not in keep:
            continue
        if x - out[-1] < 1e-6:
            continue
        out.append(x)
    return np.array(out)


def build_model(L, mesh="coarse"):
    """Rectilinear brick model of the assembly housing: exact wall thickness, coarser in-plane."""
    if isinstance(mesh, (int, float)):           # legacy: uniform element size
        mesh = "coarse"
    layers, s_rel, s_min = MESH_LEVELS[mesh]
    W, H, D, t, yc = L["W"], L["H"], L["D"], L["wall_mm"], L["yc"]
    s = max(s_rel * t, s_min)
    xe, ye = [], [yc]
    for bo in L["bores"]:
        r = bo["d"] / 2
        xe += [bo["x"] - r, bo["x"] + r, bo["x"]]
        ye += [yc - r, yc + r]
    for sh in L["shafts"]:                       # blind seats: give load patches their own planes
        xe += [sh["x"] - sh["d"] / 2, sh["x"] + sh["d"] / 2, sh["x"]]
    # forced planes at bore edges would create many different brick sizes (slow); holes are
    # stair-stepped on the regular grid instead
    xs = _axis_planes(W, t, layers, s)
    ys = _axis_planes(H, t, layers, s)
    zs = _axis_planes(D, t, layers, s)
    cx, cy, cz = (0.5 * (pl[1:] + pl[:-1]) for pl in (xs, ys, zs))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    cavity = (X > t) & (X < W - t) & (Y > t) & (Y < H - t) & (Z > t) & (Z < D - t)
    solid = ~cavity
    for bo in L["bores"]:
        r2 = (X - bo["x"]) ** 2 + (Y - yc) ** 2
        in_wall = (Z < t) if bo["wall"] == "front" else (Z > D - t)
        solid &= ~(in_wall & (r2 < (bo["d"] / 2) ** 2))
    return VoxelModel(solid, planes=(xs, ys, zs))


def wall_thickness_in_model(L, h):
    """Through-wall thickness the voxel model actually represents (≈ multiples of h)."""
    t = L["wall_mm"]
    return max(1, round(t / h)) * h


def shaft_support_loads(L):
    """Reactions each shaft puts on the housing walls (front wall z=t/2, back wall z=D−t/2).

    Lever rule along each shaft between the two wall planes. Coaxial input/output shafts of a
    2-stage reverted box are both carried by the walls at axis A (pilot bearing between them
    transfers no net force to the housing).
    """
    t, D = L["wall_mm"], L["D"]
    zf, zb = t / 2, D - t / 2
    per_axis = {}
    for me in L["meshes"]:
        for role, f in (("pinion", me["f_on_pinion"]), ("gear", me["f_on_gear"])):
            g = L["gears"][me[role]]
            x = round(g["x"], 6)
            wb = (me["z"] - zf) / (zb - zf)
            acc = per_axis.setdefault(x, {"front": np.zeros(2), "back": np.zeros(2)})
            acc["front"] += (1 - wb) * f
            acc["back"] += wb * f
    return per_axis


def shaft_supports_z(L, si):
    """Axial positions of the two bearings assumed for shaft si (z, mm)."""
    t, D = L["wall_mm"], L["D"]
    front, back = t / 2, D - t / 2
    if L["stages"] == 2 and si in (0, 2):
        pilot = (L["z1"] + L["face_width_mm"] + L["z2"]) / 2    # pilot bearing between coaxial shafts
        return (front, pilot) if si == 0 else (pilot, back)
    return front, back


def per_shaft_reactions(L):
    """Radial reactions [N, (Fx, Fy)] at the two bearings of every shaft (lever rule)."""
    out = []
    for si, s in enumerate(L["shafts"]):
        za, zb = shaft_supports_z(L, si)
        Ra, Rb = np.zeros(2), np.zeros(2)
        for me in L["meshes"]:
            for role, f in (("pinion", me["f_on_pinion"]), ("gear", me["f_on_gear"])):
                if L["gears"][me[role]]["shaft"] != si:
                    continue
                w = (me["z"] - za) / (zb - za)
                Ra += (1 - w) * f
                Rb += w * f
        out.append((Ra, Rb))
    return out


def bearing_loads(model, L):
    """Nodal loads on the housing: each axis' wall reaction spread over the bore edge (cosine law),
    or over a Ø(shaft+2h) patch of the inner wall face where assembly3d has no bore (blind seat)."""
    h = model.h
    P = model.nodes
    t, D, yc = L["wall_mm"], L["D"], L["yc"]
    loads = np.zeros((model.n_nodes, 3))
    applied = []
    zs = np.unique(np.round(P[:, 2], 6))
    z_in_front = zs[np.argmin(np.abs(zs - t))]           # node plane of the inner face of each wall
    z_in_back = zs[np.argmin(np.abs(zs - (D - t)))]
    bore_at = {(bo["wall"], round(bo["x"], 6)): bo for bo in L["bores"]}
    for x, walls in shaft_support_loads(L).items():
        names = [s["name"] for s in L["shafts"] if abs(s["x"] - x) < 1e-6]
        dmax = max(s["d"] for s in L["shafts"] if abs(s["x"] - x) < 1e-6)
        for wall, Fw in walls.items():
            mag = float(np.linalg.norm(Fw))
            if mag < 1e-9:
                continue
            rv = P[:, :2] - np.array([x, yc])
            r = np.linalg.norm(rv, axis=1)
            in_wall = (P[:, 2] <= z_in_front + 1e-6) if wall == "front" else (P[:, 2] >= z_in_back - 1e-6)
            bo = bore_at.get((wall, round(x, 6)))
            if bo is not None:
                sel = in_wall & (r < bo["d"] / 2 + 0.9 * h)
                seat = f"bore Ø{bo['d']:.0f}"
            else:
                face_z = z_in_front if wall == "front" else z_in_back
                sel = (np.abs(P[:, 2] - face_z) < 1e-6) & (r < dmax / 2 + h)
                seat = "blind seat (no bore in assembly)"
            idx = np.where(sel)[0]
            if len(idx) == 0:
                raise ValueError("could not find load nodes at a shaft support (mesh too coarse)")
            cosang = (rv[idx] @ (Fw / mag)) / np.maximum(r[idx], 1e-9)
            w = np.clip(cosang, 0, None) if bo is not None else np.ones(len(idx))
            if w.sum() == 0:
                w = np.ones(len(idx))
            loads[idx, :2] += np.outer(w / w.sum(), Fw)
            applied.append({"name": f"{' / '.join(names)} support ({wall} wall)",
                            "force_n": [round(float(Fw[0]), 2), round(float(Fw[1]), 2), 0.0],
                            "at_mm": [round(x, 2), round(yc, 2), round(t / 2 if wall == 'front' else D - t / 2, 2)],
                            "seat": seat})
    return loads, applied


def fixture_nodes(model, L, fixture):
    P = model.nodes
    if fixture == "base":
        return np.where(P[:, 1] < 1e-6)[0]
    if fixture == "flange":
        return np.where(P[:, 2] > L["D"] - 1e-6)[0]
    raise ValueError("fixture must be 'base' or 'flange'")


def wetted_face_mask(thermal_model, L):
    """True for faces touching oil (the cavity); bores and outer faces see air."""
    c, n = thermal_model.centres, thermal_model.normals
    probe = c + 0.25 * thermal_model.m.h * n
    I = L["inner"]
    return ((probe[:, 0] > I["x0"]) & (probe[:, 0] < I["x1"]) & (probe[:, 1] > I["y0"])
            & (probe[:, 1] < I["y1"]) & (probe[:, 2] > I["z0"]) & (probe[:, 2] < I["z1"]))
