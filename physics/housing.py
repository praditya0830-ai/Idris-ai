"""Gearbox housing model for 3D analyses — SAME geometry and frame as assembly3d.py.

Frame (identical to /generate-assembly STL): housing occupies x∈[0,W], y∈[0,H], z∈[0,D] mm,
shafts parallel to +z at y = H/2. 1 stage: input axis A, output axis B. 2 stages ("reverted"):
input and output coaxial on axis A, intermediate shaft on axis B, and (layout v4, manufacturable)
a central web between the stages that carries the inner bearings of the input and output shafts.
Every shaft runs in two bearings; through-bores (bearing OD) where a shaft leaves the housing.
"""
import math
import numpy as np
from .fem3d import VoxelModel
from .gears import (ALPHA, CLEARANCE, ETA_STAGE, STAGE_GAP, WEB_GAP, WEB_SEPTUM, shaft_diameter_mm,
                    bearing_width_estimate, web_thickness_estimate, bearing_od_estimate, web_hub_diameter,
                    web_plate_thickness)

SHAFT_STICKOUT = 22.0       # mm minimum shaft stick-out beyond the wall face
SEAL_W = 7.0                # mm, radial lip seal width (outside the bearing, in the boss)
COUPLING_L = 15.0           # mm of free shaft beyond the boss for the coupling
SEAL_LIP_W = 3.0            # mm, internal lip at a bolted joint: sealing land = wall + lip ≥ O-ring groove + lands
SEAL_LIP_Z = 3.0            # mm deep along z (gear faces stay ≥ CLEARANCE − SEAL_LIP_Z = 3 mm away)

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
    # bearing widths per shaft [B at the lower-z support, B at the upper-z support]; provisional until
    # the shaft solver has picked the bearings (variant_shafts.shaft_plan iterates on this)
    bw = hp.get("bearing_widths") if isinstance(hp, dict) else None
    d_list = [d_in, d_out] if stages == 1 else [d_in, d_mid, d_out]
    if not (isinstance(bw, list) and len(bw) == len(d_list)):
        bw = [[bearing_width_estimate(d), bearing_width_estimate(d)] for d in d_list]
    if stages == 1:
        x_a = t + CLEARANCE + dap / 2.0
        x_b = x_a + a
        x_right = x_b + dag / 2.0 + CLEARANCE + t
        d_need = b + 2 * CLEARANCE + 2 * t
        web_t = 0.0
    else:
        x_a = t + CLEARANCE + dag / 2.0
        x_b = x_a + a
        x_right = x_b + dag / 2.0 + CLEARANCE + t
        web_t = _num(hp, "web_mm") or web_thickness_estimate(d_in, d_out)
        d_need = 2 * b + 2 * WEB_GAP + web_t + 2 * CLEARANCE + 2 * t
    h_need = max(dap, dag) + 2 * CLEARANCE + 2 * t
    # end walls / bolted covers can be thicker than the side walls (Tahap 1.1: cover stiffness for the
    # O-ring between the screws); every z-direction wall below uses tc
    tc = max(t, _num(hp, "cover_mm") or t)
    d_need += 2 * (tc - t)
    W = _num(hp, "envelope_width", math.ceil(x_right))
    H = _num(hp, "envelope_height", math.ceil(h_need))
    D = _num(hp, "envelope_depth", math.ceil(d_need))
    if x_right > W + 1.0 or d_need > D + 1.0 or h_need > H + 1.0:
        raise ValueError("housing envelope is too small for this gear layout")
    shift = (W - x_right) / 2.0
    x_a += shift
    x_b += shift
    yc = H / 2.0
    z1 = tc + CLEARANCE
    if stages == 1:
        z2 = z1 + b + STAGE_GAP          # unused for 1 stage
        web = None
    else:
        zw0 = z1 + b + WEB_GAP
        zw1 = zw0 + web_t
        z2 = zw1 + WEB_GAP
        # ribbed web: a plate (max(wall, 6) mm) across the section at the septum between the two
        # bearing pockets + a hub of the full web length around axis A that holds the pockets
        zc = zw0 + bw[0][1] + WEB_SEPTUM / 2.0
        tp = min(web_plate_thickness(t), web_t)
        zp0 = min(max(zw0, zc - tp / 2.0), zw1 - tp)
        hub_d = _num(hp, "web_hub_diameter") or web_hub_diameter(
            max(bearing_od_estimate(d_in), bearing_od_estimate(d_out)))
        web = {"z0": zw0, "z1": zw1, "t": web_t, "plate": (zp0, zp0 + tp), "hub_d": float(hub_d)}
    # bearings: outer face of the bearing flush with the inner wall face, the rest in an outward boss
    # (nothing intrudes into the gear space); web pockets open towards their shaft
    front = lambda B: tc - B / 2.0
    back = lambda B: D - tc + B / 2.0
    # shaft ends leave through boss + lip seal (7 mm) and keep 15 mm for the coupling
    stick_f = max(SHAFT_STICKOUT, bw[0][0] - tc + SEAL_W + COUPLING_L)
    stick_b = max(SHAFT_STICKOUT, bw[-1][1] - tc + SEAL_W + COUPLING_L)
    stick = stick_f

    # shafts (z extents as in assembly3d v4: every shaft ends inside its two bearing seats)
    if stages == 1:
        supports = [(front(bw[0][0]), back(bw[0][1])), (front(bw[1][0]), back(bw[1][1]))]
        shafts = [{"name": "Input shaft", "x": x_a, "d": d_in, "z0": -stick, "z1": D - tc + bw[0][1],
                   "torque": T_in, "rpm_factor": 1.0},
                  {"name": "Output shaft", "x": x_b, "d": d_out, "z0": tc - bw[1][0], "z1": D + stick_b,
                   "torque": T_in * u * ETA_STAGE, "rpm_factor": 1 / u}]
        gears = [{"name": "Pinion", "shaft": 0, "x": x_a, "z0": z1, "d": dp, "tip": dap, "teeth": zp, "stage": 0},
                 {"name": "Gear", "shaft": 1, "x": x_b, "z0": z1, "d": dg, "tip": dag, "teeth": zg, "stage": 0}]
        bores = [{"wall": "front", "x": x_a, "d": d_in + 2, "shaft": 0},
                 {"wall": "back", "x": x_b, "d": d_out + 2, "shaft": 1}]
    else:
        supports = [(front(bw[0][0]), web["z0"] + bw[0][1] / 2.0),
                    (front(bw[1][0]), back(bw[1][1])),
                    (web["z1"] - bw[2][0] / 2.0, back(bw[2][1]))]
        shafts = [{"name": "Input shaft", "x": x_a, "d": d_in, "z0": -stick, "z1": web["z0"] + bw[0][1],
                   "torque": T_in, "rpm_factor": 1.0},
                  {"name": "Intermediate shaft", "x": x_b, "d": d_mid, "z0": tc - bw[1][0], "z1": D - tc + bw[1][1],
                   "torque": T_in * u * ETA_STAGE, "rpm_factor": 1 / u},
                  {"name": "Output shaft", "x": x_a, "d": d_out, "z0": web["z1"] - bw[2][0], "z1": D + stick_b,
                   "torque": T_in * u * u * ETA_STAGE ** 2, "rpm_factor": 1 / u ** 2}]
        gears = [{"name": "Pinion (stage 1)", "shaft": 0, "x": x_a, "z0": z1, "d": dp, "tip": dap, "teeth": zp, "stage": 0},
                 {"name": "Gear (stage 1)", "shaft": 1, "x": x_b, "z0": z1, "d": dg, "tip": dag, "teeth": zg, "stage": 0},
                 {"name": "Pinion (stage 2)", "shaft": 1, "x": x_b, "z0": z2, "d": dp, "tip": dap, "teeth": zp, "stage": 1},
                 {"name": "Gear (stage 2)", "shaft": 2, "x": x_a, "z0": z2, "d": dg, "tip": dag, "teeth": zg, "stage": 1}]
        bores = [{"wall": "front", "x": x_a, "d": d_in + 2, "shaft": 0},
                 {"wall": "back", "x": x_a, "d": d_out + 2, "shaft": 2}]

    # designed bores (bearing OD from the ACE shaft solver) and the bolted end cover of assembly3d v4
    fb, bb = _num(hp, "front_bore_diameter"), _num(hp, "back_bore_diameter")
    if fb:
        bores[0]["d"] = fb
    if bb:
        bores[1]["d"] = bb

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
            "center_distance_mm": a, "W": W, "H": H, "D": D, "wall_mm": t, "cover_mm": tc, "yc": yc, "z1": z1, "z2": z2,
            "x_a": x_a, "x_b": x_b, "shafts": shafts, "gears": gears, "bores": bores, "meshes": meshes,
            "input_torque_Nm": T_in, "designed_bearings": bool(fb or bb),
            "web": web, "supports": supports, "bearing_widths": bw, "stickout": (stick_f, stick_b),
            "bearing_ods": hp.get("bearing_ods") if isinstance(hp, dict) else None,
            "web_hole_mm": _num(hp, "web_hole_diameter") or ((d_mid or 0) + 4 if stages == 2 else None),
            "split_cover": bool(isinstance(hp, dict) and hp.get("split_cover")),
            "inner": {"x0": t, "x1": W - t, "y0": t, "y1": H - t, "z0": tc, "z1": D - tc},
            "lip": {"width_mm": SEAL_LIP_W, "depth_mm": SEAL_LIP_Z,
                    "walls": ["back"] if stages == 1 else ["front", "back"]},
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


def seat_features(L):
    """Bearing seats as modelled in 3D (v0.12): for every support its bore radius, width and centre, and for
    cover seats the outward boss (wall b_w = max(4, 0.12 D)) with a lip-seal recess (exit shafts) or a 3 mm
    closing cap. Needs L['bearing_ods'] (set by the shaft plan); [] for older layouts."""
    ods = L.get("bearing_ods")
    if not ods:
        return []
    tc = L.get("cover_mm", L["wall_mm"])
    D = L["D"]
    n = len(L["shafts"])
    out = []
    for si, sh in enumerate(L["shafts"]):
        for k in (0, 1):
            seat = support_seat(L, si, k)
            Dbr = float(ods[si][k])
            B = float(L["bearing_widths"][si][k])
            zb = shaft_supports_z(L, si)[k]
            f = {"shaft": si, "k": k, "seat": seat, "x": sh["x"], "R": Dbr / 2.0, "B": B, "z": zb}
            if seat in ("front", "back"):
                exits = (si == 0 and seat == "front") or (si == n - 1 and seat == "back")
                out_len = max(0.0, B - tc) + (7.0 if exits else 3.0)
                f.update(exit=exits, wb=max(4.0, 0.12 * Dbr),
                         boss=(-out_len, 0.0) if seat == "front" else (D, D + out_len))
            out.append(f)
    return out


def boss_extra_area_mm2(L):
    """Outer area the bearing bosses add to the plain box (side of every boss; an exit bore is open, so its
    hole area is removed). Used by the lumped thermal screen so it matches the 3D model."""
    a = 0.0
    for f in seat_features(L):
        if "boss" not in f:
            continue
        ro = f["R"] + f["wb"]
        a += 2 * math.pi * ro * (f["boss"][1] - f["boss"][0])
        if f["exit"]:
            a -= math.pi * f["R"] ** 2
    return a


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
    web = L.get("web")
    tc = L.get("cover_mm", t)
    lip = L.get("lip")
    extra_z = (web["z0"], web["z1"]) + tuple(web.get("plate", ())) if web else ()
    if lip:
        extra_z += tuple(tc + lip["depth_mm"] if w == "front" else D - tc - lip["depth_mm"] for w in lip["walls"])
    zs = _axis_planes(D, tc, layers, s, extra=extra_z)
    feats = seat_features(L)
    zmin = min([f["boss"][0] for f in feats if "boss" in f] + [0.0])
    zmax = max([f["boss"][1] for f in feats if "boss" in f] + [D])
    if zmin < 0 or zmax > D:                     # bearing bosses outside the box: extra planes along z
        front = [z for z in np.linspace(zmin, 0.0, max(2, int(round(-zmin / s))) + 1)][:-1] if zmin < 0 else []
        back = [z for z in np.linspace(D, zmax, max(2, int(round((zmax - D) / s))) + 1)][1:] if zmax > D else []
        cap_planes = []
        for f in feats:
            if "boss" in f and not f["exit"]:
                cap_planes.append(f["boss"][0] + 3.0 if f["seat"] == "front" else f["boss"][1] - 3.0)
        zs = np.array(sorted(set(np.round(np.r_[front, zs, back, cap_planes], 6))))
        keep = [zs[0]]
        for z in zs[1:]:                         # merge planes closer than 1 mm (keeps the mesh small)
            if z - keep[-1] >= 1.0 or z in (0.0, D):
                keep.append(z)
        zs = np.array(keep)
    cx, cy, cz = (0.5 * (pl[1:] + pl[:-1]) for pl in (xs, ys, zs))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    cavity = (X > t) & (X < W - t) & (Y > t) & (Y < H - t) & (Z > tc) & (Z < D - tc)
    solid = ~cavity & (Z > 0) & (Z < D)
    for bo in L["bores"]:
        r2 = (X - bo["x"]) ** 2 + (Y - yc) ** 2
        in_wall = (Z < tc) if bo["wall"] == "front" else (Z > D - tc)
        solid &= ~(in_wall & (r2 < (bo["d"] / 2) ** 2))
    if lip:                                      # internal sealing lip at the bolted joints
        w_, dz = lip["width_mm"], lip["depth_mm"]
        ring = cavity & ((X < t + w_) | (X > W - t - w_) | (Y < t + w_) | (Y > H - t - w_))
        for wall in lip["walls"]:
            band = (Z < tc + dz) if wall == "front" else (Z > D - tc - dz)
            solid |= ring & band
    if web:
        # central web (2-stage): plate + bearing hub, clearance hole of the intermediate shaft
        solid |= web_solid(L, X, Y, Z)
    for f in feats:                              # bearing seats (v0.12): pockets, bosses, caps
        r2 = (X - f["x"]) ** 2 + (Y - yc) ** 2
        if f["seat"] == "web":
            pocket = (r2 < f["R"] ** 2) & (Z > f["z"] - f["B"] / 2) & (Z < f["z"] + f["B"] / 2)
            solid &= ~pocket
            continue
        in_cover = (Z < tc) if f["seat"] == "front" else (Z > D - tc)
        z0, z1 = f["boss"]
        ring = (r2 < (f["R"] + f["wb"]) ** 2) & (Z > z0) & (Z < z1)
        solid |= ring
        solid &= ~((in_cover | ((Z > z0) & (Z < z1))) & (r2 < f["R"] ** 2))      # bore through cover + boss
        if not f["exit"]:                        # closed boss: 3 mm cap at its outer end
            cap = (Z < z0 + 3.0) if f["seat"] == "front" else (Z > z1 - 3.0)
            solid |= cap & (r2 < f["R"] ** 2) & (Z > z0) & (Z < z1)
    return VoxelModel(solid, planes=(xs, ys, zs))


def web_solid(L, X, Y, Z):
    """True where the central web has material (plate across the section, hub around axis A)."""
    web = L.get("web")
    if not web:
        return np.zeros_like(X, dtype=bool)
    yc = L["yc"]
    zp0, zp1 = web.get("plate", (web["z0"], web["z1"]))
    in_plate = (Z > zp0) & (Z < zp1)
    in_hub = (Z > web["z0"]) & (Z < web["z1"]) & \
        (((X - L["x_a"]) ** 2 + (Y - yc) ** 2) < (web.get("hub_d", 1e9) / 2) ** 2)
    hole = ((X - L["x_b"]) ** 2 + (Y - yc) ** 2) < (L["web_hole_mm"] / 2) ** 2
    return (in_plate | in_hub) & ~hole


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
    """Axial positions (bearing centres, z mm) of the two bearings of shaft si (layout v4)."""
    if L.get("supports"):
        return tuple(L["supports"][si])
    t, D = L["wall_mm"], L["D"]
    return t / 2, D - t / 2


def support_seat(L, si, k):
    """Where support k (0 = lower z, 1 = upper z) of shaft si sits: 'front', 'back' or 'web'."""
    z = shaft_supports_z(L, si)[k]
    t, D = L.get("cover_mm", L["wall_mm"]), L["D"]
    if z <= t + 1e-6:
        return "front"
    if z >= D - t - 1e-6:
        return "back"
    return "web"


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
    """Nodal loads on the housing from every bearing (layout v4): the reaction of each shaft support
    (lever rule along the shaft) goes into its seat — a through-bore edge (cosine law), the inner wall
    face behind a blind pocket, or the central web around the axis."""
    h = model.h
    P = model.nodes
    t, D, yc = L["wall_mm"], L["D"], L["yc"]
    loads = np.zeros((model.n_nodes, 3))
    applied = []
    zs = np.unique(np.round(P[:, 2], 6))
    near = lambda z: zs[np.argmin(np.abs(zs - z))]
    tc = L.get("cover_mm", t)
    z_in_front, z_in_back = near(tc), near(D - tc)
    web = L.get("web")
    reac = per_shaft_reactions(L)
    for si, s in enumerate(L["shafts"]):
        for k in (0, 1):
            Fw = reac[si][k]
            mag = float(np.linalg.norm(Fw))
            if mag < 1e-9:
                continue
            seat_at = support_seat(L, si, k)
            z_b = shaft_supports_z(L, si)[k]
            rv = P[:, :2] - np.array([s["x"], yc])
            r = np.linalg.norm(rv, axis=1)
            bo = next((b for b in L["bores"] if b["shaft"] == si and b["wall"] == seat_at), None)
            B = L["bearing_widths"][si][k] if L.get("bearing_widths") else 12.0
            r_seat = max(s["d"] / 2 + h, (bo["d"] / 2) if bo else 0)
            feat = next((f for f in seat_features(L) if f["shaft"] == si and f["k"] == k), None)
            if feat is not None:
                # v0.12: the outer ring presses on its seat bore (cover + boss, or the web pocket):
                # cosine-law pressure on the bore surface over the bearing width
                zlo, zhi = feat["z"] - feat["B"] / 2 - 1e-6, feat["z"] + feat["B"] / 2 + 1e-6
                sel = (r > feat["R"] - 0.1 * h) & (r < feat["R"] + 0.95 * h) & (P[:, 2] >= zlo) & (P[:, 2] <= zhi)
                where = {"front": "front cover boss", "back": "back cover boss", "web": "central web pocket"}[seat_at]
                seat = f"bearing seat Ø{2 * feat['R']:.0f} ({where})"
                cosw = True
            elif seat_at in ("front", "back") and bo is not None:
                in_wall = (P[:, 2] <= z_in_front + 1e-6) if seat_at == "front" else (P[:, 2] >= z_in_back - 1e-6)
                sel = in_wall & (r < bo["d"] / 2 + 0.9 * h)
                seat = f"bore Ø{bo['d']:.0f} ({seat_at} wall)"
                cosw = True
            elif seat_at in ("front", "back"):
                face_z = z_in_front if seat_at == "front" else z_in_back
                sel = (np.abs(P[:, 2] - face_z) < 1e-6) & (r < r_seat + 1.5 * h)
                seat = f"blind bearing pocket ({seat_at} wall)"
                cosw = False
            else:
                face_z = near(web["z0"]) if z_b < 0.5 * (web["z0"] + web["z1"]) else near(web["z1"])
                sel = (np.abs(P[:, 2] - face_z) < 1e-6) & (r < r_seat + 1.5 * h)
                seat = "bearing pocket in the central web"
                cosw = False
            idx = np.where(sel)[0]
            if len(idx) == 0:
                raise ValueError("could not find load nodes at a shaft support (mesh too coarse)")
            cosang = (rv[idx] @ (Fw / mag)) / np.maximum(r[idx], 1e-9)
            w = np.clip(cosang, 0, None) if cosw else np.ones(len(idx))
            if w.sum() == 0:
                w = np.ones(len(idx))
            loads[idx, :2] += np.outer(w / w.sum(), Fw)
            applied.append({"name": f"{s['name']} bearing ({seat_at})",
                            "force_n": [round(float(Fw[0]), 2), round(float(Fw[1]), 2), 0.0],
                            "at_mm": [round(s["x"], 2), round(yc, 2), round(z_b, 2)], "seat": seat})
    return loads, applied


def fixture_nodes(model, L, fixture):
    P = model.nodes
    if fixture == "base":
        return np.where(P[:, 1] < 1e-6)[0]
    if fixture == "flange":
        return np.where(np.abs(P[:, 2] - L["D"]) < 1e-6)[0]
    raise ValueError("fixture must be 'base' or 'flange'")


def seat_face_mask(thermal_model, L):
    """True for faces of the bearing-seat bores (covers, bosses, web pockets): the bearing outer ring sits
    there, so they are neither oil-wetted cavity wall nor air-cooled outer skin (treated as adiabatic)."""
    c, n = thermal_model.centres, thermal_model.normals
    probe = c + 0.25 * thermal_model.m.h * n
    yc, D, tc = L["yc"], L["D"], L.get("cover_mm", L["wall_mm"])
    mask = np.zeros(len(c), dtype=bool)
    for f in seat_features(L):
        r2 = (probe[:, 0] - f["x"]) ** 2 + (probe[:, 1] - yc) ** 2
        if f["seat"] == "web":
            zr = (probe[:, 2] > f["z"] - f["B"] / 2) & (probe[:, 2] < f["z"] + f["B"] / 2)
        else:
            z0, z1 = f["boss"]
            lo, hi = (z0, tc) if f["seat"] == "front" else (D - tc, z1)
            zr = (probe[:, 2] > lo) & (probe[:, 2] < hi)
        mask |= (r2 < f["R"] ** 2) & zr
    return mask


def wetted_face_mask(thermal_model, L):
    """True for faces touching oil (the cavity); bores and outer faces see air."""
    c, n = thermal_model.centres, thermal_model.normals
    probe = c + 0.25 * thermal_model.m.h * n
    I = L["inner"]
    wet = ((probe[:, 0] > I["x0"]) & (probe[:, 0] < I["x1"]) & (probe[:, 1] > I["y0"])
           & (probe[:, 1] < I["y1"]) & (probe[:, 2] > I["z0"]) & (probe[:, 2] < I["z1"]))
    web = L.get("web")
    if web:                                   # faces whose probe lands inside the web are not wetted
        wet &= ~web_solid(L, probe[:, 0], probe[:, 1], probe[:, 2])
    return wet & ~seat_face_mask(thermal_model, L)
