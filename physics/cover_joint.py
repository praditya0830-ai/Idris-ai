"""Cover bolts and dowels of the manufacturable housing (layout v4), sized by the bolted-joint solver.

Loads on a bolted cover: the radial reactions of the bearings seated in that cover (lever rule along
each shaft), × K_A = 1.25 (same application factor as the gear rating). In the joint plane they are a
shear force V (vector sum); because each bearing centre sits B/2 outside the joint plane they also tip
the cover: moment M = Σ (Δz × F). Spur gears give no axial force. The screws (ISO 4762, tapped into
corner columns of the body) are chosen as the smallest size / lowest class that passes every check of
physics.bolted_joint; two dowel pins (ISO 8734, Ø = screw size, m6) locate each cover and carry V if it slips.
"""
import math
from .housing import HOUSING_MATERIALS, per_shaft_reactions, support_seat, shaft_supports_z
from .bolted_joint import select_cover_bolts, METRIC

K_A = 1.25
# ultimate strength [MPa] for the stripping check (representative, like the rest of the material data)
SU = {"Al7075-T6": 572.0, "Al6061-T6": 310.0, "Al2024-T3": 483.0, "A356-T6": 234.0, "Ti-6Al-4V": 950.0,
      "Steel-1045": 625.0, "Steel-4140": 1020.0, "CastIron-GG25": 250.0, "Mg-AZ91D": 230.0}
STD_LENGTHS = (6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50)
# Bolt pitch along a sealed cover joint. Design guideline (rule of thumb used for gasketed / O-ring cover
# flanges: pitch ≈ 5–10 × d): screws at most 10·d apart, so the cover is clamped all round, not only at
# the corners (Arya 28 Oct: four corner screws on a 200 mm cover looked wrong — and they are).
PITCH_MAX_D = 10.0


def _material(name):
    key = (name or "Al7075-T6").replace("Steel4140", "Steel-4140").replace("Steel1045", "Steel-1045")
    if key not in HOUSING_MATERIALS:
        key = "Al7075-T6"
    E, nu, Sy, rho, _ = HOUSING_MATERIALS[key]
    return key, E, Sy, SU.get(key, Sy * 1.1)


def column_size(size):
    """Corner column [mm] that holds the tapped hole: ≥ 12 mm, wall ≥ max(3 mm, d/2) around the thread."""
    d = METRIC[size][0]
    return max(12.0, d + 2 * max(3.0, 0.5 * d))


def _rect_clear(rect, tips, yc, extra_clear=2.0):
    x0, y0, w, h = rect
    for x, r in tips:
        nx, ny = min(max(x, x0), x0 + w), min(max(yc, y0), y0 + h)
        if math.hypot(nx - x, ny - yc) <= r + extra_clear:
            return False
    return True


def bolt_positions(L, tips, size, extra_clear=2.0):
    """Corner (and, if needed, mid-side) column positions that clear every gear tip and bearing boss."""
    t, W, H, yc = L["wall_mm"], L["W"], L["H"], L["yc"]
    col = column_size(size)
    corners = [(t, t), (W - t - col, t), (t, H - t - col), (W - t - col, H - t - col)]
    mids = [((W - col) / 2, t), ((W - col) / 2, H - t - col), (t, (H - col) / 2), (W - t - col, (H - col) / 2)]
    ok = [c for c in corners if _rect_clear((c[0], c[1], col, col), tips, yc, extra_clear)]
    if len(ok) < 4:
        ok += [c for c in mids if _rect_clear((c[0], c[1], col, col), tips, yc, extra_clear)]
    return [(cx0, cy0, col) for cx0, cy0 in ok]


def _overlap(a, b, gap=1.0):
    return not (a[0] + a[2] + gap <= b[0] or b[0] + b[2] + gap <= a[0] or
                a[1] + a[3] + gap <= b[1] or b[1] + b[3] + gap <= a[1])


def perimeter_pads(L, tips, size, corners, avoid=(), extra_clear=2.0, allow=(0.0, 0.0)):
    """Extra screw pads along the four walls so that neighbouring screws are at most PITCH_MAX_D·d apart.
    A pad is a col × col boss against the inner wall face at the joint; it is moved along the wall (up to
    half a pitch) to clear gear tips and bearing bosses, and left out if no clear spot exists.
    Where no internal spot is free (gears fill the housing), an external lug outside the wall is used if the
    envelope allows it (allow = (x, y) room per side). Returns (internal pad rects [x0, y0, w, h],
    external lug rects, screw centres in perimeter order incl. corners, max gap, limit)."""
    t, W, H, yc = L["wall_mm"], L["W"], L["H"], L["yc"]
    d = METRIC[size][0]
    col = column_size(size)
    pmax = PITCH_MAX_D * d
    lo_x, hi_x, lo_y, hi_y = t + col / 2, W - t - col / 2, t + col / 2, H - t - col / 2
    corner_c = {(round(cx0 + c / 2, 3), round(cy0 + c / 2, 3)) for cx0, cy0, c in corners}
    sides = [("bottom", (lo_x, lo_y), (hi_x, lo_y)), ("right", (hi_x, lo_y), (hi_x, hi_y)),
             ("top", (hi_x, hi_y), (lo_x, hi_y)), ("left", (lo_x, hi_y), (lo_x, lo_y))]
    pads, lugs, ring = [], [], []
    out = col - t                                   # how far a lug stands outside the wall
    for name, a, b in sides:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, math.ceil(length / pmax))
        if (round(a[0], 3), round(a[1], 3)) in corner_c:
            ring.append(a)
        for k in range(1, n):
            f0 = k / n
            best = None
            step = 2.0 / length
            for j in range(0, int(0.5 * length / n / 2.0) + 1):
                for sgn in ((0,) if j == 0 else (1, -1)):
                    fr = f0 + sgn * j * step
                    if not 0.0 < fr < 1.0:
                        continue
                    cx, cy = a[0] + fr * (b[0] - a[0]), a[1] + fr * (b[1] - a[1])
                    rect = [cx - col / 2, cy - col / 2, col, col]
                    if not _rect_clear(rect, tips, yc, extra_clear):
                        continue
                    if any(_overlap(rect, o) for o in avoid) or any(_overlap(rect, p) for p in pads):
                        continue
                    best = (cx, cy, rect)
                    break
                if best:
                    break
            if best:
                pads.append(best[2])
                ring.append((best[0], best[1]))
                continue
            # external lug at the ideal spot: the lug and a matching cover ear stand outside the wall
            cx, cy = a[0] + f0 * (b[0] - a[0]), a[1] + f0 * (b[1] - a[1])
            room = allow[0] if name in ("left", "right") else allow[1]
            if out > room + 1e-6:
                continue
            if name == "bottom":
                rect = [cx - col / 2, t - col, col, col]
            elif name == "top":
                rect = [cx - col / 2, H - t, col, col]
            elif name == "left":
                rect = [t - col, cy - col / 2, col, col]
            else:
                rect = [W - t, cy - col / 2, col, col]
            if any(_overlap(rect, o) for o in lugs):
                continue
            lugs.append(rect)
            ring.append((rect[0] + col / 2, rect[1] + col / 2))
    gaps = [math.hypot(ring[i][0] - ring[i - 1][0], ring[i][1] - ring[i - 1][1]) for i in range(len(ring))] \
        if len(ring) > 1 else [float("inf")]
    return pads, lugs, ring, max(gaps), pmax


def dowel_layout(L, tips, cols, d):
    """Two dowels in two diagonal columns: each of those columns is lengthened along the wall by d + 6 mm
    and the dowel hole sits in the extension (the screw keeps the column centre). Returns
    (column_rects, dowel_positions) or (None, None) if no diagonal pair has room."""
    t, W, H, yc = L["wall_mm"], L["W"], L["H"], L["yc"]
    rects = [[cx0, cy0, c, c] for cx0, cy0, c in cols]
    ext = d + 6.0
    pairs = sorted(((i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))),
                   key=lambda ij: -math.hypot(cols[ij[0]][0] - cols[ij[1]][0], cols[ij[0]][1] - cols[ij[1]][1]))
    for i, j in pairs:
        trial = [list(r) for r in rects]
        dowels = []
        for k in (i, j):
            cx0, cy0, c = cols[k]
            done = False
            for axis in (0, 1):                      # lengthen along x first (long side), then along y
                inward = (cx0 + c / 2 < W / 2) if axis == 0 else (cy0 + c / 2 < H / 2)
                r = [cx0, cy0, c, c]
                if axis == 0:
                    r = [cx0, cy0, c + ext, c] if inward else [cx0 - ext, cy0, c + ext, c]
                    dpos = (cx0 + c + ext / 2, cy0 + c / 2) if inward else (cx0 - ext / 2, cy0 + c / 2)
                else:
                    r = [cx0, cy0, c, c + ext] if inward else [cx0, cy0 - ext, c, c + ext]
                    dpos = (cx0 + c / 2, cy0 + c + ext / 2) if inward else (cx0 + c / 2, cy0 - ext / 2)
                if r[0] >= t - 1e-6 and r[0] + r[2] <= W - t + 1e-6 and r[1] >= t - 1e-6 and \
                        r[1] + r[3] <= H - t + 1e-6 and _rect_clear(r, tips, yc):
                    trial[k] = r
                    dowels.append(dpos)
                    done = True
                    break
            if not done:
                break
        if len(dowels) == 2:
            return trial, dowels
    return None, None


def cover_loads(L, wall):
    """(V [N], (Mx, My) [N·mm], forces) on the cover `wall` ('front' / 'back'), K_A included."""
    t, D = L.get("cover_mm", L["wall_mm"]), L["D"]
    z_joint = t if wall == "front" else D - t
    reac = per_shaft_reactions(L)
    Fx = Fy = Mx = My = 0.0
    items = []
    for si in range(len(L["shafts"])):
        zz = shaft_supports_z(L, si)
        for k in (0, 1):
            if support_seat(L, si, k) != wall:
                continue
            f = K_A * reac[si][k]
            dz = zz[k] - z_joint
            Fx += f[0]
            Fy += f[1]
            Mx += -dz * f[1]
            My += dz * f[0]
            items.append({"shaft": L["shafts"][si]["name"], "force_n": [round(float(f[0]), 1), round(float(f[1]), 1)],
                          "offset_from_joint_mm": round(float(dz), 2)})
    return float(math.hypot(Fx, Fy)), (float(Mx), float(My)), items


ORING_LOAD_N_MM = 1.5       # O-ring compression force per length, 2.62 mm cord, 70 Sh A at ~25 % squeeze,
                            # upper end of the Parker O-Ring Handbook load curves (conservative)
SQUEEZE_NOM, SQUEEZE_MIN, CORD = 0.25, 0.15, 2.62


def required_cover_thickness(W, H, t, E, lip=3.0, col=12.0, q=ORING_LOAD_N_MM, max_mm=12.0):
    """Cover thickness so that, between two corner screws, the O-ring force does not lift the cover edge by
    more than the squeeze margin (25 % → 15 % of the cord). Edge strip = sealing land (wall + lip) as a
    beam fixed at both screws: δ = q s⁴ / (384 E I), I = a tc³ / 12 (conservative: no help from the plate).
    Returns (tc, span, deflection_at_tc, ok)."""
    a = t + lip
    s = max(W - 2 * t - col, H - 2 * t - col)
    d_allow = (SQUEEZE_NOM - SQUEEZE_MIN) * CORD
    tc_req = (12 * q * s ** 4 / (384 * E * a * d_allow)) ** (1 / 3)
    tc = max(t, math.ceil(tc_req * 2) / 2)
    delta = q * s ** 4 / (384 * E * a * tc ** 3 / 12)
    return min(tc, max_mm), s, delta, tc <= max_mm


def design_cover_joints(L, tips, housing_material="Al7075-T6", bosses=(), allow=(0.0, 0.0)):
    """Per bolted cover: screws, dowels, columns, perimeter pads. Returns {wall: result} and an `ok` flag.
    bosses: (x, r) keep-out circles of the bearing bosses at y = yc (screws never go through them)."""
    tips = list(tips) + list(bosses)
    key, E, Sy, Su = _material(housing_material)
    t = L["wall_mm"]
    tc = L.get("cover_mm", t)
    walls = ["back"] if L["stages"] == 1 else ["front", "back"]
    out, ok = {}, True
    for wall in walls:
        V, M, items = cover_loads(L, wall)
        pos_cache = {}

        pad_cache = {}

        def positions_for(size):
            if size not in pos_cache:
                pos_cache[size] = bolt_positions(L, tips, size)
                cols_ = pos_cache[size]
                # dowel columns are lengthened along the wall: keep pads clear of that extension too
                d_ = METRIC[size][0]
                ext = [[cx0 - d_ - 6, cy0 - d_ - 6, c + 2 * (d_ + 6), c + 2 * (d_ + 6)] for cx0, cy0, c in cols_]
                pad_cache[size] = perimeter_pads(L, tips, size, cols_, avoid=ext, allow=allow)
            ring = pad_cache[size][2]
            return [tuple(p) for p in ring] or [(cx0 + c / 2, cy0 + c / 2) for cx0, cy0, c in pos_cache[size]]
        # M5 is the smallest screw used in a tapped aluminium housing (thread wear on reassembly, handling)
        r = select_cover_bolts(tc, E, E, positions_for, V, M, sizes=("M5", "M6", "M8"), Su_body=Su, Sy_cover=Sy)
        r["loads"] = {"shear_n": round(V, 1), "moment_nmm": [round(M[0], 0), round(M[1], 0)],
                      "application_factor": K_A, "from_bearings": items}
        if r.get("pass"):
            d = METRIC[r["size"]][0]
            need = tc + 2.0 * d
            r["screw_length_mm"] = next((s for s in STD_LENGTHS if s >= need), STD_LENGTHS[-1])
            r["column_mm"] = column_size(r["size"])
            cols = pos_cache[r["size"]]
            r["columns"] = [[round(cx0, 2), round(cy0, 2), c] for cx0, cy0, c in cols]
            rects, dpos = dowel_layout(L, tips, cols, d)
            if rects is None:
                r["pass"] = False
                r["reason"] = "no room for two diagonal dowel pins next to the gears"
                ok = False
                rects, dpos = [[cx0, cy0, c, c] for cx0, cy0, c in cols], []
            r["column_rects"] = [[round(v, 2) for v in rc] for rc in rects]
            pads, lugs, ring, gmax, pmax = pad_cache[r["size"]]
            r["pads"] = [[round(v, 2) for v in p] for p in pads]
            r["lugs"] = [[round(v, 2) for v in p] for p in lugs]
            W_, H_ = L["W"], L["H"]
            r["lug_protrusion_mm"] = [round(max([0.0] + [max(-p[0], p[0] + p[2] - W_) for p in lugs]), 2),
                                      round(max([0.0] + [max(-p[1], p[1] + p[3] - H_) for p in lugs]), 2)]
            r["pad_depth_mm"] = round(2.5 * d + 3.0, 1)
            r["pitch"] = {"max_gap_mm": round(gmax, 1), "limit_mm": round(pmax, 1),
                          "ok": bool(gmax <= pmax + 0.5),
                          "basis": f"neighbouring screws ≤ {PITCH_MAX_D:g}·d apart (design guideline for sealed covers)",
                          "note": None if gmax <= pmax + 0.5 else
                          (f"Between two screws the gap is {gmax:.0f} mm (guideline {pmax:.0f} mm): the gears leave no "
                           "room for a screw boss there and the envelope none for an outside lug. The cover "
                           "thickness is sized for the corner span by the O-ring lift check, so the joint stays "
                           "sealed by calculation.")}
            r["dowels"] = {"count": 2, "diameter_mm": d, "fit": "m6 (holes H7)", "standard": "ISO 8734",
                           "positions": [[round(x, 2), round(y, 2)] for x, y in dpos],
                           "length_mm": next((s for s in STD_LENGTHS if s >= tc + 1.5 * d), STD_LENGTHS[-1])}
            r["spec"] = (f"{r['count']} x ISO 4762 {r['size']} x {r['screw_length_mm']}, class {r['class']}, "
                         f"tightening torque {r['tightening_torque_nm']:g} N·m (K = 0.2); 2 dowel pins ISO 8734 "
                         f"Ø{d:g} m6 x {r['dowels']['length_mm']} in the two lengthened diagonal columns")
        else:
            ok = False
        r["housing_material"] = key
        out[wall] = r
    return out, ok
