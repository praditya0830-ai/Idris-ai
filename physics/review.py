"""Rule-based design review of a gearbox variant (before manufacturing).

Works on the SAME layout as /generate-assembly (physics.housing.build_layout), so part names,
positions (`at`, mm, assembly frame) and dimensions match the 3D model. Each finding carries a
number, a concrete fix and a location. Rules: AGMA/ISO gear proportions, SKF bearing practice,
casting/machining DFM guidelines, assembly-sequence logic. Bearings are sized with the ACE
bearing solver (ISO 281) from the actual shaft reactions.
"""
import math
import numpy as np
from .bearings import select_bearing
from .housing import per_shaft_reactions

SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}
CAST_MATERIALS = {"A356-T6", "CastIron-GG25", "Mg-AZ91D"}
MIN_WALL_CAST_MM = 4.0
BEARING_LIFE_H = 5000
BEARING_RELIABILITY = 0.90     # basic L10 rating


def _coprime_suggestion(Np, Ng):
    ratio = Ng / Np
    best = None
    for p in range(max(17, Np - 3), Np + 4):
        for g in range(int(p * ratio) - 2, int(p * ratio) + 3):
            if g > p and math.gcd(p, g) == 1:
                err = abs(g / p - ratio) / ratio
                if best is None or err < best[2]:
                    best = (p, g, err)
    return best


class _Names:
    """Use the exact part names sent by the UI (from /generate-assembly) when they are present."""

    def __init__(self, parts):
        self.names = {str(p.get("name")) for p in (parts or []) if isinstance(p, dict) and p.get("name")}

    def __call__(self, canonical):
        if not self.names or canonical in self.names:
            return canonical
        low = {n.lower(): n for n in self.names}
        return low.get(canonical.lower(), canonical)


def review(design, kinematics, L, material, checks, parts=None, input_rpm=None):
    m = L["module_mm"]
    zp, zg = L["teeth"]
    b = L["face_width_mm"]
    S = L["stages"]
    W, H, D, t, yc = L["W"], L["H"], L["D"], L["wall_mm"], L["yc"]
    gears, shafts = L["gears"], L["shafts"]
    dp, dap, dag = m * zp, m * (zp + 2), m * (zg + 2)
    nm = _Names(parts)
    out = []

    def add(check, sev, part, msg, fix, at):
        out.append({"check": check, "severity": sev, "part": nm(part), "message": msg, "fix": fix,
                    "at": [round(float(v), 2) for v in at]})

    rpm_in = input_rpm
    if not rpm_in and isinstance(kinematics.get("output_rpm"), (int, float)) \
            and isinstance(kinematics.get("ratio_actual"), (int, float)):
        rpm_in = kinematics["output_rpm"] * kinematics["ratio_actual"]
    rpm_in = rpm_in or 3000.0

    # ------------------------------------------------------------- interference
    if "interference" in checks:
        for g in gears:
            r = g["tip"] / 2
            c = min(g["x"] - r - t, W - t - g["x"] - r, H - t - yc - r, yc - r - t)
            zc = g["z0"] + b / 2
            if c < 0:
                add("interference", "critical", g["name"],
                    f"Tip circle Ø{g['tip']:.1f} mm cuts into the housing wall by {-c:.1f} mm.",
                    f"Enlarge the housing by at least {2 * (-c + 3):.0f} mm or reduce the gear diameter.",
                    [g["x"], H - t, zc])
            elif c < max(3.0, 1.5 * m):
                add("interference", "warning", g["name"],
                    f"Radial clearance tip-to-wall is {c:.1f} mm (recommended ≥ {max(3.0, 1.5 * m):.1f} mm for oil "
                    "churning and casting tolerance).",
                    f"Add {2 * (max(3.0, 1.5 * m) - c + 1):.0f} mm to the housing width/height.",
                    [g["x"], H - t, zc])
            # gear vs. any other shaft that overlaps it axially (its own and its mate's shaft excluded)
            mate_shafts = {g["shaft"]}
            for me in L["meshes"]:
                if gears[me["pinion"]] is g:
                    mate_shafts.add(gears[me["gear"]]["shaft"])
                if gears[me["gear"]] is g:
                    mate_shafts.add(gears[me["pinion"]]["shaft"])
            for j, s in enumerate(shafts):
                if j in mate_shafts or s["z1"] < g["z0"] or s["z0"] > g["z0"] + b:
                    continue
                cl = abs(s["x"] - g["x"]) - r - s["d"] / 2
                if cl < 3.0:
                    add("interference", "critical" if cl < 0 else "warning", g["name"],
                        f"Gear tip passes {cl:.1f} mm from the {s['name']} (Ø{s['d']:.0f} mm).",
                        "Increase the center distance or relieve the shaft under the gear.",
                        [g["x"] + math.copysign(r, s["x"] - g["x"]), yc, zc])
        # gear faces vs. front/back walls
        for g in gears:
            gap_f, gap_b = g["z0"] - t, D - t - (g["z0"] + b)
            if min(gap_f, gap_b) < 2.0:
                add("interference", "warning", g["name"],
                    f"Gear face is {min(gap_f, gap_b):.1f} mm from the housing wall.",
                    "Keep ≥ 2 mm axial clearance (more if a bearing or circlip sits there).",
                    [g["x"], yc, g["z0"] if gap_f < gap_b else g["z0"] + b])
        if S == 2:
            s_in, s_out = shafts[0], shafts[2]
            gap = s_out["z0"] - s_in["z1"]
            add("interference", "info", s_in["name"],
                f"Input and output shafts are coaxial with a {gap:.1f} mm axial gap (reverted layout).",
                "Support the inner end of the input shaft with a pilot bearing in a bore of the output shaft "
                "(or a central housing web), and keep ≥ 1 mm end float.",
                [s_in["x"], yc, (s_in["z1"] + s_out["z0"]) / 2])

    # ------------------------------------------------------------- fits / bearings
    if "fits" in checks:
        reac = per_shaft_reactions(L)
        for si, s in enumerate(shafts):
            Fr = max(float(np.linalg.norm(v)) for v in reac[si])
            rpm = rpm_in * s["rpm_factor"]
            brg = select_bearing(s["d"], Fr, 0.0, rpm, BEARING_LIFE_H, BEARING_RELIABILITY, 1.2)
            bore = [bo for bo in L["bores"] if bo["shaft"] == shafts.index(s)]
            z_at = (t / 2) if (bore and bore[0]["wall"] == "front") else (D - t / 2)
            if brg is None:
                add("fits", "critical", s["name"],
                    f"No deep-groove bearing ≥ Ø{s['d']:.0f} mm in the catalogue carries {Fr:.0f} N for "
                    f"{BEARING_LIFE_H} h at {rpm:.0f} rpm.",
                    "Use tapered/cylindrical roller bearings or reduce the load.", [s["x"], yc, z_at])
                continue
            hole = (f"runs in a Ø{bore[0]['d']:.0f} mm clearance hole" if bore
                    else "ends in the wall without any bore")
            sev = "warning"
            add("fits", sev, s["name"],
                f"The shaft {hole} and has no bearing. For {Fr:.0f} N at {rpm:.0f} rpm "
                f"(L10 ≥ {BEARING_LIFE_H} h) the smallest deep-groove bearing is {brg['designation']} "
                f"(Ø{brg['d_mm']}×Ø{brg['D_mm']}×{brg['B_mm']} mm, L10 ≈ {brg['life_h_at_reliability']:.0f} h)"
                + (f"; the Ø{s['d']:.0f} mm shaft must step up to Ø{brg['d_mm']} at the seat." if brg['d_mm'] > s['d'] else "."),
                f"Bore the wall to Ø{brg['D_mm']} {'N7' if material.startswith(('Al', 'A356', 'Mg')) else 'H7'}, "
                f"shaft seat Ø{brg['d_mm']} {'k5' if brg['d_mm'] <= 40 else 'm5'}, and add a boss so the seat is "
                f"≥ {brg['B_mm']} mm long (wall is {t:.0f} mm).",
                [s["x"], yc, z_at])
        if material.startswith(("Al", "A356", "Mg")):
            add("fits", "warning", "Housing",
                f"{material} expands ≈2× more than steel bearing rings (23 vs 12 µm/m·K): at +60 °C a bearing "
                "bore loosens by tens of µm and the outer ring can creep.",
                "Use a tighter housing fit (N7), steel bearing liners, or anti-creep O-ring bearings.",
                [L["x_a"], yc, D - t / 2])
        for g in gears:
            if g["teeth"] != zp:
                continue
            d_sh = shafts[g["shaft"]]["d"]
            d_root = m * (zp - 2.5)
            rim = (d_root - d_sh) / 2
            if rim < 1.2 * 2.25 * m:
                add("fits", "warning", g["name"],
                    f"Root Ø{d_root:.1f} mm leaves a {rim:.1f} mm rim over the Ø{d_sh:.0f} mm shaft (AGMA backup "
                    f"ratio needs ≥ {1.2 * 2.25 * m:.1f} mm): no room for a keyway.",
                    "Cut the pinion directly on the shaft (integral pinion shaft) or use a spline.",
                    [g["x"], yc + dp / 2, g["z0"] + b / 2])

    # ------------------------------------------------------------- DFM
    if "dfm" in checks:
        p1 = gears[0]
        if zp < 17:
            add("dfm", "critical" if zp < 14 else "warning", p1["name"],
                f"Pinion has {zp} teeth: standard 20° teeth are undercut below 17.",
                "Use a positive profile shift x ≥ (17 − z)/17 or more teeth.", [p1["x"], yc + dp / 2, p1["z0"] + b / 2])
        bm, bd = b / m, b / dp
        if bm > 20 or bd > 1.2:
            add("dfm", "warning", p1["name"],
                f"Face width {b:.0f} mm is wide for this pinion (b/m = {bm:.0f}, b/d₁ = {bd:.2f}; guideline b/m 8–16, "
                "b/d₁ ≤ 1.2): load concentrates at one end under shaft misalignment.",
                f"Narrow the face to ≈{min(16 * m, dp):.0f} mm with a larger module, or specify crowning.",
                [p1["x"], yc + dp / 2, p1["z0"] + b / 2])
        g = math.gcd(zp, zg)
        if g > 1:
            sug = _coprime_suggestion(zp, zg)
            fix = (f"Use {sug[0]}/{sug[1]} teeth (ratio changes {100 * sug[2]:.1f} %) so every tooth meets every other."
                   if sug else "Change one tooth count by ±1 to make them coprime.")
            add("dfm", "info", gears[1]["name"],
                f"Tooth counts {zp}/{zg} share the factor {g}: the same tooth pairs always mesh (no hunting tooth), "
                "so wear and defects repeat.", fix, [gears[1]["x"], yc + dag / 2, gears[1]["z0"] + b / 2])
        if material in CAST_MATERIALS and t < MIN_WALL_CAST_MM:
            add("dfm", "warning", "Housing",
                f"Wall {t:.1f} mm is below the {MIN_WALL_CAST_MM:.0f} mm minimum for sand-cast {material}.",
                f"Increase the wall to ≥ {MIN_WALL_CAST_MM:.0f} mm or use die casting.", [W, yc, D / 2])
        if material not in CAST_MATERIALS:
            v_box = W * H * D
            v_part = v_box - (W - 2 * t) * (H - 2 * t) * (D - 2 * t)
            ratio = v_box / max(v_part, 1)
            if ratio > 3:
                add("dfm", "warning", "Housing",
                    f"Machining this housing from a {material} billet removes ≈{100 * (1 - 1 / ratio):.0f} % of the "
                    f"stock (buy-to-fly ≈ {ratio:.1f}).",
                    "For series production cast it (A356-T6) or build it from plates; keep billet machining for prototypes.",
                    [0, H, D])
        add("dfm", "info", "Housing", "Inner and outer corners of the housing are sharp in the current model.",
            f"Specify fillets R ≥ {max(2, round(t / 2)):.0f} mm inside (stress, castability, cutter radius).", [t, t, t])

    # ------------------------------------------------------------- assembly
    if "assembly" in checks:
        max_tip = max(dap, dag)
        max_hole = max(bo["d"] for bo in L["bores"])
        add("assembly", "critical", "Housing",
            f"The housing is a closed box: the gears (tip Ø{max_tip:.0f} mm) cannot be inserted through the "
            f"largest opening (Ø{max_hole:.0f} mm bore).",
            "Split the housing in the shaft plane (y = H/2) into two dowel-located bolted halves, or make the "
            "back wall a removable bolted cover with a gasket.",
            [W / 2, yc, D])
        add("assembly", "info", "Housing", "No breather, oil fill, level or drain plug in the model.",
            f"Breather on top, oil-level plug at the dip depth of the largest gear (≈{2.5 * m:.0f} mm), magnetic "
            "drain plug at the lowest point.", [W / 2, H, D / 2])
        add("assembly", "info", shafts[0]["name"], "Shafts leave the housing without seals.",
            "Fit radial lip seals (e.g. FKM) in the input and output bores.", [shafts[0]["x"], yc, 0.0])
    out.sort(key=lambda f: SEV_ORDER[f["severity"]])
    return out
