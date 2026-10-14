"""Rule-based design review of a gearbox variant (before manufacturing).

Every finding carries a number, a concrete fix and a 3D location in the solver frame.
Rules come from standard practice: AGMA/ISO gear proportions, SKF bearing-fit tables,
casting/machining DFM guidelines. Geometry comes from physics.housing.build_layout.
"""
import math
from .bearings import DEEP_GROOVE

SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}
CAST_MATERIALS = {"A356-T6", "CastIron-GG25", "Mg-AZ91D"}
MIN_WALL_CAST_MM = 4.0


def _designation(bore, OD):
    for k, v in DEEP_GROOVE.items():
        if v[0] == bore and v[1] == OD:
            return k
    return f"Ø{bore}×Ø{OD}"


def _coprime_suggestion(Np, Ng):
    ratio = Ng / Np
    best = None
    for p in range(max(12, Np - 3), Np + 4):
        for g in range(int(p * ratio) - 2, int(p * ratio) + 3):
            if g > p and math.gcd(p, g) == 1:
                err = abs(g / p - ratio) / ratio
                if best is None or err < best[2]:
                    best = (p, g, err)
    return best


class _Names:
    """Map findings to the exact part names sent by /generate-assembly (best effort)."""

    def __init__(self, parts, stages):
        self.parts = [p for p in (parts or []) if isinstance(p, dict) and p.get("name")]
        self.stages = stages

    def _find(self, *keys, index=None):
        hits = [p["name"] for p in self.parts
                if any(k in (str(p.get("kind", "")) + " " + p["name"]).lower() for k in keys)]
        if not hits:
            return None
        if index is not None and index < len(hits):
            return hits[index]
        return hits[0]

    def housing(self):
        return self._find("housing", "case", "casing") or "Housing"

    def pinion(self, s):
        return self._find("pinion", index=s) or f"Stage {s + 1} pinion"

    def gear(self, s):
        hits = [p["name"] for p in self.parts
                if "gear" in (str(p.get("kind", "")) + " " + p["name"]).lower()
                and "pinion" not in (str(p.get("kind", "")) + " " + p["name"]).lower()]
        return hits[s] if s < len(hits) else f"Stage {s + 1} gear"

    def shaft(self, i, n):
        names = ["input", "intermediate", "output"] if n == 3 else ["input", "output"]
        return self._find(names[i] + " shaft") or self._find("shaft", index=i) or f"{names[i].capitalize()} shaft"


def review(design, kinematics, shaft, L, material, checks, parts=None, oil_temp_c=None):
    m = float(design["module_mm"])
    Np, Ng = int(design["pinion_teeth"]), int(design["gear_teeth"])
    F = float(design["face_width_mm"])
    S = L["stages"]
    xs, I, t, off = L["shaft_x_mm"], L["inner"], L["wall_mm"], L["offset"]
    dp, dg = m * Np, m * Ng
    rp_tip, rg_tip = dp / 2 + m, dg / 2 + m
    a = L["center_distance_mm"]
    gap = (I["z1"] - I["z0"] - S * F) / (S + 1)
    zm = [I["z0"] + gap + F / 2 + s * (F + gap) for s in range(S)]
    names = _Names(parts, S)
    out = []

    def add(check, sev, part, msg, fix, at):
        out.append({"check": check, "severity": sev, "part": part, "message": msg, "fix": fix,
                    "at": [round(float(v + o), 2) for v, o in zip(at, off)]})

    nsh = len(xs)
    # ------------------------------------------------------------- interference
    if "interference" in checks:
        for s in range(S):
            for (x, r, nm) in ((xs[s], rp_tip, names.pinion(s)), (xs[s + 1], rg_tip, names.gear(s))):
                c = min(x - r - I["x0"], I["x1"] - x - r, I["y1"] - r)
                if c < 0:
                    add("interference", "critical", nm,
                        f"Tip circle (Ø{2 * r:.1f} mm) cuts into the housing wall by {-c:.1f} mm.",
                        f"Increase the inner housing size by at least {2 * (-c + 3):.0f} mm or reduce the gear diameter.",
                        [x, I["y1"], zm[s]])
                elif c < max(2.0, 1.5 * m):
                    add("interference", "warning", nm,
                        f"Radial clearance tip-to-wall is only {c:.1f} mm (recommended ≥ {max(2.0, 1.5 * m):.1f} mm for oil churning and casting tolerance).",
                        f"Add {max(2.0, 1.5 * m) - c + 1:.0f} mm to the inner width/height of the housing.",
                        [x, I["y1"], zm[s]])
        # large gear of stage s vs. shaft of the next stage that passes behind it (inline layout)
        for s in range(S):
            g_shaft = s + 1
            for j in range(nsh):
                if j in (s, s + 1):
                    continue
                d_j = L["shaft_dia_mm"][j]
                c = abs(xs[j] - xs[g_shaft]) - rg_tip - d_j / 2
                if c < 1.0:
                    add("interference", "critical" if c < 0 else "warning", names.gear(s),
                        f"Gear tip (Ø{2 * rg_tip:.0f} mm) passes {c:.1f} mm from the {names.shaft(j, nsh)} (Ø{d_j:.0f} mm).",
                        "Increase the center distance, reduce the stage gear diameter, or turn the shaft down locally under the gear (relief groove).",
                        [xs[g_shaft] + math.copysign(rg_tip, xs[j] - xs[g_shaft]), 0.0, zm[s]])
                elif c < 3.0:
                    add("interference", "warning", names.gear(s),
                        f"Gear tip runs only {c:.1f} mm from the {names.shaft(j, nsh)} (Ø{d_j:.0f} mm); shaft deflection and runout can close this gap.",
                        "Keep ≥ 3 mm: increase center distance by a few mm or reduce the shaft diameter locally.",
                        [xs[g_shaft] + math.copysign(rg_tip, xs[j] - xs[g_shaft]), 0.0, zm[s]])
        for i in range(nsh - 1):
            OD1, OD2 = L["bearings"][i][1], L["bearings"][i + 1][1]
            lig = abs(xs[i + 1] - xs[i]) - (OD1 + OD2) / 2
            if lig < 0.5 * t:
                add("interference", "critical" if lig < 0 else "warning", names.housing(),
                    f"Wall ligament between neighbouring bearing bores is {lig:.1f} mm (bores Ø{OD1:.0f} and Ø{OD2:.0f} mm).",
                    "Use a lighter bearing series (60xx instead of 62xx) or increase the center distance.",
                    [(xs[i] + xs[i + 1]) / 2, 0.0, I["z0"] - t / 2])
        if S == 2 and gap < 3.0:
            add("interference", "warning", names.gear(0),
                f"Axial gap between stage 1 and stage 2 gears is {gap:.1f} mm.",
                "Keep ≥ 3 mm axial gap (housing_params.axial_gap_mm).",
                [xs[1], 0.0, (zm[0] + zm[1]) / 2])

    # ------------------------------------------------------------- fits
    if "fits" in checks:
        for i in range(nsh):
            bore, OD, Bw = L["bearings"][i]
            d_need = L["shaft_dia_mm"][i]
            shaft_fit = "k5" if bore <= 40 else "m5"
            hous_fit = "N7 (or M7)" if material.startswith(("Al", "A356", "Mg")) else "H7"
            add("fits", "info", names.shaft(i, nsh),
                f"Bearing seat Ø{bore} mm ({_designation(bore, OD)}) for a strength-required shaft of Ø{d_need:.1f} mm: rotating inner ring needs an interference fit.",
                f"Specify shaft {shaft_fit} at the bearing seats and housing bore Ø{OD} {hous_fit}.",
                [xs[i], 0.0, I["z0"] - t / 2])
        if material.startswith(("Al", "A356", "Mg")):
            add("fits", "warning", names.housing(),
                f"{material} expands about 2× more than the steel bearing rings (≈23 vs 12 µm/m·K): at 60 °C above assembly temperature a Ø{L['bearings'][-1][1]} mm bore grows ≈{L['bearings'][-1][1] * 11e-6 * 60 * 1000:.0f} µm more than the ring and the fit loosens.",
                "Use a tighter housing fit (N7), a steel bearing liner/insert, or an anti-creep O-ring bearing.",
                [xs[-1], 0.0, I["z1"] + t / 2])
        for s in range(S):
            d_root = m * (Np - 2.5)
            d_sh = L["shaft_dia_mm"][s]
            rim = (d_root - d_sh) / 2
            if rim < 1.2 * 2.25 * m:
                add("fits", "warning", names.pinion(s),
                    f"Pinion root Ø{d_root:.1f} mm leaves only {rim:.1f} mm rim over a Ø{d_sh:.0f} mm shaft (AGMA backup ratio needs ≥ {1.2 * 2.25 * m:.1f} mm); no room for a keyway.",
                    "Make the pinion integral with its shaft (teeth cut on the shaft) or use a spline.",
                    [xs[s], dp / 2, zm[s]])

    # ------------------------------------------------------------- DFM
    if "dfm" in checks:
        if Np < 17:
            add("dfm", "critical" if Np < 14 else "warning", names.pinion(0),
                f"Pinion has {Np} teeth: a standard 20° tooth is undercut below 17 teeth.",
                "Use a positive profile shift x ≥ (17 − z)/17 on the pinion, or increase the tooth count.",
                [xs[0], dp / 2, zm[0]])
        bm, bd = F / m, F / dp
        if bm > 20 or bd > 1.2:
            add("dfm", "warning", names.pinion(0),
                f"Face width {F:.0f} mm is wide for this pinion (b/m = {bm:.0f}, b/d₁ = {bd:.2f}; guideline b/m 8–16, b/d₁ ≤ 1.2): load concentrates at one end under shaft misalignment.",
                f"Reduce face width to ≈{min(16 * m, 1.0 * dp):.0f} mm and raise the module, or specify crowning / helix correction.",
                [xs[0], dp / 2, zm[0]])
        if m < 1.0:
            add("dfm", "info", names.pinion(0), f"Module {m} mm needs fine-pitch hobs and grinding.",
                "Check supplier capability for fine-pitch gears or use module ≥ 1 mm.", [xs[0], dp / 2, zm[0]])
        g = math.gcd(Np, Ng)
        if g > 1:
            sug = _coprime_suggestion(Np, Ng)
            fix = (f"Use {sug[0]}/{sug[1]} teeth (ratio changes {100 * sug[2]:.1f} %) so every tooth meets every other tooth."
                   if sug else "Change one tooth count by ±1 to make them coprime.")
            add("dfm", "info", names.gear(0),
                f"Tooth counts {Np}/{Ng} share a factor {g}: the same tooth pairs always mesh, so wear and defects repeat (no hunting tooth).",
                fix, [xs[1], dg / 2, zm[0]])
        if material in CAST_MATERIALS and t < MIN_WALL_CAST_MM:
            add("dfm", "warning", names.housing(),
                f"Wall thickness {t:.1f} mm is below the {MIN_WALL_CAST_MM:.0f} mm minimum for sand casting {material}.",
                f"Increase the wall to ≥ {MIN_WALL_CAST_MM:.0f} mm or use die casting.",
                [I["x1"] + t / 2, 0.0, (I["z0"] + I["z1"]) / 2])
        if material not in CAST_MATERIALS:
            Wo = I["x1"] - I["x0"] + 2 * t
            Ho = I["y1"] - I["y0"] + 2 * t
            Do = I["z1"] - I["z0"] + 2 * t + 2 * L["boss_mm"]
            v_box = Wo * Ho * Do
            v_part = v_box - (I["x1"] - I["x0"]) * (I["y1"] - I["y0"]) * (I["z1"] - I["z0"])
            ratio = v_box / max(v_part, 1)
            if ratio > 3:
                add("dfm", "warning", names.housing(),
                    f"Machining this housing from a {material} billet removes ≈{100 * (1 - 1 / ratio):.0f} % of the stock (buy-to-fly ≈ {ratio:.1f}).",
                    "For series production cast it (A356-T6) or weld/bolt plates; keep billet machining for prototypes.",
                    [I["x0"] - t, I["y1"] + t, I["z1"] + t])
        add("dfm", "info", names.housing(),
            "Inner corners of the housing are sharp in the current model.",
            f"Specify inner fillets R ≥ {max(2, round(t / 2)):.0f} mm (stress relief, castability, cutter radius).",
            [I["x0"], I["y0"], I["z0"]])

    # ------------------------------------------------------------- assembly
    if "assembly" in checks:
        max_tip = max(rg_tip, rp_tip) * 2
        max_bore = max(b[1] for b in L["bearings"])
        if max_tip > max_bore:
            add("assembly", "critical", names.housing(),
                f"The gears (tip Ø{max_tip:.0f} mm) cannot pass through the largest bearing bore (Ø{max_bore} mm) of a closed housing: the box cannot be assembled.",
                "Split the housing at the shaft plane (y = 0) with dowel-located bolted halves, or make the back wall a removable bolted cover.",
                [(I["x0"] + I["x1"]) / 2, 0.0, I["z1"] + t / 2])
        add("assembly", "info", names.housing(),
            "Bearings have no axial retention in the model (shoulders, circlips or covers).",
            "Locate one bearing per shaft axially (fixed side) and let the other float; add bearing covers or circlip grooves.",
            [xs[-1], 0.0, I["z1"] + t])
        add("assembly", "info", names.housing(),
            "No breather, oil fill, drain or level plug modelled.",
            "Add a breather on top, an oil-level plug at the dip level of the largest gear (≈2–3 × module), and a magnetic drain plug at the lowest point.",
            [(I["x0"] + I["x1"]) / 2, I["y1"] + t, (I["z0"] + I["z1"]) / 2])
        add("assembly", "info", names.shaft(0, nsh),
            "Input and output shafts leave the housing without seals.",
            "Add radial shaft seals (e.g. FKM lip seals) in the outer bores of the input and output shafts.",
            [xs[0], 0.0, I["z0"] - t - L["boss_mm"]])
    out.sort(key=lambda f: SEV_ORDER[f["severity"]])
    return out
