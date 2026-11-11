"""Standard-part catalog (Mark 1 #3) + bolt/hole checks for Auto Assembly (ui/rencana-sketch-correction-dan-auto-assembly.md).

Sources and how each table was checked (kebenaran angka dulu):
  * ISO 273:1979 clearance holes (fine/medium/coarse = close/normal/loose, H12/H13/H14). Values as published in
    IS 1821:1987 (identical to ISO 273) and cross-checked against Albany County Fasteners and AMESweb tables.
    NOTE: the open bd_warehouse table differs for M12-M16 (13/14/15, 15/16/17, 17/18/19); those are NOT ISO 273 and
    are not used here.
  * ISO 261 coarse pitch; tap drill (ISO 2306-based, AMESweb / madegooddesigns tables).
  * ISO 4762 socket head cap screw dk (max), k, s; ISO 4032 hex nut m, s; ISO 7089 washer d1, d2, h (max):
    bd_warehouse (Apache-2.0) data, dk and nut values cross-checked against madegooddesigns (fasteners.eu / omdl).
  * Deep-groove ball bearings d/D/B (ISO 15) with C, C0 as tabulated in bd_warehouse (SKF catalogue numbers);
    ratings marked verified=False until spot-checked against the manufacturer catalogue.
  * Thread engagement: practical rule (>= 1.0 d in steel, >= 1.5 d in aluminium, >= 2.0 d in plastics),
    labelled as a rule of thumb, not a standard.
"""
from __future__ import annotations

import re

# size: (pitch coarse, tap drill, fine, medium, coarse)
METRIC = {
    "M1.6": (0.35, 1.25, 1.7, 1.8, 2.0), "M2": (0.4, 1.6, 2.2, 2.4, 2.6), "M2.5": (0.45, 2.05, 2.7, 2.9, 3.1),
    "M3": (0.5, 2.5, 3.2, 3.4, 3.6), "M3.5": (0.6, 2.9, 3.7, 3.9, 4.2), "M4": (0.7, 3.3, 4.3, 4.5, 4.8),
    "M5": (0.8, 4.2, 5.3, 5.5, 5.8), "M6": (1.0, 5.0, 6.4, 6.6, 7.0), "M7": (1.0, 6.0, 7.4, 7.6, 8.0),
    "M8": (1.25, 6.8, 8.4, 9.0, 10.0), "M10": (1.5, 8.5, 10.5, 11.0, 12.0), "M12": (1.75, 10.2, 13.0, 13.5, 14.5),
    "M14": (2.0, 12.0, 15.0, 15.5, 16.5), "M16": (2.0, 14.0, 17.0, 17.5, 18.5), "M18": (2.5, 15.5, 19.0, 20.0, 21.0),
    "M20": (2.5, 17.5, 21.0, 22.0, 24.0), "M22": (2.5, 19.5, 23.0, 24.0, 26.0), "M24": (3.0, 21.0, 25.0, 26.0, 28.0),
    "M27": (3.0, 24.0, 28.0, 30.0, 32.0), "M30": (3.5, 26.5, 31.0, 33.0, 35.0),
}
SERIES = ("fine", "medium", "coarse")
SHCS_ISO4762 = {   # dk max, k, s (hex key)
    "M2": (3.98, 2, 1.5), "M3": (5.68, 3, 2.5), "M4": (7.22, 4, 3), "M5": (8.72, 5, 4), "M6": (10.22, 6, 5),
    "M8": (13.27, 8, 6), "M10": (16.27, 10, 8), "M12": (18.27, 12, 10), "M14": (21.33, 14, 12), "M16": (24.33, 16, 14),
    "M20": (30.33, 20, 17), "M24": (36.39, 24, 19), "M30": (45.39, 30, 22)}
SHCS_LENGTHS = (2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120,
                130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300)
NUT_ISO4032 = {"M3": (2.4, 5.5), "M4": (3.2, 7), "M5": (4.7, 8), "M6": (5.2, 10), "M8": (6.8, 13), "M10": (8.4, 16),
               "M12": (10.8, 18), "M14": (12.8, 21), "M16": (14.8, 24), "M20": (18, 30)}     # m, s
WASHER_ISO7089 = {"M3": (3.2, 7, 0.55), "M4": (4.3, 9, 0.9), "M5": (5.3, 10, 1.1), "M6": (6.4, 12, 1.8),
                  "M8": (8.4, 16, 1.8), "M10": (10.5, 20, 2.2), "M12": (13, 24, 2.7), "M14": (15, 28, 2.7),
                  "M16": (17, 30, 3.3), "M20": (21, 37, 3.3)}                                    # d1, d2, h max
WASHER_H_NOM = {"M3": 0.5, "M4": 0.8, "M5": 1.0, "M6": 1.6, "M8": 1.6, "M10": 2.0, "M12": 2.5, "M14": 2.5,
                "M16": 3.0, "M20": 3.0}                                                          # ISO 7089 nominal h
SHCS_RANGE = {"M2": (3, 20), "M3": (5, 30), "M4": (6, 40), "M5": (8, 50), "M6": (8, 60), "M8": (10, 80),
              "M10": (16, 100), "M12": (20, 120), "M14": (25, 140), "M16": (25, 160), "M20": (30, 200),
              "M24": (40, 200), "M30": (45, 200)}                                               # ISO 4762 lengths
BEARINGS = {   # designation: d, D, B, C kN, C0 kN, source, verified
    "634": (4, 16, 5, 1.11, 0.38, None, False), "625": (5, 16, 5, 1.14, 0.38, None, False),
    "608": (8, 22, 7, 3.45, 1.37, None, False), "629": (9, 26, 8, 4.75, 1.96, None, False),
    "6000": (10, 26, 8, 4.75, 1.96, None, False), "6200": (10, 30, 9, 5.4, 2.36, None, False),
    # d > 10 mm: each value read from a manufacturer data page (ratings differ by brand; shown with the brand)
    "6201": (12, 32, 10, 7.5, 3.05, "NSK 6201 data sheet (RS Components docs)", True),
    "6002": (15, 32, 9, 5.85, 2.85, "SKF 6002 catalogue table", True),
    "6202": (15, 35, 11, 8.06, 3.75, "SKF 6202 catalogue table", True),
    "6203": (17, 40, 12, 10.0, None, "SKF Explorer 6203 data sheet (C0 not read)", True),
    "6205": (25, 52, 15, 15.4, 7.85, "NSK 6205 product page", True)}
ENGAGEMENT_D = {"steel": 1.0, "stainless": 1.0, "titanium": 1.0, "cast_iron": 1.25, "brass": 1.25,
                "aluminium": 1.5, "pla": 2.0, "abs": 2.0, "nylon": 2.0}

SOURCES = {
    "clearance": "ISO 273:1979 (= IS 1821:1987), cross-checked Albany County Fasteners, AMESweb",
    "pitch_tap": "ISO 261 coarse pitch; tap drill ISO 2306-based (AMESweb)",
    "shcs": "ISO 4762 via bd_warehouse (Apache-2.0); dk = max",
    "nut": "ISO 4032 via bd_warehouse (Apache-2.0)",
    "washer": "ISO 7089 via bd_warehouse (Apache-2.0); h = nominal ISO 7089, h_max from bd_warehouse",
    "bearing": "ISO 15 dimensions; C, C0 = manufacturer catalogue via bd_warehouse (not yet spot-checked)",
    "engagement": "practical rule of thumb (not a standard)",
}


def norm_size(s) -> str | None:
    m = re.fullmatch(r"\s*[Mm]\s*(\d+(?:[.,]\d+)?)\s*(?:[xX×-].*)?", str(s or ""))
    if not m:
        return None
    v = m.group(1).replace(",", ".")
    v = v[:-2] if v.endswith(".0") else v
    return "M" + v if "M" + v in METRIC else None


def fastener(size) -> dict | None:
    k = norm_size(size)
    if not k:
        return None
    p, tap, fine, med, coarse = METRIC[k]
    out = {"size": k, "d": float(k[1:]), "pitch": p, "tap_drill": tap,
           "clearance": {"fine": fine, "medium": med, "coarse": coarse},
           "sources": {"clearance": SOURCES["clearance"], "pitch_tap": SOURCES["pitch_tap"]}}
    if k in SHCS_ISO4762:
        dk, kk, s = SHCS_ISO4762[k]
        lo, hi = SHCS_RANGE.get(k, (SHCS_LENGTHS[0], SHCS_LENGTHS[-1]))
        lengths = [x for x in SHCS_LENGTHS if lo <= x <= hi]
        out["socket_head"] = {"dk": dk, "k": kk, "s": s, "lengths": lengths, "standard": "ISO 4762", "dk_is": "max"}
        out["socket_head_iso4762"] = {"dk_max": dk, "k": kk, "s": s, "lengths": lengths}       # v0.26 name
        out["sources"]["shcs"] = SOURCES["shcs"]
    if k in NUT_ISO4032:
        m, sw = NUT_ISO4032[k]
        out["nut"] = {"m": m, "s": sw, "standard": "ISO 4032"}
        out["hex_nut_iso4032"] = {"m": m, "s": sw}                                             # v0.26 name
        out["sources"]["nut"] = SOURCES["nut"]
    if k in WASHER_ISO7089:
        d1, d2, hmax = WASHER_ISO7089[k]
        out["washer"] = {"d1": d1, "d2": d2, "h": WASHER_H_NOM.get(k, hmax), "h_max": hmax, "standard": "ISO 7089"}
        out["washer_iso7089"] = {"d1": d1, "d2": d2, "h_max": hmax}                            # v0.26 name
        out["sources"]["washer"] = SOURCES["washer"]
    return out


def bearings(d_min=None, d_max=None) -> list:
    out = []
    for name, (d, D, B, C, C0, src, ok) in BEARINGS.items():
        if (d_min is None or d >= d_min) and (d_max is None or d <= d_max):
            out.append({"designation": name, "type": "deep groove ball", "d": d, "D": D, "B": B, "C_kN": C,
                        "C0_kN": C0, "verified": ok, "source": src or SOURCES["bearing"]})
    return out


def size_for_hole(hole_d: float) -> list:
    """Which bolt sizes and series a hole diameter matches (within 0.05 mm)."""
    hits = []
    for k, (p, tap, *cl) in METRIC.items():
        for series, v in zip(SERIES, cl):
            if abs(v - hole_d) <= 0.05:
                hits.append({"size": k, "kind": "clearance", "series": series})
        if abs(tap - hole_d) <= 0.05:
            hits.append({"size": k, "kind": "tap_drill"})
    return hits


E_MPA = {"steel": 207000.0, "stainless": 193000.0, "titanium": 114000.0, "cast_iron": 100000.0, "brass": 105000.0,
         "aluminium": 71000.0, "pla": 3500.0, "abs": 2200.0, "nylon": 2800.0}
PLASTICS = ("pla", "abs", "nylon")
CODES = ("unknown_size", "clearance_small", "clearance_tight", "clearance_loose", "tap_mismatch",
         "lower_tap_mismatch", "lower_clearance_small", "engagement_short", "too_long_for_thread", "nut_too_short",
         "strength_yield", "strength_separation", "strength_not_valid")


def _hole_findings(f, hole, kind, series, where=""):
    """where = '' for the top part, 'lower_' for the part below."""
    out, d = [], f["d"]
    if hole is None:
        return out
    hole = float(hole)
    part = "lower part: " if where else ""
    if kind == "tapped":
        if abs(hole - f["tap_drill"]) > 0.15:
            match = [h["size"] for h in size_for_hole(hole) if h["kind"] == "tap_drill"]
            out.append({"level": "error", "code": f"{where}tap_mismatch", "message":
                        f"{part}tap hole Ø{hole:g} does not match {f['size']} (tap drill Ø{f['tap_drill']:g})"
                        + (f"; it is the tap drill of {match[0]}" if match else ""),
                        "fix": f"hole Ø{f['tap_drill']:g} for {f['size']}", "fix_value": f["tap_drill"],
                        "source": SOURCES["pitch_tap"]})
        return out
    fine, coarse = f["clearance"]["fine"], f["clearance"]["coarse"]
    if hole < d:
        match = [h["size"] for h in size_for_hole(hole) if h["kind"] == "clearance"]
        out.append({"level": "error", "code": f"{where}clearance_small" if where else "clearance_small", "message":
                    f"{part}hole Ø{hole:g} is smaller than the {f['size']} bolt (Ø{d:g})"
                    + (f"; it is the clearance hole of {match[0]}" if match else ""),
                    "fix": f"hole Ø{f['clearance'][series]:g} (ISO 273 {series}) or a smaller bolt",
                    "fix_value": f["clearance"][series], "source": SOURCES["clearance"]})
    elif hole < fine - 1e-9:
        out.append({"level": "warning", "code": "clearance_tight", "message": f"{part}hole Ø{hole:g} is tighter than "
                    f"ISO 273 fine (Ø{fine:g}) for {f['size']}", "fix": f"Ø{f['clearance'][series]:g}",
                    "fix_value": f["clearance"][series], "source": SOURCES["clearance"]})
    elif hole > coarse + 1e-9:
        out.append({"level": "warning", "code": "clearance_loose", "message": f"{part}hole Ø{hole:g} is looser than "
                    f"ISO 273 coarse (Ø{coarse:g}) for {f['size']}: the head may not bear fully; use a washer",
                    "source": SOURCES["clearance"]})
    return out


def bolt_check(p: dict) -> dict:
    """Bolt <-> hole check for Auto Assembly (+ optional joint strength).
    p: size, hole_d + hole_kind (top part, 'clearance' default), lower_hole_d + lower_hole_kind ('tapped' |
       'clearance'), series, clamp_mm (parts the bolt passes through), bolt_length_mm, washers, thread_material,
       thread_depth_mm, nut (True = through-bolt with nut; implied by lower_hole_kind = 'clearance'),
       optional strength: load_n (tension per bolt), property_class ('8.8' | '10.9' | '12.9'),
       clamp_material (material of the clamped parts, default = thread_material or steel)."""
    f = fastener(p.get("size"))
    if not f:
        return {"ok": False, "findings": [{"level": "error", "code": "unknown_size",
                                           "message": f"unknown bolt size {p.get('size')!r}"}]}
    d = f["d"]
    series = p.get("series") if p.get("series") in SERIES else "medium"
    findings = _hole_findings(f, p.get("hole_d"), p.get("hole_kind", "clearance"), series)
    lower_kind = p.get("lower_hole_kind")
    if p.get("lower_hole_d") is not None and lower_kind in ("tapped", "clearance"):
        findings += _hole_findings(f, p.get("lower_hole_d"), lower_kind, series, "lower_")
    nut = bool(p.get("nut")) or lower_kind == "clearance"
    L, clamp = p.get("bolt_length_mm"), p.get("clamp_mm")
    wh = 0.0
    if L is not None and clamp is not None:
        L, clamp = float(L), float(clamp)
        washers = int(p.get("washers") or 0)
        wsh = f.get("washer") or {}
        wh = wsh.get("h", 0.0) * washers
        if nut:
            m = (f.get("nut") or {}).get("m") or d
            need = clamp + wh + m + 2 * f["pitch"]
            if L < need - 1e-9:
                findings.append({"level": "error", "code": "nut_too_short", "message":
                                 f"bolt {L:g} mm is too short: clamp {clamp:g} + washers {wh:g} + nut {m:g} + "
                                 f"2 thread pitches = {need:.1f} mm", "fix": _next_length(need, f),
                                 "fix_value": _next_value(need, f), "source": SOURCES["nut"]})
        else:
            mat = str(p.get("thread_material") or "steel").lower()
            factor = ENGAGEMENT_D.get(mat, 1.5)
            need_eng = factor * d
            eng = L - clamp - wh
            if eng < need_eng - 1e-9:
                findings.append({"level": "error", "code": "engagement_short", "message":
                                 f"thread engagement {eng:.1f} mm < {factor:g}·d = {need_eng:.1f} mm in {mat}",
                                 "fix": _next_length(clamp + wh + need_eng, f),
                                 "fix_value": _next_value(clamp + wh + need_eng, f), "source": SOURCES["engagement"]})
            depth = p.get("thread_depth_mm")
            if depth is not None and eng > float(depth) + 1e-9:
                findings.append({"level": "error", "code": "too_long_for_thread", "message":
                                 f"bolt reaches {eng:.1f} mm into the tapped hole but the thread is only {float(depth):g} mm deep",
                                 "fix": "shorter bolt or deeper thread",
                                 "fix_value": _prev_value(clamp + wh + float(depth), f), "source": "geometry"})
    out = {"ok": not any(x["level"] == "error" for x in findings), "bolt": f["size"], "findings": findings,
           "fastener": f}
    if p.get("load_n") is not None:
        st = joint_strength(f, p, nut, wh)
        out["strength"] = st
        findings.extend(st.pop("findings"))
        out["ok"] = not any(x["level"] == "error" for x in findings)
    return out


def joint_strength(f: dict, p: dict, nut: bool, washer_h: float = 0.0) -> dict:
    """Screening strength of one bolt in tension: Shigley 10e ch. 8 (same equations as the gearbox cover joints,
    physics/bolted_joint.py): A_t (eq. 8-1), k_b (8-17), frustum k_m (8-20), C (8-24), F_i = 0.75 F_p (8-31) with
    +/-25 % torque scatter, n_p yield (8-28 with F_i,max), n_0 separation (8-30 with F_i,min)."""
    from physics import bolted_joint as bj
    import math
    findings = []
    cls = str(p.get("property_class") or "8.8")
    if cls not in bj.CLASSES:
        return {"label": "not computed", "findings": [{"level": "warning", "code": "strength_not_valid",
                "message": f"property class {cls} unknown (use 8.8, 10.9 or 12.9)"}]}
    clamp = p.get("clamp_mm")
    cmat = str(p.get("clamp_material") or p.get("thread_material") or "steel").lower()
    bmat = str(p.get("thread_material") or cmat).lower()
    if cmat in PLASTICS or (not nut and bmat in PLASTICS):
        return {"label": "not computed", "findings": [{"level": "warning", "code": "strength_not_valid",
                "message": "bolt preload method is not valid for plastic parts (creep, crushing); "
                           "use a load test or inserts", "source": "Shigley ch. 8 assumptions"}]}
    if clamp is None:
        return {"label": "not computed", "findings": [{"level": "warning", "code": "strength_not_valid",
                "message": "clamp_mm is needed for the joint stiffness"}]}
    d, P = f["d"], f["pitch"]
    Sp = bj.CLASSES[cls][0]
    At = bj.tensile_area(d, P)
    Fp = Sp * At
    Fi = 0.75 * Fp
    Fi_max, Fi_min = 1.25 * Fi, 0.75 * Fi
    grip = float(clamp) + washer_h
    Ec, Eb = E_MPA.get(cmat, 207000.0), E_MPA.get(bmat, 207000.0)
    if nut:
        l_eff, layers = grip, [(grip, Ec)]
    else:
        l_eff, layers = grip + d / 2.0, [(grip, Ec), (d / 2.0, Eb)]       # tapped hole: Shigley Table 8-7
    kb = At * bj.E_STEEL / l_eff
    km = bj.member_stiffness(d, layers)
    C = kb / (kb + km)
    Pn = float(p.get("load_n"))
    n_p = Sp * At / (Fi_max + C * Pn)
    n_0 = Fi_min / (Pn * (1 - C)) if Pn > 1e-9 else math.inf
    if n_p < 1.0:
        findings.append({"level": "error", "code": "strength_yield", "message":
                         f"bolt yield safety {n_p:.2f} < 1.0 (class {cls}, load {Pn:g} N)",
                         "fix": "larger bolt, higher class or more bolts", "source": "Shigley 10e eq. 8-28"})
    if n_0 < 2.0:
        findings.append({"level": "error" if n_0 < 1.0 else "warning", "code": "strength_separation", "message":
                         f"separation safety {n_0:.2f} < 2.0: the joint may open under {Pn:g} N",
                         "fix": "more preload (higher class / larger bolt) or more bolts",
                         "source": "Shigley 10e eq. 8-30"})
    return {"label": "screening", "method": "Shigley 10e ch. 8 (eq. 8-1, 8-17, 8-20, 8-24, 8-28, 8-30, 8-31); "
                                            "preload scatter +/-25 % (torque wrench)",
            "property_class": cls, "tensile_area_mm2": round(At, 2), "proof_load_n": round(Fp, 0),
            "preload_n": round(Fi, 0), "preload_min_n": round(Fi_min, 0), "preload_max_n": round(Fi_max, 0),
            "bolt_stiffness_n_mm": round(kb, 0), "member_stiffness_n_mm": round(km, 0), "joint_constant": round(C, 4),
            "load_n": Pn, "sf_yield": round(n_p, 3), "sf_separation": round(n_0, 3) if math.isfinite(n_0) else None,
            "tightening_torque_nm": round(0.2 * Fi * d / 1000.0, 2), "findings": findings,
            "note": "Screening until checked against a published worked example."}


def _lengths(f):
    return (f.get("socket_head") or {}).get("lengths") or list(SHCS_LENGTHS)


def _next_value(need: float, f: dict):
    return next((x for x in _lengths(f) if x >= need - 1e-9), None)


def _prev_value(limit: float, f: dict):
    return next((x for x in reversed(_lengths(f)) if x <= limit + 1e-9), None)


def _next_length(need: float, f: dict | None = None) -> str:
    L = _next_value(need, f or {})
    return f"use a {L:g} mm bolt (ISO 4762 standard length)" if L else "no standard length long enough"


def index() -> dict:
    return {"fasteners": sorted(METRIC, key=lambda s: float(s[1:])), "bearings": list(BEARINGS),
            "series": SERIES, "engagement_rule_d": ENGAGEMENT_D, "sources": SOURCES}
