"""Stage 5: joining and manufacturing — a design that can be built in a workshop (v0.42).

Arya, 21 Nov, comparing Zookeeper: the parts looked like a product but could not be made — the tube and the plate
passed through each other, the tube did not reach the blade, nothing said how each part is cut or welded. This
module answers the questions a workshop asks, and every answer is either measured or labelled:

  * `stock_for` says how a part is made: laser/plasma cut plate of a standard thickness, a tube/profile from a
    sourced table, a machined block, a casting, a print, or a purchased standard part. Nothing silently assumes a
    process: a part without one says so and is asked for.
  * `analyse` checks the assembly geometry: overlap volume between parts (must be 0 inside the mesh tolerance),
    contact gap at every welded joint (must fit up), bolt holes on the same axis, and a part that hangs in the air
    with no joint. The overlap is measured from each part's DECLARED bounding box, so it is labelled an estimate,
    never "kernel geometry"; and a shift that would clear it is a PROPOSAL the engineer accepts (`fix=False`), so
    a tube is never quietly moved away from the blade it was meant to touch.
  * `joints_for` writes one joint per interface: a fillet weld whose minimum leg comes from AWS D1.1 Table 7.7 as
    already implemented in `physics.dfm_rules`, a bolted joint with the ISO 273 clearance hole from the catalogue,
    a pin, a press fit or a slot-tab — each with its reason and the features the parts still need.
  * `outputs` writes the BOM, the cut list, the weld list, the bolt list and the assembly order.

Truth rules carried from the rest of IDRIS: a weld leg, a clearance hole and a bolt length come from a table with its
source; a load is never assumed, so a joint strength is "Not checked by IDRIS" until the engineer gives one; and a
bolt size nobody stated is M8 as an ASSUMPTION in `assumptions`, with the Stage-5 question that asks for the real one
(ISO 273 is the source of the clearance hole, never of the M8 choice).
Geometry here is the part's declared shape (a box or a cylinder with its position). A curved blade is approximated by
its bounding box, so an overlap volume on a cylinder is a conservative bound, not the exact intersection — the note in
the answer says so.
"""
from __future__ import annotations

from physics import catalog
from physics import dfm_rules

FIT_TOL_MM = 0.5                 # fit-up tolerance at a welded joint (0..0.5 mm gap)
OVERLAP_TOL_MM3 = 1.0            # mesh/decimal tolerance: below this an overlap counts as none
BOLT_AXIS_TOL_MM = 0.5
ASSUMED_BOLT = "M8"              # only when neither the engineer nor the plan gives a size, and it says so
ASSUMED_BOLT_NOTE = "assumed M8 — no load given, change anytime"
OVERLAP_SOURCE = "bounding-box estimate (not kernel geometry)"
BOLT_QUESTION = {"id": "q_bolt_size", "text": "Bolt size?", "stage": 5,
                 "options": [{"label": "M8 — recommended", "recommended": True,
                              "note": "ISO 273 medium clearance 9 mm; move up if the load says so"},
                             {"label": "M10", "note": "ISO 273 medium clearance 11 mm"},
                             {"label": "Not sure — you choose", "assumed": True}]}

# Standard mill plate thicknesses (mm). A workshop stock list replaces this one; the answer says which was used.
PLATE_MM = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
PLATE_SOURCE = ("IDRIS plate stock list: the standard mill thicknesses 1-25 mm a laser/plasma shop keeps. "
                "A workshop's own stock list replaces it.")
# EN 10219-2 square/rectangular hollow sections: (a x b x t). Only the sizes in this table are offered.
SHS = {(40, 40, 3), (50, 50, 3), (60, 60, 3), (40, 40, 2.5), (50, 50, 2.5), (80, 40, 3), (80, 80, 4), (100, 50, 3)}
SHS_SOURCE = "EN 10219-2 cold-formed square/rectangular hollow sections (the sizes in this table)."
DENSITY_KG_M3 = {"steel": 7850.0, "s235": 7850.0, "s355": 7850.0, "stainless": 7900.0, "aluminium": 2700.0,
                 "aluminum": 2700.0, "brass": 8500.0, "nylon": 1140.0, "pla": 1240.0, "petg": 1270.0}
DENSITY_SOURCE = "Density of the declared material (steel 7850 kg/m3, aluminium 2700, brass 8500; polymer 1140-1270)."
PROCESSES = ("laser_cut", "plasma_cut", "press_brake", "tube_cut", "machined", "cast", "print_3d", "purchased")


# ------------------------------------------------------------------ geometry (declared shapes, mm)
def _aabb(part: dict):
    """-> (min[3], max[3]) or None. A box is exact; a cylinder uses its bounding box (a conservative bound)."""
    sh = part.get("shape") if isinstance(part.get("shape"), dict) else {}
    at = sh.get("at") or part.get("at") or [0.0, 0.0, 0.0]
    try:
        at = [float(at[0]), float(at[1]), float(at[2])]
    except (TypeError, ValueError, IndexError):
        return None
    kind = str(sh.get("type") or "box").lower()
    if kind in ("box", "plate", "block"):
        try:
            s = [float(x) for x in (sh.get("size") or [])][:3]
        except (TypeError, ValueError):
            return None
        if len(s) != 3 or min(s) <= 0:
            return None
        return ([at[i] - s[i] / 2 for i in range(3)], [at[i] + s[i] / 2 for i in range(3)])
    if kind in ("cylinder", "tube", "rod", "pipe"):
        try:
            d, l = float(sh.get("d")), float(sh.get("l"))
        except (TypeError, ValueError):
            return None
        if d <= 0 or l <= 0:
            return None
        axis = str(sh.get("axis") or "x").lower()[:1]
        half = [d / 2, d / 2, d / 2]
        half["xyz".index(axis)] = l / 2
        return ([at[i] - half[i] for i in range(3)], [at[i] + half[i] for i in range(3)])
    return None


def _inter(a, b):
    lo = [max(a[0][i], b[0][i]) for i in range(3)]
    hi = [min(a[1][i], b[1][i]) for i in range(3)]
    return lo, hi


def overlap_volume(a: dict, b: dict) -> float:
    """Intersection volume in mm3 (0 when the parts do not overlap)."""
    ba, bb = _aabb(a), _aabb(b)
    if not ba or not bb:
        return 0.0
    lo, hi = _inter(ba, bb)
    vol = 1.0
    for i in range(3):
        d = hi[i] - lo[i]
        if d <= 0:
            return 0.0
        vol *= d
    return vol


def contact_gap(a: dict, b: dict) -> float | None:
    """Distance between the two declared volumes along the axis that separates them, or None when they overlap."""
    ba, bb = _aabb(a), _aabb(b)
    if not ba or not bb:
        return None
    # positive on an axis = the two volumes are apart on that axis; 0 = they touch or overlap
    gaps = [max(ba[0][i], bb[0][i]) - min(ba[1][i], bb[1][i]) for i in range(3)]
    return max(0.0, max(gaps))


def contact_area(a: dict, b: dict) -> float:
    ba, bb = _aabb(a), _aabb(b)
    if not ba or not bb:
        return 0.0
    dims = []
    for i in range(3):
        d = min(ba[1][i], bb[1][i]) - max(ba[0][i], bb[0][i])
        dims.append(max(0.0, d))
    dims.sort(reverse=True)
    return dims[0] * dims[1]


def _thickness(part: dict) -> float:
    ba = _aabb(part)
    if not ba:
        return 0.0
    return min(ba[1][i] - ba[0][i] for i in range(3))


def volume_mm3(part: dict) -> float:
    sh = part.get("shape") if isinstance(part.get("shape"), dict) else {}
    if str(sh.get("type") or "").lower() in ("cylinder", "tube", "rod", "pipe"):
        try:
            d, l = float(sh.get("d")), float(sh.get("l"))
            wall = float(sh.get("wall"))
            inner = max(0.0, d - 2 * wall)
            return 3.141592653589793 / 4 * (d * d - inner * inner) * l
        except (TypeError, ValueError):
            pass
    ba = _aabb(part)
    if not ba:
        return 0.0
    return (ba[1][0] - ba[0][0]) * (ba[1][1] - ba[0][1]) * (ba[1][2] - ba[0][2])


def mass_kg(part: dict) -> float:
    mat = str(part.get("material") or "steel").lower()
    rho = DENSITY_KG_M3.get(mat, DENSITY_KG_M3["steel"])
    return volume_mm3(part) / 1e9 * rho


# ------------------------------------------------------------------ stock and process (A1)
def stock_for(part: dict) -> dict:
    """How this part is made. Never a silent default: a part without a process is reported as one."""
    name = str(part.get("name") or "part")
    declared = str(part.get("process") or "").strip().lower().replace("-", "_").replace(" ", "_")
    sh = part.get("shape") if isinstance(part.get("shape"), dict) else {}
    kind = str(sh.get("type") or "box").lower()
    thick = _thickness(part)
    if declared in ("purchased", "bought", "standard"):
        return {"part": name, "process": "purchased", "stock": {"kind": "purchased"},
                "source": "declared a purchased standard part", "needs": ""}
    if kind in ("cylinder", "tube", "pipe", "rod"):
        try:
            d = round(float(sh.get("d")))
            wall = round(float(sh.get("wall") or 0))
        except (TypeError, ValueError):
            d, wall = 0, 0
        if (d, d, wall) in SHS or (d, wall) in {(40, 3), (50, 3), (60, 3)}:
            profile = f"SHS {d}x{d}x{wall}"
            return {"part": name, "process": declared if declared in PROCESSES else "tube_cut",
                    "stock": {"kind": "tube", "profile": profile, "source": SHS_SOURCE},
                    "source": SHS_SOURCE, "needs": ""}
        return {"part": name, "process": declared if declared in PROCESSES else "tube_cut",
                "stock": {"kind": "tube", "profile": f"round tube d{d} x {wall}", "source": "declared tube size"},
                "source": "declared tube size", "needs": "" if wall else
                "the wall thickness is not stated, so the cut length cannot become a stock order"}
    if thick and 0 < thick <= max(PLATE_MM) and ba_dim(part) > 0:
        chosen = max([t for t in PLATE_MM if t <= thick + 1e-9] or [PLATE_MM[0]])
        return {"part": name, "process": declared if declared in PROCESSES else "laser_cut",
                "stock": {"kind": "plate", "thickness_mm": chosen, "source": PLATE_SOURCE},
                "source": PLATE_SOURCE,
                "needs": "" if declared else "the workshop process was not stated (laser, plasma, press brake?)"}
    # no table stock matches the shape: a machined block is ASSUMED, and the answer carries that as an assumption
    # (it used to be written only in the source line, where nobody reads it)
    assumed = "no plate/tube stock matches this shape; a machined block is assumed"
    return {"part": name, "process": declared or "machined",
            "stock": {"kind": "block", "source": assumed}, "source": assumed, "assumed": assumed,
            "needs": "" if declared else "the workshop process was not stated"}


def ba_dim(part):
    ba = _aabb(part)
    return (ba[1][0] - ba[0][0]) if ba else 0.0


# ------------------------------------------------------------------ joints (A2)
def weld_joint(a: dict, b: dict, area_mm2: float = 0.0) -> dict:
    t = max(_thickness(a), _thickness(b))
    leg = dfm_rules.weld_fillet_min_mm(t)
    length = length_mm(a, b, area_mm2)
    return {"between": [a.get("name"), b.get("name")], "kind": "fillet_weld",
            "reason": "two metal parts touch; a fillet weld joins them without a fastener",
            "features_required": ["both faces fit up with no gap (0-0.5 mm)",
                                  "torch access to the joint on both sides where the drawing asks"],
            "size": {"value": leg, "unit": "mm", "what": "minimum fillet leg (thickest base metal)",
                     "source": ("AWS D1.1 Table 7.7 minimum fillet leg, via physics.dfm_rules.WELD_MIN_FILLET_MM "
                                f"for {t:.1f} mm base metal")},
            "length_mm": length}


def length_mm(a: dict, b: dict, area_mm2: float = 0.0) -> float:
    if area_mm2 > 0:
        # a fillet runs around the contact: its length is at least the perimeter of the shared face
        side = area_mm2 ** 0.5
        return round(4 * side, 1)
    return round(contact_area(a, b) ** 0.5 * 4, 1)


def clearance_hole_mm(size: str) -> tuple:
    """ISO 273 medium clearance hole for that bolt, straight from the catalogue."""
    f = catalog.fastener(size)
    if not f:
        return (None, None)
    med = (f.get("clearance") or {}).get("medium")
    return (med, f.get("d"))


def bolted_joint(a: dict, b: dict, size: str | None = None) -> dict:
    """A bolted joint. `size` is the engineer's (or the plan's) bolt; without one the joint says M8 is assumed.

    ISO 273 is the source of the CLEARANCE HOLE, never of the M8 choice: that choice is an assumption until the
    engineer gives a size or a load, and it is listed as one (v0.42 review, item c).
    """
    given = bool(str(size or "").strip())
    size = str(size or ASSUMED_BOLT).strip()
    hole, d = clearance_hole_mm(size)
    clamp = _thickness(a) + _thickness(b)
    out = {"between": [a.get("name"), b.get("name")], "kind": "bolted",
           "reason": "the two parts are assembled on site, so the joint can be taken apart",
           "features_required": [f"a {hole} mm clearance hole (ISO 273 medium) through both parts, on one axis",
                                 "edge distance at least 1.5 d from the hole centre to the part edge"],
           "size": {"value": hole, "unit": "mm", "what": f"clearance hole for {size}",
                    "source": "ISO 273 medium series, physics.catalog"},
           "bolt": {"size": size, "d_mm": d, "clamp_mm": round(clamp, 1),
                    "length_mm": _bolt_length(size, clamp),
                    "source": "ISO 273 clearance + catalogue bolt length above the clamp thickness"}}
    if given:
        out["bolt_selection"] = {"size": size, "source": "given by the engineer"}
    else:
        out["bolt_selection"] = {"size": size, "source": ASSUMED_BOLT_NOTE}
        out["assumed_size"] = True
        out["assumptions"] = [{"about": f"{a.get('name')} + {b.get('name')}", "what": f"bolt size {size}",
                               "source": ASSUMED_BOLT_NOTE}]
    return out


def _bolt_length(size: str, clamp: float) -> float | None:
    """Shortest catalogue length that clamps the stack plus a nut (ISO 4017/4762 lengths)."""
    f = catalog.fastener(size) or {}
    lengths = f.get("lengths") or f.get("length_mm") or ()
    need = clamp + (f.get("nut_height") or (f.get("nut") or {}).get("m") or 0) + 2 * ((f.get("washer") or {}).get("h") or 0)
    for L in sorted(x for x in lengths if isinstance(x, (int, float))):
        if L >= need:
            return L
    return None


def _has_hole(part: dict) -> bool:
    return bool(part.get("holes"))


def _holes_align(a: dict, b: dict) -> bool:
    """A bolt needs a hole in BOTH parts, on one axis: the declared hole centres must coincide."""
    for ha in (a.get("holes") or []):
        for hb in (b.get("holes") or []):
            try:
                pa = ha.get("at") or ha.get("center")
                pb = hb.get("at") or hb.get("center")
                dist = sum((float(pa[i]) - float(pb[i])) ** 2 for i in range(3)) ** 0.5
            except (TypeError, ValueError, IndexError):
                continue
            if dist <= BOLT_AXIS_TOL_MM and str(ha.get("axis") or "z") == str(hb.get("axis") or "z"):
                return True
    return False


def joints_for(parts: list, interfaces=None, bolt_size: str | None = None) -> list:
    """One joint per touching interface, plus the interfaces the plan already declared."""
    by = {str(p.get("name")): p for p in parts if isinstance(p, dict) and p.get("name")}
    joints, seen = [], set()

    def add(j):
        key = (tuple(sorted([str(j["between"][0]), str(j["between"][1])])), j["kind"])
        if key not in seen:
            seen.add(key)
            joints.append(j)

    for i in (interfaces or []):
        if not isinstance(i, dict):
            continue
        pair = i.get("between") or []
        if len(pair) != 2 or pair[0] not in by or pair[1] not in by:
            continue
        a, b = by[pair[0]], by[pair[1]]
        kind = str(i.get("kind") or "contact")
        if kind in ("bolt", "bolted"):
            add(bolted_joint(a, b, (i.get("params") or {}).get("size") or bolt_size))
        elif kind in ("weld", "fillet_weld", "las"):
            add(weld_joint(a, b, contact_area(a, b)))
        elif kind in ("press", "press_fit", "pinned", "pin", "slot_tab", "contact"):
            joints.append({"between": [a.get("name"), b.get("name")],
                           "kind": {"press": "press_fit", "pin": "pinned"}.get(kind, kind),
                           "reason": "declared by the assembly plan", "features_required": [], "size": None})
            seen.add((tuple(sorted([str(a.get("name")), str(b.get("name"))])), kind))
    names = [str(p.get("name")) for p in parts if isinstance(p, dict) and p.get("name")]
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            a, b = by[na], by[nb]
            if overlap_volume(a, b) > OVERLAP_TOL_MM3:
                continue
            gap = contact_gap(a, b)
            if gap is None or gap > FIT_TOL_MM:
                continue
            if _has_hole(a) and _has_hole(b) and _holes_align(a, b):
                add(bolted_joint(a, b, bolt_size))
            else:
                add(weld_joint(a, b, contact_area(a, b)))
    return joints


# ------------------------------------------------------------------ the assembly check (A3)
def _shift(part: dict, axis: int, by: float) -> dict:
    out = {**part, "shape": {**(part.get("shape") or {})}}
    at = list((part.get("shape") or {}).get("at") or part.get("at") or [0.0, 0.0, 0.0])
    at[axis] = float(at[axis]) + by
    out["shape"]["at"] = at
    return out


def _fix_overlap(a: dict, b: dict) -> tuple:
    """The shift that would clear the overlap of b out of a. -> (shifted_b, proposal dict or None).

    Nothing is moved here: `analyse` reports the proposal and only applies it when the engineer accepts it.
    """
    ba, bb = _aabb(a), _aabb(b)
    if not ba or not bb:
        return b, None
    lo, hi = _inter(ba, bb)
    pen = [hi[i] - lo[i] for i in range(3)]
    axis = min(range(3), key=lambda i: pen[i])
    if pen[axis] <= 0:
        return b, None
    # push b clear of a on that axis: a part that sits INSIDE the other needs more than the penetration depth,
    # so the shift is measured to the far face and the two then touch instead of passing through
    a_c = (ba[0][axis] + ba[1][axis]) / 2
    b_c = (bb[0][axis] + bb[1][axis]) / 2
    if b_c >= a_c:
        by = ba[1][axis] - bb[0][axis] + 1e-6
    else:
        by = -(bb[1][axis] - ba[0][axis] + 1e-6)
    why = (f"{b.get('name')} moved {'+' if by > 0 else ''}{by:.1f} mm on {'xyz'[axis]} would clear it "
           f"(only if the engineer accepts — the parts may be meant to touch)")
    return _shift(b, axis, by), {"part": b.get("name"), "with": a.get("name"), "axis": "xyz"[axis],
                                 "shift_mm": round(by, 3), "why": why}


def analyse(parts: list, joints=None, fix: bool = False) -> dict:
    """Overlap, contact at every welded joint, aligned bolt holes, no floating part. -> checks + proposals.

    `fix=False` (the default) is the honest one: an overlap becomes a `proposed_fixes` entry the engineer accepts,
    because shifting a part on our own can pull a tube away from the curved blade it was designed to meet. `fix=True`
    applies those shifts and reports them in `fixes`.
    """
    parts = [p for p in (parts or []) if isinstance(p, dict) and p.get("name")]
    by = {str(p["name"]): p for p in parts}
    issues, fixes, proposals = [], [], []
    names = list(by)
    for _pass in range(6 if fix else 1):                     # moving one part can bring it near another
        moved = False
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                v = overlap_volume(by[na], by[nb])
                if v <= OVERLAP_TOL_MM3:
                    continue
                new_b, prop = _fix_overlap(by[na], by[nb])
                if not prop:
                    continue
                prop = {**prop, "overlap_mm3": round(v, 1), "accepted": bool(fix)}
                proposals.append(prop)
                if not fix:
                    continue
                by[nb] = new_b
                moved = True
                what = (f"{prop['part']} moved {'+' if prop['shift_mm'] > 0 else ''}{prop['shift_mm']:.1f} mm "
                        f"on {prop['axis']}")
                fixes.append(f"{what}; overlap {v:.0f} mm3 removed")
                issues.append({"level": "fixed", "code": "overlap",
                               "parts": [na, nb], "volume_mm3": round(v, 1), "fix": what})
        if not moved:
            break
    if not fix:
        issues = [{"level": "proposed", "code": "overlap", "parts": [x["with"], x["part"]],
                   "volume_mm3": x["overlap_mm3"], "fix": x["why"]} for x in proposals]
    checks = []
    names = list(by)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            v = overlap_volume(by[na], by[nb])
            checks.append({"check": "overlap", "parts": [na, nb], "value_mm3": round(v, 1),
                           "status": "pass" if v <= OVERLAP_TOL_MM3 else "fail",
                           "source": OVERLAP_SOURCE, "fidelity": "estimate"})
    floating = [n for n in names
                if not any(j for j in (joints or []) if n in (j.get("between") or []))]
    checks.append({"check": "every part is joined", "value": len(floating), "status": "pass" if not floating else "fail",
                   "parts": floating, "source": "joint list: a part with no joint hangs in the air"})
    for j in (joints or []):
        if j.get("kind") != "fillet_weld":
            continue
        a, b = by.get(str(j["between"][0])), by.get(str(j["between"][1]))
        if not a or not b:
            continue
        gap = contact_gap(a, b) or 0.0
        checks.append({"check": "contact at the weld", "parts": j["between"], "value_mm": round(gap, 2),
                       "limit_mm": FIT_TOL_MM, "status": "pass" if gap <= FIT_TOL_MM else "fail",
                       "source": "fit-up tolerance for a fillet weld (0-0.5 mm gap)"})
    for j in (joints or []):
        if j.get("kind") != "bolted":
            continue
        a, b = by.get(str(j["between"][0])), by.get(str(j["between"][1]))
        if not a or not b:
            continue
        ok = _has_hole(a) and _has_hole(b) and _holes_align(a, b)
        checks.append({"check": "bolt holes on one axis", "parts": j["between"], "status": "pass" if ok else "fail",
                       "source": "the two hole centres must coincide within 0.5 mm and share an axis"})
    failed = [c for c in checks if c["status"] == "fail"]
    return {"parts": [by[n] for n in names], "checks": checks, "fixes": fixes, "issues": issues,
            "proposed_fixes": proposals, "ok": not failed, "failed": failed, "floating": floating}


# ------------------------------------------------------------------ production outputs (A4)
def outputs(parts: list, joints: list) -> dict:
    parts = [p for p in (parts or []) if isinstance(p, dict) and p.get("name")]
    bom = []
    for p in parts:
        st = stock_for(p)
        bom.append({"part": p.get("name"), "process": st["process"], "stock": st["stock"].get("kind"),
                    "stock_detail": st["stock"].get("profile") or st["stock"].get("thickness_mm"),
                    "qty": int(p.get("qty") or 1), "mass_kg": round(mass_kg(p), 3),
                    "material": p.get("material") or "steel", "source": st["source"]})
    cut = []
    for p in parts:
        st = stock_for(p)
        ba = _aabb(p)
        if not ba:
            continue
        if st["stock"].get("kind") == "plate":
            cut.append({"part": p.get("name"), "stock": f"plate {st['stock']['thickness_mm']} mm",
                        "length_mm": round(ba[1][0] - ba[0][0], 1), "width_mm": round(ba[1][1] - ba[0][1], 1),
                        "angle_deg": 90, "qty": int(p.get("qty") or 1)})
        elif st["stock"].get("kind") == "tube":
            ax = str((p.get("shape") or {}).get("axis") or "x").lower()[:1]
            i = "xyz".index(ax) if ax in "xyz" else 0
            cut.append({"part": p.get("name"), "stock": st["stock"].get("profile"),
                        "length_mm": round(ba[1][i] - ba[0][i], 1), "angle_deg": 90, "qty": int(p.get("qty") or 1)})
    welds = [{"between": j["between"], "kind": j["kind"], "leg_mm": (j.get("size") or {}).get("value"),
              "length_mm": j.get("length_mm"), "source": (j.get("size") or {}).get("source")}
             for j in joints if j.get("kind") in ("fillet_weld", "butt_weld")]
    bolts = [{"between": j["between"], **(j.get("bolt") or {})} for j in joints if j.get("kind") == "bolted"]
    order = []
    if welds:
        order.append({"step": len(order) + 1, "what": "weld the sub-assemblies",
                      "detail": "; ".join(f"{w['between'][0]} + {w['between'][1]}" for w in welds[:6])})
    if bolts:
        order.append({"step": len(order) + 1, "what": "bolt the sub-assemblies together",
                      "detail": "; ".join(f"{b['between'][0]} + {b['between'][1]}" for b in bolts[:6])})
    order.append({"step": len(order) + 1, "what": "final check",
                  "detail": "check the diagonal and the fit on the machine before painting"})
    return {"bom": bom, "cut_list": cut, "welds": welds, "bolts": bolts, "assembly_order": order,
            "mass_total_kg": round(sum(b["mass_kg"] * b["qty"] for b in bom), 3)}


def plan(parts: list, interfaces=None, bolt_size: str | None = None, fix: bool = False) -> dict:
    """The whole stage: joints, geometry check (with the shifts it proposes), then the workshop documents.

    `fix=False` is the default: an overlap is reported with the shift that would clear it, and the engineer decides.
    A bolt size that neither the engineer nor the plan gives is M8 AS AN ASSUMPTION, listed in `assumptions` with
    the Stage-5 question that asks for the real one (v0.42 review, items b and c).
    """
    # geometry first, joints second: a tube that passed through a plate only touches it after the shift is accepted,
    # and a joint list written before that would say the part hangs in the air
    first = analyse(parts, [], fix)
    fixed = first["parts"]
    joints = joints_for(fixed, interfaces, bolt_size)
    chk = analyse(fixed, joints, fix=False)
    chk["fixes"], chk["issues"] = first["fixes"], first["issues"]
    chk["proposed_fixes"] = first["proposed_fixes"]
    out = outputs(chk["parts"], joints)
    assumptions = []
    for p in chk["parts"]:
        st = stock_for(p)
        if st.get("assumed"):
            assumptions.append({"about": p.get("name"), "what": st["assumed"], "source": st["source"]})
    for j in joints:
        assumptions.extend(j.get("assumptions") or [])
    strength = [{"joint": j["between"], "status": "not_checked",
                 "why": "no load was given by the engineer, so IDRIS does not claim a joint strength"}
                for j in joints if j.get("kind") in ("fillet_weld", "bolted")]
    return {"ok": chk["ok"], "joints": joints, "checks": chk["checks"], "fixes": chk["fixes"],
            "proposed_fixes": chk["proposed_fixes"], "issues": chk["issues"], "failed": chk["failed"],
            "assumptions": assumptions,
            "questions": [BOLT_QUESTION] if any(j.get("assumed_size") for j in joints) else [],
            "parts": [{"name": p.get("name"), "process": stock_for(p)["process"],
                       "mass_kg": round(mass_kg(p), 3)} for p in chk["parts"]],
            **out, "joint_strength": strength,
            "notes": ["Overlap volume uses each part's declared volume; on a curved part the declared box is a "
                      "conservative bound, not the exact intersection.",
                      "A weld leg, a clearance hole and a bolt length come from their tables; a joint strength "
                      "needs a load from the engineer and is otherwise Not checked by IDRIS."],
            "process_needs": [n for n in (stock_for(p).get("needs") for p in chk["parts"]) if n]}
