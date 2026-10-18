"""2D working drawings (SVG, A3) of the manufacturable gearbox (Tahap 1.4).

Outlines are TRUE sections of the 3D parts (plane cuts of the same meshes that go into the STL/STEP),
so the drawing cannot disagree with the model. Dimensions, fits (ISO 286) and notes come from the
layout and the solver results. Sheets: housing (two sections), covers, shafts.
"""
import math
from .iso2768 import note as iso2768_note
import numpy as np
from .iso286 import label as fit_label
from .bolted_joint import METRIC

A3 = (420.0, 297.0)
SCALES = (1.0, 0.5, 0.4, 0.25, 0.2)               # 1:1, 1:2, 1:2.5, 1:4, 1:5
INK, DIM, CUT = "#111", "#1f4e9c", "#8a8a8a"


def section(parts, axis, value, names=None):
    """Plane cut of the given parts: axis 0/1/2 = x/y/z constant. Returns [(part, (p, q))] with 3D points."""
    segs = []
    for p in parts:
        if names and p["name"] not in names and p["kind"] not in names:
            continue
        V = np.reshape(p["verts"], (-1, 3))
        F = np.reshape(p["faces"], (-1, 3))
        T = V[F]
        d = T[:, :, axis] - value
        for tri, dd in zip(T, d):
            pts = []
            for i in range(3):
                j = (i + 1) % 3
                if dd[i] * dd[j] < 0:
                    s = dd[i] / (dd[i] - dd[j])
                    pts.append(tri[i] + s * (tri[j] - tri[i]))
            if len(pts) == 2:
                segs.append((p, (pts[0], pts[1])))
    # faces shared by two touching solids (e.g. strips of a plate) cut to the same segment twice: they are
    # internal, not edges of the part — drop both copies
    key = lambda a, b: tuple(sorted((tuple(np.round(a, 3)), tuple(np.round(b, 3)))))
    count = {}
    for p_, (a, b) in segs:
        count[key(a, b)] = count.get(key(a, b), 0) + 1
    return [(p_, (a, b)) for p_, (a, b) in segs if count[key(a, b)] == 1]


class Sheet:
    def __init__(self, title, subtitle, info):
        self.title, self.subtitle, self.info = title, subtitle, info
        self.body = []

    def line(self, x1, y1, x2, y2, color=INK, w=0.35, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
                         f'stroke-width="{w}"{d}/>')

    def text(self, x, y, s, size=3.0, color=INK, anchor="start", rot=0):
        tr = f' transform="rotate({rot} {x:.2f} {y:.2f})"' if rot else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.body.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Helvetica, Arial, sans-serif" '
                         f'font-size="{size}" fill="{color}" text-anchor="{anchor}"{tr}>{s}</text>')

    def arrow(self, x, y, ang):
        a = math.radians(ang)
        for da in (math.radians(160), -math.radians(160)):
            self.line(x, y, x + 2.5 * math.cos(a + da), y + 2.5 * math.sin(a + da), DIM, 0.25)

    def dim(self, p, q, off, text, vertical=False):
        """Linear dimension between sheet points p and q, offset `off` mm perpendicular to it."""
        (x1, y1), (x2, y2) = p, q
        if vertical:
            xd = x1 + off
            self.line(x1, y1, xd + math.copysign(1.5, off), y1, DIM, 0.18)
            self.line(x2, y2, xd + math.copysign(1.5, off), y2, DIM, 0.18)
            self.line(xd, y1, xd, y2, DIM, 0.25)
            self.arrow(xd, y1, 90 if y2 > y1 else -90)
            self.arrow(xd, y2, -90 if y2 > y1 else 90)
            self.text(xd - 1.2, (y1 + y2) / 2, text, 2.8, DIM, "middle", -90)
        else:
            yd = y1 + off
            self.line(x1, y1, x1, yd + math.copysign(1.5, off), DIM, 0.18)
            self.line(x2, y2, x2, yd + math.copysign(1.5, off), DIM, 0.18)
            self.line(x1, yd, x2, yd, DIM, 0.25)
            self.arrow(x1, yd, 0 if x2 < x1 else 180)
            self.arrow(x2, yd, 180 if x2 < x1 else 0)
            self.text((x1 + x2) / 2, yd - 1.2, text, 2.8, DIM, "middle")

    def leader(self, x, y, tx, ty, text):
        """Numbered callout: a short leader to a balloon; the text goes to the notes list of the sheet
        (no overlapping labels on the views)."""
        self.notes = getattr(self, "notes", [])
        self.notes.append(text)
        k = len(self.notes)
        dx, dy = tx - x, ty - y
        L_ = math.hypot(dx, dy) or 1.0
        bx, by = x + dx / L_ * 9, y + dy / L_ * 9
        self.line(x, y, bx, by, DIM, 0.2)
        self.body.append(f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="2.4" fill="#fff" stroke="{DIM}" stroke-width="0.25"/>')
        self.text(bx, by + 1.0, str(k), 2.6, DIM, "middle")

    def notes_block(self, x, y, width=200):
        for k, n in enumerate(getattr(self, "notes", [])):
            self.text(x, y + 4.2 * k, f"{k + 1}.  {n}", 2.7)

    def svg(self):
        W, H = A3
        if getattr(self, "notes", None) and not getattr(self, "_notes_done", False):
            self._notes_done = True
            self.text(20, H - 18 - 4.2 * len(self.notes) - 5, "NOTES", 3.2)
            self.notes_block(20, H - 18 - 4.2 * len(self.notes))
        tb = [f'<rect x="10" y="10" width="{W - 20}" height="{H - 20}" fill="none" stroke="{INK}" stroke-width="0.7"/>',
              f'<rect x="{W - 190}" y="{H - 42}" width="180" height="32" fill="none" stroke="{INK}" stroke-width="0.5"/>']
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
               '<rect width="100%" height="100%" fill="#fff"/>'] + tb + self.body
        x0, y0 = W - 187, H - 35
        esc = lambda v: v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(f'<text x="{x0}" y="{y0}" font-family="Helvetica, Arial" font-size="5" font-weight="bold">'
                   f'{esc(self.title)}</text>')
        out.append(f'<text x="{x0}" y="{y0 + 6}" font-family="Helvetica, Arial" font-size="3">{esc(self.subtitle)}</text>')
        for k, s in enumerate(self.info):
            out.append(f'<text x="{x0}" y="{y0 + 12 + 4 * k}" font-family="Helvetica, Arial" font-size="2.6">{esc(s)}</text>')
        out.append("</svg>")
        return "\n".join(out)


def _scale_for(w, h, box_w, box_h):
    for s in SCALES:
        if w * s <= box_w and h * s <= box_h:
            return s
    return SCALES[-1]


def _scale_txt(s):
    return "1:1" if s == 1 else f"1:{1 / s:g}"


def housing_sheet(asm, materials=None):
    lay = asm["layout"]
    hp = asm["housing_params"]
    W, H, D = lay["housing_box_mm"]
    t = float(hp.get("wall_thickness", 4.0))
    tc = float(lay.get("cover_mm", t))
    xa, yc = lay["axis_a_mm"]
    xb = lay["axis_b_mm"][0]
    mat = (materials or {}).get("housing") or "Al7075-T6"
    hole_fit = "N7" if mat.startswith(("Al", "A356", "Mg")) else "H7"
    sh = Sheet("HOUSING BODY", f"Idris gearbox · body with integral web · {W:g} × {H:g} × {D:g} mm",
               [f"Material: {mat} · general tolerances ISO 2768-mK", "Bearing bores Ra 1.6, joint faces Ra 3.2, "
                f"inner fillets R ≥ {max(2, round(t / 2))} mm", "Generated by Idris (ACE) — screening-level design"])
    housing_names = {"Housing", "Housing corner columns", "Bearing bosses"}
    # view A: section through the shaft plane (y = yc): z → right, x → down
    s = _scale_for(D + 60, W, 170, 190)
    ox, oy = 45, 50
    P = lambda z, x: (ox + 25 * s + z * s, oy + x * s)
    for part, (a, b) in section(asm["parts"], 1, yc + 0.37, housing_names):
        sh.line(*P(a[2], a[0]), *P(b[2], b[0]))
    sh.text(ox, oy - 18, f"SECTION A–A (shaft plane y = {yc:g})  scale {_scale_txt(s)}", 3.4)
    sh.dim(P(0, 0), P(D, 0), -6, f"{D:g}")
    sh.dim(P(0, 0), P(0, W), -8, f"{W:g}", vertical=True)
    web = lay.get("web")
    if web:
        sh.dim(P(web["z0"], W), P(web["z1"], W), 8, f"web {web['t']:g}")
        sh.dim(P(0, W), P(web["z0"], W), 14, f"{web['z0']:g}")
        sh.leader(*P((web["z0"] + web["z1"]) / 2, xa), *P(web["z1"] + 18, xa - 30),
                  f"hub Ø{web['hub_d']:g}, bearing pockets {hole_fit}")
    bores = lay.get("bores_mm") or {}
    notes_y = oy + W * s + 28
    if bores.get("front"):
        sh.leader(*P(-4, xa), *P(8, W + 30), fit_label(bores["front"], hole_fit) + " bearing seat, front cover boss")
    if bores.get("back"):
        sh.leader(*P(D + 4, xa), *P(D + 6, W + 22), fit_label(bores["back"], hole_fit) + " bearing seat, back cover boss")
    sh.text(ox, notes_y + 12, f"Covers ({tc:g} mm) are separate parts — see sheet COVERS. Side walls {t:g} mm.", 2.8)
    sh.text(ox, notes_y + 17, "Internal sealing lip 3 × 3 mm at both joint faces (land for the cover O-ring).", 2.8)
    sh.text(ox, notes_y + 22, "Untoleranced dimensions " + iso2768_note([W, H, D, t, tc, xa, xb]), 2.8)
    # view B: section through the back sealing lip (z just inside the back joint): x → right, y → up
    lip_z = D - tc - 1.5
    s2 = _scale_for(W + 40, H + 40, 170, 130)
    bx, by = 245, 40 + H * s2 + 15
    Q = lambda x, y: (bx + x * s2, by - y * s2)
    for part, (a, b) in section(asm["parts"], 2, lip_z, housing_names):
        sh.line(*Q(a[0], a[1]), *Q(b[0], b[1]))
    sh.text(bx, by - H * s2 - 24, f"SECTION B–B (z = {lip_z:g}, back lip)  scale {_scale_txt(s2)}", 3.4)
    sh.dim(Q(0, 0), Q(W, 0), 7, f"{W:g}")
    sh.dim(Q(W, 0), Q(W, H), 7, f"{H:g}", vertical=True)
    sh.dim(Q(0, H), Q(xa, H), -6, f"{xa:g}")
    sh.dim(Q(0, H), Q(xb, H), -12, f"{xb:g}")
    sh.dim(Q(0, 0), Q(0, yc), -7, f"{yc:g}", vertical=True)
    for cx, cy in ((xa, yc), (xb, yc)):
        sh.line(*Q(cx - 6, cy), *Q(cx + 6, cy), CUT, 0.2, "3,1")
        sh.line(*Q(cx, cy - 6), *Q(cx, cy + 6), CUT, 0.2, "3,1")
    j = (lay.get("joints") or {}).get("back")
    if j and j.get("pass"):
        d = METRIC[j["size"]][0]
        for k, (cx, cy) in enumerate(j["positions"]):
            sh.line(*Q(cx - 3, cy), *Q(cx + 3, cy), CUT, 0.2)
            sh.line(*Q(cx, cy - 3), *Q(cx, cy + 3), CUT, 0.2)
        cx, cy = j["positions"][0]
        sh.leader(*Q(cx, cy), *Q(cx + 25, cy - 18),
                  f"{j['count']}× {j['size']} thread depth {2 * d:g}, drill Ø{d * 0.8:.1f} depth {2 * d + 4:g}")
        for dx, dy in j["dowels"].get("positions") or []:
            sh.body.append(f'<circle cx="{Q(dx, dy)[0]:.2f}" cy="{Q(dx, dy)[1]:.2f}" r="{d / 2 * s2:.2f}" '
                           f'fill="none" stroke="{INK}" stroke-width="0.3"/>')
        if j["dowels"].get("positions"):
            dx, dy = j["dowels"]["positions"][0]
            sh.leader(*Q(dx, dy), *Q(dx + 30, dy + 22), f"2× Ø{d:g} H7 dowel holes depth {1.5 * d + 2:g}")
    sh.text(bx, by + 22, "Joint faces flat 0.05 mm, Ra 3.2. Same pattern at the front joint (2-stage).", 2.8)
    return sh.svg()


def covers_sheet(asm, plan, materials=None):
    lay = asm["layout"]
    W, H, D = lay["housing_box_mm"]
    tc = float(lay.get("cover_mm", 4.0))
    mat = (materials or {}).get("housing") or "Al7075-T6"
    hole_fit = "N7" if mat.startswith(("Al", "A356", "Mg")) else "H7"
    sh = Sheet("COVERS", f"bolted covers · {W:g} × {H:g} × {tc:g} mm · O-ring face seal",
               [f"Material: {mat} · ISO 2768-mK · sealing face flat 0.05, Ra 1.6",
                "O-ring groove 3.56 × 1.96 (ISO 3601, 2.62 cord), corners R ≥ 8", "Generated by Idris (ACE)"])
    plan_shafts = plan.get("shafts", [])
    covers = [(n, z) for n, z in (("Housing front cover", tc / 2), ("Housing cover", D - tc / 2))
              if any(p["name"] == n for p in asm["parts"])]
    s = _scale_for(W + 30, H + 30, 180, 190)
    for k, (name, zc) in enumerate(covers):
        bx, by = 30 + k * 200, 50 + H * s
        Q = lambda x, y: (bx + x * s, by - y * s)
        for part, (a, b) in section(asm["parts"], 2, zc, {name}):
            sh.line(*Q(a[0], a[1]), *Q(b[0], b[1]))
        wall = "front" if "front" in name else "back"
        sh.text(bx, by - H * s - 10, f"{name.upper()} (seen from inside)  scale {_scale_txt(s)}", 3.4)
        sh.dim(Q(0, 0), Q(W, 0), 8, f"{W:g}")
        sh.dim(Q(W, 0), Q(W, H), 8, f"{H:g}", vertical=True)
        j = (lay.get("joints") or {}).get(wall)
        if j and j.get("pass"):
            d, P_, dk, dh = METRIC[j["size"]]
            for cx, cy in j["positions"]:
                r = dh / 2 * s
                sh.body.append(f'<circle cx="{Q(cx, cy)[0]:.2f}" cy="{Q(cx, cy)[1]:.2f}" r="{r:.2f}" fill="none" '
                               f'stroke="{INK}" stroke-width="0.35"/>')
            cx, cy = j["positions"][0]
            sh.leader(*Q(cx, cy), *Q(cx + 20, cy + 25), f"{j['count']}× Ø{dh:g} (ISO 273 medium), counterbore Ø{dk + 1:g}"
                      f" for DIN 7603 washer")
            sh.text(bx, by + 18, f"Screws: {j['spec']}", 2.5)
        exit_shaft = 0 if wall == "front" else len(plan_shafts) - 1
        for si, ps in enumerate(plan_shafts):
            for b in ps["bearings"]:
                if b["seat"] != wall:
                    continue
                x_, y_ = ps["x"], lay["axis_a_mm"][1]
                what = "through, lip seal outside" if si == exit_shaft else "boss closed by a cap"
                sh.leader(*Q(x_, y_), *Q(x_ + 12, y_ + 0.5 * H - 8 + 6 * si),
                          f"{fit_label(b['D'], hole_fit)} · {b['designation']} · {what}")
        if j and j.get("pass"):
            for dx, dy in j["dowels"].get("positions") or []:
                sh.body.append(f'<circle cx="{Q(dx, dy)[0]:.2f}" cy="{Q(dx, dy)[1]:.2f}" r="{d / 2 * s:.2f}" '
                               f'fill="none" stroke="{INK}" stroke-width="0.3" stroke-dasharray="1,0.6"/>')
            if j["dowels"].get("positions"):
                dx, dy = j["dowels"]["positions"][-1]
                sh.leader(*Q(dx, dy), *Q(dx - 25, dy - 20), f"2× Ø{d:g} H7 dowel holes")
        sh.text(bx, by + 24, f"Thickness {tc:g} mm (from the O-ring sealing check: cover lift between screws ≤ 0.26 mm).",
                2.5)
    return sh.svg()


def shafts_sheet(asm, plan):
    sh = Sheet("SHAFTS", "stepped shafts · AISI 4140 Q&T · bearing seats per ISO 286",
               ["Material: AISI 4140 Q&T (Sut 760 MPa) · ISO 2768-mK", "Bearing seats Ra 0.8, seal seats Ra 0.4 "
                "(plunge-ground), shoulder fillets r ≤ bearing chamfer", "Generated by Idris (ACE)"])
    shafts = [p for p in asm["parts"] if p["kind"] == "shaft"]
    y = 45
    for si, p in enumerate(shafts):
        segs = p.get("segments_mm") or []
        if not segs:
            continue
        z0, z1 = segs[0][0], segs[-1][1]
        dmax = max(d for _, _, d in segs)
        s = _scale_for(z1 - z0 + 40, dmax + 30, 360, 70)
        ox = 30
        X = lambda z: ox + (z - z0) * s
        sh.text(ox, y - 4, f"{p['name'].upper()}  scale {_scale_txt(s)}  length {z1 - z0:.1f}", 3.4)
        yc = y + dmax * s / 2 + 6
        for a, b, d in segs:
            sh.line(X(a), yc - d / 2 * s, X(b), yc - d / 2 * s)
            sh.line(X(a), yc + d / 2 * s, X(b), yc + d / 2 * s)
        edges = sorted({segs[0][0]} | {b for _, b, _ in segs})
        for k, (a, b, d) in enumerate(segs):
            prev = segs[k - 1][2] if k else 0
            h = max(d, prev) / 2 * s
            sh.line(X(a), yc - h, X(a), yc + h)
        sh.line(X(z1), yc - segs[-1][2] / 2 * s, X(z1), yc + segs[-1][2] / 2 * s)
        sh.line(X(z0) - 3, yc, X(z1) + 3, yc, CUT, 0.2, "6,1.5,1,1.5")
        brs = plan["shafts"][si]["bearings"] if si < len(plan.get("shafts", [])) else []
        for b in brs:
            zb = b["z"]
            seg = next((sg for sg in segs if sg[0] - 1e-6 <= zb <= sg[1] + 1e-6), None)
            if seg:
                fit = "k5" if seg[2] <= 40 else "m5"
                sh.leader(X(zb), yc - seg[2] / 2 * s, X(zb) + 6, yc - dmax * s / 2 - 4,
                          f"{fit_label(seg[2], fit)} · {b['designation']}")
        for k, (a, b, d) in enumerate(segs):
            sh.dim((X(a), yc + dmax / 2 * s), (X(b), yc + dmax / 2 * s), 5 + (k % 2) * 4, f"{b - a:.1f}")
            sh.text((X(a) + X(b)) / 2, yc + 1, f"Ø{d:g}", 2.4, INK, "middle")
        y = yc + dmax * s / 2 + 22
    for n in asm.get("notes", []):
        if "integral pinion" in n:
            sh.leader(30, y - 10, 40, y - 4, n)
    sh.text(30, y, "Gear seats: k6 with parallel key DIN 6885-A where a gear is mounted (keyway not drawn); "
            "integral pinions are cut on the shaft (see notes).", 2.7)
    return sh.svg()


def make_drawings(asm, plan, materials=None):
    return {"housing": housing_sheet(asm, materials), "covers": covers_sheet(asm, plan, materials),
            "shafts": shafts_sheet(asm, plan)}
