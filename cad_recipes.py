"""The four Studio recipes (ui/resep-cad-contoh.md, UI v56), verbatim. They build in the browser kernel and pass the
UI checks, so the AI uses them as reference code (few-shot) for technique and style."""

STYLE_NOTES = """Style notes from the Studio recipes:
- Shapes from 2D profiles (draw().lineTo().customCorner(r)) then extrude/revolve, not stacked boxes.
- Curves with cubicBezierCurveTo(end, c1, c2) and explicit control points. Complex shapes = intersection of 3 views
  (side ∩ top ∩ front), like the car body.
- 3D fillets only inside try {}; if they fail the shape still builds.
- Functional details: cable notch, lightening holes with 45° pointed ends (FDM without supports), standard motor mount.
- probes = reference objects not built (phone, wheels) or measuring slices (...Cut); the UI measures centre of mass,
  collisions and diameters from them.
- Never put the same point twice in a row in draw() (zero-length segment -> kernel error); never use forbidden words,
  not even in comments."""

RECIPES = [
    {"id": "phone-stand", "title": "Phone stand", "kind": "functional", "process": "fdm",
     "params": "T (Phone thickness incl. case, mm), Hp (Phone height, mm), a (Lean angle from the desk, deg), "
               "W (Stand width, mm), t (Wall thickness, mm), Ls (Back support length, mm), lip (Front lip height, mm)",
     "code": r"""const { T, Hp, a, W, t, Ls, lip } = P;
const r = a * Math.PI / 180, s = Math.sin(r), c = Math.cos(r);
const xf = Ls * c + t + 2;            // foot of the back support
const xr = xf + t / s;                // where the back of the phone meets the base
const xl = xr + T / s + 1;            // inside of the front lip, 1 mm clearance
const B = xl + t + 4;                 // base length
const k = Math.min(1.6, t * 0.4), kt = Math.min(t * 0.45, 2.2);
// side profile with rounded corners, extruded across the width
const prof = draw([B / 2, 0]).lineTo([B, 0]).lineTo([B, t]).customCorner(k).lineTo([xl + t, t]).customCorner(k)
  .lineTo([xl + t, t + lip]).customCorner(k).lineTo([xl, t + lip]).customCorner(k).lineTo([xl, t]).customCorner(k).lineTo([xr, t])
  .lineTo([xf - Ls * c + t * s, t + Ls * s + t * c]).customCorner(kt).lineTo([xf - Ls * c, t + Ls * s]).customCorner(kt)
  .lineTo([xf, t]).customCorner(k).lineTo([0, t]).customCorner(k).lineTo([0, 0]).close();
let body = prof.sketchOnPlane('XZ', -W).extrude(W);
// opening in the back support (lighter, and it looks designed) and a cable notch in the lip
const win = Math.min(W * 0.45, 40), wl = Ls * 0.55, x0 = -Ls * 0.55, y0 = W / 2, hw = win / 2;
// pointed ends at 45° so the opening prints without supports
if (W > 30 && Ls > 50) body = body.cut(draw([x0 - wl / 2, y0]).lineTo([x0 - wl / 2 + hw, y0 - hw]).customCorner(hw * 0.3).lineTo([x0 + wl / 2 - hw, y0 - hw]).customCorner(hw * 0.3)
  .lineTo([x0 + wl / 2, y0]).lineTo([x0 + wl / 2 - hw, y0 + hw]).customCorner(hw * 0.3).lineTo([x0 - wl / 2 + hw, y0 + hw]).customCorner(hw * 0.3).close()
  .sketchOnPlane('XY', -5).extrude(t + 10).rotate(a, [0, 0, 0], [0, 1, 0]).translate([xf, 0, t]));
const notch = Math.min(16, W * 0.3);
body = body.cut(makeBox([xl - 1, W / 2 - notch / 2, t + 2], [xl + t + 1, W / 2 + notch / 2, t + lip + 1]));
const phone = makeBox([0, 0, 0], [T, W, Hp]).rotate(-(90 - a), [0, 0, 0], [0, 1, 0]).translate([xr, 0, t]);
return { body, probes: { phone } };"""},
    {"id": "handlebar", "title": "Handlebar", "kind": "interface", "process": "cnc",
     "params": "D (Clamp diameter, mm), Wt (Total width, mm), rise (Rise, mm), cl (Straight centre length, mm), "
               "run (Bend run (sideways), mm), R (Bend radius, mm), wall (Wall thickness, mm)",
     "code": r"""const { D, Wt, rise, cl, run, R, wall } = P;
const g = 22.2;                                   // 7/8 in grip and lever size
const h = Wt / 2, xa = cl / 2 + run;
const path = draw([-h, rise]).hLineTo(-xa).customCorner(R).lineTo([-cl / 2, 0]).customCorner(R).hLineTo(cl / 2).customCorner(R)
  .lineTo([xa, rise]).customCorner(R).hLineTo(h).done().sketchOnPlane('XZ');
const tube = (d) => path.clone().sweepSketch((plane, origin) => sketchCircle(d / 2, { plane, origin }));
let body = tube(g);
if (D > g + 0.1) {
  const tang = R * Math.tan(Math.atan2(rise, run) / 2), sl = cl / 2 - tang - 2, tp = Math.min(30, sl / 3);
  const sleeve = makeCylinder(D / 2, 2 * (sl - tp), [-(sl - tp), 0, 0], [1, 0, 0]);
  const taper = (x0, x1) => sketchCircle(D / 2, { plane: 'YZ', origin: [x0, 0, 0] }).loftWith(sketchCircle(g / 2, { plane: 'YZ', origin: [x1, 0, 0] }));
  body = body.fuse(sleeve).fuse(taper(sl - tp, sl)).fuse(taper(-(sl - tp), -sl));
}
body = body.cut(tube(g - 2 * wall));
const clampCut = body.clone().intersect(makeBox([-1, -60, -60], [1, 60, 60]));
const gripCut = body.clone().intersect(makeBox([h - 21, -60, -60], [h - 19, 60, 60]));
return { body, probes: { clampCut, gripCut } };"""},
    {"id": "car-body", "title": "Car body · proportions", "kind": "styling", "process": None,
     "params": "Lc (Length, mm), Wc (Width, mm), Hc (Height, mm), WB (Wheelbase, mm), Dw (Wheel diameter, mm), "
               "gc (Ground clearance, mm), cab (Cabin position (0 front … 1 rear))",
     "code": r"""const { Lc, Wc, Hc, WB, Dw, gc, cab } = P;
const H = Hc - gc, fo = (Lc - WB) * 0.45, belt = gc + 0.50 * H, X = (u) => u * Lc, Z = (v) => gc + v * H;
// three views intersected: side silhouette, plan (top) and front section with tumblehome
const side = draw([X(0.06), gc]).lineTo([X(0.94), gc])
  .cubicBezierCurveTo([Lc, Z(0.28)], [X(0.985), gc], [Lc, Z(0.12)])
  .cubicBezierCurveTo([X(0.985), Z(0.55)], [Lc, Z(0.42)], [Lc, Z(0.52)])
  .cubicBezierCurveTo([X(cab + 0.06), Hc], [X(0.85), Z(0.62)], [X(cab + 0.22), Hc])
  .cubicBezierCurveTo([X(cab - 0.24), Z(0.55)], [X(cab - 0.08), Hc], [X(cab - 0.16), Z(0.62)])
  .cubicBezierCurveTo([0, Z(0.20)], [X(0.18), Z(0.46)], [0, Z(0.34)])
  .cubicBezierCurveTo([X(0.06), gc], [0, Z(0.08)], [X(0.02), gc]).close()
  .sketchOnPlane('XZ', -(Wc / 2 + 10)).extrude(Wc + 20);
const plan = draw([0, -0.30 * Wc]).cubicBezierCurveTo([X(0.20), -0.5 * Wc], [0, -0.42 * Wc], [X(0.08), -0.5 * Wc]).lineTo([X(0.85), -0.5 * Wc])
  .cubicBezierCurveTo([Lc, -0.40 * Wc], [X(0.95), -0.5 * Wc], [Lc, -0.46 * Wc]).lineTo([Lc, 0.40 * Wc])
  .cubicBezierCurveTo([X(0.85), 0.5 * Wc], [Lc, 0.46 * Wc], [X(0.95), 0.5 * Wc]).lineTo([X(0.20), 0.5 * Wc])
  .cubicBezierCurveTo([0, 0.30 * Wc], [X(0.08), 0.5 * Wc], [0, 0.42 * Wc]).close().sketchOnPlane('XY', gc - 20).extrude(H + 60);
const front = draw([-Wc / 2 - 5, gc - 20]).lineTo([Wc / 2 + 5, gc - 20]).lineTo([Wc / 2 + 5, belt]).customCorner(0.06 * Wc)
  .lineTo([0.33 * Wc, Hc + 20]).customCorner(0.08 * Wc).lineTo([-0.33 * Wc, Hc + 20]).customCorner(0.08 * Wc).lineTo([-Wc / 2 - 5, belt]).customCorner(0.06 * Wc).close()
  .sketchOnPlane('YZ', -10).extrude(Lc + 20);
let body = side.intersect(plan).intersect(front);
const tw = 0.13 * Wc, wy = Wc / 2 - tw;
const wheel = (x, y) => makeCylinder(Dw / 2, tw, [x, y, Dw / 2], [0, 1, 0]);
const aw = tw + 60;                                  // wheel wells: only the outer sides are cut
const arch = (x, y) => makeCylinder(Dw / 2 + 35, aw + 100, [x, y, Dw / 2], [0, 1, 0]);
for (const x of [fo, fo + WB]) body = body.cut(arch(x, wy - 60)).cut(arch(x, -wy - tw - 100));
const wheelsFront = wheel(fo, wy).fuse(wheel(fo, -wy - tw));
const wheelsRear = wheel(fo + WB, wy).fuse(wheel(fo + WB, -wy - tw));
return { body, probes: { wheelsFront, wheelsRear } };"""},
    {"id": "rocket", "title": "Model rocket", "kind": "functional", "process": None,
     "params": "D (Body tube diameter, mm), Lb (Body length (no nose), mm), nr (Nose length ÷ diameter), n (Number of fins), "
               "cr (Fin root chord, mm), ct (Fin tip chord, mm), span (Fin span, mm), sw (Fin sweep, mm), "
               "tf (Fin thickness, mm), wall (Tube wall, mm), Dm (Motor diameter, mm)",
     "code": r"""const { D, Lb, nr, n, cr, ct, span, sw, tf, wall, Dm } = P;
const R = D / 2, Ln = nr * D, rho = (R * R + Ln * Ln) / (2 * R);
// tangent ogive nose cone, revolved; profile in the XZ plane (x = radius, z = height)
let pen = draw([0, Lb]).lineTo([R, Lb]);
for (let i = 1; i < 48; i++) { const x = Ln * i / 48; pen = pen.lineTo([Math.sqrt(Math.max(rho * rho - x * x, 0)) + R - rho, Lb + x]); }
const nose = pen.lineTo([0, Lb + Ln]).close().sketchOnPlane('XZ').revolve();
const tube = makeCylinder(R, Lb, [0, 0, 0], [0, 0, 1]).cut(makeCylinder(R - wall, Lb + 2, [0, 0, -1], [0, 0, 1]));
// fins: trapezoid from the tube wall, swept back, repeated around the axis
const fin = draw([R - 0.5, 0]).lineTo([R - 0.5, cr]).lineTo([R + span, cr - sw]).lineTo([R + span, cr - sw - ct]).close().sketchOnPlane('XZ', -tf / 2).extrude(tf);
let body = tube.fuse(nose);
for (let i = 0; i < n; i++) body = body.fuse(fin.clone().rotate(360 * i / n, [0, 0, 0], [0, 0, 1]));
// motor mount tube and centering ring at the aft end
const mm = makeCylinder(Dm / 2 + 1, 75, [0, 0, 0], [0, 0, 1]).cut(makeCylinder(Dm / 2, 77, [0, 0, -1], [0, 0, 1]));
const ring = makeCylinder(R - wall + 0.01, 3, [0, 0, 60], [0, 0, 1]).cut(makeCylinder(Dm / 2 + 0.5, 5, [0, 0, 59], [0, 0, 1]));
body = body.fuse(mm).fuse(ring);
const midCut = body.clone().intersect(makeBox([-200, -200, Lb / 2 - 1], [200, 200, Lb / 2 + 1]));
return { body, probes: { midCut } };"""},
]


KEYWORDS = {
    "phone-stand": ("holder", "stand", "dudukan", "bracket", "braket", "hook", "mount", "rak", "tray", "clip", "klem"),
    "handlebar": ("stang", "handlebar", "pipa", "tube", "frame", "rangka", "sepeda", "bicycle", "bike", "rail", "handle",
                  "gagang", "sweep", "bend"),
    "car-body": ("mobil", "car", "body", "kapal", "boat", "hull", "pesawat", "aircraft", "wing", "sayap", "helm",
                 "helmet", "mouse", "casing", "cover", "styling", "konsep", "concept"),
    "rocket": ("roket", "rocket", "nozzle", "vas", "vase", "pot", "botol", "bottle", "cup", "gelas", "knob", "pulley",
               "puli", "spacer", "flange", "revolve", "fin", "sirip", "propeller", "corong"),
}


def pick(message: str, n: int = 2) -> list:
    """The n recipes closest to the request (a short prompt answers faster); always at least the phone stand."""
    m = (message or "").lower()
    scored = sorted(RECIPES, key=lambda r: -sum(w in m for w in KEYWORDS[r["id"]]))
    hit = [r for r in scored if any(w in m for w in KEYWORDS[r["id"]])][:n]
    for r in RECIPES:
        if len(hit) >= n:
            break
        if r not in hit:
            hit.append(r)
    return hit


def prompt_block(message: str | None = None, n: int = 2) -> str:
    recipes = RECIPES if message is None else pick(message, n)
    out = ["Reference code from the Studio recipes. It builds in our kernel and passes the checks: reuse its "
           "techniques (profiles + customCorner, cubicBezierCurveTo, intersection of views, sweepSketch, loftWith, "
           "revolve, probes).", STYLE_NOTES]
    for r in recipes:
        out.append(f"--- {r['title']} (kind {r['kind']}, process {r['process']}) params: {r['params']}\n{r['code']}")
    return "\n\n".join(out)
