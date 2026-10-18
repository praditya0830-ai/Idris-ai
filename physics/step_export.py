"""STEP AP214 export of the assembly parts (Tahap 1.3), pure Python, no CAD kernel.

What it writes: every closed mesh component becomes a MANIFOLD_SOLID_BREP whose faces are planar
polygons (coplanar triangles are merged, so boxes come out with 6 faces and holes as inner loops);
round surfaces (gears, shafts, bores) stay faceted. CAD tools import it as solid bodies that can be
measured, sectioned and used for clearance checks. Limits (stated in the API): no analytic cylinders /
NURBS, overlapping solids of one part are not boolean-united. Exact-geometry STEP needs an OpenCascade
kernel on the server (planned, see solver/rencana-solver-setelah-mvp.md).
"""
import time
from collections import defaultdict
import numpy as np


class _W:
    def __init__(self):
        self.lines, self.n = [], 0

    def add(self, text):
        self.n += 1
        self.lines.append(f"#{self.n}={text};")
        return self.n


def _fmt(v):
    return f"{float(v):.6f}".rstrip("0").rstrip(".") if abs(v) > 1e-12 else "0."


def _pt(v):
    return "(" + ",".join(_fmt(c) if "." in _fmt(c) else _fmt(c) + "." for c in v) + ")"


def _components(F):
    """Connected components of a triangle mesh (by shared vertices)."""
    parent = list(range(int(F.max()) + 1)) if len(F) else []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in F:
        for x, y in ((a, b), (b, c)):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
    groups = defaultdict(list)
    for i, tri in enumerate(F):
        groups[find(tri[0])].append(i)
    return list(groups.values())


def _weld(V, F, tol=1e-5):
    key = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[idx], inv.reshape(-1)[F]


def _faces_of(V, F):
    """Group coplanar triangles; return list of (normal, [loops]) where a loop is a vertex-index list.
    Falls back to single triangles when a group's boundary is not a set of simple loops."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-12
    F, n, ln = F[keep], n[keep], ln[keep]
    n = n / ln[:, None]
    off = np.einsum("ij,ij->i", n, V[F[:, 0]])
    key = np.round(np.c_[n, off / 100.0], 5)
    groups = defaultdict(list)
    for i, k in enumerate(map(tuple, key)):
        groups[k].append(i)
    out = []
    for k, tris in groups.items():
        edges = defaultdict(int)
        for i in tris:
            a, b, c = F[i]
            for e in ((a, b), (b, c), (c, a)):
                edges[e] += 1
        bnd = [e for e in edges if (e[1], e[0]) not in edges]
        nxt = defaultdict(list)
        for a, b in bnd:
            nxt[a].append(b)
        if any(len(v) != 1 for v in nxt.values()):
            out += [(n[i], [list(F[i])]) for i in tris]
            continue
        loops, seen = [], set()
        for a, _ in bnd:
            if a in seen:
                continue
            loop, cur = [], a
            while cur not in seen:
                seen.add(cur)
                loop.append(cur)
                cur = nxt[cur][0]
            loops.append(loop)
        # boundary edges keep the triangle orientation: outer loops turn counter-clockwise about the
        # normal (positive area), holes clockwise. Several outer loops = disjoint regions of one plane.
        nn = n[tris[0]]
        e1 = np.cross(nn, [1, 0, 0] if abs(nn[0]) < 0.9 else [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nn, e1)

        def proj(loop):
            P = V[loop]
            return np.c_[P @ e1, P @ e2]

        def signed_area(q):
            return 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))

        def inside(pt, q):
            x, y = pt
            c = False
            for i in range(len(q)):
                x1, y1 = q[i]
                x2, y2 = q[(i + 1) % len(q)]
                if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                    c = not c
            return c
        outers, holes = [], []
        for loop in loops:
            q = proj(loop)
            (outers if signed_area(q) > 0 else holes).append((loop, q))
        if not outers:
            out += [(n[i], [list(F[i])]) for i in tris]
            continue
        faces = [[o] for o in outers]
        for hl, hq in holes:
            owner = [f for f in faces if inside(hq[0] * 0.999999 + hq[1] * 0.000001, f[0][1])]
            (owner[0] if owner else faces[0]).append((hl, hq))
        for f in faces:
            out.append((nn, [lp for lp, _ in f]))
    return out


def _solid(w, V, F, name):
    V, F = _weld(V, F)
    vtx, pts = {}, {}

    def vertex(i):
        if i not in vtx:
            pts[i] = w.add(f"CARTESIAN_POINT('',{_pt(V[i])})")
            vtx[i] = w.add(f"VERTEX_POINT('',#{pts[i]})")
        return vtx[i]
    edges = {}

    def edge(a, b):
        if (b, a) in edges:
            return edges[(b, a)], False
        if (a, b) not in edges:
            d = V[b] - V[a]
            L = float(np.linalg.norm(d))
            dr = w.add(f"DIRECTION('',{_pt(d / L)})")
            vec = w.add(f"VECTOR('',#{dr},{_fmt(L) if '.' in _fmt(L) else _fmt(L) + '.'})")
            p0 = w.add(f"CARTESIAN_POINT('',{_pt(V[a])})")
            line = w.add(f"LINE('',#{p0},#{vec})")
            edges[(a, b)] = w.add(f"EDGE_CURVE('',#{vertex(a)},#{vertex(b)},#{line},.T.)")
        return edges[(a, b)], True
    faces = []
    for nrm, loops in _faces_of(V, F):
        bounds = []
        for li, loop in enumerate(loops):
            oes = []
            for j in range(len(loop)):
                e, same = edge(loop[j], loop[(j + 1) % len(loop)])
                oes.append(w.add(f"ORIENTED_EDGE('',*,*,#{e},{'.T.' if same else '.F.'})"))
            el = w.add("EDGE_LOOP('',(" + ",".join(f"#{o}" for o in oes) + "))")
            bounds.append(w.add(f"{'FACE_OUTER_BOUND' if li == 0 else 'FACE_BOUND'}('',#{el},.T.)"))
        p = w.add(f"CARTESIAN_POINT('',{_pt(V[loops[0][0]])})")
        dz = w.add(f"DIRECTION('',{_pt(nrm)})")
        ref = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0])
        dx = w.add(f"DIRECTION('',{_pt(ref / np.linalg.norm(ref))})")
        ax = w.add(f"AXIS2_PLACEMENT_3D('',#{p},#{dz},#{dx})")
        pl = w.add(f"PLANE('',#{ax})")
        faces.append(w.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{pl},.T.)"))
    shell = w.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
    return w.add(f"MANIFOLD_SOLID_BREP('{name}',#{shell})"), len(faces), len(edges)


def write_step(parts, title="Idris assembly", author="Idris AI (ACE solver)"):
    """parts: [{"name", "verts", "faces"}]. One product with all solids (each part may give several)."""
    w = _W()
    ctx = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})")
    pctx = w.add(f"PRODUCT_CONTEXT('',#{ctx},'mechanical')")
    prod = w.add(f"PRODUCT('{title}','{title}','',(#{pctx}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')")
    pd = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
    pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    lu = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    au = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#{lu},'distance_accuracy_value','')")
    gctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
                 f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))REPRESENTATION_CONTEXT('',''))")
    solids, stats = [], []
    for p in parts:
        V = np.asarray(p["verts"], float).reshape(-1, 3)
        F = np.asarray(p["faces"], int).reshape(-1, 3)
        for k, comp in enumerate(_components(F)):
            used = np.unique(F[comp])
            remap = -np.ones(len(V), int)
            remap[used] = np.arange(len(used))
            sid, nf, ne = _solid(w, V[used], remap[F[comp]], f"{p['name']}" + (f" #{k + 1}" if k else ""))
            solids.append(sid)
            stats.append({"part": p["name"], "solid": k + 1, "faces": nf, "edges": ne})
    o = w.add("CARTESIAN_POINT('',(0.,0.,0.))")
    z = w.add("DIRECTION('',(0.,0.,1.))")
    x = w.add("DIRECTION('',(1.,0.,0.))")
    origin = w.add(f"AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})")
    rep = w.add("ADVANCED_BREP_SHAPE_REPRESENTATION('',(" + ",".join(f"#{s}" for s in solids) +
                f",#{origin}),#{gctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    head = ["ISO-10303-21;", "HEADER;",
            f"FILE_DESCRIPTION(('{title} - faceted B-rep (planar faces)'),'2;1');",
            f"FILE_NAME('{title}.step','{stamp}',('{author}'),('Idris AI'),'Idris ACE','Idris ACE','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    text = "\n".join(head + w.lines + ["ENDSEC;", "END-ISO-10303-21;"]) + "\n"
    return text, stats


def check_step(text):
    """Self-check of the written topology: every EDGE_CURVE is used exactly twice, once per direction,
    inside its shell. Returns (ok, message)."""
    import re
    uses = defaultdict(list)
    for m in re.finditer(r"ORIENTED_EDGE\('',\*,\*,#(\d+),\.(T|F)\.\)", text):
        uses[m.group(1)].append(m.group(2))
    bad = [e for e, u in uses.items() if sorted(u) != ["F", "T"]]
    n_edges = len(re.findall(r"EDGE_CURVE\(", text))
    ok = not bad and n_edges == len(uses)
    return ok, f"{n_edges} edges, {len(bad)} not used exactly once per direction"


def step_solid_volumes(text):
    """Independent re-read of the written STEP: rebuild every face loop from VERTEX/EDGE entities and
    return each solid's volume (divergence theorem). Used to prove the export reproduces the mesh."""
    import re
    ent = {}
    for m in re.finditer(r"#(\d+)=([A-Z_0-9]+)\((.*)\);", text):
        ent[int(m.group(1))] = (m.group(2), m.group(3))
    refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", s)]

    def point(pid):
        return np.array([float(x) for x in re.search(r"\(([^()]*)\)\s*$", ent[pid][1]).group(1).split(",")])
    vols = {}
    for sid, (typ, args) in ent.items():
        if typ != "MANIFOLD_SOLID_BREP":
            continue
        shell = refs(args.split("'")[-1])[0]
        vol = 0.0
        for fid in refs(ent[shell][1]):
            f_args = ent[fid][1]
            bounds = refs(f_args.split(")")[0])
            for b in bounds:
                loop = refs(ent[b][1])[0]
                pts = []
                for oe in refs(ent[loop][1]):
                    e = refs(ent[oe][1])[0]
                    forward = ent[oe][1].strip().endswith(".T.")
                    v0, v1 = refs(ent[e][1])[:2]
                    start = v0 if forward else v1
                    pts.append(point(refs(ent[start][1])[0]))
                P = np.array(pts)
                vol += float(np.dot(np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0), P[0])) / 6.0
        vols[re.search(r"'([^']*)'", args).group(1)] = vol
    return vols
