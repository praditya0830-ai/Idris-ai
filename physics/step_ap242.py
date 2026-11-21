"""STEP AP242 export (P5a): named products, colours, millimetres, and an assembly tree with placements.

Why a second writer instead of a flag on physics/step_export.py: that module writes ONE product for all solids
(AP214), which is what the gearbox download uses and what its tests lock down. This one writes the shape a
buyer's CAD expects when the file is opened as an assembly:

    a PRODUCT + PRODUCT_DEFINITION per part, with its name
    an ADVANCED_BREP_SHAPE_REPRESENTATION per part (faceted, planar faces -- same geometry writer)
    a NEXT_ASSEMBLY_USAGE_OCCURRENCE + REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION per instance, so a part
      lands where it belongs (rotation + translation)
    a COLOUR_RGB per part, attached through STYLED_ITEM
    millimetre / radian units, stated once in the geometric context

What it is not: analytic cylinders or NURBS (faces stay faceted; the CAD tool can measure and section them, but a
hole is a polygon, not a cylinder), and not a SolidWorks native file -- that cannot be written without SolidWorks.

v0.39.1: the writer no longer trusts the client's vertex sharing. Before a shell is built, coincident vertices
are welded with a relative tolerance, degenerate triangles are dropped and every edge must be shared by exactly
two triangles. A Studio that sends one vertex per triangle used to produce nine separate solids for one bracket;
now it produces one, and the per-part solid count is reported instead of being sent silently.
"""
from __future__ import annotations

import time

import numpy as np

from .step_export import _W, _components, _fmt, _pt, _solid, check_step, step_solid_volumes

SCHEMA_AP242 = "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }"
DEFAULT_COLOUR = (0.72, 0.73, 0.75)
WELD_REL_TOL = 1e-6          # x the part's diagonal
WELD_MIN_MM = 1e-4           # never coarser than this, never so fine that a rounded float fails to weld


def _weld_part(verts, faces):
    """Merge coincident vertices and drop degenerate triangles -> (V, F, info).

    A client may send every triangle with its own three vertices (a triangle soup); the shell then has no shared
    edges and each triangle becomes a solid of its own. Welding here means the writer does not depend on the
    client having done it.
    """
    V = np.asarray(verts, float).reshape(-1, 3)
    F = np.asarray(faces, int).reshape(-1, 3)
    if not len(V) or not len(F):
        raise ValueError("needs verts and faces")
    if F.max() >= len(V) or F.min() < 0:
        raise ValueError("faces point outside verts")
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    tol = max(WELD_REL_TOL * diag, WELD_MIN_MM)
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[keep]
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    F = F[np.linalg.norm(np.cross(e1, e2), axis=1) > 1e-12]
    if not len(F):
        raise ValueError("every triangle is degenerate (zero area)")
    key = np.round(V / tol).astype(np.int64)
    _k, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)            # numpy 2 returns it shaped like the input
    Fw = inv[F]
    keep = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Fw = Fw[keep]
    info = {"vertices_in": int(len(V)), "vertices_used": int(len(np.unique(Fw))),
            "triangles_in": int(len(faces)), "triangles_used": int(len(Fw)),
            "tolerance_mm": round(float(tol), 6)}
    return V[first], Fw, info


def _open_edges(F, nvert):
    """-> (edges not shared by exactly two triangles, total edges). A solid needs a closed shell."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    key = lo * max(int(nvert), 1) + hi
    _u, counts = np.unique(key, return_counts=True)
    return int((counts != 2).sum()), int(len(key))


def _axis(w, point, z, x):
    p = w.add(f"CARTESIAN_POINT('',{_pt(point)})")
    dz = w.add(f"DIRECTION('',{_pt(z)})")
    dx = w.add(f"DIRECTION('',{_pt(x)})")
    return w.add(f"AXIS2_PLACEMENT_3D('',#{p},#{dz},#{dx})")


def _colour(w, rgb):
    r, g, b = (max(0.0, min(1.0, float(c))) for c in (list(rgb or DEFAULT_COLOUR) + [0.0, 0.0, 0.0])[:3])
    return w.add(f"COLOUR_RGB('',{_fmt(r) if '.' in _fmt(r) else _fmt(r) + '.'},"
                 f"{_fmt(g) if '.' in _fmt(g) else _fmt(g) + '.'},"
                 f"{_fmt(b) if '.' in _fmt(b) else _fmt(b) + '.'})")


def _styled(w, colour_id, solid_id):
    c2 = w.add(f"FILL_AREA_STYLE_COLOUR('',#{colour_id})")
    c3 = w.add(f"FILL_AREA_STYLE('',(#{c2}))")
    c4 = w.add(f"SURFACE_STYLE_FILL_AREA(#{c3})")
    c5 = w.add(f"SURFACE_SIDE_STYLE('',(#{c4}))")
    c6 = w.add(f"SURFACE_STYLE_USAGE(.BOTH.,#{c5})")
    c7 = w.add(f"PRESENTATION_STYLE_ASSIGNMENT((#{c6}))")
    return w.add(f"STYLED_ITEM('colour',(#{c7}),#{solid_id})")


def write_step_ap242(parts, title="Idris part", author="Idris AI (ACE solver)"):
    """parts: [{"name", "verts", "faces", "colour": [r,g,b], "rotation": 3x3, "translation": [x,y,z]}].

    -> (step text, stats). Each part may contain several closed components; each becomes one solid body named
    after its part, `#k` appended when there is more than one.
    """
    if not parts:
        raise ValueError("no part to export")
    w = _W()
    ctx = w.add("APPLICATION_CONTEXT('managed model based 3d engineering')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','ap242_managed_model_based_3d_engineering',"
          f"2014,#{ctx})")
    pctx = w.add(f"PRODUCT_CONTEXT('',#{ctx},'mechanical')")
    pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')")
    lu = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    au = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#{lu},'distance_accuracy_value','')")
    gctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
                 f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))REPRESENTATION_CONTEXT('',''))")

    asm_name = title or "Idris part"
    prod = w.add(f"PRODUCT('{asm_name}','{asm_name}','',(#{pctx}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    asm_pd = w.add(f"PRODUCT_DEFINITION('assembly','',#{pdf},#{pdc})")
    asm_pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{asm_pd})")
    asm_origin = _axis(w, [0, 0, 0], [0, 0, 1], [1, 0, 0])
    asm_rep = w.add(f"SHAPE_REPRESENTATION('{asm_name}',(#{asm_origin}),#{gctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{asm_pds},#{asm_rep})")

    solids, stats, styled, nauo = [], [], [], []
    solids_per_part: dict = {}
    weld_stats: list = []
    warnings: list = []
    for i, p in enumerate(parts):
        name = str(p.get("name") or f"part {i + 1}").replace("'", "")
        V, F, welded = _weld_part(p["verts"], p["faces"])
        weld_stats.append({"part": name, **welded})
        nopen, nedges = _open_edges(F, len(V))
        if nopen:
            raise ValueError(f"the mesh of part '{name}' is not closed: {nopen} of {nedges} edge(s) are not "
                             f"shared by exactly two triangles, so no solid can be built from it; weld the "
                             f"vertices or close the shell")
        ncomp = len(_components(F))
        solids_per_part[name] = ncomp
        if ncomp > 1:
            warnings.append(f"part '{name}' is {ncomp} separate closed bodies; they are written as {ncomp} "
                            f"solids of one product. Send one connected shell if it should be a single body")
        pp = w.add(f"PRODUCT('{name}','{name}','',(#{pctx}))")
        ppf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{pp})")
        ppd = w.add(f"PRODUCT_DEFINITION('part','',#{ppf},#{pdc})")
        ppds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{ppd})")
        comp_solids = []
        for k, comp in enumerate(_components(F)):
            used = np.unique(F[comp])
            remap = -np.ones(len(V), int)
            remap[used] = np.arange(len(used))
            solid_name = name + (f" #{k + 1}" if ncomp > 1 else "")
            sid, nf, ne = _solid(w, V[used], remap[F[comp]], solid_name)
            comp_solids.append(sid)
            stats.append({"part": name, "solid": k + 1, "faces": nf, "edges": ne})
        part_axis = _axis(w, [0, 0, 0], [0, 0, 1], [1, 0, 0])
        rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',("
                    + ",".join(f"#{s}" for s in comp_solids) + f",#{part_axis}),#{gctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{ppds},#{rep})")
        # placement: the part's own frame -> its frame in the assembly
        rot = np.asarray(p.get("rotation") or np.eye(3), float).reshape(3, 3)
        trans = np.asarray(p.get("translation") or [0, 0, 0], float).reshape(3)
        z_col = rot @ np.array([0, 0, 1.0])
        x_col = rot @ np.array([1, 0, 0.0])
        placed = _axis(w, trans, z_col, x_col)
        idt = w.add(f"ITEM_DEFINED_TRANSFORMATION('','',#{part_axis},#{placed})")
        w.add(f"(REPRESENTATION_RELATIONSHIP('','',#{rep},#{asm_rep})"
              f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idt})SHAPE_REPRESENTATION_RELATIONSHIP())")
        nauo.append(w.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{i + 1}','{name}','',#{asm_pd},#{ppd},$)"))
        col = _colour(w, p.get("colour"))
        for sid in comp_solids:
            styled.append(_styled(w, col, sid))
        solids.extend(comp_solids)
    w.add("MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',("
          + ",".join(f"#{s}" for s in styled) + f"),#{gctx})")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    head = ["ISO-10303-21;", "HEADER;",
            f"FILE_DESCRIPTION(('{asm_name} - AP242 faceted B-rep, colours, mm'),'2;1');",
            f"FILE_NAME('{asm_name}.step','{stamp}',('{author}'),('Idris AI'),'Idris ACE','Idris ACE','');",
            f"FILE_SCHEMA(('{SCHEMA_AP242}'));", "ENDSEC;", "DATA;"]
    text = "\n".join(head + w.lines + ["ENDSEC;", "END-ISO-10303-21;"]) + "\n"
    return text, {"format": "STEP AP242", "products": [str(p.get("name") or "") for p in parts],
                  "solids": len(solids), "instances": len(nauo), "styled_items": len(styled),
                  "units": "mm, rad", "faces": sum(s["faces"] for s in stats), "parts_detail": stats,
                  "solids_per_part": solids_per_part, "warnings": warnings, "weld": weld_stats,
                  "sew": "coincident vertices welded in the writer (relative tolerance); OCP is not installed "
                         "in this environment, so BRepBuilderAPI_Sewing was not used"}


def step_references_ok(text: str) -> tuple:
    """Every #id a line points at must exist in the file, and every edge must be used once per direction.

    This is the only self-check that can run without a CAD kernel in the sandbox; it cannot prove a reader likes
    the file, only that the file is not broken.
    """
    import re
    defined = {int(m.group(1)) for m in re.finditer(r"^#(\d+)=", text, re.M)}
    used = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        body = line.split("=", 1)[1]
        used.update(int(x) for x in re.findall(r"#(\d+)", body))
    dangling = sorted(used - defined)
    ok_edges, msg = check_step(text)
    return (not dangling and ok_edges), {"dangling_references": dangling, "edge_check": msg}


def read_back(text: str) -> dict:
    """What a reader can get back out of the file without a kernel: product names, volumes, instances."""
    import re
    names = re.findall(r"PRODUCT\('([^']*)','[^']*','',\(", text)
    vols = step_solid_volumes(text)
    return {"products": names, "solids": len(vols), "volumes_mm3": vols,
            "instances": len(re.findall(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\(", text)),
            "styled_items": len(re.findall(r"STYLED_ITEM\(", text)),
            "schema": (re.search(r"FILE_SCHEMA\(\('([^']+)'\)\)", text) or [None, None])[1]}
