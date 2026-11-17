"""The only module that touches OpenCascade (Fase 1, pilar A).

The server has cadquery-ocp (OCP); a developer machine or CI without it must not silently pretend a
solid was built. So every entry point raises KernelUnavailable with a plain message, and callers
report that instead of a fake success.

Kept deliberately small: build, boolean, fillet, measure, STEP read/write, tessellate. Anything else
belongs in a caller. This replaces the scattered OCP imports in physics/occt_export.py and
physics/simple_part.py over time; nothing here duplicates their file formats.
"""
from __future__ import annotations

import math
import os


class KernelUnavailable(RuntimeError):
    """No CAD kernel on this machine. The caller must say so, not fall back to a wrong answer."""


def available() -> bool:
    try:
        import OCP.BRepPrimAPI  # noqa: F401
        return True
    except Exception:
        return False


def require() -> None:
    if not available():
        raise KernelUnavailable(
            "no CAD kernel on this server (cadquery-ocp / OCP is not installed); "
            "geometry cannot be built, measured or converted here")


def measure(solid, mass_kg: float | None = None) -> dict:
    """-> summary used by geometry.fingerprint and geometry.invariants."""
    require()
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    volume = abs(props.Mass())
    area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(solid, area)
    box = Bnd_Box()
    BRepBndLib.Add_s(solid, box)
    summary = {"volume_mm3": volume, "area_mm2": abs(area.Mass()),
               "bbox_mm": _extents(box),
               "solids": len(shapes_of(solid, "solid")),
               "faces": len(shapes_of(solid, "face")),
               "edges": len(shapes_of(solid, "edge")),
               "vertices": len(shapes_of(solid, "vertex")),
               "degenerate_faces": _degenerate_faces(solid),
               "watertight": _is_closed(solid)}
    out = dict(summary)
    if mass_kg is not None:
        out["mass_kg"] = mass_kg
    return out


_TOPABS = {"solid": "TopAbs_SOLID", "shell": "TopAbs_SHELL", "face": "TopAbs_FACE",
           "edge": "TopAbs_EDGE", "vertex": "TopAbs_VERTEX"}


def shapes_of(shape, kind: str) -> list:
    """All sub-shapes of one kind (solid, shell, face, edge, vertex)."""
    import OCP.TopAbs as TA
    from OCP.TopExp import TopExp_Explorer
    out, exp = [], TopExp_Explorer(shape, getattr(TA, _TOPABS[kind]))
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _extents(box) -> list:
    """Bounding box extents. Bnd_Box.Get() returns a struct that OCP cannot convert to a Python tuple
    (found on the live server), so the corners are used instead."""
    from OCP.Bnd import Bnd_Box  # noqa: F401  (type only)
    if box.IsVoid():
        return [0.0, 0.0, 0.0]
    lo, hi = box.CornerMin(), box.CornerMax()
    return [abs(hi.X() - lo.X()), abs(hi.Y() - lo.Y()), abs(hi.Z() - lo.Z())]


def _typed(kind: str, shape):
    """Downcast a TopoDS_Shape. OCP names differ between builds: TopoDS.Face_s on some,
    TopoDS.Face on others (the live server has the latter). Guessing one name breaks the whole call."""
    from OCP.TopoDS import TopoDS
    for name in (kind, f"{kind}_s"):
        fn = getattr(TopoDS, name, None)
        if callable(fn):
            return fn(shape)
    raise KernelUnavailable(f"this OCP build exposes neither TopoDS.{kind} nor TopoDS.{kind}_s")


def as_face(shape):
    return _typed("Face", shape)


def as_edge(shape):
    """Downcast to an edge. BRepFilletAPI_MakeFillet.Add wants a TopoDS_Edge, not a TopoDS_Shape:
    passing the shape raised "Add(): incompatible function arguments" on the live server."""
    return _typed("Edge", shape)


def as_wire(shape):
    """Downcast to a wire. Every transform here returns a generic TopoDS_Shape, and
    BRepOffsetAPI_ThruSections.AddWire refuses that ("AddWire(): incompatible function arguments")."""
    return _typed("Wire", shape)


def _is_closed(shape) -> bool:
    """Closed means no free boundary. BRep_Tool.IsClosed answers it per shell (asking a solid directly
    returned False on the live server for a closed box), so the shells are asked first, then the solid,
    then edge->face ancestry. If none of that is available the answer is "not closed": we never claim a
    watertight solid we could not check."""
    from OCP.BRep import BRep_Tool
    closed = getattr(BRep_Tool, "IsClosed_s", None) or getattr(BRep_Tool, "IsClosed", None)
    if callable(closed):
        shells = shapes_of(shape, "shell")
        if shells:
            try:
                if all(bool(closed(s)) for s in shells):
                    return True
            except Exception:
                pass
        try:
            if bool(closed(shape)):
                return True
        except Exception:
            pass
    try:
        from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape as _Map
    except Exception:
        return False
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    ancestors = _Map()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, ancestors)
    n = ancestors.Extent()
    if n == 0:
        return False
    for i in range(1, n + 1):
        if ancestors.FindFromIndex(i).Extent() < 2:
            return False
    return True


def box(dx: float, dy: float, dz: float):
    require()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(float(dx), float(dy), float(dz)).Shape()


def cylinder(radius: float, height: float):
    require()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    return BRepPrimAPI_MakeCylinder(float(radius), float(height)).Shape()


def boolean(op: str, a, b):
    """op ∈ {fuse, cut, common}; raises ValueError for anything else (an AI must not invent ops)."""
    require()
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
    maker = {"fuse": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut, "common": BRepAlgoAPI_Common}.get(op)
    if maker is None:
        raise ValueError(f"unknown boolean op {op!r}; use fuse, cut or common")
    m = maker(a, b)
    m.Build()
    if not m.IsDone():
        raise RuntimeError(f"boolean {op} failed in the kernel")
    return m.Shape()


def fillet(shape, radius: float, edges=None):
    require()
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    maker = BRepFilletAPI_MakeFillet(shape)
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    n = 0
    while exp.More():
        if edges is None or n in edges:
            maker.Add(float(radius), as_edge(exp.Current()))
        n += 1
        exp.Next()
    maker.Build()
    if not maker.IsDone():
        raise RuntimeError("fillet failed in the kernel (radius too large for the geometry)")
    return maker.Shape()


def tessellate(shape, deflection: float = 0.1, angle: float = 0.5):
    """-> (positions, indices) triangles for the browser and for mesh_checks()."""
    require()
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(shape, float(deflection), False, float(angle), True)
    positions, indices = [], []
    lookup = {}                      # welded by position: each face meshes on its own, so without
                                     # welding every shared edge looks like a boundary (24 of them
                                     # on a closed box, found on the live server)

    def vid(p) -> int:
        key = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
        i = lookup.get(key)
        if i is None:
            i = len(positions)
            lookup[key] = i
            positions.append([p.X(), p.Y(), p.Z()])
        return i

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = as_face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            local = [vid(tri.Node(i).Transformed(trsf)) for i in range(1, tri.NbNodes() + 1)]
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                indices += [local[a - 1], local[b - 1], local[c - 1]]
        exp.Next()
    return positions, indices


# --------------------------------------------------------------- modelling primitives
# The sketch/extrude/hole layer needs a handful of primitives beyond the boolean helpers above. They
# live here because this is the only module allowed to import OCP; model/builder.py stays kernel-free
# maths so it can be tested with a stand-in kernel.

PLANE_AXES = {"XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
              "XZ": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
              "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))}


def plane_axes(plane: str):
    """-> (u axis, v axis, normal) in world coordinates. Sketch (u, v) -> world u*e1 + v*e2."""
    try:
        return PLANE_AXES[plane]
    except KeyError:
        raise ValueError(f"unknown sketch plane {plane!r}; use {sorted(PLANE_AXES)}") from None


def version():
    """The kernel version the record should name, or None when this build does not say."""
    try:
        from importlib.metadata import version as _pkg_version
    except Exception:
        return None
    for name in ("cadquery-ocp", "cadquery_ocp", "OCP"):
        try:
            return _pkg_version(name)
        except Exception:
            continue
    return None


def _degenerate_faces(shape, area_min: float = 1e-9) -> int:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    bad = 0
    for face in shapes_of(shape, "face"):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        if abs(props.Mass()) <= area_min:
            bad += 1
    return bad


def _world(plane: str, u: float, v: float):
    from OCP.gp import gp_Pnt
    e1, e2, _ = plane_axes(plane)
    return gp_Pnt(u * e1[0] + v * e2[0], u * e1[1] + v * e2[1], u * e1[2] + v * e2[2])


def sketch_wire(plane: str, profile: str, w: float, h: float | None = None, n: int = 6,
                thickness: float | None = None):
    """-> closed planar wire on `plane`.

    Conventions, so a caller can predict where the material lands:
      rectangle  (0,0)..(w,h);  lshape   an L inside that same box, leg thickness `thickness`;
      circle     centred on the origin, diameter w;
      polygon    `n`-gon, circumscribed radius w/2, centred on the origin;
      triangle   polygon with n = 3, flat side down.
    """
    require()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
    w = float(w)
    if w <= 0:
        raise ValueError(f"sketch: w must be > 0, got {w}")
    if profile == "circle":
        e1, _, normal = plane_axes(plane)
        circ = gp_Circ(gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(*normal), gp_Dir(*e1)), w / 2.0)
        return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire()
    if profile == "rectangle":
        hh = w if h is None else float(h)
        return _wire_through(plane, [(0.0, 0.0), (w, 0.0), (w, hh), (0.0, hh)])
    if profile == "lshape":
        if h is None:
            raise ValueError("sketch: an lshape needs h")
        hh = float(h)
        t = float(thickness) if thickness else min(w, hh) / 3.0
        if not (0.0 < t < min(w, hh)):
            raise ValueError(f"sketch: lshape thickness must be inside (0, {min(w, hh)}), got {t}")
        return _wire_through(plane, [(0.0, 0.0), (w, 0.0), (w, t), (t, t), (t, hh), (0.0, hh)])
    if profile in ("polygon", "triangle"):
        sides = 3 if profile == "triangle" else int(n)
        if not (3 <= sides <= 64):
            raise ValueError(f"sketch: polygon needs 3..64 sides, got {sides}")
        r = w / 2.0
        phase = -90.0 if profile == "triangle" else 0.0
        pts = [(r * math.cos(math.radians(phase + 360.0 * k / sides)),
                r * math.sin(math.radians(phase + 360.0 * k / sides))) for k in range(sides)]
        return _wire_through(plane, pts)
    raise ValueError(f"unknown sketch profile {profile!r}; use rectangle, circle, polygon, triangle or lshape")


def sketch_face(plane: str, profile: str, w: float, h: float | None = None, n: int = 6,
                thickness: float | None = None):
    """-> planar face: the wire above, filled."""
    require()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    return BRepBuilderAPI_MakeFace(sketch_wire(plane, profile, w, h, n, thickness)).Face()


def _wire_through(plane: str, pts):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    mk = BRepBuilderAPI_MakeWire()
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        mk.Add(BRepBuilderAPI_MakeEdge(_world(plane, *a), _world(plane, *b)).Edge())
    return mk.Wire()


def prism(face, vector):
    """Extrude a face along a world-space vector."""
    require()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    vec = [float(c) for c in vector]
    if all(abs(c) < 1e-12 for c in vec):
        raise ValueError("extrude: the distance resolves to zero")
    return BRepPrimAPI_MakePrism(face, gp_Vec(*vec)).Shape()


def translate(shape, vector):
    require()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(*[float(c) for c in vector]))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def rotate(shape, axis: str, angle_deg: float, origin=(0.0, 0.0, 0.0)):
    require()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(*[float(c) for c in origin]), gp_Dir(*_axis(axis))),
                     math.radians(float(angle_deg)))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _axis(axis: str):
    try:
        return {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[str(axis).lower()]
    except KeyError:
        raise ValueError(f"unknown axis {axis!r}; use x, y or z") from None


def revolve(face, axis: str, angle_deg: float):
    require()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt
    mk = BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(*_axis(axis))),
                               math.radians(float(angle_deg)))
    if not mk.IsDone():
        raise RuntimeError("revolve failed in the kernel")
    return mk.Shape()


def loft(bottom, top):
    """Solid between two planar WIRES (draft: the top profile is the offset one).

    Wires, not faces: BRepTools.OuterWire has a different signature than expected on this OCP build
    ("OuterWire_s(): incompatible function arguments" on the live server), so nothing here has to
    extract a wire out of a face.
    """
    require()
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    mk = BRepOffsetAPI_ThruSections(True, True)
    for wire in (bottom, top):
        mk.AddWire(as_wire(wire))
    mk.Build()
    if not mk.IsDone():
        raise RuntimeError("loft failed in the kernel (draft angle too large for the profile?)")
    return mk.Shape()


def cylinder_along(radius: float, height: float, origin, direction):
    """A cylinder from `origin` along `direction` (used for holes)."""
    require()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    if float(radius) <= 0 or float(height) <= 0:
        raise ValueError("cylinder_along: radius and height must be > 0")
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*[float(c) for c in origin]),
                                          gp_Dir(*[float(c) for c in direction])),
                                    float(radius), float(height)).Shape()


def bbox_range(shape):
    """-> (lo, hi) corners in world coordinates."""
    require()
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    lo, hi = box.CornerMin(), box.CornerMax()
    return [lo.X(), lo.Y(), lo.Z()], [hi.X(), hi.Y(), hi.Z()]


def edge_filter(shape, selector: str = "all"):
    """Edge indices for a fillet selector; None means every edge.

    Direction comes from each edge's own bounding box, so no vertex downcast is needed. A closed
    circular edge has no direction to classify and is only ever chosen by "all".
    """
    if selector in (None, "all"):
        return None
    if selector not in ("vertical", "horizontal", "top", "bottom"):
        raise ValueError(f"unknown edge selection {selector!r}; use all, vertical, horizontal, top, bottom")
    require()
    lo, hi = bbox_range(shape)
    out = []
    for i, edge in enumerate(shapes_of(shape, "edge")):
        elo, ehi = bbox_range(edge)
        d = [ehi[k] - elo[k] for k in range(3)]
        length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if selector in ("vertical", "horizontal"):
            if length <= 1e-9:
                continue
            dz = abs(d[2]) / length
            if (selector == "vertical" and dz > 0.9) or (selector == "horizontal" and dz < 0.1):
                out.append(i)
        else:
            want = hi[2] if selector == "top" else lo[2]
            tol = 1e-6 * max(1.0, abs(hi[2]) + abs(lo[2]))
            if all(abs(z - want) <= tol for z in (elo[2], ehi[2])):
                out.append(i)
    return out


def write_step(shape, path: str) -> str:
    require()
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(os.fspath(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed for {path}")
    return os.fspath(path)


def read_step(path: str):
    """STEP in -> one shape. Feature recognition is NOT done here (see exchange/step.py)."""
    require()
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    reader = STEPControl_Reader()
    if reader.ReadFile(os.fspath(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed for {path}")
    reader.TransferRoots()
    return reader.OneShape()
