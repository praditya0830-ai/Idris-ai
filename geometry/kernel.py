"""The only module that touches OpenCascade (Fase 1, pilar A).

The server has cadquery-ocp (OCP); a developer machine or CI without it must not silently pretend a
solid was built. So every entry point raises KernelUnavailable with a plain message, and callers
report that instead of a fake success.

Kept deliberately small: build, boolean, fillet, measure, STEP read/write, tessellate. Anything else
belongs in a caller. This replaces the scattered OCP imports in physics/occt_export.py and
physics/simple_part.py over time; nothing here duplicates their file formats.
"""
from __future__ import annotations

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
               "solids": len(_solids(solid)), "watertight": _is_closed(solid)}
    out = dict(summary)
    if mass_kg is not None:
        out["mass_kg"] = mass_kg
    return out


def _solids(shape) -> list:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    out, exp = [], TopExp_Explorer(shape, TopAbs_SOLID)
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


def _is_closed(shape) -> bool:
    """Closed means no free boundary: BRep_Tool.IsClosed answers it for a solid. If this build does not
    expose it, fall back to edge->face ancestry, and if even that is unavailable say "not closed"
    rather than claim a watertight solid we could not check."""
    from OCP.BRep import BRep_Tool
    for name in ("IsClosed_s", "IsClosed"):
        fn = getattr(BRep_Tool, name, None)
        if callable(fn):
            try:
                return bool(fn(shape))
            except Exception:
                break
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
            maker.Add(float(radius), exp.Current())
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
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = as_face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            base = len(positions)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(loc.Transformation())
                positions.append([p.X(), p.Y(), p.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                indices += [base + a - 1, base + b - 1, base + c - 1]
        exp.Next()
    return positions, indices


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
