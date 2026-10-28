"""OpenCascade (OCCT) — optional geometry kernel, stage 1 of solver/arahan-occt-untuk-solver.md.

Stage 1 only MEASURES whether OCCT can run on the server (no feature depends on it yet):
- status(): is the OCP package installed? (no import, so /health stays light)
- selftest(): import OCP, build an exact cylinder and a box with a through hole, write them to STEP,
  read the file back, compare each solid's volume with the closed-form value (πr²h, box − hole) and check
  that the cylinder faces come back as real cylindrical surfaces. Reports RSS memory before/after the
  import and the time of each step.

Hard rule: the faceted STEP path (physics/step_export.py) stays the default and the fallback. Nothing in
the server imports OCP at start-up; every call is wrapped so a missing or broken OCCT only produces a
report, never an error for the rest of the API.
"""
import importlib.util
import math
import os
import tempfile
import time

PACKAGE_CANDIDATES = ("cadquery-ocp", "cadquery_ocp", "OCP")
_last = {"selftest": None}


def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return None


def _version():
    try:
        from importlib import metadata
        for name in PACKAGE_CANDIDATES:
            try:
                return f"{name} {metadata.version(name)}"
            except metadata.PackageNotFoundError:
                continue
    except Exception:
        pass
    return None


def status():
    """Cheap: is OCP importable in principle? (find_spec does not import the C++ library)."""
    try:
        avail = importlib.util.find_spec("OCP") is not None
    except Exception as e:  # pragma: no cover
        return {"available": False, "version": None, "error": f"{type(e).__name__}: {e}"}
    st = _last["selftest"]
    return {"available": avail, "version": _version() if avail else None,
            "error": None if avail else "OCP (OpenCascade Python binding) is not installed on this server",
            "selftest_passed": st["pass"] if st else None,
            "step_mode_default": "faceted"}


def selftest():
    rep = {"stage": "occt stage 1 selftest", "rss_mb_before_import": _rss_mb(), "steps": [], "pass": False}
    t0 = time.perf_counter()

    def step(name, fn):
        t = time.perf_counter()
        try:
            out = fn()
            rep["steps"].append({"step": name, "ok": True, "s": round(time.perf_counter() - t, 3)})
            return out
        except Exception as e:
            rep["steps"].append({"step": name, "ok": False, "s": round(time.perf_counter() - t, 3),
                                 "error": f"{type(e).__name__}: {e}"})
            raise

    try:
        def _imp():
            global gp_Pnt, gp_Ax2, gp_Dir, BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepAlgoAPI_Cut
            global STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs, IFSelect_RetDone
            global GProp_GProps, BRepGProp, TopExp_Explorer, TopAbs_SOLID, TopAbs_FACE, TopoDS
            global BRepAdaptor_Surface, GeomAbs_Cylinder
            from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir
            from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
            from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
            from OCP.IFSelect import IFSelect_RetDone
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_Cylinder
        step("import OCP", _imp)
        rep["rss_mb_after_import"] = _rss_mb()
        rep["version"] = _version()

        r, h = 10.0, 50.0                        # shaft-like cylinder
        bx, by, bz, rh = 40.0, 30.0, 20.0, 5.0   # housing-like plate with a through hole
        cyl = step("make cylinder", lambda: BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), r, h).Shape())
        box = step("make box", lambda: BRepPrimAPI_MakeBox(bx, by, bz).Shape())
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(bx / 2, by / 2, -1), gp_Dir(0, 0, 1)), rh, bz + 2).Shape()
        plate = step("boolean cut (box − hole)", lambda: BRepAlgoAPI_Cut(box, hole).Shape())

        path = os.path.join(tempfile.gettempdir(), "idris_occt_selftest.step")

        def _write():
            w = STEPControl_Writer()
            for shp in (cyl, plate):
                w.Transfer(shp, STEPControl_AsIs)
            return w.Write(path)
        step("write STEP", _write)
        rep["step_file_bytes"] = os.path.getsize(path)

        def _read():
            rd = STEPControl_Reader()
            if rd.ReadFile(path) != IFSelect_RetDone:
                raise RuntimeError("STEP re-read failed")
            rd.TransferRoots()
            return rd.OneShape()
        back = step("read STEP back", _read)

        def _vols():
            vols, ncyl = [], 0
            ex = TopExp_Explorer(back, TopAbs_SOLID)
            while ex.More():
                pr = GProp_GProps()
                BRepGProp.VolumeProperties_s(ex.Current(), pr)
                vols.append(pr.Mass())
                ex.Next()
            ef = TopExp_Explorer(back, TopAbs_FACE)
            while ef.More():
                if BRepAdaptor_Surface(TopoDS.Face_s(ef.Current())).GetType() == GeomAbs_Cylinder:
                    ncyl += 1
                ef.Next()
            return sorted(vols), ncyl
        vols, ncyl = step("volumes + surface types", _vols)
        exact = sorted([math.pi * r * r * h, bx * by * bz - math.pi * rh * rh * bz])
        errs = [abs(v / e - 1) for v, e in zip(vols, exact)] if len(vols) == 2 else [1.0]
        rep.update(solids=len(vols), volumes_mm3=[round(v, 3) for v in vols],
                   exact_mm3=[round(e, 3) for e in exact], max_volume_error_pct=round(100 * max(errs), 5),
                   cylindrical_faces=ncyl)
        rep["pass"] = len(vols) == 2 and max(errs) < 1e-3 and ncyl >= 2
    except Exception:
        pass
    rep["total_s"] = round(time.perf_counter() - t0, 3)
    rep["rss_mb_end"] = _rss_mb()
    rep["criteria"] = "2 solids, volume error < 0.1 %, cylinder faces read back as cylinders (arahan OCCT §5)"
    _last["selftest"] = rep
    return rep
