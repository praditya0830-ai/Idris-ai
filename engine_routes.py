"""HTTP surface for the engine core: geometry, model, assembly, exchange.

Handlers are plain functions returning (status, payload); main.py only forwards to them. The point is
that main.py's dispatcher stays thin: every rule (validation, kernel availability, payload budget)
lives in the modules under geometry/, model/, assembly/ and exchange/, where it can be tested without
an HTTP server.

    GET  /geometry/checks     contract: the invariant ids and what each one means
    GET  /geometry/kernel     is a CAD kernel present on this server right now
    GET  /geometry/selftest   run the engine checks here and say which ones passed
    POST /model/validate      {"model": {...}} -> graph errors, topological order, resolved dims
    POST /model/build         the same model, built with the CAD kernel (+ RunRecord)
    POST /assembly/manifest   an assembly tree -> compact manifest + LOD plan (no geometry)
    POST /assembly/node       one node's geometry, in the LOD the viewer asked for (+ RunRecord)
    POST /exchange/step       {"step_base64": "..."} -> summary, invariants, feature candidates
"""
from __future__ import annotations

import base64
import os
import tempfile

from assembly import lod as LOD
from assembly.resolver import Resolver
from assembly.tree import Assembly, AssemblyError
from audit import run as RUN
from exchange import roundtrip, step as exch_step
from geometry import fingerprint as FP
from geometry import invariants as INV
from geometry import kernel
from model import builder as MODEL
from model.graph import Graph, GraphError, regenerate

GET_PATHS = ("/geometry/checks", "/geometry/kernel", "/geometry/selftest", "/engineer/materials")
POST_PATHS = ("/model/validate", "/model/build", "/assembly/manifest", "/assembly/node",
              "/exchange/step", "/exchange/step-ap242", "/exchange/solidworks-macro",
              "/engineer/solve", "/engineer/verify")

KERNEL = kernel          # the kernel these endpoints use; tests swap in a stand-in
_BUILDS: dict = {}       # model_id -> {"parts": {fid: Part}, "keys": {fid: params_key}}
_MAX_BUILDS = 32


def _solver_version() -> str:
    try:
        import ace_core
        return str(getattr(ace_core, "SOLVER_VERSION", "unknown") or "unknown")
    except Exception:
        return "unknown"


def _record(kind, inputs, summary=None, checks=(), status="ok", extra=None):
    """One RunRecord per answer. If the record itself fails, say so instead of dropping provenance."""
    try:
        return RUN.make(inputs=inputs, solver_version=_solver_version(), geometry_summary=summary,
                        kernel={"name": "occt" if KERNEL.available() else None,
                                "version": KERNEL.version()},
                        checks=list(checks or ()), status=status, kind=kind, extra=extra)
    except Exception as exc:
        return {"error": f"record not produced: {type(exc).__name__}: {exc}"}


def _summary_of(payload):
    """Pull a kernel summary out of whatever shape a helper returned, or None if there is none."""
    if not isinstance(payload, dict):
        return None
    for key in ("summary", "geometry", "measure", "solid"):
        val = payload.get(key)
        if isinstance(val, dict) and "volume_mm3" in val:
            return val
    return payload if "volume_mm3" in payload else None


def _box_mesh(extents):
    """A closed 12-triangle proxy for the LOD 'bbox' level: a box, nothing else."""
    x, y, z = [float(c) for c in (list(extents) + [1.0, 1.0, 1.0])[:3]]
    u, v, w = max(x, 1e-6), max(y, 1e-6), max(z, 1e-6)
    p = [[0, 0, 0], [u, 0, 0], [u, v, 0], [0, v, 0], [0, 0, w], [u, 0, w], [u, v, w], [0, v, w]]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return p, idx


def _decode_part(spec) -> bytes:
    """Client-supplied part geometry: base64 STEP/IGES, or the file as text."""
    if isinstance(spec, str):
        return base64.b64decode(spec)
    if not isinstance(spec, dict):
        raise ValueError("expected an object with step_base64 (or step as text)")
    raw = spec.get("step_base64") or spec.get("iges_base64")
    if raw:
        return base64.b64decode(raw)
    text = spec.get("step") or spec.get("iges")
    if text:
        return str(text).encode("utf-8")
    raise ValueError("expected step_base64 (or step as text)")


def _audit_status(out: dict) -> str:
    if out.get("quarantined"):
        return "quarantined"
    if out.get("blocked"):
        return "blocked"
    return "ok"


def _trim_builds():
    while len(_BUILDS) > _MAX_BUILDS:
        _BUILDS.pop(next(iter(_BUILDS)))


CHECKS = [
    {"id": "mesh.nonempty", "applies_to": "mesh", "means": "there is at least one triangle"},
    {"id": "mesh.indices_in_range", "applies_to": "mesh", "means": "no index points outside the vertex list"},
    {"id": "mesh.no_degenerate_triangles", "applies_to": "mesh", "means": "no zero-area triangle"},
    {"id": "mesh.manifold", "applies_to": "mesh", "means": "no edge shared by more than two triangles"},
    {"id": "mesh.closed", "applies_to": "mesh", "means": "no boundary edge: the surface has no hole"},
    {"id": "solid.single", "applies_to": "solid", "means": "exactly one solid body"},
    {"id": "solid.watertight", "applies_to": "solid", "means": "the shell is closed"},
    {"id": "solid.volume_positive", "applies_to": "solid", "means": "the enclosed volume is above 1e-6 mm3"},
    {"id": "solid.bbox_positive", "applies_to": "solid", "means": "all three extents are greater than zero"},
    {"id": "solid.has_faces", "applies_to": "solid", "means": "the body has faces"},
    {"id": "solid.no_degenerate_faces", "applies_to": "solid", "means": "no zero-area face"},
]


def _cube():
    p = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    idx = []
    for a, b, c, d in quads:
        idx += [a, b, c, a, c, d]
    return p, idx


def _demo_assembly(sub=5, bolts=20):
    """An instanced tree: every subassembly needs its own nodes, so ids are built per sub."""
    subs = []
    for s in range(sub):
        kids = [{"id": f"b-{s}-{i}", "kind": "instance", "part": "bolt", "triangles": 480,
                 "mass_kg": 0.012, "transform": {"translate": [0.0, i * 8.0, 0.0]}} for i in range(bolts)]
        subs.append({"id": f"sub-{s}", "kind": "subassembly", "children": kids})
    return {"model_id": "demo", "tree": {"id": "root", "kind": "assembly", "children":
            [{"id": "housing", "kind": "part", "part": "housing", "triangles": 12000, "mass_kg": 6.4}]
            + subs}}


def _r(check_id, ok, detail=None):
    out = {"id": check_id, "ok": bool(ok)}
    if detail is not None:
        out["detail"] = detail
    return out


def _selftest():
    """The honest version of "does the engine work here": every check says what it exercised."""
    results = []

    p, idx = _cube()
    mesh = INV.mesh_checks(p, idx)
    results.append(_r("mesh.closed_cube", INV.all_ok(mesh), INV.failures(mesh) or "cube passes all mesh checks"))
    open_mesh = INV.mesh_checks(p, idx[:-3])
    results.append(_r("mesh.rejects_open_surface", not INV.all_ok(open_mesh),
                      [c["id"] for c in INV.failures(open_mesh)]))

    summary = {"volume_mm3": 4800.0, "area_mm2": 1760.0, "bbox_mm": [40, 20, 3],
               "solids": 1, "faces": 6, "edges": 12, "vertices": 8}
    shuffled = dict(reversed(list(summary.items())))
    results.append(_r("fingerprint.deterministic", FP.fingerprint(summary) == FP.fingerprint(shuffled),
                      FP.fingerprint(summary)[:16]))
    results.append(_r("fingerprint.sensitive",
                      FP.fingerprint(summary) != FP.fingerprint({**summary, "volume_mm3": 4801.0})))

    g = Graph("selftest", dims={"L": 40, "W": 20, "t": 3})
    g.add("f1", "sketch", [], {"plane": "XZ", "w": "L", "h": "W"})
    g.add("f2", "extrude", ["f1"], {"distance": "t"})
    g.add("f3", "hole", ["f2"], {"d": 4, "x": "L*0.5", "y": "W/2"})
    cache = {fid: fid for fid in g.topo()}
    g.update("f2", {"distance": 4.0})
    out = regenerate(g, changed=["f2"], evaluate=lambda f, d: f.id, cache=cache)
    results.append(_r("model.partial_regeneration", out["regenerated"] == ["f2", "f3"] and out["reused"] == ["f1"],
                      {"regenerated": out["regenerated"], "reused": out["reused"]}))
    bad = regenerate(g, changed=["f1"], evaluate=lambda f, d: (_ for _ in ()).throw(RuntimeError("boom")), cache={})
    results.append(_r("model.quarantine_and_block", bool(bad["quarantined"]) and bool(bad["blocked"]),
                      {"quarantined": [q["id"] for q in bad["quarantined"]],
                       "blocked": [b["id"] for b in bad["blocked"]]}))

    asm = Assembly.from_dict(_demo_assembly())
    resolver = Resolver(asm, load_part=lambda pid, level: {"part": pid})
    size = resolver.manifest_bytes()
    results.append(_r("assembly.manifest_compact", size < 2_000_000,
                      {"bytes": size, "instances": asm.instance_count(),
                       "unique_geometry": asm.unique_geometry()}))
    results.append(_r("assembly.mass_matches_bom",
                      abs(asm.mass_kg() - sum(b["mass_total"] for b in asm.bom())) < 1e-9, asm.mass_kg()))

    if not kernel.available():
        results.append(_r("kernel.available", False,
                          "no CAD kernel on this server (cadquery-ocp / OCP not installed): "
                          "solid build, STEP read/write and round-trip were NOT exercised"))
    else:
        # One check per call: a kernel that fails on one operation must not hide the others, and the
        # detail says which call failed (this is how the Bnd_Box bug was found on the live server).
        state = {}

        def _run(check_id, fn):
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results.append(_r(check_id, ok, detail))

        def shape_fn():
            if "shape" not in state:
                state["shape"] = kernel.box(40.0, 20.0, 3.0)
            return state["shape"]

        def build():
            s = kernel.measure(shape_fn())
            state["summary"] = s
            good = (abs(s["volume_mm3"] - 2400.0) < 24.0
                    and all(abs(a - b) < 0.01 for a, b in zip(s["bbox_mm"], [40.0, 20.0, 3.0])))
            return good, s

        def invariants():
            s = state.get("summary") or kernel.measure(shape_fn())
            checks = INV.solid_checks({**s, "degenerate_faces": 0})
            return INV.all_ok(checks), INV.failures(checks) or checks

        def tessellation():
            pos, ind = kernel.tessellate(shape_fn())
            checks = INV.mesh_checks(pos, ind)
            return INV.all_ok(checks), INV.failures(checks) or {"vertices": len(pos),
                                                               "triangles": len(ind) // 3}

        def step_gate():
            with tempfile.TemporaryDirectory(prefix="idris-st-") as tmp:
                rt = roundtrip.step_roundtrip(shape_fn(), workdir=tmp)
            return rt["status"] == "ok", rt["diffs"] or rt["before_hash"][:16]

        def recognition():
            cut = kernel.boolean("cut", shape_fn(), kernel.cylinder(4.0, 10.0))
            rec = exch_step.candidates_from_counts(exch_step.face_kinds(cut))
            return bool(rec["candidates"]), {"recognised": rec["recognised"],
                                             "dumb_solid": rec["dumb_solid"],
                                             "candidates": rec["candidates"]}

        _run("kernel.build_and_measure", build)
        _run("kernel.solid_invariants", invariants)
        _run("kernel.tessellation_closed", tessellation)
        _run("exchange.step_roundtrip", step_gate)
        _run("exchange.face_recognition", recognition)

    return 200, {"success": True, "pass": all(c["ok"] for c in results), "checks": results,
                 "kernel": kernel.available()}


def handle_get(path: str):
    """-> (status, payload) or None when the path is not ours."""
    p = path.split("?")[0]
    if p == "/geometry/checks":
        return 200, {"success": True, "checks": CHECKS,
                     "note": "a design failing any check is quarantined, never passed on as ok"}
    if p == "/geometry/kernel":
        ok = kernel.available()
        return 200, {"success": True, "available": ok,
                     "detail": "OCP present: builds, STEP and round-trip can run"
                               if ok else "no OCP: solid operations will answer 503, not a fake result"}
    if p == "/geometry/selftest":
        return _selftest()
    if p == "/engineer/materials":
        from ai import engineer as ENG
        return 200, {"success": True, "materials": ENG.MATERIALS,
                     "note": "typical handbook values; a real part needs the mill certificate or the "
                             "standard's guaranteed minimum for its grade"}
    return None


def _validate_features(model: dict):
    """Build the graph feature by feature so every broken feature is named, not just the first.

    X3: every problem keeps the feature it came from, so the Studio can put the message next to the feature
    instead of showing one list. -> (graph or None, [{"feature", "op", "error"}])
    """
    features = model.get("features")
    if not isinstance(features, list):
        return None, [{"feature": None, "op": None, "error": "'features' must be a list"}]
    dims = model.get("dims") if isinstance(model.get("dims"), dict) else {}
    g = Graph(str(model.get("model_id") or "model"), dims=dims)
    problems = []
    for i, spec in enumerate(features):
        if not isinstance(spec, dict) or not spec.get("id") or not spec.get("op"):
            problems.append({"feature": None, "op": None,
                             "error": f"feature #{i}: an object with id and op is required"})
            continue
        try:
            g.add(str(spec["id"]), str(spec["op"]), spec.get("deps") or (), spec.get("params") or {})
        except GraphError as exc:
            problems.append({"feature": str(spec["id"]), "op": str(spec["op"]), "error": str(exc)})
        except Exception as exc:
            problems.append({"feature": str(spec["id"]), "op": str(spec["op"]),
                             "error": f"{spec['id']}: {type(exc).__name__}: {exc}"})
    return g, problems


def _model_body(params):
    model = params.get("model") if isinstance(params.get("model"), dict) else params
    if not isinstance(model, dict) or "features" not in model:
        return None
    return model


def handle_post(base: str, params: dict):
    """-> (status, payload) or None when the path is not ours."""
    if not isinstance(params, dict):
        return 400, {"success": False, "error": "body must be a JSON object"}

    if base in ("/engineer/solve", "/engineer/verify"):
        from ai import engineer as ENG
        if base == "/engineer/verify":
            eng = params.get("engineering") if isinstance(params.get("engineering"), dict) else None
            return 200, {"success": True, **ENG.verify(str(params.get("answer") or ""), eng)}
        given = params.get("given") if isinstance(params.get("given"), dict) else None
        eng = ENG.solve(str(params.get("message") or ""), given)
        if params.get("answer"):
            eng["verification"] = ENG.verify(str(params["answer"]), eng)
        return 200, {"success": True, **eng}

    if base == "/model/validate":
        model = _model_body(params)
        if model is None:
            return 400, {"success": False, "error": "expected a model object with a 'features' list"}
        g, problems = _validate_features(model)
        if g is None:
            return 400, {"success": False, "error": problems[0]["error"],
                         "errors": [p["error"] for p in problems], "problems": problems}
        errors = [p["error"] for p in problems]
        try:
            dims = g.resolved_dims()
        except Exception as exc:                      # a dimension that cannot be resolved is an error, not a 500
            msg = f"dims: {exc}"
            errors.append(msg)
            problems.append({"feature": None, "op": None, "error": msg})
            dims = {}
        # features the exchange path carries but the server-side kernel does not build (X2/X3)
        exchange_only = [f.id for f in g.features() if f.op in ("chamfer", "mirror", "not_translated")]
        return 200, {"success": True, "ok": not errors, "errors": errors, "problems": problems,
                     "order": g.topo(), "dims": dims, "features": len(g),
                     "rejected": len(model.get("features") or []) - len(g),
                     "exchange_only": exchange_only,
                     "model": g.to_dict() if not errors else None}

    if base == "/assembly/manifest":
        try:
            asm = Assembly.from_dict(params.get("assembly") if isinstance(params.get("assembly"), dict) else params)
        except AssemblyError as exc:
            return 400, {"success": False, "error": f"invalid assembly: {exc}"}
        distance = params.get("distance_mm")
        budget = params.get("budget_triangles")
        try:
            resolver = Resolver(asm, load_part=lambda pid, level: None,
                                revision=int(params.get("revision") or 0),
                                budget=int(budget) if budget else 800_000)
            manifest = resolver.manifest(float(distance) if distance is not None else None)
        except (TypeError, ValueError) as exc:
            return 400, {"success": False, "error": f"invalid assembly: {exc}"}
        checks = [
            {"id": "assembly.mass_matches_bom",
             "ok": abs(asm.mass_kg() - round(sum(b["mass_total"] for b in manifest["bom"]), 6)) < 1e-6,
             "detail": f"{asm.mass_kg()} kg over {asm.instance_count()} instances"},
            {"id": "assembly.under_triangle_budget",
             "ok": not manifest["lod"]["over_budget"],
             "detail": f"{manifest['lod']['triangles_total']} of {manifest['lod']['budget']} triangles"},
        ]
        record = _record("assembly", {"assembly": asm.to_dict(), "distance_mm": distance, "budget": budget},
                         checks=checks, status="ok" if all(c["ok"] for c in checks) else "quarantined",
                         extra={"instances": asm.instance_count(), "revision": resolver.revision})
        return 200, {"success": True, **manifest,
                     "checks": checks, "run": record,
                     "note": "structure only: no vertex data here; geometry per node is served by "
                             "/assembly/node with the part's STEP, or by the existing /generate-assembly path",
                     "geometry_included": False}

    if base == "/exchange/step":
        raw = params.get("step_base64") or params.get("iges_base64")
        text = params.get("step") or params.get("iges")
        if not raw and not text:
            return 400, {"success": False, "error": "send step_base64 (or step as text) and a filename"}
        name = str(params.get("filename") or ("part.iges" if params.get("iges_base64") or params.get("iges") else "part.step"))
        if not kernel.available():
            return 503, {"success": False, "error": "kernel_unavailable",
                         "detail": "this server has no cadquery-ocp, so STEP cannot be read; "
                                   "no substitute answer is produced"}
        try:
            blob = base64.b64decode(raw) if raw else str(text).encode("utf-8")
        except Exception as exc:
            return 400, {"success": False, "error": f"step_base64 is not valid base64: {exc}"}
        with tempfile.TemporaryDirectory(prefix="idris-step-") as tmp:
            path = os.path.join(tmp, os.path.basename(name))
            with open(path, "wb") as fh:
                fh.write(blob)
            try:
                shape = exch_step.import_shape(path)
                rec = exch_step.inspect(shape)
                rt = roundtrip.check_shape(shape)
            except exch_step.ImportFailed as exc:
                return 422, {"success": False, "error": str(exc)}
        summary = _summary_of(rec)
        checks = [{"id": c.get("id"), "ok": bool(c.get("ok")), "detail": c.get("detail", "")}
                  for c in (rt.get("checks") or [])]
        record = _record("exchange_step", {"filename": name, "bytes": len(blob)}, summary=summary,
                         checks=checks, status="ok" if rt.get("all_ok") else "quarantined")
        return 200, {"success": True, "bytes": len(blob), **rec, "run": record,
                     "roundtrip_self_check": {"all_ok": rt["all_ok"], "checks": rt["checks"]}}

    if base == "/exchange/step-ap242":
        # v0.39 P5a: the same faceted geometry, written as AP242 with named products, colours, millimetre
        # units and an assembly tree (instance placement per part). Not a native SolidWorks file.
        from physics.step_ap242 import read_back, step_references_ok, write_step_ap242
        parts = params.get("parts")
        if not isinstance(parts, list) or not parts:
            return 400, {"success": False,
                         "error": "send parts: [{name, verts, faces, colour?, rotation?, translation?}]"}
        for i, p in enumerate(parts):
            verts, faces = (p.get("verts"), p.get("faces")) if isinstance(p, dict) else (None, None)
            if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                return 400, {"success": False, "error": f"part {i + 1} needs verts and faces"}
        title = str(params.get("title") or params.get("name") or "Idris part")
        try:
            text, stats = write_step_ap242(parts, title=title)
        except (ValueError, KeyError, IndexError) as exc:
            return 400, {"success": False, "error": f"cannot write this geometry: {exc}"}
        ok, detail = step_references_ok(text)
        back = read_back(text)
        checks = [{"id": "exchange.ap242.references_resolve", "ok": ok, "detail": str(detail)},
                  {"id": "exchange.ap242.solids_read_back", "ok": back["solids"] == stats["solids"],
                   "detail": f"{back['solids']} of {stats['solids']} solids"}]
        record = _record("exchange_step_ap242", {"title": title, "parts": len(parts)},
                         checks=checks, status="ok" if all(c["ok"] for c in checks) else "quarantined",
                         extra={"solids": stats["solids"], "faces": stats["faces"],
                                "solids_per_part": stats["solids_per_part"]})
        return 200, {"success": True, "filename": f"{title.replace(' ', '_')}_AP242.step",
                     "format": "STEP AP242", "step": text, "stats": stats, "read_back": back,
                     "checks": checks, "run": record,
                     "solids_per_part": stats["solids_per_part"], "warnings": stats["warnings"],
                     "note": "faceted planar faces (no analytic cylinders); colours and product names are in "
                             "the file, and the assembly tree carries one placement per part"}

    if base == "/exchange/solidworks-macro":
        # v0.39 P5b: a VBA macro that rebuilds the feature tree as PARAMETRIC features in SolidWorks. A native
        # .SLDPRT cannot be written without SolidWorks; this is what a buyer can actually run there.
        from physics.sw_macro import check_vba, macro_from_model
        model = _model_body(params)
        if model is None:
            return 400, {"success": False,
                         "error": "expected a model object with a 'features' list (the IDRIS feature tree)"}
        name = str(params.get("name") or model.get("model_id") or "Idris part")
        try:
            rep = macro_from_model(model, name=name)
        except ValueError as exc:
            return 400, {"success": False, "error": str(exc),
                         "hint": "a part written by the AI as code has no feature tree: export STEP AP242 instead"}
        ok, problems = check_vba(rep["vba"])
        checks = [{"id": "export.macro.vba_structure", "ok": ok,
                   "detail": "statements and blocks parse" if ok else "; ".join(problems[:4])}]
        record = _record("exchange_solidworks_macro", {"name": name, "features": len(model.get("features") or [])},
                         checks=checks, status="ok" if ok else "quarantined",
                         extra={"translated": len(rep["translated"]), "skipped": len(rep["skipped"])})
        return 200, {"success": True, "filename": f"{name.replace(' ', '_')}_rebuild.bas",
                     "macro": rep["vba"], "translated": rep["translated"], "manual": rep["manual"],
                     "skipped": rep["skipped"], "notes": rep["notes"], "vba_check": {"ok": ok,
                     "problems": problems}, "checks": checks, "run": record,
                     "not_executed": "this macro has not been run in SolidWorks: no SolidWorks is available "
                                      "to IDRIS, so a person has to run it once and check the rebuild"}

    if base == "/model/build":
        model = _model_body(params)
        if model is None:
            return 400, {"success": False, "error": "expected a model object with a 'features' list"}
        try:
            g = Graph.from_dict(model)
        except GraphError as exc:
            return 400, {"success": False, "error": str(exc)}
        if not KERNEL.available():
            return 503, {"success": False, "error": "kernel_unavailable",
                         "detail": "this server has no cadquery-ocp, so no solid can be built; "
                                   "nothing is simulated or approximated instead"}
        entry = _BUILDS.setdefault(g.model_id, {"parts": {}, "keys": {}})
        cache, keys = {}, {}
        for fid in g.topo():
            keys[fid] = MODEL.params_key(g.get(fid))
            if fid in entry["parts"] and entry["keys"].get(fid) == keys[fid]:
                cache[fid] = entry["parts"][fid]
        state: dict = {}
        try:
            out = MODEL.build(g, changed=params.get("changed"), cache=cache, keys=entry["keys"],
                              k=KERNEL, parts_dir=str(params.get("parts_dir") or "."), collect=state)
        except GraphError as exc:
            return 400, {"success": False, "error": str(exc)}
        except Exception as exc:
            return 500, {"success": False, "error": "build failed",
                         "detail": f"{type(exc).__name__}: {exc}"}
        entry["parts"] = {fid: p for fid, p in state.items() if getattr(p, "shape", None) is not None}
        entry["keys"] = out["keys"]
        _trim_builds()
        final = out.get("final") or {}
        record = _record("model", {"model": model, "changed": params.get("changed")},
                         summary=final.get("summary"), checks=final.get("checks") or [],
                         status=_audit_status(out),
                         extra={"model_id": g.model_id, "regenerated": out["regenerated"],
                                "reused": out["reused"]})
        return 200, {"success": True, **out, "run": record, "solver_version": _solver_version()}

    if base == "/assembly/node":
        spec = params.get("assembly") if isinstance(params.get("assembly"), dict) else params
        try:
            asm = Assembly.from_dict(spec)
        except AssemblyError as exc:
            return 400, {"success": False, "error": f"invalid assembly: {exc}"}
        node_id = params.get("node") or params.get("node_id")
        if not node_id:
            return 400, {"success": False, "error": "send node: the id of the node to resolve"}
        if not KERNEL.available():
            return 503, {"success": False, "error": "kernel_unavailable",
                         "detail": "this server has no cadquery-ocp, so a part's STEP cannot be read; "
                                   "no substitute geometry is produced"}
        parts = params.get("parts") if isinstance(params.get("parts"), dict) else {}
        budget = int(params.get("budget_triangles") or LOD.DEFAULT_BUDGET)
        distance = params.get("distance_mm")

        def load_part(part_id, level):
            part_spec = parts.get(part_id)
            if part_spec is None:
                raise LookupError(f"no geometry for part {part_id!r}: send parts[{part_id!r}] = "
                                  '{"step_base64": "..."}')
            try:
                blob = _decode_part(part_spec)
            except Exception as exc:
                raise ValueError(f"parts[{part_id!r}]: {exc}") from None
            with tempfile.TemporaryDirectory(prefix="idris-node-") as tmp:
                name = str((part_spec or {}).get("filename") if isinstance(part_spec, dict) else "")
                path = os.path.join(tmp, os.path.basename(name or f"{part_id}.step"))
                with open(path, "wb") as fh:
                    fh.write(blob)
                shape = exch_step.import_shape(path, k=KERNEL)   # raises ImportFailed with a reason
            return {"shape": shape, "summary": KERNEL.measure(shape)}

        try:
            resolver = Resolver(asm, load_part=load_part,
                                revision=int(params.get("revision") or asm.revision), budget=budget)
            node = resolver.node(str(node_id), float(distance) if distance is not None else None)
        except AssemblyError as exc:
            return 400, {"success": False, "error": str(exc)}
        except (LookupError, ValueError, exch_step.ImportFailed) as exc:
            return 400, {"success": False, "error": str(exc)}

        meta = asm.find(str(node_id))
        level = node["level"]
        geometry = node.get("geometry") or {}
        summary = geometry.get("summary")
        note = None
        if level == "bbox":
            positions, indices = _box_mesh((summary or {}).get("bbox_mm") or [1, 1, 1])
            note = "bbox proxy: a box stands in for the part at this distance"
        else:
            positions, indices = KERNEL.tessellate(geometry["shape"])
            if level == "decimated":
                note = "no mesh decimator on this server: the full mesh is sent and the level says so"
        checks = INV.mesh_checks(positions, indices)
        record = _record("assembly_node", {"node": node_id, "distance_mm": distance, "part": meta.part},
                         summary=summary, checks=checks,
                         status="ok" if INV.all_ok(checks) else "quarantined",
                         extra={"level": level, "cached": node["cached"], "budget": budget})
        return 200, {"success": True, "id": meta.id, "kind": meta.kind, "part": meta.part,
                     "level": level, "cached": node["cached"], "transform": meta.transform,
                     "positions": positions, "indices": indices,
                     "vertices": len(positions), "triangles": len(indices) // 3,
                     "summary": summary, "checks": checks,
                     "geometry_included": True, "note": note, "run": record}

    return None
