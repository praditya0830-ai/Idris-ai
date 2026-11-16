"""HTTP surface for the engine core: geometry, model, assembly, exchange.

Handlers are plain functions returning (status, payload); main.py only forwards to them. The point is
that main.py's dispatcher stays thin: every rule (validation, kernel availability, payload budget)
lives in the modules under geometry/, model/, assembly/ and exchange/, where it can be tested without
an HTTP server.

    GET  /geometry/checks     contract: the invariant ids and what each one means
    GET  /geometry/kernel     is a CAD kernel present on this server right now
    GET  /geometry/selftest   run the engine checks here and say which ones passed
    POST /model/validate      {"model": {...}} -> graph errors, topological order, resolved dims
    POST /assembly/manifest   an assembly tree -> compact manifest + LOD plan (no geometry)
    POST /exchange/step       {"step_base64": "..."} -> summary, invariants, feature candidates
"""
from __future__ import annotations

import base64
import os
import tempfile

from assembly.resolver import Resolver
from assembly.tree import Assembly, AssemblyError
from exchange import roundtrip, step as exch_step
from geometry import fingerprint as FP
from geometry import invariants as INV
from geometry import kernel
from model.graph import Graph, GraphError, regenerate

GET_PATHS = ("/geometry/checks", "/geometry/kernel", "/geometry/selftest")
POST_PATHS = ("/model/validate", "/assembly/manifest", "/exchange/step")

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
        try:
            shape = kernel.box(40, 20, 3)
            s = kernel.measure(shape)
            solid = INV.solid_checks({**s, "degenerate_faces": 0})
            results.append(_r("kernel.solid_invariants", INV.all_ok(solid), INV.failures(solid) or s))
            pos, ind = kernel.tessellate(shape)
            tess = INV.mesh_checks(pos, ind)
            results.append(_r("kernel.tessellation_closed", INV.all_ok(tess), INV.failures(tess) or len(pos)))
            with tempfile.TemporaryDirectory(prefix="idris-st-") as tmp:
                rt = roundtrip.step_roundtrip(shape, workdir=tmp)
            results.append(_r("exchange.step_roundtrip", rt["status"] == "ok", rt["diffs"] or rt["before_hash"][:16]))
            cut = kernel.boolean("cut", shape, kernel.cylinder(4, 10))
            rec = exch_step.candidates_from_counts(exch_step.face_kinds(cut))
            results.append(_r("exchange.face_recognition", bool(rec["candidates"]),
                              {"recognised": rec["recognised"], "dumb_solid": rec["dumb_solid"],
                               "candidates": rec["candidates"]}))
        except Exception as exc:                        # a kernel that fails is a result too
            results.append(_r("kernel.available", False, f"{type(exc).__name__}: {exc}"))

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
    return None


def _validate_features(model: dict):
    """Build the graph feature by feature so every broken feature is named, not just the first."""
    features = model.get("features")
    if not isinstance(features, list):
        return None, ["'features' must be a list"]
    dims = model.get("dims") if isinstance(model.get("dims"), dict) else {}
    g = Graph(str(model.get("model_id") or "model"), dims=dims)
    errors = []
    for i, spec in enumerate(features):
        if not isinstance(spec, dict) or not spec.get("id") or not spec.get("op"):
            errors.append(f"feature #{i}: an object with id and op is required")
            continue
        try:
            g.add(str(spec["id"]), str(spec["op"]), spec.get("deps") or (), spec.get("params") or {})
        except GraphError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"{spec.get('id')}: {type(exc).__name__}: {exc}")
    return g, errors


def _model_body(params):
    model = params.get("model") if isinstance(params.get("model"), dict) else params
    if not isinstance(model, dict) or "features" not in model:
        return None
    return model


def handle_post(base: str, params: dict):
    """-> (status, payload) or None when the path is not ours."""
    if not isinstance(params, dict):
        return 400, {"success": False, "error": "body must be a JSON object"}

    if base == "/model/validate":
        model = _model_body(params)
        if model is None:
            return 400, {"success": False, "error": "expected a model object with a 'features' list"}
        g, errors = _validate_features(model)
        if g is None:
            return 400, {"success": False, "error": errors[0]}
        return 200, {"success": True, "ok": not errors, "errors": errors, "order": g.topo(),
                     "dims": g.resolved_dims(), "features": len(g),
                     "rejected": len(model.get("features") or []) - len(g),
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
        return 200, {"success": True, **manifest,
                     "note": "structure only: fetch a node's geometry through /assembly/node"}

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
        return 200, {"success": True, "bytes": len(blob), **rec,
                     "roundtrip_self_check": {"all_ok": rt["all_ok"], "checks": rt["checks"]}}

    return None
