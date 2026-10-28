"""
Idris Core — POC Backend (Step 1: Geometry Kernel + API)
=========================================================

API server for the `/generate-housing` endpoint, per the original spec.

Environment note: FastAPI could not be installed here (no outbound access
to PyPI in this sandbox — see README.md for details), so this uses only
Python's built-in `http.server`. The API contract (route, JSON in/out) is
identical to what FastAPI would expose, so swapping to FastAPI later is a
drop-in change, not a redesign.

Run:
    python3 main.py

Then POST to http://localhost:8000/generate-housing — see README.md for a
curl example.
"""

from __future__ import annotations

import os

# Render's free tier gives a fraction of one CPU. Multi-threaded BLAS (used by numpy) then
# fights the CPU quota and gets throttled hard (FEA ran ~50-400x slower than locally).
# One math thread per process is much faster there. Must be set BEFORE numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from geometry.housing import generate_housing, export_mesh_to_stl
from physics.gears import Requirements
from physics.variants import generate_variants
from assembly3d import build_assembly

import numpy as np
import llm

# ACE solver framework (ace_core/ + solvers/): /fea, /solve/<name>, /solvers, /verify.
# Guarded import: if anything in the solver packages is broken, the rest of the
# backend (chat, variants, assembly) keeps running and /verify reports the error.
try:
    from ace_core.http_routes import handle_get as ace_get, handle_post as ace_post, encode_json
    ACE_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    traceback.print_exc()
    ace_get = ace_post = encode_json = None
    ACE_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
from physics.references import (REFERENCES, BENDING_METHOD, LOSS_REFERENCES, resolve as resolve_reference,
                                basis as reference_basis)
try:
    from ace_core import SOLVER_VERSION
except Exception:  # pragma: no cover
    SOLVER_VERSION = "unknown"

DEFAULT_INPUT_RPM = 6000.0            # assumption when the client does not send input_rpm
DEFAULT_GEAR_MATERIAL = "AMS6265"     # aerospace gear steel (VTOL actuator use case)

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))  # cloud hosts (Render/Railway/etc.) inject PORT
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_FIELDS = ["envelope_width", "envelope_height", "envelope_depth", "gear_ratio", "wall_thickness"]


def _occt_status():
    try:
        from physics.occt_export import status
        return status()
    except Exception as e:      # never break /health
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


def _requirements_from(params):
    return Requirements(
        power_kw=float(params["power_kw"]),
        input_rpm=float(params.get("input_rpm", DEFAULT_INPUT_RPM)),
        gear_ratio=float(params["gear_ratio"]),
        env_w=float(params["envelope_width"]),
        env_h=float(params["envelope_height"]),
        env_d=float(params["envelope_depth"]),
        wall_thickness=float(params.get("wall_thickness", 4.0)),
        gear_material=str(params.get("gear_material", DEFAULT_GEAR_MATERIAL)),
        housing_material=str(params.get("housing_material", "Al7075-T6")),
        min_sf_bending=float(params.get("min_sf_bending", 1.4)),
        min_sf_contact=float(params.get("min_sf_contact", 1.1)),
        bending_method=BENDING_METHOD[resolve_reference(params.get("reference"))],
    )


def _waitlist_status():
    try:
        import waitlist
        return waitlist.status()
    except Exception as e:      # never break /health
        return {"error": f"{type(e).__name__}: {e}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "IdrisCorePOC/0.1"

    def _cors_headers(self) -> None:
        # Needed so a Lovable-hosted frontend (a different origin/domain) is
        # allowed by the browser to call this API with fetch().
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict) -> None:
        if isinstance(payload, dict):
            payload = {**payload, "solver_version": SOLVER_VERSION}
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_ace(self, status: int, payload: dict) -> None:
        """JSON reply for ACE routes: compact, gzip when the browser accepts it (FEA surfaces are large)."""
        if isinstance(payload, dict):
            payload = {**payload, "solver_version": SOLVER_VERSION}
        body, extra = encode_json(payload, self.headers.get("Accept-Encoding", ""))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in extra.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _ace_unavailable(self, path: str) -> bool:
        """If the solver framework failed to import, answer ACE paths with a clear error."""
        p = path.split("?")[0]
        if ACE_IMPORT_ERROR and (p in ("/fea", "/modal", "/cfd", "/review", "/solvers", "/solvers/tools", "/verify", "/selfcheck")
                                 or p.startswith(("/fea/", "/modal/", "/cfd/", "/review/", "/solve/", "/verify/"))):
            self._send_json(503, {"success": False, "ok": False,
                                  "error": f"Solver framework failed to load: {ACE_IMPORT_ERROR}"})
            return True
        return False

    def do_OPTIONS(self):
        # CORS preflight request — browsers send this before the real POST
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def log_message(self, fmt, *args):  # quieter default logging
        print("[idris-core-backend]", fmt % args)

    def _waitlist_get(self) -> bool:
        from urllib.parse import urlsplit, parse_qs
        u = urlsplit(self.path)
        if not u.path.startswith("/waitlist"):
            return False
        import waitlist
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/waitlist/options":
                self._send_json(200, {"success": True, **waitlist.options()})
            elif u.path == "/waitlist/stats":
                self._send_json(200, {"success": True, **waitlist.stats(q.get("token"))})
            elif u.path == "/waitlist/export.csv":
                body = waitlist.export_csv(q.get("token"), q.get("include_test") == "1").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="idris-waitlist.csv"')
                self.send_header("Content-Length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json(404, {"success": False, "error": "unknown endpoint"})
        except waitlist.WaitlistError as e:
            self._send_json(e.status, {"success": False, "error": str(e)})
        return True

    def do_GET(self):
        if self._waitlist_get():
            return
        if self.path.split("?")[0] in ("/", "/health"):
            self._send_json(200, {
                "service": "idris-core-backend",
                "status": "ok",
                "endpoints": {
                    "POST /generate-variants": "requirements in -> up to 5 physics-checked, scored gearbox variants",
                    "POST /generate-assembly": "full manufacturable 3D assembly of one variant as separate STL parts + BOM (CSV); body \"export_step\": true or ?step=1 adds a faceted STEP file",
                    "POST /generate-housing": "generate a parametric gearbox housing shell",
                    "POST /chat": "free-text message -> structured requirements via Claude/Gemini/OpenAI (validated; falls back to the built-in parser)",
                    "GET /chat/providers": "which AI providers are configured on this server",
                    "POST /edit-variant": "change one parameter of a variant (wall, face width, material) -> new fully re-checked variant + before/after diff + allowed_range",
                    "GET /design-methods": "how the design is shaped: parametric (available), topology optimisation and computational engineering (planned)",
                    "GET /references": "calculation references the engineer can choose (lewis, iso6336) and their status",
                    "GET /chat/selftest": "QA: 5 bait questions to the live AI with a real thermal result as context",
                    "GET /download/<filename>": "download a previously generated file (STL, BOM CSV, drawing)",
                    "POST /fea": "3D static FEA of the housing for one variant (job: poll GET /fea/<job_id>)",
                    "POST /modal": "natural frequencies + mode shapes of the housing (job: poll GET /modal/<job_id>)",
                    "POST /cfd": "thermal analysis: oil and housing temperatures (job: poll GET /cfd/<job_id>)",
                    "POST /review": "design review: interference, fits, DFM, assembly findings (job: poll GET /review/<job_id>)",
                    "POST /solve/<name>": "run any ACE solver (see GET /solvers)",
                    "GET /solvers": "catalog of ACE solvers: fidelity, input schema, benchmarks, limitations",
                    "GET /selfcheck": "end-to-end check: 3 demo scenarios × generate, assembly, review, FEA, modal, thermal, timed (background; ?refresh=1)",
                    "GET /occt/selftest": "optional OpenCascade kernel: import, exact solids, STEP write/read, volumes, memory (stage 1)",
                    "GET /verify": "solver benchmarks vs published reference values (runs in background, cached; ?refresh=1)",
                    "POST /waitlist": "website sign-up (waitlist + design partner); needs IDRIS_DATA_DIR on a persistent disk",
                    "GET /waitlist/stats?token=": "sign-up numbers for Arya (test sign-ups excluded)",
                    "GET /waitlist/export.csv?token=": "all sign-ups as CSV",
                },
                "solver_version": SOLVER_VERSION,
                "ace_solvers": "ok" if not ACE_IMPORT_ERROR else f"failed to load: {ACE_IMPORT_ERROR}",
                "occt": _occt_status(),
                "waitlist": _waitlist_status(),
            })
            return
        if self.path.split("?")[0] == "/occt/selftest":
            try:
                from physics.occt_export import selftest
                self._send_json(200, selftest())
            except Exception as e:
                self._send_json(200, {"pass": False, "error": f"{type(e).__name__}: {e}"})
            return

        if self._ace_unavailable(self.path):
            return
        if ace_get is not None:
            res = ace_get(self.path)
            if res is not None:
                self._send_ace(*res)
                return

        if self.path.split("?")[0] == "/chat/providers":
            self._send_json(200, {"success": True, "providers": llm.configured(),
                                  "models": {p: llm.model_for(p) for p in llm.PROVIDERS}})
            return

        if self.path.split("?")[0] == "/design-methods":
            from physics.design_methods import DESIGN_METHODS, DEFAULT_METHOD
            self._send_json(200, {"success": True, "default": DEFAULT_METHOD, "methods": DESIGN_METHODS,
                                  "note": "Send \"design_method\" to POST /generate-variants. Only methods with "
                                          "status 'available' generate designs; the others are planned."})
            return

        if self.path.split("?")[0] == "/references":
            self._send_json(200, {"success": True, "references": REFERENCES, "loss_references": LOSS_REFERENCES,
                                  "note": "Send \"reference\": \"lewis\" | \"iso6336\" to POST /generate-variants; "
                                          "send \"loss_model\": \"constant\" | \"iso14179_load\" to POST /cfd."})
            return

        if self.path.split("?")[0] == "/chat/selftest":
            # Browser-friendly QA check (Q-018): S1 Variant B thermal result from the real solver as context,
            # then the five bait questions to the configured AI. Costs 5 AI calls.
            try:
                from ace_core.analysis import handle_post as _ap
                from solvers._example import EXAMPLE_VARIANT
                code, b = _ap("/cfd?sync=1", {**EXAMPLE_VARIANT, "power_kw": 11.19, "input_rpm": 6000})
                r = b.get("result") or {}
                advice = {"topic": "thermal", "status": r.get("verdict"), "oil_temp_c": r.get("oil_temp_c"),
                          "limit_c": r.get("oil_limit_c"), "heat_w": r.get("heat_generated_w"),
                          "surface_area_m2": r.get("surface_area_m2"), "cooling": "natural", "air_speed_ms": 0,
                          "ambient_c": 25, "variant": "Variant B",
                          "options": [{k: o.get(k) for k in ("kind", "label", "change", "oil_temp_c", "pass",
                                                             "added_mass_kg", "assumption")}
                                      for o in (r.get("options") or [])[:8]]}
                ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
                res = llm.selftest({"advice": advice}, ip)
                self._send_json(200, {"success": True, "context_used": advice, "results": res,
                                      "how_to_read": "numbers_verified=false means the AI said a number that is "
                                                     "not in the solver context; the answer then carries a note."})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"success": False, "error": f"selftest failed: {type(e).__name__}: {e}"})
            return

        if self.path.startswith("/chat/try"):
            # Browser-friendly smoke test:  /chat/try?provider=claude&q=15 kW gearbox ratio 8
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            try:
                ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
                llm.check_rate(ip)
                res = llm.interpret((qs.get("provider") or [None])[0], (qs.get("q") or [""])[0])
                self._send_json(200, {"success": True, **res})
            except llm.LLMError as e:
                self._send_json(e.status, {"success": False, "error": str(e), "attempts": e.attempts, "fallback": True})
            return

        if self.path.startswith("/download/"):
            filename = self.path[len("/download/"):]
            filename = os.path.basename(filename)  # prevent path traversal
            filepath = os.path.join(OUTPUT_DIR, filename)
            if not os.path.isfile(filepath):
                self._send_json(404, {"success": False, "error": f"file not found: {filename}"})
                return
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            ctype = {".csv": "text/csv; charset=utf-8", ".step": "application/step", ".stp": "application/step",
                     ".svg": "image/svg+xml", ".pdf": "application/pdf"}.get(os.path.splitext(filename)[1].lower(),
                                                                                "model/stl")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json(404, {"success": False, "error": "not found"})

    def _handle_edit(self, params: dict) -> None:
        """POST /edit-variant: requirements (as /generate-variants) + variant + change -> new re-checked variant."""
        from physics.edit_variant import edit_variant
        v = params.get("variant")
        if not isinstance(v, dict) or not isinstance(v.get("design"), dict):
            self._send_json(400, {"success": False, "error": "variant (one entry of /generate-variants variants) is required"})
            return
        missing = [f for f in ("power_kw", "gear_ratio", "envelope_width", "envelope_height", "envelope_depth")
                   if f not in params]
        if missing:
            self._send_json(400, {"success": False, "error": f"missing required fields: {missing} "
                                                             "(send the same requirements as /generate-variants)"})
            return
        try:
            req = _requirements_from(params)
        except (TypeError, ValueError) as e:
            self._send_json(400, {"success": False, "error": f"invalid parameter type: {e}"})
            return
        goals = params.get("goals") if isinstance(params.get("goals"), dict) else None
        try:
            code, body = edit_variant(req, v, params.get("change"), str(params.get("envelope_basis", "housing")),
                                      goals, float(params.get("ambient_c", 25.0)))
        except ValueError as e:
            code, body = 422, {"success": False, "error": str(e)}
        except Exception as e:
            traceback.print_exc()
            code, body = 500, {"success": False, "error": f"edit failed: {e}"}
        self._send_json(code, body)

    def _handle_variants(self, params: dict) -> None:
        """POST /generate-variants: requirements in -> up to 5 scored, physics-checked design variants out."""
        required = ["power_kw", "gear_ratio", "envelope_width", "envelope_height", "envelope_depth"]
        missing = [f for f in required if f not in params]
        if missing:
            self._send_json(400, {"success": False, "error": f"missing required fields: {missing}"})
            return
        try:
            req = _requirements_from(params)
            reference = resolve_reference(params.get("reference"))
        except (TypeError, ValueError) as e:
            self._send_json(400, {"success": False, "error": f"invalid parameter type: {e}"})
            return
        from physics.design_methods import resolve as resolve_method, DESIGN_METHODS
        method, method_err = resolve_method(params.get("design_method"))
        if method_err:
            self._send_json(422 if method else 400, {"success": False, "error": method_err,
                                                     "design_method": method,
                                                     "available_methods": [k for k, m in DESIGN_METHODS.items()
                                                                           if m["status"] == "available"]})
            return
        goals = params.get("goals")
        if goals is not None and not isinstance(goals, dict):
            self._send_json(400, {"success": False, "error": "goals must be an object, e.g. {\"oil_temp_max_c\": 90}"})
            return
        basis = str(params.get("envelope_basis", "housing"))
        try:
            out = generate_variants(req, envelope_basis=basis, goals=goals or None,
                                    ambient_c=float(params.get("ambient_c", 25.0)),
                                    compare_basis=bool(params.get("compare_basis", True)))
        except ValueError as e:
            self._send_json(422, {"success": False, "error": str(e)})
            return
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"success": False, "error": f"variant generation failed: {e}"})
            return

        alt_vs = [v for op in ((out.get("envelope_choice") or {}).get("options") or []) for v in (op.get("variants") or [])]
        for v in out["variants"] + alt_vs:             # QA Q-016: the reference on every stress number
            v["reference"] = reference
            if isinstance(v.get("stress"), dict):
                v["stress"]["basis"] = reference_basis(reference)
        used_defaults = [k for k in ("input_rpm", "gear_material", "housing_material", "wall_thickness",
                                     "min_sf_bending", "min_sf_contact", "reference") if k not in params]
        self._send_json(200, {
            "success": True,
            "requirements": {
                "power_kw": req.power_kw, "input_rpm": req.input_rpm, "gear_ratio": req.gear_ratio,
                "envelope_mm": {"width": req.env_w, "height": req.env_h, "depth": req.env_d},
                "wall_thickness_mm": req.wall_thickness, "gear_material": req.gear_material,
                "housing_material": req.housing_material,
                "min_sf_bending": req.min_sf_bending, "min_sf_contact": req.min_sf_contact,
                "reference": reference,
            },
            "reference": {"key": reference, **REFERENCES[reference]},
            "design_method": method,
            "output_level": "screening",
            "assumed_defaults": used_defaults,
            "variants": out["variants"],
            "diagnostics": out["diagnostics"],
            "suggestions": out["suggestions"],
            "repairs": out.get("repairs", []),
            "envelope_basis": out.get("envelope_basis", "housing"),
            "envelope_note": out.get("envelope_note"),
            "envelope_choice": out.get("envelope_choice"),
            "goal_summary": out.get("goal_summary"),
            "disclaimer": ("Screening-level results from closed-form formulas (Lewis bending, Hertz contact, "
                           "AGMA-style load factors; shafts: Shigley fatigue, deflection, critical speed; bearings: "
                           "ISO 281 L10; oil temperature: L1 lumped thermal screen) with representative material data. "
                           "Not a certified analysis. Housing stress, vibration and temperature: run /fea, /modal "
                           "and /cfd on a chosen variant."),
        })


    def _cleanup_old_files(self, max_age_s: int = 3600) -> None:
        now = time.time()
        try:
            for fn in os.listdir(OUTPUT_DIR):
                fp = os.path.join(OUTPUT_DIR, fn)
                if fn.endswith((".stl", ".csv", ".step", ".svg", ".pdf")) and now - os.path.getmtime(fp) > max_age_s:
                    os.remove(fp)
        except OSError:
            pass

    def _handle_assembly(self, params: dict) -> None:
        """POST /generate-assembly: one variant (as returned by /generate-variants) -> separate STL parts."""
        missing = [k for k in ("design", "kinematics", "shaft", "housing_params") if not isinstance(params.get(k), dict)]
        if missing:
            self._send_json(400, {"success": False, "error": f"missing or invalid fields: {missing}"})
            return
        try:
            asm = build_assembly(params["design"], params["kinematics"], params["shaft"], params["housing_params"],
                                 params.get("materials") if isinstance(params.get("materials"), dict) else
                                 ({"housing": params["housing_material"]} if params.get("housing_material") else None))
        except (KeyError, TypeError) as e:
            self._send_json(400, {"success": False, "error": f"invalid variant data: {e}"})
            return
        except ValueError as e:
            self._send_json(422, {"success": False, "error": str(e)})
            return
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"success": False, "error": f"assembly failed: {e}"})
            return

        self._cleanup_old_files()
        tag = uuid.uuid4().hex[:8]
        parts_out, all_v, all_f, offset = [], [], [], 0
        for i, part in enumerate(asm["parts"]):
            fname = f"asm_{tag}_{i}.stl"
            export_mesh_to_stl(part["verts"], part["faces"], os.path.join(OUTPUT_DIR, fname), name=part["name"])
            meta = {k: v for k, v in part.items() if k not in ("verts", "faces", "name", "kind")}
            parts_out.append({"name": part["name"], "kind": part["kind"], "triangles": int(part["faces"].shape[0]),
                              "download_url": f"/download/{fname}", "meta": meta})
            all_v.append(part["verts"])
            all_f.append(part["faces"] + offset)
            offset += part["verts"].shape[0]
        bom_name = f"asm_{tag}_bom.csv"
        from physics.bom import bom_csv
        with open(os.path.join(OUTPUT_DIR, bom_name), "w", encoding="utf-8", newline="") as fh:
            fh.write(bom_csv(asm.get("bom") or []))
        step_out = None
        if params.get("export_step") or "step=1" in self.path:
            from physics.step_export import write_step, check_step
            step_name = f"asm_{tag}.step"
            txt, _stats = write_step(asm["parts"], title="Idris gearbox")
            ok_step, msg = check_step(txt)
            with open(os.path.join(OUTPUT_DIR, step_name), "w", encoding="ascii") as fh:
                fh.write(txt)
            step_out = {"filename": step_name, "download_url": f"/download/{step_name}", "self_check": msg,
                        "self_check_ok": ok_step,
                        "note": "Faceted STEP AP214: solid bodies with planar faces (round surfaces faceted, "
                                "overlapping solids of one part not united). Exact-geometry STEP needs an "
                                "OpenCascade kernel (planned)."}
        drawings_out = None
        if params.get("drawings", True):
            try:
                from physics.drawing import make_drawings
                from physics.variant_shafts import shaft_plan as _sp
                plan_ = _sp({"design": params["design"], "kinematics": params["kinematics"], "shaft": params["shaft"],
                             "housing_params": params["housing_params"], "loads": {},
                             "materials": params.get("materials") or {}})
                drawings_out = []
                for sheet, svg in make_drawings(asm, plan_, params.get("materials")).items():
                    fn = f"asm_{tag}_{sheet}.svg"
                    with open(os.path.join(OUTPUT_DIR, fn), "w", encoding="utf-8") as fh:
                        fh.write(svg)
                    drawings_out.append({"sheet": sheet, "filename": fn, "download_url": f"/download/{fn}"})
            except Exception as e:          # never block the assembly on the drawings
                traceback.print_exc()
                drawings_out = {"error": f"drawings unavailable: {e}"}
        combined = f"asm_{tag}_all.stl"
        export_mesh_to_stl(np.vstack(all_v), np.vstack(all_f), os.path.join(OUTPUT_DIR, combined), name="idris_assembly")
        self._send_json(200, {
            "success": True,
            "parts": parts_out,
            "combined": {"filename": combined, "download_url": f"/download/{combined}",
                         "triangles": int(sum(p["triangles"] for p in parts_out))},
            "bbox_mm": asm["bbox"],
            "layout": asm["layout"],
            "warnings": asm["warnings"],
            "notes": list(asm.get("notes", [])) + [
                "Tooth profile: standard 20 deg involute, no root fillet. Bearings, seals, cover bolts and oil plugs "
                "are drawn as simplified envelopes (catalogue size, no balls or threads); keyways and fillets are not drawn."],
            "housing_params": asm.get("housing_params"),
            "bom": asm.get("bom"),
            "step": step_out,
            "drawings": drawings_out,
            "bom_csv": {"filename": bom_name, "download_url": f"/download/{bom_name}"},
        })

    def do_POST(self):
        base = self.path.split("?")[0]
        is_ace = base in ("/fea", "/modal", "/cfd", "/review") or base.startswith("/solve/")
        if base not in ("/generate-housing", "/generate-variants", "/generate-assembly", "/chat", "/edit-variant",
                        "/waitlist", "/visit", "/waitlist/delete") and not is_ace:
            self._send_json(404, {"success": False, "error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            params = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self._send_json(400, {"success": False, "error": f"invalid JSON body: {e}"})
            return

        if is_ace:
            if self._ace_unavailable(self.path):
                return
            res = ace_post(self.path, params)
            if res is None:
                self._send_json(404, {"success": False, "error": "unknown endpoint"})
            else:
                self._send_ace(*res)
            return

        if base in ("/waitlist", "/visit", "/waitlist/delete"):
            import waitlist
            if not isinstance(params, dict):
                self._send_json(400, {"success": False, "error": "body must be a JSON object"})
                return
            ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
            try:
                if base == "/waitlist":
                    self._send_json(*waitlist.signup(params, ip))
                elif base == "/visit":
                    self._send_json(*waitlist.visit(params))
                else:
                    self._send_json(200, waitlist.delete(params.get("token"), params.get("email")))
            except waitlist.WaitlistError as e:
                self._send_json(e.status, {"success": False, "error": str(e)})
            except Exception:
                traceback.print_exc()
                self._send_json(500, {"success": False, "error": "sign-up failed"})
            return

        if base == "/chat":
            try:
                ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
                llm.check_rate(ip)
                res = llm.interpret(params.get("provider"), params.get("message"), params.get("current"),
                                    params.get("variants"), params.get("selected"),
                                    params.get("context") if isinstance(params.get("context"), dict) else None)
                self._send_json(200, {"success": True, **res})
            except llm.LLMError as e:
                self._send_json(e.status, {"success": False, "error": str(e), "attempts": e.attempts, "fallback": True})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"success": False, "error": "chat failed", "fallback": True})
            return
        if base == "/generate-variants":
            self._handle_variants(params)
            return
        if base == "/edit-variant":
            self._handle_edit(params)
            return
        if base == "/generate-assembly":
            self._handle_assembly(params)
            return

        missing = [f for f in REQUIRED_FIELDS if f not in params]
        if missing:
            self._send_json(400, {"success": False, "error": f"missing required fields: {missing}"})
            return

        try:
            envelope_width = float(params["envelope_width"])
            envelope_height = float(params["envelope_height"])
            envelope_depth = float(params["envelope_depth"])
            gear_ratio = float(params["gear_ratio"])
            wall_thickness = float(params["wall_thickness"])
            bore_diameter = float(params["bore_diameter"]) if params.get("bore_diameter") is not None else None
        except (TypeError, ValueError) as e:
            self._send_json(400, {"success": False, "error": f"invalid parameter type: {e}"})
            return

        try:
            result = generate_housing(
                envelope_width=envelope_width,
                envelope_height=envelope_height,
                envelope_depth=envelope_depth,
                gear_ratio=gear_ratio,
                wall_thickness=wall_thickness,
                bore_diameter=bore_diameter,
            )
        except ValueError as e:
            self._send_json(422, {"success": False, "error": str(e)})
            return
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"success": False, "error": f"generation failed: {e}"})
            return

        filename = f"housing_{uuid.uuid4().hex[:8]}.stl"
        filepath = os.path.join(OUTPUT_DIR, filename)
        export_mesh_to_stl(result.verts_mm, result.faces, filepath, name=filename)

        self._send_json(200, {
            "success": True,
            "input_parameters": {
                "envelope_width_mm": envelope_width,
                "envelope_height_mm": envelope_height,
                "envelope_depth_mm": envelope_depth,
                "gear_ratio": gear_ratio,
                "wall_thickness_mm": wall_thickness,
            },
            "computed": {
                "bore_diameter_mm": round(result.bore_diameter_mm, 3),
                "volume_mm3": round(result.volume_mm3, 3),
                "bounding_box_mm": {k: round(v, 3) for k, v in result.bbox_mm.items()},
                "triangle_count": int(result.faces.shape[0]),
            },
            "file": {
                "filename": filename,
                "format": "STL (binary)",
                "server_path": filepath,
                "download_url": f"/download/{filename}",
            },
        })


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Idris Core POC backend running on http://{HOST}:{PORT}")
    print(f"Generated files are saved to: {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()