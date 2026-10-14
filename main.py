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

DEFAULT_INPUT_RPM = 6000.0            # assumption when the client does not send input_rpm
DEFAULT_GEAR_MATERIAL = "AMS6265"     # aerospace gear steel (VTOL actuator use case)

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))  # cloud hosts (Render/Railway/etc.) inject PORT
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_FIELDS = ["envelope_width", "envelope_height", "envelope_depth", "gear_ratio", "wall_thickness"]


class Handler(BaseHTTPRequestHandler):
    server_version = "IdrisCorePOC/0.1"

    def _cors_headers(self) -> None:
        # Needed so a Lovable-hosted frontend (a different origin/domain) is
        # allowed by the browser to call this API with fetch().
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_ace(self, status: int, payload: dict) -> None:
        """JSON reply for ACE routes: compact, gzip when the browser accepts it (FEA surfaces are large)."""
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
        if ACE_IMPORT_ERROR and (p in ("/fea", "/solvers", "/solvers/tools", "/verify")
                                 or p.startswith(("/fea/", "/solve/", "/verify/"))):
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

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/health"):
            self._send_json(200, {
                "service": "idris-core-backend",
                "status": "ok",
                "endpoints": {
                    "POST /generate-variants": "requirements in -> up to 5 physics-checked, scored gearbox variants",
                    "POST /generate-assembly": "full 3D assembly of one variant (housing, involute gears, shafts) as separate STL parts",
                    "POST /generate-housing": "generate a parametric gearbox housing shell",
                    "POST /chat": "free-text message -> structured requirements via Claude/Gemini/OpenAI (validated; falls back to the built-in parser)",
                    "GET /chat/providers": "which AI providers are configured on this server",
                    "GET /download/<filename>": "download a previously generated STL file",
                    "POST /fea": "3D static FEA of the housing for one variant (job: poll GET /fea/<job_id>)",
                    "POST /solve/<name>": "run any ACE solver (see GET /solvers)",
                    "GET /solvers": "catalog of ACE solvers: fidelity, input schema, benchmarks, limitations",
                    "GET /verify": "solver benchmarks vs published reference values (runs in background, cached; ?refresh=1)",
                },
                "ace_solvers": "ok" if not ACE_IMPORT_ERROR else f"failed to load: {ACE_IMPORT_ERROR}",
            })
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
            self.send_header("Content-Type", "model/stl")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json(404, {"success": False, "error": "not found"})

    def _handle_variants(self, params: dict) -> None:
        """POST /generate-variants: requirements in -> up to 5 scored, physics-checked design variants out."""
        required = ["power_kw", "gear_ratio", "envelope_width", "envelope_height", "envelope_depth"]
        missing = [f for f in required if f not in params]
        if missing:
            self._send_json(400, {"success": False, "error": f"missing required fields: {missing}"})
            return
        try:
            req = Requirements(
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
            )
        except (TypeError, ValueError) as e:
            self._send_json(400, {"success": False, "error": f"invalid parameter type: {e}"})
            return
        try:
            out = generate_variants(req)
        except ValueError as e:
            self._send_json(422, {"success": False, "error": str(e)})
            return
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"success": False, "error": f"variant generation failed: {e}"})
            return

        used_defaults = [k for k in ("input_rpm", "gear_material", "housing_material", "wall_thickness",
                                     "min_sf_bending", "min_sf_contact") if k not in params]
        self._send_json(200, {
            "success": True,
            "requirements": {
                "power_kw": req.power_kw, "input_rpm": req.input_rpm, "gear_ratio": req.gear_ratio,
                "envelope_mm": {"width": req.env_w, "height": req.env_h, "depth": req.env_d},
                "wall_thickness_mm": req.wall_thickness, "gear_material": req.gear_material,
                "housing_material": req.housing_material,
                "min_sf_bending": req.min_sf_bending, "min_sf_contact": req.min_sf_contact,
            },
            "assumed_defaults": used_defaults,
            "variants": out["variants"],
            "diagnostics": out["diagnostics"],
            "suggestions": out["suggestions"],
            "disclaimer": ("Screening-level results from closed-form formulas (Lewis bending, Hertz contact, "
                           "AGMA-style load factors) with representative material data. Not a certified analysis; "
                           "housing structure is not yet checked (FEA is a later milestone)."),
        })


    def _cleanup_old_files(self, max_age_s: int = 3600) -> None:
        now = time.time()
        try:
            for fn in os.listdir(OUTPUT_DIR):
                fp = os.path.join(OUTPUT_DIR, fn)
                if fn.endswith(".stl") and now - os.path.getmtime(fp) > max_age_s:
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
            asm = build_assembly(params["design"], params["kinematics"], params["shaft"], params["housing_params"])
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
            "notes": ["Tooth profile: standard 20 deg involute, no root fillet. No bearings, keyways or fasteners are modelled yet."],
        })

    def do_POST(self):
        base = self.path.split("?")[0]
        is_ace = base == "/fea" or base.startswith("/solve/")
        if base not in ("/generate-housing", "/generate-variants", "/generate-assembly", "/chat") and not is_ace:
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

        if self.path == "/chat":
            try:
                ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
                llm.check_rate(ip)
                res = llm.interpret(params.get("provider"), params.get("message"), params.get("current"),
                                    params.get("variants"), params.get("selected"))
                self._send_json(200, {"success": True, **res})
            except llm.LLMError as e:
                self._send_json(e.status, {"success": False, "error": str(e), "attempts": e.attempts, "fallback": True})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"success": False, "error": "chat failed", "fallback": True})
            return
        if self.path == "/generate-variants":
            self._handle_variants(params)
            return
        if self.path == "/generate-assembly":
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