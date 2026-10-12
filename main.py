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

import json
import os
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from geometry.housing import generate_housing, export_mesh_to_stl
from physics.gears import Requirements
from physics.variants import generate_variants

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

    def do_OPTIONS(self):
        # CORS preflight request — browsers send this before the real POST
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def log_message(self, fmt, *args):  # quieter default logging
        print("[idris-core-backend]", fmt % args)

    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self._send_json(200, {
                "service": "idris-core-backend",
                "status": "ok",
                "endpoints": {
                    "POST /generate-variants": "requirements in -> up to 5 physics-checked, scored gearbox variants",
                    "POST /generate-housing": "generate a parametric gearbox housing shell",
                    "GET /download/<filename>": "download a previously generated STL file",
                },
            })
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

    def do_POST(self):
        if self.path not in ("/generate-housing", "/generate-variants"):
            self._send_json(404, {"success": False, "error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            params = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self._send_json(400, {"success": False, "error": f"invalid JSON body: {e}"})
            return

        if self.path == "/generate-variants":
            self._handle_variants(params)
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
