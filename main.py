"""
Idris Core — POC Backend (Step 1: Geometry Kernel + API)
==========================================================

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
from part_view import PART_VIEW_HTML

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


# rest of file unchanged; only add /model/plan to POST_ROUTES and the dedicated handler branch below
