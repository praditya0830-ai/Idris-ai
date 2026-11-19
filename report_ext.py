"""Bagian D2 route: POST /report/part -> a traceable HTML report for one part.

Body: {request, cad (params with their source), material, measurements, part_check (FEA result, optional),
       rank (optional), references (optional)} -> {html, filename, download_url}

Numbers are copied, never computed here. Without a load, the strength section says so in one sentence instead
of showing a pass.
"""
from __future__ import annotations

import part_report

POST_ROUTES = ("/report/part",)


def handle_get(path: str):
    return None


def handle_post(base: str, params: dict):
    if base != "/report/part":
        return None
    try:
        out = part_report.build(params)
    except ValueError as e:
        return 400, {"success": False, "error": str(e)}
    except Exception as e:                     # a broken body is a clear error, never a 500 page
        return 400, {"success": False, "error": f"report failed: {type(e).__name__}: {e}"}
    return 200, {"success": True, **out}
