"""Bagian D2: a calculation report an engineer can trace, for one part.

Every number in the page is copied from the request or from a solver result; the report computes nothing.
Each input row names its source (user / standard table / default + reason), each result names the solver and
the benchmark that solver was checked against, a candidate formula from the reference flow is marked
"unverified candidate — not used for any number", and the last section says what was NOT checked.

Built on audit/report.py: the same escaping and the same style, so two reports from IDRIS look like one tool.
"""
from __future__ import annotations

import html
import os
import time
import uuid

from audit.report import STYLE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
ISO2768 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics", "iso2768.py")


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _num(value, unit: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _e(value)
    txt = f"{value:g}"
    return _e(txt + (" " + unit if unit else ""))


def _table(headers: list, rows: list, empty: str) -> str:
    if not rows:
        return f"<p class='empty'>{_e(empty)}</p>"
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _benchmark(solver: str) -> str:
    """The benchmark the named solver was checked against, read from the registry; unknown stays unknown."""
    try:
        import solvers                     # noqa: F401  (registers every solver once)
        from ace_core import registry
        spec = registry.get(solver)
        if spec is None:
            return "not a registered solver in this build"
        bench = [b.source for b in (spec.benchmarks or [])]
        return bench[0] if bench else "no published benchmark for this solver"
    except Exception as e:
        return f"solver registry unavailable ({type(e).__name__})"


def _inputs(body: dict) -> list:
    rows = []
    for p in (body.get("cad") or {}).get("params") or []:
        if not isinstance(p, dict):
            continue
        src = str(p.get("source") or "not stated")
        if src == "user":
            src = "user (the engineer stated it)"
        elif p.get("standard"):
            src = f"standard table: {p['standard']}"
        elif src.startswith("default"):
            src = src
        rows.append([_e(p.get("label") or p.get("k")), _num(p.get("value"), str(p.get("unit") or "")), _e(src)])
    for p in (body.get("requirements") or {}).get("params") or []:
        if isinstance(p, dict):
            rows.append([_e(p.get("label")), _num(p.get("value"), str(p.get("unit") or "")),
                         _e(p.get("source") or "not stated")])
    return rows


def _material(body: dict) -> list:
    mat = body.get("material") if isinstance(body.get("material"), dict) else {}
    name = mat.get("name") or (body.get("material") if isinstance(body.get("material"), str) else None)
    if not name:
        return []
    rows = [["material", _e(name), _e(mat.get("source") or "the engineer named it")]]
    if mat.get("density_kg_m3") is not None:
        rows.append(["density", _num(mat["density_kg_m3"], "kg/m3"), _e(mat.get("density_source") or
                                                                       "materials table in physics/")])
    if mat.get("yield_mpa") is not None:
        rows.append(["yield strength", _num(mat["yield_mpa"], "MPa"), _e(mat.get("strength_source") or
                                                                        "materials table in physics/")])
    return rows


def _fea(body: dict) -> tuple:
    """-> (settings rows, result rows). Only what the solver returned is printed."""
    pc = body.get("part_check") if isinstance(body.get("part_check"), dict) else None
    if not pc:
        return [], []
    mesh = pc.get("mesh") if isinstance(pc.get("mesh"), dict) else {}
    settings = []
    for key, label in (("elements", "elements"), ("element_type", "element type"), ("nodes", "nodes")):
        if mesh.get(key) is not None:
            settings.append([label, _e(mesh[key]), "from the FEA job"])
    for key, label in (("loads", "loads"), ("supports", "supports")):
        if pc.get(key):
            settings.append([label, _e(pc[key]), "from the request"])
    conv = pc.get("convergence") if isinstance(pc.get("convergence"), dict) else {}
    if conv:
        settings.append(["convergence study", _e("; ".join(f"{k}: {v}" for k, v in conv.items())),
                         _e(conv.get("source") or "the same FEA run at two mesh sizes")])
    results = []
    for key, label, unit in (("von_mises_max_mpa", "maximum von Mises stress", "MPa"),
                             ("sf", "safety factor", ""), ("safety_factor", "safety factor", ""),
                             ("displacement_max_mm", "maximum displacement", "mm"),
                             ("max_displacement_mm", "maximum displacement", "mm")):
        if pc.get(key) is not None:
            results.append([_e(label), _num(pc[key], unit), _e(pc.get("solver") or "part-check"),
                            _e(_benchmark(str(pc.get("solver") or "fea_part_3d")))])
    return settings, results


def _ranking(body: dict) -> list:
    rank = body.get("rank") if isinstance(body.get("rank"), dict) else None
    rows = []
    for r in (rank or {}).get("ranking") or []:
        if isinstance(r, dict):
            rows.append([_e(r.get("name")), _num(r.get("score")),
                         _e(", ".join(f"{k}={v}" for k, v in (r.get("values") or {}).items()))])
    return rows


def _candidates(body: dict) -> list:
    rows = []
    for ref in (body.get("references") or {}).get("items") or []:
        if not isinstance(ref, dict):
            continue
        rows.append([_e(ref.get("title") or ref.get("slug")), _e(ref.get("formula")),
                     _e(ref.get("source") or ref.get("url") or "candidate"),
                     "unverified candidate - not used for any number"])
    return rows


def _not_checked(body: dict) -> list:
    out = [str(x) for x in (body.get("not_checked") or []) if x]
    if not body.get("part_check"):
        out.insert(0, "stiffness, vibration and thermal behaviour of the part")
    out.append("fatigue and buckling: no solver in this build")
    out.append("manufacturability (DFM): the rule table is in the design job, not in this report")
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def build(body: dict) -> dict:
    """-> {html, filename, download_url}. Raises ValueError when the body is not usable."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    request = str(body.get("request") or "").strip()
    cad = body.get("cad") if isinstance(body.get("cad"), dict) else {}
    if not request and not cad:
        raise ValueError("send the request sentence and the part (cad) the report is about")
    settings, results = _fea(body)
    inputs = _inputs(body)
    material = _material(body)
    ranking = _ranking(body)
    candidates = _candidates(body)
    not_checked = _not_checked(body)
    title = str(body.get("title") or (cad.get("title") or "Part"))[:120]
    stamp = time.strftime("%d %b %Y, %H:%M UTC", time.gmtime())

    head = (f"<h1>IDRIS engineering report: {_e(title)}</h1>"
            f"<p class='note'>{_e(stamp)} &middot; Idris Core &middot; units mm unless stated &middot; "
            f"every number below is copied from an input or a solver result; this page computes nothing.</p>")
    sections = [
        ("1. The request", f"<blockquote>{_e(request) or '<em>no sentence was given</em>'}</blockquote>"),
        ("2. Inputs and where each number comes from",
         _table(["input", "value", "source"], inputs,
                "No input values were sent with the part, so none are listed.")),
        ("3. Material", _table(["property", "value", "source"], material,
                               "No material was stated for this part, so no density or strength is claimed.")),
        ("4. Analysis settings",
         _table(["setting", "value", "source"], settings,
                "No finite-element run was sent with this part, so no settings are listed.")),
        ("5. Strength result",
         _table(["result", "value", "solver", "benchmark"], results,
                "No load was given; no strength check was run.")),
    ]
    if ranking:
        sections.append(("6. Ranking of the variants", _table(["variant", "score", "measured"], ranking, "")))
    if candidates:
        sections.append(("7. Candidate formulas (not used)",
                         "<p class='note'>These were found by the reference flow and were NOT used to produce "
                         "any number in this report.</p>" +
                         _table(["reference", "formula", "source", "status"], candidates, "")))
    sections.append(("8. Not checked by IDRIS",
                     "<ul>" + "".join(f"<li>{_e(x)}</li>" for x in not_checked) + "</ul>"))
    sections.append(("9. Notes", "<p class='note'>General tolerances ISO 2768-m unless the drawing says "
                                 "otherwise. Dimensions not listed are taken from the 3D model.</p>"))

    body_html = "".join(f"<h2>{_e(t)}</h2>{c}" for t, c in sections)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>IDRIS report: {_e(title)}</title>"
           f"<style>{STYLE}h1{{font-size:20px}}h2{{font-size:16px;margin-top:22px}}blockquote{{margin:6px 0;"
           f"padding:6px 10px;border-left:3px solid #ccc;color:#333}}.empty{{color:#666}}ul{{margin:6px 0 0 "
           f"18px}}</style></head><body>{head}{body_html}</body></html>")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"report_part_{uuid.uuid4().hex[:8]}.html"
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as fh:
        fh.write(doc)
    return {"html": doc, "filename": filename, "download_url": f"/download/{filename}"}
