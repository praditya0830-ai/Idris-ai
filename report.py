"""Print-ready engineering report for the Export tab (POST /report).

The browser receives one self-contained HTML document and prints it to PDF
(Ctrl/Cmd+P -> Save as PDF), so the backend needs no PDF dependency. Everything
from the request is escaped: a report can quote user text without injecting HTML.

Expected payload (all fields optional except at least one section):
    {
      "title": "Gearbox S1 - Variant B",
      "project": "VTOL actuator",
      "author": "A. Engineer",
      "summary": {"mass_kg": 6.49, "min_fos": 2.1, "verdict": "PASS"},
      "requirements": {...},
      "variant": {"name": ..., "design": {...}, "kinematics": {...}, "scores": {...}},
      "checks": [{"name": ..., "value": ..., "limit": ..., "unit": ..., "pass": true}],
      "bom": [{"part": ..., "qty": 1, "material": ...}],
      "analyses": {"fea": {"summary": {...}}, "modal": {...}, "thermal": {...}},
      "review": {"counts": {...}, "findings": [...]},
      "solvers": [{"name": ..., "version": ..., "fidelity": ..., "benchmarks": [...]}]
    }
"""
from __future__ import annotations

import html
import json
import time
from collections.abc import Mapping

from ace_core import SOLVER_VERSION

MAX_PAYLOAD_BYTES = 512_000
MAX_TABLE_ROWS = 200

CSS = """
:root{--ink:#111318;--muted:#5b6472;--line:#d8dee8;--soft:#f3f6fa;--accent:#0b6bcb;
--pass:#137a3d;--fail:#b3261e;--warn:#9a6700}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);
font:13.5px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.sheet{max-width:1040px;margin:0 auto;padding:34px 40px 54px}
header{border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:22px}
.brand{font-size:11px;letter-spacing:.24em;color:var(--accent);font-weight:700}
h1{font-size:25px;line-height:1.25;margin:8px 0 4px;letter-spacing:-.01em}
.meta{color:var(--muted);font-size:12px;margin:0}
h2{font-size:15px;margin:26px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}
h3{font-size:13px;margin:16px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
table{width:100%;border-collapse:collapse;font-size:12.5px;page-break-inside:avoid}
th{background:var(--soft);text-align:left;font-weight:600;padding:6px 9px;border:1px solid var(--line)}
td{padding:6px 9px;border:1px solid var(--line);vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 2px}
.card{flex:1 1 150px;border:1px solid var(--line);border-radius:9px;padding:9px 12px;background:var(--soft)}
.card .k{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.card .v{font-size:18px;font-weight:650;margin-top:2px}
.pass{color:var(--pass);font-weight:700}.fail{color:var(--fail);font-weight:700}.warn{color:var(--warn);font-weight:700}
.note{color:var(--muted);font-size:12px;margin:8px 0}
.empty{color:var(--muted);font-style:italic}
footer{margin-top:34px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:11px}
@media print{body{font-size:12px}.sheet{padding:0}h2{page-break-after:avoid}header{page-break-after:avoid}}
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _kv_table(data: Mapping, caption: str | None = None) -> str:
    rows = "".join(f"<tr><th>{_e(k)}</th><td>{_e(_fmt(v))}</td></tr>" for k, v in data.items())
    cap = f"<caption>{_e(caption)}</caption>" if caption else ""
    return f"<table>{cap}{rows}</table>" if rows else '<p class="empty">no data</p>'


def _checks_table(checks: list) -> str:
    rows = []
    for c in checks[:MAX_TABLE_ROWS]:
        if not isinstance(c, Mapping):
            continue
        ok = c.get("pass")
        verdict = ('<span class="pass">PASS</span>' if ok else '<span class="fail">FAIL</span>') \
            if ok is not None else ""
        rows.append(
            "<tr>"
            f"<td>{_e(c.get('name', ''))}</td>"
            f"<td class='num'>{_e(_fmt(c.get('value')))}</td>"
            f"<td class='num'>{_e(_fmt(c.get('limit')))}</td>"
            f"<td>{_e(c.get('unit', ''))}</td><td>{verdict}</td>"
            f"<td>{_e(c.get('note', ''))}</td></tr>")
    if not rows:
        return '<p class="empty">no checks</p>'
    return ("<table><thead><tr><th>Criterion</th><th>Value</th><th>Limit</th><th>Unit</th>"
            "<th>Verdict</th><th>Note</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _bom_table(bom: list) -> str:
    rows = [r for r in bom[:MAX_TABLE_ROWS] if isinstance(r, Mapping)]
    if not rows:
        return '<p class="empty">no BOM</p>'
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(_fmt(r.get(h)))}</td>" for h in headers) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _findings_table(findings: list) -> str:
    rows = []
    for f in findings[:MAX_TABLE_ROWS]:
        if not isinstance(f, Mapping):
            continue
        sev = str(f.get("severity", "")).lower()
        cls = "fail" if sev == "critical" else ("warn" if sev == "warning" else "")
        rows.append(
            "<tr>"
            f"<td><span class='{cls}'>{_e(sev)}</span></td>"
            f"<td>{_e(f.get('check', ''))}</td>"
            f"<td>{_e(f.get('part', ''))}</td>"
            f"<td>{_e(f.get('message', ''))}</td>"
            f"<td>{_e(f.get('fix', ''))}</td></tr>")
    if not rows:
        return '<p class="empty">no findings</p>'
    return ("<table><thead><tr><th>Severity</th><th>Check</th><th>Part</th><th>Finding</th>"
            "<th>Suggested fix</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _analyses_block(analyses: Mapping) -> str:
    blocks = []
    for name, result in analyses.items():
        if not isinstance(result, Mapping):
            continue
        body = result.get("summary") if isinstance(result.get("summary"), Mapping) else result
        blocks.append(f"<h3>{_e(name)}</h3>{_kv_table(body)}")
    return "".join(blocks) or '<p class="empty">no analyses attached</p>'


def _solvers_table(solvers: list) -> str:
    rows = []
    for s in solvers[:MAX_TABLE_ROWS]:
        if not isinstance(s, Mapping):
            continue
        marks = ", ".join(str(b.get("source", b.get("id", ""))) for b in (s.get("benchmarks") or [])
                           if isinstance(b, Mapping))
        rows.append(
            "<tr>"
            f"<td>{_e(s.get('name', ''))}</td><td>{_e(s.get('version', ''))}</td>"
            f"<td>{_e(s.get('fidelity', ''))}</td><td>{_e(s.get('method', ''))}</td>"
            f"<td>{_e(marks)}</td></tr>")
    if not rows:
        return ('<p class="note">No solver evidence attached. The live audit is '
                '<code>GET /verify</code> and <code>GET /solvers</code>.</p>')
    return ("<table><thead><tr><th>Solver</th><th>Version</th><th>Fidelity</th><th>Method</th>"
            "<th>Benchmarks</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _summary_cards(payload: Mapping) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    if not summary:
        summary = {}
        variant = payload.get("variant") if isinstance(payload.get("variant"), Mapping) else {}
        mass = variant.get("mass_kg") if isinstance(variant.get("mass_kg"), Mapping) else {}
        if mass.get("total") is not None:
            summary["mass"] = f"{_fmt(mass['total'])} kg"
        checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
        if checks:
            passed = sum(1 for c in checks if isinstance(c, Mapping) and c.get("pass"))
            summary["checks"] = f"{passed}/{len(checks)} pass"
        review = payload.get("review") if isinstance(payload.get("review"), Mapping) else {}
        if review.get("counts"):
            summary["review"] = _fmt(review["counts"])
    cards = "".join(f'<div class="card"><div class="k">{_e(k)}</div>'
                    f'<div class="v">{_e(_fmt(v))}</div></div>' for k, v in summary.items())
    return f'<div class="cards">{cards}</div>' if cards else '<p class="empty">no summary supplied</p>'


def render(payload: Mapping) -> str:
    """Return a complete HTML document. Raises ValueError for unusable input."""
    if not isinstance(payload, Mapping):
        raise ValueError("body must be a JSON object")
    try:
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        raise ValueError("body must be JSON-serialisable") from None
    if size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"report payload is {size} bytes; keep it under {MAX_PAYLOAD_BYTES}")

    title = str(payload.get("title") or "IDRIS Engineering Report")
    stamp = time.strftime("%d %b %Y, %H:%M UTC", time.gmtime())
    meta = [stamp, f"Idris Core {SOLVER_VERSION}"]
    if payload.get("project"):
        meta.append(f"Project: {payload['project']}")
    if payload.get("author"):
        meta.append(f"Prepared by: {payload['author']}")

    variant = payload.get("variant") if isinstance(payload.get("variant"), Mapping) else {}
    variant_parts = []
    if variant:
        if variant.get("name"):
            variant_parts.append(f"<p class='note'>Variant: <strong>{_e(variant['name'])}</strong></p>")
        for key in ("design", "kinematics", "materials", "scores", "housing_params"):
            if isinstance(variant.get(key), Mapping) and variant[key]:
                variant_parts.append(f"<h3>{_e(key.replace('_', ' '))}</h3>{_kv_table(variant[key])}")
    requirements = payload.get("requirements")
    requirements_html = _kv_table(requirements) if isinstance(requirements, Mapping) and requirements \
        else '<p class="empty">no requirements supplied</p>'
    review = payload.get("review") if isinstance(payload.get("review"), Mapping) else {}
    findings = review.get("findings") if isinstance(review.get("findings"), list) else []
    analyses = payload.get("analyses") if isinstance(payload.get("analyses"), Mapping) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    bom = payload.get("bom") if isinstance(payload.get("bom"), list) else []
    solvers = payload.get("solvers") if isinstance(payload.get("solvers"), list) else []

    sections = [
        ("1. Summary", _summary_cards(payload)),
        ("2. Requirements", requirements_html),
    ]
    if variant_parts:
        sections.append(("3. Variant", "".join(variant_parts)))
    if checks:
        sections.append(("4. Design checks", _checks_table(checks)))
    if analyses:
        sections.append(("5. Analyses", _analyses_block(analyses)))
    if review:
        counts = review.get("counts")
        intro = f"<p class='note'>Findings: {_e(_fmt(counts))}</p>" if counts else ""
        sections.append(("6. Design review", intro + _findings_table(findings)))
    if bom:
        sections.append(("7. Bill of materials", _bom_table(bom)))
    sections.append(("8. Solver evidence", _solvers_table(solvers)))

    body = "".join(f"<section><h2>{_e(heading)}</h2>{content}</section>" for heading, content in sections)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{CSS}</style></head>
<body><div class="sheet">
<header><div class="brand">IDRIS ENGINEERING REPORT</div><h1>{_e(title)}</h1>
<p class="meta">{_e(" · ".join(meta))}</p></header>
<main>{body}</main>
<footer>Generated by Idris Core {_e(SOLVER_VERSION)}. Screening values from deterministic solvers and
declared material data; they support, but do not replace, certified engineering sign-off.
Print this page to PDF for the engineering package.</footer>
</div></body></html>"""
