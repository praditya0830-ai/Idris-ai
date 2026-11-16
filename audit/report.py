"""Printable report from RunRecords (Fase 4).

Every number in the page comes from a record, and every record names its evidence level and sources.
Text from outside (part names, quote text, URLs) is HTML-escaped: a report that renders user text raw
is an injection hole with a nice font.
"""
from __future__ import annotations

import html

STYLE = ("body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#111}"
         "table{border-collapse:collapse;margin:8px 0 18px}"
         "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}"
         "th{background:#f4f4f4}.bad{color:#a00;font-weight:600}.ok{color:#060}")


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def section(record: dict) -> str:
    rows = [("run_id", record.get("run_id")), ("kind", record.get("kind")),
            ("created", record.get("created_at")), ("status", record.get("status")),
            ("solver", record.get("solver_version")),
            ("kernel", " ".join(str(v) for v in (record.get("kernel") or {}).values() if v)),
            ("inputs_hash", record.get("inputs_hash")),
            ("geometry_hash", record.get("geometry_hash"))]
    ev = record.get("evidence") or {}
    if ev:
        rows.append(("evidence", f"level {ev.get('level')} - {ev.get('level_name', ev.get('method', ''))}"))
        if ev.get("not_yet"):
            rows.append(("not yet", "; ".join(map(str, ev["not_yet"]))))
    body = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows)

    checks = record.get("checks") or []
    check_rows = "".join(
        f"<tr><td>{_e(c.get('id'))}</td><td class='{'ok' if c.get('ok') else 'bad'}'>"
        f"{'pass' if c.get('ok') else 'FAIL'}</td><td>{_e(c.get('detail'))}</td></tr>"
        for c in checks) or "<tr><td colspan='3'>no checks recorded</td></tr>"

    sources = record.get("sources") or []
    source_rows = "".join(
        f"<tr><td>{_e(s.get('k'))}</td><td><a href='{_e(s.get('url'))}' rel='noreferrer'>"
        f"{_e(s.get('url'))}</a></td><td>{_e(s.get('quote'))}</td></tr>" for s in sources
    ) or "<tr><td colspan='3'>no sources: every number here comes from a deterministic solver</td></tr>"

    return (f"<section><h2>Run {_e(record.get('run_id'))[:12]}</h2><table>{body}</table>"
            f"<h3>Checks</h3><table><tr><th>id</th><th>result</th><th>detail</th></tr>{check_rows}</table>"
            f"<h3>Sources</h3><table><tr><th>k</th><th>url</th><th>quote</th></tr>{source_rows}</table>"
            "</section>")


def page(records, title: str = "IDRIS run report") -> str:
    body = "".join(section(r) for r in records) or "<p>no runs</p>"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{_e(title)}</title><style>{STYLE}</style></head>"
            f"<body><h1>{_e(title)}</h1>{body}</body></html>")
