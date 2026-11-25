"""The steps IDRIS really took in one /chat turn (doc: how IDRIS got here, 25 Nov 2026).

The Studio draws "How IDRIS got here" from `trace` on the /chat answer. A trace row is written at the point where
the work happened — reading the message, checking the brief, the reference lookup, the route, the build — so a step
that never ran is simply not in the list. Nothing here invents a step, a duration or a status: `detail` is copied
from the answer's own fields and `source` names the module that did the work.
"""
from __future__ import annotations

MAX_STEPS = 12
MAX_STEP_CHARS = 80
MAX_DETAIL_CHARS = 240
MAX_SOURCE_CHARS = 80

STEP_READ = "Read the request"
STEP_BRIEF = "Checked the brief"
STEP_REFERENCES = "Searched design references"
STEP_ROUTE = "Picked a build route"


class Trace:
    """The rows of one turn, in the order the work happened."""

    def __init__(self):
        self._rows: list[dict] = []

    def add(self, step: str, detail: str = "", source: str = "", status: str = "done") -> dict:
        row = {"step": str(step or "")[:MAX_STEP_CHARS],
               "status": status if status in ("done", "running", "skipped", "failed") else "done",
               "detail": str(detail or "")[:MAX_DETAIL_CHARS],
               "source": str(source or "")[:MAX_SOURCE_CHARS]}
        self._rows.append(row)
        del self._rows[:-MAX_STEPS]
        return row

    def update(self, step: str, detail: str | None = None, append: str | None = None,
               status: str | None = None, source: str | None = None) -> dict | None:
        """Change a row that already exists (the latest one with this name); a step that did not run is not created."""
        for row in reversed(self._rows):
            if row["step"] != str(step or "")[:MAX_STEP_CHARS]:
                continue
            if detail is not None:
                row["detail"] = str(detail)[:MAX_DETAIL_CHARS]
            if append:
                row["detail"] = (row["detail"] + str(append))[:MAX_DETAIL_CHARS]
            if status in ("done", "running", "skipped", "failed"):
                row["status"] = status
            if source is not None:
                row["source"] = str(source)[:MAX_SOURCE_CHARS]
            return row
        return None

    def rows(self) -> list[dict]:
        return [dict(row) for row in self._rows]


def attach(out, trace: Trace) -> dict:
    """The answer plus `trace`; a non-object answer is returned untouched."""
    if not isinstance(out, dict):
        return out
    return {**out, "trace": trace.rows()}


def record_build(trace: Trace, out) -> None:
    """One build row from the ladder's own fields: the rung it reached, or the failure report it wrote."""
    if not isinstance(out, dict):
        return
    cls = out.get("design_shape_class")
    cls = cls.get("shape_class") if isinstance(cls, dict) else cls
    if cls:
        trace.update(STEP_ROUTE, append=f" · shape class: {cls}")
    report = out.get("design_report") if isinstance(out.get("design_report"), dict) else None
    rung = out.get("rung")
    if rung == "failed" or (report and not out.get("cad")):
        tried = report.get("rungs_tried") if report else None
        why = "; ".join(f'{t.get("rung")}: {str(t.get("error") or "")[:110]}'
                        for t in (tried or []) if isinstance(t, dict))
        if not why:
            why = str((report or {}).get("suggestion") or "the part could not be built")[:200]
        trace.add("Build failed", why, "design_ladder.py", status="failed")
        return
    if rung in ("features", "code"):
        attempts = out.get("cad_attempts") if isinstance(out.get("cad_attempts"), list) else []
        detail = f'rung "{rung}" · fidelity "{out.get("fidelity") or "-"}"'
        if attempts:
            detail += f" · {len(attempts)} repair attempt(s)"
        trace.add(f"Built with {rung}", detail, "design_ladder.py")
