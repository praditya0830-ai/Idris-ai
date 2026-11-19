"""One design as a background job (Bagian G): plan -> code -> measure -> DFM -> review -> FEA -> repair.

Why a job: the HTTP path is cut at about 85 s by the proxy, and a design that is worth checking needs more
than that. This runs on the existing single-worker queue (`ace_core/jobs.py`), writes a readable log in
`events`, and is polled through `GET /design/job/<id>`.

Honest split of work, because the solver does not run the CAD kernel:
  * code      — the AI writes replicad code (`cad_codegen.generate`), the server checks it statically;
  * build     — the kernel runs in the browser. It counts as *checked* only when the client sends the kernel
                measurements with the request (`measurements`); otherwise the step says so;
  * review    — needs renders from the browser, so it is `not_checked` unless a reviewer is passed in;
  * FEA       — needs a mesh, so it runs only when a `fea` callable is passed in (Part Stress in the browser,
                or a test double); a step that did not run is never reported as a pass.
No number here is invented: every value in `checks` comes from `cad_codegen`, `physics/dfm_rules.py`, or the
caller's kernel/FEA numbers, and every limit carries its source.
"""
from __future__ import annotations

import contextlib
import os
import re
import threading
import time

from ace_core import jobs
from physics import dfm_rules

# Bagian G1: loose defaults, so quality is not limited by the old 85 s / 45 s / 80-line budget.
DEFAULTS = {"IDRIS_CAD_CALL_S": 120.0, "IDRIS_PART_BUDGET_S": 600.0,
            "IDRIS_CAD_MAX_LINES": 200.0, "IDRIS_DESIGN_ROUNDS": 4.0}
STEPS = ("plan", "code", "build", "measure", "dfm", "review", "fea", "repair")

# Sent with every design request, digit-free so it can never be mistaken for an engineer's size: the workspace
# replays the build order, and the old "keep it short" habit is lifted by IDRIS_CAD_MAX_LINES (Bagian G1/G4).
_STEP_ASK = ("\n\nBuild order: the workspace builds the part one step at a time, so put one short comment before "
             "each feature: // step: base plate, // step: holes, // step: ribs (two to six steps). The server allows "
             "code up to the line limit in IDRIS_CAD_MAX_LINES, so do not cut the design short to stay small.")

_STORE: dict = {}
_LOCK = threading.Lock()
_DURATIONS: list = []                      # seconds per finished design, for the E4b/G1 estimate


def setting(name: str) -> float:
    """A limit from the environment, with the loose default Bagian G asks for (never a crash on a typo)."""
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else DEFAULTS[name]
    except (TypeError, ValueError):
        return DEFAULTS[name]


@contextlib.contextmanager
def _call_budget():
    """Bagian G1: one AI call gets more room than the old 45 s default, for this call only.

    cad_codegen reads IDRIS_CAD_CALL_S at call time, so the job lifts the default around the call and puts the
    environment back afterwards: an operator's own value still wins, and nothing else in the process sees it.
    """
    key = "IDRIS_CAD_CALL_S"
    if key in os.environ:
        yield
        return
    os.environ[key] = str(int(DEFAULTS["IDRIS_CAD_CALL_S"]))
    try:
        yield
    finally:
        os.environ.pop(key, None)


def average_design_s() -> float:
    """Running average of a finished design, or 60 s before the first one (G1/E4b: an estimate, not a promise)."""
    with _LOCK:
        done = list(_DURATIONS)
    return round(sum(done) / len(done), 1) if done else 60.0


def estimate_s(unique_parts: int = 1, standard_parts: int = 0) -> dict:
    """What to tell the engineer before starting: parts to design x measured average, standard parts are free."""
    per = average_design_s()
    unique_parts = max(0, int(unique_parts or 0))
    return {"parts_to_design": unique_parts, "standard_parts": max(0, int(standard_parts or 0)),
            "per_part_s": per, "estimate_s": round(unique_parts * per, 1)}


# ------------------------------------------------------------------ the log (G3)
def _append(job_id, event: dict, on_event=None) -> None:
    if job_id:
        with _LOCK:
            box = _STORE.setdefault(job_id, {"events": [], "result": None, "created": time.time()})
            box["events"].append(event)
    if on_event:
        on_event(event)


# ------------------------------------------------------------------ build order (G4)
_STEP_LINE = re.compile(r"^\s*//\s*step\s*:\s*(.+?)\s*$", re.M)


def split_steps(code: str) -> list:
    """`// step: name` markers -> [{name, code_until_here}], the order the workspace replays.

    Text before the first marker becomes its own step, so a part with no markers still has one step and never
    looks empty. The code is cumulative: each entry builds everything up to that point.
    """
    code = str(code or "")
    if not code.strip():
        return []
    marks = [(m.start(), m.group(1)[:40]) for m in _STEP_LINE.finditer(code)]
    if not marks:
        return [{"name": "part", "code_until_here": code}]
    steps = []
    if code[:marks[0][0]].strip():
        steps.append({"name": "setup", "code_until_here": code[:marks[0][0]].strip("\n")})
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(code)
        steps.append({"name": name, "code_until_here": code[:end].rstrip("\n")})
    return steps


# ------------------------------------------------------------------ step 3: measure
def measure_check(cad: dict, measurements: dict | None = None) -> dict:
    """Engineer sizes must be in the model; the built solid is measured by the kernel.

    Without a kernel measurement the answer says so (`not_checked`) instead of claiming the size matches.
    """
    expect = [e for e in (cad.get("expect") or []) if isinstance(e, dict)]
    user = [p for p in (cad.get("params") or []) if isinstance(p, dict) and p.get("source") == "user"]
    checks = [{"id": "engineer_sizes", "check": "engineer sizes in the model", "status": "pass",
               "value": len(user), "unit": "size(s)", "source": "the request",
               "message": f"{len(user)} size(s) from the engineer are in the model and are never rewritten"}]
    if not isinstance(measurements, dict) or not measurements:
        if expect:
            checks.append({"id": "kernel_size", "check": "measured size", "status": "not_checked",
                           "message": "no kernel measurement was sent, so the built solid was not measured here"})
        return {"checks": checks, "ok": True, "failed": []}
    bad = []
    for e in expect:
        axis, want = e.get("axis"), e.get("value")
        got = measurements.get(axis)
        if axis not in ("x", "y", "z") or want is None or got is None:
            continue
        tol = float(e.get("tol") or 0.5)
        try:
            if abs(float(got) - float(want)) > tol:
                bad.append({"label": str(e.get("label") or axis), "want": float(want),
                            "got": round(float(got), 3), "tol": tol})
        except (TypeError, ValueError):
            continue
    checks.append({"id": "kernel_size", "check": "measured size", "status": "fail" if bad else "pass",
                   "value": len(expect), "unit": "size(s)", "source": "the CAD kernel in the browser",
                   "message": ("measured " + ", ".join(f"{b['label']} {b['got']:g} mm vs {b['want']:g} mm"
                                                       for b in bad) if bad else
                               f"{len(expect)} engineer size(s) match the built solid")})
    return {"checks": checks, "ok": not bad, "failed": bad}


# ------------------------------------------------------------------ step 7: the label (G3)
def trust_label(checks: list, load_given: bool, fea_ran: bool) -> str:
    """Validated / Geometry only / Not validated — decided from the checks that actually ran."""
    if any(c.get("status") == "fail" for c in checks):
        return "Not validated"
    if load_given:
        return "Validated" if fea_ran else "Not validated"
    return "Geometry only"


def not_checked_text(checks: list) -> list:
    """Everything IDRIS did not check, in one place, so a report can print it without guessing."""
    return [f"{c['check']}: {c.get('message') or 'not checked'}"
            for c in checks if c.get("status") == "not_checked"]


# ------------------------------------------------------------------ one design, in rounds
def run(request: str, context: dict | None = None, provider: str | None = None, review=None, fea=None,
        fea_trial=None, cad_fn=None, measurements: dict | None = None, on_event=None, job_id=None) -> dict:
    """The validation circle. Returns the last design with its log, checks and label.

    `fea(cad)` -> {"sf": number, "numbers": {...}} and `fea_trial(cad, target)` -> {"param","from","to","sf"}
    are injected because FEA needs a mesh (the browser builds it); without them the step is honestly
    `not_checked`. `review(cad)` -> {"verdict": "ok"|"revise", "notes": [...]} likewise needs renders.
    """
    if cad_fn is None:
        import cad_codegen
        cad_fn = lambda msg, ctx, prov: cad_codegen.generate(msg, ctx, prov)     # noqa: E731
    rounds = max(1, int(setting("IDRIS_DESIGN_ROUNDS")))
    budget = setting("IDRIS_PART_BUDGET_S")
    start = time.time()
    ctx = dict(context or {})
    msg_for_ai = str(request) + _STEP_ASK
    events, checks, cad = [], [], None
    label, answer, proposal = "Not validated", "", None
    rnd, prev_user = 0, None

    def event(step: str, **kw) -> None:
        e = {"t": round(time.time() - start, 2), "step": step, "round": rnd}
        e.update(kw)
        events.append(e)
        _append(job_id, e, on_event)

    event("plan", status="running", message=f"request: {str(request)[:160]}",
          numbers={"rounds_max": rounds, "budget_s": budget, "call_s": setting("IDRIS_CAD_CALL_S")},
          source="IDRIS design job")

    for rnd in range(1, rounds + 1):
        if time.time() - start > budget:
            event("code", status="fail", message=f"the {budget:g} s budget was used up before round {rnd}")
            break
        with _call_budget():
            got = cad_fn(msg_for_ai, ctx, provider)
        cad = got.get("cad") if isinstance(got, dict) else None
        if not cad or not got.get("ok"):
            message = str((got or {}).get("answer") or "the AI could not write code that builds")[:300]
            event("code", status="fail", message=message, round=rnd)
            answer = message
            break
        user_now = {p.get("k"): p.get("value") for p in (cad.get("params") or [])
                    if isinstance(p, dict) and p.get("source") == "user"}
        changed = []
        if prev_user:
            for k, was in prev_user.items():
                now = user_now.get(k)
                try:
                    if now is not None and abs(float(now) - float(was)) > 1e-9:
                        changed.append(f"{k} {float(was):g} -> {float(now):g} mm")
                except (TypeError, ValueError):
                    continue
        prev_user = user_now
        if changed:
            # Bagian G2: a repair may never change a size the engineer gave; stop and ask instead
            why = "a repair tried to change an engineer size: " + ", ".join(changed)
            checks.append({"id": "engineer_sizes", "check": "engineer sizes kept", "status": "fail",
                           "message": why, "source": "the request"})
            event("repair", status="fail", message=why, numbers={"changed": len(changed)},
                  source="IDRIS design job")
            label = "Not validated"
            answer = (why + ". IDRIS will not change it on its own: say the new size, or let IDRIS change "
                              "something that is not yours.")
            break
        cad["steps"] = split_steps(cad.get("code"))
        event("code", status="pass", message=f"{len(cad['steps'])} step(s), {len(cad.get('params') or [])} parameter(s)",
              numbers={"steps": len(cad["steps"]), "lines": len(str(cad.get("code") or "").splitlines()),
                       "max_lines": int(setting("IDRIS_CAD_MAX_LINES"))},
              source=str(cad.get("read_by") or "AI"))

        # 2) build: the kernel is the browser's, so this is only known when a measurement came with the request
        event("build", status="pass" if measurements else "not_checked",
              message=("kernel measurements were sent with the request" if measurements else
                       "the CAD kernel runs in the browser; no measurement was sent, so the build is not "
                       "confirmed here"), source="the CAD kernel in the browser")

        # 3) measure against what the engineer asked for
        measured = measure_check(cad, measurements)
        checks = list(measured["checks"])
        event("measure", status="fail" if measured["failed"] else
              ("pass" if measurements else "not_checked"),
              message="; ".join(c["message"] for c in measured["checks"])[:300],
              numbers={"failed": len(measured["failed"])}, source="the CAD kernel in the browser")

        # 4) DFM, from the table with a source per rule
        dfm = dfm_rules.check(cad)
        checks.extend(dfm["checks"])
        event("dfm", status="fail" if dfm["failed"] else
              ("warn" if dfm.get("warned") else
               ("pass" if any(c["status"] == "pass" for c in dfm["checks"]) else "not_checked")),
              message=dfm_rules.summary(dfm)[:300],
              numbers={"rules": len(dfm["checks"]), "failed": len(dfm["failed"])},
              source="physics/dfm_rules.py (table with a source per rule)")

        # 5) visual review: renders come from the browser
        if review is not None:
            try:
                rv = review(cad) or {}
            except Exception as e:                       # a review that cannot run is not a pass
                rv = {"verdict": "not_checked", "notes": [f"review could not run: {type(e).__name__}"]}
            status = {"ok": "pass", "revise": "fail"}.get(str(rv.get("verdict")), "not_checked")
            checks.append({"id": "review", "check": "visual review", "status": status,
                           "message": "; ".join(str(n) for n in (rv.get("notes") or []))[:300] or
                                      f"verdict: {rv.get('verdict')}", "source": "/cad/review"})
            event("review", status=status, message=f"verdict: {rv.get('verdict')}",
                  source="/cad/review")
        else:
            checks.append({"id": "review", "check": "visual review", "status": "not_checked",
                           "message": "the review reads renders of the built part, which the browser has",
                           "source": "/cad/review"})
            event("review", status="not_checked", message="no renders were supplied", source="/cad/review")

        # 6) strength, only when the engineer gave a load
        req = cad.get("requirements") if isinstance(cad.get("requirements"), dict) else None
        load_given = bool(req and req.get("loads"))
        fea_ran, fea_out = False, None
        if not load_given:
            checks.append({"id": "fea", "check": "strength (FEA)", "status": "not_checked",
                           "message": "no load was given, so no strength check was run",
                           "source": "/part-check"})
            event("fea", status="not_checked", message="no load in the request; no strength check",
                  source="/part-check")
        elif fea is None:
            checks.append({"id": "fea", "check": "strength (FEA)", "status": "not_checked",
                           "message": "FEA needs the mesh the browser builds; none was supplied",
                           "source": "/part-check"})
            event("fea", status="not_checked", message="no mesh was supplied, so FEA did not run",
                  source="/part-check")
        else:
            try:
                fea_out = fea(cad) or {}
                fea_ran = fea_out.get("sf") is not None
            except Exception as e:                       # a failed analysis never becomes a pass
                fea_out = {"error": f"{type(e).__name__}: {e}"}
                fea_ran = False
            target = req.get("design_factor")
            sf = fea_out.get("sf")
            ok = bool(fea_ran and (target is None or float(sf) >= float(target)))
            checks.append({"id": "fea", "check": "strength (FEA)", "status": "pass" if ok else "fail",
                           "value": sf, "limit": target, "unit": "safety factor",
                           "message": (f"SF {sf:g}" + (f" >= target {float(target):g}" if target is not None else ""))
                           if fea_ran else str(fea_out.get("error") or "FEA produced no safety factor")[:200],
                           "source": str(fea_out.get("source") or "/part-check (fea_part_3d)")})
            event("fea", status="pass" if ok else "fail",
                  message=f"SF {sf:g}" if fea_ran else str(fea_out.get("error"))[:200],
                  numbers={"sf": sf, "target": target}, source=str(fea_out.get("source") or "/part-check"))

            if fea_ran and not ok and fea_trial is not None:
                # a concrete proposal, only with the numbers of an FEA that really ran
                try:
                    trial = fea_trial(cad, target) or {}
                except Exception as e:
                    trial = {"error": f"{type(e).__name__}: {e}"}
                if trial.get("sf") is not None:
                    proposal = (f"SF {float(sf):g} < {float(target):g}; {trial.get('param') or 'the wall'} "
                                f"{trial.get('from')} -> {trial.get('to')} gives SF {float(trial['sf']):g} "
                                f"according to FEA round {rnd}. Change it?")
                    event("repair", status="waiting", message=proposal,
                          numbers={"sf": float(sf), "trial_sf": float(trial["sf"]),
                                   "from": trial.get("from"), "to": trial.get("to")},
                          source="FEA, trial run")

        label = trust_label(checks, load_given, fea_ran)
        if not any(c["status"] == "fail" for c in checks) or proposal:
            break

        # a failed check that a code change can fix: repair, but never touch a size the engineer gave
        if rnd == rounds:
            break
        why = "; ".join(f"{c['check']}: {c.get('message')}" for c in checks if c["status"] == "fail")[:600]
        before = {p.get("k"): p.get("value") for p in (cad.get("params") or [])
                  if isinstance(p, dict) and p.get("source") == "user"}
        ctx = dict(ctx or {})
        # cad_codegen's own counter stays at two repairs; the job's rounds (IDRIS_DESIGN_ROUNDS) decide how many
        # rounds a design gets, so the attempt handed to it never pushes it over its own guard.
        ctx["cad_error"] = {"error": why, "code": cad.get("code"), "params": cad.get("params"),
                            "attempt": min(rnd, 2)}
        event("repair", status="running", message=f"round {rnd} failed: {why[:200]}",
              numbers={"rounds_left": rounds - rnd, "engineer_sizes": len(before)},
              source="IDRIS design job")

    result = {"ok": bool(cad) and label != "Not validated", "fidelity": "full", "cad": cad,
              "checks": checks, "label": label, "rounds": rnd, "events": events,
              "elapsed_s": round(time.time() - start, 1), "proposal": proposal,
              "not_checked": not_checked_text(checks), "budget_s": budget,
              "stop_reason": str(answer).strip() or None}
    # Review item 7b: when a round stopped early (the AI budget, a rejected design), the reason comes first in
    # the answer; the table below it stays the same.
    result["answer"] = ((str(answer).strip() + "\n") if answer else "") + answer_text(result)
    with _LOCK:
        box = _STORE.setdefault(job_id or "-", {"events": [], "result": None, "created": time.time()})
        box["result"] = result
        if job_id:
            _DURATIONS.append(result["elapsed_s"])
    return result


def answer_text(result: dict) -> str:
    """The answer an engineer reads: the validation table, the label, and what was not checked."""
    cad = (result or {}).get("cad") or {}
    lines = []
    title = cad.get("title") or "Part"
    if result.get("proposal"):
        lines.append(result["proposal"])
    lines.append(f"{title}: {result.get('label')} ({result.get('rounds')} round(s), "
                 f"{result.get('elapsed_s')} s).")
    rows = []
    for c in result.get("checks") or []:
        val = "" if c.get("value") is None else f" {c['value']:g}" if isinstance(c["value"], (int, float)) else f" {c['value']}"
        lim = "" if c.get("limit") is None else f" (limit {c['limit']})"
        rows.append(f"- {c['check']}: {c['status']}{val}{lim} — {c.get('message') or ''}".strip())
    if rows:
        lines.append("Checks:")
        lines.extend(rows[:12])
    missing = result.get("not_checked") or []
    if missing:
        lines.append("Not checked by IDRIS: " + "; ".join(missing[:6]))
    if (cad.get("steps") or []) and result.get("label") != "Validated":
        lines.append("Build order: " + ", ".join(s["name"] for s in cad["steps"][:8]) + ".")
    return "\n".join(lines)


# ------------------------------------------------------------------ job plumbing
def submit(request: str, **kw) -> str | None:
    """Start a design on the shared worker. -> job_id, or None when the queue is full."""
    gate = threading.Event()
    holder: dict = {}

    def runner(_progress):
        gate.wait(30)                      # the id is handed over right after submit()
        return run(request, job_id=holder.get("id"), **kw), None

    job_id = jobs.submit("/design/job", runner)
    if job_id is None:
        gate.set()
        return None
    holder["id"] = job_id
    with _LOCK:
        _STORE[job_id] = {"events": [], "result": None, "created": time.time(), "request": str(request)[:300]}
    gate.set()
    return job_id


def status(job_id: str) -> dict | None:
    """What the workspace polls: status, stage, the log so far, and the result once it is done."""
    job = jobs.status(job_id)
    with _LOCK:
        box = dict(_STORE.get(job_id) or {})
    if job is None and not box:
        return None
    status_ = (job or {}).get("status") or ("done" if box.get("result") else "queued")
    body = {"job_id": job_id, "status": status_, "stage": (job or {}).get("stage"),
            "progress": (job or {}).get("progress"), "events": box.get("events") or [],
            "elapsed_s": round(time.time() - box["created"], 1) if box.get("created") else None}
    if status_ == "done":
        body["result"] = (job or {}).get("result") or box.get("result")
    if status_ == "failed":
        body["error"] = (job or {}).get("error")
    if box.get("request"):
        body["request"] = box["request"]
    return body
