"""Bagian E3/E4b: the assembly job — plan once, design the parts one at a time, show progress and the estimate.

POST /assembly/plan    {"request": "whole bicycle 170 cm, 700c"}      -> plan, estimate, questions (no job yet)
POST /assembly/design  {"request": ..., "plan": <confirmed plan>}     -> job_id, estimate
GET  /assembly/design/<job_id>                                        -> status, parts, eta_s, elapsed_s, assembly
POST /assembly/job/<job_id>/review  {"action": "approve"|"redesign"} -> the review gate: nothing is placed or
                                      assembled until every part is seen and approved (or skipped, which is recorded)

The gate is opt-in: POST /assembly/design runs the old flow (status done, assembly placed) unless it carries
"review_gate": true, which is what the Studio review panel will send. A gate that nobody can answer would leave
the Studio as it is today polling a job that never reaches done.

Parts are designed one at a time on purpose (one AI call per part, a small server): each part gets its own
needs and the interface sizes fixed in the skeleton, so two parts that meet never disagree. Each part goes
through the Bagian F ladder when cad_fallback is in this build, so a part that fails still becomes geometry
and the assembly still finishes.

Honest limit (E7): IDRIS does not analyse the assembly as a system. Geometry, interference and per-part
checks only; the reply carries "Not checked by IDRIS: ...".
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid

import ai_budget
import assembly_plan
import llm

JOB_TTL_S = 3600
REVIEW_TTL_S = 24 * 3600                # a review nobody answered is dropped; it would otherwise sit in memory
REVIEW_STATUS = "awaiting_review"      # every part is designed, nothing is assembled until a human says so
REVIEW_EXPIRED_ERROR = "review expired after 24 h, start the assembly again"
REVIEW_ACTIONS = ("approve", "redesign")
DEFAULT_MAX_JOBS = 1                    # review item 3: one assembly designs parts at a time by default
QUEUE_MAX = 2                           # a job that would be the third one waiting is refused (503), not queued
_JOBS: dict = {}
_EXPIRED: dict = {}                     # job ids whose review ran out, so a GET can say why instead of "unknown"
_LOCK = threading.Lock()
_AVG = {"n": 0, "total": 0.0}          # running average design time of a part, in seconds
_RUNNING = {"n": 0}
_SLOTS = threading.Condition()          # one slot per job that is allowed to design parts right now


def max_jobs() -> int:
    """IDRIS_ASSEMBLY_MAX_JOBS: how many assembly jobs may design parts at the same time (default 1)."""
    try:
        return max(1, int(os.environ.get("IDRIS_ASSEMBLY_MAX_JOBS", DEFAULT_MAX_JOBS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_JOBS


def running_jobs() -> int:
    return _RUNNING["n"]


def queued_jobs() -> int:
    with _LOCK:
        return sum(1 for j in _JOBS.values() if j.get("status") == "queued")


def _is_expired(job: dict, now: float) -> bool:
    """A job that has waited at the gate longer than REVIEW_TTL_S is stale: nobody is going to answer it."""
    return (job.get("status") == REVIEW_STATUS and job.get("review_since") is not None
            and now - job["review_since"] > REVIEW_TTL_S)


def _expire_reviews(now: float) -> None:
    """Drop stale gate jobs and remember why (review WAJIB 2). The caller holds _LOCK."""
    for old, j in list(_JOBS.items()):
        if _is_expired(j, now):
            _JOBS.pop(old, None)
            _EXPIRED[old] = now


def review_expired(job_id: str) -> bool:
    """True when this job was dropped because its review was never answered."""
    return job_id in _EXPIRED


def _acquire_slot() -> None:
    """Wait until fewer than max_jobs() assembly jobs are designing: each one spends AI money per part."""
    with _SLOTS:
        while _RUNNING["n"] >= max_jobs():
            _SLOTS.wait(timeout=1.0)
        _RUNNING["n"] += 1


def _release_slot() -> None:
    with _SLOTS:
        _RUNNING["n"] = max(0, _RUNNING["n"] - 1)
        _SLOTS.notify_all()

NOT_CHECKED = ("kinematics of the assembly, loads between parts, and the assembly as one system: IDRIS checks "
               "geometry, interference and each part on its own")
FRAME_CONTRACT = ("a part is placed at origin_mm and turned so its own +Z points along axis and its own +X along "
                  "x_axis; y_axis = axis x x_axis. x_axis_source says whether the plan chose x_axis or it was derived "
                  "(derived means the part could have been turned differently about axis)")


def average_s() -> float:
    return round(_AVG["total"] / _AVG["n"], 1) if _AVG["n"] else 60.0


def _record_part_time(seconds: float) -> None:
    _AVG["n"] += 1
    _AVG["total"] += max(0.1, float(seconds))


def _designer(prompt: str, provider=None, context=None) -> dict:
    """One part through the design ladder. Bagian F's ladder when it is in this build, else the full path.

    `context` carries the part_spec read from the plan (Tahap A1), so a part of an assembly is built from its own
    numbers by the feature rung instead of starting life as AI code.
    """
    try:
        import cad_fallback
        return cad_fallback.design(prompt, context, provider)
    except ImportError:                    # F not merged yet: the full path only, and it says so
        import cad_codegen
        r = cad_codegen.generate(prompt, {}, provider, None)
        return {**r, "fidelity": "full" if r.get("ok") else None, "validated": False,
                "not_checked": ["the Bagian F ladder is not in this build, so no placeholder was built"]}
    except llm.LLMError as e:
        return {"ok": False, "cad": None, "answer": str(e), "fidelity": None, "validated": False,
                "not_checked": []}


def _takes_context(fn) -> bool:
    """A designer written before this change takes two arguments; its part simply gets no spec."""
    try:
        import inspect
        params = list(inspect.signature(fn).parameters.values())
        return len([x for x in params
                    if x.kind in (x.POSITIONAL_ONLY, x.POSITIONAL_OR_KEYWORD)]) >= 3
    except (TypeError, ValueError):
        return False


def design_parts(plan: dict, designer=None, provider=None, on_event=None) -> tuple:
    """E3: design every unique part in order. -> (parts, assembly). No AI here: the designer is injected."""
    designer = designer or (lambda prompt, prov=None: _designer(prompt, prov))
    to_design = [p for p in plan.get("parts") or [] if not p.get("standard")]
    parts = []
    for p in plan.get("parts") or []:
        parts.append({"id": p["id"], "name": p["name"], "qty": p.get("qty", 1),
                      "status": "queued" if not p.get("standard") else "standard",
                      "fidelity": None, "cad": None, "error": None, "repaired": False})
    index = {x["id"]: x for x in parts}
    for p in to_design:
        slot = index[p["id"]]
        why = ai_budget.gate()             # review item 7b: a job stops, with the reason, when AI is capped
        if why:
            slot.update(status="failed", fidelity=None, cad=None, error=why)
            if on_event:
                on_event("design", f"{p['name']} not designed: {why}"[:200], p["id"])
            continue
        slot["status"] = "designing"
        if on_event:
            on_event("design", f"Designing {p['name']}", p["id"])
        t0 = time.time()
        import part_spec
        spec = part_spec.from_plan(p, plan)
        pctx = {"part_spec": spec, "assembly_part": p.get("id")} if spec else None
        try:
            if _takes_context(designer):
                r = designer(assembly_plan.part_prompt(plan, p), provider, pctx) or {}
            else:
                r = designer(assembly_plan.part_prompt(plan, p), provider) or {}
        except Exception as e:                     # one broken part must not stop the assembly
            traceback.print_exc()
            r = {"ok": False, "cad": None, "answer": f"{type(e).__name__}: {e}"}
        _record_part_time(time.time() - t0)
        cad = r.get("cad") if isinstance(r.get("cad"), dict) else None
        if r.get("ok") and cad:
            slot.update(status="done", fidelity=r.get("fidelity") or "full", cad=cad)
        else:
            fixed, changed = assembly_plan.repair_return(str((cad or {}).get("code") or ""))
            slot.update(status="failed", fidelity=r.get("fidelity"), cad=cad,
                        repaired=changed,
                        error=(r.get("answer") or "the part could not be designed")
                              + (" (IDRIS added the missing return statement locally, so the browser can build "
                                 "this code again)" if changed else ""))
            if changed and cad is not None:
                cad = {**cad, "code": fixed}
                slot["cad"] = cad
            if on_event:
                on_event("design", f"{p['name']}: {slot['error'][:160]}", p["id"])
            continue
        if on_event:
            on_event("design", f"{p['name']} ready ({slot['fidelity']})", p["id"])
    return parts, assembly_of(plan, parts)


def assembly_of(plan: dict, parts: list) -> dict:
    """Place every part at its frame. A pile of parts is not an analysis: the note says what was not checked.

    Every placed part carries the full orientation (origin_mm + axis/x_axis/y_axis as unit vectors), not just a
    direction, so the Studio can rotate a part as well as move it; see assembly_plan.basis for where x_axis comes
    from and why a missing one is marked "derived".
    """
    frames = {f["part_id"]: f for f in plan.get("frames") or []}
    placed = []
    for p in parts:
        fr = assembly_plan.placed_frame(frames.get(p["id"]) or {})
        placed.append({"id": p["id"], "name": p["name"], "status": p["status"],
                       "fidelity": p.get("fidelity"), **fr})
    return {"parts": placed, "interfaces": plan.get("interfaces") or [],
            "frames_complete": all(p["id"] in frames for p in parts),
            "frame_contract": FRAME_CONTRACT,
            "not_checked": NOT_CHECKED}


def _rung_of(part: dict) -> str:
    """Which rung really produced this part: features, code, failed — or standard (bought, not designed)."""
    if part.get("status") == "standard":
        return "standard"
    if part.get("status") == "failed":
        return "failed"
    cad = part.get("cad") if isinstance(part.get("cad"), dict) else {}
    if cad.get("feature_tree"):
        return "features"
    if cad.get("code"):
        return "code"
    return "failed"


def _plan_part(plan: dict, part_id: str) -> dict:
    for p in plan.get("parts") or []:
        if p.get("id") == part_id:
            return p
    return {}


def _part_review(plan: dict, part: dict) -> dict:
    """One part as the engineer sees it at the gate: rung, the sizes with their source, the assumptions, the report."""
    import part_spec
    ppart = _plan_part(plan, part.get("id"))
    spec = part_spec.from_plan(ppart, plan) if ppart else {}
    rung = _rung_of(part)
    out = {"id": part["id"], "name": part.get("name"), "qty": part.get("qty", 1), "rung": rung,
           "dims": [{"k": n.get("k"), "value": n.get("value"), "unit": n.get("unit"),
                     "source": n.get("source"), "method": n.get("method")}
                    for n in (spec.get("numbers") or [])][:8],
           "assumptions": [str(a) for a in (ppart.get("assumptions") or [])][:6],
           "questions": [str(q.get("question") or q.get("k")) for q in (spec.get("questions") or [])][:4],
           "failed_report": None}
    if rung == "failed":
        out["failed_report"] = {"error": part.get("error") or "the part could not be designed",
                                "repaired": bool(part.get("repaired"))}
    return out


def review_of(job: dict) -> dict:
    """The gate payload: one row per part, and the list of failed parts that still block assembly."""
    skipped = list(job.get("skipped") or [])
    rows = [_part_review(job.get("plan") or {}, p) for p in job.get("parts") or []]
    blocked = [r["id"] for r in rows if r["rung"] == "failed" and r["id"] not in skipped]
    return {"parts": rows, "blocked": blocked, "can_assemble": not blocked, "skipped": skipped,
            "redesigns": job.get("redesigns") or [],
            "note": "No part is placed and the assembly is not built until this review is approved.",
            "how_to_continue": ("POST /assembly/job/<job_id>/review with action approve (add \"skip\": [part ids] "
                                "to accept a failed part, which is written into the report), or action "
                                "redesign with part_id and changes {\"dims\": {\"length_mm\": 130}}")}


def _job_public(job: dict) -> dict:
    elapsed = round((job.get("finished") or time.time()) - job["started"], 1)
    remaining = [p for p in job["parts"] if p["status"] in ("queued", "designing")]
    return {"success": True, "job_id": job["id"], "status": job["status"], "request": job["request"],
            "plan": job["plan"], "parts": job["parts"], "assembly": job.get("assembly"),
            "review": job.get("review"), "skipped": job.get("skipped") or [],
            "review_gate": bool(job.get("review_gate")),
            "events": job["events"][-40:], "elapsed_s": elapsed,
            "eta_s": round(len(remaining) * average_s(), 1), "seconds_per_part": average_s(),
            "not_checked": [NOT_CHECKED],
            "error": job.get("error"), "stopped": job.get("stopped")}


def run_job(job: dict, designer=None, provider=None) -> None:
    """The worker body. Kept separate so a test can run it without a thread.

    A job waits here for a free slot (review item 3): the maximum number of jobs designing parts at once is
    IDRIS_ASSEMBLY_MAX_JOBS, so several costly assemblies cannot run together by accident.
    """
    _acquire_slot()
    try:
        with _LOCK:
            job.update(status="running", started=job.get("started") or time.time())
        try:
            parts, asm = design_parts(job["plan"], designer, provider,
                                      on_event=lambda step, msg, pid=None: job["events"].append(
                                          {"t": round(time.time(), 2), "step": step, "message": msg, "part": pid}))
            job["parts"] = parts
            if job.get("review_gate") and any(p.get("status") != "standard" for p in parts):
                # The gate: every part is designed, nothing is assembled yet. The assembly is built on approve,
                # so a failed part cannot be quietly placed into a rack of parts nobody looked at.
                with _LOCK:
                    job["review"] = review_of(job)
                    job["review_since"] = time.time()
                    job["status"] = REVIEW_STATUS
                job["events"].append({"t": round(time.time(), 2), "step": "review",
                                      "message": ("Every part is designed. Review them before assembly: "
                                                  + ("all parts are ready" if job["review"]["can_assemble"]
                                                     else "blocked by " + ", ".join(job["review"]["blocked"])))})
            else:
                job.update(assembly=asm, status="done", finished=time.time(), stopped=ai_budget.gate())
        except Exception as e:
            traceback.print_exc()
            job.update(status="failed", error=f"{type(e).__name__}: {e}", finished=time.time())
    finally:
        _release_slot()


def start_job(request: str, plan: dict, provider=None, designer=None, review_gate: bool = False):
    """-> job_id, or None when the queue is full (the route answers 503; nothing is designed).

    review_gate=False keeps the flow the live Studio already polls (status done + assembly); the gate is opt-in
    until the Studio review panel is in production.
    """
    if queued_jobs() >= QUEUE_MAX:
        return None
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "request": request, "plan": plan, "parts": [], "events": [], "status": "queued",
           "started": time.time(), "finished": None, "error": None, "assembly": None, "stopped": None,
           "provider": provider, "designer": designer, "review": None, "skipped": [], "redesigns": [],
           "review_gate": bool(review_gate), "review_since": None}
    with _LOCK:
        now = time.time()
        _JOBS[job_id] = job
        _expire_reviews(now)
        for old, j in list(_JOBS.items()):
            if j.get("finished") and now - j["finished"] > JOB_TTL_S:
                _JOBS.pop(old, None)
        for old, t in list(_EXPIRED.items()):
            if now - t > JOB_TTL_S:
                _EXPIRED.pop(old, None)
    t = threading.Thread(target=run_job, args=(job, designer, provider), daemon=True)
    t.start()
    return job_id


def _job(job_id: str):
    with _LOCK:
        _expire_reviews(time.time())
        return _JOBS.get(job_id)


def status(job_id: str):
    job = _job(job_id)
    return None if job is None else _job_public(job)


def _clean_changes(changes) -> tuple:
    """-> (dims, error). A changed size is a number the engineer stands behind, so it is sourced \"you\"."""
    raw = changes if isinstance(changes, dict) else {}
    flat = raw.get("dims") if isinstance(raw.get("dims"), dict) else raw
    source = str(raw.get("source") or "you").strip()
    if source != "you":
        return {}, f"a size changed by hand is sourced \"you\", not '{source}'"
    dims = {}
    for k, v in (flat or {}).items():
        if str(k) in ("source", "dims"):
            continue
        try:
            dims[str(k)] = float(v)
        except (TypeError, ValueError):
            return {}, f"{k} is not a number: '{v}'"
    return dims, None


def _unknown_job(job_id: str) -> dict:
    """404 body: an expired review says so, because "unknown job" would send the engineer looking for a bug."""
    if review_expired(job_id):
        return {"success": False, "expired": True, "error": REVIEW_EXPIRED_ERROR}
    return {"success": False, "error": f"unknown assembly job '{job_id[:40]}' (jobs are kept one hour)"}


def review_job(job_id: str, params: dict):
    """The gate's door: approve (and assemble) at once; a redesign answers 202 and comes back to the gate."""
    job = _job(job_id)
    if job is None:
        return 404, _unknown_job(job_id)
    action = str(params.get("action") or "").strip().lower()
    if action not in REVIEW_ACTIONS:
        return 400, {"success": False, "error": "action must be \"approve\" or \"redesign\""}
    with _LOCK:
        if job.get("status") != REVIEW_STATUS:
            return 409, {"success": False, "error": f"the job is {job.get('status')}, not {REVIEW_STATUS}",
                         "status": job.get("status")}
        skip = params.get("skip") or []
        if isinstance(skip, str):
            skip = [skip]
        skip = [str(s) for s in skip]
        known = {p["id"] for p in job.get("parts") or []}
        unknown = [s for s in skip if s not in known]
        if unknown:
            return 400, {"success": False, "error": "skip names a part this job does not have",
                         "unknown": unknown}
        job["skipped"] = sorted(set(job.get("skipped") or []) | set(skip))
        review = review_of(job)
        if action == "approve":
            if review["blocked"]:
                return 409, {"success": False, "blocked": review["blocked"], "review": review,
                             "error": ("These parts failed and were not skipped: "
                                       + ", ".join(review["blocked"])
                                       + ". Redesign them, or approve again with \"skip\": [their ids] to "
                                         "accept them and record that in the report.")}
            job["status"] = "assembling"
            job["assembly"] = assembly_of(job["plan"], job["parts"])
            job["finished"] = time.time()
            job["status"] = "done"
            job["stopped"] = ai_budget.gate()
            job["review"] = review_of(job)
            job["events"].append({"t": round(time.time(), 2), "step": "review",
                                  "message": "Approved: the parts are placed and the assembly is built"
                                             + (f"; skipped {', '.join(job['skipped'])}" if job["skipped"] else "")})
            return 200, _job_public(job)
        # redesign: one part, with the engineer's own sizes, then back to the gate
        part_id = str(params.get("part_id") or "").strip()
        part = next((p for p in job.get("parts") or [] if p["id"] == part_id), None)
        if part is None:
            return 400, {"success": False, "error": "part_id is required and must name a part of this job"}
        ppart = _plan_part(job["plan"], part_id)
        if ppart.get("standard"):
            return 400, {"success": False, "error": f"{part_id} is a standard part: it is bought, not redesigned"}
        dims, err = _clean_changes(params.get("changes"))
        if err:
            return 400, {"success": False, "error": err}
        if not dims:
            return 400, {"success": False, "error": "changes must carry at least one size to redesign with"}
        ppart["dims"] = {**(ppart.get("dims") if isinstance(ppart.get("dims"), dict) else {}), **dims}
        ppart["sources"] = {**(ppart.get("sources") if isinstance(ppart.get("sources"), dict) else {}),
                            **{k: "you" for k in dims}}
        job["status"] = "redesigning"
        part.update(status="designing", error=None, repaired=False)
        job["events"].append({"t": round(time.time(), 2), "step": "review",
                              "message": (f"Redesigning {part.get('name') or part_id} with the engineer's sizes"),
                              "part": part_id})
    # Review WAJIB 3: the design runs off the request thread, in its own assembly slot, so a slow designer cannot
    # hang the HTTP call and a hand redesign cannot race an assembly that is already spending AI money.
    threading.Thread(target=_redesign_worker,
                     args=(job, part, ppart, part_id, dims, job.get("designer") or _designer,
                           job.get("provider")),
                     daemon=True).start()
    return 202, {**_job_public(job),
                 "how_to_continue": f"poll GET /assembly/design/{job_id} until status is {REVIEW_STATUS}"}


def _redesign_worker(job: dict, part: dict, ppart: dict, part_id: str, dims: dict, designer, provider) -> None:
    """Redesign one part with the engineer's sizes, then put the job back at the gate.

    It waits for an assembly slot like a whole job does: what spends AI money is the number of designs running
    at once, and a redesign is a design.
    """
    _acquire_slot()
    try:
        spec = None
        try:
            import part_spec
            spec = part_spec.from_plan(ppart, job["plan"])
        except Exception:
            spec = None
        ctx = {"part_spec": spec, "assembly_part": part_id} if spec else None
        t0 = time.time()
        try:
            if _takes_context(designer):
                r = designer(assembly_plan.part_prompt(job["plan"], ppart), provider, ctx) or {}
            else:
                r = designer(assembly_plan.part_prompt(job["plan"], ppart), provider) or {}
        except Exception as e:                 # a broken redesign returns to the gate with the reason
            traceback.print_exc()
            r = {"ok": False, "cad": None, "answer": f"{type(e).__name__}: {e}"}
        _record_part_time(time.time() - t0)
        cad = r.get("cad") if isinstance(r.get("cad"), dict) else None
        if r.get("ok") and cad:
            part.update(status="done", fidelity=r.get("fidelity") or "full", cad=cad)
        else:
            part.update(status="failed", fidelity=r.get("fidelity"), cad=cad, repaired=False,
                        error=r.get("answer") or "the part could not be redesigned")
        with _LOCK:
            job["redesigns"].append({"part_id": part_id, "changes": dims, "source": "you",
                                      "rung": _rung_of(part), "error": part.get("error")})
            job["review"] = review_of(job)
            job["status"] = REVIEW_STATUS
            job["review_since"] = time.time()
            job["events"].append({"t": round(time.time(), 2), "step": "review",
                                  "message": (f"{part.get('name') or part_id} redesigned with "
                                              + ", ".join(f"{k} = {v:g} mm" for k, v in dims.items())
                                              + "; back at the review gate"), "part": part_id})
    finally:
        _release_slot()


# ------------------------------------------------------------------ routes
POST_ROUTES = ("/assembly/plan", "/assembly/design")


def handle_get(path: str):
    p = path.split("?")[0].rstrip("/")
    if p.startswith("/assembly/job/"):
        job_id = p[len("/assembly/job/"):]
    elif p.startswith("/assembly/design/"):
        job_id = p[len("/assembly/design/"):]
    else:
        return None
    st = status(job_id)
    if st is None:
        return 404, _unknown_job(job_id)
    return 200, st


def handle_post(base: str, params: dict):
    p = str(base or "").rstrip("/")
    if p.startswith("/assembly/job/") and p.endswith("/review"):
        return review_job(p[len("/assembly/job/"):-len("/review")], params)
    if base == "/assembly/plan":
        request = str(params.get("request") or "").strip()
        if not request:
            return 400, {"success": False, "error": "request is required (one sentence describing the assembly)"}
        out = assembly_plan.plan_for(request, provider=params.get("provider"))
        return (200 if out["ok"] else 422), {"success": out["ok"], **out}

    if base == "/assembly/design":
        request = str(params.get("request") or "").strip()
        if not request:
            return 400, {"success": False, "error": "request is required (one sentence describing the assembly)"}
        plan_raw = params.get("plan")
        if plan_raw is None:
            out = assembly_plan.plan_for(request, provider=params.get("provider"))
            if not out["ok"]:
                return 422, {"success": False, "need": "plan", "error": out["error"],
                             "errors": out["errors"], "plan": out["plan"],
                             "how_to_continue": "confirm or correct the plan with POST /assembly/design "
                                                "and the plan object, or split the request"}
            return 200, {"success": True, "need": "confirmation", "plan": out["plan"],
                         "estimate": out["estimate"], "questions": out["plan"].get("questions") or [],
                         "how_to_continue": "send the same request with this plan to start the job"}
        plan, errs = assembly_plan.validate(plan_raw if isinstance(plan_raw, dict) else {})
        if errs:
            return 422, {"success": False, "error": "; ".join(errs[:4]), "errors": errs}
        if params.get("confirm"):
            plan = assembly_plan.confirm(plan, params.get("confirm"))
        if plan.get("unsourced"):
            # review item 4: a size without a source is never accepted silently, so the job does not start
            return 200, {"success": True, "need": "confirmation", "plan": plan,
                         "unsourced": plan["unsourced"], "questions": plan.get("questions") or [],
                         "how_to_continue": "send the same request with \"confirm\": [the ids you accept], or "
                                            "send the plan again with a source for each size (user, "
                                            "standard: <name>, default: <reason>)"}
        job_id = start_job(request, plan, params.get("provider"),
                           review_gate=bool(params.get("review_gate")))
        if job_id is None:
            return 503, {"success": False, "error": "The assembly queue is full. Please retry in a minute.",
                         "running": running_jobs(), "max_jobs": max_jobs(), "queued": queued_jobs()}
        return 200, {"success": True, "job_id": job_id, "status": "queued", "plan": plan,
                     "estimate": assembly_plan.estimate(plan),
                     "poll": f"/assembly/design/{job_id}"}
    return None
