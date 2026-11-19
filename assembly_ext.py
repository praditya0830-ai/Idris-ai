"""Bagian E3/E4b: the assembly job — plan once, design the parts one at a time, show progress and the estimate.

POST /assembly/plan    {"request": "whole bicycle 170 cm, 700c"}      -> plan, estimate, questions (no job yet)
POST /assembly/design  {"request": ..., "plan": <confirmed plan>}     -> job_id, estimate
GET  /assembly/design/<job_id>                                        -> status, parts, eta_s, elapsed_s, assembly

Parts are designed one at a time on purpose (one AI call per part, a small server): each part gets its own
needs and the interface sizes fixed in the skeleton, so two parts that meet never disagree. Each part goes
through the Bagian F ladder when cad_fallback is in this build, so a part that fails still becomes geometry
and the assembly still finishes.

Honest limit (E7): IDRIS does not analyse the assembly as a system. Geometry, interference and per-part
checks only; the reply carries "Not checked by IDRIS: ...".
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid

import assembly_plan
import llm

JOB_TTL_S = 3600
_JOBS: dict = {}
_LOCK = threading.Lock()
_AVG = {"n": 0, "total": 0.0}          # running average design time of a part, in seconds

NOT_CHECKED = ("kinematics of the assembly, loads between parts, and the assembly as one system: IDRIS checks "
               "geometry, interference and each part on its own")


def average_s() -> float:
    return round(_AVG["total"] / _AVG["n"], 1) if _AVG["n"] else 60.0


def _record_part_time(seconds: float) -> None:
    _AVG["n"] += 1
    _AVG["total"] += max(0.1, float(seconds))


def _designer(prompt: str, provider=None) -> dict:
    """One part through the design ladder. Bagian F's ladder when it is in this build, else the full path."""
    try:
        import cad_fallback
        return cad_fallback.design(prompt, None, provider)
    except ImportError:                    # F not merged yet: the full path only, and it says so
        import cad_codegen
        r = cad_codegen.generate(prompt, {}, provider, None)
        return {**r, "fidelity": "full" if r.get("ok") else None, "validated": False,
                "not_checked": ["the Bagian F ladder is not in this build, so no placeholder was built"]}
    except llm.LLMError as e:
        return {"ok": False, "cad": None, "answer": str(e), "fidelity": None, "validated": False,
                "not_checked": []}


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
        slot["status"] = "designing"
        if on_event:
            on_event("design", f"Designing {p['name']}", p["id"])
        t0 = time.time()
        try:
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
    """Place every part at its frame. A pile of parts is not an analysis: the note says what was not checked."""
    frames = {f["part_id"]: f for f in plan.get("frames") or []}
    placed = []
    for p in parts:
        f = frames.get(p["id"]) or {}
        placed.append({"id": p["id"], "name": p["name"], "origin_mm": f.get("origin_mm") or [0, 0, 0],
                       "axis": f.get("axis") or [0, 0, 1], "status": p["status"],
                       "fidelity": p.get("fidelity")})
    return {"parts": placed, "interfaces": plan.get("interfaces") or [],
            "frames_complete": all(p["id"] in frames for p in parts),
            "not_checked": NOT_CHECKED}


def _job_public(job: dict) -> dict:
    elapsed = round((job.get("finished") or time.time()) - job["started"], 1)
    remaining = [p for p in job["parts"] if p["status"] in ("queued", "designing")]
    return {"success": True, "job_id": job["id"], "status": job["status"], "request": job["request"],
            "plan": job["plan"], "parts": job["parts"], "assembly": job.get("assembly"),
            "events": job["events"][-40:], "elapsed_s": elapsed,
            "eta_s": round(len(remaining) * average_s(), 1), "seconds_per_part": average_s(),
            "not_checked": [NOT_CHECKED],
            "error": job.get("error")}


def run_job(job: dict, designer=None, provider=None) -> None:
    """The worker body. Kept separate so a test can run it without a thread."""
    with _LOCK:
        job.update(status="running", started=job.get("started") or time.time())
    try:
        parts, asm = design_parts(job["plan"], designer, provider,
                                  on_event=lambda step, msg, pid=None: job["events"].append(
                                      {"t": round(time.time(), 2), "step": step, "message": msg, "part": pid}))
        job.update(parts=parts, assembly=asm, status="done", finished=time.time())
    except Exception as e:
        traceback.print_exc()
        job.update(status="failed", error=f"{type(e).__name__}: {e}", finished=time.time())


def start_job(request: str, plan: dict, provider=None, designer=None) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "request": request, "plan": plan, "parts": [], "events": [], "status": "queued",
           "started": time.time(), "finished": None, "error": None, "assembly": None}
    with _LOCK:
        _JOBS[job_id] = job
        for old, j in list(_JOBS.items()):
            if j.get("finished") and time.time() - j["finished"] > JOB_TTL_S:
                _JOBS.pop(old, None)
    t = threading.Thread(target=run_job, args=(job, designer, provider), daemon=True)
    t.start()
    return job_id


def status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    return None if job is None else _job_public(job)


# ------------------------------------------------------------------ routes
POST_ROUTES = ("/assembly/plan", "/assembly/design")


def handle_get(path: str):
    p = path.split("?")[0].rstrip("/")
    if not p.startswith("/assembly/design/"):
        return None
    job_id = p[len("/assembly/design/"):]
    st = status(job_id)
    if st is None:
        return 404, {"success": False, "error": f"unknown assembly job '{job_id[:40]}' (jobs are kept one hour)"}
    return 200, st


def handle_post(base: str, params: dict):
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
        job_id = start_job(request, plan, params.get("provider"))
        return 200, {"success": True, "job_id": job_id, "status": "queued", "plan": plan,
                     "estimate": assembly_plan.estimate(plan),
                     "poll": f"/assembly/design/{job_id}"}
    return None
