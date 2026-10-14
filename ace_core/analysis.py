"""Analysis endpoints in the UI team's format (spec "Spesifikasi API /fea", 2026-10-13).

    POST /fea            body = variant + user choices  -> {"success", "status", "job_id"} or result
    GET  /fea/{job_id}   -> {"success", "status": queued|running|done|failed, "stage", "progress",
                             "result" | "error"}

Same pattern for /modal, /cfd, /review later: add an entry to ANALYSES.
Internally every analysis is an ordinary ACE solver; this module only adapts the shape.
"""
from . import registry, jobs

SYNC_ELEMENT_HINT = {"coarse"}      # small meshes may be answered directly with ?sync=1


def _fea_result(env):
    o = env["outputs"]
    return {"surface": o["surface"], "yield_mpa": o["yield_mpa"], "mesh": o["mesh"],
            "solve_time_s": o["solve_time_s"], "applied_loads": o["applied_loads"],
            "warnings": env["warnings"],
            # extra (UI may ignore): verdict and engineering summary from the ACE envelope
            "verdict": env["verdict"], "checks": env["checks"], "material": o["material"],
            "summary": {"max_von_mises_mpa": o["max_von_mises_MPa"],
                        "max_displacement_mm": o["max_displacement_mm"],
                        "min_safety_factor": o["min_safety_factor"]},
            "model": o["model"], "solver": f"{env['solver']} {env['version']}",
            "method": env["method"], "limitations": env["limitations"]}


ANALYSES = {
    "/fea": {"solver": "fea_housing_3d", "adapt": _fea_result},
}


def _error_text(env):
    errs = env.get("errors") or ["Unknown solver error"]
    return " ".join(str(e) for e in errs)


def _runner(spec, payload):
    def fn(progress):
        env = registry.run(spec["solver"], payload, on_progress=progress)
        if not env.get("ok"):
            return None, _error_text(env)
        return spec["adapt"](env), None
    return fn


def handle_post(path, payload):
    base, _, query = path.partition("?")
    spec = ANALYSES.get(base.rstrip("/"))
    if spec is None:
        return None
    if not isinstance(payload, dict):
        return 400, {"success": False, "status": "failed", "error": "Request body must be a JSON object."}
    if "sync=1" in query:
        res, err = _runner(spec, payload)(lambda *a, **k: None)
        if err:
            return 400, {"success": False, "status": "failed", "error": err}
        return 200, {"success": True, "status": "done", "result": res}
    job_id = jobs.submit(base, _runner(spec, payload))
    if job_id is None:
        return 503, {"success": False, "status": "failed",
                     "error": "The solver queue is full. Please retry in a minute."}
    return 202, {"success": True, "status": "queued", "job_id": job_id, "stage": "Queued", "progress": 0.0}


def handle_get(path):
    base = path.split("?")[0].rstrip("/")
    for prefix in ANALYSES:
        if base.startswith(prefix + "/"):
            job_id = base[len(prefix) + 1:]
            j = jobs.status(job_id)
            if j is None:
                return 404, {"success": False, "status": "failed",
                             "error": "Job not found (it may have expired or the server restarted). Please run again."}
            body = {"success": j["status"] != "failed", "status": j["status"], "job_id": job_id,
                    "stage": j.get("stage"), "progress": j.get("progress")}
            if j["status"] == "done":
                body["result"] = j["result"]
            if j["status"] == "failed":
                body["error"] = j.get("error")
            return 200, body
    return None
