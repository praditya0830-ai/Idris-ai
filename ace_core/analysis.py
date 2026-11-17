"""Analysis endpoints in the UI team's format (spec "Spesifikasi API /fea", 2026-10-13).

    POST /fea            body = variant + user choices  -> {"success", "status", "job_id"} or result
    GET  /fea/{job_id}   -> {"success", "status": queued|running|done|failed, "stage", "progress",
                             "result" | "error"}

/modal, /cfd (thermal) and /review follow the same pattern (entries in ANALYSES).
Internally every analysis is an ordinary ACE solver; this module only adapts the shape.
"""
from . import registry, jobs, SOLVER_VERSION

SYNC_ELEMENT_HINT = {"coarse"}      # small meshes may be answered directly with ?sync=1


def _evidence(env):
    try:
        from physics.evidence import EVIDENCE
        return EVIDENCE.get(env.get("solver"))
    except Exception:
        return None


def _fea_result(env):
    o = env["outputs"]
    return {"surface": o["surface"], "yield_mpa": o["yield_mpa"], "mesh": o["mesh"],
            "solve_time_s": o["solve_time_s"], "applied_loads": o["applied_loads"],
            "warnings": env["warnings"],
            # extra (UI may ignore): verdict and engineering summary from the ACE envelope
            "verdict": env["verdict"], "checks": env["checks"], "material": o["material"],
            "summary": {"max_von_mises_mpa": o["max_von_mises_MPa"],
                        "max_displacement_mm": o["max_displacement_mm"],
                        "min_safety_factor": o["min_safety_factor"],
                        "margin_of_safety": o.get("margin_of_safety"), "factor_of_safety": o.get("factor_of_safety")},
            "model": o["model"], "solver": f"{env['solver']} {env['version']}",
            "options": o.get("options", []), "outer_size_mm": o.get("outer_size_mm"),
            "fits_envelope": o.get("fits_envelope"), "mesh_convergence": o.get("mesh_convergence"),
            "method": env["method"], "limitations": env["limitations"], "evidence": _evidence(env)}


def _common(env, o):
    return {"evidence": _evidence(env), "warnings": env["warnings"], "verdict": env["verdict"], "checks": env["checks"],
            "solver": f"{env['solver']} {env['version']}", "method": env["method"],
            "limitations": env["limitations"]}


def _modal_result(env):
    o = env["outputs"]
    return {"surface": o["surface"], "mesh": o["mesh"], "solve_time_s": o["solve_time_s"],
            "frequencies_hz": o["frequencies_hz"], "excitations_hz": o["excitations_hz"],
            "eigen_method": o["eigen_method"], "housing_mass_kg": o["housing_mass_kg"],
            "speed_map": o.get("speed_map", []), "options": o.get("options", []),
            "material": o["material"], **_common(env, o)}


def _thermal_result(env):
    o = env["outputs"]
    keep = ("oil_temp_c", "heat_dissipated_w", "heat_generated_w", "max_surface_temp_c",
            "mean_outer_surface_temp_c", "h_oil_w_m2k", "h_outer_convection_w_m2k",
            "h_outer_radiation_w_m2k", "area_outer_m2", "area_wetted_m2", "material", "conductivity_w_mk",
            "target_oil_temp_c", "fixes", "options", "surface_area_m2", "heat_to_housing_w",
            "oil_cooler_w", "housing_box_mm", "housing_scale", "fins", "efficiency_total", "oil_limit_c",
            "goal_report", "best_without_assumption", "closest_without_assumption", "validation",
            "outer_size_mm", "fits_envelope", "loss_model", "power_loss")
    return {"surface": o["surface"], "mesh": o["mesh"], "solve_time_s": o["solve_time_s"],
            **{k: o[k] for k in keep}, **_common(env, o)}


def _review_result(env):
    o = env["outputs"]
    return {"findings": o["findings"], "counts": o["counts"], "checked_parts": o["checked_parts"],
            "solve_time_s": o["solve_time_s"], **_common(env, o)}


def _part_result(env):
    o = env["outputs"]
    # the same findings the chat shows, so a number reviewed in the studio and a number quoted in the
    # chat come from one rule set (ai/review.py)
    from ai import review as REVIEW
    findings = REVIEW.review_part(o)
    return {"surface": o["surface"], "yield_mpa": o["yield_mpa"], "mesh": o["mesh"], "material": o["material"],
            "review": findings, "review_summary": REVIEW.summary(findings),
            "solve_time_s": o["solve_time_s"], "applied_loads": o["applied_loads"], "reaction_n": o["reaction_N"],
            "summary": {"max_von_mises_mpa": o["max_von_mises_MPa"], "von_mises_p995_mpa": o["von_mises_p995_MPa"],
                        "max_displacement_mm": o["max_displacement_mm"], "min_safety_factor": o["min_safety_factor"],
                        "margin_of_safety": o["margin_of_safety"], "factor_of_safety": o["design_factor"],
                        "mass_kg": o["mass_kg"]},
            "volume": o["volume"], "equilibrium_error": o["equilibrium_error"],
            "mesh_convergence": o.get("mesh_convergence"), **_common(env, o)}


def _weldment_result(env):
    o = env["outputs"]
    return {"surface": o["surface"], "yield_mpa": o["yield_mpa"], "mesh": o["mesh"], "material": o["steel"],
            "solve_time_s": o["solve_time_s"], "applied_loads": o["applied_loads"], "reaction_n": o["reaction_N"],
            "supports": o["supports"], "dimensions_mm": o["dimensions_mm"],
            "summary": {"max_von_mises_mpa": o["max_von_mises_MPa"], "von_mises_p99_mpa": o["von_mises_p99_MPa"],
                        "max_von_mises_sheet_mpa": o["max_von_mises_sheet_MPa"],
                        "max_von_mises_ribs_mpa": o["max_von_mises_ribs_MPa"],
                        "max_displacement_mm": o["max_displacement_mm"], "min_safety_factor": o["min_safety_factor"],
                        "safety_factor_p99": o["safety_factor_p99"], "margin_of_safety": o["margin_of_safety"],
                        "factor_of_safety": o["design_factor"], "mass_kg": o["mass_kg"]},
            "equilibrium_error": o["equilibrium_error"], "mesh_convergence": o.get("mesh_convergence"),
            **_common(env, o)}


ANALYSES = {
    "/weldment-check": {"solver": "fea_plow_blade", "adapt": _weldment_result},
    "/part-check": {"solver": "fea_part_3d", "adapt": _part_result},
    "/fea": {"solver": "fea_housing_3d", "adapt": _fea_result},
    "/modal": {"solver": "modal_housing_3d", "adapt": _modal_result},
    "/cfd": {"solver": "thermal_housing_3d", "adapt": _thermal_result},
    "/review": {"solver": "review_gearbox", "adapt": _review_result},
}


def _error_text(env):
    errs = env.get("errors") or ["Unknown solver error"]
    return " ".join(str(e) for e in errs)


def _runner(spec, payload):
    # The UI may send the whole variant object (name, tag, scores, mass, ...). Keep only the
    # fields this solver understands and report the rest instead of rejecting the request.
    solver = registry.get(spec["solver"])
    known = set(solver.schema)
    mats = payload.get("materials") if isinstance(payload, dict) else None
    if isinstance(mats, dict) and mats.get("housing") and "housing_material" not in payload \
            and "housing_material" in known:
        payload = {**payload, "housing_material": mats["housing"]}       # variant from /generate-variants
    payload = {k: v for k, v in payload.items() if k != "materials"}
    ignored = sorted(k for k in payload if k not in known)
    clean = {k: v for k, v in payload.items() if k in known}

    def fn(progress):
        env = registry.run(spec["solver"], clean, on_progress=progress)
        if not env.get("ok"):
            return None, _error_text(env)
        res = spec["adapt"](env)
        res["solver_version"] = SOLVER_VERSION
        if ignored:
            res["warnings"] = list(res.get("warnings", [])) + [f"Fields not used by this solver: {ignored}"]
        return res, None
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
