"""GET /selfcheck — end-to-end check of the live engine, the same path an engineer takes in the UI.

For the three demo scenarios (S1 2-stage, S2 1-stage, S3 2-stage): generate variants → full assembly
(STL parts, BOM, drawings) → /review → /fea (coarse) → /modal (coarse) → /cfd, each timed. Runs in the
background (like /verify) so it never blocks the server; the report is cached until ?refresh=1.

Pass criteria per scenario (the MVP promises):
  variants > 0 · every assembly part closed · review 0 critical / 0 warning · FEA, modal, thermal return
  results · every analysis < 60 s on this server.
"""
import threading
import time
import traceback
import numpy as np

from . import SOLVER_VERSION
from .analysis import handle_post

SCENARIOS = {
    "S1": dict(power_kw=11.19, input_rpm=6000, gear_ratio=8, env_w=200, env_h=200, env_d=150),
    "S2": dict(power_kw=11.0, input_rpm=6000, gear_ratio=6, env_w=250, env_h=200, env_d=180),
    "S3": dict(power_kw=7.5, input_rpm=6000, gear_ratio=12, env_w=300, env_h=250, env_d=200),
}
TIME_LIMIT_S = 60.0
_state = {"status": "idle", "report": None, "started": None, "finished": None}
_lock = threading.Lock()


def _closed(part):
    V = np.reshape(part["verts"], (-1, 3))
    F = np.reshape(part["faces"], (-1, 3))
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    return set(c.tolist()) == {2} and vol > 0


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t, 2)


def _scenario(name, kw):
    from physics.gears import Requirements
    from physics.variants import generate_variants
    from assembly3d import build_assembly
    steps, problems = {}, []
    req = Requirements(wall_thickness=4.0, gear_material="AMS6265", housing_material="Al7075-T6",
                       min_sf_bending=1.4, min_sf_contact=1.1, **kw)
    o, dt = _timed(lambda: generate_variants(req))
    vs = o.get("variants") or []
    steps["generate_variants"] = {"s": dt, "variants": len(vs)}
    if not vs:
        return {"pass": False, "steps": steps, "problems": ["no variants generated"]}
    v = vs[1] if len(vs) > 1 else vs[0]
    asm, dt = _timed(lambda: build_assembly(v["design"], v["kinematics"], v["shaft"], v["housing_params"],
                                            v.get("materials")))
    open_parts = [p["name"] for p in asm["parts"] if not _closed(p)]
    steps["assembly"] = {"s": dt, "parts": len(asm["parts"]), "bom_rows": len(asm.get("bom") or []),
                         "open_parts": open_parts}
    if open_parts:
        problems.append(f"parts not closed: {open_parts}")
    body = {**v, "power_kw": kw["power_kw"], "input_rpm": kw["input_rpm"], "mesh": "coarse", "modes": 4,
            "checks": ["interference", "fits", "dfm", "assembly"]}
    for path, key in (("/review?sync=1", "review"), ("/fea?sync=1", "fea"), ("/modal?sync=1", "modal"),
                      ("/cfd?sync=1", "thermal")):
        (code, b), dt = _timed(lambda: handle_post(path, body))
        r = b.get("result") or {}
        st = {"s": dt, "ok": code == 200}
        if code != 200:
            problems.append(f"{key}: {b.get('error')}")
        elif key == "review":
            st["counts"] = r.get("counts")
            if r.get("counts", {}).get("critical") or r.get("counts", {}).get("warning"):
                problems.append(f"review findings {r.get('counts')}")
        elif key == "fea":
            st["max_von_mises_mpa"] = (r.get("summary") or {}).get("max_von_mises_mpa")
            st["margin_of_safety"] = (r.get("summary") or {}).get("margin_of_safety")
        elif key == "modal":
            st["f1_hz"] = (r.get("frequencies_hz") or [None])[0]
        elif key == "thermal":
            st["oil_temp_c"] = r.get("oil_temp_c")
        if dt > TIME_LIMIT_S:
            problems.append(f"{key} took {dt} s (> {TIME_LIMIT_S:g} s)")
        steps[key] = st
    return {"pass": not problems, "variant": v.get("name"), "steps": steps, "problems": problems}


def _worker():
    rep = {"solver_version": SOLVER_VERSION, "scenarios": {}}
    try:
        for name, kw in SCENARIOS.items():
            try:
                rep["scenarios"][name] = _scenario(name, kw)
            except Exception as ex:
                rep["scenarios"][name] = {"pass": False, "problems": [f"{type(ex).__name__}: {ex}"],
                                          "trace": traceback.format_exc()[-800:]}
        rep["all_pass"] = all(s.get("pass") for s in rep["scenarios"].values())
        with _lock:
            _state.update(status="done", report=rep, finished=time.time())
    except Exception as ex:  # pragma: no cover
        with _lock:
            _state.update(status="failed", report={"error": f"{type(ex).__name__}: {ex}"}, finished=time.time())


def start(force=False):
    with _lock:
        if _state["status"] == "running" or (_state["status"] == "done" and not force):
            return
        _state.update(status="running", started=time.time(), finished=None)
    threading.Thread(target=_worker, daemon=True, name="ace-selfcheck").start()


def page(query=""):
    start(force="refresh=1" in query)
    with _lock:
        s = dict(_state)
    if s["status"] == "done":
        return 200, {"status": "done", "computed_in_s": round(s["finished"] - s["started"], 1),
                     "computed_at_unix": round(s["finished"]), **s["report"]}
    if s["status"] == "failed":
        return 500, {"status": "failed", **s["report"]}
    return 200, {"status": "running", "running_for_s": round(time.time() - (s["started"] or time.time()), 1),
                 "message": "End-to-end check is running (3 scenarios × generate, assembly, review, FEA, modal, "
                            "thermal). Refresh in 1–2 minutes."}
