"""ACE exploration for free-form parts (v0.32): choose variants, rank the FEA results, search wider with a fitted model.

Division of work (numbers stay honest):
  solver  POST /cad/explore   -> which parameter sets to try next + ranking of what has been checked
  browser builds every variant with the CAD kernel, finds the support/load faces again, calls POST /part-check
  solver  /part-check         -> stress, deflection, safety factor (the benchmarked voxel FEA)

Round 1 (no results): the current design + the AI's named styles + a Latin hypercube over the free parameters.
Round 2+ (results sent back): rank them; fit a power-law model  SF = c * prod(p_i ^ a_i)  and  mass = c * prod(p_i ^ b_i)
to the CHECKED results; score a few thousand candidate parameter sets with that model; return the few predicted
to be the lightest that still meet the safety factor. Those go back to the browser for a real FEA.

The fitted model only CHOOSES what to check next. Every number shown as a result comes from /part-check.
Sizes the engineer typed (source "user") are never changed unless the request says free_user_params: true.
"""
from __future__ import annotations

import math

import numpy as np

MAX_VARIANTS = 24
CANDIDATES = 4000
MAX_ROUNDS = 4


def _f(v):
    try:
        x = float(v)
        return x if x == x and abs(x) < 1e9 else None
    except (TypeError, ValueError):
        return None


def variables(cad: dict, free_user: bool = False) -> tuple:
    """-> (free, fixed). free = [{k, label, lo, hi, base, step}] = parameters IDRIS may vary."""
    free, fixed = [], []
    for p in (cad or {}).get("params") or []:
        if not isinstance(p, dict) or not p.get("k"):
            continue
        v, lo, hi = _f(p.get("value")), _f(p.get("min")), _f(p.get("max"))
        if v is None:
            continue
        why = None
        if lo is None or hi is None or hi - lo < 1e-9:
            why = "no range"
        elif p.get("source") in ("user", "table") and not free_user:
            why = "given by the engineer" if p.get("source") == "user" else "standard table value"
        elif lo <= 0:
            why = "range starts at zero or below"
        if why:
            fixed.append({"k": p["k"], "label": p.get("label") or p["k"], "value": v, "why": why})
        else:
            span = hi - lo
            step = 0.1 if span <= 5 else (0.5 if span <= 40 else 1.0)
            free.append({"k": p["k"], "label": p.get("label") or p["k"], "lo": lo, "hi": hi,
                         "base": min(max(v, lo), hi), "step": step, "unit": p.get("unit") or "mm"})
    return free[:8], fixed


def _snap(x: float, var: dict) -> float:
    s = var["step"]
    return float(min(max(round(round(x / s) * s, 4), var["lo"]), var["hi"]))


def latin_hypercube(n: int, free: list, seed: int = 7) -> list:
    """n parameter sets spread over the ranges: each range is cut into n slices and every slice is used once."""
    rng = np.random.default_rng(seed)
    cols = []
    for var in free:
        u = (rng.permutation(n) + rng.random(n)) / n
        cols.append([_snap(var["lo"] + ui * (var["hi"] - var["lo"]), var) for ui in u])
    return [{var["k"]: cols[j][i] for j, var in enumerate(free)} for i in range(n)]


def _key(params: dict, free: list) -> tuple:
    return tuple(round(float(params.get(v["k"], v["base"])), 4) for v in free)


def _label(params: dict, free: list) -> str:
    return " · ".join(f"{v['k']} {params[v['k']]:g}" for v in free if v["k"] in params)


# ------------------------------------------------------------------ ranking of checked variants
def rank(results: list, targets: dict | None = None) -> dict:
    """results = [{name, params, min_safety_factor, mass_kg | mass_g, max_displacement_mm, status?, geometry_ok?}].
    Order: meets every target, lightest first; then the ones that fail, closest to the target first; unchecked last.
    score 100 = lightest variant that meets the targets; passing 50..100 by mass; failing 0..49 by safety factor."""
    t = targets or {}
    df = _f(t.get("design_factor")) or _f(t.get("sf_min")) or 2.0
    dmax = _f(t.get("max_deflection_mm"))
    rows = []
    for i, r in enumerate(results or []):
        if not isinstance(r, dict):
            continue
        sf = _f(r.get("min_safety_factor"))
        if sf is None:
            sf = _f(r.get("min_sf"))
        mass = _f(r.get("mass_kg"))
        if mass is None and _f(r.get("mass_g")) is not None:
            mass = _f(r.get("mass_g")) / 1000.0
        disp = _f(r.get("max_displacement_mm"))
        row = {"name": str(r.get("name") or f"V{i + 1}")[:40], "params": r.get("params") if isinstance(r.get("params"), dict) else {},
               "min_safety_factor": sf, "mass_kg": mass, "max_displacement_mm": disp}
        reasons = []
        if r.get("geometry_ok") is False:
            row["status"] = "not built"
            reasons.append(str(r.get("reason") or "the CAD kernel could not build this variant")[:160])
        elif sf is None or mass is None or mass <= 0 or str(r.get("status") or "") in ("not checked", "error"):
            row["status"] = "not checked"
            reasons.append(str(r.get("reason") or "no FEA result for this variant")[:160])
        else:
            ok = sf >= df
            reasons.append(f"SF {sf:.3g} {'meets' if ok else 'is below'} the design factor {df:g}")
            if dmax is not None and disp is not None:
                d_ok = disp <= dmax
                ok = ok and d_ok
                reasons.append(f"deflection {disp:.3g} mm {'within' if d_ok else 'over'} the limit {dmax:g} mm")
            row["status"] = "pass" if ok else "fail"
            row["sf_ratio"] = sf / df
        row["reasons"] = reasons
        rows.append(row)
    passed = sorted((r for r in rows if r["status"] == "pass"), key=lambda r: r["mass_kg"])
    failed = sorted((r for r in rows if r["status"] == "fail"), key=lambda r: -r["min_safety_factor"])
    other = [r for r in rows if r["status"] not in ("pass", "fail")]
    m_best = passed[0]["mass_kg"] if passed else None
    for r in passed:
        r["score"] = round(50 + 50 * m_best / r["mass_kg"], 1)
        r["reasons"].append("lightest that meets the targets" if r is passed[0]
                            else f"{(r['mass_kg'] / m_best - 1) * 100:.0f} % heavier than the best")
    for r in failed:
        r["score"] = round(min(49.0, 49 * r["sf_ratio"]), 1)
    for r in other:
        r["score"] = None
    order = passed + failed + other
    for i, r in enumerate(order):
        r["rank"] = i + 1
    best = passed[0] if passed else None
    if best:
        summary = (f"Checked {len(passed) + len(failed)} variants by FEA: {len(passed)} meet the design factor {df:g}. "
                   f"Best: {best['name']} (SF {best['min_safety_factor']:.3g}, {best['mass_kg'] * 1000:.3g} g), "
                   "the lightest one that passes.")
    elif failed:
        c = failed[0]
        summary = (f"Checked {len(failed)} variants by FEA: none meets the design factor {df:g}. Closest: {c['name']} "
                   f"(SF {c['min_safety_factor']:.3g}). Allow larger sizes, a stronger material, or a lower load.")
    else:
        summary = "No variant has an FEA result yet."
    return {"ranking": order, "best": best["name"] if best else None, "design_factor": df, "summary": summary,
            "counts": {"pass": len(passed), "fail": len(failed), "not_checked": len(other)}}


# ------------------------------------------------------------------ fitted model: wide search, then real FEA
def fit_power_law(points: list, free: list, field: str):
    """log(y) = c + sum a_i log(p_i), least squares with a small ridge. -> (coef, rms relative error) or None."""
    X, y = [], []
    for r in points:
        v = r.get(field)
        if v is None or v <= 0:
            continue
        X.append([1.0] + [math.log(float(r["params"].get(var["k"], var["base"]))) for var in free])
        y.append(math.log(v))
    if len(y) < len(free) + 2:
        return None
    X, y = np.asarray(X), np.asarray(y)
    lam = 1e-6 * np.eye(X.shape[1])
    lam[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + lam, X.T @ y)
    rms = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    return coef, rms


def _predict(coef, params: dict, free: list) -> float:
    return math.exp(coef[0] + sum(c * math.log(params[v["k"]]) for c, v in zip(coef[1:], free)))


def plan(cad: dict, targets: dict | None = None, results: list | None = None, max_variants: int = 8,
         round_: int = 1, free_user_params: bool = False) -> dict:
    """-> the /cad/explore answer."""
    free, fixed = variables(cad, free_user_params)
    n = int(min(max(int(max_variants or 8), 2), MAX_VARIANTS))
    out = {"round": int(round_ or 1), "variables": [{k: v[k] for k in ("k", "label", "lo", "hi", "base", "unit")} for v in free],
           "fixed": fixed, "variants": [], "ranking": [], "best": None, "done": False, "notes": []}
    if not free:
        out.update(done=True, method="nothing to vary")
        out["notes"].append("No parameter is free to vary: every size is given by the engineer, comes from a standard "
                            "table, or has no range. Send free_user_params: true to let IDRIS vary the engineer's sizes too.")
        if results:
            out.update(rank(results, targets))
        return out
    base = {v["k"]: v["base"] for v in free}
    checked = [r for r in (results or []) if isinstance(r, dict) and isinstance(r.get("params"), dict)]
    seen = {_key({**base, **r["params"]}, free) for r in checked}

    def add(params: dict, name: str, **extra):
        full = {**base, **{k: _snap(float(v), next(x for x in free if x["k"] == k)) for k, v in params.items()
                           if any(x["k"] == k for x in free) and _f(v) is not None}}
        key = _key(full, free)
        if key in seen or len(out["variants"]) >= n:
            return False
        seen.add(key)
        out["variants"].append({"name": name, "params": full, "label": _label(full, free), **extra})
        return True

    if not checked:                                   # ---------- round 1: spread over the ranges
        add(base, "Current")
        for v in (cad.get("variants") or [])[:3]:
            if isinstance(v, dict) and isinstance(v.get("params"), dict) and not v.get("code"):
                add(v["params"], str(v.get("name") or "Style")[:24])
        i = 0
        for ps in latin_hypercube(max(n, 4) * 2, free):
            i += add(ps, f"E{i + 1}")
        out["method"] = (f"round 1: current design, the AI's styles and a Latin hypercube over {len(free)} free "
                         f"parameter(s); {len(out['variants'])} variants to build and check by FEA")
        return out

    out.update(rank(checked, targets))                # ---------- later rounds: rank, model, wider search
    df = out["design_factor"]
    usable = [{"params": {**base, **r["params"]}, "sf": _f(r.get("min_safety_factor") if r.get("min_safety_factor") is not None
                                                           else r.get("min_sf")),
               "mass": _f(r.get("mass_kg")) if _f(r.get("mass_kg")) is not None else ((_f(r.get("mass_g")) or 0) / 1000 or None)}
              for r in checked if r.get("geometry_ok") is not False]
    usable = [u for u in usable if u["sf"] and u["mass"]]
    if out["round"] >= MAX_ROUNDS:
        out.update(done=True, method=f"stopped after {MAX_ROUNDS} rounds")
        return out
    m_sf, m_mass = fit_power_law(usable, free, "sf"), fit_power_law(usable, free, "mass")
    if not m_sf or not m_mass:
        need = len(free) + 2
        i = 0
        for ps in latin_hypercube(n * 2, free, seed=11 + out["round"]):
            i += add(ps, f"R{out['round']}-{i + 1}")
        out["method"] = (f"round {out['round']}: only {len(usable)} checked variant(s), a model of {len(free)} "
                         f"parameter(s) needs {need}; more spread-out variants instead")
        return out
    (c_sf, e_sf), (c_m, e_m) = m_sf, m_mass
    margin = 1.0 + min(0.5, 2.0 * e_sf)               # the worse the model fits, the more reserve it keeps
    cands = latin_hypercube(CANDIDATES, free, seed=101 + out["round"])
    scored = []
    for ps in cands:
        sf_p, m_p = _predict(c_sf, ps, free), _predict(c_m, ps, free)
        if sf_p >= df * margin:
            scored.append((m_p, sf_p, ps))
    scored.sort(key=lambda x: x[0])
    best_mass = min((u["mass"] for u in usable if u["sf"] >= df), default=None)
    i = 0
    for m_p, sf_p, ps in scored:
        if best_mass is not None and m_p >= best_mass * 0.995:
            break                                     # the model sees nothing lighter than what is already checked
        i += add(ps, f"R{out['round']}-{i + 1}", predicted={"min_safety_factor": round(sf_p, 3), "mass_kg": round(m_p, 6),
                                                              "label": "model prediction, not a result"})
        if len(out["variants"]) >= max(2, n // 2):
            break
    out["model"] = {"form": "y = c * prod(p_i ^ a_i), fitted to the FEA-checked variants",
                    "candidates_scored": len(cands), "candidates_predicted_to_pass": len(scored),
                    "safety_factor_exponents": {v["k"]: round(float(a), 3) for a, v in zip(c_sf[1:], free)},
                    "mass_exponents": {v["k"]: round(float(a), 3) for a, v in zip(c_m[1:], free)},
                    "fit_error": {"safety_factor": round(e_sf, 4), "mass": round(e_m, 4)}, "reserve_on_design_factor": round(margin, 3),
                    "use": "chooses what to check next; never shown as a result"}
    if not out["variants"]:
        out.update(done=True)
        out["notes"].append("The model finds no lighter candidate that still meets the design factor: the best checked "
                            "variant stands." if best_mass is not None else
                            "The model predicts that no size inside the allowed ranges meets the design factor.")
    out["method"] = (f"round {out['round']}: power-law model fitted to {len(usable)} FEA results, {len(cands)} candidate "
                     f"parameter sets scored, {len(out['variants'])} sent for a real FEA check")
    return out
