"""Bagian H1: a goal in words -> the physical quantities that can be measured for it.

Arya's point: a design variant is not only a set of sizes, it is a shape; ACE should vary the SHAPE to chase
the goal ("a quiet electric motor"). This module does the part that must never be guessed: which quantities
the goal needs, and which of them IDRIS can actually measure today.

Rules kept here, not in the AI:
- A quantity is scoreable only when a registered solver (GET /solvers) computes it AND that solver ships at
  least one benchmark. Mass is the one exception, and it is not an estimate: the kernel measures the volume
  of the built solid and the density comes from the materials table, so the number is measured, not predicted.
- Everything else is written "Not checked by IDRIS" and no AI number is accepted for it.
- The AI may propose shape families and their design space; it may not put a number into a score.
"""
from __future__ import annotations

import json
import re

import llm

# quantity -> what measures it. "solver" names are checked against the live registry (ace_core.registry),
# never assumed: a name that is not registered becomes not_available, which is the honest answer.
QUANTITIES = {
    "mass": {"label": "Mass", "unit": "kg", "better": "low", "solver": "kernel_volume",
             "measured_by": "kernel: volume of the built solid x material density"},
    "max_deflection": {"label": "Maximum deflection", "unit": "mm", "better": "low", "solver": "fea_part_3d",
                       "measured_by": "FEAssembler/fea_part_3d"},
    "min_sf": {"label": "Minimum safety factor", "unit": "-", "better": "high", "solver": "fea_part_3d",
               "measured_by": "fea_part_3d (von Mises vs yield)"},
    "max_temperature": {"label": "Maximum temperature", "unit": "C", "better": "low", "solver": "free_convection_plate",
                        "measured_by": "free_convection_plate"},
    "resonance_gap": {"label": "Resonance gap (natural vs excitation frequency)", "unit": "Hz", "better": "high",
                      "solver": "modal_solid_block", "measured_by": "modal_solid_block"},
    "cogging_torque": {"label": "Cogging torque", "unit": "Nm", "better": "low", "solver": None,
                       "measured_by": None},
    "torque_ripple": {"label": "Torque ripple", "unit": "%", "better": "low", "solver": None, "measured_by": None},
    "radial_force_harmonics": {"label": "Radial electromagnetic force harmonics", "unit": "N", "better": "low",
                               "solver": None, "measured_by": None},
    "fan_blade_tone": {"label": "Fan blade passing tone (SPL)", "unit": "dB", "better": "low", "solver": None,
                       "measured_by": None},
    "aerodynamic_drag": {"label": "Aerodynamic drag", "unit": "N", "better": "low", "solver": None,
                         "measured_by": None},
    "fatigue_life": {"label": "Fatigue life", "unit": "cycles", "better": "high", "solver": None,
                     "measured_by": None},
}

GOALS = {
    "quiet": {"words": ("bising", "senyap", "suara", "quiet", "noise", "akustik", "acoustic", "getar", "vibrat"),
              "quantities": ("cogging_torque", "torque_ripple", "radial_force_harmonics", "fan_blade_tone",
                             "resonance_gap")},
    "light": {"words": ("ringan", "lighter", "light", "massa", "mass", "berat"),
              "quantities": ("mass",)},
    "stiff": {"words": ("kaku", "stiff", "defleksi", "deflection", "lentur"),
              "quantities": ("max_deflection",)},
    "strong": {"words": ("kuat", "strong", "kekuatan", "strength", "patah", "safe", "aman"),
               "quantities": ("min_sf",)},
    "cool": {"words": ("dingin", "panas", "suhu", "cool", "thermal", "temperature", "heat"),
             "quantities": ("max_temperature",)},
    "fast": {"words": ("cepat", "speed", "drag", "aerodinamik", "aerodynamic"),
             "quantities": ("aerodynamic_drag",)},
    "durable": {"words": ("awet", "fatigue", "lelah", "umur", "life"),
                "quantities": ("fatigue_life",)},
}

NOT_CHECKED_NOTE = ("Not checked by IDRIS: no solver in this build measures the quantities above, so no AI "
                   "number is accepted for them. A shape variant can still be designed and shown.")


def _registry():
    try:
        import solvers                    # noqa: F401  (registers every solver once)
        from ace_core import registry
        return registry
    except Exception:                     # the solver package failed to import: every quantity is unmeasured
        return None


def solver_status(name: str | None) -> dict:
    """-> {status: solver | kernel | not_available, solver, benchmark}. Benchmarks are the gate for a score."""
    if not name:
        return {"status": "not_available", "solver": None, "benchmark": None}
    if name == "kernel_volume":
        return {"status": "kernel", "solver": None, "benchmark": "measured volume x materials table density"}
    reg = _registry()
    spec = reg.get(name) if reg is not None else None
    if spec is None:
        return {"status": "not_available", "solver": name, "benchmark": None}
    bench = [b.source for b in (spec.benchmarks or [])]
    if not bench:
        return {"status": "not_available", "solver": name, "benchmark": None}
    return {"status": "solver", "solver": name, "benchmark": bench[0]}


def objectives_for(request: str) -> list:
    """Which goals the sentence states. Deterministic on purpose: a rule can be checked, an AI guess cannot."""
    text = (request or "").lower()
    out = []
    for key, goal in GOALS.items():
        # whole words only: "nyaman" must not read as "aman" (safe), and "ringan" must not read as "angin"
        if any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text) for w in goal["words"]) and key not in out:
            out.append(key)
    return out


def measurable(request: str) -> dict:
    """H1: the goal sentence -> quantities, each with the solver that measures it and its status.

    -> {goals, quantities: [{key, label, unit, better, status, solver, benchmark, measured_by}],
        scoreable: [keys], not_checked: [labels], note}
    """
    goals = objectives_for(request)
    keys = []
    for g in goals:
        for k in GOALS[g]["quantities"]:
            if k not in keys:
                keys.append(k)
    quantities = []
    for k in keys:
        spec = QUANTITIES[k]
        st = solver_status(spec["solver"])
        quantities.append({"key": k, "label": spec["label"], "unit": spec["unit"], "better": spec["better"],
                           "status": st["status"], "solver": spec["solver"], "benchmark": st["benchmark"],
                           "measured_by": spec["measured_by"]})
    scoreable = [q["key"] for q in quantities if q["status"] in ("solver", "kernel")]
    not_checked = [q["label"] for q in quantities if q["status"] == "not_available"]
    return {"request": request, "goals": goals, "quantities": quantities, "scoreable": scoreable,
            "not_checked": not_checked,
            "note": None if scoreable else (NOT_CHECKED_NOTE if not_checked else
                                            "No measurable goal was found in that sentence.")}


# ------------------------------------------------------------------ H2: shape families (topology level)
FAMILY_SYSTEM = (
    "You are an engineering design lead. The engineer gives one request and a goal. Propose shape FAMILIES: "
    "concepts that differ in PRINCIPLE (topology), not in size, so the goal can be pursued by shape.\n"
    "Answer with JSON only: {\"families\":[{\"name\":\"...\",\"principle\":\"one sentence of physics, no numbers\","
    "\"changes\":[\"what differs from the other families\"],\"design_space\":[{\"name\":\"param\",\"min\":number,"
    "\"max\":number,\"unit\":\"mm\",\"locked\":false}]}]}\n"
    "Rules: every family must be buildable as one solid; sizes the engineer stated are LOCKED (locked: true) and "
    "their min and max equal that size; do not state performance numbers, savings or efficiencies."
)

COUNTS = (3, 5, 7, 10)


def _user_sizes(request: str) -> list:
    return [float(x.replace(",", ".")) for x in re.findall(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(?:mm|milimeter)\b",
                                                           request or "", flags=re.I)]


def _clean_family(fam: dict, locked_sizes: list) -> dict | None:
    if not isinstance(fam, dict) or not str(fam.get("name") or "").strip():
        return None
    space = []
    for var in fam.get("design_space") or []:
        if not isinstance(var, dict):
            continue
        name = str(var.get("name") or "").strip()[:40]
        try:
            lo, hi = float(var.get("min")), float(var.get("max"))
        except (TypeError, ValueError):
            continue
        if not name or hi < lo:
            continue
        locked = bool(var.get("locked")) or (lo == hi and lo in locked_sizes)
        space.append({"name": name, "min": lo, "max": hi, "unit": str(var.get("unit") or "mm"), "locked": locked})
    return {"name": str(fam["name"]).strip()[:80], "principle": str(fam.get("principle") or "")[:300],
            "changes": [str(c)[:160] for c in (fam.get("changes") or [])][:6], "design_space": space}


def families(request: str, count: int = 3, call=None, provider: str | None = None) -> dict:
    """H2: N shape families, each parametric, with the engineer's own sizes locked.

    `call(system, user) -> str` is injected in tests; without it the AI is asked through llm.call_provider.
    -> {ok, count, families, locked_sizes, error}
    """
    count = count if count in COUNTS else 3
    locked = _user_sizes(request)
    user = (f"Request: {request}\nGoal quantities asked for: "
            f"{[QUANTITIES[k]['label'] for g in objectives_for(request) for k in GOALS[g]['quantities']]}\n"
            f"Sizes the engineer stated (mm): {locked or 'none'}\nFamilies requested: {count}")
    try:
        if call is None:
            prov = provider or (llm.provider_order() or [None])[0]
            if not prov:
                return {"ok": False, "count": count, "families": [], "locked_sizes": locked,
                        "error": "no AI provider is configured on the server, so no shape family was written"}
            text, _model = llm.call_provider(prov, FAMILY_SYSTEM, user, max_tokens=1400, timeout=45)
        else:
            text = call(FAMILY_SYSTEM, user)
        data = llm.extract_json(text)
    except llm.LLMError as e:
        return {"ok": False, "count": count, "families": [], "locked_sizes": locked, "error": str(e)}
    except Exception as e:                       # a broken reply is a clear error, never a fake family
        return {"ok": False, "count": count, "families": [], "locked_sizes": locked,
                "error": f"{type(e).__name__}: {e}"}
    raw = data.get("families") if isinstance(data.get("families"), list) else []
    fams = [f for f in (_clean_family(x, locked) for x in raw) if f][:count]
    names = {f["name"].lower() for f in fams}
    return {"ok": len(fams) == count, "count": count, "families": fams, "locked_sizes": locked,
            "distinct_names": len(names) == len(fams),
            "error": None if len(fams) == count else
                     f"the AI returned {len(fams)} usable shape families, {count} were asked for"}


# ------------------------------------------------------------------ H4: ranking from measured values only
def _norm(value: float, lo: float, hi: float, better: str) -> float:
    if hi <= lo:
        return 1.0
    x = (value - lo) / (hi - lo)
    return 1.0 - x if better == "low" else x


def rank(results: list, weights: dict | None = None, measured: dict | None = None) -> dict:
    """Rank shape variants using ONLY values that were measured, and name the solver for each one.

    results: [{name, values: {quantity: number}, checks: {pass: bool}, sources: {quantity: "solver name"}}]
    measured: the quantity map from `measurable()` (defaults to the union of the results' values).
    -> {ranking: [{name, score, values, measured_by, gates}], not_ranked: [{name, reason}], weights, summary}
    """
    measured = measured or {}
    keys = [k for k in measured if measured[k].get("status") in ("solver", "kernel")]
    weights = {k: float((weights or {}).get(k, 1.0)) for k in keys} or {}
    live = [r for r in (results or []) if isinstance(r, dict) and r.get("name")]
    scored, skipped = [], []
    for r in live:
        values = r.get("values") if isinstance(r.get("values"), dict) else {}
        checks = r.get("checks") if isinstance(r.get("checks"), dict) else {}
        if checks and checks.get("pass") is False:
            skipped.append({"name": r["name"], "reason": "a pass/fail gate failed: " +
                            str(checks.get("reason") or "see checks")})
            continue
        have = [k for k in keys if isinstance(values.get(k), (int, float))]
        if not have:
            skipped.append({"name": r["name"], "reason": "no measured objective value yet, so it is not ranked"})
            continue
        scored.append((r, have, values))
    if not scored:
        return {"ranking": [], "not_ranked": skipped, "weights": weights,
                "summary": "No variant has a measured objective value yet, so nothing is ranked."}
    # per-quantity min/max over the variants that have that value
    rng = {}
    for _r, _have, values in scored:
        for k in _have:
            lo, hi = rng.get(k, (values[k], values[k]))
            rng[k] = (min(lo, values[k]), max(hi, values[k]))
    ranking = []
    for r, have, values in scored:
        num = den = 0.0
        for k in have:
            w = weights.get(k, 1.0)
            lo, hi = rng[k]
            num += w * _norm(float(values[k]), lo, hi, QUANTITIES.get(k, {}).get("better", "low"))
            den += w
        sources = r.get("sources") if isinstance(r.get("sources"), dict) else {}
        ranking.append({"name": r["name"], "score": round(num / den, 4) if den else 0.0,
                        "values": {k: values[k] for k in have},
                        "measured_by": {k: sources.get(k) or QUANTITIES.get(k, {}).get("measured_by") for k in have},
                        "gates": checks_note(r)})
    ranking.sort(key=lambda x: -x["score"])
    missing = [measured[k]["label"] for k in keys if k not in rng]
    return {"ranking": ranking, "not_ranked": skipped, "weights": weights,
            "summary": ("Ranked " + str(len(ranking)) + " variant(s) on " + str(len(rng)) +
                        " measured objective(s).") +
                       ((" Not measured for any variant: " + ", ".join(missing) + ".") if missing else "")}


def checks_note(r: dict) -> str:
    checks = r.get("checks") if isinstance(r.get("checks"), dict) else {}
    if not checks:
        return "no pass/fail gate was run: geometry and measurement only"
    return "pass" if checks.get("pass") else "fail"


def as_json(block: dict) -> str:
    """Small helper for /health-style summaries and logs."""
    return json.dumps(block, ensure_ascii=False, sort_keys=True)[:2000]
