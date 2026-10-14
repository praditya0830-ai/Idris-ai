"""Solver registry + safe runner + live verification."""
import math
import time
import traceback
from .contract import SolverSpec, Check, envelope
from .schema import validate, to_llm_parameters

_SOLVERS = {}


def register(spec: SolverSpec):
    if spec.name in _SOLVERS:
        raise ValueError(f"solver '{spec.name}' already registered")
    _SOLVERS[spec.name] = spec
    return spec


def get(name):
    return _SOLVERS.get(name)


def catalog():
    return [s.describe() for s in _SOLVERS.values()]


def llm_tools():
    """Function declarations for the chat LLM (LLM picks a solver, ACE computes)."""
    return [{"name": f"ace_{s.name}", "description": f"{s.summary} Method: {s.method}",
             "parameters": to_llm_parameters(s.schema)} for s in _SOLVERS.values()]


def _clean(x):
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return round(x, 6) if abs(x) < 1e-2 else round(x, 4)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "tolist"):          # numpy scalars / arrays
        return _clean(x.tolist())
    return x


def run(name, payload, on_progress=None):
    spec = _SOLVERS.get(name)
    if spec is None:
        return {"ok": False, "errors": [f"unknown solver '{name}'"], "available": list(_SOLVERS)}
    clean, errs = validate(spec.schema, payload)
    if errs:
        return _clean(envelope(spec, ok=False, inputs=payload if isinstance(payload, dict) else {},
                               errors=errs))
    t0 = time.perf_counter()
    try:
        r = (spec.run(clean, on_progress or (lambda *a, **k: None)) if spec.progress
             else spec.run(clean)) or {}
    except Exception as ex:  # never leak a stack trace to the UI; keep it in logs
        traceback.print_exc()
        return _clean(envelope(spec, ok=False, inputs=clean,
                               errors=[f"solver crashed: {type(ex).__name__}: {ex}"]))
    ms = (time.perf_counter() - t0) * 1e3
    checks = [c if isinstance(c, Check) else Check(**c) for c in r.get("checks", [])]
    return _clean(envelope(spec, ok=r.get("ok", True), inputs=clean, outputs=r.get("outputs"),
                           checks=checks, warnings=r.get("warnings"), errors=r.get("errors"),
                           iterations=r.get("iterations"), elapsed_ms=ms))


def _dig(d, path):
    for part in path.split("."):
        if isinstance(d, list):
            d = d[int(part)]
        else:
            d = d[part]
    return d


def verify(name=None):
    """Run every benchmark of one (or all) solvers. Returns a verification report."""
    names = [name] if name else list(_SOLVERS)
    report = []
    for n in names:
        spec = _SOLVERS.get(n)
        if spec is None:
            report.append({"solver": n, "error": "unknown solver"})
            continue
        for b in spec.benchmarks:
            t0 = time.perf_counter()
            try:
                res = run(n, b.inputs)
                actual = b.extract(res) if b.extract else {k: _dig(res, k) for k in b.expected}
                rows = []
                for k, exp in b.expected.items():
                    act = actual[k]
                    err = abs(act - exp) / abs(exp) if exp else abs(act)
                    rows.append({"quantity": k, "expected": exp, "actual": act,
                                 "rel_error": err, "pass": err <= b.rel_tol})
                ok = all(r["pass"] for r in rows)
            except Exception as ex:
                rows, ok = [{"error": f"{type(ex).__name__}: {ex}"}], False
            report.append({"solver": n, "benchmark": b.id, "source": b.source,
                           "tolerance": b.rel_tol, "pass": ok, "results": rows,
                           "elapsed_ms": (time.perf_counter() - t0) * 1e3})
    return _clean({"all_pass": all(r.get("pass") for r in report),
                   "passed": sum(1 for r in report if r.get("pass")), "total": len(report),
                   "report": report})
