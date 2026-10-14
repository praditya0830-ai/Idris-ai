"""ACE solver contract — every solver (gear, shaft, FEA, CFD, thermal, ...) obeys this.

Rules (enforced by registry.run):
1. Input  is a JSON object, validated against the solver's schema BEFORE any math.
2. Output is the Result envelope below — the UI and the LLM only ever read this shape.
3. Physics is deterministic and traceable (closed-form or numerical method), never an LLM.
4. Every solver ships benchmarks with a published source; /verify runs them live.
5. A solver states its fidelity level honestly (see FIDELITY).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

CONTRACT_VERSION = "1.0"

# How "deep" the physics is. Each tier is slower and more general than the one above.
FIDELITY = {
    "L0": "closed-form handbook / standard (Shigley, AGMA, ISO) — milliseconds",
    "L1": "1D numerical (beam integration, lumped networks, Rayleigh) — < 1 s",
    "L2": "2D/3D field solver: FEM structural / thermal — seconds",
    "L3": "CFD / coupled multiphysics (FVM, FSI) — minutes",
    "L4": "physical test correlation (prototype data) — the final truth",
}


@dataclass
class Check:
    """One pass/fail engineering criterion — the UI renders these as the verdict panel."""
    name: str
    value: float
    limit: float
    unit: str
    kind: str = "max"          # "max": value <= limit passes; "min": value >= limit passes
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.value <= self.limit if self.kind == "max" else self.value >= self.limit

    def to_dict(self):
        return {"name": self.name, "value": self.value, "limit": self.limit, "unit": self.unit,
                "kind": self.kind, "pass": self.passed, "note": self.note}


@dataclass
class Benchmark:
    """A known answer from a published source. Tolerance is relative (0.02 = 2 %)."""
    id: str
    source: str                          # citation, e.g. "Shigley 10e, Ex. 7-1"
    inputs: dict
    expected: dict                       # {"path.to.output": value}
    rel_tol: float = 0.02
    extract: Optional[Callable] = None   # optional custom extractor(result) -> dict of actuals


@dataclass
class SolverSpec:
    name: str                            # url-safe id, e.g. "shaft_bearing"
    version: str                         # e.g. "0.1.0"
    domain: str                          # "machine_elements" | "structural_fea" | "cfd" | ...
    fidelity: str                        # key of FIDELITY
    summary: str
    method: str                          # human-readable method + references
    schema: dict                         # input schema (see schema.py)
    run: Callable                        # run(validated_inputs: dict) -> dict (outputs, checks, ...)
    benchmarks: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    example: Optional[dict] = None       # example payload for UI / LLM tool docs
    progress: bool = False               # run(inputs, on_progress) receives a progress callback

    def describe(self):
        return {"name": self.name, "version": self.version, "domain": self.domain,
                "fidelity": self.fidelity, "fidelity_meaning": FIDELITY[self.fidelity],
                "summary": self.summary, "method": self.method, "schema": self.schema,
                "benchmarks": [{"id": b.id, "source": b.source} for b in self.benchmarks],
                "limitations": self.limitations, "example": self.example}


def envelope(spec: SolverSpec, *, ok, inputs=None, outputs=None, checks=None,
             warnings=None, errors=None, iterations=None, elapsed_ms=None):
    """The ONE response shape every solver returns."""
    checks = checks or []
    cd = [c.to_dict() if isinstance(c, Check) else c for c in checks]
    return {
        "contract": CONTRACT_VERSION,
        "ok": bool(ok),
        "solver": spec.name, "version": spec.version, "domain": spec.domain,
        "fidelity": spec.fidelity, "method": spec.method,
        "verdict": ("PASS" if all(c["pass"] for c in cd) else "FAIL") if ok and cd else ("OK" if ok else "ERROR"),
        "inputs": inputs or {}, "outputs": outputs or {}, "checks": cd,
        "warnings": warnings or [], "errors": errors or [],
        "iterations": iterations or [], "limitations": spec.limitations,
        "elapsed_ms": elapsed_ms,
    }
