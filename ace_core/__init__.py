"""ACE core — the solver framework of Idris Core (Autonomous Computational Engineering)."""
SOLVER_VERSION = "0.12.0"          # ace-solver package version, sent in every HTTP response
from .contract import SolverSpec, Benchmark, Check, FIDELITY, CONTRACT_VERSION  # noqa: F401
from .registry import register, run, verify, catalog, llm_tools  # noqa: F401
