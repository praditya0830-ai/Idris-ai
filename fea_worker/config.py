"""Configuration read from the environment, exactly once per process.

Cloud Run sets PORT and nothing else; the rest is set in the service's
environment variables (see DEPLOY.md). Every value has a safe default so the
container starts and reports its capabilities even when something is missing.
"""
from __future__ import annotations

import os

DEFAULT_MAX_BODY = 20 * 1024 * 1024   # 20 MB: a STEP file plus JSON framing
DEFAULT_MAX_ELEMENTS = 200_000
DEFAULT_TIMEOUT_S = 240.0


class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.token = (env.get("FEA_WORKER_TOKEN") or "").strip()
        self.port = int(env.get("PORT") or 8080)
        self.max_body = int(env.get("FEA_MAX_BODY") or DEFAULT_MAX_BODY)
        self.max_elements = int(env.get("FEA_MAX_ELEMENTS") or DEFAULT_MAX_ELEMENTS)
        self.timeout_s = float(env.get("FEA_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
        self.run_benchmarks = (env.get("FEA_RUN_BENCHMARKS") or "1").strip() not in ("0", "false", "no")

    def describe_limits(self):
        return {
            "max_body_bytes": self.max_body,
            "max_elements": self.max_elements,
            "timeout_s": self.timeout_s,
        }
