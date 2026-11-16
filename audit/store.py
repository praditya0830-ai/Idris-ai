"""Immutable run store on disk (Fase 4).

A run is written once. Saving a different body under an existing run_id is refused: an audit trail
that can be silently overwritten is not an audit trail. Identical re-saves are harmless and say so.
"""
from __future__ import annotations

import json
import os

from . import run as R


class StoreError(RuntimeError):
    """Refused write: invalid record, or a different body under an existing run_id."""


class Store:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, run_id: str) -> str:
        if not run_id or os.sep in run_id or ".." in run_id:
            raise StoreError(f"bad run_id {run_id!r}")
        return os.path.join(self.root, f"{run_id}.json")

    def save(self, record: dict) -> dict:
        problems = R.validate(record)
        if problems:
            raise StoreError("; ".join(problems))
        path = self._path(record["run_id"])
        if os.path.exists(path):
            existing = self.get(record["run_id"])
            if R.body_hash(existing) != R.body_hash(record):
                raise StoreError(
                    f"run {record['run_id']} already stored with different content; runs are immutable")
            return {"run_id": record["run_id"], "path": path, "existing": True}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return {"run_id": record["run_id"], "path": path, "existing": False}

    def get(self, run_id: str) -> dict:
        path = self._path(run_id)
        if not os.path.exists(path):
            raise StoreError(f"no run {run_id!r} in {self.root}")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def ids(self) -> list:
        return sorted(f[: -len(".json")] for f in os.listdir(self.root) if f.endswith(".json"))

    def list(self, status: str | None = None) -> list:
        out = [self.get(i) for i in self.ids()]
        return [r for r in out if status is None or r.get("status") == status]
