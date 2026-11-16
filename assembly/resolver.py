"""Lazy assembly resolution (Fase 2, pilar C).

The manifest carries structure, LOD and a parts table - never geometry. Geometry is fetched per node
and cached under (node_id, revision); because instances share a part, 500 bolts cause ONE load.

    r = Resolver(assembly, load_part=fn, revision=7)
    r.manifest()          -> small JSON: tree + parts + LOD plan
    r.node("bolt-42")     -> {"id", "part", "geometry", "cached": False}
"""
from __future__ import annotations

import json

from . import lod as L


class Resolver:
    def __init__(self, assembly, load_part, revision: int = 0, budget: int = L.DEFAULT_BUDGET):
        if not callable(load_part):
            raise TypeError("load_part must be callable: (part_id, level) -> geometry")
        self.assembly = assembly
        self._load = load_part
        self.revision = int(revision)
        self.budget = budget
        self._cache: dict = {}             # (part_id, level) -> geometry
        self.stats = {"loads": 0, "cache_hits": 0}

    def _key(self, part_id: str, level: str) -> tuple:
        return (part_id, level, self.revision)

    def geometry(self, part_id: str, level: str = "full"):
        key = self._key(part_id, level)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]
        value = self._load(part_id, level)
        self.stats["loads"] += 1
        self._cache[key] = value
        return value

    def node(self, node_id: str, distance_mm: float | None = None) -> dict:
        n = self.assembly.find(node_id)
        if n.kind not in ("part", "instance"):
            raise ValueError(f"{node_id!r} is a {n.kind}, not a geometry node")
        key = self._key(n.part, "full")
        cached = key in self._cache
        share = 1.0 / max(1, self.assembly.instance_count())
        level = L.level_for(n.triangles, distance_mm, self.budget, share)
        return {"id": n.id, "part": n.part, "level": level,
                "geometry": self.geometry(n.part, level), "cached": cached}

    def manifest(self, distance_mm: float | None = None) -> dict:
        plan = L.plan(self.assembly, distance_mm, self.budget)
        levels = {x["id"]: x["level"] for x in plan["nodes"]}
        parts = {pid: {"triangles": p["triangles"], "instances": p["instances"],
                       "mass_kg": p["mass_kg"]}
                 for pid, p in self.assembly.parts().items()}

        def walk(n):
            out = {"id": n.id, "kind": n.kind}
            if n.part:
                out["part"] = n.part
            if n.id in levels:
                out["lod"] = levels[n.id]
            if n.transform:
                out["transform"] = n.transform
            if n.children:
                out["children"] = [walk(c) for c in n.children]
            return out

        return {"model_id": self.assembly.model_id, "revision": self.revision,
                "tree": walk(self.assembly.root), "parts": parts,
                "lod": {"distance_mm": distance_mm, "triangles_total": plan["triangles_total"],
                        "over_budget": plan["over_budget"], "budget": self.budget},
                "bom": self.assembly.bom(), "instances": self.assembly.instance_count(),
                "unique_geometry": self.assembly.unique_geometry()}

    def manifest_bytes(self, distance_mm: float | None = None) -> int:
        return len(json.dumps(self.manifest(distance_mm), separators=(",", ":")).encode())

    def forget(self) -> None:
        self._cache.clear()
        self.stats = {"loads": 0, "cache_hits": 0}
