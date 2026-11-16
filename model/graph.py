"""Feature graph and deterministic partial regeneration (Fase 1, pilar B).

A design is a DAG of typed operations with explicit dependencies. Changing one parameter regenerates
only that node and its dependents; a node that fails is quarantined and its dependents are blocked,
never rebuilt on top of a broken shape. That is the whole difference between "predictive regeneration"
and re-running everything and hoping.

Geometry-agnostic: `regenerate(..., evaluate=fn)` takes the builder. Production passes a kernel-backed
builder; tests pass a fake, which is why this layer is testable without OpenCascade.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from . import expr as E
from . import ops as O

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STATUSES = ("ok", "repaired", "quarantined", "blocked", "failed")


class GraphError(ValueError):
    """The graph would be invalid: bad id, unknown op, missing dep, or a cycle."""


@dataclass(frozen=True)
class Feature:
    id: str
    op: str
    deps: tuple = ()
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "op": self.op, "deps": list(self.deps), "params": dict(self.params)}


class Graph:
    def __init__(self, model_id: str = "model", dims: dict | None = None, revision: int = 0):
        self.model_id = model_id
        self.dims = dict(dims or {})
        self.revision = int(revision)
        self._features: dict[str, Feature] = {}

    # ---------------------------------------------------------------- read
    def __len__(self):
        return len(self._features)

    def __contains__(self, fid):
        return fid in self._features

    def get(self, fid: str) -> Feature:
        if fid not in self._features:
            raise GraphError(f"no feature {fid!r}")
        return self._features[fid]

    def features(self) -> list:
        """Insertion order, which is the order the user built the model in."""
        return list(self._features.values())

    def dependents(self, fid: str, transitive: bool = True) -> list:
        """Everything that must be rebuilt when `fid` changes (excluding `fid` itself)."""
        self.get(fid)
        direct = {f.id for f in self._features.values() if fid in f.deps}
        if not transitive:
            return sorted(direct)
        out, frontier = set(direct), list(direct)
        while frontier:
            cur = frontier.pop()
            for f in self._features.values():
                if cur in f.deps and f.id not in out:
                    out.add(f.id)
                    frontier.append(f.id)
        return sorted(out)

    def topo(self) -> list:
        """Deterministic topological order; raises GraphError on a cycle."""
        order, state = [], {}

        def visit(fid, stack):
            mark = state.get(fid)
            if mark == "done":
                return
            if mark == "open":
                raise GraphError(f"dependency cycle: {' -> '.join(stack + [fid])}")
            state[fid] = "open"
            for dep in self.get(fid).deps:
                visit(dep, stack + [fid])
            state[fid] = "done"
            order.append(fid)

        for fid in self._features:          # insertion order -> stable output
            visit(fid, [])
        return order

    def resolved_dims(self) -> dict:
        return E.resolve(self.dims)

    # --------------------------------------------------------------- write
    def add(self, fid: str, op: str, deps=(), params=None) -> Feature:
        if not ID_RE.match(fid or ""):
            raise GraphError(f"feature id {fid!r} must match {ID_RE.pattern}")
        if fid in self._features:
            raise GraphError(f"feature {fid!r} already exists; use update()")
        return self._insert(Feature(fid, op, tuple(deps or ()), dict(params or {})))

    def add_many(self, specs) -> list:
        added = []
        for spec in specs:
            added.append(self.add(**spec) if isinstance(spec, dict) else self.add(*spec))
        return added

    def _insert(self, feature: Feature) -> Feature:
        if feature.op not in O.SCHEMAS:
            raise GraphError(f"{feature.id}: unknown op {feature.op!r}; known: {list(O.OPS)}")
        for dep in feature.deps:
            if dep == feature.id:
                raise GraphError(f"{feature.id}: a feature cannot depend on itself")
            if dep not in self._features:
                raise GraphError(f"{feature.id}: unknown dependency {dep!r}")
        # "exprs" is derived output, not an input field: feed the original text back so a model that
        # went through to_dict/from_dict keeps the engineer's expressions instead of bare numbers.
        raw = {k: v for k, v in feature.params.items() if k != "exprs"}
        for name, text in (feature.params.get("exprs") or {}).items():
            if name in raw:
                raw[name] = text
        clean, errors = O.validate_params(feature.op, raw, self.dims)
        if errors:
            raise GraphError(f"{feature.id}: " + "; ".join(errors))
        feature = replace(feature, params=clean)
        self._features[feature.id] = feature
        try:
            self.topo()
        except GraphError:
            del self._features[feature.id]
            raise
        self.revision += 1
        return feature

    def update(self, fid: str, params: dict | None = None, deps=None) -> Feature:
        cur = self.get(fid)
        new = Feature(cur.id, cur.op, tuple(deps) if deps is not None else cur.deps,
                      {**cur.params, **(params or {})})
        backup = self._features[fid]
        del self._features[fid]
        try:
            return self._insert(new)
        except GraphError:
            self._features[fid] = backup
            raise

    def remove(self, fid: str, cascade: bool = False) -> list:
        self.get(fid)
        kids = self.dependents(fid, transitive=False)
        if kids and not cascade:
            raise GraphError(f"{fid!r} still has dependents {kids}; pass cascade=True to drop them")
        victims = [fid] + (self.dependents(fid) if cascade else [])
        for v in victims:
            del self._features[v]
        self.revision += 1
        return victims

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self, resolved: bool = False) -> dict:
        out = {"model_id": self.model_id, "revision": self.revision,
               "units": "mm", "features": [f.to_dict() for f in self.features()],
               "dims": self.resolved_dims() if resolved else dict(self.dims)}
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        if not isinstance(data, dict) or "features" not in data:
            raise GraphError("a model must be an object with a 'features' list")
        g = cls(str(data.get("model_id") or "model"), data.get("dims") or {}, 0)
        for spec in data["features"]:
            if not isinstance(spec, dict) or "id" not in spec or "op" not in spec:
                raise GraphError(f"bad feature spec {spec!r}")
            g.add(spec["id"], spec["op"], spec.get("deps") or (), spec.get("params") or {})
        return g


def regenerate(graph: Graph, changed=None, evaluate=None, cache: dict | None = None) -> dict:
    """Rebuild only what changed. `changed=None` means everything.

    evaluate(feature, dep_values) -> value  (number, shape, dict: the caller decides)
    -> {"status", "state", "regenerated", "reused", "quarantined", "blocked", "errors"}
    """
    if evaluate is None:
        raise GraphError("regenerate() needs an evaluate callable; there is no default builder")
    dirty = None if changed is None else set(_as_list(changed))
    if dirty is not None:
        for fid in list(dirty):
            graph.get(fid)
            dirty.update(graph.dependents(fid))
    cache = {} if cache is None else cache

    state, regenerated, reused, quarantined, blocked = {}, [], [], [], []
    dead = {}
    for fid in graph.topo():
        feature = graph.get(fid)
        upstream = [d for d in feature.deps if d in dead]
        if upstream:
            dead[fid] = f"blocked by {upstream[0]}"
            blocked.append({"id": fid, "reason": dead[fid]})
            continue
        if dirty is not None and fid not in dirty and fid in cache:
            state[fid] = cache[fid]
            reused.append(fid)
            continue
        try:
            state[fid] = evaluate(feature, {d: state.get(d) for d in feature.deps})
        except Exception as exc:                       # a broken feature is data, not a crash
            dead[fid] = f"{type(exc).__name__}: {exc}"
            quarantined.append({"id": fid, "reason": dead[fid]})
            continue
        cache[fid] = state[fid]
        regenerated.append(fid)

    status = "quarantined" if (quarantined or blocked) else "ok"
    return {"status": status, "state": state, "regenerated": regenerated, "reused": reused,
            "quarantined": quarantined, "blocked": blocked,
            "errors": [q["reason"] for q in quarantined] + [b["reason"] for b in blocked]}


def _as_list(changed) -> list:
    if isinstance(changed, str):
        return [changed]
    return list(changed)
