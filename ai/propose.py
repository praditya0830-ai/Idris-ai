"""The AI proposes, the contract decides (Fase 4, principle 4).

An LLM may propose feature operations; it may not add a field, invent an operation, or reference a
dimension that does not exist. Everything it sends passes through the same schema and graph rules as a
human request, and a rejected proposal keeps its reason so the user can see what was refused and why.
"""
from __future__ import annotations

from model.graph import Graph, GraphError

MAX_FEATURES = 200


def _features_of(proposal) -> list:
    if isinstance(proposal, dict):
        if "features" not in proposal:
            raise ValueError("a proposal must be an object with a 'features' list")
        return list(proposal["features"] or [])
    if isinstance(proposal, list):
        return list(proposal)
    raise ValueError("a proposal must be an object with 'features' or a list of features")


def validate_proposal(proposal, dims: dict | None = None, model_id: str = "proposal") -> dict:
    """-> {"ok", "accepted", "rejected", "graph"} where graph holds only the accepted features.

    Rejection is per feature: one bad operation does not throw away a good model, and one good
    operation does not smuggle in a bad one.
    """
    dims = dict(dims or {})
    feats = _features_of(proposal)
    accepted, rejected = [], []
    graph = Graph(model_id, dims=dims)
    if len(feats) > MAX_FEATURES:
        return {"ok": False, "accepted": [], "rejected": [
            {"index": None, "reason": f"{len(feats)} features exceeds the limit of {MAX_FEATURES}"}],
            "graph": graph}

    for i, spec in enumerate(feats):
        if not isinstance(spec, dict):
            rejected.append({"index": i, "reason": f"feature must be an object, got {type(spec).__name__}"})
            continue
        fid = spec.get("id")
        if not fid:
            rejected.append({"index": i, "reason": "feature has no id"})
            continue
        try:
            graph.add(str(fid), str(spec.get("op") or ""), spec.get("deps") or (),
                      spec.get("params") or {})
        except GraphError as exc:
            rejected.append({"index": i, "id": fid, "reason": str(exc)})
            continue
        accepted.append({"id": fid, "op": spec.get("op"), "deps": list(spec.get("deps") or []),
                         "params": dict(graph.get(str(fid)).params)})
    return {"ok": not rejected, "accepted": accepted, "rejected": rejected, "graph": graph}


def apply_proposal(graph: Graph, proposal, dims: dict | None = None) -> dict:
    """Add the acceptable features to `graph` (mutates it). -> {"applied", "rejected"}"""
    result = validate_proposal(proposal, dims if dims is not None else graph.dims)
    applied = []
    for feat in result["accepted"]:
        if feat["id"] in graph:
            result["rejected"].append({"id": feat["id"], "reason": "already exists in the model"})
            continue
        try:
            graph.add(feat["id"], feat["op"], feat["deps"], feat["params"])
            applied.append(feat["id"])
        except GraphError as exc:
            result["rejected"].append({"id": feat["id"], "reason": str(exc)})
    return {"applied": applied, "rejected": result["rejected"], "ok": not result["rejected"]}
