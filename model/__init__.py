"""CAD model layer: feature graph, dimensions, deterministic regeneration (Fase 1, pilar B).

Geometry-agnostic on purpose: the graph knows nothing about OpenCascade. `regenerate()` takes an
`evaluate` callable supplied by the caller (kernel-backed in production, a fake in tests), which is
what makes partial regeneration and failure quarantine testable without a CAD kernel.
"""
from .graph import Feature, Graph, GraphError, regenerate
from . import expr, ops

__all__ = ["Feature", "Graph", "GraphError", "regenerate", "expr", "ops"]
