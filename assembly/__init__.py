"""Assembly layer: tree, instancing, LOD and lazy resolution (Fase 2, pilar C)."""
from .resolver import Resolver
from .tree import Assembly, AssemblyError, Node, node_from_dict
from . import lod

__all__ = ["Assembly", "AssemblyError", "Node", "node_from_dict", "Resolver", "lod"]
