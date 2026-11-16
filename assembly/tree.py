"""Hierarchical assembly tree with instancing (Fase 2, pilar C).

An assembly is a tree of nodes: assembly -> subassembly -> part, where a repeated part is an
`instance` node with its own transform and the geometry is referenced, not copied. That is what keeps
a 500-bolt gearbox from becoming 500 meshes in the payload and in the browser.

Mass and BOM aggregate over instances, and the tests compare the result with the naive per-part sum,
so "instancing" cannot quietly drop a bolt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("assembly", "subassembly", "part", "instance")


class AssemblyError(ValueError):
    """The tree is malformed: duplicate id, unknown kind, instanced node without a part."""


@dataclass
class Node:
    id: str
    kind: str = "part"
    part: str | None = None            # geometry reference (part/instance); shared between instances
    transform: dict | None = None      # {"translate": [x,y,z], "rotate": [ax,ay,az,deg]}
    triangles: int = 0                 # of the referenced geometry
    mass_kg: float = 0.0               # of ONE instance
    children: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {"id": self.id, "kind": self.kind}
        if self.part:
            out["part"] = self.part
        if self.transform:
            out["transform"] = self.transform
        if self.triangles:
            out["triangles"] = self.triangles
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out


def node_from_dict(spec: dict) -> Node:
    if not isinstance(spec, dict) or not spec.get("id"):
        raise AssemblyError(f"a node needs an id: {spec!r}")
    kind = spec.get("kind") or ("assembly" if spec.get("children") else "part")
    if kind not in KINDS:
        raise AssemblyError(f"{spec['id']}: unknown kind {kind!r}; use {list(KINDS)}")
    n = Node(str(spec["id"]), kind, spec.get("part"),
             spec.get("transform"), int(spec.get("triangles") or 0),
             float(spec.get("mass_kg") or 0.0),
             [node_from_dict(c) for c in spec.get("children") or []])
    if n.kind in ("part", "instance") and not n.part:
        n.part = n.id
    return n


class Assembly:
    def __init__(self, root, model_id: str = "assembly", revision: int = 0):
        self.root = root if isinstance(root, Node) else node_from_dict(root)
        self.model_id = model_id
        self.revision = int(revision)
        self._index: dict[str, Node] = {}
        self._paths: dict[str, str] = {}
        for n, path, _ in self.walk():
            if n.id in self._index:
                raise AssemblyError(f"duplicate node id {n.id!r}")
            self._index[n.id] = n
            self._paths[n.id] = path

    # ---------------------------------------------------------------- read
    def walk(self, node: Node | None = None, path: str = ""):
        """(node, path, depth) in pre-order. Path is /root/sub/child, stable enough to cache on."""
        node = node or self.root
        here = f"{path}/{node.id}"
        yield node, here, here.count("/") - 1
        for c in node.children:
            yield from self.walk(c, here)

    def find(self, node_id: str) -> Node:
        if node_id not in self._index:
            raise AssemblyError(f"no node {node_id!r} in this assembly")
        return self._index[node_id]

    def path_of(self, node_id: str) -> str:
        self.find(node_id)
        return self._paths[node_id]

    def leaves(self) -> list:
        """The nodes that actually carry geometry (parts and instances)."""
        return [n for n, _, _ in self.walk() if n.kind in ("part", "instance")]

    def parts(self) -> dict:
        """part id -> {triangles, mass_kg, instances}. Geometry is listed once per unique part."""
        out = {}
        for n in self.leaves():
            entry = out.setdefault(n.part, {"triangles": n.triangles or 0,
                                            "mass_kg": n.mass_kg or 0.0, "instances": 0})
            entry["instances"] += 1
            entry["triangles"] = max(entry["triangles"], n.triangles or 0)
            entry["mass_kg"] = max(entry["mass_kg"], n.mass_kg or 0.0)
        return out

    def mass_kg(self) -> float:
        return round(sum(n.mass_kg or 0.0 for n in self.leaves()), 6)

    def bom(self) -> list:
        """[{part, count, triangles, mass_total}] sorted by part; the naive per-part sum."""
        rows = [{"part": pid, "count": p["instances"], "triangles": p["triangles"],
                 "mass_total": round(p["mass_kg"] * p["instances"], 6)}
                for pid, p in self.parts().items()]
        return sorted(rows, key=lambda r: r["part"])

    def triangle_count(self) -> int:
        """Triangles actually sent if every instance were drawn (instanced parts count once each)."""
        return sum(p["triangles"] * p["instances"] for p in self.parts().values())

    def unique_geometry(self) -> int:
        return len(self.parts())

    def instance_count(self) -> int:
        return len(self.leaves())

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "revision": self.revision, "tree": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "Assembly":
        if not isinstance(data, dict) or "tree" not in data:
            raise AssemblyError("an assembly must be an object with a 'tree'")
        return cls(data["tree"], str(data.get("model_id") or "assembly"), int(data.get("revision") or 0))
