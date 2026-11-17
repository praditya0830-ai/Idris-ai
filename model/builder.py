"""Kernel-backed builder for the feature graph (Fase 1, pilar A dan B).

`model.graph` decides WHAT to rebuild; this module decides HOW a feature becomes a solid. It talks to
the kernel only through the small surface in geometry/kernel.py and takes the kernel module as a
parameter, so the mapping from features to geometry can be tested without OpenCascade installed.

Rules that are not negotiable here:

  * a feature that cannot be built raises; graph.regenerate() then quarantines it and blocks its
    dependents. Nothing is silently rebuilt on top of a broken or invalid solid;
  * every solid passes geometry.invariants before it is handed to the next feature, so a body that is
    not watertight cannot become the parent of a hole;
  * a hole that removes no material is an error, not an empty result: the (x, y) misses the part and
    the caller must hear about it.

Part carries the sketch frame along, because a hole is drilled along the normal of the plane its base
was sketched on, not along world Z.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from geometry import fingerprint as FP
from geometry import invariants as INV
from geometry import kernel as _kernel

from . import graph as G

SOLID_OPS = ("extrude", "revolve", "hole", "fillet", "pattern", "import")


@dataclass(frozen=True)
class Part:
    """One node's geometry plus the frame it was built in."""

    shape: object
    op: str
    plane: str = "XY"
    profile: str | None = None
    size: tuple = ()
    summary: dict | None = None      # kernel summary once the node is a solid
    args: dict = None               # the sketch parameters, so a later op can rebuild the profile
    checks: tuple = ()              # the invariant checks this shape passed (pattern allows several bodies)

    @property
    def is_solid(self) -> bool:
        return self.summary is not None


class BuildError(RuntimeError):
    """The kernel could not produce what the feature asked for, and the reason is stated."""


def params_key(feature) -> str:
    """Stable key of everything that changes a feature's geometry (used to decide what to reuse)."""
    payload = {"op": feature.op, "deps": list(feature.deps),
               "params": {k: v for k, v in feature.params.items() if k != "exprs"}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- helpers
def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _scale(vec, k):
    return [vec[0] * k, vec[1] * k, vec[2] * k]


def _add(*vecs):
    return [sum(v[i] for v in vecs) for i in range(3)]


def _parent(deps: dict, op: str) -> Part:
    values = [v for v in deps.values() if v is not None]
    if not values:
        raise BuildError(f"{op}: no input geometry; it needs a dependency")
    if len(values) > 1:
        raise BuildError(f"{op}: exactly one input shape is supported, got {len(values)}")
    parent = values[0]
    if not isinstance(parent, Part):
        raise BuildError(f"{op}: input is not a shape this builder produced")
    return parent


def _solid(parent: Part, op: str) -> Part:
    if not parent.is_solid:
        raise BuildError(f"{op}: input {parent.op!r} is a face, not a solid; extrude or revolve it first")
    return parent


def _verify(k, shape, op: str, allow_multi: bool = False) -> dict:
    """Measure, then gate on the invariants. A failing solid never leaves the builder.

    `allow_multi` is for a pattern of copies that do not touch: that is a compound of several bodies,
    which is a legitimate result, so `solid.single` is replaced by a body-count check instead of being
    quietly dropped.
    """
    summary = k.measure(shape)
    checks = INV.solid_checks(summary)
    if allow_multi:
        bodies = int(summary.get("solids") or 0)
        checks = [c for c in checks if c["id"] != "solid.single"]
        checks.append({"id": "solid.bodies_positive", "ok": bodies >= 1, "detail": f"solids = {bodies}"})
    failed = [c["id"] for c in INV.failures(checks)]
    if failed:
        raise BuildError(f"{op}: the kernel produced a shape that fails {failed} "
                         f"(summary: {json.dumps(summary, sort_keys=True, default=str)})")
    return summary, checks


def _solid_part(k, shape, op: str, parent: Part, allow_multi: bool = False) -> Part:
    """Wrap a finished body, keeping the checks it passed so the report cannot disagree with the gate."""
    summary, checks = _verify(k, shape, op, allow_multi)
    return Part(shape, op, parent.plane, parent.profile, parent.size, summary,
                getattr(parent, "args", None), tuple(checks))


def _wire_for(k, parent: Part, grow: float = 0.0):
    """Rebuild the parent's sketch wire, optionally grown by `grow` on every side (draft)."""
    a = getattr(parent, "args", None) or {}
    if not a:
        raise BuildError("draft or offset needs the sketch parameters, and this input was not sketched")
    w = float(a["w"]) + 2.0 * grow
    h = None if a.get("h") is None else float(a["h"]) + 2.0 * grow
    wire = k.sketch_wire(a["plane"], a["profile"], w, h, int(a.get("n") or 6), a.get("thickness"))
    offset = float(a.get("offset") or 0.0)
    if offset:
        _, _, normal = k.plane_axes(a["plane"])
        wire = k.translate(wire, _scale(normal, offset))
    return wire


def _extents(k, shape):
    lo, hi = k.bbox_range(shape)
    return lo, hi


# --------------------------------------------------------------------------- operations
def _sketch(k, p: dict, op: str) -> Part:
    plane = str(p.get("plane") or "XY")
    profile = str(p.get("profile") or "rectangle")
    w = float(p["w"])
    h = float(p["h"]) if p.get("h") is not None else None
    n = int(p.get("n") or 6)
    thickness = float(p["thickness"]) if p.get("thickness") is not None else None
    offset = float(p.get("offset") or 0.0)
    if profile in ("rectangle", "lshape") and h is None:
        raise BuildError(f"sketch: a {profile} needs h")
    if profile == "lshape":
        leg = thickness if thickness is not None else min(w, h) / 3.0
        if not (0.0 < leg < min(w, h)):
            raise BuildError(f"sketch: lshape thickness must be inside (0, {min(w, h)}), got {leg}")
    face = k.sketch_face(plane, profile, w, h, n, thickness)
    if offset:
        _, _, normal = k.plane_axes(plane)
        face = k.translate(face, _scale(normal, offset))
    args = {"plane": plane, "profile": profile, "w": w, "h": h, "n": n,
            "thickness": thickness, "offset": offset}
    return Part(face, op, plane, profile, (w, h if h is not None else w), None, args)


def _extrude(k, parent: Part, p: dict, op: str) -> Part:
    _, _, n = k.plane_axes(parent.plane)
    distance = float(p["distance"])
    direction = str(p.get("direction") or "+normal")
    draft = float(p.get("draft_deg") or 0.0)
    if direction == "-normal":
        vec = _scale(n, -distance)
    else:
        vec = _scale(n, distance)

    if direction == "symmetric":
        base = k.translate(parent.shape, _scale(vec, -0.5))
    else:
        base = parent.shape

    if abs(draft) < 1e-9:
        shape = k.prism(base, vec)
    else:
        # A draft is a real offset, not a scale: the top profile is grown by distance*tan(draft).
        grow = abs(distance) * math.tan(math.radians(draft))
        bottom = _wire_for(k, parent)
        top = k.translate(_wire_for(k, parent, grow), vec)
        if direction == "symmetric":
            bottom = k.translate(bottom, _scale(vec, -0.5))
        shape = k.loft(bottom, top)
    return _solid_part(k, shape, op, parent)


def _revolve(k, parent: Part, p: dict, op: str) -> Part:
    angle = float(p.get("angle_deg") or 360.0)
    axis = str(p.get("axis") or "z").lower()
    perpendicular = {"XY": "z", "XZ": "y", "YZ": "x"}.get(parent.plane)
    if axis == perpendicular:
        raise BuildError(
            f"revolve: axis {axis!r} is perpendicular to the {parent.plane} sketch plane, so the profile "
            "would only sweep out of its own plane into a flat sheet; use an axis lying in the plane "
            f"(not {perpendicular!r})")
    shape = k.revolve(parent.shape, axis, angle)
    return _solid_part(k, shape, op, parent)


def _hole(k, parent: Part, p: dict, op: str) -> Part:
    e1, e2, n = k.plane_axes(parent.plane)
    x, y = float(p["x"]), float(p["y"])
    radius = float(p["d"]) / 2.0
    lo, hi = _extents(k, parent.shape)
    t_lo, t_hi = _dot(lo, n), _dot(hi, n)
    depth = p.get("depth")
    through = bool(p.get("through", True)) or depth is None
    if through:
        start = t_lo - 1.0
        length = (t_hi - t_lo) + 2.0
    else:
        start = t_hi - float(depth)
        length = float(depth) + 1.0
    if length <= 0:
        raise BuildError(f"{op}: depth {depth!r} leaves nothing to drill")

    origin = _add(_scale(e1, x), _scale(e2, y), _scale(n, start))
    tool = k.cylinder_along(radius, length, origin, n)
    shape = k.boolean("cut", parent.shape, tool)

    before = float(parent.summary["volume_mm3"])
    after = float(k.measure(shape)["volume_mm3"])
    if after >= before - 1e-9:
        raise BuildError(
            f"{op}: the hole at (x={x}, y={y}) removed no material, so it misses the part "
            f"(volume {before:.3f} -> {after:.3f} mm3)")
    return _solid_part(k, shape, op, parent)


def _fillet(k, parent: Part, p: dict, op: str) -> Part:
    radius = float(p["radius"])
    selector = str(p.get("edges") or "all")
    edges = k.edge_filter(parent.shape, selector)
    if edges == []:
        raise BuildError(f"{op}: no edge on this shape matches {selector!r}")
    try:
        shape = k.fillet(parent.shape, radius, edges)
    except Exception as exc:                     # the kernel's own message is the useful one
        raise BuildError(f"{op}: {exc}") from None
    return _solid_part(k, shape, op, parent)


def _pattern(k, parent: Part, p: dict, op: str) -> Part:
    mode = str(p.get("mode") or "linear")
    count = int(p.get("count") or 2)
    if count < 2:
        raise BuildError(f"{op}: count must be >= 2, got {count}")
    e1, e2, n = k.plane_axes(parent.plane)
    shape = parent.shape
    if mode == "linear":
        dx, dy = float(p.get("dx") or 0.0), float(p.get("dy") or 0.0)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            raise BuildError(f"{op}: a linear pattern needs dx or dy")
        step = _add(_scale(e1, dx), _scale(e2, dy))
        for i in range(1, count):
            shape = k.boolean("fuse", shape, k.translate(parent.shape, _scale(step, i)))
    elif mode == "circular":
        angle = float(p.get("angle_deg") or 360.0) / count
        # rotate about the sketch normal: the axis name whose direction matches the plane normal
        axis = {"XY": "z", "XZ": "y", "YZ": "x"}[parent.plane]
        for i in range(1, count):
            shape = k.boolean("fuse", shape, k.rotate(parent.shape, axis, angle * i))
    else:
        raise BuildError(f"{op}: unknown pattern mode {mode!r}")
    # Copies that do not touch stay separate bodies: that is a valid pattern, not a failed solid.
    return _solid_part(k, shape, op, parent, allow_multi=True)


def _import(k, p: dict, op: str, parts_dir: str) -> Part:
    path = str(p.get("path") or "")
    full = os.path.abspath(os.path.join(parts_dir, path))
    root = os.path.abspath(parts_dir)
    if os.path.commonpath([root, full]) != root:
        raise BuildError(f"{op}: {path!r} resolves outside the parts directory")
    if not os.path.exists(full):
        raise BuildError(f"{op}: {path!r} not found under the parts directory")
    shape = k.read_step(full)
    return Part(shape, op, "XY", None, (), _verify(k, shape, op))


def evaluate(feature, deps: dict, k=None, parts_dir: str = ".") -> Part:
    """One feature -> one Part. Raises on anything the kernel cannot honour."""
    k = _kernel if k is None else k
    op, p = feature.op, dict(feature.params)

    if op == "sketch":
        return _sketch(k, p, op)
    if op == "import":
        return _import(k, p, op, parts_dir)
    if op == "opaque":
        raise BuildError(
            "opaque: AI-generated CAD code is stored as data in this model and is NOT executed on this "
            "server; build the feature from typed operations instead (sketch/extrude/hole/fillet/...)")

    parent = _parent(deps, op)
    if op in ("hole", "fillet", "pattern"):
        _solid(parent, op)          # extrude/revolve take a face; the rest need a body
    if op == "extrude":
        return _extrude(k, parent, p, op)
    if op == "revolve":
        return _revolve(k, parent, p, op)
    if op == "hole":
        return _hole(k, parent, p, op)
    if op == "fillet":
        return _fillet(k, parent, p, op)
    if op == "pattern":
        return _pattern(k, parent, p, op)
    raise BuildError(f"unknown op {op!r}")


def build(graph, changed=None, cache: dict | None = None, keys: dict | None = None,
          k=None, parts_dir: str = ".", collect: dict | None = None) -> dict:
    """Regenerate the graph with the kernel, reusing nodes whose parameters did not change.

    `keys` maps feature id -> params_key() from a previous build. When `changed` is not given, the
    difference between those keys and the current graph is the change set: that is what makes the
    partial regeneration reachable over HTTP instead of only in a test.
    """
    k = _kernel if k is None else k
    k.require()

    if changed is None and keys:
        changed = [fid for fid in graph.topo() if keys.get(fid) != params_key(graph.get(fid))]
        changed += [fid for fid in graph.topo() if fid not in keys]
    cache = {} if cache is None else cache

    out = G.regenerate(graph, changed=changed, cache=cache,
                       evaluate=lambda feature, deps: evaluate(feature, deps, k=k, parts_dir=parts_dir))
    if collect is not None:
        collect.update(out["state"])

    features, final = {}, None
    for fid in graph.topo():
        part = out["state"].get(fid)
        if not isinstance(part, Part):
            continue
        entry = {"op": part.op}
        if part.is_solid:
            entry["summary"] = part.summary
            entry["fingerprint"] = FP.fingerprint(part.summary)
            entry["checks"] = list(part.checks) or INV.solid_checks(part.summary)
            final = {"feature": fid, **entry}
        else:
            entry["kind"] = "face"
            entry["plane"] = part.plane
            entry["profile"] = part.profile
        features[fid] = entry

    return {"status": out["status"], "changed": sorted(changed) if changed is not None else None,
            "regenerated": out["regenerated"], "reused": out["reused"],
            "quarantined": out["quarantined"], "blocked": out["blocked"], "errors": out["errors"],
            "features": features, "final": final,
            "keys": {fid: params_key(graph.get(fid)) for fid in graph.topo()},
            "parts": len(graph)}
