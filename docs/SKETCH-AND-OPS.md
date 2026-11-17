# Sketch profiles and operations

The authoritative list is `model/ops.py` (schema) and `model/builder.py` (behaviour). This page says
what each one means, and records what the live server actually proved.

## Sketch

| profile | where the material sits | needs |
|---|---|---|
| `rectangle` | (0,0)..(w,h) | `w`, `h` |
| `lshape` | an L inside (0,0)..(w,h), leg `thickness` (default min(w,h)/3) | `w`, `h`, optional `thickness` |
| `circle` | centred on the origin, diameter `w` | `w` |
| `polygon` | `n`-gon (3..64, default 6), circumscribed radius w/2, centred | `w`, optional `n` |
| `triangle` | polygon with 3 sides, flat side down | `w` |

`plane` is `XY` (normal +Z), `XZ` (-Y) or `YZ` (+X). `offset` moves the sketch along the plane's normal,
which is what makes a multi-level part possible: before it, every sketch had to sit exactly on the plane.
Parameters may be expressions over `dims` (`"L/2"`, `"t*2"`).

## Operations

| op | input | what it does |
|---|---|---|
| `extrude` | a face | prism along the plane normal; `direction` `+normal`/`-normal`/`symmetric`; `draft_deg` builds a loft whose top profile is grown by `distance*tan(draft)` |
| `revolve` | a face | revolves about a world axis; an axis perpendicular to the sketch plane is refused with a reason, because it would sweep a flat sheet |
| `hole` | a solid | drills along the sketch normal at `(x, y)`; `through` or `depth`; a hole that removes no material is an error |
| `fillet` | a solid | `all`, `vertical`, `horizontal`, `top`, `bottom`; a selector that matches no edge is an error |
| `pattern` | a solid | `linear` (dx, dy) or `circular` (angle_deg); copies that do not touch are reported as several bodies (`solid.bodies_positive`) instead of failing `solid.single` |
| `import` | - | reads a STEP/IGES from `parts_dir`, path confined to that directory |
| `opaque` | - | **refused**: AI-written CAD code is data in this model, never executed here |

Every solid is measured and gated on the `solid.*` invariants before it becomes the parent of the next
feature. A node that fails is quarantined with the kernel's reason and its dependents are blocked.

## OCP traps that only show up on the real kernel

Both were found by `/model/build` on the live server, not by the fake kernel in the tests:

1. `BRepFilletAPI_MakeFillet.Add` wants a `TopoDS_Edge`. Passing `TopExp_Explorer.Current()` - a generic
   `TopoDS_Shape` - raised `Add(): incompatible function arguments`, and that silently broke **every**
   fillet.
2. `BRepOffsetAPI_ThruSections.AddWire` rejects the generic shape a transform returns, so a wire that
   went through `translate` has to be downcast too (`as_wire`).

Same root cause as `as_face` earlier: OCP bindings insist on the concrete TopoDS type. The helpers
`as_face`, `as_edge` and `as_wire` in `geometry/kernel.py` are the only place that should happen.

## Live matrix (Render, OCP 8.0.1.1.0)

All of these answered `status: ok` with a volume that matches the closed-form result:
rectangle/circle/polygon(6, 8)/triangle/lshape, sketch `offset`, extrude `+normal`/`-normal`/`symmetric`,
draft on a rectangle and on a circle (frustum volume), revolve 270 deg, hole with `depth`, pattern
`linear` and `circular`, and fillet with all five selectors. `revolve` with an axis perpendicular to the
sketch plane is refused with the stated reason.
