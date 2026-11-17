# CAD planning helper

`model.planning.plan_model(model)` provides a deterministic part-vs-assembly decision and
measurement-based review findings for callers that already have explicit part/interface
metadata. It validates IDs, interface references, coordinates, quantities, and finite
measurement values. It reports `MEMBER_SLENDER`, `WALL_THIN`, `CORNER_SHARP`, and
`JOINT_GAP` only when the corresponding dimensions are explicitly supplied.

This helper is not a geometry kernel, does not build or union solids, and must not be used
to label browser-only geometry as kernel-verified. The response includes an `evidence`
field to make that limitation explicit. Route integration and live geometry-backed checks
remain separate work.
