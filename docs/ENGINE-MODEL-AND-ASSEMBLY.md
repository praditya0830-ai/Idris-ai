# Model build and per-node geometry

The two endpoints added after the first engine release, and the rules they follow. Everything here is
verified by `tests/test_model_build.py` with a stand-in kernel and, for OCP itself, on the live server.

## POST /model/build

Input: the same model object `/model/validate` takes.

```json
{"model": {"model_id": "bracket", "dims": {"L": 40, "W": 20, "t": 3},
  "features": [
    {"id": "f1", "op": "sketch",  "deps": [],          "params": {"plane": "XY", "w": "L", "h": "W"}},
    {"id": "f2", "op": "extrude", "deps": ["f1"],      "params": {"distance": "t"}},
    {"id": "f3", "op": "hole",    "deps": ["f2"],      "params": {"d": 4, "x": 10, "y": 10}}]}}
```

Output: `status`, `changed`, `regenerated`, `reused`, `quarantined`, `blocked`, per-feature `summary` /
`fingerprint` / `checks`, `final`, and `run` (a RunRecord).

Rules:

* `changed` may be omitted. The change set is then derived by comparing each feature's parameter key
  with the key stored from the previous build of that `model_id`, so an edit does not have to announce
  itself. Only the change set and its dependents are rebuilt; the rest is reused from the cache.
* Every solid is measured and gated on `solid.*` before the next feature runs. A body that is not
  watertight cannot become the parent of a hole.
* A feature that cannot be built is quarantined with the kernel's own reason, its dependents are
  blocked, and the response says `status: "quarantined"`. It never returns a partial success silently.
* A hole that removes no material is an error (`removed no material, so it misses the part`), not an
  empty result.
* `opaque` nodes (AI-written CAD code kept as data) are refused: this server does not execute model
  code. Build the feature from typed operations instead.
* No kernel on the host: `503 kernel_unavailable`, nothing approximated.

Sketch frames: `XY` normal `+Z`, `XZ` normal `-Y`, `YZ` normal `+X`. A hole is drilled along the normal
of the plane its base was sketched on, at the sketch coordinates `(x, y)`.

## POST /assembly/node

Resolves ONE node's geometry, in the LOD the viewer asked for. The parts' STEP/IGES come from the
caller, because this service stores no part geometry.

```json
{"assembly": {...}, "node": "b-2", "distance_mm": 5000,
 "parts": {"bolt": {"filename": "bolt.step", "step_base64": "..."}}}
```

Output: `id`, `part`, `kind`, `level`, `cached`, `transform`, `positions`, `indices`, `vertices`,
`triangles`, `summary`, `checks` (`mesh.*`), `run`.

* `level` comes from `assembly/lod.py`: `bbox` (a 12-triangle box proxy), `decimated`, `full`. There is
  no mesh decimator on this server, so a `decimated` request is served as the full mesh and the
  response says so in `note` rather than pretending.
* Instances share geometry: one load per part per revision, not one per node.
* A part with no STEP supplied is a `400` that names the exact key to send; an unknown node id is a
  `400` that says so.
* Every returned mesh is checked with `mesh_checks` (closed, manifold, no degenerate triangles) and the
  result travels in `checks` and in the RunRecord.

## RunRecords

`/model/build`, `/assembly/manifest`, `/assembly/node` and `/exchange/step` now return a `run` object:
`run_id`, `kind`, `inputs_hash`, `geometry_hash`, `kernel {name, version}`, `solver_version`,
`checks`, `status`, `created_at`. `run_id` is derived from the inputs and the geometry, so the same run
computed twice gets the same id. Records are returned with the answer; they are not persisted, because
the Render filesystem is ephemeral - `audit/store.py` is ready for a host with a volume.

## Live facts (Render, OCP 8)

The kernel primitives above were exercised on the live server; the failures found there are recorded in
`docs/ARCH-CONTRACT.md` section 9 and were fixed rather than worked around.
