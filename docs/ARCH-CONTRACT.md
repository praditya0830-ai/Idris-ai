# IDRIS — Architectural contract (engine core)

Status: living document. Phase 0 freeze. Owner: solver team.
Scope: the contracts every later phase (geometry kernel, feature graph, assembly, exchange, audit)
must satisfy. If code and this document disagree, the document wins until it is amended in a commit.

## 1. Principles

1. **One authoritative kernel.** OpenCascade (OCP) on the server decides what is "valid". The browser
   kernel (replicad) is a viewport. A shape that only exists in the viewport is never a result.
2. **The model is data, not code.** A design is a feature graph (JSON) plus dimensions, not a
   monolithic CAD function. AI-generated CAD code is allowed only as a single `opaque` node.
3. **Nothing leaves without a `RunRecord`.** Inputs, solver version, kernel version, geometry
   fingerprint, evidence level, sources, and checks travel with every number.
4. **The AI is a language layer.** It proposes feature operations that are validated against
   schemas; it never emits geometry directly and never invents a parameter.
5. **Honest failure.** Missing kernel, unrecognised STEP feature, unsolved model: each returns a
   stated status, never a plausible-looking substitute.

## 2. Feature graph (`model/`)

```json
{
  "model_id": "bracket-1", "revision": 7, "units": "mm",
  "features": [
    {"id": "f1", "op": "sketch",  "deps": [], "params": {"plane": "XZ"}},
    {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": "W"}},
    {"id": "f3", "op": "hole",    "deps": ["f2"], "params": {"d": "h", "x": "L*0.65", "through": true}}
  ],
  "dims": {"L": 40, "W": 20, "h": 4.5},
  "dims_source": {"L": "user", "W": "default", "h": "catalog:ISO273:M4"}
}
```

Rules (enforced by code, not convention):

- `deps` must be explicit. A node without `deps` is a root. This is what makes partial regeneration
  possible at all.
- `id` unique, `[A-Za-z0-9_-]{1,64}`; a feature may not depend on itself or form a cycle.
- Every `op` has a schema in `model/ops.py`. Unknown `op` and unknown `params` keys are **rejected**
  (same policy as `ace_core/schema.py`: an invented parameter fails loudly).
- Dimensions referenced by expressions must exist in `dims`; expressions are restricted to numbers,
  `+ - * / **`, parentheses and names (no attribute access, no calls, no imports).
- `dims_source` vocabulary: `user`, `default`, `catalog:<...>`, `research` — the same words already
  used by `cad_codegen.validate()`.

Operations in the MVP: `sketch`, `extrude`, `revolve`, `hole`, `fillet`, `pattern`, `import`,
`opaque`. `opaque` is the migration door for AI-generated CAD code: one node, one hash, no partial
regeneration inside it.

Regeneration contract (`model/regenerate.py`):

| outcome | meaning | downstream |
|---|---|---|
| `ok` | node rebuilt and passed invariants | rebuild dependents |
| `unchanged` | not dirty, reused from cache | untouched |
| `quarantined` | build or invariant failed; reason recorded | **blocked**, never rebuilt on top |
| `blocked` | a dependency is quarantined | not attempted |

## 3. Geometry invariants (`geometry/invariants.py`)

Check ids are contract:

- mesh: `mesh.nonempty`, `mesh.indices_in_range`, `mesh.no_degenerate_triangles`, `mesh.manifold`,
  `mesh.closed`
- solid: `solid.single`, `solid.watertight`, `solid.volume_positive`, `solid.bbox_positive`,
  `solid.has_faces`, `solid.no_degenerate_faces`

A design failing any check is `quarantined`. Warnings are not a category here.

## 4. Geometry fingerprint (`geometry/fingerprint.py`)

- Canonical summary: `volume_mm3`, `area_mm2`, `bbox_mm`, `solids`, `faces`, `edges`, `vertices`,
  `triangles`, `mass_kg`; rounded to 6 significant decimals; keys sorted; unknown fields ignored.
- `fingerprint(summary)` = sha256 of that canonical form. Same physical part ⇒ same hash on any
  machine; a different part ⇒ a different hash.
- `diff(before, after, tol)` compares with a relative tolerance for sizes (default 1e-3) and exactly
  for counts. Used by the STEP round-trip gate.

## 5. Assembly (`assembly/`)

- Node kinds: `assembly`, `subassembly`, `part`, `instance`.
- An `instance` references a part by id and carries a transform; geometry is shared, not duplicated.
- BOM and mass aggregate over instances; the sum must equal the naive per-part sum (cross-checked in
  tests).
- LOD levels: `bbox` (proxy only) → `decimated` (reduced triangles) → `full`. Choice is a pure
  function of triangle count and view distance, so it is testable.
- Solving is lazy per node and cached under `(node_id, revision)`.

## 6. Exchange (`exchange/`)

- Import (STEP) is shape-level: `kernel.read_step` gives one shape. Feature recognition is a
  **candidate** step and reports `confidence`; nothing is silently converted into constraints.
- A shape whose features are not recognised carries `recognised: false` and is reported as a
  dumb solid to the user.
- Round-trip gate: export → import → `fingerprint.diff` within tolerance, and the same face/edge
  counts. Exceeding the gate is a failure (`status: "failed"`), not a warning.
- Sidecar: `<part>.step` may ship with `<part>.idris.json` holding parameters, features, evidence
  and sources, so a STEP file is not a dumb solid when it leaves the system.

## 7. RunRecord (`audit/`)

```json
{
  "run_id": "sha256...", "created_at": "2026-11-16T09:12:03Z",
  "inputs_hash": "...", "geometry_hash": "...",
  "kernel": {"name": "occt", "version": "8.0.1"},
  "solver_version": "0.35.0",
  "evidence": {"level": 3, "method": "...", "verified_on": [], "not_yet": []},
  "sources": [{"k": "T", "url": "https://...", "quote": "..."}],
  "checks": [{"id": "solid.watertight", "ok": true, "detail": ""}],
  "status": "ok|repaired|quarantined|failed"
}
```

`run_id` is derived from `inputs_hash`, `geometry_hash` and file kind, so the same run computed twice
gets the same id. A run is immutable once stored: writing a different body under an existing
`run_id` is refused.

## 8. Non-goals (this contract)

- No second CAD kernel, no rewrite of the physics solvers, no container/deploy change.
- No claim of evidence level 5 (experimental) or 6 (engineer approval): `physics/evidence.py`
  remains the single source of those levels.
- No silent fallback from kernel failure to a mesh-only "result".
