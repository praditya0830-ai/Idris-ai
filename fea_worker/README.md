# IDRIS FEA worker

CalculiX (solver) and gmsh (mesher) behind one small HTTP service, deployed to
Google Cloud Run and called by the IDRIS solver through `fea3d.py`.

| Method | Path | Auth | Answer |
|---|---|---|---|
| GET | `/health` | none | `{"ok": true}` — Cloud Run's probe |
| GET | `/capabilities` | bearer | what this build can run, versions, limits |
| GET | `/verify` | bearer | re-runs the benchmark gate and returns it |

Everything except `/health` requires `Authorization: Bearer $FEA_WORKER_TOKEN`.
A wrong or missing token is `401`; the token is never logged.

`/capabilities` is the only source of truth about features: a solver reports
`available: true` only when its module is in this build and its binaries exist,
and `validated: true` only when its benchmark gate ran here and passed. The Studio
greys out what is not available and shows the `reason` string.

## Run locally

```bash
FEA_WORKER_TOKEN=dev PORT=8080 python3 -m fea_worker.app
curl -s localhost:8080/health
curl -s -H "Authorization: Bearer dev" localhost:8080/capabilities
```

Without `ccx` or `gmsh` on PATH the worker still starts: it reports both versions
as `null`, every analysis as unavailable, and the reason names the missing binary.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | set by Cloud Run |
| `FEA_WORKER_TOKEN` | (empty) | shared secret with the solver; empty means every protected route answers 401 |
| `FEA_MAX_BODY` | `20971520` | request body limit, bytes |
| `FEA_MAX_ELEMENTS` | `200000` | element limit; larger meshes are refused with a message |
| `FEA_TIMEOUT_S` | `240` | CalculiX wall-clock limit per request |
| `FEA_RUN_BENCHMARKS` | `1` | `0` skips the startup benchmark gate (CI smoke build); capabilities then report `validated: false` |

## Deploy

`DEPLOY.md` has the click-by-click path through the Cloud Run console, written for
someone on a tablet. `LICENSES.md` records the GPL programs this runs and the
subprocess boundary that keeps them out of IDRIS code.
