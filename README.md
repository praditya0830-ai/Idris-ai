# Idris Core — ACE engine

Backend for **IDRIS Design Studio**: deterministic engineering solvers plus parametric CAD.
Requirements in → validated gearbox variants → 3D assembly, BOM, drawings → FEA, modal and
thermal analyses. Every solver ships benchmarks with a published source and states its own
limitations; physics is never delegated to an LLM.

- Live: <https://idris-ai-rwd6.onrender.com> (`GET /health` lists every route)
- Solver version: **0.35.0** · **27** registered solvers · contract **1.0**
- Frontend: [`Idris-studio`](https://github.com/praditya0830-ai/Idris-studio) (separate repo)
- API contract: [`docs/API-CONTRACT.md`](docs/API-CONTRACT.md)

## What it does

| Capability | Endpoint |
|---|---|
| Requirements → up to 5 scored, physics-checked variants | `POST /generate-variants` |
| Full manufacturable assembly: STL per part, BOM CSV, drawings, optional STEP | `POST /generate-assembly` |
| Housing shell geometry | `POST /generate-housing` |
| Static FEA / natural frequencies / thermal, as pollable jobs | `POST /fea`, `/modal`, `/cfd` |
| Design review: interference, fits, DFM, assembly | `POST /review` |
| FEA for an arbitrary part mesh | `POST /part-check` |
| Standard-part library and catalogs, with ISO sources | `GET /library`, `/catalog` |
| Material table for the simulator and reports (`E`, `nu`, `Sy`, `rho`) | `GET /materials` |
| Print-ready engineering report: requirements, checks, analyses, BOM, solver evidence | `POST /report` |
| Chat → requirements or CAD code (LLM, with built-in fallback parser) | `POST /chat` |
| Live solver benchmarks vs published reference values | `GET /verify` |
| End-to-end check of the three demo scenarios | `GET /selfcheck` |

## Quick start

Python 3.10+ (tested on 3.12). `cadquery-ocp` is optional — the server runs without it and
`GET /occt/selftest` reports the fallback status.

```bash
pip install -r requirements.txt
python3 main.py
# in another shell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate-housing \
  -H 'Content-Type: application/json' \
  -d '{"envelope_width":200,"envelope_height":200,"envelope_depth":150,
       "gear_ratio":8.0,"wall_thickness":5.0}'
```

Generated STL/STEP/BOM files go to `generated/` (git-ignored).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

`conftest.py` keeps three files out of pytest collection because they are standalone QA
scripts, not test modules:

```bash
python3 tests/test_validation.py   # Shigley/ISO benchmark report (prints PASS/FAIL, exits 1 on failure)
python3 tests/test_chat.py         # live chat/LLM end-to-end check (needs provider keys)
```

`tests/test_waitlist.py` belongs to the `Idris-company-web` repo and is ignored here.

The full suite includes FEA and live benchmark runs and can take 20+ minutes on a small
2-vCPU machine. CI runs a fast, deterministic subset on every push (see
`.github/workflows/tests.yml`); the full suite is available via manual dispatch.

## Layout

```
main.py              stdlib HTTP server: routing, jobs hand-off, CAD/assembly responses
ace_core/            solver contract, registry, schema validation, job queue, self-check
physics/             gears (ISO 6336), bearings (ISO 281), shafts, bolted joints,
                     FEM 2D/3D, modal, thermal, DFM, ISO 286/2768, STEP/STL export
solvers/             27 registered solvers, each with schema, benchmarks and limitations
geometry/            housing geometry kernel
cad_codegen.py       LLM → replicad CAD code, repair loop, review
llm.py               provider fallback (Claude, Gemini, OpenAI, NVIDIA) + rate limiting
assembly3d.py        assembly, closed-solid checks, BOM
docs/                API contract and UI integration notes
tests/               pytest suite + standalone QA scripts
```

## Configuration

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port (default `8000`; set automatically by Render/Railway) |
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `NVIDIA_API_KEY` | Enable LLM providers; without keys `/chat` uses the built-in parser |
| `IDRIS_LLM_ORDER` | Provider preference, default `claude,gemini,openai,nvidia` |
| `IDRIS_MODEL_<PROVIDER>` | Override the model per provider |
| `IDRIS_LLM_PER_MIN`, `IDRIS_LLM_DAILY_CAP` | LLM rate limits (default 12/min, 400/day) |
| `IDRIS_REPLY_LANGUAGE` | Reply language (`en` default, `auto` follows the request) |

Deploy as a Render Web Service with start command `python3 main.py`. Mount a disk only if
you need generated files to survive deploys.

## Fidelity, honestly

Solvers declare a fidelity level (`L0` closed-form handbook/ISO, `L1` 1D numerical, `L2`
2D/3D FEM field solver, `L3` CFD/multiphysics, `L4` physical test correlation). The catalog
currently spans L0–L2. Known scope limits are listed per solver and surfaced by
`GET /solvers`; for example the ISO 6336 rating takes `K_Hbeta` as an input and the default
STEP export is faceted. There is no PINN/meshless solver, and no claim is made that there is.

## Contributing

Small, reviewable commits from a local clone are preferred over uploading whole files through
the GitHub web UI: uploads produce coarse history (`Add files via upload`) that is hard to
review or bisect. Run the fast test subset before pushing; CI runs the same set.
