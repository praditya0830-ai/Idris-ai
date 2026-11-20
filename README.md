# Idris Core — ACE engine

Backend for **IDRIS Design Studio**: deterministic engineering solvers plus parametric CAD.
Requirements in → validated gearbox variants → 3D assembly, BOM, drawings → FEA, modal and
thermal analyses. Every solver ships benchmarks with a published source and states its own
limitations; physics is never delegated to an LLM.

- Live: <https://idris-ai-rwd6.onrender.com> (`GET /health` lists every route)
- Solver version: **0.37.0** · **27** registered solvers · contract **1.0**
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
| Natural frequencies and mode shapes of any part mesh | `POST /solve/modal_part_3d` |
| Eigen-buckling of a slender member (the four Euler column cases) | `POST /solve/buckling_beam` |
| Pipe flow: friction factor, Darcy-Weisbach pressure drop, regimes | `POST /solve/pipe_flow_darcy` |
| Standard-part library and catalogs, with ISO sources | `GET /library`, `/catalog` |
| Material table for the simulator and reports (`E`, `nu`, `Sy`, `rho`), with a source on every row | `physics/part_fea.py` (`material_catalog()`); the ids are what `POST /part-check` accepts |
| Print-ready engineering report: requirements, checks, analyses, BOM, solver evidence | `POST /report` |
| Chat → requirements or CAD code (LLM, with built-in fallback parser) | `POST /chat` |
| Chat: a whole product becomes a plan of parts, and the part-by-part job starts only after the engineer says Start | `POST /chat`, `POST /assembly/plan`, `POST /assembly/design` |
| Calculation references (formula + source) stored as candidates | `GET /references/db`, `/references/db/<slug>` |
| Free-form design as a background job, with every check it ran | `POST /design/job`, `GET /design/job/<job_id>` |
| One door to check the reference lookup, and a self-test of it | `GET /references/status`, `/references/selftest` |
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
research/            sourced lookups: web search, page fetch, quote-checked extraction,
                     reference database (candidates on the reference-db branch)
docs/                API contract and UI integration notes
tests/               pytest suite + standalone QA scripts
```

## Configuration

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port (default `8000`; set automatically by Render/Railway) |
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `NVIDIA_API_KEY` | Enable LLM providers; without keys `/chat` uses the built-in parser |
| `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` (Qwen), `MOONSHOT_API_KEY` (Kimi) | Enable the OpenAI-compatible providers (DeepSeek, Qwen, Kimi) |
| `IDRIS_LLM_ORDER` | Provider preference, default `claude,gemini,openai,nvidia,deepseek,qwen,kimi` |
| `IDRIS_MODEL_<PROVIDER>` | Override the model per provider |
| `IDRIS_LLM_PER_MIN`, `IDRIS_LLM_DAILY_CAP` | LLM rate limits (default 12/min, 400/day) |
| `IDRIS_REPLY_LANGUAGE` | Reply language (`en` default, `auto` follows the request) |
| `IDRIS_CAD_CALL_S` | Timeout for one AI CAD call (default `120` s for a design job; was 45) |
| `IDRIS_PART_BUDGET_S` | Whole-design budget for the background job (default `600` s) |
| `IDRIS_CAD_MAX_LINES` | Code length a design job asks the AI for (default `200` lines; was 80) |
| `IDRIS_DESIGN_ROUNDS` | Validation rounds per design (default `4`) |
| `GITHUB_TOKEN` | Fine-grained token, **this repo only**, Contents **read/write**: reads and stores calculation references. Without it the chat still answers, without the reference database |
| `IDRIS_REFERENCE_BRANCH` | Branch that holds `references/*.json` (default `reference-db`; `main` is refused in code) |
| `IDRIS_GITHUB_REPO` | Repository for the reference database (default `praditya0830-ai/Idris-ai`) |

### Menambah provider selain Claude (DeepSeek, Qwen, Kimi)

Ketiganya memakai adapter OpenAI-compatible yang sama, jadi tidak ada kode baru per provider:

1. Isi key di Render → Environment: `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` (Qwen), `MOONSHOT_API_KEY` (Kimi).
   Jangan pernah menaruh key di repo atau di berkas `.env` yang ter-commit.
2. Cek satu per satu: `GET /chat/ping?provider=deepseek` (juga `qwen`, `kimi`). Balasannya menyebut model
   yang dicoba dan error persisnya, misalnya model yang sudah EOL atau key yang salah region.
3. Atur urutannya: `IDRIS_LLM_ORDER=claude,deepseek,qwen,kimi` — hanya provider yang punya key yang dipakai,
   dan provider berikutnya dicoba bila yang di depan gagal. Ganti model dengan `IDRIS_MODEL_DEEPSEEK`
   (daftar dipisah koma, yang pertama dipilih).

Default model: `deepseek-flash` / `deepseek-v4-pro`, `qwen3.8-max` / `qwen3.7-plus`, `kimi-k3.1-pro`.
Model seri Kimi K3 hanya bisa berpikir (tidak bisa dimatikan), jadi adapter memberi mereka token lebih;
model lain dikirimi setelan thinking-off (`thinking: disabled` untuk DeepSeek, `enable_thinking: false`
untuk Qwen/Kimi) supaya jawaban JSON pendek tidak habis dipakai reasoning.

### Database acuan perhitungan (references on their own branch)

Every message in `/chat` that asks for a formula settles its calculation reference in this order, before the model
speaks:

1. **Solver registry** (`GET /solvers`, the bench in `ai/engineer.py`). If a solver covers it, that solver is used and
   named with its benchmark. Nothing is browsed.
2. **Stored references** (`references/<slug>.json` on the `reference-db` branch). An existing candidate for the topic is
   shown, not searched again.
3. **Web** (standard → textbook → datasheet; forums and blogs are never stored). What could be read is stored as a
   **candidate** and shown with the label
   `**Unverified formula from <source>: not used for any IDRIS result yet**`.

A candidate is text only. It may not produce a result, a safety factor, a ranking or any CAD/requirements field: the
number gate in `ai/engineer.py` still treats every one of its numbers as a guess. `status` is never changed by code —
Arya promotes a reference to `verified` by hand, and only after a worked example from the same source, solver code and a
passing benchmark in `/verify` exist.

The lookup only runs for a message that asks for a formula, a standard or a calculation reference. A design request
(whatever route the router picks), a greeting, and a message that never asks for a formula never look one up, and a
message with no recognised topic is never searched or stored. Since v0.36.2 a formula question asked while a design is
open **is** still looked up; only a request that changes or repairs that design (`context.cad_error`, or an answer to a
missing size) is skipped. When a registered solver covers the topic it is named with its benchmark and nothing is
browsed, even if the bench did not run on that message.

When the lookup is empty the chat says why, in the answer and in `references.reasons`: `search_off` (no web-search key),
`db_unreachable` (GitHub or the token), `no_formula` (the pages that were read had none) or `no_topic`.
`GET /references/status` shows the same configuration as booleans plus the provider name only — never a key — and
`GET /references/selftest?q=...` runs exactly the lookup `/chat` would run (no final AI answer), at most once a minute.

Why a branch: Render deploys automatically from `main`, so a commit there restarts the live server. The database writes
through the GitHub Contents API only, refuses `main`/`master` in code, never overwrites an existing file, and refreshes
its in-memory list at most every 10 minutes. With GitHub unreachable, or without a token, the chat keeps answering and
says the reference could not be stored.

Setup, once:

1. Create the branch: `git push origin main:reference-db` (or **Branches → New branch** on GitHub). The first write
   needs it to exist; until then a candidate is shown in the chat but reported as not stored.
2. Render → Environment: `GITHUB_TOKEN` = fine-grained token, **only this repository**, Contents **read/write**.
   Nothing else is needed; the token never reaches the browser or the repo.

```bash
curl https://idris-ai-rwd6.onrender.com/references/status
curl "https://idris-ai-rwd6.onrender.com/references/selftest?q=what+is+the+formula+for+fillet+weld+strength"
curl https://idris-ai-rwd6.onrender.com/references/db
curl https://idris-ai-rwd6.onrender.com/references/db/<slug>
```

### Chat → assembly: the plan comes first, and nothing starts without a yes

A sentence for a whole product ("design whole bicycle 170 cm", "seluruh sepeda") is not designed as one solid. The
router reads it as an **assembly**, and `/chat` answers with a plan: the parts and their roles, every interface size
with its source, the open questions, and the estimate — `"This assembly has 4 parts: 3 to design (about 1 min
each), 1 standard parts. Estimated time: about 3 minutes."` — ending in **`Start?`**. That reply also carries the
`assembly` block: `{plan, questions, estimate, job_id: null, poll: null, eta_s, not_checked}`.

The job starts only on a later message that carries **both** `context.assembly_plan` and
`context.assembly_confirm: true`, and that message goes through the same door as `POST /assembly/design`, so no
guard is bypassed:

- a size with **no source** (`user`, `standard: <name>`, `default: <reason>`) keeps the job waiting, and the reply asks
  for that one size; `context.assembly_confirm_dims` accepts the ids the engineer confirmed;
- a **full queue** (`IDRIS_ASSEMBLY_MAX_JOBS`, `QUEUE_MAX`) is answered with "Another assembly is being designed right
  now, so this one was not started. Try again in a few minutes." — never queued silently behind another costly job;
- a spent **AI budget** (`ai_budget.gate()`) writes no plan and starts no job, and the reply is the budget message.

`POST /assembly/plan` is the planner alone (no job), `POST /assembly/design` starts the job once the plan is in the
body, and `GET /assembly/design/<job_id>` shows each part as it finishes. The assembly as one system is still not
analysed: the block says what was not checked.

### The orientation contract of a placed part (UI v65)

`origin_mm` alone is not a placement: a part can sit at the right point and still be turned wrong, and with only
`axis` it is free to spin about that axis. Every frame therefore carries both directions, in assembly coordinates:

- `axis` — where the part's own **+Z** points;
- `x_axis` — where its own **+X** points; it may not be parallel to `axis` (the plan is refused with the reason,
  because the part would still be free to spin);
- `y_axis` — computed as `axis × x_axis`, so the three are a right-handed unit basis the Studio builds a rotation
  from. A sloppy `x_axis` is projected perpendicular to `axis` first, so the basis is always orthonormal.

When a plan names no `x_axis`, one is derived from the world axis least aligned with `axis` and the frame says
`x_axis_source: "derived"` — deterministic, but not a decision the engineer made, and the reply never pretends
otherwise. `GET /assembly/design/<job_id>` returns the same three vectors per part, plus
`assembly.frame_contract`. IDRIS checks geometry and interference; it does not verify that the planned orientation
is the one the product needs.

### `context.shape_family`: the Studio's own family for one part (UI v65)

`POST /chat` accepts `context.shape_family = {name, principle, design_space}` — the family the Studio can edit
("l_bracket", "two legs at 90 degrees carry the load into the wall", `{L: [20, 80], leg_t: 3}`). It is checked
before it reaches a prompt: name (≤ 60 chars), principle (≤ 400), and a design space of numbers, strings, booleans
or lists of at most 8 of them (≤ 24 entries, ≤ 2000 characters). The accepted family is added to the CAD writer's
user block, and the answer carries `shape_family: {used, name, principle, design_space, applied_to, not_checked}`.
A family that cannot be read is **dropped**, and the answer says why (`{used: false, why}`) instead of half-using it.
An assembly answer says it was not applied, because parts are designed one at a time.

What this does not do: nothing verifies that the generated code stays inside the design space, that its parameters
match the family, or that the family is the right one for the load. `not_checked` in the reply says exactly that.

### Material table: every density and strength names its source

`physics/part_fea.py` holds the material table the UI selector reads and `POST /part-check` accepts
(`aluminium`, `steel`, `stainless`, `titanium`, `brass`, `cast_iron`, `pla`, `abs`, `nylon`). Each row carries
`source: {density, strength, stiffness}` — ASM Handbook Vol. 2, EN 10025-2, ASTM A240, EN 12164, EN 1561, filament
and `EOS PA 2200` datasheets — so a mass or a safety factor can be traced back to a page instead of a habit.
Two limits are stated, not hidden: `stainless` is 304 at the **standard minimum** (`Rp0.2 ≥ 205 MPa`, annealed), and
`cast_iron` (GJL-250) does not yield, so its reported safety factor is `Rm / von Mises` and its row says so
(`brittle: true`). Printed plastics are datasheet values at 100 % infill; a real FDM part is weaker across layers.

Deploy as a Render Web Service with start command `python3 main.py`. Mount a disk only if
you need generated files to survive deploys.

## New solvers in v0.39, and what is still "Not yet"

Three solvers were added, each with published benchmarks that `GET /verify/<name>` runs live:

- **`modal_part_3d`** — natural frequencies of any part from its mesh, on the same voxel engine as
  `/part-check`. Benchmarked against Euler-Bernoulli for a cantilever (Blevins Table 8-1) and a square plate with
  all four edges clamped (Leissa, NASA SP-160; $\omega a^2\sqrt{\rho h/D} = 35.99$), both within 5 %/4 %. It reports
  the voxel volume error and says that clamped supports make it an upper bound for a bolted joint. It does not
  compute forced response or any electromagnetic excitation: only the structure's own frequencies.
- **`buckling_beam`** — linear eigen-buckling of a slender prismatic member from a beam-column finite element
  (bending stiffness + geometric stiffness). All four Euler column cases are benchmarks and all four come out
  exact to the printed value: K = 2, 1, 0.699, 0.5. It says in the answer that linear buckling ignores
  imperfections and is therefore an **upper bound**, and that it cannot do plate/shell buckling, torsion or
  lateral-torsional buckling of an open section.
- **`pipe_flow_darcy`** — Darcy-Weisbach with `f = 64/Re` below Re 2300 and Colebrook above Re 4000, plus a sum
  of minor-loss K values. Benchmarked against Hagen-Poiseuille (exact) and two Colebrook factors read off the
  Moody chart. The transitional band 2300 < Re < 4000 is reported as *transitional* and flagged, not used.

`GET /solvers` now answers with two lists: `solvers` (registered, benchmarked) and `planned` (named, with the
reason it does not exist and what would unblock it). Planned today: `drop_test` (decision and reasoning in
`docs/P2-drop-test-decision.md`), `topology_simp_3d` (a qualitative MBB picture is not the quantitative benchmark
IDRIS requires), `buckling_plate` (the shell element has no geometric stiffness yet) and `cfd_3d`.

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
