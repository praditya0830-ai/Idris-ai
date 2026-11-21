# Idris Core — ACE engine

Backend for **IDRIS Design Studio**: deterministic engineering solvers plus parametric CAD.
Requirements in → validated gearbox variants → 3D assembly, BOM, drawings → FEA, modal and
thermal analyses. Every solver ships benchmarks with a published source and states its own
limitations; physics is never delegated to an LLM.

- Live: <https://idris-ai-rwd6.onrender.com> (`GET /health` lists every route)
- Solver version: **0.40.0** · **27** registered solvers · contract **1.0**
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
| STEP AP242 with named products, colours, mm units and an assembly tree | `POST /exchange/step-ap242` |
| SolidWorks VBA macro that rebuilds the feature tree as parametric features | `POST /exchange/solidworks-macro` |
| Manufacturability of any part from its mesh: measured wall thickness, overhang, draft, drill series | `POST /dfm/part` |
| Standard-part library and catalogs, with ISO sources | `GET /library`, `/catalog` |
| Material table for the simulator and reports (`E`, `nu`, `Sy`, `rho`), with a source on every row | `physics/part_fea.py` (`material_catalog()`); the ids are what `POST /part-check` accepts |
| Print-ready engineering report: requirements, checks, analyses, BOM, solver evidence | `POST /report` |
| Chat → requirements or CAD code (LLM, with built-in fallback parser) | `POST /chat` |
| Chat: a whole product becomes a plan of parts, and the part-by-part job starts only after the engineer says Start | `POST /chat`, `POST /assembly/plan`, `POST /assembly/design` |
| Calculation references (formula + source) stored as candidates | `GET /references/db`, `/references/db/<slug>` |
| Free-form design as a background job, with every check it ran and the code the browser must build (`waiting_cad`) | `POST /design/job`, `GET /design/job/<job_id>` |
| How many design concepts to compare (3 / 5 / 7 / 10), asked before any code is written | `POST /chat` with `concept_count` |
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

### The design job and the browser: `waiting_cad` (UI v66 A)

A design job and the Studio each do half of the work: the job writes code, the browser builds it, measures it and
runs the strength check. `GET /design/job/<job_id>` therefore says not only that it is waiting, but for what:

- `waiting_for` — `"browser_build"` or `"fea"`, with `waiting_round` and `waiting_s`;
- `waiting_cad` — the **contract of that round**, the same `cad` object `/chat` returns (`params`, `code`,
  `expect`, `requirements`, `steps`, `concept` when there is one), so the Studio can build exactly that;
- `waiting_cad_hash` — sha256 of the round's code. `POST /design/job/<id>/feedback` accepts an optional `cad_hash`;
  a hash from another round is refused with **409** and the hash the job is waiting for, so numbers measured on one
  code are never counted for another (omitting it still works, for an older Studio);
- `rounds_data` — one entry per finished round: `{round, cad_hash, label, checks, changed}`. `changed` lists the
  parameters a repair really changed (`{param, from, to, why}`, `why` being the failed check or FEA number that
  caused it). The full code of old rounds is not repeated; only the round being waited for and the final
  `result.cad` carry code, so a status polled every second stays small.

### Before the design: the brief (v0.40)

A new design — a single part **or** a whole product — is judged for completeness before anything is written. The
minimum is: (a) what the object is **and** what it is for, (b) its key sizes or the space it must fit, and (c) when
strength matters, the load or service condition. If one is missing, `POST /chat` returns route **`brief`** and starts
no code generator and no assembly planner. The reply is an engineer talking, not a form:

```json
{"route": {"route": "brief", "of": "free_design"}, "answer": "…",
 "brief": {"object": "drone", "purpose": "", "kind": "part", "envelope": "", "key_sizes": "", "loads": "",
           "assumptions": [], "decided": [], "open": [], "ready": false},
 "questions": [{"id": "q1", "text": "What should it carry?", "options": [{"label": "FPV 5-inch", "note": "fast, agile"}],
                "allow_free_text": true}],
 "suggestions": [{"id": "s1", "text": "Folding landing gear", "why": "fits in a backpack"}],
 "ready": false}
```

- **at most three questions**, each with 2–4 real options and a short qualitative trade-off. A performance number is
  never invented and is stripped from AI prose (`flies 25 minutes` cannot leave the module); a size class written as
  the option itself (`5-inch`, `450 mm`) is allowed.
- **`brief` is the memory of the conversation.** The Studio sends it back as `context.brief` every turn; `merge` only
  accepts a digit the engineer actually wrote (mm, or cm / m / inch written with that unit), so a reader can fill
  words but cannot smuggle in a size.
- **ready**: the reply repeats the brief in one engineer sentence and ends with `Shall I start? How many concepts:
  3 / 5 / 7 / 10?` plus `ask: {k: "concept_count", …}` — the same question UI v66 B already understands, still with
  no code written. For an assembly the next turn goes to the existing plan (`POST /assembly/plan`), which still needs
  its own Start.
- **just design it with your assumptions**: `langsung aja desain`, `just design it`, `go ahead`, … continues the same
  turn, and every gap is written into `brief.assumptions` as `{what, value, why, assumed: true}`; the request handed
  to the code writer carries them, marked assumed.
- a sentence that is complete from the start never sees the brief: the router sends it to the design path as before.
  A question in the middle of a brief is answered by the engineer chat; the brief stays.
- one AI call per brief turn (max_tokens 800) through `ai_budget`. With no AI reachable the route falls back to the
  old path instead of showing questions nobody wrote. `IDRIS_BRIEF_TIMEOUT_S` (20 s) bounds that call.

The language follows the message (`reply_language` wins when it is set): an Indonesian sentence is answered in
Indonesian, and technical terms may stay English.

### How many concepts? (UI v66 B)

`POST /chat` accepts a top-level **`concept_count`** (integer 1–10). A fresh free design with no `concept_count`
writes **no code**: the answer is `"How many concepts should I design?"` with
`ask: {k: "concept_count", options: [3, 5, 7, 10], default: 3}` — one question, no AI code call. With a count:

- **1–4**: that many concepts are written now, at most four AI calls at a time; the /chat time budget
  (`IDRIS_CAD_BUDGET_S`) is shared, and a concept it cannot reach comes back `ok: false` with
  `"time budget used up"` instead of quietly missing;
- **5–10**: the concepts do not fit in one chat reply, so `/chat` starts `POST /design/job` with the same
  `concept_count` and returns **202** with `job_id` and `poll`; the job writes them in the background and
  `GET /design/job/<job_id>` returns `result.concepts` (label `Not validated`: nothing is built or checked until
  the browser does it);
- every AI call goes through `ai_budget`; when the budget runs out mid-way, the remaining concepts are returned
  `ok: false` with that reason.

A follow-up (`context.pending_cad`), a repair (`context.cad_error`) and a change to the open part
(`context.current_cad`) are never asked, and neither is a library, primitive, gearbox or assembly route: those are
about a design that already exists. `IDRIS_CONCEPTS=0` turns the question off and keeps the old one-design path.

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
## Export to SolidWorks: what can be written and what cannot (v0.39)

A native **`.SLDPRT` cannot be written** without SolidWorks or its licensed API. IDRIS does not claim to. What a
buyer can be given instead:

- **STEP AP242** (`POST /exchange/step-ap242`) — the same faceted geometry as the AP214 export, but written as an
  AP242 file with one named `PRODUCT` per part, a `COLOUR_RGB` per part attached through `STYLED_ITEM`, millimetre
  units stated in the geometric context, and an assembly tree: one `NEXT_ASSEMBLY_USAGE_OCCURRENCE` plus a
  `REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION` per instance, so each part lands at its own rotation and
  translation. Faces stay planar polygons — a bore is a polygon, not an analytic cylinder.
  The writer does not trust the client's vertex sharing: before a shell is built, coincident vertices are welded
  with a relative tolerance (1e-6 × the diagonal, at least 1e-4 mm), degenerate triangles are dropped, and every
  edge has to be shared by exactly two triangles. A triangle soup that used to become nine solids for one part
  now becomes one (v0.39.1); `solids_per_part` and `warnings` are returned, and a mesh with open edges is refused
  with the number of open edges rather than written silently.
  Self-check: every `#id` reference resolves, every edge is used once per direction, and the file is read back
  (products, solid count, volumes by the divergence theorem) before it is returned. Reading it with OpenCascade
  could not be done here: `OCP` is not installed in this environment, so that check is reported as not run.
- **"Rebuild in SolidWorks" macro** (`POST /exchange/solidworks-macro`) — a VBA module generated from the IDRIS
  feature graph: one sketch per sketch feature (with real `AddDimension2` dimensions, values in metres because that
  is the API unit), `FeatureExtrusion2` for extrudes, `FeatureRevolve2`, `FeatureFillet3`, a circular cut for a
  hole, and linear/circular patterns. Features with no mapping (an imported solid, an opaque part) are skipped and
  written into the macro as `' NOT TRANSLATED` comments and into the export report. A part written by the AI as
  code has no feature tree at all: for that, only STEP is offered, and the endpoint says so.

  v0.39.2 (X2) adds the features a classic Studio part uses and the macro used to refuse:

  | IDRIS feature | parameter | SolidWorks call |
  |---|---|---|
  | `sketch` off centre | `cx`, `cy` (mm) | the profile is drawn at its centre: `CreateCornerRectangle` / `CreateCircleByRadius` / `CreatePolygon` |
  | `extrude` cut | `op: "cut"` | `FeatureCut4`, with `distance`, `direction` (+normal / -normal / symmetric) or `through: true` |
  | `hole` blind | `depth`, `through: false` | `FeatureCut4` to `depth` instead of through all |
  | `hole` from the top | `from: "top"`, `z_top` | a reference plane at `z_top` (`InsertRefPlane`, 8 = `swRefPlaneReferenceConstraint_Distance`), then the cut from there |
  | `hole` counterbore | `cbore_d`, `cbore_depth` | two cuts (the bore, then the wider counterbore); a Hole Wizard hole needs a standard-table selection a macro cannot make, and the report says so |
  | `chamfer` | `distance`, `edges` | `InsertFeatureChamfer` in equal-distance mode (18 = `swChamferDistanceDistance Or swChamferEqualDistance`) |
  | `mirror` | `plane` (XY / XZ / YZ) | `InsertMirrorFeature2(False, False, False, False, 0)`, the last feature marked 1 and the plane marked 2 |

  Example (a plate with an off-centre pocket, a blind hole from the top, a counterbored hole, a chamfer and a
  mirror — the model S3 sends):

  ```json
  {"model_id": "plate", "features": [
    {"id": "f1", "op": "sketch", "params": {"plane": "XY", "profile": "rectangle", "w": 80, "h": 50}},
    {"id": "f2", "op": "extrude", "deps": ["f1"], "params": {"distance": 10}},
    {"id": "f3", "op": "sketch", "deps": ["f2"], "params": {"plane": "XY", "profile": "circle", "w": 20, "cx": 20, "cy": 5}},
    {"id": "f4", "op": "extrude", "deps": ["f3"], "params": {"distance": 3, "op": "cut"}},
    {"id": "f5", "op": "hole", "deps": ["f4"], "params": {"x": -20, "y": 0, "d": 6, "through": false, "depth": 8, "from": "top", "z_top": 10}},
    {"id": "f6", "op": "hole", "deps": ["f5"], "params": {"x": 30, "y": -10, "d": 5, "through": true, "cbore_d": 9, "cbore_depth": 4}},
    {"id": "f7", "op": "chamfer", "deps": ["f6"], "params": {"distance": 1}},
    {"id": "f8", "op": "mirror", "deps": ["f7"], "params": {"plane": "YZ"}}
  ]}
  ```

  Still not translated, and still stated as such in the macro and the report: tube, sphere, slot and free-line
  sketches, a hole from the top without `z_top`, and anything the Studio sends with `op: "not_translated"`.
  `chamfer` and `mirror` are carried by the exchange path (macro and `/model/validate`); the server-side kernel
  builder answers with a named `BuildError` rather than guessing geometry it cannot build.

**Not tested in SolidWorks.** There is no SolidWorks in this environment, so the macro is checked only for VBA
structure (balanced blocks, only VBA statements, `Application.SldWorks` present) and for content (every feature is
either an API call or a stated skip). Someone with SolidWorks has to run it once and confirm the rebuild. The
endpoint returns `not_executed` saying exactly that.

DWG is still not written: the drawing export is DXF (see the Drawing route).

## DFM for any part: measured, not guessed (v0.39)

`POST /dfm/part` takes the same triangle mesh as `POST /part-check` plus `process` (cnc, fdm, sls, sheet, weld,
cast) and `material`, and answers with findings that each carry a number, a limit and the limit's source:

- **wall thickness** — measured, not read from a parameter list: an exact ray-triangle cast (Möller–Trumbore) is
  fired from sampled surface triangles along the inward face normal to the first other surface of the part, so
  the value is as accurate as the mesh rather than one voxel wide. (v0.39.1: the cast used to run through a voxel
  grid whose edge came from the shortest side of the bounding box — an 80 × 60 × 60 bracket with 5 mm walls read
  "7.5 ± 5 mm", because its cell was 7.5 mm. It is now 5.0 mm.) Above 150k triangles the voxel fallback is used
  and reports its cell as `resolution_mm`; a limit finer than a quarter of that cell makes the check
  `not_checked` with that reason, never pass/fail. `method` and `resolution_mm` travel with the numbers.
  Statistics: min, 5th percentile (what the limit is compared with), median, max.
- **units** — every finding states the unit of what it measured: `mm` for a length, `count` for a number of
  holes, `fraction` for a share of the surface. A count is never labelled millimetres (v0.39.1).
- **hole diameters** — checked against the DIN 338 / ISO 235 twist-drill series *when the caller declares the
  holes*. A hole cannot be recognised reliably in a faceted mesh (a bore, a slot and a tessellation artefact look
  alike), so an undeclared hole is `not_checked`, never a pass.
- **overhang** (FDM) — the share of the surface whose normal leans more than 45 deg from the build direction.
- **draft** (cast) — the share of the surface within 1 deg of parallel to the pull direction.
- **undercuts, tool access, inside corner radius** — `not_checked` with the reason: they need the solid and the
  sketch, not just the triangles.

Limits keep their status: the weld fillet leg (AWS D1.1 Table 7.7) and the drill series are quoted from a
standard and can fail a part; the wall minimums per process (including the new SLS and casting rows) and the
overhang/draft angles are workshop guidelines, so they are **warnings**, never failures. `wall.thickness` per
vertex is returned for the Studio's colour map.

What it is not: no cost, no cycle time, no tolerance stack-up, and no feature recognition.

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
