# Ship project developer experience without lab

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. The
canonical product authority is `docs/post-training/README.md` and the six
documents it indexes. The DX working authority for vocabulary and golden path is
`docs/developer-experience.md` (locked 2026-07-23). This plan implements that
brief. It builds on portable project layout
(`docs/decisions/0013-portable-project-layout.md`,
`docs/plan/portable-project-layout-and-consumer.md`) and the primary CLI /
`posttrain.work` extraction in `docs/plan/polished-framework-release.md`. It
does not change the frozen meanings of selections, job kinds, work packages,
runs, artifacts, or observation.

## Purpose / Big Picture

After this work, a CarbonTeq developer can create an ordinary repository with
`posttrain init` (layout **and** dependency install in one command), bind
models/datasets/environments/settings from a **global catalog** and/or project
overlay, run tracked jobs through framework-shipped definition ids such as
`train/trl-sft@1` and `train/trl-grpo@1` **without** importing `posttrain_lab`
or passing `--host`, and open local Observatory with
`posttrain observatory up`.

Capability already exists in packages: `posttrain.data` adapters (Hugging Face
SFT formats `auto|messages|prompt-completion|alpaca|sharegpt`, preferences,
Nemo, Verifiers traces) and `VerifiersEnvironmentRolloutBridge` in
`posttrain.train.integrations`. This plan **wires** those into catalog decode,
materialize-on-first-use, and `posttrain.jobs` standard definitions. It does
not rebuild adapters or bridges, and it does not defer GRPO/distill/eval to a
later phase.

Today the only end-to-end seat→request compositions and many dataset/env
factories live under `apps/lab`. `posttrain init` creates an empty skeleton.
`posttrain work-package run` requires `--host`. Developers rationally copy lab.
That is the failure this plan closes.

Visible proof: external (or temp) projects from `posttrain init` that, from
installed wheels, validate and run an SFT path and an environment-backed path
(GRPO or distill/eval) with Trackio evidence and Observatory bring-up, with no
`posttrain_lab` import. Lab remains the qualification suite.

## Progress

- [x] (2026-07-23T14:40Z) Locked DX brief at `docs/developer-experience.md`.
- [x] (2026-07-23T14:40Z) Authored this implementation plan.
- [x] (2026-07-23T15:05Z–16:40Z) Locked authoring model: global + project
  catalog, work-package bindings, standard jobs wire existing data/env bridges,
  init installs (no `sync`), Observatory up, materialize on first use, no
  phase-2 deferral for datasets/GRPO.
- [x] (2026-07-23T16:45Z) Consolidated plan rewrite to match locked brief and
  “wire existing packages” scope.
- [x] (2026-07-23T17:25Z) Added autonomous agent runbook
  `docs/plan/project-developer-experience-agent-runbook.md` (pasteable command +
  milestone goal stack + validate/fix loop).
- [x] (2026-07-23T17:27Z) Milestone A — Baseline amendment (`04` / `05` /
  README); canonical docs now define `posttrain.jobs`, project-entry/job-runtime
  vocabulary, global catalog materialization, adapter/bridge wiring, init
  install, and Observatory bring-up.
- [x] (2026-07-23T17:32Z) Milestone A1 — Usable global catalog hub slice:
  packaged SFT fixture selection, empty-overlay discovery proof, and
  idempotent first-use dataset validation/materialization.
- [x] (2026-07-23T17:39Z) Milestone B0 — Catalog dataset decode/materialize,
  installed environment factory discovery, and public GRPO/distillation
  request builders wired to existing adapters/Verifiers bridge.
- [x] (2026-07-23T17:47Z) Milestone B — `packages/jobs`
  (`posttrain.jobs`) standard definitions + `build_job_runtime` for SFT, DPO,
  GRPO, distill, serve, eval, and transform.
- [x] (2026-07-23T17:49Z) Milestone C — `project.toml` tracking/entry, CLI
  validation/run without required `--host`, entry override, and additive
  Host→JobRuntime public aliases.
- [x] (2026-07-23T17:54Z) Milestone D — installable SFT and GRPO starter
  templates with visible catalog settings, standard work packages, selected
  extras, project `.venv`, and no lab dependency.
- [x] (2026-07-23T17:58Z) Milestone E — project-aware
  `posttrain observatory up` composes and serves the read product in-process
  for Trackio or W&B, with URL output and install guidance.
- [x] (2026-07-23T18:05Z) Milestone F — isolated wheel consumer proofs for
  SFT and GSM8K-backed GRPO paths, plus current quickstart and Lab guidance.
- [x] (2026-07-23T18:06Z) Final Validation and Acceptance — locked sync,
  repository-wide ruff/pyright/import contracts, full package tests, isolated
  consumer wheels, lock digest, and diff whitespace checks pass.
- [ ] Optional in-plan if cheap: `dataset add` / `environment add` helpers,
  `job plan` / `run show` aliases, `catalog materialize`, extra hub packages in
  `project.toml`.

## Surprises & Discoveries

- Observation: Lab GRPO and distillation definition factories currently close
  over lab-specific request types (`VerifiersGRPOJobRequest`,
  `GSM8KDistillationJobRequest`) rather than public `GRPORequest` /
  `OnPolicyDistillationRequest` in `packages/train`.
  Evidence: `apps/lab/src/posttrain_lab/work_packages/definitions.py`.
- Observation: `04-framework` says `packages/work` must not own concrete job
  definitions. Standard jobs need `posttrain.jobs`, not a dump into
  `posttrain.work`.
  Evidence: `docs/post-training/04-framework.md`.
- Observation: Dataset and GRPO capability largely **already exists**. The DX
  gap is composition through catalog + standard jobs, while lab still owns
  end-to-end wiring and some scenario bridges.
  Evidence: `packages/data/.../adapters/huggingface.py`,
  `packages/train/.../integrations/verifiers.py`,
  `packages/catalog/.../__init__.py` `catalog_decoders()`.
- Observation: the earlier primary-CLI amendment still said concrete
  definitions and provider wiring came from a host.
  Evidence: `docs/post-training/README.md` before Milestone A. The new
  project-DX amendment explicitly supersedes that composition boundary while
  preserving the older entry as release history.
- Observation: catalog selection ids may carry an `@revision` suffix, while
  canonical `posttrain.data` dataset identities deliberately do not.
  Materialization therefore preserves the full catalog selection id in its
  manifest and passes the unversioned logical id plus explicit revision to the
  existing adapters.
  Evidence: `posttrain.common` catalog-id grammar versus
  `packages/data/src/posttrain/data/models.py`.
- Observation: a resolved Verifiers `EnvironmentBinding.factory` already
  contains enough native `EnvConfig` to construct its installed taskset.
  Training needs only deterministic task selection plus the existing
  `VerifiersEnvironmentRolloutBridge`; the lab's package-specific request
  wrappers were unnecessary.
  Evidence: `verifiers.v1.env.Environment` and
  `packages/train/src/posttrain/train/integrations/verifiers.py`.
- Observation: after a clean all-package sync, uv reused an incomplete cached
  wheel for the already-pinned Trackio commit. Rebuilding that same immutable
  commit from a cleared package cache restored the full `trackio` import
  package; no pin change was required.
  Evidence: `uv sync --all-packages --locked` followed by a forced
  `carbonteq-trackio` cache rebuild.
- Observation: invoking a generated environment's absolute `posttrain` binary
  while the shell remains in another Posttrain project still discovers that
  shell directory, as a normal project CLI should. The original D validation
  command omitted changing into the generated project.
  Evidence: the first `/tmp/posttrain-sft-demo/.venv/bin/posttrain doctor`
  resolved the repository project; the same command after `cd` resolved
  `sft-demo`.
- Observation: a Trackio project is unavailable to Observatory until its first
  tracked run exists; once a canonical run is finished, the same live server
  transitions from degraded to ready and lists that run without restart.
  Evidence: live `/health/ready` and `/api/v1/runs` responses for
  `sft-demo` before and after `observatory-proof-1`.
- Observation: uv requires transitive Git URL dependencies to be repeated as
  project-root direct requirements or constraints. A generated wheel consumer
  could not resolve Trackio/TRL/Verifiers until init preserved those immutable
  references from installed package metadata.
  Evidence: the first isolated SFT wheel init failed on
  `carbonteq-trackio`; the corrected generated `pyproject.toml` installed both
  starter environments from the wheelhouse.
- Observation: CUDA-bearing starter extras exceed the desktop pytest tmpfs.
  Consumer proof environments need a disk-backed temporary directory even
  though the wheelhouse itself is small.
  Evidence: tmpfs installation failed with `No space left on device`; the same
  test under `/tmp` passed and cleaned itself afterward.

## Decision Log

- Decision: Lock DX direction in `docs/developer-experience.md`; implement via
  this plan (not the superseded audit’s “host SDK”).
  Rationale: Agreed product model.
  Date/Author: 2026-07-23 / agent with user

- Decision: Ship standard job definitions in `packages/jobs` (`posttrain.jobs` /
  `posttrain-jobs`).
  Rationale: train/eval/serve must not import each other; work must not own
  concrete definitions; jobs package is the composition layer.
  Date/Author: 2026-07-23 / agent with user

- Decision: `posttrain init` creates the project **and** installs dependencies.
  No `posttrain sync` command. Later refreshes use the project’s own `uv sync`
  / documented wheelhouse install. Optional `--no-install` for nested CI only.
  Rationale: One bootstrap verb.
  Date/Author: 2026-07-23 / agent with user

- Decision: **No phase-2 deferral** for dataset formats or GRPO/distill/eval
  bridges. Productize existing `posttrain.data` adapters and
  `VerifiersEnvironmentRolloutBridge` / eval factories through catalog decode,
  materialize, and standard jobs. Lab keeps scenario policy only.
  Rationale: User correction — packages already support this; wire it.
  Date/Author: 2026-07-23 / agent with user

- Decision: Catalog dataset format names must match existing adapter literals
  (`auto`, `messages`, `prompt-completion`, `alpaca`, `sharegpt` for SFT;
  preference formats as in `posttrain.data`). Do not invent parallel names such
  as `chat-messages`.
  Rationale: Avoid a second vocabulary on top of working adapters.
  Date/Author: 2026-07-23 / agent with user

- Decision: Global published catalog (`posttrain-catalog` base, later optional
  extra hub packages) is auto-composed and discoverable via
  `posttrain catalog list`. Project overlay is for proprietary selections and
  overrides. Work packages may ref global ids directly. Global entries are
  **pointers**; first validate/run materializes locally into
  `.posttrain/state/` / project env (idempotent). Env materialize checks
  importability against project pins; prefer lockfile pins over silent
  unconstrained Hub installs on every run.
  Rationale: Hub DX + reproducibility.
  Date/Author: 2026-07-23 / agent with user

- Decision: Dataset registration in `.posttrain/catalog/` (or global catalog);
  bytes on Hub (cache in state) or under project `data/`. Environment
  **implementation** is a Verifiers package in dependencies; **registration**
  is catalog family `environment`. Install → register → bind; discover with
  catalog list/show/doctor.
  Rationale: Clear layout and linkage.
  Date/Author: 2026-07-23 / agent with user

- Decision: Additive rename `WorkPackageHost*` → `ProjectExecutionRequest` /
  `ProjectEntry` / `JobRuntime` with aliases; deprecate required `--host`.
  Rationale: Vocabulary; avoid flag day.
  Date/Author: 2026-07-23 / agent with user

- Decision: Keep `source_layer: base | overlay` as the serialized provenance
  contract while using “global catalog” in developer-facing language.
  Rationale: Milestone A changes the authoring model without an unnecessary
  persistence-format rename.
  Date/Author: 2026-07-23 / agent

- Decision: Use a two-row packaged `messages` fixture for the first global
  dataset entry, while supporting the same declarative load-plan shape needed
  for Hugging Face and project JSONL sources.
  Rationale: Milestone A1 needs a real, deterministic first-use proof that does
  not make the catalog/CLI test suite depend on network availability.
  Date/Author: 2026-07-23 / agent

- Decision: Discover third-party environment factories through the
  `posttrain.environment_factories` Python entry-point group, layered after
  built-ins and before explicit project/runtime overrides.
  Rationale: Published environment packages remain independently installable;
  the catalog resolves their registered factory without importing lab.
  Date/Author: 2026-07-23 / agent

- Decision: Build `GRPORequest` and `OnPolicyDistillationRequest` directly via
  public `posttrain.train` helpers. Keep the old lab request names as aliases
  during migration, not as parallel dataclasses.
  Rationale: One request contract and one existing Verifiers bridge.
  Date/Author: 2026-07-23 / agent

- Decision: Keep project-specific materialization out of `posttrain.work`.
  The work runner exposes a provider-neutral `SeatResolver` hook; the default
  `posttrain.jobs` runtime uses it to turn dataset plans into canonical data
  sources and to preflight environment packages before definition type checks.
  Rationale: Work remains contracts/runner-only while standard jobs own
  cross-capability composition.
  Date/Author: 2026-07-23 / agent

- Decision: A project entry returns a complete `JobRuntime`; the CLI verifies
  that every standard id is still present with the standard operation code,
  kind, and seats. The deprecated `--host` path remains less strict for the
  compatibility window.
  Rationale: Entries may add project definitions but cannot silently change
  shipped semantics.
  Date/Author: 2026-07-23 / agent

- Decision: Starter projects depend on the installed Posttrain release with
  template-specific extras (`trackio,trl` for SFT and
  `trackio,trl,verifiers` for GRPO). When init itself is running from this
  workspace, generated uv source overrides point at the local package members;
  released wheels omit those development-only paths.
  Rationale: The same one-command bootstrap must work during repository
  qualification and after packages are published, without copying lab code.
  Date/Author: 2026-07-23 / agent

- Decision: The primary CLI imports Observatory's public settings/server seam
  lazily and serves it in-process. The project manifest selects Trackio or
  W&B; provider connection details remain environment configuration.
  Rationale: `posttrain observatory up` is the happy-path product command while
  Observatory remains an optional application package and read-only boundary.
  Date/Author: 2026-07-23 / agent

- Decision: Generate direct immutable URL requirements by reading installed
  distribution metadata for the selected template extras, rather than copying
  pin constants into the CLI.
  Rationale: uv gets the project-root declarations it requires while the
  package owners remain authoritative for Trackio, TRL, Verifiers, and
  environment revisions.
  Date/Author: 2026-07-23 / agent

## Outcomes & Retrospective

- (2026-07-23) DX brief locked; plan consolidated. Implementation not started.
  Success = external projects run SFT and an environment-backed job without
  `posttrain_lab`, with init-install and Observatory up.
- (2026-07-23T17:27Z) Milestone A complete. The frozen baseline now authorizes
  standard framework jobs, project runtime vocabulary, first-use
  materialization, init-install, and primary-CLI Observatory bring-up.
- (2026-07-23T17:32Z) Milestone A1 complete. `posttrain-catalog` 0.2.0 ships
  `datasets/posttrain-sft-smoke@1`; empty overlays list shared model, dataset,
  and pinned environment ids; `posttrain dataset validate` writes normalized
  JSONL plus a digest manifest under `.posttrain/state/` and reuses it on the
  second call. Focused catalog/data/CLI tests passed (34), pyright passed,
  import contracts passed, and `git diff --check` was clean.
- (2026-07-23T17:39Z) Milestone B0 complete. Dataset plans support fixture,
  Hugging Face, JSONL, and Parquet sources through existing adapters and return
  canonical trainer-neutral sources. Installed environment factories are
  discoverable, environment preflight validates native config/importability,
  and public GRPO/distillation builders construct the existing bridge without
  `posttrain_lab`. The exact B0 test gate passed (29), lab migration tests
  passed (10), focused pyright and import contracts passed, and
  `git diff --check` was clean.
- (2026-07-23T17:47Z) Milestone B complete. `posttrain-jobs` now owns nine
  stable definition ids and the default tracked/local runtime, rejects
  standard-id shadowing, and resolves catalog data/environment seats without
  lab imports. Lab re-exports the standard technique definitions and retains
  only qualification-specific evaluation/transform wrappers. Jobs tests
  passed (3), the full lab suite passed (55, 2 optional-dependency skips),
  focused pyright and import contracts passed, and `git diff --check` was
  clean.
- (2026-07-23T17:49Z) Milestone C complete. Project manifests now carry
  `tracking` (default Trackio) and optional `entry`; the primary CLI constructs
  the standard runtime by default and can run through a discovered entry with
  no `--host`. Legacy Host symbols/flag remain aliases. CLI/catalog tests
  passed (16), including no-host execution and standard SFT preflight; focused
  pyright, import contracts, and `git diff --check` passed.
- (2026-07-23T17:54Z) Milestone D complete. `posttrain init --template sft`
  and `--template grpo` now write installable Python projects, visible settings
  overlays, standard work packages, state ignores, and selected release
  extras, then create `.venv` with uv. Fresh SFT and GRPO projects both
  installed; project-local doctor and `import posttrain.jobs` passed; SFT
  dataset materialization and both standard-job preflights passed. Focused
  CLI/catalog tests passed (19), ruff/pyright/import contracts passed, and
  `git diff --check` was clean.
- (2026-07-23T17:58Z) Milestone E complete. The primary CLI now derives
  Observatory source settings from the discovered project's tracking
  selection, prints the listening URL, and calls the application's public
  in-process server API. A live Trackio-backed server passed liveness and
  readiness; after a canonical tracked SFT proof run, `/api/v1/runs` returned
  that project run. Focused CLI/settings tests passed (24), pyright and import
  contracts passed, and the live process shut down cleanly.
- (2026-07-23T18:05Z) Milestone F complete. The isolated consumer suite builds
  and installs wheels outside the workspace, then proves generated SFT
  materialization/standard-job preflight and generated GRPO
  standard-job/Verifiers preflight. The environment proof constructs a real
  one-task GSM8K bridge, not an import-only fake. Both starters exclude
  `posttrain_lab`; full GPU updates remain documented release gates. Consumer
  tests passed (2), focused ruff/pyright/import contracts passed, and the root
  quickstart, DX golden path, and new Lab README now match the shipped CLI.
- (2026-07-23T18:06Z) Final acceptance complete. `uv sync --all-packages
  --locked --python 3.12`, repository-wide ruff and pyright, all eight import
  contracts, `git diff --check`, and the catalog/lock digest check passed. The
  full package suite passed 273 tests with 15 expected optional-dependency
  skips; the isolated wheel consumer suite separately passed 2 tests.

## Context and Orientation

Repository root: `/home/hammad/projects/rl` (Python 3.12 `uv` workspace).

Terms:

- **Standard job** — versioned id + seat→request function that calls package
  ops and applies data/Verifiers bridges. Does not embed project LR/data/GPU.
- **Global catalog** — published `posttrain-catalog` (and optional hub packages),
  auto-composed.
- **Project overlay** — `.posttrain/catalog/`.
- **Materialize** — first local fetch/install check for a bound global/project
  pointer (state cache / venv).
- **Project entry** — optional escape-hatch configure hook in `project.toml`.
- **Lab** — qualification reference project; not a consumer dependency.

Authoring (normative):

    # Catalog format kinds = adapter literals
    dataset:
      datasets/support-sft@1:
        revision: "1"
        source:
          kind: huggingface
          repo: org/support-conversations
          revision: <immutable>
          split: train
        format:
          kind: messages   # or auto | prompt-completion | alpaca | sharegpt

    data/support_sft/train.jsonl          # optional local source
    # source.kind: jsonl, path: data/support_sft/train.jsonl

    environment:
      support-tool-use-grpo:
        revision: "1"
        source: { package: my-env-v1, revision: <immutable> }
        factory: my-env-training
        sampling: { max_tokens: 2048, temperature: 1.0 }
        num_tasks: 8

Work package binds ids to `train/trl-sft@1` or `train/trl-grpo@1`. Full layout
and discovery rules: `docs/developer-experience.md`.

Key paths:

- DX brief: `docs/developer-experience.md`
- Data adapters: `packages/data/src/posttrain/data/adapters/`
- GRPO bridge: `packages/train/src/posttrain/train/integrations/verifiers.py`
- Catalog decode: `packages/catalog/src/posttrain/catalog/__init__.py`
- Work runner: `packages/work/src/posttrain/work/runner.py`
- Lab definitions to extract:
  `apps/lab/src/posttrain_lab/work_packages/definitions.py`
- CLI: `apps/cli/src/posttrain_cli/cli.py`
- Train API: `packages/train/src/posttrain/train/api.py`
- Consumer proof: `tests/consumer/`

Boundaries: train/eval/serve do not import one another; `posttrain.jobs` may.
No reusable package imports lab. Run `uv run lint-imports` after boundary edits.

Baseline amendment required before code that renames normative host seams or
claims standard jobs / declarative dataset decode as product contract.

## Plan of Work

### Milestone A — Baseline amendment

Amend `docs/post-training/README.md`, `04-framework.md`, and `05-apis.md`:

- Project entry / job runtime vocabulary; lab = qualification project.
- `posttrain.jobs` owns standard definitions + default runtime construction.
- `posttrain.work` remains contracts/runner only.
- Global catalog vs project overlay; materialize-on-first-use.
- Standard jobs wire `posttrain.data` + Verifiers bridges (existing packages).
- Dataset catalog source kinds and format literals aligned with adapters.
- `posttrain init` installs; `posttrain observatory up` on primary CLI.
- Document `JobRuntime`, `ProjectExecutionRequest`, `ProjectEntry` (Host aliases).

Do not change job-kind strings or selection type names without an explicit note.

Acceptance: docs-only; `git diff --check` clean on those files.

### Milestone A1 — Global catalog hub slice

Expand `posttrain-catalog` base so an empty project overlay still lists shared
models, at least one declarative HF (or fixture) dataset id, and at least one
environment binding with a documented package pin. Document how owners publish
new global entries (bump catalog package version).

Acceptance: CLI/consumer test with empty overlay lists global ids; binding a
global dataset id materializes on first validate and is idempotent on second.

### Milestone B0 — Wire existing data adapters and env bridges

Do not reimplement adapters/bridges. Connect catalog + jobs to:

- `supervised_from_huggingface` / preference helpers / JSONL and related
  adapters in `packages/data`
- `VerifiersEnvironmentRolloutBridge` in `posttrain.train.integrations`
- Existing catalog env factory maps (`evaluation_catalog_decoders`,
  `GENERAL_ENVIRONMENT_FACTORIES`, AutomationBench training factory)

Implement catalog `dataset` decode → load plan; `posttrain dataset validate`;
materialize into `.posttrain/state/`; expand env factory registry for
published/Prime-installed packages; ensure GRPO/distill/eval can build bridges
from `EnvironmentBinding` using **public** request types (migrate lab private
request wrappers).

Acceptance: catalog dataset + SFT definition path without project
`datasets.py`; catalog environment + GRPO/distill/eval preflight without
`posttrain_lab` bridge imports.

### Milestone B — `posttrain.jobs`

Add workspace package `packages/jobs` (`posttrain-jobs` / `posttrain.jobs`)
depending on work, common, train, eval, serve, data as needed.

Move technique-stable definitions from lab into `posttrain.jobs` for
**SFT, DPO, GRPO, distill, serve benchmark/smoke, eval, model transform**.
Preserve definition ids (`train/trl-sft@1`, `train/trl-dpo@1`,
`train/trl-grpo@1`, `train/trl-distill@1`, `serve/vllm-benchmark@1`,
`serve/vllm-smoke@1`, `model/llm-compressor@2`, `eval/verifiers-general@1`,
…).

Add `standard_definitions()` and `build_job_runtime(...)` (tracking + scratch +
`execute_run_tracked`; reject shadowing standard ids). Lab re-exports or thin
wraps; scenario-only extras stay in lab entry.

Acceptance: `uv run pytest packages/jobs/tests`; lab tests still pass;
`uv run lint-imports` passes.

### Milestone C — Defaults and CLI without `--host`

Extend `project.toml` with tracking backend (default trackio) and optional
`entry`. CLI builds JobRuntime from `posttrain.jobs` by default; `--host` /
`--entry` as overrides during compatibility window. Rename Host types with
aliases.

Escape-hatch `entry` may register extra definitions or unshipped factories; it
must not redefine `train/trl-sft@1` with different semantics.

Acceptance: work-package run without `--host`; one entry escape-hatch test.

### Milestone D — Init templates with install

`posttrain init PATH --template sft` and `--template grpo` (or
`env-train`): write installable `pyproject.toml`, `.posttrain/`, visible
settings overlays, declarative dataset and/or environment bindings (prefer
global ids where A1 published them), work packages bound to standard
definition ids, then create `.venv` and install. No `posttrain sync`.

Acceptance: single init command → doctor + `import posttrain.jobs` without
lab; no mandatory `datasets.py`.

### Milestone E — `posttrain observatory up`

Primary CLI starts Observatory for the discovered project’s tracking config,
prints URL. Prefer library import over shelling out; fail with install hint if
extra missing.

Acceptance: after a tracked consumer run, Observatory serves runs for that
project.

### Milestone F — External proof and docs

Consumer tests from installed wheels:

1. SFT starter: init → dataset validate/materialize → job run (CPU-safe or
   documented GPU gate) → optional Observatory.
2. Environment-backed starter: catalog env + `train/trl-grpo@1` or distill/eval
   preflight/run path without `posttrain_lab`.

Update quickstart and lab README. Keep audit superseded banner.

Acceptance: CI consumer tests pass; grep starters for no `posttrain_lab`.

## Concrete Steps

From repository root:

    uv sync --all-packages --locked --python 3.12

After A:

    git diff --check docs/post-training docs/developer-experience.md docs/plan/project-developer-experience.md

After B0:

    uv run pytest packages/data/tests packages/catalog/tests packages/train/tests/test_verifiers_grpo_bridge.py -q

After B:

    uv sync --all-packages --python 3.12
    uv run lint-imports
    uv run pytest packages/jobs/tests packages/work/tests apps/lab/tests/test_work_packages.py -q

After C:

    uv run pytest apps/cli/tests packages/catalog/tests -q

After D:

    rm -rf /tmp/posttrain-sft-demo
    uv run --package posttrain posttrain init /tmp/posttrain-sft-demo \
      --template sft --project-id sft-demo
    cd /tmp/posttrain-sft-demo
    .venv/bin/posttrain doctor

After E/F:

    uv run pytest tests/consumer -q
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Expected shape:

    $ posttrain init support-agent --template sft
    initialized … Installing dependencies… Environment ready.

    $ posttrain dataset validate datasets/support-sft@1
    # materializes if needed

    $ posttrain work-package run .posttrain/work_packages/sft.yaml --job train
    run_id=… status=succeeded

    $ posttrain observatory up
    Observatory listening at http://127.0.0.1:8787

## Validation and Acceptance

Done when:

1. `posttrain init --template sft` (and env/GRPO template) installs and needs no
   `posttrain_lab`.
2. Jobs run without `--host` via `posttrain.jobs` standard definitions.
3. Catalog datasets resolve through existing `posttrain.data` adapters;
   environments through existing Verifiers bridge/eval factories.
4. Global ids listable with empty overlay; first use materializes locally.
5. `posttrain observatory up` works for the project tracking config.
6. Consumer wheel proofs cover SFT and an environment-backed path.
7. Baseline docs + DX brief agree; `lint-imports` / package tests / consumer
   tests pass.

GPU full qualification may remain a separate release gate; it must not block
merging the wheel/CPU (or documented smoke) proofs of the wired DX path.

## Idempotence and Recovery

- Init refuses non-empty targets unless forced; install failure fails init.
- Materialize and sync-via-project-package-manager are re-runnable.
- Host type aliases until callers migrated.
- Do not revert unrelated dirty worktree changes.

## Artifacts and Notes

    docs/developer-experience.md          # locked DX authority
    packages/jobs/                        # new
    packages/data/src/posttrain/data/adapters/
    packages/train/src/posttrain/train/integrations/verifiers.py
    packages/catalog/src/posttrain/catalog/
    packages/work/src/posttrain/work/runner.py
    apps/cli/src/posttrain_cli/cli.py
    apps/lab/src/posttrain_lab/work_packages/definitions.py
    tests/consumer/
    docs/post-training/README.md
    docs/post-training/04-framework.md
    docs/post-training/05-apis.md

## Interfaces and Dependencies

`posttrain.jobs` exposes at least:

    def sft_definition(...) -> JobDefinition: ...
    def dpo_definition(...) -> JobDefinition: ...
    def grpo_definition(...) -> JobDefinition: ...
    def distillation_definition(...) -> JobDefinition: ...
    def serve_benchmark_definition(...) -> JobDefinition: ...
    def serve_smoke_definition(...) -> JobDefinition: ...
    def general_evaluation_definition(...) -> JobDefinition: ...
    def model_transform_definition(...) -> JobDefinition: ...

    def standard_definitions() -> dict[str, JobDefinition]: ...

    def build_job_runtime(
        request: ProjectExecutionRequest,
        *,
        tracking: str | None = None,
        extra_definitions: Mapping[str, JobDefinition] | None = None,
    ) -> JobRuntime: ...

Catalog dataset format field uses adapter literals from
`posttrain.data.adapters` (`SFTFormat`, `PreferenceFormat`).

Work renames (aliases): `JobRuntime`, `ProjectExecutionRequest`, `ProjectEntry`.

CLI:

    posttrain init PATH --template sft|grpo
    posttrain dataset validate ID
    posttrain work-package run PATH --job ID    # no required --host
    posttrain observatory up [--port N]

Do not add `posttrain sync`. Reuse Trackio/W&B adapters and Observatory
`create_service` / `create_http_app` / `ObservatorySettings`.

## Revision history

- 2026-07-23: Initial plan from locked DX brief.
- 2026-07-23: Authoring model, catalog layout, init-install, Observatory,
  global hub + materialize, HF/Prime/custom env discovery.
- 2026-07-23: No phase-2 — wire existing data adapters and Verifiers bridges;
  standard jobs include SFT/DPO/GRPO/distill/eval/serve/transform.
- 2026-07-23: Consolidated rewrite for coherence; format literals aligned with
  `posttrain.data`; dual consumer proofs (SFT + environment-backed).
- 2026-07-23: Added agent runbook for goal-setting agents
  (`docs/plan/project-developer-experience-agent-runbook.md`).
