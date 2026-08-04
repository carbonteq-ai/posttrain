# Add a SkyRL-BIRD SQL GRPO experiment to Posttrain Lab

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as
work proceeds. Maintain it according to `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this change, a Posttrain Lab user can evaluate and train
`google/gemma-4-12B-it` on the verified ReViSQL subset of BIRD by using the
SkyRL-SQL multi-turn text protocol. The model may reason, execute read-only
exploratory SQL against a local SQLite database, inspect bounded observations,
and submit one final SQL query. Posttrain's existing Verifiers bridge, TRL GRPO,
vLLM, LoRA, work-package, and Trackio primitives own execution and evidence.

The observable workflow is base evaluation, a one-step full-shape canary, a
516-step full training pass, and held-out evaluation of an immutable adapter
artifact. There is no pilot configuration. Implementation is restricted to
`apps/lab`, except for the generic Gemma renderer contract that now lives beside
the existing Qwen and LFM contracts in `packages/common`. That narrow reusable
edit was explicitly approved after packaged execution exposed the missing
registration. A further reusable-package or external-fork edit must still be
justified to the user before it is made.

This composition does not change the frozen product baseline. GRPO consumes an
environment selection rather than a dataset seat, and held-out evaluation is a
separate qualification job, matching the canonical baseline.

## Progress

- [x] (2026-08-04 00:00Z) Read the canonical product documents, plan template,
  current Lab GRPO jobs, catalogs, work packages, gates, and native Verifiers v1
  environment patterns.
- [x] (2026-08-04 00:00Z) Verify immutable model, ReViSQL, SkyRL-SQL, Verifiers,
  and database-mirror inputs and resolve the authoritative 2,064/398 row counts.
- [x] (2026-08-04 00:00Z) Implement the native `skyrl-bird-sql-v1`
  environment and offline tests.
- [x] (2026-08-04 00:00Z) Add Lab-local Gemma renderer registration and
  project catalog selections.
- [x] (2026-08-04 00:00Z) Add base evaluation, full-shape canary, full run, and adapter evaluation
  scenarios and work packages.
- [x] (2026-08-04 00:00Z) Run focused tests, workspace sync, Ruff, Pyright,
  import-linter, Lab tests, gate validation, and `git diff --check`.
- [x] (2026-08-04 13:30Z) Replace the three local `project-path` environment
  sources with the pushed immutable Git revision after v0.3.0 packing exposed
  inconsistent nested-source digest roots.
- [x] (2026-08-04 13:57Z) Pack, smoke-test, attest, and publish the canary
  actual-job image from the complete v0.3.0 framework wheel set on `pypi.lan`.
  Package identity is
  `33841e94d792408ad1b83f86707159bdb234cff0c50a0943b9429771d7d7d0d1`;
  image identity is
  `registry.lan/carbonteq/posttrain-job@sha256:23b9c118551e8a17e8471e46257e918ffdf7577aa4f85b0b2880d63ec1e8efaf`.
- [x] (2026-08-04 14:30Z) Move `gemma4-tools@1` into the static common renderer
  registry after the first submitted worker failed while decoding the catalog
  before importing the Lab project entry.
- [x] (2026-08-04 14:45Z) Verify the common and Lab suites (101 passed, 5
  skipped), all eight import contracts, Pyright, plain `posttrain doctor`, and
  `git diff --check` after the static registration change.
- [ ] Complete the full repository pytest run (blocked by an unchanged
  Observatory discovery test that hangs when run alone in this environment).
- [ ] Run the network and GPU release gates on a suitable pod and record evidence.

## Surprises & Discoveries

- Observation: the pinned ReViSQL README reports 2,088 train and 374 validation
  rows, but the immutable JSON files contain 2,064 train and 398 validation rows.
  Evidence: the pinned files hash to
  `a3829a81a02299e3a0155afa1321e1a4cca58ba90645bb95a59e6b8a1de8b3ec`
  and `a2781fab072244928d4d4e452b626d7ee2133050353b0bba6c631159d99ed39e`.
- Observation: Transformers 5.14.1 already maps `gemma4_unified` through
  `AutoModelForCausalLM` to `Gemma4UnifiedForConditionalGeneration`; no trainer
  loader change is required.
- Observation: `ToolCallProtocol.id` in `posttrain.common` is restricted to the
  Qwen and LFM identifiers. This experiment has no tool calls: SkyRL SQL uses
  visible text tags. The Lab Gemma renderer therefore omits `tool_calls` rather
  than broadening a shared contract for an unused feature.
- Observation: v0.3.0 cannot pack a nested `project-path` environment because
  planning hashes the selected directory with its project-relative prefix,
  while environment-wheel verification hashes the package subtree without
  that prefix. For `environments/skyrl_bird_sql_v1`, the observed digests were
  `bc168132bf591bea2b9bce4e108b5345dfbd9a63248b73343a4996b059da7a98`
  and `7a3200f864b28de09a5f38cfbd5553e98b1091f24023499f033be2b98a0f3016`.
  The Lab overlay now selects the same package from the pushed full commit SHA;
  a generic framework correction is deliberately deferred.
- Observation: installed-distribution packing succeeds against the protected
  project `UV_INDEX_URL=https://pypi.lan/carbonteq/stable/+simple/`. The final
  image installed all 20 requested v0.3.0 framework distributions plus current
  `posttrain-lab`, passed `posttrain-runtime --help`, parsed its package
  manifest, and reported only the intentionally deferred
  `skyrl-bird-sql-train-canary` qualification.
- Observation: the first submitted worker failed before creating a training
  run because `posttrain-runtime` opens the project catalog before importing
  `posttrain_lab.entry`; the former Lab-only registration therefore did not
  exist in that process. The exception was `unknown renderer contract:
  'gemma4-tools@1'`.
- Observation: the full repository suite reaches unchanged Observatory
  discovery tests and then stops making progress. The test
  `test_success_removes_missing_projects_but_failure_retains_last_snapshot`
  exits with timeout status 124 when run alone for 30 seconds; when it is
  deselected, `test_start_refreshes_once_and_stop_cancels_periodic_wait`
  reports failure and hangs during teardown. All new Lab tests pass before this
  point.
- Observation: after importing the pinned Verifiers v1 API, SQLite work run in
  asyncio's default executor completes but its thread-safe completion callback
  does not wake an otherwise idle selector loop in either CPython 3.12.13 or
  3.13.14. A live server's other I/O can mask the problem, while isolated tests
  and loop shutdown hang. The environment now uses a private daemon worker and
  an event-loop heartbeat, preserving non-blocking SQL execution without
  depending on that cross-thread wakeup path.

## Decision Log

- Decision: treat the pinned JSON contents, 2,064 train and 398 validation
  examples, as authoritative and document the stale README counts.
  Rationale: immutable bytes are executable provenance.
  Date/Author: 2026-08-04 / Codex and user.
- Decision: reproduce ReViSQL terminal result grading, but use Posttrain GRPO
  rather than ReViSQL's CISPO configuration.
  Rationale: the experiment evaluates the requested Posttrain GRPO composition.
  Date/Author: 2026-08-04 / Codex and user.
- Decision: use a one-step canary with the full run's four prompts by sixteen
  generations, effective batch 64 geometry; do not add a pilot.
  Rationale: one canary should exercise the memory and rollout shape that matters.
  Date/Author: 2026-08-04 / Codex and user.
- Decision: register Gemma's tokenizer-backed renderer with reasoning disabled
  and no tool-call protocol, then promote that unchanged contract to the static
  common renderer registry when packaged execution demonstrated that a
  project-local registration cannot precede worker catalog decoding.
  Rationale: literal `<think>`, `<sql>`, `<observation>`, and `<solution>` tags are
  ordinary visible text and the environment never emits structured tool calls.
  Static registration matches the existing Qwen/LFM mechanism and adds no new
  runtime discovery or import behavior.
  Date/Author: 2026-08-04 / Codex and user.
- Decision: use Trackio and require an explicit immutable `vN` adapter version
  for descendant evaluation.
  Rationale: work-package outputs are artifact values, not implicit config links.
  Date/Author: 2026-08-04 / Codex and user.
- Decision: select the environment package from
  `carbonteq-ai/posttrain@0c04712edd013576bb245a17cd8619040750f4dc` using the
  existing immutable Git-source primitive.
  Rationale: the implementation is now pushed, and this Lab-only selection
  avoids v0.3.0's inconsistent nested `project-path` digest roots without
  changing reusable framework packages during the GPU qualification pass.
  Date/Author: 2026-08-04 / Codex and user.
- Decision: execute SQLite on a private bounded worker thread and observe its
  completion from the rollout loop instead of using `asyncio.to_thread`.
  Rationale: this avoids the verified idle-selector integration deadlock while
  retaining the SQL progress-handler timeout and keeping database work off the
  rollout event loop.
  Date/Author: 2026-08-04 / Codex.

## Outcomes & Retrospective

The CPU implementation is complete within the approved boundary. The native
environment has a locked dependency graph and its synthetic suite passes; the
Lab catalog, three static work packages, dynamic immutable-adapter evaluation,
and 29-entry gate inventory resolve successfully. Workspace sync, Ruff,
Pyright, all eight import contracts, and the 84-test Lab suite pass. Three Lab
tests skip as designed because Transformers/GPU dependencies are not installed
in the CPU environment. The standalone environment reports one expected network
skip and 27 offline tests passing. The 20 GB asset integration and Gemma GPU
canary remain release gates. The otherwise unrelated full-suite Observatory
hang is recorded above rather than attributed to this experiment.

The immutable canary actual-job image is now published and has passed its
build-time runtime smoke and deferred-qualification checks. This establishes
packaging and publication only; GPU execution, prepared BIRD assets, model
loading, rollout generation, the optimizer update, and retained run evidence
remain open release gates.

The first submitted execution of that image failed before training because the
v0.3.0 common wheel did not contain the formerly Lab-local renderer. The branch
now places the same contract in common's static registry, and plain
`posttrain doctor` resolves all 102 project selections without the Lab wrapper.
A replacement image containing the branch version of `posttrain-common` must be
packed and submitted; the earlier image digest remains known-bad for execution.

## Context and Orientation

`apps/lab` is the reference composition host. Its `.posttrain/catalog` directory
contains project overlays, `.posttrain/work_packages` contains declarative work
packages, `src/posttrain_lab/cli.py` composes named scenarios, and
`src/posttrain_lab/qualification/gates.toml` inventories every work package
exactly once. Standard `train/trl-grpo@1` consumes model, environment, settings,
training, and rollout-inference seats.

The new standalone environment package lives at
`apps/lab/environments/skyrl_bird_sql_v1`. It implements native Verifiers v1
task classes and does not import Posttrain. Its cache is selected by
`POSTTRAIN_SKYRL_BIRD_CACHE`; tracked configuration must never contain pod-local
absolute paths.

The environment protocol is textual. Assistant turns contain non-empty
`<think>` followed by either one exploratory `<sql>` or one terminal
`<solution>`. The user simulator executes exploratory SQL and replies with a
bounded `<observation>`. The final solution is executed only for scoring.

## Plan of Work

First, build the environment package with pinned metadata download, safe and
idempotent 20 GB database-archive preparation, deterministic task loading,
schema introspection, strict protocol parsing, read-only SQLite execution, and
the pinned ReViSQL grading methods. Add synthetic-database tests that run without
network or credentials, plus a marked integration test for exact source counts,
hashes, database coverage, disjoint IDs, and gold-query execution.

Second, add a Lab-local `gemma4-tools@1` renderer using the tokenizer chat
template and `enable_thinking=False`. Register it before opening the Lab catalog.
Add the pinned Gemma model, 96 GB target, train/eval environments and evaluation
plan, training settings, LoRA training binding, and vLLM rollout/evaluation
bindings to a dedicated overlay.

Third, add static work packages for base evaluation, the one-step canary, and
the 516-step full run, listing each once in `gates.toml`. Add CLI scenarios for
those three and a dynamic adapter evaluation scenario accepting an explicit
Trackio artifact version. Reuse standard job definitions and existing managed
evaluation/GRPO operations.

Finally, run focused tests and the full validation ladder. GPU execution must
prove BF16 text-only Gemma loading, LoRA isolation below `language_model`, a
finite update, full-shape reward variance, no truncation/OOM, artifact
persistence, and adapter reload before the full run is admitted.

## Concrete Steps

From the repository root, install and validate with:

    uv sync --all-packages --locked --python 3.13
    cd apps/lab/environments/skyrl_bird_sql_v1
    uv run --python 3.12 pytest tests
    cd ../../../..
    uv run pytest apps/lab/tests
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

On a GPU pod, configure:

    export HF_TOKEN=...
    export POSTTRAIN_SKYRL_BIRD_CACHE=/workspace/cache/skyrl-bird-sql
    export POSTTRAIN_SCRATCH_ROOT=/workspace/posttrain-scratch
    export TRACKIO_DIR=/workspace/trackio

Then prepare and validate the immutable assets, run base evaluation, run the
canary, and only after its acceptance run the full scenario. Adapter evaluation
must name the produced immutable `vN` Trackio artifact version.

## Validation and Acceptance

Offline tests cover every parser state, malformed/mixed tags, correction and
termination, SQL read-only enforcement and timeout, deterministic schema and
task loading, observation truncation, all ReViSQL graders, catalog decoding,
work-package seats, gate uniqueness, and immutable adapter selection.

The canary processes four deterministic grading-family tasks with sixteen
rollouts each. It must show at least one nonconstant reward group, finite loss
and gradients, no OOM or truncation, language-model-only LoRA parameters, peak
VRAM and duration evidence, and a reloadable Trackio adapter.

The full run performs 516 optimizer steps, consuming all 2,064 train prompts
once in groups of four. Base and adapter evaluations each cover all 398 held-out
tasks and report reward, protocol validity, execution success, correctness,
turn counts, and grading-method slices with immutable lineage.

## Idempotence and Recovery

Asset preparation uses a lock, temporary paths, SHA-256 verification, safe ZIP
extraction, an atomic publish, and a manifest. It may be retried after failure
and never redownloads a valid archive. Training resumes only from a verified
checkpoint with matching source, lock, model, environment, and settings
lineage. Hash mismatch and unsafe archive entries are fatal.

## Artifacts and Notes

Pinned inputs are Gemma revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, ReViSQL commit
`9fac371aa22019e9912dcbd6572e8fe8194d352a`, SkyRL-SQL reference commit
`cf220ee86500e94b5415d8b226557ce5d950f1db`, Verifiers commit
`284a868d6a9022109b749710672a0460e8a996d4`, and BIRD mirror revision
`7877a1bfee6b3794f5026b1f00fcc4dd43e529be`. The database archive is
20,683,638,742 bytes with SHA-256
`54424b2004cea43f1fd89605b3df41836df3a46bc68ffd5444c6549c112172f3`.

## Interfaces and Dependencies

The environment exposes `SkyRLBirdSQLTaskset` as its native Verifiers entry.
The asset module exposes `prepare` and `validate` CLI subcommands. Catalog-facing
configuration uses existing `EnvironmentBinding`, `GRPOSettings`,
`TrainingBinding`, `InferenceBinding`, `EvaluationPlan`, and `ModelVariant`
contracts. `packages/common/src/posttrain/common/variants/gemma4.py` supplies the
generic renderer value through the existing static `RENDERER_CONTRACTS` map; no
new reusable framework API or discovery behavior is introduced.

Revision note (2026-08-04): this initial living plan incorporates the decision
to replace a pilot with a full-shape one-step canary and records the Lab-only
renderer deviation required to avoid an unnecessary common-package edit.

Revision note (2026-08-04): packaged worker execution proved that a Lab-only
renderer cannot be registered before runtime catalog decoding. The user
approved moving the unchanged generic contract into common beside Qwen/LFM,
while retaining the Lab registration function as a compatibility shim.
