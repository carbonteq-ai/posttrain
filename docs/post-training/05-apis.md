# 05 · APIs


> **Frozen baseline (2026-07-21).** Product authority: [post-training README](./README.md). Prefer implementation-plan / code changes over redesigning this doc unless explicitly unfrozen.

Target **public surface** of the framework: how developers name selections,
resolve them from a catalog, invoke job kinds, and read evidence.

Derived from [01](./01-workflow.md)–[04](./04-framework.md). Existing code is
**not** the contract. Implementation converges here.

## Naming stack (normative)

One idea → one noun. Do not invent parallel vocabularies in the API.

| Noun | Meaning | Not |
| --- | --- | --- |
| **Selection** | Concrete value of one primitive family (02) | “Profile”, untyped config blob |
| **CatalogRef** | `{family, id}` pointer into the composed catalog | A second type system per family |
| **Seat** | Named slot a job kind requires (`starting_model`, …) | Free-form kwargs |
| **Job kind** | Stable semantic id (`train.sft`, `serve.benchmark`) | The Python function alone |
| **Job definition** | Versioned implementation of exactly one job kind | Adapter class name as public API |
| **Operation** | Public function that realizes a job kind (`train.sft(…)`) | “Job API” as a separate product layer |
| **Request / Result** | Inputs and evidence for one operation call | Control-plane decisions |
| **Run** | One observed execution of one job definition | Work package |
| **Work package** | Stage-tied bindings + which jobs to run | Python package / catalog package |
| **RunContext** | Operation-facing identity, observer, workspace | Backend handle or Trackio import |
| **JobRuntime** | Resolved definitions, catalog, tracking, and scratch used to execute jobs | A trainer or provider SDK |
| **ProjectExecutionRequest** | Project-scoped inputs used to construct a job runtime | One operation request |
| **ProjectEntry** | Optional hook that adds custom definitions or factories | Required host factory |

```text
CatalogRef  --resolve-->  Selection  --bind seats-->  Work package
                                                        │
                                              for each enabled job
                                                        ▼
                              RunContext + Request  -->  Operation  -->  Result
                                                        │
                                                        ▼
                                                     Observer / Observatory
```

### Selection type names (align with 02)

| Primitive (02) | Public type | Catalog `family` |
| --- | --- | --- |
| Model variant | `ModelVariant` | `model` |
| Dataset selection | `DatasetSelection` | `dataset` |
| Environment binding | `EnvironmentBinding` | `environment` |
| Inference binding | `InferenceBinding` | `inference` |
| Training selection | `SFTSettings` \| `DPOSettings` \| `GRPOSettings` (kind-specific) | `training` |
| Evaluation plan | `EvaluationPlan` | `evaluation` |
| Workload | `Workload` | `workload` |
| Execution target | `ExecutionTarget` | `target` |

Rules:

1. **Selections are values**, not “refs.” A `CatalogRef` resolves *to* a selection.
2. **Training is kind-specific.** There is no shared `TrainingSelection` mega-type
   that also carries `job_kind`. The job kind lives on the run / request; settings
   types live in `posttrain.train`.
3. **Artifact identity** for lineage is the selection’s stable id (plus digest /
   revision). Do not introduce a parallel `ModelRef` type for the same thing.
4. Prefer catalog ids in configs; inline selections are allowed for one-offs and
   must validate as the same types.

## Design stance

### Config-first, dual authoring

Primary developer artifact: **named bindings** (which catalog entries fill which
seats, which jobs run). Author as **YAML + schema** or **typed Python**; both
validate into the same models. Neither is a second platform language.

Jobs return evidence. They do **not** accept / revise / reject descendants.

### Small surface

- seats are explicit and typed
- every run stores the **resolved** selections (`source_layer` included)
- backends stay behind capability packages
- `train` / `eval` / `serve` do not import each other
- tracking readers and Observatory are read-only

### Internal framework

Reusable across Carbonteq projects. Prefer a first-party module over plugin
hooks. No third-party marketplace SDK.

## API layers

| Layer | Developer touches | Lives in |
| --- | --- | --- |
| Selections | Typed primitive values | `common` (+ settings types in owning package) |
| Catalog | `open` / `resolve` / `list` | `common` + catalog modules |
| Operations | `data.prepare`, `serve.benchmark`, `train.sft`, … | `data`, `serve`, `eval`, `train` |
| Composition | Recipe, work-package bindings | project + thin helpers |
| Standard jobs | Stable seat-to-operation definitions and default runtime | `jobs` |
| RunContext | Identity, observer, workspace | job runtime injects; protocol in `common` |
| Evidence | Query raw runs/artifacts; compute job-aware views | raw contracts in `tracking`; product queries and views in Observatory |

```mermaid
flowchart LR
    Ref["CatalogRef"] --> Cat["Catalog.resolve"]
    Cat --> Sel["Selection"]
    Sel --> WP["Work package bindings"]
    WP --> Op["Operation = job kind"]
    Op --> Ctx["RunContext"]
    Ctx --> Obs["Observer"]
    Op --> Res["Result"]
    Res --> Observatory["Observatory"]
```

## Primary project CLI

The `posttrain` distribution is the primary command for repository and
remote-server workflows. It coordinates the public APIs in this document; it
does not introduce new selection, work, run, artifact, or evidence types.

```text
posttrain init [PATH] --template sft|grpo [--project-id ID] [--no-install]
posttrain version
posttrain doctor
posttrain runtime images list
posttrain runtime images verify [--variant VARIANT]
posttrain runtime images build [--variant VARIANT] [--push]
posttrain runtime images mirror --registry PREFIX [--variant VARIANT]
posttrain project show
posttrain catalog list [--family FAMILY]
posttrain catalog show FAMILY ID
posttrain catalog validate
posttrain dataset validate DATASET_ID
posttrain work-package validate PATH
posttrain work-package run PATH --job JOB_ID
posttrain job plan WORK_PACKAGE --job JOB_ID
posttrain job pack WORK_PACKAGE --job JOB_ID
posttrain job run WORK_PACKAGE --job JOB_ID [--provider local|dstack] [--build-missing]
posttrain job diff WORK_PACKAGE --job JOB_ID [--from KEY] [--to KEY]
posttrain run list
posttrain run status RUN_ID
posttrain run wait RUN_ID
posttrain run logs RUN_ID
posttrain run cancel RUN_ID
posttrain run retry-submit RUN_ID
posttrain run reconcile RUN_ID
posttrain run cleanup RUN_ID
posttrain run show RUN_ID
posttrain observatory up [--port PORT]
```

Project and catalog commands load the same `ProjectLayout`, `CatalogRef`, and
composed `Catalog` values used by Python callers. Work-package commands validate
all seats and catalog references before opening a run or invoking an operation.
Commands return zero on success, one for expected project or contract failures,
and two for invalid command syntax. The primary CLI is built with Typer; that is
an implementation detail and does not change the public noun surface. Readable
terminal output is the default; `--json` provides deterministic automation
output.

`job` owns immutable planning, packing, and submission. Once a canonical
`run_id` exists, `run` owns provider lifecycle, admission state, retained
evidence reconciliation, and cleanup. This noun split supersedes the earlier
draft spelling of lifecycle commands under `posttrain job`.

An actual-job image is identified by content, not by a declared version. The
package key is a digest over resolved catalog selections, resolved
configuration, project configuration, project and framework source, the
dependency closure, every materialized dataset, and the job-kind image. Any
change to any of those produces a different package key and therefore a
different published image, so a job image is never overwritten and never needs
a version bump; two different configurations cannot share an identity. `job
diff` reports which of those inputs differ between two packages, because a
digest alone states that something changed without saying what, and the honest
answer is often narrower than a reader assumes.

`runtime` owns the framework-published image levels that exist before any
project: the universal base and the job-kind images. The framework publishes
these once per release and pins their digests in the distribution, so an
installed wheel already knows the exact image identity it requires. `list`
reports those pinned digests, `verify` compares them against a registry,
`build` reproduces them from the shipped definitions for a site that can reach
neither the public registry nor a mirror, and `mirror` copies already-published
digests into a site's own registry without rebuilding. Every one of these is a
consumer operation. Publishing the release itself — pushing base and kind
images to the framework's public registry and rewriting the pinned manifest —
is a framework-owner operation and is deliberately absent from this surface.
`runtime` never touches actual-job images; those remain owned by `job pack`.

Two registries are in play and they are not the same thing. The framework
publishes its base and job-kind images to one public registry per release; that
location is a property of the framework, is identical for every consumer, and
is recorded in the distribution rather than configured. A project publishes its
own actual-job images somewhere else entirely, usually a private registry, and
that location is per-site configuration. Conflating them makes a project's
private registry look like a framework release channel.

The `POSTTRAIN_REGISTRY` environment variable names the second one: the OCI
registry prefix a project pushes its own actual-job images to. A site that
cannot reach the public registry may also mirror the framework's base and kind
images into it; because image digests are content-addressed, mirroring
preserves image identity exactly, so a mirrored image satisfies the same digest
the distribution pins. `POSTTRAIN_REGISTRY` is a location, not a credential;
registry authentication stays in the environment's own Docker or OCI credential
store. When it is unset, `job pack` and `job run` fail with a project contract
error rather than defaulting to a registry. `doctor` reports its presence and
reachability, and fails when a reachable kind image does not carry the lock
digest the installed framework expects.

Initialization writes the project layout and an installable project package,
then creates the project environment and installs dependencies. There is no
separate `posttrain sync` command. Work-package execution builds a default
`JobRuntime` from framework standard definitions and project tracking config;
`--host` / `--entry` remain temporary compatibility overrides only.

`posttrain-lab` remains the reference-project command for concrete
qualification jobs. Its fixed scenario names are not the portable CLI contract.

## Selections

Field lists are the minimum contract; they may grow without breaking the seat
model. Semantics stay in [02 · Primitives](./02-primitives.md).

### `ModelVariant`

Exact loadable weight state. Weight quantization (AWQ, GPTQ, GGUF Q4, …) is a
**new variant**, not an inference setting. See
[02 · model variant](./02-primitives.md#model-variant).

| Field | Role |
| --- | --- |
| `id` / `revision` or `digest` | Stable identity |
| `artifact_uri` | Where weights live |
| `form` | foundation \| adapter \| merged \| weight-quantized \| … |
| `weight_precision` | bf16 \| fp16 \| int4 \| … (materialized) |
| `quantization` | Optional method/scheme metadata when `form` is weight-quantized |
| `parent` | Optional parent `ModelVariant` |
| `renderer_contract` | Chat/tool/reasoning contract id |
| `capabilities` | Context, modality claims |
| `provenance` | Source / producing run |

### `DatasetSelection`

| Field | Role |
| --- | --- |
| `id` / `digest` | Immutable snapshot when materialized |
| `kind` | supervised \| preference \| prompt-task \| … |
| `split` | train \| val \| heldout \| named slice |
| `schema_version` | Example contract |
| `provenance` | Upstream + transforms |
| `environment_compatibility` | Optional env revision for task-shaped data |

Packing, tokenization, and render max lengths belong on **training settings**,
not on the dataset.

### `EnvironmentBinding`

| Field | Role |
| --- | --- |
| `id` / `revision` | Env/taskset identity |
| `package` | Installable Verifiers (or compatible) package name |
| `repository` / `source_revision` / `subdirectory` | Secret-free Git URL, full commit SHA, and package root |
| `activation` | Declarative Verifiers config by default; optional real `module:callable` for custom packages |
| `split` / subset | Task selection for this binding |
| `parameters` | Task-meaningful timeouts/limits |
| `reward_components` | Declared raw signal names (meanings owned by env) |

### `InferenceBinding`

How a `ModelVariant` generates tokens or scores existing token sequences for a
purpose. Replaces the old “inference / serve profile” idea. Engine-level config
is a **field** (`engine`), not its own catalog family. See
[02 · inference binding](./02-primitives.md#inference-binding).

| Field | Role |
| --- | --- |
| `id` / `revision` | Binding identity when catalogued |
| `model` | `ModelVariant` (or catalog ref resolved to one) |
| `backend` | Engine kind + version (e.g. `vllm@…`) |
| `renderer` | Renderer contract |
| `engine` | Backend-owned runtime settings (KV cache, TP, speculative, mem util, …) |
| `sampling` | Defaults for generation or token scoring for this purpose |
| `target` | `ExecutionTarget` |
| `purpose` | screen \| eval \| rollout \| teacher-score \| smoke \| handoff |

`engine` schema is owned by the `serve` (or colocated train) adapter for that
`backend`. Changing engine or sampling → new binding revision; same model
variant. Changing on-disk weight quant → new `ModelVariant`, then a binding that
points at it.

Illustrative composition:

```text
ModelVariant          models/qwen-2b@awq-int4
InferenceBinding      inference/qwen-2b-awq-vllm-screen@1
  model ────────────► models/qwen-2b@awq-int4
  backend             vllm@…
  engine              { gpu_memory_utilization, kv_cache, … }
  sampling            { temperature, max_tokens, … }
  target              hardware/rtx3070ti-8gb
  purpose             [screen, eval]
```

### Training settings (`SFTSettings`, `DPOSettings`, `GRPOSettings`, `OnPolicyDistillationSettings`)

Owned by `posttrain.train`. **Algorithm + learning schedule only** — not vLLM
topology, not mandatory QLoRA.

Also first-class (cataloguable where useful):

| Type | Role |
| --- | --- |
| `ParameterUpdatePlan` | Discriminated: full \| LoRA \| QLoRA \| quantization-aware |
| `TrainingBinding` | Train backend, update plan, normalized train parallelism/runtime, namespaced backend options, train `ExecutionTarget` |

`TrainingBinding.runtime` uses backend-neutral names wherever two trainers share
the same concept (for example global batch size, node count, devices per node,
parameter offload, and optimizer offload). `TrainingBinding.backend_options` is
the explicit escape hatch for trainer-native settings. A backend adapter
translates normalized values into its native vocabulary; native overrides must
not replace selected model, data, environment, target, or artifact identities.
| `QuantizationPlan` | Recipe + calibration + formats; offline PTQ and/or QAT |

Reward **weights** live in algorithm settings; reward **meanings** live on the
environment. Rollout engine knobs live on `InferenceBinding.engine`.
`GRPOSettings.algorithm` selects `grpo` or `dapo`. The DAPO selection owns its
token-level aggregation, asymmetric clipping, bounded retained-group dynamic
sampling, truncation handling, and optional soft-overlong shaping. Run evidence
records these settings explicitly. Backend adapters reject unsupported
semantics rather than approximating DAPO with another objective.

`SAMPOSettings` belongs to the separate `train.sampo` operation. It owns the
discount factor, turn-advantage weight, sequence clipping bounds, reward
normalization, and bounded dynamic filtering. The rollout contract supplies
sampled assistant-turn token spans and preceding observation keys. A backend
must support both the sequence objective and hierarchical episode/turn
advantages or reject the request before launch.

Request seats assemble model, data/env, settings, training binding, and (for
online RL) inference binding. Train target and rollout target **may differ**.

Async GRPO policies are **out of MVP scope**.

### `EvaluationPlan`

| Field | Role |
| --- | --- |
| `id` / `revision` | Plan identity |
| `environments` | Env bindings / suite composition (one cell → one Verifiers run) |
| `inference_requirements` | Compatible binding constraints |
| `sampling` | Repetition, seeds, limits |
| `metrics_and_slices` | Required measures (projected from traces) |
| `aggregation` | Coverage / missing-evidence rules |
| `comparison` | Parent / foundation / baseline policy |

The model under test is **not** part of the plan; it is a seat on the eval
request.

Verifiers mapping (implementation target): each enabled plan cell becomes an
`EnvConfig` from the env package, merged with client/sampling/budget from the
request into Verifiers `EvalConfig`, then `run_eval` → native traces. See
[02 · Verifiers-backed eval evidence](./02-primitives.md#verifiers-backed-eval-evidence)
and [06 · Verifiers ingest](./06-observation-and-lineage.md#verifiers-ingest-notes).

### `Workload`

Request shapes, a versioned prompt-corpus identity, representative or
controlled cohort, ordered bounded concurrency sweep, saturation methodology,
warmup/repeat policy, and required operating measures. Project throughput,
latency, context, and reliability thresholds are not workload fields.

### `ExecutionTarget`

Device class, memory, placement, host constraints — recorded on every GPU-bound
run.

## Catalog

One **logical** catalog at resolve time: published **global catalog** + zero or
more **project overlays**.

```text
Global (framework-shared package resource)
  + Overlay(s) (project / work-package)
        │
        ▼
  Catalog.resolve(CatalogRef) -> Resolved[Selection]
```

| Rule | Meaning |
| --- | --- |
| Single read API | `catalog.resolve(ref)` only (no dual base-then-project lookup in configs) |
| Overlay wins | Same id in overlay replaces base |
| Scope | Usually project-scoped so overlays do not leak |
| Provenance | Snapshot records `source_layer: base \| overlay` (+ overlay id); `base` is the serialized name for the global layer |
| No silent global mutation | Publishing a shared entry requires a catalog package release |
| First-use materialization | Dataset pointers materialize idempotently into project state; environment bindings verify an installed pinned package |

```text
CatalogRef
  family: model | dataset | environment | inference | training
          | evaluation | workload | target | recipe
  id: str                    # e.g. models/qwen-2b@bf16

Resolved[T]
  ref: CatalogRef
  value: T                   # Selection of that family
  source_layer: base | overlay
  overlay_id: str | None

Catalog
  open(base, overlays, scope) -> Catalog
  resolve(ref) -> Resolved[Selection]
  contains(ref) -> bool
  list(family=None) -> [CatalogRef]
```

YAML / Python configs emit `CatalogRef`s (or inline selections). The job runtime
resolves before calling operations.

```yaml
bindings:
  starting_model: { family: model, id: models/qwen-2b@bf16 }
  training_data: { family: dataset, id: datasets/memory-sft-v3@<digest> }
```

Declarative dataset entries use existing `posttrain.data` adapter vocabulary:

```yaml
dataset:
  datasets/support-sft@1:
    revision: "1"
    source:
      kind: huggingface        # or jsonl | parquet | nemo
      repo: org/support-conversations
      revision: <immutable-revision>
      split: train
    format:
      kind: messages           # auto | messages | prompt-completion | alpaca | sharegpt
```

NeMo project files use `source.kind: nemo` with a project-relative JSONL
`path`. Supervised NeMo entries use format `messages` (or `auto`); preference
entries use format `nemo-ranked` (or `auto`). Materialization routes through
`supervised_from_nemo` / `preferences_from_nemo` and caches the same canonical
HF-normalized JSONL used by other sources.

Environment entries bind a pinned Verifiers package source and serializable
activation. Catalog loading does not import that package. `job pack`
fetches the full Git commit, may build several selected environment
subdirectories from one checkout, records each tree and wheel digest, and
qualifies every activation in the actual-job image. Standard GRPO, distillation,
and evaluation definitions build the existing environment bridges from the
resolved `EnvironmentBinding`; projects do not supply a parallel dataset seat
for environment-only GRPO.

## RunContext

Every operation takes a `RunContext` injected by the `JobRuntime` (or directly
by a library caller):

| Concern | Expectation |
| --- | --- |
| Identity | `project_id`, `work_package_id`, `run_id`, `job_kind`, job definition version |
| Observer | Optional sink for metrics, traces, artifacts, status |
| Workspace | Temp paths; durable evidence via observer/artifacts |
| Cancellation | Cooperative cancel |

Capability packages must not import Trackio or W&B. The job runtime injects an
observer opened by a provider-neutral tracking backend; projects may select
Trackio, W&B, no-op, or another conforming implementation. Trackio remains the
default local backend.

The observer is the operation-facing emission surface. Run start/finish,
provider translation, durable reads, and artifact materialization live behind
`posttrain.tracking` contracts. The framework's `run_id`, `work_package_id`,
logical metric step, artifact identity, and outcome remain authoritative;
provider IDs, groups, row numbers, and states are storage metadata.

## Operations (job kinds)

An **operation** is the public function for a **job kind**. Naming rule:

```text
job kind          package.operation
──────────        ─────────────────
data.prepare   →  posttrain.data.prepare
serve.benchmark→  posttrain.serve.benchmark
serve.smoke    →  posttrain.serve.smoke
eval.general   →  posttrain.eval.general
eval.domain    →  posttrain.eval.domain
train.sft      →  posttrain.train.sft
train.dpo      →  posttrain.train.dpo
train.grpo     →  posttrain.train.grpo
train.sampo    →  posttrain.train.sampo
train.distill  →  posttrain.train.distill
model.transform→  posttrain.train.transform   # or dedicated owner when split
```

Each call:

1. accepts a **typed request** (resolved seats + settings)
2. validates compatibility
3. uses / creates run identity via `RunContext`
4. executes through an internal adapter
5. returns a **typed result** (status, artifacts, summaries; traces via observer)

### Requests are seat-shaped

```text
# serve.benchmark
ServeBenchmarkRequest
  inference: InferenceBinding
  workload: Workload
  target: ExecutionTarget          # if not already fixed on the binding

# eval.domain / eval.general
EvaluateRequest
  model: ModelVariant | RemotePolicy
  plan: EvaluationPlan
  inference: InferenceBinding | RemoteEvaluationBinding
  target: ExecutionTarget

# train.sft
SFTRequest
  model: ModelVariant
  data: DatasetSelection
  settings: SFTSettings
  training: TrainingBinding          # includes update plan + train target

# train.grpo
GRPORequest
  policy: ModelVariant
  reference: ModelVariant | None
  environment: EnvironmentBinding
  settings: GRPOSettings             # selects grpo or dapo; algorithm only
  training: TrainingBinding          # train target (may differ from rollout)
  inference: InferenceBinding        # rollouts; owns rollout target
  quantization: QuantizationPlan | None   # when QAT

# train.sampo
SAMPORequest
  policy: ModelVariant
  environment: EnvironmentBinding
  settings: SAMPOSettings            # hierarchical agentic objective only
  training: TrainingBinding          # train target (may differ from rollout)
  inference: InferenceBinding        # multi-turn rollouts
  quantization: QuantizationPlan | None

# train.distill
OnPolicyDistillationRequest
  student: ModelVariant
  teacher: ModelVariant                    # frozen; scores student token ids
  environment: EnvironmentBinding          # Verifiers task interaction
  settings: OnPolicyDistillationSettings   # fully on-policy algorithm only
  training: TrainingBinding                # student update + train target
  rollout_inference: InferenceBinding      # current-student generation
  teacher_inference: InferenceBinding      # exact-token teacher scoring
  quantization: QuantizationPlan | None    # when student update requires it
```

`ServeBenchmarkRequest` does not accept a project-requirements seat. The
project runtime snapshots the typed project brief beside the resolved seats,
while the operation returns the measured concurrency points and their safe
failure states. Representative and controlled workloads retain distinct
identities.

`train.grpo` is environment-only in the current API. A work package binds an
`EnvironmentBinding` reference; it does not bind a dataset, prompt collection,
or exact task-id list. Category filters, deterministic sampling, and task or
interaction budgets belong to the versioned environment binding. The host may
project the environment's resolved tasks into the trainer's internal rollout
dataset shape, but that projection is an adapter detail and never becomes a
second public GRPO seat. Concrete task identities are retained in Verifiers
traces for replay.

`train.sampo` uses the same environment-owned task selection boundary but
requires multi-turn rollout evidence. Every optimized trajectory identifies its
sampled assistant turns by token span and stable preceding user/tool observation
key. Missing intermediate rewards use the explicit sparse-reward convention:
zero before the final turn and the trajectory reward on the final turn.

`train.distill` accepts only fresh, consume-once trajectories generated by the
current student through the bound Verifiers environment. The rollout binding
must select the student and declare purpose `rollout`; the teacher binding must
select the teacher and declare purpose `teacher-score`. Preflight verifies an
immutable tokenizer fingerprint covering ordered vocabulary and special-token
ids. Historical traces and teacher-generated completions are not accepted by
this operation.

Do not require `training.target == inference.target`. Colocation is a work-package
choice.

Evaluation has a second, evaluation-only subject path for a remote policy. It
binds a remote model selector to a versioned external service descriptor rather
than forcing an API model into `ModelVariant`. The descriptor carries an
OpenAI-compatible protocol revision, secret-variable name, safe headers, and
safe request defaults. It is not accepted by train, serve, or token-level
rollout APIs. The service and policy remain separate because the same policy
can be served locally, directly by a provider, or through a router.

Do not pass a `GenerationHandle` as a public seat across packages. If local eval
or train needs live generation, the **host** (or work-package runner) may start
serve-side generation and pass an opaque, host-owned endpoint descriptor that
satisfies the inference seat — capability packages still speak
`InferenceBinding`, not foreign handles. Remote evaluation instead uses the
typed evaluation-only remote binding described above; its client remains inside
Verifiers and does not become a cross-package generation handle.

### Package surfaces

```text
posttrain.data
  prepare(ctx, DatasetPrepareRequest) -> DatasetPrepareResult

posttrain.serve
  benchmark(ctx, ServeBenchmarkRequest) -> ServeBenchmarkResult
  smoke(ctx, ServeSmokeRequest) -> ServeSmokeResult

posttrain.eval
  general(ctx, EvaluateRequest) -> EvaluateResult
  domain(ctx, EvaluateRequest) -> EvaluateResult
```

`EvaluateResult` points at the retained **native Verifiers evaluation artifact**
(at least `traces.jsonl` + resolved config) and sync stats. Per-rollout detail
lives in observer `VerifiersTrace` records and/or that artifact — not as a
framework-owned parallel score table. Plan-level views in Observatory aggregate
those traces with explicit missing-evidence states.

```text
posttrain.train
  sft(ctx, SFTRequest) -> TrainResult
  dpo(ctx, DPORequest) -> TrainResult
  grpo(ctx, GRPORequest) -> TrainResult
  sampo(ctx, SAMPORequest) -> TrainResult
  distill(ctx, OnPolicyDistillationRequest) -> TrainResult
  transform(ctx, TransformRequest) -> ModelVariant   # weight-quant / merge / …
```

`TrainResult` includes recovery checkpoint identities, metric summaries, and an
optional nominated `ModelVariant` when materialization policy says so. Online
RL and on-policy distillation rollouts use the same Verifiers trace authority
as eval when an environment is bound. A distillation result also summarizes
teacher scoring and identifies the fresh policy revision and consumed batch
ids; these summaries do not replace the native trace artifact.

### Status vocabulary

Every result carries one of at least:

`succeeded` | `failed` | `cancelled` | `unsupported` | `partial`

Missing evidence is never coerced to numeric zero.

## Work composition

Thin by design. Projects stay ordinary Python / YAML.

### Recipe

```text
Recipe
  id, revision, stage
  seats: { name -> SelectionFamily }     # what must be bound
  jobs: [ { kind, definition, optional? } ]
  expected_artifacts: [...]
```

### Job definition

```text
JobDefinition
  id, kind
  description                              # stable human-readable purpose
  seats: { name -> expected Selection type }
  operation
```

The description is authored with the versioned definition and copied into each
run snapshot. Evidence readers do not resolve mutable definition text later.

Framework-shipped definitions live in `posttrain.jobs` and include stable ids
for SFT, DPO, GRPO, on-policy distillation, evaluation, serving smoke/benchmark,
and model transform. They resolve catalog seats, invoke existing data adapters
or Verifiers bridges where required, then call the public capability operation.
They do not freeze project learning rates, datasets, environments, targets, or
acceptance policy.

```text
standard_definitions() -> { definition_id: JobDefinition }
```

Projects customize standard jobs through bindings. A `ProjectEntry` may add a
new versioned definition or an unshipped factory, but cannot shadow a standard
definition id with different semantics.

### Work package

```text
WorkPackage
  project_id
  work_package_id
  stage: screen | train | qualify
  description                              # stage-level purpose
  recipe: CatalogRef | inline Recipe
  bindings: { seat -> CatalogRef | Selection }
  enabled_optional_jobs: [...]
  metadata: notes, labels                 # not job outcomes
```

The work-package description is project-authored and copied into every run in
the package. It complements decision questions in metadata; it is not a result,
status, or conclusion.

### Optional runner

```text
run_work_package(ctx, WorkPackage) -> WorkPackageResult
```

Resolves catalog refs, validates seats per job kind, runs enabled jobs, returns
per-job results. Convenience only — not a workflow engine and not a decision
system.

### Project runtime

```text
ProjectExecutionRequest
  project_layout
  catalog
  tracking
  optional entry

ProjectEntry
  configure(runtime) -> None

JobRuntime
  catalog
  definitions
  tracking
  scratch

build_job_runtime(
  ProjectExecutionRequest,
  tracking=None,
  extra_definitions=None,
) -> JobRuntime
```

`JobRuntime` is the preferred public name for the resolved execution registry.
`ProjectExecutionRequest` and `ProjectEntry` describe project composition
without making “host” a developer concept. Existing `WorkPackageHost*` symbols
remain additive compatibility aliases during migration.

## Evidence, tracking, and Observatory

| Concept | Rule |
| --- | --- |
| Artifact | Immutable selection identity (model, dataset, eval bundle, …) + parent links |
| Metric | Named value at lowest trustworthy grain; correlated to run |
| Trace | High-cardinality payload via observer; not jammed into scalars |
| Run snapshot | Resolved selections + job definition version + code revision + target |

```text
posttrain.tracking  # provider-neutral lifecycle and raw evidence reads
  get_run(run_id) -> RunRecord
  list_runs(filter) -> [RunRecord]
  metric_series(run_id, names) -> [MetricSeries]
  traces(run_id, query) -> TracePage
  artifacts(run_id) -> ArtifactSet

posttrain_observatory  # dedicated read product and query/intelligence service
  get_job_telemetry_schema(job_kind) -> JobTelemetryDefinition
  get_run_view(run_id) -> RunView
  get_run_alerts(run_id) -> [RunAlert]
  get_run_delta(run_id, cursor) -> RunDelta
  compare_runs(run_ids) -> RunComparison
  work_package_view(work_package_id) -> WorkPackageView
  stage_view(project_id, stage) -> StageView
  lineage_view(model: ModelVariant) -> LineageView
  serving_capacity_run_view(run_id) -> ServingCapacityRunView
  serving_pareto_view(project_id, screen_work_package_id) -> ParetoView
  export_report(view, format) -> MaterializedReport
```

Views must expose `missing` | `failed` | `unsupported` | `not_run` |
`reused_from_framework` | `incomparable`. No report API picks a production winner.

Serving-capacity views apply project constraints only after reading evidence.
Their versioned calculator selects the highest-throughput measured point that
satisfies context, latency, reliability, and evidence-completeness constraints.
It reports eligibility against the project minimum throughput and may compute
a Pareto set only across runs with the same requirement snapshot,
representative workload/corpus, target, and calculator version.

Each supported job kind has one versioned `JobTelemetryDefinition` owned by
Observatory and describing summary fields, chart series, health rules,
comparison keys, trace sections, and artifact roles. Python, HTTP, report
exports, the custom frontend, and MCP tools consume the same query service.
`get_run_delta` returns projection changes and alerts, not an unbounded stream
of raw provider rows.

## Config layout (illustrative)

```text
<project>/
  .posttrain/
    project.toml           # identity, tracking, optional entry, paths, state policy
    catalog/               # tracked project overlay
      datasets/...
      inference/...
    work_packages/
      screen_contenders.yaml
      train_sft_bootstrap.yaml
      qualify_sft.yaml
    state/                 # ignored scratch/cache/recovery/provider state
  pyproject.toml           # installed project, dependency pins, and optional [tool.posttrain.pack] source selection
  project_entry.py         # optional escape hatch; absent on the happy path
```

Prefer directory name `work_packages/` (not `packages/`) so it does not collide
with Python packages. The framework base catalog is a versioned package
resource; `.posttrain/catalog/` contains project overlays only. Durable
artifacts remain observer/backend values and do not derive identity from
`.posttrain/state/`.

`[tool.posttrain.pack]` may declare sorted `project_packages` and
`source_includes`. The normal single-package project defaults to installing
`.` and snapshots `pyproject.toml`, `src/`, and a declared readme. Monorepos
declare package roots explicitly. These values select code only; the framework
always adds the selected work package, project manifest, overlays, and project
brief as a closed configuration bundle. Repeatable CLI source options may
override the committed values for an experiment, and their selected bytes
become a different package identity. The packer never implicitly copies
`.git/`, `.posttrain/state/`, credentials, model weights, or unrelated
repository contents.

## Validation

Composition validation and detached planning perform only checks that are safe
on the developer machine: immutable references, required seats, selection
types, and cross-seat compatibility. They do not import CUDA backends,
Verifiers, or independently packaged environments. Explicit materialization
commands may install and validate local dependencies. The selected execution
runtime performs native backend and environment activation immediately before
the operation starts.

Before operation side effects, the composition layer or execution runtime
checks:

- required seats present for the job kind
- renderer/tokenizer alignment across model, dataset, inference, env
- execution target compatible with binding claims
- dataset kind matches training kind

Failures are typed validation errors when possible.

## Extensibility (first-party)

| Need | Path |
| --- | --- |
| New model / inference / workload / target | Catalog entry |
| New environment | Publish env package; `EnvironmentBinding` |
| New evaluation plan | Catalog `evaluation` entry |
| New dataset | `data.prepare` → `DatasetSelection` |
| New technique | New job kind + settings type + adapter in `train` |
| New engine | Adapter inside `serve` / `train`; keep request seats stable |
| New tracking backend | Implement writer and reader contracts; pass logical conformance tests |
| New run view / MCP surface | Extend the shared job telemetry definition and normalized view service |

Non-goals: plugin discovery, user-defined job kinds without a package change,
entry-point backend swapping, a generic operator-graph API.

## Non-goals

- universal training config shared by SFT/DPO/GRPO without kind-specific seats
- automatic checkpoint promotion
- project thresholds inside shared evaluation plans
- Trackio as a required dependency of capability packages
- W&B as a required dependency of capability packages
- provider-specific SQL/API calls in Observatory services or consumers
- cross-imports between `train`, `eval`, and `serve`

## Relationship to other docs

| Doc | Role |
| --- | --- |
| [02 · Primitives](./02-primitives.md) | What selections mean |
| [03 · Work and Evidence](./03-work-and-evidence.md) | Packaging and lineage |
| [04 · Framework](./04-framework.md) | Packages and DX |
| **05 · APIs** | Public names and call shapes |
| [06 · Observation](./06-observation-and-lineage.md) | Metrics, traces, artifacts, wiring |

## Reading order

1. [01 · Workflow](./01-workflow.md)
2. [02 · Primitives](./02-primitives.md)
3. [03 · Work and Evidence](./03-work-and-evidence.md)
4. [04 · Framework](./04-framework.md)
5. **05 · APIs**
6. [06 · Observation, Artifacts, and Lineage](./06-observation-and-lineage.md)
