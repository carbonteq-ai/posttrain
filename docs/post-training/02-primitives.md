# 02 · Primitives


> **Frozen baseline (2026-07-21).** Product authority: [post-training README](./README.md). Prefer implementation-plan / code changes over redesigning this doc unless explicitly unfrozen.

The [workflow](./01-workflow.md) explains the decisions a post-training project
makes. [03 · Work and Evidence](./03-work-and-evidence.md) explains how projects,
work packages, jobs, runs, artifacts, and views organize that work.

This document answers a different question: **what exact things does a developer
select when composing that work?**

- Which exact model and weight form?
- Which inference backend and settings generate tokens?
- Which trainer, technique, and optimization settings update the model?
- Which dataset revision, split, and transformation are consumed?
- Which task environment, tools, verifier, and raw reward signals apply?
- Which evaluation plan and operating workload measure the result?
- Which hardware and execution constraints apply?

Those selections are the **primitives**. Job definitions and recipes refer to
them by name and version. They are the concrete inputs that make a run
reproducible and comparable.

## What this document is not

Primitives are **not** required to share one construction lifecycle.

There is no universal rule of the form:

```text
definition + config → profile → outputs
```

That pattern fits some published presets, but it fails as a law:

| Primitive style | Example | Why a single lifecycle fails |
| --- | --- | --- |
| Immutable artifact | Model weights, dataset snapshot | Identity is the artifact digest; “config” does not create it |
| Executable specification | Environment, evaluation plan | Behavior is declared; outputs appear only when a job runs it |
| Concrete binding | Inference binding, training selection | Several selections composed for one purpose |
| Constraint / target | Execution target, workload | Inputs to comparability; they do not “emit” products alone |

**Outputs belong to runs of jobs**, not to primitives in isolation. A model
variant does not “produce traces.” An evaluation job that loads that variant
through an inference binding, against an evaluation plan, on an execution
target, produces traces and metrics.

## How primitives enter work

The work-and-evidence model remains authoritative for packaging. Primitives plug
into it as **slot values**:

```text
Job kind
  └── implemented by a versioned Job definition
        └── declares required primitive seats

Recipe
  └── reusable template for one Work package
        ├── selects job definitions (which jobs may run)
        └── declares primitive-selection slots (what must be bound)

Work package
  └── concrete recipe instance for one project and stage
        └── fills every required slot with exact versions

Run
  └── executes one Job definition with the package’s resolved selections
        └── records those selections + produced artifacts/observations
```

A recipe may be framework-shared or project-local. A work package always binds
exact revisions. Every run records the resolved selections even when they were
inherited from a recipe default.

**Example.** An SFT recipe requires seats for starting model, supervised data,
training settings, and execution target. The work package
`train/memory-sft-bootstrap` binds those seats to exact digests and revisions.
Optional diagnostic eval jobs in the same recipe may remain unbound until the
team chooses to run them.

A multi-stage path such as screen → SFT → qualify → GRPO → qualify is a
**post-training plan**: several work packages, each possibly from its own
recipe. It is not one mega-recipe and not one primitive.

## Selection map

Not every job kind consumes every primitive. Together, the families answer:

| Question | Primitive | Typical consumers |
| --- | --- | --- |
| Which exact weights and model interface? | **Model variant** | Almost every train / eval / serve / transform job |
| Which immutable supervised or preference examples are consumed? | **Dataset selection** | `data.prepare`, `train.sft`, `train.dpo` |
| Which task interaction, tools, sandbox, and verification? | **Environment binding** | `eval.domain`, `train.grpo`, data generation from rollouts |
| Which backend generates tokens, and how? | **Inference binding** | `serve.benchmark`, `serve.smoke`, eval generation, GRPO rollouts |
| Which trainer, technique, update plan, and train/rollout bindings? | **Training selection** (settings + `TrainingBinding` + optional `QuantizationPlan`) | `train.sft`, `train.dpo`, `train.grpo`, `model.transform` |
| Which tasks, slices, metrics, and comparison policy? | **Evaluation plan** | `eval.general`, `eval.domain` |
| Which request shapes and concurrency should be measured? | **Workload** | `serve.benchmark`, sometimes smoke / requalification |
| Which devices and resource limits execute the work? | **Execution target** | Every GPU-bound job |

The framework may publish reusable choices in any family. Projects still bind
exact versions and may create project-local choices when shared ones do not
match.

---

## Model variant

A model selection must identify one exact loadable weight state. “Qwen 2B” is
not enough.

Together with **inference binding** and **execution target**, this is how Q4 /
AWQ weights, serving recipes, and engine knobs are defined — as **three
layers**, not one profile:

```text
ModelVariant          → what weights are on disk (BF16, AWQ-int4, adapter, …)
InferenceBinding      → how those weights generate tokens for a purpose
  ├── backend         → which engine binary (vLLM@…, …)
  ├── engine          → engine-level config (KV cache, TP, speculative, …)
  ├── sampling        → generation defaults for that purpose
  └── target          → ExecutionTarget (device / VRAM / placement)
```

There is no separate “inference profile” type. Reusable serving setups are
**named inference bindings** in the catalog.

### What it establishes

| Field | Meaning |
| --- | --- |
| Artifact reference | Immutable weights or adapter revision (URI + digest/revision) |
| Family and architecture | Compatibility with trainers, renderers, and inference backends |
| Artifact form | Foundation weights, adapter, merged descendant, or weight-quantized descendant |
| Weight precision / quantization | BF16, FP16, AWQ, GPTQ, GGUF-Q4, or another *materialized* representation |
| Base and parent artifacts | Lineage for adapters, merges, and trained descendants |
| Tokenizer and renderer contract | Chat template, roles, reasoning mode, tool-call representation |
| Capabilities | Modalities, claimed context limits, and other model facts |
| Provenance | Source, license, producing run, and transformation history |

### Quantization and transformation boundary

| Change | Layer | Why |
| --- | --- | --- |
| Weight quantization (AWQ / GPTQ / GGUF Q4 / …) | **new `ModelVariant`** | Stored weights changed |
| Adapter train / merge / export that changes the loadable artifact | **new `ModelVariant`** | New artifact + parent link |
| KV-cache dtype / quant, scheduler, speculative decoding, kernels, GPU mem util | **`InferenceBinding.engine`** | Same weights; runtime only |
| Temperature, max tokens, stop policy for a purpose | **`InferenceBinding.sampling`** | Purpose-specific generation |
| Device class / VRAM / placement policy | **`ExecutionTarget`** (usually referenced by the binding) | Hardware constraints |

This split lets one quantized variant be screened under several engine configs
without pretending each runtime knob creates new weights — and lets BF16 and
Q4 stay comparable as distinct model identities.

### Catalog examples (model layer)

```text
models/qwen-2b@bf16
  artifact: hf://Qwen/...@<immutable revision>
  form: foundation
  weight_precision: bf16
  renderer_contract: qwen-tools@1

models/qwen-2b@awq-int4
  artifact: artifact://...@<digest>
  form: weight-quantized
  weight_precision: int4
  quantization: { method: awq, bits: 4, scheme: ... }
  parent: models/qwen-2b@bf16
  produced_by: run://…/model.transform/…    # or imported pre-quantized

models/qwen-2b@gguf-q4-k-m
  artifact: artifact://...@<digest>
  form: weight-quantized
  quantization: { method: gguf, type: Q4_K_M }
  parent: models/qwen-2b@bf16

models/qwen-memory-sft@<digest>
  form: adapter-or-merged-descendant
  parent: models/qwen-2b@bf16
  produced_by: run://train/memory-sft-bootstrap/...
```

Weight-quantized variants are normally created by `model.transform` (or an
import that records the same fields). Projects bind the variant id; they do not
embed quant recipes inside inference bindings.

### Common mistakes

- Treating a family name as a variant
- Folding KV-cache or speculative settings into model identity
- Treating “Q4” as an inference profile instead of a weight variant (when
  weights are actually quantized on disk)
- Losing parent links when materializing a checkpoint

---

## Dataset selection

A dataset selection identifies the semantic data contract and the exact data a
job consumes.

For the current product boundary, datasets are public training inputs for SFT
and DPO. GRPO does not expose a dataset or prompt-task seat: it binds a
Verifiers environment, which owns its task population and resolves the actual
tasks used by each run. A later offline-RL operation may introduce a trajectory
dataset contract, but it is not part of `train.grpo`.

### What it establishes

| Field | Meaning |
| --- | --- |
| Data kind | Supervised conversation, prompt/completion, preference, prompt/task, rollout/trajectory, or another declared contract |
| Snapshot or source | Immutable materialized revision, or a source that `data.prepare` must materialize |
| Split and subset | Exact train, validation, held-out, or named slice |
| Schema version | Meaning of examples, messages, tools, scores, and targets |
| Provenance | Upstream datasets, traces, human review, and producing runs |
| Transformations | Filtering, dedup, mixture, conversion, and transformation revision |
| Access policy | License, privacy, redaction, retention |

### What does *not* change dataset identity

Packing, tokenization, chat rendering, and max sequence length belong to the
**consuming training selection** (or to a deliberately materialized
pre-tokenized derived dataset with its own provenance).

### Examples

```text
datasets/memory-sft-v3@<digest>
  kind: supervised
  split: train
  derived_from: reviewed memory-agent demonstrations
  source_runs: [run://..., run://...]

datasets/memory-preference-v2@<digest>
  kind: preference
  split: train
  derived_from: accepted vs rejected trace branches

datasets/memory-tasks-v1@<digest>
  kind: prompt-task
  split: train
  environment_compatibility: envs/memory-agent-v2@<revision>
```

Trace-derived datasets must keep references to source runs, traces, environment
revision, and the transform that produced them.

### Common mistakes

- Pointing training at a moving Hub tip without a digest
- Baking renderer/packing into “the dataset” so the same examples cannot be
  reused with another model contract
- Dropping environment lineage when exporting rollouts to SFT data

---

## Environment binding

An environment defines the task behavior in which a model acts and is checked.
In this lab it is typically a **Verifiers v1** environment: a published package
that contributes a **Taskset** (and usually pairs with a Harness via
`EnvConfig`). One environment may support evaluation, online training, data
generation, or all three.

The framework does **not** add another task-row schema, reward interface, or
eval suite inside Verifiers. Env packages own tasks; our evaluation **plan**
only selects and aggregates them. Rollout evidence is Verifiers **traces** —
see [Verifiers-backed eval evidence](#verifiers-backed-eval-evidence).

### What it establishes

| Field | Meaning |
| --- | --- |
| Taskset | Task schema, source, revision, available splits (`Taskset.load` / Hub id) |
| Harness | Interaction loop, context management, stop behavior |
| Tools and interfaces | Tool schemas, user simulation, external services, sandboxes |
| Verification | Deterministic checks, judges, rubrics, **raw** reward signals (`@reward` / `@metric`) |
| Environment parameters | Task-meaningful timeouts, limits, configurable behavior |
| Compatibility | Required modalities, tool protocol, renderer behavior, sandbox needs |

### Ownership split (critical)

| Owner | Owns |
| --- | --- |
| **Environment** | What each reward/component *means*; task legality; verification truth |
| **Evaluation plan / eval job** | Repetition, held-out selection, slicing, aggregation, missing-evidence rules |
| **Training selection (e.g. GRPO)** | Rollout count, sampling, advantage/KL, **reward scalarization / weights** |
| **Job / execution target** | Worker count, scheduling, infrastructure |

If an environment exposes several reward components, the environment defines
those components. A GRPO training selection decides how they are weighted or
transformed for optimization.

### Example

```text
envs/memory-agent-v2@<revision>
  taskset: memory-write-read-contradiction
  tools: [memory.write, memory.search, memory.delete]
  verification: [exact_recall, no_hallucinated_id, contradiction_free]
  reward_components:
    task_success: binary end-state
    contradiction: penalty signal
    tool_error: diagnostic (often not optimized directly)
```

### Common mistakes

- Putting GRPO group size or KL coef into the environment
- Treating the environment as the evaluation plan
- Changing verification silently without bumping environment revision

---

## Inference binding

An inference binding states exactly how a **model variant** generates tokens
for a particular purpose. It is the catalog object that used to be casually
called a “serve / inference profile.”

The same variant (e.g. `models/qwen-2b@awq-int4`) may have several bindings:
screen under tight latency, eval with different sampling, GRPO rollouts
colocated with TRL, handoff with speculative decoding enabled.

### What it establishes

| Field | Meaning |
| --- | --- |
| Model variant | Exact weights loaded (`ModelVariant` / catalog model id) |
| Backend | Engine product + version (vLLM, Transformers, SGLang, …) |
| Renderer | Chat template, reasoning mode, tool parsing, generation boundary |
| Engine settings | Backend-owned runtime config (see below) — **not** a separate catalog family |
| Sampling defaults | Temperature, top-p, token limits, stop policy, seed behavior |
| Execution target | Compatible hardware target and placement assumptions |
| Purpose | `screen` \| `eval` \| `rollout` \| `smoke` \| `handoff` |

### Engine-level config (`engine`)

Lives **on the binding**, owned by the `serve` (or train-colocated) adapter.
Illustrative vLLM-shaped fields — concrete schema is backend-specific:

| Concern | Examples (illustrative) |
| --- | --- |
| Parallelism | `tensor_parallel`, `pipeline_parallel` |
| Memory | `gpu_memory_utilization`, max model / seq len claims |
| Cache | KV dtype / KV quantization, prefix cache policy |
| Scheduler | max num seqs, batching policy |
| Kernels / features | attention backend, CUDA graphs, speculative decoding |
| Colocation | adapter broadcast / sync when used for online RL rollouts |

Rules:

1. Changing `engine` → new binding revision (or new binding id), **same** model
   variant.
2. Do not invent a third public type (`EngineProfile`). Reuse is “publish
   another inference binding” (base catalog or project overlay).
3. Project overlays may shadow a base binding id when only engine/sampling
   differ for that project.

### Catalog examples (binding layer)

```text
# BF16 weights, standard screen/eval engine config
inference/qwen-2b-bf16-vllm-standard@2
  model: models/qwen-2b@bf16
  backend: vllm@<version>
  renderer: qwen-tools@1
  engine:
    tensor_parallel: 1
    gpu_memory_utilization: 0.90
    kv_cache: { dtype: bf16 }
  sampling: { temperature: 0.0, max_tokens: ... }
  target: hardware/rtx3070ti-8gb
  purpose: [screen, eval, handoff]

# Same family, weight-quantized variant, tighter memory engine config
inference/qwen-2b-awq-vllm-screen@1
  model: models/qwen-2b@awq-int4
  backend: vllm@<version>
  renderer: qwen-tools@1
  engine:
    tensor_parallel: 1
    gpu_memory_utilization: 0.85
    kv_cache: { dtype: fp8 }          # runtime KV — not a new model variant
  sampling: { temperature: 0.0, ... }
  target: hardware/rtx3070ti-8gb
  purpose: [screen, eval]

# Online RL: same or descendant weights, rollout-oriented binding
inference/qwen-memory-vllm-colocate@1
  model: models/qwen-memory-sft@<digest>
  backend: vllm@<version>
  engine: { ... , colocate: true }
  sampling: { temperature: 1.0, ... }
  sync: adapter-broadcast
  purpose: [rollout]
```

A GRPO rollout binding is explicit even when it reuses pieces of a serving
configuration. Do not silently reuse a screen binding for rollouts.

### Common mistakes

- One “serve profile” silently used for screen, eval, and RL rollouts
- Embedding the model artifact digest only in logs, not in the binding identity
- Calling runtime KV quantization a new model variant
- Putting weight-quant recipes (AWQ/GPTQ/GGUF) inside `engine` instead of
  creating / selecting a weight-quantized `ModelVariant`
- Copy-pasting full engine YAML into every work package instead of a catalog
  binding id (use overlay shadow when the project must diverge)

---

## Training selection

Training is selected **per job kind**, but **algorithm**, **how parameters
update**, **train runtime**, and **rollout runtime** are different seats.

There is no universal training mega-schema. SFT, DPO, GRPO, and on-policy
distillation keep kind-specific **algorithm settings**. They must not embed
QLoRA-as-default, vLLM topology, or a single forced train=rollout target.

### Layers (normative)

| Layer | Type | Owns |
| --- | --- | --- |
| Algorithm settings | `SFTSettings` \| `DPOSettings` \| `GRPOSettings` \| `OnPolicyDistillationSettings` | Algorithm identity, generations, divergence semantics, advantages, reward **weights**, IS *learning* semantics, prompt/completion limits, opt schedule |
| Parameter update | `ParameterUpdatePlan` | Full-parameter \| LoRA \| QLoRA \| quantization-aware (QAT) |
| Training binding | `TrainingBinding` | Train backend, update plan, normalized train parallelism/runtime, backend-specific options, **training** `ExecutionTarget` |
| Rollout inference | `InferenceBinding` | Rollout backend/`engine`/sampling, **rollout** `ExecutionTarget` |
| Quantization plan | `QuantizationPlan` | Recipe + calibration data/budget + formats; used by offline transform and/or QAT |

```text
Offline PTQ:   ModelVariant → transform(QuantizationPlan) → weight-quantized child
QAT:           ModelVariant → train(…, QuantizationAwareUpdate + plan) → QAT child
Runtime quant: fake vs real quant kernels → InferenceBinding (when lineage unchanged)
```

Colocated GRPO may bind the same allocation in a work package; the **primitive**
must allow distinct train vs rollout targets (NeMo-style split topology).

**Deferred:** async GRPO (trajectory age, off-policy correction, in-flight
weight/KV policies). Not part of the current MVP.

In the API ([05](./05-apis.md)), seats stay on the request; settings types are
cataloguable under family `training` (algorithm) and related families for
bindings / quant plans.

### What a training run must identify

- training job definition and backend revision
- exact model roles (policy, reference, …)
- `ParameterUpdatePlan` and `TrainingBinding` (train target included)
- exact supervised/preference data selections for SFT/DPO, or the exact
  Verifiers environment selection for online training
- renderer contract alignment
- technique-specific **algorithm** settings
- checkpoint and materialization policy
- for online RL and on-policy distillation: student rollout `InferenceBinding`
  (may use a different target)
- for on-policy distillation: frozen teacher `ModelVariant`, teacher-scoring
  `InferenceBinding`, and an exact tokenizer token-id compatibility fingerprint
- when quantizing: `QuantizationPlan` (offline and/or QAT)

### Required selections by job kind

| Job kind | Required selections |
| --- | --- |
| `train.sft` | Starting model, supervised data, `SFTSettings`, update plan, training binding/target |
| `train.dpo` | Policy (+ reference semantics), preference data, `DPOSettings`, update plan, training binding/target |
| `train.grpo` | Policy, versioned Verifiers environment, `GRPOSettings` selecting `grpo` or `dapo`, update plan, training binding, rollout inference binding |
| `train.sampo` | Policy, versioned multi-turn Verifiers environment, `SAMPOSettings`, update plan, training binding, rollout inference binding |
| `train.distill` | Student, frozen teacher, versioned Verifiers environment, `OnPolicyDistillationSettings`, update plan, training binding, student rollout inference binding, teacher-scoring inference binding |
| `model.transform` | Source model, `QuantizationPlan` (or transform settings referencing it), target |

### Example

```text
train.grpo seats
  model: models/qwen-2b@bf16
  settings: training-settings/grpo-math@1          # algorithm only
  training: training-bindings/trl-qlora-local@1   # update plan + train target
  inference: inference/qwen-vllm-colocate@1       # rollout engine + rollout target
  environment: envs/gsm8k-train@1
  # optional QAT:
  # quantization: quantization/nvfp4-w4a16@1
```

The GRPO environment reference may carry category filters, a deterministic
sampling seed, task/rollout budgets, and interaction limits. It must not expose
hand-picked task IDs as a work-package seat. The environment resolves concrete
tasks at runtime and the run records their identities in native Verifiers
traces, so researchers can replay what happened without coupling the public job
contract to environment-internal row numbering. `GRPOSettings.loop.max_steps`
and the rollout/group settings bound how much of that population is consumed.

`GRPOSettings.algorithm` selects the group-relative update objective. `grpo`
uses the existing sequence-normalized objective and symmetric clipping. `dapo`
uses token-level policy loss, separately selected lower and upper clip
epsilons, global active-token normalization, bounded retained-group dynamic
sampling, truncation handling, and optional linear soft-overlong punishment.
The sampler keeps prompt groups with reward variation and generates only enough
replacement groups to fill the optimizer batch. Exhausting the configured
candidate bound fails the run rather than silently changing the batch.

The length punishment is algorithm-owned shaping added to the environment
reward; the environment remains the authority for task reward meaning.
Efficiency, memory, synchronization, and observability changes that preserve
these semantics remain DAPO implementation improvements, not a new algorithm.
CISPO, GSPO, and Dr. GRPO replace objective or normalization semantics and are
not DAPO flags.

SAMPO is a separate selection for multi-turn tool-using agents. It combines one
sequence-level importance ratio per trajectory with a token-aligned advantage
formed from a group-relative episode advantage and an anchor-state-relative
turn advantage. An anchor state is the latest user or tool observation before a
sampled assistant turn. The rollout records each sampled turn's half-open token
span and a stable anchor-state key; environment/tool tokens remain outside the
loss mask. Sparse environments assign the terminal trajectory reward to the
final sampled turn and zero to earlier turns before discounted returns are
computed. A backend without both sequence-level clipping and hierarchical
agentic advantages rejects `train.sampo`; GSPO alone is not SAMPO.

```text
train.distill seats
  student: models/qwen-0.8b@bf16
  teacher: models/qwen-2b@bf16
  settings: training-settings/on-policy-distill-math@1
  training: training-bindings/trl-lora-local@1
  rollout_inference: inference/qwen-0.8b-vllm-rollout@1
  teacher_inference: inference/qwen-2b-vllm-teacher-score@1
  environment: envs/gsm8k-train@1
```

For `train.distill`, on-policy means every optimized trajectory was generated
by the current student weights for that training step and is consumed once
while fresh. Verifiers owns task interaction, tools, sandbox, verification, and
the native trajectory. The frozen teacher scores the exact integer token ids
from that trajectory; it does not generate the optimized completion. Student
training, student rollout, and teacher scoring may use different execution
targets. Exact scoring requires the student and teacher tokenizers to have the
same ordered vocabulary and special-token mapping. Stored teacher completions
or historical student traces are off-policy data and belong in supervised data
generation or `train.sft`, not `train.distill`.

Recovery checkpoints are trainer state. Only owner-nominated materialization
creates a new **model variant** artifact for later jobs.

### Common mistakes

- One schema pretending SFT and GRPO take the same seats
- Putting reward component *meanings* in training instead of the environment
- Treating every checkpoint as a model variant
- Stuffing vLLM TP/memory/speculative into `GRPOSettings`
- Requiring train target == rollout target at the type level
- Calling Trackio or background rollout I/O “async RL”
- Treating Verifiers reward as a distillation loss or routing distillation
  through GRPO
- Retokenizing student text with the teacher instead of scoring the exact
  student token ids
---

## Evaluation plan

An evaluation plan defines what evidence should be collected and how it should
be interpreted. The model under test is a **run input**, not part of the
reusable plan’s identity.

For Verifiers-backed evals, the plan does **not** replace Verifiers types. It
selects environment bindings (cells), budgets, slices, aggregation, and
comparison policy. Each cell runs through Verifiers v1; see
[evidence model](#verifiers-backed-eval-evidence) below.

### What it establishes

| Field | Meaning |
| --- | --- |
| Environments and tasksets | Exact evaluation content and held-out splits (via env bindings) |
| Inference requirements | Compatible generation binding or required behavior |
| Sampling and repetition | Seeds, sample counts, generation limits, retry policy |
| Metrics and slices | Required measures and failure categories (projected from traces) |
| Aggregation | Coverage, missing-evidence rules, summary calculations |
| Comparison policy | Parent, foundation, sibling, or published baseline references |

### Examples

```text
evals/general-compact@1
  environments: [reasoning-math, instruction, tool-use, long-context]
  aggregation: per-category + missing-evidence
  comparison: optional published baseline ids

evals/memory-heldout@2
  environment: envs/memory-agent-v2@<revision>
  split: heldout
  slices: [write_fail, recall_miss, contradiction, tool_error]
  comparison: [parent_model, foundation_model]
```

Running a plan against a particular model creates one or more **runs** (typically
one run per environment cell) and evaluation evidence. It does not mint a new
evaluation-plan identity.

### Verifiers-backed eval evidence

Verifiers v1 does not expose a suite object. It exposes:

```text
EnvConfig (taskset + harness + limits)
  -> Environment
  -> EvalConfig (+ model client, sampling, num_tasks/rollouts)
  -> run_eval -> list[Trace]  (+ traces.jsonl, config.toml, …)
```

Our contract for `eval.general` / `eval.domain` (and env-backed GRPO rollouts):

| Layer | Authority |
| --- | --- |
| **Native Verifiers bundle** | Replay source of truth (`traces.jsonl`, resolved `config.toml`, logs) |
| **Observer `VerifiersTrace` records** | Idempotent projection of completed rollouts for query |
| **`eval/*` scalars / views** | Bounded aggregates extracted from traces — never a second score DB |
| **Project thresholds** | Accept/revise/reject — outside the job; read the evidence |

Rules:

1. Do **not** invent a parallel task/reward schema in `posttrain.eval`.
2. Task semantics and reward *meanings* stay in the env package (`Task` /
   `@reward` / `@metric`).
3. Missing or unsynced rollouts are evidence gaps (`missing` / sync-partial),
   not silent numeric zeros.
4. Optional later: project selected trace branches into datasets via `data`
   (still citing source runs/traces).

### Common mistakes

- Baking a specific model digest into the plan name
- Omitting missing-evidence rules so gaps become silent zeros
- Confusing the plan with the environment
- Treating Trackio aggregates as authoritative over `traces.jsonl`
- Re-implementing Verifiers scoring in reports instead of reading trace fields

---

## Workload

A workload describes the request distribution used to test **operating fit**.
It is intentionally separate from inference so several model/backend choices can
be compared under the same project conditions.

### What it may specify

- context sizes and prompt/output token shapes
- a versioned representative prompt-corpus identity, provenance, and renderer
- an ordered, bounded concurrency sweep and arrival behavior
- representative tools, modalities, or structured-output patterns
- warmup, repetition, failure, and saturation policy
- which operating metrics must be collected (TTFT, ITL, e2e, throughput, VRAM,
  KV behavior, truncation, errors)

The workload does not own project accept/reject thresholds. A representative
workload uses model-visible messages rendered through the declared model
renderer and is suitable for project capacity interpretation. A controlled
diagnostic workload may use exact token-id shapes to isolate backend behavior.
Those cohorts have different identities and must not be merged into one
capacity curve or Pareto set.

### Example

```text
workloads/general-serving-32k-sweep@1
  context: 32768
  cohort: representative
  corpus: general-serving-v1@1 (content digest recorded)
  concurrency: [1, 2, 4, 8, 12, 16]
  shapes: [reasoning, code, chat, extraction, structured-output, tool-use]
  output_tokens: 128
  measures: [ttft, e2e_p95, thruput, peak_vram, truncations]
```

### Common mistakes

- Embedding workload only inside an inference binding so comparisons drift
- Changing shapes between candidates without calling it a new workload
- Treating a project throughput or latency threshold as workload methodology
- Padding representative prompts into controlled token shapes without changing
  the workload identity

---

## Execution target

An execution target records the resources and constraints under which a run is
valid and comparable.

### What it establishes

- device class and count
- memory capacity and utilization limits
- topology and placement
- host or scheduler constraints
- container, driver, and accelerator assumptions when relevant

### Example

```text
hardware/rtx3070ti-8gb@1
  device: nvidia-3070ti
  vram_gib: 8
  policies: [no_cpu_offload_for_screen, max_concurrency_cap: 4]
```

Hardware is an input and part of evidence context. It does not produce a product
artifact by itself.

---

## Job definitions, recipes, and work packages

These compose primitives without replacing them:

| Concept | Meaning |
| --- | --- |
| Job kind | Stable semantic contract (`train.sft`, `serve.benchmark`, …) |
| Job definition | Versioned runnable implementation of exactly one job kind |
| Recipe | Reusable template for one stage-tied work package |
| Work package | Project-local recipe instance with exact primitive selections |
| Run | One execution of one job definition inside the work package |

A job definition declares which primitive seats its kind requires. A recipe
selects job definitions and exposes the slots a project must fill.

### Recipe example

```text
recipe: recipes/sft-bootstrap@2
stage: train
question: Does supervised bootstrap produce a descendant worth qualifying?

slots:
  starting_model: ModelVariant
  training_data: DatasetSelection[supervised]
  training_settings: SFTSettings
  execution_target: ExecutionTarget

jobs:
  prepare:
    kind: data.prepare
    definition: data/canonicalize-supervised@3
    optional: true
  train:
    kind: train.sft
    definition: trainers/trl-sft@5
  diagnostic:
    kind: eval.domain
    definition: eval/verifiers-domain@2
    optional: true

expected_artifacts:
  - materialized model descendant
```

### Work package binding

```text
work_package: train/memory-sft-bootstrap
project: background-memory-agent
recipe: recipes/sft-bootstrap@2

bindings:
  starting_model: models/qwen-2b@bf16
  training_data: datasets/memory-sft-v3@<digest>
  training_settings: project/memory-sft-qlora-v1
  execution_target: hardware/rtx3070ti-8gb
```

Seeds, retries, and optional diagnostics create additional **runs** inside the
same work package. The recipe stays reusable; the work package holds the
decision context.

---

## Scenario A — Background memory agent

**Problem.** An assistant must write, retrieve, and maintain long-lived
user/project memory across sessions: correct recall, no hallucinated memory
ids, contradiction avoidance, within latency/context budgets on 8 GB.

**Project.** `background-memory-agent`

### Screen

Compare at least:

| Seat | Candidate A | Candidate B |
| --- | --- | --- |
| Model variant | `models/qwen-2b@bf16` | `models/qwen-2b@awq-int4` |
| Inference binding | `inference/qwen-2b-bf16-vllm-standard@2` | `inference/qwen-2b-awq-vllm@1` |
| Workload | `workloads/memory-recall-32k-c2@1` | same |
| Execution target | `hardware/rtx3070ti-8gb@1` | same |

Job: `serve.benchmark`. Optional: skip heavy general eval if framework-shared
baselines already cover both foundations.

**Decision.** Pick one model variant + inference binding as the starting point.

### Train SFT

Work package `train/memory-sft-bootstrap` from `recipes/sft-bootstrap@2`:

| Seat | Binding |
| --- | --- |
| Starting model | selected screen variant |
| Dataset | `datasets/memory-sft-v3@<digest>` |
| Training selection | project QLoRA SFT settings + `trainers/trl-sft@5` |
| Execution target | `hardware/rtx3070ti-8gb@1` |

**Runs produce.** Checkpoints; owner materializes `models/qwen-memory-sft@<digest>`.

### Qualify SFT

Work package `qualify/memory-sft-bootstrap`:

| Seat | Binding |
| --- | --- |
| Model under test | `models/qwen-memory-sft@<digest>` |
| Evaluation plan | `evals/memory-heldout@2` |
| Inference binding | qualification-oriented binding |
| Environment (via plan) | `envs/memory-agent-v2@<revision>` |

**Runs produce.** Verifiers traces, slice metrics; optional `serve.smoke`.
**Decision.** Accept as GRPO/DPO parent, revise data, or reject.

### Branch GRPO

Work package `train/memory-grpo-reward-v2`:

| Seat | Binding |
| --- | --- |
| Policy model | SFT descendant |
| Task data | `datasets/memory-tasks-v1@<digest>` |
| Environment | `envs/memory-agent-v2@<revision>` |
| Rollout inference | `inference/qwen-memory-vllm-colocate@1` |
| Training selection | GRPO settings + reward weights over env components |
| Execution target | 8 GB target |

**Runs produce.** New model variant + train metrics + Verifiers traces during
rollouts.

### Qualify GRPO

Same pattern as SFT qualify against the GRPO descendant; add general regression
plan if risk warrants; accept / revise / reject for serving handoff (handoff
pairs model variant + inference binding — not a separate stage).

**Critique.** Are reward weights clearly in training selection? Is colocated
rollout inference a distinct binding? Did dataset export keep env lineage?

---

## Scenario B — Agentic workflow optimization

**Problem.** Improve an automation agent that plans and executes multi-step
workflows (API discovery, tool calls, goal end-state), reducing invalid actions
and raising held-out success under fixed concurrency/latency. AutomationBench-
shaped.

**Project.** `agentic-workflow-opt`

### Distinctive selections

| Seat | Example |
| --- | --- |
| Environment | `envs/automationbench-v1@<revision>` (shared or project-extended) |
| Dataset | `datasets/workflow-sft-v1@<digest>` derived from reviewed successful traces |
| Evaluation plan | `evals/automationbench-heldout@1` |
| Training | SFT bootstrap, then GRPO with dense reward weights over env components |
| Workload | workflow-like I/O sizes for screen |

### Play-through

1. **Screen** — same workload across contenders; reuse framework-shared general
   baselines when valid; optional tiny domain probe.
2. **Train SFT** — optional `data.prepare` from env traces → supervised dataset
   selection → SFT training selection → materialized descendant.
3. **Qualify SFT** — held-out AutomationBench plan; inspect assertion/end-state
   traces; decide if GRPO is justified.
4. **Train GRPO** — policy = SFT descendant; environment = AutomationBench;
   rollout inference binding explicit; reward weights in training selection.
5. **Qualify GRPO** — domain plan + optional general regression + serve smoke;
   accept/revise/reject.

**Critique.** Is dense reward mixture training config (yes) vs environment def
(no)? Does the SFT dataset selection record the environment revision that
produced its traces?

---

## Scenario C — Tool-using support agent

**Problem.** Short-turn support agent with a fixed tool set (order lookup,
refund policy, ticket create): correct tool use, refuse unsafe actions, keep
replies fast on the same 8 GB box. Less long-horizon than B; more policy than A.

**Project.** `support-tool-agent`

### Distinctive selections

| Need | Selection |
| --- | --- |
| Tools + grading | Environment `envs/support-tools-v1@<revision>` |
| Behavior cloning | Dataset `datasets/support-sft-v1@<digest>` |
| Prefer/refuse | Preference dataset + `train.dpo` selection (often before GRPO) |
| Operating fit | Short-interactive **workload** dominates screen |
| Regression | `evals/support-heldout@1` + thin instruction-following plan |

### Play-through

1. Screen on short-interactive workload; pick model variant + inference binding.
2. SFT train → qualify on support environment plan.
3. DPO train on preference data → qualify again.
4. Handoff accepted model variant with the inference binding validated for
   production-like smoke.

**Critique.** Do tools need a primitive outside Environment? (Default: no —
tools live in environment bindings.) Is handoff explicitly
`ModelVariant + InferenceBinding`?

---

## Invariants

- Every run identifies an exact job definition and job kind.
- Every run records the exact primitive selections it resolved.
- A recipe is a reusable work-package template; a work package is its concrete
  project instance.
- A primitive may be an artifact, specification, preset, or binding. It does not
  need to behave like every other primitive.
- Weight-changing transformations create new model variants.
- Runtime-only inference settings do not create new model variants.
- Dataset rendering and packing do not alter canonical dataset identity unless a
  separate derived dataset is materialized.
- Environments define task and verification semantics; training and evaluation
  define how those semantics are sampled and interpreted.
- Job runs produce artifacts and observations. Primitives do not produce outputs
  in isolation.
- Exact artifact relationships — not package order — remain the source of
  lineage.

## Critique checklist

After reading the scenarios, check:

1. Can every concrete choice in a run be pointed at as one of the eight
   primitive families?
2. Are algorithm differences isolated inside **training selections** / job
   definitions?
3. Are Verifiers traces always produced by jobs that bind an environment (eval
   or online train) — never by a floating type?
4. Is lineage only via **model variant** artifacts and their parent links?
5. What felt forced — missing family, or a boundary that should move?

Continue with [03 · Work and Evidence](./03-work-and-evidence.md).
