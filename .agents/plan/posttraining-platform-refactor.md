# Post-Training Platform Refactor Plan

**Status:** In progress, revision 3
**Created:** 2026-07-20  
**Last revised:** 2026-07-20  
**Intent:** MVP revamp; existing code is not a compatibility contract

## Summary

Refactor the playground into a small, code-first post-training platform whose
reusable engines can evolve independently across projects. The platform will
screen foundation models, evaluate general and domain behavior, apply SFT/DPO/
GRPO as branchable training operations, benchmark serving, and preserve the
resulting evidence and lineage in Trackio.

The target is not a generic orchestration framework. `train`, `eval`, `serve`,
and `reports` are reusable Python operations; code-defined jobs in `apps/lab`
compose them for a particular finetuning problem. Framework-specific semantics
remain visible through typed configs and internal adapters.

## Planning inputs and settled decisions

This plan combines three inputs that must be read together:

1. The 3/10 code critique identifying fourteen concrete implementation
   failures: misplaced benchmark data, a repository-bound `common`, CLI-owned
   behavior, untyped dictionaries, Trackio coupling, duplicate local runs,
   false concurrency semantics, CLI-to-CLI subprocess composition, eval/vLLM
   coupling, unclear aggregation, reports/schema coupling, an empty heavyweight
   train package, generic package names, and missing quality gates.
2. The functional lifecycle in `docs/functional/`: screen reusable foundation
   profiles, evaluate general and task-specific behavior, apply branchable
   training techniques, re-evaluate descendants, and carry proven reusable work
   into future jobs.
3. The architectural decisions made after the critique:

   - use `posttrain.*` rather than generic top-level imports;
   - keep foundation model definitions in `posttrain.common`;
   - make train/eval/serve/reports reusable operations, not runners;
   - make jobs mandatory, code-defined compositions owned by `apps/lab`;
   - make Trackio a pure observability/evidence layer;
   - use Verifiers v1 natively and publish environments independently;
   - use Renderers for SFT and DPO;
   - bridge Verifiers traces to TRL for GRPO;
   - support vLLM only in the MVP and defer SGLang;
   - start with Qwen3.5-2B and LFM2.5-1.2B-Thinking;
   - remove obsolete code instead of maintaining compatibility aliases.

Where an older document still mentions repository YAML as the canonical
configuration, durable local `runs/`, SGLang as an implemented backend, or
Trackio inside reusable packages, this plan is the migration authority.

## Critique-to-refactor traceability

Every original critique finding has an explicit destination and acceptance
condition. A phase is incomplete if its linked critique item remains true.

| # | Critique finding | Required change | Acceptance evidence |
| ---: | --- | --- | --- |
| 1 | Benchmark data lives outside `serve` | Move workload models, suites, and corpora into package resources; keep optimization selection out of workload definitions | Installed `posttrain-serve` wheel runs a dry-run suite outside the monorepo |
| 2 | `common` is a repository god-package | Delete root discovery, global paths, YAML resolver, durable run store, and Trackio integration; retain only shared identities and protocols | Import graph proves `common` has no TRL/Verifiers/vLLM/Trackio/YAML dependency |
| 3 | Behavior lives in CLIs | Introduce public operation functions and move parsing/presentation to thin CLI adapters | Unit/integration tests call public APIs directly; CLI tests only cover translation |
| 4 | Domain concepts are arbitrary dictionaries | Add typed public requests, results, policies, profiles, workload shapes, and artifact references; keep native backend config typed but backend-specific | Invalid combinations fail during request construction, before GPU/model loading |
| 5 | Capabilities construct Trackio runs | Add observer/execution-context protocols and implement Trackio only in `apps/lab` | Each package runs with `NullObserver`, test observer, or Trackio observer unchanged |
| 6 | Local run hierarchy duplicates Trackio | Replace durable local runs with scoped scratch/recovery workspaces and required artifact promotion | Deleting scratch after success does not remove any durable evidence or lineage |
| 7 | “Concurrency” is an offline batch | Split offline batch and online request-concurrency modes with different request/result types | Online tests record one trace per request and report TTFT/tail latency under c1/c2/c4 |
| 8 | Suite CLI shells into benchmark CLI | Add in-process suite service and an explicit process-isolation executor strategy | Suite tests invoke services directly; subprocesses are optional cleanup boundaries |
| 9 | Eval rejects non-vLLM engines | Depend on `GenerationClient`/endpoint contracts; isolate vLLM in serve | The same eval program passes against a fake client and an OpenAI-compatible endpoint |
| 10 | Aggregation ownership is unclear | Persist direct observations; move deterministic cross-trace/run/job derivations to versioned report calculators | Recomputing a calculator from the same evidence yields the same result and version |
| 11 | Reports imports Trackio SQLite internals | Introduce `EvidenceRepository`; confine Trackio/Turso physical queries to one adapter | Calculators pass against in-memory fixtures and Trackio adapter contract tests |
| 12 | `train` is empty and heavyweight | Add SFT/DPO/GRPO public operations; move Torch/TRL stacks behind extras and lazy adapters | Base wheel imports without Torch; each backend extra proves one operation smoke |
| 13 | Package names collide | Rename distributions to `posttrain-*` and imports to PEP 420 `posttrain.*` | Clean environment imports no top-level `common`, `train`, `eval`, `serve`, or `reports` |
| 14 | Tests and tooling do not guard behavior | Standardize pytest, Ruff, Pyright, coverage, import-linter, wheel tests, and CI matrices | Pytest discovers trace-sync tests; CI fails on boundary, type, lint, or coverage regressions |

## Supported product workflows

The architecture is accepted only if it supports these workflows without a new
framework abstraction for each job.

### Workflow A — Build reusable foundation evidence

1. Add an immutable instruction/thinking model profile.
2. Reuse an existing compatible renderer and serve profile where possible.
3. Run selected serving workload cells on known hardware.
4. Run a selectable subset of general Verifiers programs.
5. Store model facts, direct evidence, and compatibility outcomes in Trackio.
6. Make the profile discoverable to later finetuning jobs.

This workflow is independent of a customer/domain job. It grows a reusable pool
of foundation profiles and evidence whenever a new model is released.

### Workflow B — Select a model for a finetuning job

1. Define a mandatory job with capability, latency, throughput, memory, context,
   and concurrency requirements.
2. Select several immutable foundation or promoted model artifacts.
3. Reuse existing evidence only when model revision, serve configuration, and
   execution context remain comparable.
4. Fill missing serve/general-eval cells rather than rebuilding all profiles.
5. Run the domain environment and select one model based on the job's policy.

The platform records evidence; the job's Python code owns the selection rule.

### Workflow C — Branch through post-training

1. SFT consumes the selected model artifact and renderer-built demonstrations.
2. Promote the selected SFT output as a model artifact.
3. Re-evaluate domain capability and a general-regression subset.
4. Start DPO and GRPO from the same SFT artifact or start another branch from
   any prior model artifact.
5. Re-run domain evaluation and required serving cells for each descendant.
6. Query the complete lineage without deriving it from directory names.

### Workflow D — Improve one reusable subsystem independently

- A serving owner can add a vLLM configuration or kernel variant and benchmark
  existing model artifacts without modifying jobs or eval code.
- An evaluation owner can add a general program or environment package and run
  it against existing endpoints without modifying serve.
- A training owner can improve SFT/DPO/GRPO adapters and test them against model
  artifacts without importing Trackio or job orchestration.
- A reports owner can add derived views without changing the evidence emitted by
  train/eval/serve.

## Goals and acceptance criteria

- Installable namespace packages expose `posttrain.common`,
  `posttrain.train`, `posttrain.eval`, `posttrain.serve`, and
  `posttrain.reports`.
- Reusable packages work without Trackio; the lab application injects Trackio
  observation through a shared execution context.
- Qwen3.5-2B and `LiquidAI/LFM2.5-1.2B-Thinking` have immutable foundation
  profiles, renderer behavior, and validated vLLM serving profiles.
- Native Verifiers v1 supports upstream `gsm8k-v1` and a separately
  publishable AutomationBench v1 environment adapter.
- SFT, DPO, and GRPO produce branchable model artifacts with explicit lineage.
- A reference job proves `base -> SFT -> {DPO, GRPO}` and evaluates/benchmarks
  every produced model version.
- Trackio is the only durable run-evidence store. Local execution directories
  are temporary and may be deleted after required artifacts are uploaded.
- Non-GPU code passes Ruff, Pyright, pytest, package-boundary tests, wheel
  installation tests, and at least 85% coverage. GPU acceptance runs are
  explicit and reproducible rather than part of ordinary CI.

## Target architecture

```text
apps/lab                         code-defined jobs and composition root
packages/common                  posttrain.common
packages/train                   posttrain.train
packages/eval                    posttrain.eval
packages/serve                   posttrain.serve
packages/reports                 posttrain.reports
environments/automationbench_v1  independently publishable Verifiers package
```

### Detailed target tree

```text
pyproject.toml
apps/
  lab/
    pyproject.toml
    src/posttrain_lab/
      cli.py
      execution.py                 # job invocation and action attempts
      jobs/
        foundation_screening.py
        gsm8k_posttraining.py
      tracking/
        trackio_observer.py
        artifact_promotion.py
      policies/
        selection.py               # job-owned model selection examples

packages/
  common/
    pyproject.toml                 # distribution: posttrain-common
    src/posttrain/common/
      __init__.py
      artifacts.py                 # immutable references and produced artifacts
      execution.py                 # context, observer, cancellation
      jobs.py                      # identity only; no orchestration
      models.py                    # foundation profiles and model facts
      errors.py
      profiles/
        qwen35.py
        lfm25.py

  serve/
    pyproject.toml                 # distribution: posttrain-serve
    src/posttrain/serve/
      __init__.py                  # stable public operation exports
      api.py
      requests.py
      results.py
      clients.py                   # generation endpoint/client contracts
      profiles/
        base.py
        qwen35.py
        lfm25.py
      benchmarks/
        workloads.py
        planner.py
        offline.py
        online.py
        metrics.py
        traces.py
        resources/
          suites/
          corpora/
      backends/vllm/
        engine.py
        server.py
        compatibility.py
      system/cuda.py
      cli.py

  eval/
    pyproject.toml                 # distribution: posttrain-eval
    src/posttrain/eval/
      __init__.py
      api.py
      requests.py
      results.py
      programs/
        general_smoke.py
      backends/verifiers/
        adapter.py
        clients.py
        traces.py
        synchronization.py
      cli.py

  train/
    pyproject.toml                 # distribution: posttrain-train
    src/posttrain/train/
      __init__.py
      api.py
      requests.py
      results.py
      data/
        sft.py
        preferences.py
      renderers/
        registry.py                # typed selection, not a global model registry
        validation.py
      checkpoints/
        recovery.py
        promotion.py
      profiles/
        base.py
        qwen35.py
        lfm25.py
      backends/trl/
        common.py
        sft.py
        dpo.py
        grpo.py
        verifiers_bridge.py

  reports/
    pyproject.toml                 # distribution: posttrain-reports
    src/posttrain/reports/
      __init__.py
      evidence.py
      repository.py
      calculators/
        serving.py
        evaluation.py
        training.py
        lineage.py
      views/
        jobs.py
        models.py
      backends/trackio.py
      cli.py

environments/
  automationbench_v1/
    pyproject.toml
    src/automationbench_v1/
      environment.py
      taskset.py
      harness.py
      rewards.py
      tools.py
```

Module names may change during implementation only when the responsibility and
import boundary remain identical; such changes must be recorded in the Decision
Log.

### Package responsibilities

| Package | Owns | Must not own |
| --- | --- | --- |
| `common` | Model/artifact identities, foundation profiles, job/action identity, `ExecutionContext`, observer protocol | TRL, Verifiers, vLLM, Trackio, workspace-root globals |
| `train` | SFT/DPO/GRPO operations, training configs, renderers, checkpoint/resume semantics, TRL adapters | Job orchestration, Trackio persistence, serving |
| `eval` | Endpoint-neutral evaluation operation, Verifiers v1 adapters, general evaluation programs, native result bundles | vLLM-specific launch logic, job policy |
| `serve` | vLLM launch/generate/probe/benchmark operations, workload resources, model/backend performance profiles | Capability scoring, training, Trackio queries |
| `reports` | Stable evidence repository, derived run/job/trace views, comparisons | Direct public dependence on Trackio private SQLite/Turso tables |
| `apps/lab` | Code-defined jobs, dependency composition, Trackio observer, lifecycle policy | Reimplementation of train/eval/serve engines |
| Environment packages | Tasksets, tasks, harness/runtime wiring, tools, rewards, trace semantics | Platform orchestration and Trackio SDK usage |

### Public operation contracts

- `posttrain.common`: `ModelProfile`, `HubModelRef`, `TrackioArtifactRef`,
  `LocalArtifactRef`, `Job`, `JobAction`, `ExecutionContext`, `Observer`, and
  `ProducedArtifact`.
- `posttrain.train`: `sft(SFTRequest)`, `dpo(DPORequest)`, and
  `grpo(GRPORequest)`, each returning `TrainingResult`.
- `posttrain.eval`: `evaluate(EvaluationRequest) -> EvaluationResult` against a
  model client/endpoint contract rather than a vLLM-aware CLI.
- `posttrain.serve`: `launch`, `generate`, `probe`, and `benchmark`, returning
  typed results while preserving backend-native details as artifacts.
- `posttrain.reports`: an `EvidenceRepository` interface plus versioned pure
  calculators for run, trace, model-lineage, and job views.

Requests contain operation-native typed configuration. There is no universal
experiment YAML or framework-hiding config schema. Python definitions are the
source of truth and every run records their resolved values and Git identity.

### Core type invariants

#### Model and artifact identity

- `HubModelRef(repo_id, revision)` requires a commit SHA; branch names and tags
  are rejected for execution.
- `TrackioArtifactRef(project, name, version_or_alias)` resolves an alias to an
  immutable version before an operation starts and records both values.
- `LocalArtifactRef(path, digest)` is allowed for work in progress but cannot be
  used as durable cross-job lineage until promoted.
- `ModelProfile` contains immutable model facts: identity, family, parameter
  count, modalities, native context, reasoning modes, MTP availability,
  renderer selection, and known compatibility declarations.
- Ordinary checkpoints and adapters are lineage artifacts, not new checked-in
  model profiles. A selected descendant gets a reusable profile only when it is
  intentionally promoted for use across jobs.

#### Execution identity

- `Job.id` is stable and mandatory; `Job.version` identifies the Python job
  definition or Git revision.
- `JobAction.id` is stable within a job and expresses intent, not an attempt.
- `Invocation.id` identifies one parameterized request to an action.
- `RunAttempt.id` identifies an execution attempt. Retrying an invocation makes
  another run while preserving job/action/invocation identity.
- `ExecutionContext` contains identities, observer, cancellation token, scratch
  workspace, clock, and resolved source metadata. It contains no Trackio type.

#### Operation requests and results

- Requests are immutable after validation and carry a fully typed native config
  for their operation.
- Results contain outcome status, direct metrics, traces or trace references,
  produced artifacts, native-output artifacts, and warnings. They never create
  or finalize a Trackio run themselves.
- Backend-native settings remain under explicit types such as
  `VllmEngineConfig` and `TrlGRPOConfig`; there is no `dict[str, Any]` escape at
  public boundaries. Narrow JSON mappings are permitted only for preserved
  upstream payloads and are labeled as such.

### Profile and configuration ownership

| Definition | Owner | Reusable scope | Excludes |
| --- | --- | --- | --- |
| Foundation model profile | `common` | Cross-job immutable model facts | Workload, concurrency, job thresholds |
| Serve profile | `serve` | Backend/model-family configuration such as context, MTP, TurboQuant, kernels | Benchmark request shape and job selection policy |
| Training profile | `train` | Technique/model-family defaults for SFT, DPO, GRPO | Dataset choice, lifecycle order, Trackio |
| General eval program | `eval` | Reusable category/environment selection | Domain thresholds and job acceptance policy |
| Environment package | Independent package | Taskset/harness/runtime/reward semantics | Platform orchestration |
| Job definition | `apps/lab` or downstream app | One finetuning objective | Reusable backend implementation |

Hardware facts and workload shapes are recorded execution inputs, not global
profiles. A serve profile may declare compatibility constraints, but it does not
pretend that one benchmark result applies to every device or concurrency level.

### Dependency and extras strategy

| Distribution | Base install | Optional runtime extras |
| --- | --- | --- |
| `posttrain-common` | Standard-library/shared typing dependencies only | None |
| `posttrain-serve` | Common types and package resources | `[vllm]` contains Torch, CUDA-aligned vLLM 0.25.1, compiler components |
| `posttrain-eval` | Common types and endpoint contracts | `[verifiers]` contains pinned Verifiers v1 and environment clients |
| `posttrain-train` | Common types, request/result definitions | `[trl]`, `[trl-vllm]`; heavy Torch/TRL/PEFT dependencies live here |
| `posttrain-reports` | Evidence models and pure calculators | `[trackio]` contains the Trackio repository adapter |
| `posttrain-lab` | Composition code | Selects compatible package extras for each job environment |

The uv workspace declares incompatible extras explicitly and CI tests each
supported variant separately. A single developer environment is not required to
contain every GPU backend. Lock generation must nevertheless prove all declared
variants resolve from one committed lockfile.

### Pinned MVP baseline

These pins define the first acceptance matrix; changing one requires a Decision
Log entry and rerunning the affected package and GPU contracts.

| Component | MVP identity |
| --- | --- |
| Qwen foundation | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` |
| LFM foundation | `LiquidAI/LFM2.5-1.2B-Thinking@95053d21d8e0b7ca99421a2127ae39c64f685ff3` |
| Verifiers v1 | `PrimeIntellect-ai/verifiers@284a868d6a9022109b749710672a0460e8a996d4` |
| GSM8K environment | `environments/gsm8k_v1` at the same Verifiers revision |
| AutomationBench source | Hub package `zapier/automationbench` version `1.0.5`, wrapped by the native-v1 package |
| TRL | `carbonteq-ai/trl@72e5d176bf0820621759107aa9699bf4aed43396`, package version 1.8.0 |
| vLLM | `0.25.1` with the current CUDA 13/PyTorch lock |
| Trackio | `carbonteq-ai/trackio@02351d871050bf4b3505c7371239c698b710ec83` until the fork work is advanced |

## Jobs, runs, evidence, and lineage

- A **job** is the durable, code-defined finetuning objective and is mandatory.
- A **job action** is one logical lifecycle step such as baseline serving,
  general evaluation, SFT, or checkpoint evaluation.
- A **run** is one execution attempt of an action. Retries and repeated seeds
  create multiple runs under the same action.
- Trackio `project` separates global reusable evidence from product/domain job
  evidence. Trackio `group` is the stable job ID; run config contains action and
  invocation IDs.
- Direct observations are persisted: losses, rewards, latency samples,
  throughput, memory, errors, and trace-level evaluator results.
- Run/job aggregates and comparison views are computed by `reports`; a report
  may be materialized as a versioned artifact, but derived values are not copied
  into a second custom metrics database.
- A base Hugging Face revision is an immutable external model reference. A
  selected training output is a Trackio model artifact. Training, evaluation,
  and serving runs consume the same artifact identity, giving lineage such as
  `base -> SFT -> {DPO, GRPO}` without path-based inference.

### Observation levels

| Level | Persisted directly | Computed later |
| --- | --- | --- |
| Trace/request | Prompt/output token counts, timing events, reward components, verifier results, error/truncation, native trace payload | Trace classification or presentation projections that can be reproduced |
| Run attempt | Trainer loss steps, GPU memory samples, direct throughput, error status, consumed/produced artifacts | Percentiles, normalized scores, regressions, comparable-cell summaries |
| Job/action | Identity and relationships to runs | Best attempt, branch comparisons, capability rollups, selection views |
| Model lineage | Produced/consumed artifact edges | Ancestor/descendant views and cross-stage comparisons |

Trackio stores the direct observations and explicitly materialized report
artifacts. Reports owns calculator name, version, input selection, and output.
No component writes a second normalized result database.

### Finalization and failure semantics

1. The host creates the Trackio run and scratch workspace, then constructs the
   `ExecutionContext`.
2. The operation emits events but remains unaware of Trackio availability.
3. Recovery checkpoints may remain local while a run is active; selected outputs
   and native result bundles are required durable artifacts.
4. The host marks success only after all required artifacts and direct metrics
   are committed.
5. A required artifact failure makes the run fail even when model execution
   completed.
6. Streaming trace synchronization may be partial only if the complete native
   trace bundle is uploaded and synchronization status/error metrics are stored.
7. Retrying finalization is idempotent by run, trace type, and external trace ID.
8. Scratch is deleted after successful finalization. Failed scratch may be kept
   temporarily under an explicit retention policy, never indexed as a run store.

## Developer experience

### Use a reusable operation without the lab

```python
from posttrain.serve import BenchmarkRequest, benchmark

result = benchmark(BenchmarkRequest(model=model, workload=workload))
```

No workspace root, YAML tree, Trackio project, or local run registry is needed.

### Compose a tracked job

```python
from posttrain_lab import Lab
from posttrain.train import SFTRequest, sft

job = gsm8k_posttraining_job()
with Lab.trackio().attempt(job, action="sft", invocation=inputs) as context:
    result = sft(SFTRequest(model=selected_model, data=data, config=config), context=context)
```

The host owns run creation and artifact promotion. Calling `sft()` directly is
still valid and produces the same `TrainingResult`.

### Add a new foundation model

1. Add one immutable `ModelProfile` in `posttrain.common.profiles`.
2. Reuse or add a renderer validation in `train`.
3. Extend a compatible serve profile and validate only changed assumptions.
4. Run a general-eval subset and required serve cells through a foundation job.
5. Do not create a descendant profile for every checkpoint.

### Add a domain environment

1. Create or depend on an independently publishable Verifiers v1 package.
2. Keep dataset/taskset, tools, harness/runtime, rewards, and trace semantics in
   that package.
3. Reference it from job code and optionally from a reusable eval program.
4. Never add a reward callback or environment-specific schema to `common`.

## Implementation phases

### Phase 0 — Establish a safe baseline

- Restore or initialize valid Git metadata for this workspace; the current
  `.git` directory is empty, so the revamp cannot yet be committed safely.
- Record the current test baseline and dependency variants.
- Treat the merged TRL fork and current Trackio fork as isolated prerequisites,
  then verify their immutable pins from clean environments.
- Add CI jobs for lightweight packages and separate opt-in environments for
  mutually incompatible GPU/framework extras.

**Exit:** the repository has real history/remotes, a reproducible lock, and a
green baseline from which each migration slice can be reviewed.

### Phase 1 — Namespace and shared contracts

- Rename distributions to `posttrain-*` and imports to the PEP 420
  `posttrain.*` namespace.
- Implement the shared identities, typed artifact references, operation result
  envelope, cancellation/temporary-workspace behavior, and observer protocol.
- Move the two immutable foundation model definitions into `posttrain.common`.
- Add import-boundary tests ensuring `common` imports no execution frameworks or
  observability backend.
- Create `apps/lab` as the composition root and a thin CLI for invoking
  code-defined jobs.

**Exit:** every package builds as a wheel and a sample no-op job can execute an
operation with or without a Trackio observer.

### Phase 2 — Serving vertical slice

- Replace the anemic CLI-oriented implementation with typed `launch`,
  `generate`, `probe`, and `benchmark` services around an internal vLLM adapter.
- Move benchmark corpora, prompt shapes, chat-template settings, and workload
  definitions under `posttrain.serve` package resources.
- Separate offline batch throughput from online concurrency/latency tests.
- Support concurrency 1, 2, and 4 on this machine; retain portable support for
  8 on larger hardware.
- Cover short interactive, balanced, output-heavy, prefill-heavy, and context
  windows through 32K. Use the TurboQuant K8 profile for compatible 32K cells.
- Keep MTP and TurboQuant as typed model/serve-profile choices. Remove SGLang
  code, dependency, and support claims from the MVP.

**Exit:** both foundation profiles complete comparable vLLM smoke cells and
emit typed request traces, direct metrics, native output, and Trackio artifacts.

### Phase 3 — Evaluation and environments vertical slice

- Replace shelling out to Verifiers with its native v1 Python API.
- Implement endpoint-neutral `evaluate`; model clients determine whether the
  runtime is vLLM, an in-process model, or a future backend.
- Package general evaluation programs by category so they can be reused for
  new model releases independently of a finetuning job.
- Consume upstream `gsm8k-v1` at an immutable revision.
- Create `environments/automationbench_v1` as a thin, native-v1 adapter over a
  pinned AutomationBench release, reusing its tasks, tools, world state, and
  assertions while owning v1 taskset/harness/runtime wiring.
- Preserve the complete native Verifiers result bundle as an artifact and
  stream specialized traces to Trackio idempotently.

**Exit:** both models run a general-eval subset, GSM8K, and a small
AutomationBench smoke with trace-level metrics, errors, truncation, and native
artifacts retained.

### Phase 4 — Renderer-based SFT and DPO

- Define renderer specs in `train`: use explicit Qwen3.5 renderer support and a
  validated default renderer configuration for LFM2.5.
- Add golden tests for prompt token IDs, generation prefixes, stop tokens,
  thinking controls, assistant loss masks, and chosen/rejected prompt equality.
- Build SFT samples from GSM8K reference demonstrations through renderer token
  IDs and loss masks.
- Build DPO pairs deterministically: the reference solution is chosen; a
  completed, non-error, non-truncated rollout with lower verified reward is
  rejected. Fail clearly when the requested pair count cannot be constructed.
- Isolate any pre-tokenized DPO behavior behind a version-pinned internal TRL
  adapter rather than leaking a trainer subclass into the public API.
- Use NF4 double-quantized QLoRA, BF16, gradient checkpointing, and `all-linear`
  LoRA as the local 8 GB smoke baseline. Kernel/Liger variants are promoted per
  model and operation only after numerical parity tests.

**Exit:** each foundation profile completes a two-step SFT smoke and a two-step
DPO smoke, produces a model artifact, and can be resumed from a recovery
checkpoint without confusing recovery state with a promoted model version.

### Phase 5 — Verifiers-to-TRL GRPO bridge

- Implement a Verifiers v1 `Client` backed by the already-loaded TRL policy so
  the default 8 GB path does not duplicate weights or require an HTTP server.
- Convert one trainable Verifiers trace branch into TRL rollout outputs:
  `prompt_ids`, `completion_ids`, sampling logprobs, rewards, and `env_mask`.
- Mark model-produced tokens trainable and harness/tool/user tokens masked.
- Preserve the complete Verifiers trace for observation.
- Serialize access to the model, restore train/eval mode reliably, propagate
  cancellation, and make retries idempotent.
- Reject unsupported multi-trainable-branch traces explicitly in the MVP.
- Expose separate rollout profiles: in-process Transformers as the local
  default, pinned vLLM 0.25.1 colocation as an optimized option, and vLLM server
  mode for future larger machines.

**Exit:** both profiles complete a one-step GSM8K GRPO smoke with two
generations, aligned token/logprob lengths, recorded Verifiers traces, and no
second policy load in the default path.

### Phase 6 — Observation, reports, and reference lifecycle

- Implement the Trackio observer in `apps/lab`; reusable packages only emit
  typed lifecycle, metric, trace, and artifact events.
- Enforce mandatory job/action/run identity, resolved-input capture, produced
  and consumed artifact relationships, and finalization rules.
- A required artifact upload failure fails the run. Trace streaming may finish
  partially only when the native bundle is retained and the synchronization
  state is recorded.
- Put all Trackio physical-schema access behind the reports repository adapter;
  calculators operate on stable evidence models.
- Add code-defined foundation-screening and GSM8K post-training jobs.
- Execute the reference branch:

  ```text
  immutable base
    -> serving + general eval + GSM8K
    -> SFT artifact
       -> serving + GSM8K
       -> DPO artifact -> serving + GSM8K
       -> GRPO artifact -> serving + GSM8K
  ```

**Exit:** reports can reconstruct the complete lineage and compare direct and
derived evidence without reading a shared `runs/` directory.

### Phase 7 — Delete the prototype and harden quality

- Audit existing local `runs/`; upload any evidence required by the documented
  smoke results, then remove the directory.
- Delete YAML profile resolver/config trees, workspace-root globals, local run
  persistence, shell-to-shell runners, obsolete entrypoints, root benchmark
  data, SGLang claims, and compatibility shims superseded by the new slices.
- Remove old package names only after all internal imports and docs use
  `posttrain.*`; do not provide backwards-compatible aliases.
- Enforce Ruff formatting/lint, Pyright on public/core APIs, pytest discovery,
  coverage, import-linter boundaries, clean wheel installs, and separate GPU
  acceptance commands.

**Exit:** no legacy path is required by tests, docs, jobs, or runtime; the
reference lifecycle passes from a clean clone.

## Current-to-target migration inventory

This table is the deletion checklist. “Replace” means the old path is removed in
the same reviewed slice that introduces the new owner.

| Current path/surface | Target owner | Disposition |
| --- | --- | --- |
| `benchmarks/inference/suites/core.yaml` | `posttrain.serve.benchmarks.resources` plus typed workload definitions | Translate workload-only fields, remove TurboQuant policy from suite, then delete YAML/root path |
| `benchmarks/inference/corpora/representative-v1.jsonl` | Serve package resource | Move with schema/version loader and wheel-resource test |
| `profiles/models/*.yaml` | `posttrain.common.profiles` | Recreate as typed immutable Python profiles, golden-compare resolved facts, delete YAML |
| `profiles/serve/vllm/*.yaml` | `posttrain.serve.profiles` | Recreate as typed vLLM configs/variants; separate workload defaults; delete YAML |
| `profiles/eval/general-smoke-v1.yaml` | `posttrain.eval.programs.general_smoke` | Recreate as code-defined program referencing pinned environment packages |
| `profiles/train/README.md` placeholder | `posttrain.train.profiles` | Implement actual SFT/DPO/GRPO typed defaults; remove root profile tree |
| `common/__init__.py` workspace paths | No reusable package equivalent | Delete root discovery and exported repository paths |
| `common/profiles.py` and `profile_cli.py` | Owning package profile modules | Delete generic inheritance/path resolver and profile CLI |
| `common/runs.py` | `posttrain_lab.execution` plus temporary workspace helper | Delete durable local run hierarchy; retain explicit recovery/scratch behavior only |
| `common/tracking.py` | `posttrain_lab.tracking.trackio_observer` | Move Trackio run lifecycle and artifact promotion to host; common keeps protocol only |
| `serve/benchmark.py` | Serve API, benchmark domain services, vLLM adapter, thin CLI | Decompose responsibilities; do not retain a second legacy entrypoint |
| `serve/suite_cli.py` | Benchmark planner/executor and thin CLI | Replace CLI-to-CLI subprocess composition with service calls and optional isolation runner |
| `serve/suites.py` and `prompts.py` | `serve.benchmarks` models/resources | Preserve good immutable records while moving them behind public operations |
| `serve/cuda.py` | `serve.system.cuda` | Retain isolated toolkit behavior and add adapter contract tests |
| `serve/vllm_compat.py` | `serve.backends.vllm.compatibility` | Retain narrow state/version guards; delete patches when upstream invariant is present |
| `eval/cli.py` | Eval API, Verifiers adapter, thin CLI | Remove vLLM profile check, subprocess orchestration, Trackio construction, and aggregation |
| `eval/suites.py` | Eval programs and typed requests | Replace arbitrary mappings/YAML loading with code-defined program composition |
| `eval/results.py` | Eval execution summary plus report calculators | Keep only execution-critical direct summary; move derived rollups to reports |
| `eval/trace_sync.py` | `eval.backends.verifiers.synchronization` | Preserve partial-line, batching, retry, and dependency-injection behavior; use observer protocol |
| `eval/runtime/*` | Environment/runtime package or eval Verifiers adapter | Move only if required by a program; delete null warmup as a platform concept |
| `reports/query.py` | `EvidenceRepository` and Trackio adapter | Hide SQLite/Turso/private schema imports inside adapter; pure calculators see stable evidence |
| Empty `train/__init__.py` | Public SFT/DPO/GRPO exports | Implement real capability surface before accepting heavyweight extras |
| `jobs/README.md` | `apps/lab/src/posttrain_lab/jobs` | Replace prose-only placeholder with executable reference jobs |
| Root `tests/` | Package-local tests plus `tests/integration` | Move tests to owners, stop importing private helpers, retain cross-package lifecycle tests centrally |
| Root `runs/` if present | Trackio artifacts/metrics/traces | Audit required evidence, upload missing artifacts, then delete |

## Delivery units and review boundaries

Each unit should be reviewable, green, and deletive. Avoid a long-lived branch
that introduces the new architecture while leaving the old system active.

### Delivery 0 — Repository and quality bootstrap

**Adds**

- valid Git history/remote configuration;
- root `dev` dependency group with pytest, pytest-asyncio, Ruff, Pyright,
  coverage, and import-linter;
- CI workflow with core, package-extra, wheel, and documentation checks;
- test markers for `gpu`, `network`, and `docker`.

**Proves**

- `pytest` discovers the trace synchronization tests previously skipped by
  `unittest`;
- the existing non-GPU baseline is green from a clean environment;
- dependency variants resolve independently from the committed lock.

### Delivery 1 — Shared contracts and namespaced wheels

**Adds**

- the `posttrain.*` namespace and renamed distributions;
- immutable artifact/model/job identities and operation result envelope;
- `ExecutionContext`, observer, cancellation, clock, and scratch protocols;
- import-linter contracts and independent wheel install tests;
- minimal `apps/lab` composition root and Trackio observer skeleton.

**Removes**

- old generic imports as soon as each internal consumer migrates;
- `WORKSPACE_ROOT` and exported repository-global directories;
- Trackio from common's dependency list.

**Proves**

- a fake operation executes with no observer, a recording observer, and Trackio;
- `posttrain-common` imports in an environment without Torch, Trackio, YAML, or
  the monorepo checkout.

### Delivery 2 — Self-contained serving capability

**Adds**

- typed serving/model configs and public operation functions;
- package-owned benchmark resources and schema versioning;
- separate offline and online benchmark implementations;
- request traces, warmup exclusion, per-request errors, TTFT, inter-token
  latency, end-to-end latency, throughput, and memory samples;
- explicit runner strategies: in-process reuse and isolated subprocess cleanup;
- reusable Qwen/LFM standard, TurboQuant K8V4, and Qwen native-MTP variants.

**Removes**

- root benchmark data;
- CLI-to-CLI subprocess construction;
- SGLang extra and implemented-support claims;
- workload-to-TurboQuant mapping from suite data.

**Proves**

- a built wheel plans a suite from package resources outside the repository;
- a fake backend validates online scheduling semantics deterministically;
- local GPU smokes complete at c1/c2/c4, and a 32K compatible cell uses the
  explicitly selected TurboQuant serve variant.

### Delivery 3 — Endpoint-neutral eval and native environments

**Adds**

- `GenerationClient` contract and native Verifiers adapter;
- typed code-defined general programs;
- upstream GSM8K integration and AutomationBench native-v1 package;
- native result bundle production and observer-based trace streaming;
- trace metrics at the trace level, not flattened into custom run facts.

**Removes**

- eval's vLLM profile lookup/rejection;
- Verifiers subprocess/CLI dependency where the Python API is available;
- YAML eval suite as the canonical program definition;
- report-style cross-trace aggregation from eval.

**Proves**

- the same program runs against a fake client and a real compatible endpoint;
- AutomationBench scripted trajectories preserve world-state transitions,
  tool filtering, assertions, partial credit, and final rewards;
- trace tailing handles partial lines, retry, resume, interruption, and duplicate
  external IDs.

### Delivery 4 — SFT and DPO capability

**Adds**

- public typed operations and TRL adapters;
- renderer selection/validation for Qwen and LFM;
- reference-demonstration SFT builder and deterministic preference builder;
- QLoRA/BF16/gradient-checkpointed local profiles;
- recovery checkpoint versus promoted artifact semantics.

**Removes**

- Trackio from train dependencies and trainer lifecycle;
- any use of a universal trainer config or raw dataset schema at public APIs.

**Proves**

- renderer goldens and loss masks are stable;
- two-step SFT and DPO smokes update only expected adapter parameters;
- an interrupted run resumes from recovery while a promoted output is a new
  immutable artifact with a producer edge.

### Delivery 5 — GRPO environment bridge

**Adds**

- in-process Verifiers `Client` over the loaded TRL policy;
- trace-to-rollout conversion, reward propagation, token masks, and logprobs;
- explicit unsupported-trace-shape error;
- optional vLLM colocate/server generation profiles using the pinned fork.

**Proves**

- the default path has one policy model object/load;
- model mode and GPU lock are restored on success, cancellation, and exception;
- completion IDs, logprobs, reward vectors, and masks have exact aligned lengths;
- a one-step, two-generation GSM8K GRPO smoke produces an adapter artifact and
  full Verifiers traces for both foundation profiles.

### Delivery 6 — Reports and reference jobs

**Adds**

- stable evidence records and repository protocol;
- Trackio adapter isolated from calculators/views;
- code-defined foundation-screening and post-training jobs;
- calculators for comparable serving cells, capability/regression views, and
  model lineage.

**Proves**

- multiple attempts appear under one action without overwriting each other;
- `base -> SFT -> {DPO, GRPO}` is reconstructed only from artifact edges;
- calculators return identical output for Trackio and fixture repositories;
- the complete reference lifecycle works from a clean clone.

### Delivery 7 — Final deletion and release readiness

**Removes** every legacy path in the migration inventory and updates every
architecture, functional, tooling, and handoff document to the implemented
state.

**Proves**

- searching for old imports, workspace globals, YAML resolver calls, durable
  local run APIs, SGLang dependencies, and CLI-to-CLI execution returns no
  runtime consumers;
- every distribution installs independently;
- downstream example code can use operations without `apps/lab`;
- the final quality score is reassessed against the original fourteen findings.

## Explicit non-goals for this refactor

- A generic DAG scheduler, distributed workflow engine, or custom experiment
  configuration language.
- A second artifact registry or normalized results database outside Trackio.
- SGLang, Triton custom-kernel development, or automatic multi-backend routing.
- Rust/Tokio/Doris rewrites of Trackio, production deployment promotion,
  approvals, RBAC, or governance workflows.
- A custom reports frontend or complete Verifiers trace frontend redesign.
- Supporting every model family, multimodal training, or 32K training on the
  local 8 GB GPU.
- Materializing every derived metric. Derived comparisons remain versioned
  calculations unless a report artifact is intentionally published.

## Test and validation matrix

| Surface | Required validation |
| --- | --- |
| Contracts | Serialization, validation, cancellation, temporary cleanup, observer absent/present |
| Packaging | Build and install every wheel independently from a temporary directory |
| Boundaries | `common` framework-free; `eval` vLLM-free; packages do not import `apps/lab` |
| Renderers | Golden IDs, stops, loss masks, thinking behavior, DPO prompt equality for both models |
| Verifiers | Single/multi-turn, tool use, error, truncation, reward metrics, native trace preservation |
| GRPO bridge | Same loaded model, mode restoration, logprob alignment, `env_mask`, branch rejection, retry/cancel |
| Trackio | Multiple runs per action, artifact lineage, idempotent traces, required-artifact failure semantics |
| Reports | Deterministic derived metrics and lineage views over repository fixtures |
| Serving | Offline vs online semantics, input/output-heavy shapes, context limits, concurrency 1/2/4 |
| GPU lifecycle | Both models: base serve/eval, SFT, DPO, GRPO, and descendant serve/eval smokes |

## Migration rules

- Implement one vertical slice at a time and delete the replaced prototype in
  the same slice.
- Do not create compatibility wrappers for old imports, YAML files, CLIs, run
  directories, or result schemas.
- Preserve user data and required Trackio evidence, not obsolete code shape.
- Update this plan whenever implementation reveals a changed assumption; do
  not hide deviations in code or a disconnected handoff note.

## Progress

- [x] Product lifecycle and reusability opportunities documented.
- [x] Target functional and architectural documents drafted.
- [x] Original 3/10 critique mapped to target owners, migration tasks, and acceptance evidence.
- [x] Trackio fork and Verifiers trace direction established.
- [x] TRL 1.8 fork with vLLM 0.25.1 support merged and GPU-smoked.
- [x] Phase 0 — local Git/reproducibility and quality baseline; remote publication is deferred.
- [ ] Phase 1 — namespace and shared contracts.
  - [x] Add independently installable `posttrain-common` contracts and pinned foundation profiles.
  - [x] Add `posttrain-lab` composition root, ephemeral attempt host, Trackio observer, and no-op job.
  - [x] Migrate train/eval/serve/reports distributions and imports to `posttrain.*`.
  - [ ] Remove the legacy `common` surface as each vertical slice stops consuming it.
- [x] Phase 2 — serving vertical slice.
  - [x] Replace serving profile and workload YAML as the source of truth with typed Python definitions.
  - [x] Remove SGLang and its incompatible dependency branch from the MVP lock.
  - [x] Diagnose and correct the Qwen3.5-2B 8 GiB text-only startup profile with a real GPU run.
  - [x] Move offline benchmark execution behind a typed, observer-neutral operation.
  - [x] Add the first code-defined foundation-screening serving action.
  - [x] Move the representative corpus into the installable serve wheel and verify it from an isolated environment.
  - [x] Delete the legacy serving YAML/Trackio CLI path after the tracked operation GPU smoke.
  - [x] Add managed online launch, probe, and streaming generate operations with deterministic HTTP tests.
  - [x] Prove the online endpoint on GPU and add deterministic bounded-concurrency scheduling tests.
- [ ] Phase 3 — evaluation and environment vertical slice.
  - [x] Replace the legacy CLI/YAML suite with a typed, endpoint-neutral public evaluation operation.
  - [x] Compose the pinned Verifiers v1 Python runner directly and preserve its native result directory.
  - [x] Add code-defined general and agentic programs with per-run task, rollout, and concurrency budgets.
  - [x] Consume upstream `gsm8k-v1` and stream completed JSONL traces idempotently through the observer.
  - [x] Implement and independently lock the native Verifiers v1 AutomationBench package on Python 3.13.
  - [x] Preserve upstream AutomationBench tasks, simulated APIs, final-world assertions, dense reward, and strict outcome metrics.
  - [x] Prove Qwen and LFM GSM8K cells on GPU with queryable Trackio traces and retained native artifacts.
  - [x] Prove the native Python 3.13 AutomationBench package, MCP tool server, endpoint calls, world finalization, and scoring against live Qwen.
  - [ ] Add the isolated environment worker to the lab composition API so AutomationBench traces and artifacts flow into the same tracked attempt automatically.
- [ ] Phase 4 — renderer-based SFT and DPO.
- [ ] Phase 5 — Verifiers-to-TRL GRPO bridge.
- [ ] Phase 6 — observation, reports, and reference lifecycle.
- [ ] Phase 7 — legacy deletion and quality hardening.

## Surprises & Discoveries

- The workspace contains an empty `.git` directory rather than valid repository
  metadata. This must be resolved before a broad refactor can be safely staged.
- No existing CarbonTeq GitHub repository was found for this workspace. A local
  `main` history now provides safe refactor checkpoints; creating/pushing a
  remote remains an explicit publication action outside the code refactor.
- TRL 1.8.0 originally capped vLLM at 0.23.0, while the serving environment uses
  0.25.1. The CarbonTeq fork now carries upstream-validated compatibility
  commits without moving to TRL 1.9 development APIs.
- The current uv workspace intentionally cannot install every heavy extra in one
  environment. Validation must use separate train, eval, and serve variants.
- `unittest` discovery omitted pytest-style trace synchronization tests; pytest
  is the canonical test runner going forward.
- The measured prototype coverage baseline is 56%. Delivery 0 establishes a
  55% non-regression floor rather than spending effort testing CLIs scheduled
  for deletion; each replacement delivery raises the floor toward the final 85%.
- Verifiers v1 already separates taskset, harness, runtime, client, and trace;
  the platform should compose those abstractions rather than recreate an eval
  schema or reduce environments to reward callbacks.
- Renderer-based tokenization is essential for preserving thinking controls and
  assistant loss attribution across SFT and DPO.
- A clean `posttrain-common` wheel imports with no Trackio, Torch, TRL,
  Verifiers, vLLM, or YAML installation. `posttrain-lab` is the first and only
  new package that selects the pinned Trackio adapter.
- All six new distributions build independently as wheels and compose through
  a PEP 420 `posttrain.*` namespace; the previous top-level `eval`, `reports`,
  `serve`, and `train` import packages are no longer shipped.
- Qwen3.5-2B fits without CPU offload, but the old 82%/CUDA-graph profile OOMed
  during graph/KV profiling. The explicit text-only eager profile completed at
  65.05 output tok/s, 43.1 ms TTFT, and 6.92 GiB peak VRAM; cold start remains
  65.24 seconds and is a separate optimization target.
- The first canonical Trackio serving run preserved identity, trace, and
  artifact linkage but wrote one row per scalar metric. Run-level metric batches
  are now a shared observer primitive so one result becomes one coherent metric
  observation rather than an artificial step sequence.
- The corrected canonical foundation-screening run at Git revision `5b429cb`
  contains mandatory identity, a single 23-field metric batch, an inference
  trace, and a versioned output artifact. It measured 75.66 output tok/s,
  35.5 ms TTFT, 6.99 GiB peak VRAM, and 19.51 seconds cache-warm startup.
- LFM2.5 passed the matching canonical cell at Git revision `c13df39` with
  170.90 output tok/s, 14.2 ms TTFT, 6.35 GiB peak VRAM, and 22.55 seconds
  startup. Both pinned foundations now have comparable job-owned serving
  evidence through the new operation and observation boundaries.
- Root serving YAML, the YAML benchmark matrix, and both legacy serving CLIs
  were deleted after the typed operation smokes. The representative corpus now
  loads from `posttrain.serve` package resources in an isolated wheel-only
  environment; import-linter also proves serving has no Trackio, YAML, legacy
  `common`, or lab-host dependency.
- The pinned LFM2.5 tokenizer template advertises tools but serializes an
  OpenAI assistant tool-call history as `null`. The shared model conversation
  contract now supplies a tested package template that preserves LFM's native
  Pythonic tool call; Qwen continues to use its pinned XML tokenizer template.
  vLLM parser names remain serve-profile details (`qwen3_xml`/`qwen3` and
  `lfm2`/tag-compatible `deepseek_r1`).
- The managed LFM endpoint completed a clean-revision online smoke at
  `a8c1706`: health/model probe passed, the response stopped normally with
  final content `4.`, 22 input and 169 output tokens, 168 ms TTFT, and 953 ms
  end-to-end latency. Earlier capped attempts exposed that reasoning-only
  truncation must fail the job and that a contrived exact-word prompt caused
  repetitive reasoning. The final smoke uses a natural prompt and a 512-token
  thinking budget. Raw SSE events belong in the native output artifact rather
  than searchable trace metadata.
- The canonical refined online run is
  `serve-online/lfm2.5-1.2b-thinking-95a0371d-a1` at clean revision `371a49c`.
  It records a 42 ms TTFT and 884 ms request latency, a compact projected trace,
  a 73.7 KB native streamed-response artifact, and a versioned server log.
  The package also has a bounded, order-preserving concurrent generation
  primitive with deterministic concurrency tests.

## Decision Log

- 2026-07-20: Treat the work as an MVP revamp; remove old code instead of
  preserving compatibility.
- 2026-07-20: Use `posttrain.*` namespace packages and keep foundation model
  definitions in `common`.
- 2026-07-20: Keep chat templates, reasoning modes, roles, and native tool
  grammar in the shared model profile; keep backend parser choices in serve.
- 2026-07-20: Make train/eval/serve/reports reusable operations, with jobs as
  code-defined compositions in `apps/lab`.
- 2026-07-20: Make Trackio the observability/evidence layer, not an execution or
  configuration framework.
- 2026-07-20: Use vLLM as the only MVP serving backend; defer SGLang.
- 2026-07-20: Use Verifiers v1 natively and package domain environments
  independently.
- 2026-07-20: Use Renderers for SFT/DPO and an in-process Verifiers client as
  the default GRPO rollout path on the local 8 GB GPU.
- 2026-07-20: Retain optional vLLM colocation by pinning the CarbonTeq TRL 1.8
  fork and vLLM 0.25.1.
- 2026-07-20: Validate Qwen3.5-2B and LFM2.5-1.2B-Thinking across the reference
  lifecycle before expanding the model set.
- 2026-07-20: Expanded revision 2 so every original critique finding has a
  deletion/migration target, review boundary, and acceptance test.
- 2026-07-20: Set the initial coverage ratchet to 55% against a measured 56%
  baseline; the final target remains 85% for the refactored non-GPU code.
- 2026-07-20: Raise the coverage ratchet to 60% after typed serving profiles and
  workload definitions moved measured coverage to 60.81%.
- 2026-07-20: Raise the coverage ratchet to 65% after the public serving
  operation, internal vLLM adapter, and foundation-screening action reached
  65.07% measured coverage.

## Outcomes & Retrospective

Implementation has not started beyond the dependency/observability spikes.
Complete this section after each phase with delivered behavior, validation
evidence, deviations, and follow-up decisions. At final completion, summarize
whether the reference lifecycle works from a clean clone and which abstractions
proved reusable across more than one operation or model.

## References

- `docs/functional/overview.md`
- `docs/functional/finetuning-lifecycle.md`
- `docs/architecture.md`
- `docs/architecture/layers-and-ownership.md`
- `docs/architecture/profiles-and-model-variants.md`
- `docs/architecture/evaluation-and-environments.md`
- `docs/architecture/training-and-inference.md`
- `docs/architecture/trackio.md`
- `docs/decisions/0004-lifecycle-driven-mvp-platform.md`
- `docs/decisions/0006-trackio-observation-model.md`
- `docs/decisions/0007-trl-vllm-025-fork.md`
