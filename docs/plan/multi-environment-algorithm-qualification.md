# Qualify post-training algorithms across realistic environments and both GPU workers

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with
`docs/templates/PLAN.md`. It is intentionally separate from the Ambient Agent
research plan: it qualifies the reusable execution and algorithm substrate
before Ambient Agent becomes a training workload.

Provider-backed operator DX is implemented by
`docs/plan/dstack-execution-provider.md`. The scripts under
`scripts/qualification/` are temporary characterization and parity harnesses;
they are not the supported launch interface and must be removed after the
normal CLI produces equivalent execution and evidence.

## Purpose / Big Picture

After this work, a developer can select a bounded qualification scenario and
run it locally or through dstack without editing an environment-specific
Python program. Each scenario performs real model generation in a pinned
Verifiers environment, computes environment rewards, completes at least ten
optimizer updates, publishes model and trace artifacts through Trackio, and is
readable in the remote Observatory backed by Apache Doris.

The matrix is deliberately substantial but not a full research campaign. It
uses enough optimizer updates and rollouts to expose synchronization, memory,
reward-variance, checkpoint, cancellation, and observation failures while
keeping one experiment active at a time. Successful qualification means the
system executed the selected algorithm correctly; it does not mean the model
improved enough for promotion.

This plan does not amend the frozen post-training product baseline. It
implements the existing project, work-package, job, run, environment,
artifact, and observation contracts.

## Progress

- [x] (2026-07-26 12:02Z) Audited the existing AutomationBench TRL and veRL
  entrypoints, canonical training contracts, dstack execution adapter, remote
  Trackio/Doris deployment, and prior GPU evidence.
- [x] (2026-07-26 12:02Z) Confirmed the local RTX 4090 is idle with about
  23 GiB free and the isolated veRL runtime imports Torch 2.10, vLLM 0.18,
  Verifiers, AutomationBench, and veRL.
- [ ] Introduce framework-owned provider-neutral qualification scenarios and
  run them through the normal CLI with local and dstack execution adapters.
- [x] (2026-07-26 12:36Z) Added the immutable scenario/acceptance contract,
  manifest round-trip tests, and a local operator adapter that hides raw task
  indices. This was the local-only checkpoint before the later dstack slice.
- [x] (2026-07-26 14:19Z) Added deterministic source bundles and the dstack
  scenario path, digest-pinned BuildKit runtime receipts, typed execution
  mounts shared by local Docker and dstack, and run-scoped worker workspaces.
- [x] (2026-07-26 14:19Z) Provisioned both GPU workers with persistent
  Hugging Face, vLLM, Torch Inductor, and Triton caches plus conservative
  terminal-marker workspace retention. Both worker playbooks passed registry,
  Docker live-restore, free-space, and real CUDA probes.
- [ ] Convert AutomationBench and GSM8K into scenario adapters without changing
  their environment-owned task, reward, or trace semantics.
- [x] (2026-07-26 12:36Z) Ran AutomationBench GRPO for ten optimizer updates
  on the local RTX 4090 and published complete evidence.
- [x] (2026-07-26 12:36Z) Replayed isolated Verifiers evidence through
  `RunContext`, reconciled 80 live traces and ten trace-derived metric batches
  into remote Trackio/Doris, and verified current-source Observatory reports
  all five required groups plus tool behavior available and
  `research_ready=true`.
- [ ] Run one GSM8K GRPO scenario for at least ten optimizer updates on the
  remote RTX PRO 6000 and publish complete evidence.
- [ ] Exercise a queued second scenario, cancel one real GPU scenario after
  rollout activity begins, verify bounded cleanup, and retry it successfully.
- [ ] Run bounded DPO, on-policy distillation, `eval.general`, and
  `eval.domain` scenarios; retain at least ten optimizer updates for each
  training algorithm and a realistic task population for evaluation.
- [ ] Reconcile every successful and cancelled run through remote Trackio,
  Apache Doris, and current-source Observatory.
- [ ] Complete a developer-experience review, remove redundant knobs and
  environment-specific launch assumptions, and update this plan with measured
  outcomes.

## Surprises & Discoveries

- Observation: the old AutomationBench `tools/` qualification helpers were
  useful research probes but were not a reusable job interface, so they were
  removed in favor of `posttrain job run` work packages.
  Evidence: those scripts hard-coded run IDs, source paths, model bindings,
  execution targets, and environment-specific assertions.

- Observation: previous online-RL infrastructure jobs completed 15 backward
  passes but used a synthetic REINFORCE loss rather than Verifiers rollouts.
  Evidence: `../ambient-agent/docs/plan/ambient-agent-autoresearch-runbook.md`
  (outside this repository commit boundary) explicitly marks
  the online-RL row as optimizer/observation plumbing only and leaves actual
  GRPO, DAPO, and SAMPO unqualified.

- Observation: a realistic GRPO update can legitimately have a zero gradient
  when every generation in a prompt group receives the same reward.
  Evidence: the retained two-step AutomationBench qualification had non-zero
  gradient on step one and zero gradient on step two when both sampled rewards
  matched. Acceptance must therefore require non-zero gradients and reward
  variance somewhere in the run, not on every update.

- Observation: the development workspace has no GPU training dependencies in
  its default `uv` environment.
  Evidence: the root environment cannot import Torch, TRL, vLLM, Verifiers, or
  AutomationBench. The isolated veRL environment at
  `/home/hammad/projects/verl/.venv313` contains the qualified GPU stack. Job
  packaging must name the runtime image or lock explicitly instead of relying
  on the developer shell.

- Observation: the isolated veRL environment originally contained the GPU
  stack but not the Trackio client and its HTTP dependencies.
  Evidence: the first launch failed before GPU allocation with a missing
  `gradio_client` import. Installing the receipted Doris-candidate Trackio
  wheel into `/home/hammad/projects/verl/.venv313` made the same request
  runnable. The runtime image/package must own this dependency.

- Observation: promoting the native Verifiers JSONL artifact did not make
  traces queryable through Trackio or Observatory.
  Evidence: framework run `0bd7625a-7b09-446b-96df-e6d24a152970`
  initially had the 5.1 MB trace artifact but Trackio reported zero live
  traces, no reward standard deviation, and no rollout population series.
  Replaying provider-neutral observations in the host process produced 80
  live traces and ten complete trace-derived metric batches.

- Observation: the hand-written qualification probe recorded too little
  resolved selection metadata for conditional Observatory checks.
  Evidence: tool behavior was initially marked not applicable despite the
  selected `agentic-tool-use` environment. The probe now reuses
  `grpo_job_inputs`; Observatory also recognizes the declared environment
  category. The reconciled run now reports tool behavior available.

- Observation: replaying trace-derived reward mean duplicated the native
  trainer's reward-mean series during this one reconciliation.
  Evidence: the reconciled run has twenty reward-mean points but ten points
  for every other per-update metric. Future evidence replay omits reward mean
  and derives only dispersion, denominators, and tool behavior that the
  trainer does not own.

- Observation: the initial immutable runtime rebuilt all 246 Python packages
  when only the source/provenance digest changed.
  Evidence: `SOURCE_DIGEST` and `VERL_REVISION` were declared before dependency
  installation in the Dockerfile, causing the large uv-sync layer to miss
  cache. The corrected Dockerfile declares dynamic arguments after dependency
  installation and uses a locked BuildKit uv cache mount.

- Observation: the current zstd image has fourteen layers but one virtualenv
  layer accounts for about 10.5 GB of uncompressed history.
  Evidence: the first accepted runtime receipt records zstd level 3 and a
  builder registry round trip of eight seconds; `docker history` identifies
  the single virtualenv copy layer. Zstd compatibility is proven, but layer
  parallelism and capability-profile size remain open optimizations.

- Observation: neither worker can honestly enforce a hard per-run directory
  quota with its current mount.
  Evidence: the RTX 4090 root is XFS mounted with `noquota`; the RTX PRO root
  is ext4. The implementation therefore uses scheduler disk admission,
  free-space floors, bounded checkpoint finalization, and terminal-only aging
  rather than pretending dstack's disk request is a filesystem quota.

## Decision Log

- Decision: qualify one experiment at a time while allowing the scheduler to
  queue later submissions.
  Rationale: serial GPU experiments make memory, cleanup, and causal
  interpretation clear. Queue and cancellation behavior can still be tested
  without running two research experiments concurrently.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: migrate qualification launch ownership from repository scripts to
  the framework CLI and Python SDK.
  Rationale: qualification scenarios belong to the framework, while bundle
  selection, image resolution, provider submission, lifecycle state, and
  reconciliation belong to the shared execution application service. Existing
  scripts remain only until the CLI path passes parity.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: define a qualification scenario as data, not another public
  training API.
  Rationale: public `GRPORequest`, `DPORequest`,
  `OnPolicyDistillationRequest`, and evaluation requests already express job
  meaning. The scenario only binds an existing request to a bounded update
  budget, execution target, environment sample, and acceptance policy.
  Date/Author: 2026-07-26 / Codex.

- Decision: require at least ten completed optimizer updates for every
  algorithm qualification and realistic environment rollouts for online
  algorithms.
  Rationale: a constructor smoke, one backward pass, or synthetic loss does not
  exercise rollout collection, reward flow, policy synchronization, repeated
  checkpointing, or memory stability.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: accept a run when it has at least one non-zero gradient and at
  least one within-group reward difference, rather than requiring both on every
  update.
  Rationale: group-relative objectives correctly produce zero advantage for a
  uniform-reward group. Requiring every step to be non-zero would reject valid
  algorithm behavior and encourage cherry-picked tasks.
  Date/Author: 2026-07-26 / Codex.

- Decision: keep native Verifiers traces as rollout authority and treat trainer
  metrics as run-level summaries.
  Rationale: the trace preserves resolved task identity, model/tool turns,
  reward components, errors, and termination. Metrics must not become a second,
  lossy replay format.
  Date/Author: 2026-07-26 / Codex.

- Decision: use the remote Trackio service and its native Doris engine for all
  matrix runs.
  Rationale: local-only tracking would fail to exercise the deployed evidence
  path the user will operate. Model and trace artifacts flow through Trackio;
  jobs do not receive direct Doris or object-store credentials.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: recover observations from isolated runtimes at the framework
  boundary before promoting their retained artifacts.
  Rationale: Verifiers remains the native replay authority, while
  `RunContext` remains the only provider-neutral path to Trackio. An artifact
  is durable evidence but is not a substitute for queryable metrics and
  traces.
  Date/Author: 2026-07-26 / Codex.

- Decision: express worker storage as typed execution mounts rather than
  provider-specific target-placement dictionaries.
  Rationale: model caches, compile caches, and run workspaces are execution
  inputs with identical logical meaning for local Docker and dstack. Keeping
  them on `ExecutionRequest` removes backend leakage and lets the contract
  require run-id namespacing.
  Date/Author: 2026-07-26 / Codex.

- Decision: automatically delete only workspaces with a valid terminal marker.
  Rationale: Trackio is the durable artifact authority, but a hard-killed run
  may leave its only recoverable checkpoint on the worker. Successful
  workspaces retain seven days, failed/cancelled workspaces retain three days,
  and unmarked workspaces require explicit reconciliation.
  Date/Author: 2026-07-26 / Codex.

- Decision: reuse the composition-owned `grpo_job_inputs` projection in
  qualification probes.
  Rationale: a second hand-written resolved-input projection drifted from the
  product job contract and prevented correct conditional evidence selection.
  Date/Author: 2026-07-26 / Codex.

## Outcomes & Retrospective

The system and plan audit is complete. Native Doris, remote Observatory,
scheduler cancellation, two-worker CUDA execution, 15-step SFT, and bounded
serving are already proven. AutomationBench GRPO is now proven for ten real
updates on the local RTX 4090: 80/80 rollouts completed, a non-zero gradient
and within-group reward variance were observed, the LoRA adapter changed, and
remote Observatory is research-ready with tool evidence active. The retained
qualification receipt is
`.posttrain/state/qualification/automationbench-grpo-10-0bd7625a-7b09-446b-96df-e6d24a152970/qualification-receipt.json`;
the execution-log run is `run-1dd59a085e4d40af8ec5173439301197`.

The remaining gap is multi-environment and multi-algorithm qualification:
GSM8K through dstack on the RTX PRO 6000, real queue/cancel/retry behavior,
DPO, on-policy distillation, and canonical evaluation. No broader capability
promotion is claimed until those rows satisfy this plan.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. Public training request and
result contracts live in `packages/train/src/posttrain/train`. The
Verifiers-to-training bridge lives in
`packages/train/src/posttrain/train/integrations/verifiers.py`. The TRL and
veRL adapters live under `packages/train/src/posttrain/train/backends/`.
Scenario launches use `posttrain job run` against `.posttrain/work_packages/`.
Ephemeral `tools/` job helpers were removed.

`packages/execution` owns provider-neutral submission, status, cancellation,
and result contracts. `packages/execution-local` runs a package on the current
machine. `packages/execution-dstack` translates the same logical package to
dstack. Neither execution adapter owns training semantics.

The dstack control plane and Trackio run on `ai-control`. The two GPU workers
are `pop-os.lan`, with an RTX 4090 and 24 GiB VRAM, and
`carbonteq-ai-workstation.lan`, with an RTX PRO 6000 and 96 GiB VRAM. The
shared Trackio endpoint is `trackio.lan`; it stores structured evidence in
Apache Doris and artifact bytes behind Trackio's artifact boundary.
Observatory reads Trackio remotely and never reads Doris tables directly.

A qualification scenario is a versioned description of one bounded proof. It
names an existing job kind, model, environment or dataset, algorithm binding,
execution target class, update/task/rollout budgets, and acceptance checks. A
scenario result is still a normal observed run; the scenario is not a new
product hierarchy level.

## Plan of Work

The existing `scripts/qualification/algorithm_scenarios.py` characterizes the
initial contract. Move its reusable immutable
`QualificationScenario` and `QualificationAcceptance` values. The scenario
contains stable IDs and catalog references, not live model or environment
objects. It also contains the minimum optimizer updates, task count,
generations per task, maximum duration, and target capability requirements.
Acceptance contains evidence requirements such as a minimum trace count,
complete update count, at least one reward-variant group, at least one
non-zero gradient, produced model artifact, clean finalization, and remote
tracking visibility.

Use the normal `posttrain` CLI as the single operator entrypoint. It resolves a
named framework scenario, acquires the existing experiment
lease from `packages/work/src/posttrain/work/admission.py`, creates a unique
run ID and workspace, renders the selected public request, and dispatches
through either `posttrain_execution_local` or
`posttrain_execution_dstack`. It never imports an environment package until
after resolving and validating its immutable source revision.

Keep `scripts/qualification/run_algorithm_scenario.py` temporarily to compare
the resulting `ExecutionRequest`, native provider state, Trackio evidence, and
retained artifacts. Remove it once the CLI path passes that parity check.

Environment-specific construction belongs in work-package bindings and small
scenario adapters under `scripts/qualification/`, not ephemeral `tools/`
helpers. The AutomationBench adapter selects categories and deterministic
sampling policy through its `EnvironmentBinding`; it must not expose raw task
indices as the ordinary operator interface. The GSM8K adapter selects a bounded
train split and seed through its environment binding. Both adapters use
`create_verifiers_training_bridge`, preserving native trace payloads.

Add `scripts/qualification/validate_algorithm_run.py`. It reads the
`TrainingResult`, training summary, native Verifiers JSONL artifact, selected
checkpoint/adapter, scheduler terminal state, and remote Trackio run. It
calculates trace and reward-group evidence rather than trusting a success
string. It emits a compact terminal qualification receipt containing only
identities, counts, digests, measured peaks, and acceptance results. Full
traces and model bytes remain Trackio artifacts.

Package the exact framework source digest, environment Git revision, runtime
image digest, and scenario manifest into every dstack job. A worker receives
Trackio URL/write-token settings from the execution environment. It does not
receive Apache Doris, RustFS, registry, or Observatory credentials. The same
package can run locally by selecting the local adapter.

Execute the matrix serially. Start with AutomationBench GRPO on the RTX 4090
because its runtime and model are already cached and the machine is idle. Use
at least ten updates and enough generations for a valid group-relative
objective. Inspect reward variance and memory after the first two updates; do
not silently lower the ten-update acceptance gate. If the run fails, retain
the logical evidence, clean disposable checkpoints and caches, revise the
scenario, and retry under a new run ID.

Run GSM8K GRPO on the RTX PRO 6000 using the same scenario runner. This proves
the abstraction is not AutomationBench-specific and exercises a second worker.
Then queue one additional bounded run while the first is active, verify it
remains queued, cancel it only after the selected active/queued state is
observed, confirm the scheduler terminal reason and worker container cleanup,
and retry successfully.

After GRPO, add bounded DPO and on-policy distillation scenarios. DPO uses a
canonical preference dataset and therefore does not create a Verifiers
environment. Distillation generates fresh student trajectories in the selected
Verifiers environment and teacher-scores the exact student token IDs. Each
training algorithm completes at least ten optimizer updates. Canonical
`eval.general` and `eval.domain` scenarios consume the produced model artifact
and retain a realistic task population rather than optimizer updates.

Finally inspect each remote run through current-source Observatory. Verify job
view selection, update count, reward population, gradients, traces, failures,
phase timing, GPU memory, checkpoints, artifact lineage, and terminal status.
Record any awkwardness in this plan before changing public APIs. Prefer fixing
scenario construction, provider translation, or validation messages over
adding technique-specific flags to shared contracts.

## Concrete Steps

Run all commands from `/home/hammad/projects/rl` unless stated otherwise.

Before each GPU experiment, confirm the single-experiment lease and hardware:

    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader
    /home/hammad/projects/ai-infra/scripts/preflight-workers

Run focused contract tests while implementing:

    uv run pytest \
      packages/execution/tests \
      packages/execution-local/tests \
      packages/execution-dstack/tests \
      packages/train/tests/test_verifiers_grpo_bridge.py \
      packages/train/tests/test_verl_backend.py \
      scripts/qualification/tests -q
    uv run ruff check scripts/qualification packages/execution*
    uv run pyright scripts/qualification packages/execution*
    uv run lint-imports

The temporary characterization commands are:

    uv run python scripts/qualification/run_algorithm_scenario.py \
      automationbench-qwen35-08b-grpo-10 \
      --provider local \
      --target pop-os.lan

and:

    uv run python scripts/qualification/run_algorithm_scenario.py \
      gsm8k-qwen35-08b-grpo-10 \
      --provider dstack \
      --target carbonteq-ai-workstation.lan

The supported replacement is defined in
`docs/plan/dstack-execution-provider.md` and must use:

    posttrain work-package plan <package.yaml> --job <job> --provider dstack
    posttrain work-package run <package.yaml> --job <job> --provider dstack
    posttrain run status <run-id>
    posttrain run logs <run-id>
    posttrain run reconcile <run-id>

Each command prints the immutable run ID immediately, then only bounded status
updates. On success it prints a receipt path and remote Trackio reference. A
retry creates a new run and records `run_attempt`; it never overwrites the
failed run.

After each run, validate remote evidence:

    uv run python scripts/qualification/validate_algorithm_run.py \
      --receipt .posttrain/state/qualification/<run-id>/receipt.json \
      --trackio-url https://trackio.lan

The validator must report at least:

    status=succeeded
    optimizer_updates=10
    traces_complete=<positive count>
    reward_variant_groups=<positive count>
    nonzero_gradient_updates=<positive count>
    model_artifact=<name>:<version>
    observatory_source=healthy

Run the full repository ladder after interface changes:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

The scenario contract is accepted when unit tests prove invalid update, task,
rollout, duration, and target budgets fail before submission; manifests
round-trip without live Python objects; and local and dstack adapters receive
equivalent logical requests.

An online-RL run is accepted only when a real model generated native Verifiers
episodes, at least ten optimizer updates completed, all selected update IDs
are present, at least one group has reward variance, at least one update has a
non-zero gradient, a checkpoint or adapter contains changed parameters, and
the final policy was synchronized before a post-update rollout or evaluation.
A ten-iteration synthetic tensor loop does not pass.

Migration across environments is accepted when the same runner completes both
AutomationBench and GSM8K without environment-specific branches in the
execution providers. Resolved tasks, reward components, and turn/tool details
may differ because the environments own those semantics.

Scheduler acceptance requires an observed queued state, an explicit
cancellation request, a terminal cancelled state with reason, no remaining
job container on either worker, retained terminal evidence, and a successful
retry under a new run ID. It must not use a sleeping CUDA container as the only
proof; at least one case must cancel a real scenario after rollout activity.

Tracking acceptance requires the normal Trackio SDK/observer path to expose
every run through the remote service. Observatory must show the appropriate
job view, exact update and trace populations, model input/output lineage,
resource series, and cancelled/failed states. Direct Doris queries are
diagnostic only and cannot substitute for Trackio and Observatory reads.

The developer experience is accepted when the operator chooses a scenario,
provider, and target without editing Python, copying datasets manually,
selecting raw environment row numbers, or supplying storage/database
credentials. Error messages must name the failed gate and the retained run
reference.

## Idempotence and Recovery

Scenario IDs and source revisions are immutable, but every execution receives a
new run ID. Re-running a scenario is safe and creates another attempt. The
experiment lease prevents overlapping measured jobs while still allowing a
second dstack submission to remain queued for the scheduler test.

Failed and cancelled runs retain compact receipts, Trackio metrics, native
traces already produced, terminal reasons, and artifact metadata. Disposable
worker containers, partial upload sessions, transient caches, and unselected
recovery checkpoints are removed after finalization. Never run a global Docker
prune.

If local execution fails, release the lease in a `finally` block and verify no
trainer or rollout process remains. If dstack connectivity is interrupted,
reconcile by immutable submission ID before resubmitting. If Trackio is
temporarily unavailable, retain a bounded local observation spool and retry
idempotently. If artifact publication fails, do not mark the run succeeded
until the finalizer reads the artifact metadata back through Trackio.

Model and dataset source caches are shared, expensive inputs and are not
deleted by scenario cleanup. Large optimizer checkpoints are retained only for
an active resume or a selected diagnostic; final model artifacts and compact
summaries remain.

## Artifacts and Notes

The required durable artifacts for each training run are the terminal
qualification receipt, native Verifiers trace bundle for online jobs, training
summary, selected model adapter or weights, and input/output lineage. Full
stdout, duplicated trace projections, dataset copies, transient vLLM caches,
and every intermediate checkpoint are not durable outputs.

The Apache Doris cutover is recorded in
`docs/plan/trackio-apache-doris-engine.md` and execution-log sequence 34. This
plan consumes Trackio as a service; algorithm jobs never import a Doris client.

## Interfaces and Dependencies

In `scripts/qualification/algorithm_scenarios.py`, define:

    @dataclass(frozen=True)
    class QualificationAcceptance:
        minimum_optimizer_updates: int
        minimum_complete_traces: int
        require_reward_variance: bool
        require_nonzero_gradient: bool
        require_model_artifact: bool
        require_remote_observatory: bool

    @dataclass(frozen=True)
    class QualificationScenario:
        id: str
        job_kind: str
        model_ref: str
        environment_ref: str | None
        dataset_ref: str | None
        training_ref: str
        inference_ref: str | None
        update_budget: int | None
        task_budget: int | None
        rollouts_per_task: int | None
        maximum_duration_seconds: int
        target_capabilities: Mapping[str, object]
        acceptance: QualificationAcceptance

In `scripts/qualification/run_algorithm_scenario.py`, expose:

    def run_scenario(
        scenario: QualificationScenario,
        *,
        provider: Literal["local", "dstack"],
        target: str,
    ) -> QualificationReceipt:
        ...

In `scripts/qualification/validate_algorithm_run.py`, expose:

    def validate_run(
        receipt: QualificationReceipt,
        *,
        tracking_reader: TrackingReader,
        observatory: ObservatoryClient,
    ) -> QualificationReport:
        ...

Use existing public request types from `posttrain.train`, existing
`ExecutionProvider` contracts from `packages/execution`, and existing Trackio
readers from `packages/tracking-trackio`. Do not introduce a workflow
orchestrator, environment-specific execution provider, direct Doris client, or
second artifact registry.

Revision note (2026-07-26): Created this focused plan after the native Doris
cutover exposed the remaining qualification gap. It replaces the implicit idea
of extending one-off AutomationBench scripts with a reusable scenario,
execution, validation, and evidence boundary.

Revision note (2026-07-26): Made the provider-backed framework CLI plan a
prerequisite and reclassified qualification scripts as temporary parity
harnesses. The framework, not a repository script, owns scenario launch,
packing, submission, lifecycle, and reconciliation.
