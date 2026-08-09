# Publish reusable model views with every selected training checkpoint

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document must be maintained in accordance with
`docs/templates/PLAN.md`.

The plan implements the product meanings in `docs/post-training/01-workflow.md`
through `docs/post-training/06-observation-and-lineage.md`. Milestone 0 makes a
narrow clarification to that frozen baseline before implementation because the
current prose calls recovery checkpoints workspace state while the released
code already publishes selected recovery checkpoints as durable run outputs.
The clarification preserves the important distinction: recovery state does not
become a loadable model or a model-lineage node merely because it is retained.

## Purpose / Big Picture

After this work, a training run can expose a selected checkpoint in two useful
forms without launching a second materialization job. The recovery form restores
the exact training process, including optimizer, scheduler, trainer, random
number generator, and adapter or model state. The model form is directly
loadable by a new training branch, evaluation, or serving job. Both forms are
produced by the original training run, identify the same checkpoint step, and
reuse content-addressed storage rather than uploading duplicate bytes.

A LoRA or QLoRA run publishes only adapter parameters in the model form; it
never copies immutable base-model weights into that artifact. A full-parameter
run publishes model weights only when its backend can produce an
inference-loadable representation without an unplanned heavyweight conversion.
Packing selects and validates this behavior but cannot produce parameters that
do not exist until training reaches a checkpoint.

The visible proof is one small LoRA training run with checkpoint publication at
step one. Its run page shows a recovery artifact and a model-adapter artifact
for the same checkpoint snapshot. Four later commands then use those retained
outputs: exact training resume consumes the recovery form; a fresh training
branch, evaluation, and serving consume the model form. No provider run with a
materialization job kind appears between producer and consumer. A graceful
cancellation publishes and drains the latest complete checkpoint, while an
abrupt process loss preserves the most recently committed off-host checkpoint.

## Progress

- [x] (2026-08-09 11:19Z) Read the repository instructions, canonical
  post-training baseline, plan template, related artifact lifecycle plan,
  checkpoint and resume implementation, tracking adapters, Trackio direct-S3
  transport, work-package planning, standard job definitions, and Observatory
  artifact surfaces.
- [x] (2026-08-09 11:19Z) Resolve the core product decision: checkpoint model
  projection is an output action of the original training run, not a separate
  user-visible job; recovery and model forms remain separate typed artifacts.
- [x] (2026-08-09 11:19Z) Record the current implementation gaps and the
  multi-repository release order in this plan without modifying the existing
  dirty training or cleanup work.
- [x] (2026-08-09 11:27Z) Revise the plan with a first-class read-only
  checkpoint inspection surface covering summaries, manifests, file roles,
  integrity, compatibility, and step-to-step differences.
- [x] (2026-08-09 10:02Z) Milestone 0: clarify the frozen product baseline and
  add the durable ADR. The canonical work/evidence, API, and observation docs
  now distinguish durable recovery views from loadable model views, define the
  run-scoped inspection surface, and record the no-materialization-job decision
  in ADR 0015.
- [/] Milestone 1: add checkpoint snapshot, publication policy, model descriptor,
  and backend capability contracts with deterministic unit fixtures. Provider-
  neutral contracts, publication handles, and resolver fixtures are complete;
  broader backend capability tests remain.
- [x] (2026-08-09 12:20Z) Milestone 2: add bounded asynchronous artifact
  publication in the Trackio adapter and fork. Trackio post12 commit
  `4c73e8b6e71c3da65cac41fc1371830e4435ecea` passed its unit suite and built
  wheel/sdist; the exact wheel (`sha256:6ed1bda951a70b85611a8928d489c7b806abfb9bd19643a165a9111a78c9d0f4`)
  and sdist (`sha256:78c9b4db9b659810cd31f05693342686cc4506ab2f96676b2d3bc64c324e416c`)
  were published to the stable internal index and read back. The framework
  pin and OCI runtime lock now select post12.
- [/] (2026-08-09 12:05Z) Milestone 3: TRL now publishes paired recovery and
  adapter-only model views at `on_save`, and handled interruption uses the same
  publisher. Normal/failure unit coverage is complete; live GPU cancellation
  and abrupt-loss qualification remain.
- [/] (2026-08-09 12:05Z) Milestone 4: run-scoped inspection (`list`, `show`,
  `verify`, `diff`), exact recovery/model resolver selection, and CLI
  `--checkpoint-step`, `--model-from-run`, and `--model-checkpoint-step` are
  implemented. End-to-end packaged train/eval/serve selection still needs
  fixture and live qualification.
- [ ] Milestone 5: add veRL and full-parameter capability negotiation, admitting
  only representations that are safe to expose without a hidden conversion.
- [ ] Milestone 6: expose lightweight checkpoint summaries and a lazy detailed
  checkpoint ledger in Observatory.
- [ ] Milestone 7: integrate reference-aware retention, purge, and historical
  recovery-only adapter backfill.
- [ ] Milestone 8: run local, remote, cancellation, crash, lineage, and release
  qualification and record exact immutable evidence.

## Surprises & Discoveries

- Observation: successful training already publishes a model artifact and an
  optional recovery checkpoint from the same run.
  Evidence: `packages/train/src/posttrain/train/api.py::_finalize` emits role
  `model` as `model-adapter` or `model-weights`, then emits role `recovery` as
  `training-checkpoint`. The missing behavior is checkpoint-scoped publication
  during a run and on cancellation, not a new model-transform operation.

- Observation: interrupted TRL training publishes only the latest complete
  recovery directory.
  Evidence: `packages/train/src/posttrain/train/backends/trl/common.py::publish_interrupted_recovery_checkpoint`
  validates adapter-only recovery state and emits `training-checkpoint`; it does
  not project the loadable adapter view.

- Observation: artifact publication blocks the caller today.
  Evidence: `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py::TrackioTrackedRun.artifact`
  calls `Run.log_artifact` synchronously. The Trackio fork's
  `Artifact.wait()` is explicitly a no-op because logging is synchronous.
  Publishing every checkpoint through this path would add network and object
  storage latency directly to measured training step time.

- Observation: Trackio already provides the expensive transport primitive this
  plan needs.
  Evidence: `../trackio/trackio/direct_uploads.py` and
  `../trackio/trackio/remote_client.py` implement restart-safe, digest-bound,
  presigned multipart upload directly to any configured S3-compatible store.
  The missing part is a bounded background publication lifecycle and durable
  result handle, not another storage protocol.

- Observation: the current CLI cannot address more than one checkpoint from a
  run.
  Evidence: `apps/cli/src/posttrain_cli/commands/work_package.py` requires
  exactly one `training-checkpoint` output for `--resume-from-run`. Intermediate
  publication must replace this count-based assumption with an explicit
  complete-step selector.

- Observation: `posttrain run show` exposes a general recorded run view but no
  command owns checkpoint-specific interpretation.
  Evidence: `apps/cli/src/posttrain_cli/commands/run_cmd.py` registers run list,
  status, wait, logs, cancel, reconcile, purge, and show commands. Checkpoint
  inspection must be added as a nested read-only surface rather than hidden
  inside generic run output or implemented only in Observatory.

- Observation: model replacement is more than adding one input artifact.
  Evidence: standard eval, serve, GRPO, and distillation definitions bind a
  `ModelVariant` and one or more `InferenceBinding` values that must name the
  same model. `packages/work/src/posttrain/work/runner.py::_artifact_inputs`
  currently also uses the global names `model_adapter` and `model_weights`,
  which are ambiguous for jobs with several model seats. The model-source API
  needs a declared target seat and must update the related inference seats
  atomically.

- Observation: current required artifact roles are singular.
  Evidence: `packages/work/src/posttrain/work/execution.py` requires exactly one
  artifact for every `required_artifact_role`. Intermediate checkpoint model
  views therefore cannot all use the terminal role `model`; they need a
  checkpoint-specific role while the successful terminal output retains the
  existing singular `model` role.

- Observation: cancellation finalization cannot rely on ordinary cooperative
  observation calls after cancellation.
  Evidence: `RunContext.artifact` checks the cancellation token before calling
  the observer. A bounded finalization scope is required so a graceful stop can
  drain already-created artifacts without reopening arbitrary training work.

- Observation: the canonical baseline and current durable implementation use
  different shorthand for recovery state.
  Evidence: `docs/post-training/06-observation-and-lineage.md` calls recovery
  checkpoints workspace state, while current interrupted and successful train
  paths publish `training-checkpoint` artifacts. The baseline needs to say that
  a durable recovery artifact remains recovery state and is excluded from model
  lineage until a typed model view is emitted.

- Observation: current sibling forks are clean but their development heads are
  not automatically the framework's executable releases.
  Evidence: during planning `../trackio` was on
  `codex/attested-internal-release` at
  `4c73e8b6e71c3da65cac41fc1371830e4435ecea`, while the framework selected
  `carbonteq-trackio==0.31.5.post12`. `../trl` was on
  `codex/bounded-vllm-waves` at
  `c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1`, while
  `packages/train/pyproject.toml` selected `trl==1.9.2.post1` with published
  source revision `a82ecebc0fa081efd58302a34a553445fc73271d`.
  Implementation must resolve and publish exact fork commits before changing
  pins; a clean sibling checkout is not publication evidence.

- Observation: the existing common model contract can carry checkpoint-derived
  model metadata without coupling consumers to a trainer backend.
  Evidence: `ModelArtifactDescriptor` round-trips renderer, base, capability,
  precision, and adapter/full-form facts and reconstructs a `ModelVariant` only
  from a committed `StoredArtifactRef`; focused tests pass six cases.

- Observation: publication lifecycle methods are currently absent from the
  concrete tracking adapters, so the first contract change remains structural
  and non-breaking.
  Evidence: `TrackedRun` now declares `flush_artifacts` and `RunDataSource`
  declares `verify_artifact`, but no adapter behavior changed in this milestone.

- Observation: a recovery-only checkpoint must remain selectable for exact
  continuation even when its paired model view was not produced.
  Evidence: the resolver now exposes `recovery_ready` and `model_ready`
  independently and requires a committed digest only for the requested view;
  digestless or partial snapshots remain visible to inspection rather than
  being silently discarded.

- Observation: existing job definitions already materialize model inputs by
  stable names, so model-source selection can reuse the same path without
  changing trainer APIs.
  Evidence: selected `model_adapter`/`model_weights` inputs are now honored by
  SFT, DPO, GRPO, distillation, eval, and serve definitions, with the model
  interface and inference binding updated together.

## Decision Log

- Decision: The original training run produces checkpoint-scoped recovery and
  model artifacts; ordinary reuse does not create a materialization job or
  provider run.
  Rationale: LoRA projection is selecting a safe subset of files that already
  exists at checkpoint time. A second scheduled job adds latency, cost, lineage
  noise, and a failure boundary without changing model parameters.
  Date/Author: 2026-08-09 / user and Codex.

- Decision: A checkpoint snapshot has two separate typed views rather than one
  universal artifact kind.
  Rationale: exact resume requires optimizer and random state that vLLM and eval
  must never load. Existing strict artifact-kind validation correctly prevented
  a recovery checkpoint from masquerading as a model adapter. A shared
  `checkpoint_snapshot_id` links the views without weakening consumers.
  Date/Author: 2026-08-09 / Codex, accepted direction from user proposal.

- Decision: LoRA and QLoRA model views contain adapter files and model-interface
  metadata only; immutable base-model weights are forbidden.
  Rationale: base weights are selected separately by immutable Hub revision,
  and duplicating them makes checkpoint publication slow, expensive, and
  semantically wrong.
  Date/Author: 2026-08-09 / user and Codex.

- Decision: Packing validates and carries checkpoint-publication policy but does
  not create checkpoint outputs.
  Rationale: packing occurs before training parameters exist. It can include the
  code, schema, policy, backend capability attestation, and storage settings
  needed at runtime, but it cannot project future learned weights.
  Date/Author: 2026-08-09 / Codex.

- Decision: Artifact transfer is a bounded background activity inside the
  producing run, with explicit durable commit and final drain semantics.
  Rationale: synchronous upload would distort step time, but an unbounded
  fire-and-forget upload could lose artifacts or exhaust memory and disk.
  `model ready` means the artifact manifest and every referenced blob are
  committed, not merely queued locally.
  Date/Author: 2026-08-09 / Codex.

- Decision: Pure checkpoint projection is not a `model.transform` job. Merge,
  quantization, export, or distributed-shard consolidation remains a transform
  when the backend cannot produce a directly loadable model view at checkpoint
  time.
  Rationale: job identity should represent meaningful work and evidence, not
  internal copying. Heavy representation changes have their own resources,
  failure modes, and lineage and therefore remain explicit jobs.
  Date/Author: 2026-08-09 / Codex.

- Decision: `latest` is a planning alias only.
  Rationale: every submitted consumer must record the exact source run,
  checkpoint step, artifact version, content digest, base-model revision, and
  model-view kind. Execution must never follow a moving alias.
  Date/Author: 2026-08-09 / Codex.

- Decision: A new train job has two intentionally different source modes.
  Rationale: `resume` restores optimizer, scheduler, trainer, RNG, and parameter
  state, while `branch` consumes only the model view and starts a fresh
  optimizer. Silently inferring one from the other would make experiments
  irreproducible.
  Date/Author: 2026-08-09 / Codex.

- Decision: The normal path publishes checkpoint artifacts prospectively;
  historical recovery-only runs use an explicit idempotent CPU backfill command
  that attaches a model view to the source run without scheduling dstack.
  Rationale: existing runs, including the retained Ambient step-50 checkpoint,
  predate the new contract. They need a safe compatibility path, but hidden
  automatic mutation during eval planning would make planning non-read-only.
  Date/Author: 2026-08-09 / Codex.

- Decision: Checkpoint inspection is a first-class, read-only CLI and Python
  API with list, show, verify, and manifest-diff operations.
  Rationale: developers need to determine what exists, which view is durable,
  whether LoRA safety invariants hold, and whether a checkpoint is compatible
  before scheduling a consumer. Generic run output and provider-native artifact
  commands do not understand recovery versus model semantics. Inspection never
  publishes, aliases, repairs, or deletes artifacts.
  Date/Author: 2026-08-09 / user and Codex.

## Outcomes & Retrospective

Milestone 0 outcome as of 2026-08-09: the intended behavior, ownership
boundaries, and developer command semantics are now recorded in the canonical
baseline and ADR 0015. This closes the documentation gate; it does not claim
runtime support. The next milestone is the provider-neutral contract layer.

Planning outcome as of 2026-08-09: the intended behavior, ownership boundaries,
artifact identities, failure semantics, developer commands, migration path,
and qualification gates are specified. No implementation is complete. In
particular, no Trackio or framework release should be described as supporting
checkpoint-scoped model views until Milestone 8 records a real producer,
resume, branch, eval, serve, graceful-cancel, and abrupt-failure proof.

Milestone 1 partial outcome as of 2026-08-09: the framework now has immutable
checkpoint snapshot IDs, structural manifests, bounded publication policy,
backend projection capability, and a model-view descriptor with adapter/full
weight safety checks. The focused contract suite reports `6 passed`; concrete
publisher and planner behavior remains intentionally unimplemented.

Implementation contributors must update this section after every milestone with
what actually shipped, what differed from the plan, and the exact retained
evidence. Do not convert queued, locally saved, or deferred publication into a
success claim.

## Context and Orientation

A **checkpoint snapshot** is one atomic parameter and trainer-state save at an
exact logical training step. A snapshot is complete only after its backend has
closed all files, written a versioned manifest, and passed structural
validation. A partially written directory is never selectable.

A **recovery view** is a `training-checkpoint` artifact. It contains everything
required for an exact compatible training continuation. For LoRA and QLoRA that
means adapter parameters plus trainer, optimizer, scheduler, and random state;
it must not contain the immutable base-model weights. Recovery compatibility is
stricter than model compatibility and includes trainer backend, checkpoint
schema, parameter-update kind, base model, tokenizer/renderer, optimizer and
scheduler configuration, and any distributed topology constraint imposed by
the backend.

A **model view** is a `model-adapter` or `model-weights` artifact produced from
the same snapshot. It is inference-loadable and intentionally excludes
optimizer, scheduler, trainer, and random state. Consuming it from another
training job creates a new branch with a fresh optimizer. Consuming it from an
eval or serve job does not create or mutate a checkpoint.

An **artifact publication** is durable only after the selected tracking backend
returns an immutable provider version and digest. Copying files into a local
staging directory or queuing an upload is not publication. Trackio owns the
artifact metadata, run-artifact edge, and S3-compatible storage boundary;
dstack owns provider scheduling and container lifecycle, not artifact
semantics.

The current framework flow is as follows. `packages/train` calls a private TRL
or veRL backend. On normal completion, `packages/train/src/posttrain/train/api.py`
publishes a terminal model, recovery checkpoint, summary, and optional
retention manifest through `RunContext.artifact`. On an interrupted TRL run,
`packages/train/src/posttrain/train/backends/trl/common.py` finds the latest
native checkpoint and publishes only recovery state. The concrete Trackio
observer in `packages/tracking-trackio` synchronously uploads and commits each
artifact. `packages/work` asks the tracking backend for committed artifacts
before deleting the workspace. `apps/cli` can inject one recovery checkpoint
into a new training run through `with_recovery_checkpoint`, but it cannot select
a step or inject a model view. `apps/observatory` labels model and recovery kinds
but does not group them into checkpoint snapshots.

This plan touches four ownership areas. `packages/train` owns checkpoint policy,
snapshot manifests, backend validation, and model projection. `packages/common`,
`packages/tracking`, and concrete tracking adapters own the smallest
provider-neutral publication lifecycle. `packages/work`, `packages/jobs`,
`packages/execution-pack`, `apps/runtime`, and `apps/cli` own source resolution,
seat rebinding, immutable packaging, and submission. `apps/observatory` remains
read-only and owns checkpoint presentation. Generic Trackio background-upload
behavior belongs in `../trackio`, followed by an exact released dependency pin
in this repository. No planned change belongs in `../trl`; use TRL's public
callback hooks from the framework. If implementation proves a generic trainer
fix is unavoidable, update the fork ledger, commit and publish that fork first,
then amend this plan before changing the framework pin.

The implementation must preserve unrelated dirty work. At plan creation the
framework checkout contained active purge architecture changes, OLMo3 plan
updates, and TRL checkpoint correctness edits. Do not reset, overwrite, stage,
or commit those changes as part of this plan. Before each milestone, run
`git status --short` in every repository it touches and record overlapping
files in `Surprises & Discoveries`.

### Product behavior and command contract

The normal prospective path is automatic. A training selection chooses a local
save cadence and an independent publication policy. At every selected complete
step, the same run emits recovery and model views. LoRA defaults may publish
both views at every retained checkpoint because the model view is small. Full
updates must choose terminal or milestone publication explicitly and pass a
backend capability check.

The CLI contract after implementation is:

    posttrain [--json] run checkpoint list RUN_ID \
      [--view recovery|model|both] [--state ready|pending|failed] \
      [--limit 50] [--cursor CURSOR]

    posttrain [--json] run checkpoint show RUN_ID --step 50 \
      [--view recovery|model|both] [--files]

    posttrain [--json] run checkpoint verify RUN_ID --step 50 \
      [--view recovery|model|both] [--deep] [--max-download SIZE]

    posttrain [--json] run checkpoint diff RUN_ID \
      --from-step 25 --to-step 50 [--view recovery|model]

    posttrain job run TRAIN_WORK_PACKAGE --job JOB \
      --resume-from-run RUN_ID --checkpoint-step 50

    posttrain job run TRAIN_WORK_PACKAGE --job JOB \
      --model-from-run RUN_ID --checkpoint-step 50 [--model-seat model]

    posttrain job run EVAL_WORK_PACKAGE --job JOB \
      --model-from-run RUN_ID --checkpoint-step 50

    posttrain job run SERVE_WORK_PACKAGE --job JOB \
      --model-from-run RUN_ID --checkpoint-step latest

`checkpoint list` is the bounded discovery command. It returns one row per
snapshot with exact step, completion state, recovery/model readiness, model
kind, sizes, provider versions, digest prefixes, and publication timestamps.
It is newest-first, paginated, and does not fetch file manifests. `--step
latest` on the other inspection commands is permitted as an interactive query,
but the output must print the exact step it resolved at that moment.

`checkpoint show` explains one snapshot. It reports backend and immutable
revision, technique, parameter-update kind, base model, renderer/tokenizer
identity, publication states, artifact identities, whether exact resume is
supported, whether the model view is loadable, and any compatibility reason.
`--files` fetches the small manifests and prints component role, relative path,
size, and SHA-256; it does not download tensor bytes.

`checkpoint verify` checks that the snapshot pair is internally consistent,
artifact kinds match their descriptors, manifests reference present blobs, and
the declared compatibility and LoRA denylist rules hold. The default uses
provider metadata and server-side blob-presence checks. `--deep` downloads each
selected view into temporary project state, recomputes file and tree digests,
validates required recovery files or the adapter-only model layout, and removes
the temporary copy after reporting. It does not load the model, allocate a GPU,
or run inference. Before downloading, it reports the declared byte total,
checks free local space, and enforces the machine default or explicit
`--max-download` bound. An integrity capability unavailable from a provider is
reported as `unsupported`, never silently treated as verified.

`checkpoint diff` compares manifests and model descriptors between two steps in
the same run. It reports added, removed, and digest-changed files plus changes
to backend, update, base, renderer, and compatibility metadata. It deliberately
does not perform tensor-level numerical comparison; that is a separate analysis
operation and would require downloading and loading weights.

`--resume-from-run` is accepted only by a compatible training job and selects
the recovery view. `--model-from-run` selects the model view and is accepted by
any job definition that declares a model input seat. It can target train, eval,
or serve. When a job has one branchable model seat, `--model-seat` is optional.
When a job has several, such as student and teacher in distillation, omission is
an error unless the job definition declares one unambiguous default branch
seat. The two source modes are mutually exclusive.

The Python planning API uses the same values rather than implementing another
resolution path. `latest` resolves during `job plan` to the greatest complete
step whose requested artifact view is durably committed. `job plan`, `job
pack`, and global-JSON `posttrain --json job run` output display the resolved
step, provider version, digest,
base revision, update kind, and targeted model seat. `job pack` carries those
immutable values into the actual-job image. A worker never queries `latest`.

For historical recovery-only adapters, the compatibility command is explicit:

    posttrain run checkpoint publish-model RUN_ID --step 50 --dry-run
    posttrain run checkpoint publish-model RUN_ID --step 50 --apply PLAN_DIGEST

It runs as local CPU work, downloads and verifies the recovery artifact,
projects only the adapter files, uploads through the configured artifact
backend, attaches an audited output edge to the exact source run, and requests
framework evidence reconciliation. It is idempotent by source artifact digest,
checkpoint step, and projection schema. It never creates a dstack provider run.
The first version supports LoRA and QLoRA only. Full-weight historical export
must use an explicit `model.transform` job when format conversion is required.

### Checkpoint identities and artifact contents

Every snapshot has a stable identifier derived from the producer run and exact
step, such as `RUN_ID/step-00000050`. This is a logical identifier, not a mutable
storage path. The versioned snapshot manifest records at least:

    schema_version
    checkpoint_snapshot_id
    source_run_id
    global_step
    created_at
    training_backend and immutable backend revision
    technique and settings identity
    parameter_update_kind
    base_model repository and full revision
    renderer identity and tokenizer fingerprint
    trainer/checkpoint schema version
    complete=true
    component entries: role, relative path, size, sha256

The recovery logical name includes the step:

    training/<model>/<technique>/checkpoints/step-00000050/recovery

The model logical name also includes the step:

    training/<model>/<technique>/checkpoints/step-00000050/model

The recovery artifact uses role `checkpoint-recovery`. The model view uses role
`checkpoint-model`. The existing terminal `model` and `summary` roles remain
singular required outputs for a successful training run. If the terminal step
is already a published checkpoint model view, terminal finalization may reuse
its exact provider artifact as the terminal model edge rather than uploading a
duplicate version, but it must still satisfy the singular required role.

The model view includes a versioned model descriptor sufficient to reconstruct
a `ModelVariant` without editing the catalog. It records the model form,
precision, family, parameter count, instruction-tuned flag, base identity,
renderer/conversation contract identity, capabilities, tokenizer fingerprint,
parent model, source run, checkpoint step, and projection schema. The stored
artifact reference itself is supplied only after publication and is not baked
as a mutable alias into this descriptor.

For LoRA and QLoRA, the projection allowlist contains PEFT adapter configuration
and adapter tensor files plus the model descriptor and safe tokenizer or
renderer metadata selected by the existing model contract. The denylist rejects
full base tensor indexes and common complete-model tensor names as well as
optimizer, scheduler, scaler, trainer, RNG, and training-argument files. The
recovery view keeps the trainer files but continues to reject base-model tensor
duplication through `validate_adapter_only_directory`.

The projection uses hard links or reflinks into a temporary view directory when
the filesystem supports them, with a bounded copy fallback. Trackio hashes
individual files and its content-addressed store skips blobs already present,
so the two logical artifacts do not require duplicate off-host bytes. The
snapshot and both view manifests retain their own complete tree digests because
their allowed file sets differ.

### Publication and failure semantics

`TrainingLoop.checkpoint_steps` remains the local checkpoint cadence and
`checkpoint_limit` remains native local retention. Add a typed
`CheckpointPublicationPolicy` rather than overloading those fields. It selects
recovery publication, model-view publication, milestone steps, terminal and
cancel behavior, and whether a declared publication failure is fatal. Tracking
transport concurrency and drain timeout remain runtime/host settings rather
than algorithm settings.

Artifact publication has the states `queued`, `uploading`, `committed`,
`failed`, and `aborted`. Only `committed` is selectable. The producer emits
events carrying snapshot ID, step, view kind, bytes, duration, and safe error
type. It records metrics for checkpoint save time, projection time, queued
duration, upload duration, committed bytes, queue depth, and final drain time.
No event may expose signed URLs, credentials, or local secret-bearing paths.

The publication queue is bounded by item count and staged bytes. Blob transfer
may use a small configurable worker pool, while artifact-version allocation and
the output edge commit remain serialized for one run. A full queue applies
backpressure at the next checkpoint boundary; it never drops a required
checkpoint silently. Trackio's presigned multipart upload remains restart-safe
and digest-bound. Default Trackio behavior stays synchronous for external
callers; the framework explicitly opts into background publication and waits on
the returned handles.

The run's success finalization calls `flush_artifacts` before inspecting
required roles or deleting the workspace. Graceful cancellation enters a
bounded finalization scope that stops new training work, publishes the latest
complete selected snapshot if it has not already been queued, and drains
required publications within the provider's stop grace. A cancelled run remains
cancelled even if an optional model-view upload fails, but the failure is
durably visible and no model-ready claim is made. A policy-required recovery
publication failure makes the evidence inconsistent and blocks cleanup until
reconciled; implementation must not rewrite provider cancellation as training
success.

An abrupt process, container, worker, or power loss cannot execute a finalizer.
Recovery is therefore the latest artifact already committed off-host. The live
qualification deliberately kills the producer after a later local checkpoint
exists and proves that planning selects the earlier committed step, never the
uncommitted directory.

### Retention and lineage

Local checkpoint retention and durable artifact retention are separate. The
first production release does not automatically delete committed checkpoint
artifacts during training. It exposes dry-run retention planning that can
remove unreferenced old snapshot views after the run is terminal. A checkpoint
is protected when any downstream run consumes either view, when the artifact is
explicitly retained/pinned, when it is the latest valid recovery point under
policy, or while the producer is nonterminal.

Recovery and model views share a `checkpoint_snapshot_id`, but lineage consumers
link to the exact artifact they used. Exact resume creates a recovery input edge.
Branch, eval, and serve create a model input edge. The model descriptor names the
base model and parent; the recovery artifact does not become a model parent.
Aliases such as `latest-checkpoint` are navigation only and are absent from
execution snapshots.

Trackio's consumer-aware purge remains the byte-deletion authority. Framework
retention planning must preview provider links and CAS references before apply,
use immutable plan digests, and leave a receipt across provider, tracking,
registry, and local planes when those planes are in scope. Artifact manifests
and storage accounting must show that publishing two views does not double
shared adapter blobs.

## Plan of Work

### Milestone 0: clarify product meaning and record the architecture decision

Amend `docs/post-training/03-work-and-evidence.md` so a recovery checkpoint may
be durably retained as a run-scoped recovery artifact while remaining distinct
from a loadable model artifact. Amend
`docs/post-training/06-observation-and-lineage.md` so the train API may publish a
checkpoint-scoped model view from the same run and so only that view enters
model lineage. Amend `docs/post-training/05-apis.md` with exact step selection,
resume versus branch semantics, and the prospective CLI. Do not change the
meaning of stage, job, run, or candidate status.

Create `docs/decisions/0015-checkpoint-scoped-model-artifacts.md` using
`docs/templates/ADR.md`. It records why ordinary projection is not a job, why
typed views remain separate, why packing cannot create learned outputs, why
heavy conversion remains `model.transform`, and why durable commit rather than
local save defines availability. Update `docs/decisions/README.md` and add a
revision history entry. The baseline amendment and ADR must be reviewed before
code changes because later milestones depend on these meanings.

Acceptance for this milestone is a documentation review that can answer, with
no chat context, which artifact each train/eval/serve operation consumes, what
happens after graceful and abrupt failure, and when a transform job is still
required.

### Milestone 1: define snapshot, model descriptor, policy, and capability contracts

Create `packages/train/src/posttrain/train/checkpoints.py`. Define frozen,
JSON-safe values for `CheckpointSnapshotId`, `CheckpointComponent`,
`CheckpointSnapshotManifest`, `CheckpointPublicationPolicy`,
`CheckpointView`, and `CheckpointProjectionCapability`. Put training-specific
state here rather than in `posttrain.common`. Add deterministic read, write,
digest, exact-step, and latest-complete helpers. Pydantic or frozen dataclasses
may be used consistently with nearby train contracts, but unknown schema fields
must fail closed and schema versions must be explicit.

Add `ModelArtifactDescriptor` and its JSON conversion beside `ModelVariant` in
`packages/common/src/posttrain/common/models.py` because train produces it and
train, eval, serve, planning, and Observatory consume it. The descriptor must
not import any backend SDK. Add a method that constructs a new `ModelVariant`
from a descriptor plus an immutable `StoredArtifactRef`, validating artifact
kind against model form and base/renderer compatibility.

Extend `TrainingLoop` and the catalog schema in
`packages/train/src/posttrain/train/profiles.py` and
`packages/train/src/posttrain/train/catalog_schema.py` with a typed publication
policy. Keep current catalog entries behavior-compatible by default: terminal
model publication and existing configured recovery retention continue, while
intermediate durable publication is off until selected. Add explicit LoRA
qualification profiles that publish both views at every test checkpoint. Do not
put upload worker count or S3 settings in algorithm/catalog selections.

Add a backend capability method under
`packages/train/src/posttrain/train/backends/common.py` that reports whether
recovery and model views are supported for each update kind, whether the model
view is adapter-only or full weights, and whether it requires a transform. Job
planning rejects unsupported policies before GPU submission.

Focused tests belong in `packages/common/tests/test_models.py`,
`packages/train/tests/test_checkpoints.py`, and existing catalog/profile tests.
Fixtures include complete and incomplete manifests, two snapshots with ordered
steps, invalid component paths, digest mismatch, adapter-only descriptor
round-trip, incompatible model kind, and a full-update capability that requires
transform.

### Milestone 2: make artifact publication bounded, asynchronous, and durable

In `packages/tracking/src/posttrain/tracking/contracts.py`, formalize artifact
publication without forcing capability packages to import Trackio or W&B.
Add an `ArtifactPublicationHandle` protocol exposing a stable submission ID,
state, and `wait()` result, and add `flush_artifacts()` to `TrackedRun`. Preserve
the existing `Observer.artifact` call as the operation-facing submission method;
concrete immediate backends may return after commit, while background backends
queue the work. Update `packages/work/src/posttrain/work/execution.py` so success
finalization flushes before resolving required outputs and terminal failure or
cancellation invokes the bounded flush path before workspace cleanup. Add a
finalization-only emission path in `RunContext` that cannot restart ordinary
operation work after cooperative cancellation.

Implement the generic background upload feature first in `../trackio`. Keep
`Run.log_artifact` synchronous by default. Add an explicit background option
returning an artifact whose `wait()` blocks for the committed version. Use a
bounded executor and staged-byte budget, direct multipart transport for S3,
idempotent blob digests, serialized version/edge commit per run, exception
propagation, cancellation, and finish-time drain. No signed URL or token may be
persisted in the task receipt. Add unit tests for two concurrent artifacts,
backpressure, missing blob retry, digest failure, duplicate submission,
finish-time drain, and failure propagation. Add a separate-process test against
a real local S3-compatible service or the existing RustFS canary path proving
that the producer uploads directly and Trackio server memory does not scale with
artifact size.

Add an authenticated, read-only artifact-integrity operation to Trackio that
checks one exact artifact version's stored manifest and reports whether every
content-addressed blob is still present in the selected artifact store. Return
only bounded missing-digest evidence and never a signed URL. Expose this as an
optional provider-neutral tracking capability; providers without it return
`unsupported`, after which CLI deep verification remains available. Unit and
real-S3 tests must cover a complete artifact, a removed blob, a stale direct
upload receipt, authorization, and response redaction.

Update `../trackio/CARBONTEQ_FORK.md` and
`docs/tooling/trackio/README.md` in their respective repositories. Run the fork
test and build ladder, commit and push the fork, publish a new immutable
`carbonteq-trackio` release, verify distribution/import version and hashes in a
clean environment, then update
`packages/tracking-trackio/pyproject.toml` and `uv.lock`. Do not update the
framework dependency to a working-tree SHA or claim release before the package
exists in the selected index.

Implement `TrackioTrackedRun.artifact` as background submission using the new
fork API, append `PublishedArtifact` only after `wait()` returns a committed
identity, and make `published_artifacts()` fail or flush explicitly rather than
returning an incomplete prefix. W&B may remain synchronously committed behind a
completed handle, but it must pass the same tracking conformance tests. Add
provider-neutral conformance cases for publication ordering, duplicate logical
names, required failure, flush idempotence, and no workspace deletion before
commit.

### Milestone 3: publish both views from TRL LoRA checkpoints

Add shared projection helpers under
`packages/train/src/posttrain/train/backends/retention.py` or a new focused
`checkpoint_projection.py`. One helper validates a complete native checkpoint
and writes the versioned snapshot manifest. Another creates the adapter-only
model view from the allowlisted components and writes `model-variant.json`.
Both compute deterministic content digests. Reuse
`validate_adapter_only_directory` for recovery and add a separate strict
`validate_model_adapter_directory` that rejects every recovery-only or
full-model component.

Install one framework-owned TRL `TrainerCallback` from the shared builder in
`packages/train/src/posttrain/train/backends/trl/common.py`. Its post-save hook
receives the exact saved step, validates that checkpoint, creates selected
views, and submits artifacts. Reuse it from SFT, DPO, GRPO/DAPO/OLMo3, SAMPO,
and on-policy distillation modules. Do not copy callbacks into every technique
module and do not change the TRL fork unless its public callback contract is
proven insufficient by a focused reproducer.

Refactor normal completion and
`publish_interrupted_recovery_checkpoint` to use the same snapshot and
projection code. Deduplicate by snapshot ID and view kind so a checkpoint
already queued during training is not uploaded again on cancellation or
terminal success. On LoRA and QLoRA, assert that neither recovery nor model view
contains base weights. Terminal success reuses the final committed model view
when possible and still publishes the existing summary and retention manifest.

Instrument `checkpoint_saved`, `checkpoint_projection_started`,
`checkpoint_publication_queued`, `checkpoint_view_committed`, and
`checkpoint_publication_failed` events plus bounded timing/queue metrics. Keep
runtime phase boundaries accurate: local save, projection, upload, and final
drain are distinct from actor update and rollout generation.

Add focused tests in `packages/train/tests/test_trl_common.py`,
`test_sft_validation.py`, `test_dpo_observability.py`, GRPO tests, and a new
shared projection suite. Prove two steps produce four checkpoint-view artifacts
with unique logical names, a terminal model role remains singular, cancellation
does not duplicate the latest step, interrupted upload is retried, and every
model adapter passes a recursive assertion that no base or trainer state is
present.

### Milestone 4: resolve checkpoint sources for every model-consuming job

Add provider-neutral checkpoint queries to `packages/tracking`. A
`CheckpointSelector` carries source run ID, exact non-negative step or
`latest-complete`, and required view `recovery` or `model`. A resolver reads the
source run's output artifact edges, groups them by `checkpoint_snapshot_id`,
rejects missing, duplicate, incomplete, conflicting, or digest-less entries,
and returns a `ResolvedCheckpointSource` containing exact immutable identities.
It never downloads bytes during planning.

Add pure checkpoint inspection values and functions beside the snapshot
contracts in `packages/train/src/posttrain/train/checkpoints.py`. They accept
provider-neutral artifact records rather than importing Trackio or a tracking
backend. `CheckpointInspection` represents one grouped snapshot;
`CheckpointVerification` represents metadata, server-presence, and optional
deep-file checks; `CheckpointDiff` represents manifest and descriptor changes.
The CLI and Observatory adapt their existing `ArtifactLink` values into these
pure functions so both surfaces use one interpretation of view pairing and
LoRA safety without moving job-specific semantics into `packages/tracking`.

Register a nested checkpoint Typer application from
`apps/cli/src/posttrain_cli/commands/run_cmd.py`, with implementation helpers in
a focused `apps/cli/src/posttrain_cli/commands/checkpoint.py` module. Implement
`list`, `show`, `verify`, and `diff` exactly as described in the command
contract. All four commands are read-only. Bound list pages to at most 200
snapshots, bound manifest output, require explicit `--deep` before downloading
bytes, use project state for temporary downloads, and redact presigned URLs,
tokens, provider credentials, and unsafe absolute worker paths. JSON output uses
versioned response models and stable field names rather than terminal text
parsing.

Extend `JobDefinition` in `packages/work/src/posttrain/work/contracts.py` with
declared model input roles. Each role names a `ModelVariant` seat, the related
`InferenceBinding` seats that must be rebound with it, and whether it is the
default branch target. Update framework definitions in
`packages/jobs/src/posttrain/jobs/definitions.py`: SFT and DPO target `model`;
GRPO/OLMo3/SAMPO target policy `model` and rollout inference; distillation
targets `student` by default and its student rollout inference while retaining
an explicit teacher role; eval targets `model` and evaluation inference; serve
targets `model` and its inference binding. Static validation must reject a
declared role whose seats do not exist or disagree.

Replace the global `model_adapter` and `model_weights` input names in
`packages/work/src/posttrain/work/runner.py` with seat-scoped names, such as
`model/model_adapter` and `student/model_adapter`, while retaining a documented
compatibility read for existing packed images. Add
`override_job_model_source`, parallel to execution-target override, that
replaces the selected `ModelVariant`, related inference bindings, resolved
snapshot, and artifact input as one immutable operation. It reconstructs the
model only from the committed model descriptor and stored artifact. It cannot
accept `training-checkpoint`.

Extend `apps/cli/src/posttrain_cli/commands/job.py`,
`commands/work_package.py`, and `execution_planning.py` with
`--checkpoint-step`, `--model-from-run`, and `--model-seat`. Change
`--resume-from-run` to use the same selector with the recovery view. Reject
source flags for in-process compatibility execution until that path uses the
same immutable planning and materialization contract. Reject resume plus model
source together, checkpoint step without a source, ambiguous roles, incompatible
base/backend/update state, and a model source whose descriptor disagrees with
the artifact kind.

Update execution package manifests and `apps/runtime` validation so the
resolved source is part of actual-job identity. A change from step 25 to step 50
must produce a different execution package and image identity even if all YAML
is unchanged. At runtime the existing tracking materializer downloads only the
selected typed artifact. Eval and serve never download recovery state; resume
never substitutes the model view.

Add CLI and planning tests for paginated listing, exact and interactive-latest
inspection, summary-versus-manifest fetch boundaries, deep verification cleanup,
LoRA forbidden-file detection, incomplete snapshot pairs, digest mismatch,
manifest diff, exact and latest source selection, no moving execution aliases,
ambiguous distillation seats, model/inference rebinding, adapter and full-weight
inputs, resume compatibility, package identity, and expected developer-facing
errors. Update `docs/getting-started.md`, `docs/developer-experience.md`, and the
Trackio consumer page with concise inspection and reuse workflows that do not
require catalog edits.

### Milestone 5: negotiate veRL and full-parameter support explicitly

Extend the veRL result contract and worker under
`packages/train/src/posttrain/train/backends/verl/` to emit the same snapshot
manifest and capability result. First qualify adapter-only veRL checkpoints if
their retained layout exposes a complete loadable adapter without actor or
critic shards. If the layout cannot satisfy the model-view validator, publish
recovery only and reject a policy requesting checkpoint model views before
submission. Preserve existing shared checkpoint cadence and retention mappings.

For full-parameter updates, distinguish a directly loadable checkpoint from a
native distributed training checkpoint. A directly loadable safetensors model
may be exposed as `model-weights` inside the original run. A sharded optimizer,
FSDP, Megatron, or veRL checkpoint that requires gather, merge, reshard, or
format export reports `requires_transform=true`. The planner then rejects
automatic model-view publication with an actionable message naming the required
`model.transform` definition; it must not silently schedule one. This preserves
the user's rule that ordinary projection stays in the producer while admitting
that real conversion is separate work.

Use a tiny CPU model fixture to qualify full-weight manifests and a backend
fixture to prove transform-required rejection. Run a bounded real veRL LoRA
canary only after its package and checkpoint layout are immutable and available.
No unqualified backend may advertise model-ready checkpoints in Observatory.

### Milestone 6: present checkpoint readiness without overfetching

Add a provider-neutral checkpoint projection in
`apps/observatory/src/posttrain_observatory/models.py` and the query service.
The run overview receives only a checkpoint count, latest complete step, latest
recovery-ready step, latest model-ready step, pending count, failed count, and
last publication timestamp. The detailed list is fetched only when the user
opens the Artifacts & lineage checkpoint section. It is paginated by step and
does not download artifact manifests or files unless one detail row is opened.

The detailed ledger groups recovery and model artifacts by snapshot ID. Each row
shows step, save time, update kind, base revision, recovery readiness, model
readiness and kind, size, provider version, digest prefix, publication duration,
and safe failure state. It must distinguish `saved locally`, `queued`,
`uploading`, `committed`, and `failed`; only committed rows offer copyable source
commands. Observatory remains read-only: actions copy or explain the exact
`posttrain job run` command and do not submit work.

Add training telemetry definitions for checkpoint timing and publication health
in `apps/observatory/src/posttrain_observatory/telemetry.py`. Update the frontend
labels and Artifacts & lineage components in
`apps/observatory/frontend/src/App.tsx`, with focused API and UI tests. Verify
that runs with hundreds of checkpoints load the overview through one bounded
aggregate query and that the detailed tab requests one page rather than every
artifact or trace.

### Milestone 7: add safe retention and historical adapter backfill

Extend framework retention planning and the existing purge architecture to
understand checkpoint snapshot groups. Preview identifies protected downstream
consumers, retained aliases, the latest policy-required recovery point, shared
CAS blobs, and reclaimable unique bytes. Apply remains digest-bound and
idempotent. Deleting one unreferenced view must not remove blobs still referenced
by the sibling view or another artifact version. No automatic deletion occurs
while the producer run is active or artifact publication is pending.

Implement the historical
`posttrain run checkpoint publish-model` command in `apps/cli`. It uses the same
selector, validator, projector, and tracking publisher as prospective training,
but runs locally as an audited administrative action. Dry-run reports the
source recovery identity, expected adapter file set, forbidden-file scan,
estimated bytes, target logical name, and immutable plan digest. Apply requires
that digest, publishes at most once, attaches the output to the original run,
and writes a machine-scoped receipt. Trackio needs an authenticated exact-run
late-artifact commit that is idempotent and audit-recorded; it must not reopen
metrics, alter the terminal status, or create a provider job.

Use the retained Ambient OLMo3 step-50 recovery checkpoint as the first real
backfill only after the generic disposable fixture passes. Verify its projected
artifact contains LoRA files and descriptor metadata only, then run the planned
small evaluation from that model view. Preserve the existing recovery artifact
and production lineage. Do not treat the scratch `model_adapter` input directory
as the recovery source.

### Milestone 8: qualify the complete lifecycle and release it

Start with provider-free unit and integration tests. Then deploy the exact new
Trackio release against a dedicated S3 bucket or prefix and a disposable project.
Run a two-step LoRA producer with checkpoint cadence one and background
publication enabled. While the producer is still running, query Trackio and
Observatory and prove the first model view becomes committed and selectable.
Record training step time separately from artifact upload time.

From step one, run four consumers: exact resume to step two, a fresh optimizer
training branch, a bounded evaluation, and a bounded serving smoke. Inspect
their input edges and actual-job manifests. The resume must restore trainer,
optimizer, scheduler, RNG, and adapter state. The branch must start new optimizer
state. Eval and serve must load only the model adapter plus immutable base. The
provider history must contain the producer and four requested consumers and no
materialization run.

Repeat with graceful cancellation after a complete checkpoint. Confirm the
cancelled producer drains its selected publications inside the provider grace
period and remains terminal cancelled. Repeat with an abrupt kill after a later
local checkpoint but before its upload commits. Confirm `latest` resolves to the
last committed earlier step. Restart the Trackio service during a multipart
upload and prove idempotent resume, digest verification, and one committed
artifact version.

Measure local and remote bytes. For the LoRA fixture, publishing recovery and
model views must not duplicate base weights, and the object store's unique blob
growth must approximately equal the union of the two file sets rather than the
sum of two directory sizes. Exercise reference-aware purge in dry-run and apply
on disposable snapshots, then prove a consumed checkpoint remains blocked.

After all evidence passes, run the repository validation ladder, build the
framework distributions and immutable runtime images, update release manifests
and fork ledgers, deploy a canary, and only then promote the framework release.
Trackio must be committed, pushed, published, and deployed before the framework
pin and images are finalized. If a TRL fork change became necessary, publish it
before the framework as well. Record exact commits, package hashes, image
digests, run IDs, artifact versions/digests, storage measurements, and purge
receipts in this plan and the relevant tooling pages.

## Concrete Steps

All framework commands run from `/home/hammad/projects/rl` unless another
working directory is stated. Before implementation and at every stopping point:

    git status --short
    git -C ../trackio status --short
    git -C ../trl status --short

Do not proceed across a repository boundary until overlapping dirty changes are
classified. Milestone 0 documentation checks are:

    uv run pytest packages/common/tests packages/train/tests -q
    git diff --check

During Milestone 1, run the smallest contract suites first:

    uv run pytest packages/common/tests/test_models.py \
      packages/train/tests/test_checkpoints.py \
      packages/train/tests/test_api.py -q
    uv run ruff check packages/common packages/train
    uv run pyright packages/common packages/train

During the Trackio fork milestone, run from `/home/hammad/projects/trackio` the
focused upload tests selected by the fork ledger, followed by its complete
CPU-safe test and build ladder. The exact commands must be refreshed from the
fork's current `CARBONTEQ_FORK.md` before execution; record their outputs here.
After publication, verify in a clean temporary environment that distribution
metadata and `trackio.__version__` match and record the wheel and sdist hashes.

After updating the framework Trackio pin:

    uv lock
    uv sync --all-packages --locked --python 3.13
    uv run pytest packages/tracking/tests packages/tracking-trackio/tests \
      packages/tracking-wandb/tests packages/work/tests -q

For checkpoint projection and planning:

    uv run pytest packages/train/tests packages/jobs/tests packages/work/tests \
      apps/cli/tests apps/runtime/tests packages/execution-pack/tests -q
    uv run ruff check packages/train packages/jobs packages/work apps/cli apps/runtime
    uv run pyright packages/train packages/jobs packages/work apps/cli apps/runtime
    uv run lint-imports

For Observatory:

    uv run pytest apps/observatory/tests -q
    npm --prefix apps/observatory/frontend test
    npm --prefix apps/observatory/frontend run check
    npm --prefix apps/observatory/frontend run build

Before live GPU work, plan and inspect the exact job without submitting:

    posttrain --json job plan <producer-work-package> --job <train-job>
    posttrain --json job pack <producer-work-package> --job <train-job>

The output must include the checkpoint publication policy and backend capability
result but no future artifact identity. After step one commits:

    posttrain --json run checkpoint list <producer-run-id>
    posttrain --json run checkpoint show <producer-run-id> --step 1 --files
    posttrain --json run checkpoint verify <producer-run-id> --step 1 --view both
    posttrain --json run checkpoint verify <producer-run-id> --step 1 --view model --deep
    posttrain --json job plan <eval-work-package> --job <eval-job> \
      --model-from-run <producer-run-id> --checkpoint-step 1

After step two commits:

    posttrain --json run checkpoint diff <producer-run-id> \
      --from-step 1 --to-step 2 --view model

The consumer plan must include exact provider version and digest and must not
contain a materialization job. Use disposable run IDs and storage prefixes for
failure, restart, and purge tests. Never kill or purge the retained Ambient
production lineage while qualifying generic behavior.

The final framework validation ladder is:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Update this section with exact test counts, skips, live commands, run IDs, and
observed outputs as milestones complete. Do not copy expected counts from an
older release.

## Validation and Acceptance

The change is accepted only when all of the following behaviors are directly
observed.

A complete LoRA checkpoint produces exactly one recovery view and one model
view with the same snapshot ID and step. The recovery view restores exact
training state and contains no base-model tensors. The model view loads in vLLM
and contains no optimizer, scheduler, trainer, RNG, or base-model tensors. Its
descriptor reconstructs a valid `ModelVariant` using the exact published
adapter version and immutable base revision.

Publication is nonblocking within its configured bound. Training records local
checkpoint/projection time separately from upload time. A queued artifact is not
returned by checkpoint resolution. Final success drains required publications
before workspace cleanup. Graceful cancellation preserves the latest complete
selected checkpoint. Abrupt loss exposes only the last off-host committed
checkpoint. Trackio restart resumes rather than duplicates a multipart upload.

The CLI resolves an exact step and immutable artifact identity. `latest` in a
plan becomes an exact step in the package. Resume, branch, eval, and serve use
the correct typed view and record input edges. An attempt to eval a recovery
artifact, resume from a model artifact, omit an ambiguous model seat, use an
incompatible base/backend schema, or request an unsupported full-weight export
fails during planning with an actionable message and starts no provider run.

Checkpoint inspection is complete enough to diagnose source readiness without
opening Trackio directly. List remains paginated and does not fetch manifests.
Show reports both typed views and compatibility; `--files` displays only the
bounded manifest. Verify detects a missing blob, mismatched digest, malformed
descriptor, incomplete recovery state, and forbidden LoRA file. Deep verify
recomputes local digests without loading a model or leaving downloaded state.
Diff reports manifest and descriptor changes without claiming tensor-level
numerical comparison. Every command is read-only and prints or returns the
exact step when `latest` is used.

The original train run is the only producer of prospective checkpoint views.
The provider contains no hidden projection/materialization run. The historical
backfill is local CPU work with dry-run/apply receipts and attaches an audited
artifact edge without altering terminal status.

Observatory overview loading remains bounded as checkpoint count grows. It
shows latest readiness rather than every artifact. The detailed checkpoint
ledger loads only on demand, paginates, groups both views correctly, and never
labels queued/local state as usable. Its copied commands contain exact step
selection.

Retention preview protects active, latest-required, pinned, and consumed
snapshots. Applying a disposable purge removes only unreferenced manifests and
unique blobs. Shared adapter blobs survive while any sibling or downstream
artifact references them. Provider, Trackio, object-storage, and local receipts
agree before cleanup is called complete.

The final release uses published Trackio and, if changed, TRL artifacts from
immutable commits. The framework pin, lockfile, job-kind images, actual-job
image, deployed Trackio service, and recorded runtime provenance agree exactly.

## Idempotence and Recovery

Snapshot creation writes into a temporary directory and atomically renames only
after validation and manifest completion. Repeating projection for the same
source digest, step, view kind, and schema returns the same content digest.
Repeating publication either resumes missing multipart parts or observes the
already committed artifact; it must not allocate a second version silently.

The publication queue persists enough digest-bound state to resume transfers
while the producing process remains available, but a host loss is recovered
from Trackio/S3 session state and the last committed manifest, not from an
in-memory future. Abandoned multipart sessions remain subject to Trackio's
existing age-based cleanup and cannot be mistaken for artifacts.

`job plan`, `job pack`, and checkpoint list/show/verify/diff are read-only. Historical
backfill requires a dry-run plan digest before apply. Retention and purge remain
dry-run by default and consumer-aware. Live qualification uses dedicated
projects, prefixes, run IDs, and small artifacts so failed attempts can be
removed without touching retained production evidence.

If a milestone fails, preserve its exact logs and artifact/session IDs, update
`Surprises & Discoveries`, and retry only that boundary. Do not reset the
framework's dirty worktree. Do not update package pins until fork publication
succeeds. Do not remove compatibility reads until new producer and consumer
images have been deployed and old queued jobs have drained.

Rollback is additive. Disable intermediate publication in catalog policy while
retaining terminal model and recovery behavior. Revert the framework package
pin only to a still-deployed compatible Trackio release. Previously committed
checkpoint artifacts remain readable by immutable version. A schema reader must
continue to support version 1 for the declared compatibility window even if a
later writer version is introduced.

## Artifacts and Notes

The intended checkpoint list output is structurally equivalent to:

    run: ambient-example
    step  recovery          model               state
    25    v3 sha256:...     v7 sha256:...        ready
    50    v4 sha256:...     queued               partial

The intended checkpoint show output is structurally equivalent to:

    checkpoint: ambient-example/step-00000025
    backend: trl@a82ecebc...
    update: lora
    base: Qwen/Qwen3.5-2B@15852e8c...
    recovery: ready, training-checkpoint, v3, sha256:...
    model: ready, model-adapter, v7, sha256:...
    exact resume: supported
    model consumers: train branch, eval, serve

With `--files`, the model view must contain adapter and descriptor roles and no
optimizer, scheduler, trainer, RNG, or base-weight role. Actual filenames remain
backend-versioned evidence rather than a hard-coded UI contract.

The intended eval plan source excerpt is structurally equivalent to:

    model_source:
      source_run_id: ambient-example
      checkpoint_step: 25
      checkpoint_snapshot_id: ambient-example/step-00000025
      artifact_kind: model-adapter
      provider: trackio
      version: v7
      digest: sha256:...
      target_model_seat: model

The intended event order for one asynchronously committed checkpoint is:

    checkpoint_saved
    checkpoint_projection_started
    checkpoint_publication_queued       view=recovery
    checkpoint_publication_queued       view=model
    checkpoint_view_committed           view=recovery
    checkpoint_view_committed           view=model

These are examples of shape, not hard-coded identities. Replace them with real
retained evidence during implementation.

## Interfaces and Dependencies

The following names are the intended public or cross-package interfaces. A
milestone may refine field spelling, but it must update this plan before
shipping a semantically different contract.

In `packages/train/src/posttrain/train/checkpoints.py`:

    CheckpointSnapshotId(run_id: str, global_step: int)
    CheckpointComponent(role: str, relative_path: str, size_bytes: int, sha256: str)
    CheckpointSnapshotManifest(..., complete: bool, components: tuple[CheckpointComponent, ...])
    CheckpointPublicationPolicy(...)
    CheckpointProjectionCapability(update_kind: str, recovery: bool, model_view: bool, requires_transform: bool)
    project_checkpoint_views(checkpoint: Path, manifest: CheckpointSnapshotManifest, ...)

In `packages/common/src/posttrain/common/models.py`:

    ModelArtifactDescriptor.from_model_variant(model: ModelVariant, ...)
    ModelArtifactDescriptor.to_model_variant(reference: StoredArtifactRef, kind: str) -> ModelVariant

In `packages/tracking/src/posttrain/tracking/contracts.py`:

    ArtifactPublicationHandle.wait(timeout: float | None = None) -> PublishedArtifact
    TrackedRun.flush_artifacts(timeout: float | None = None) -> tuple[PublishedArtifact, ...]
    ArtifactIntegrityResult(... state: verified|failed|unsupported, bounded failures ...)
    RunDataSource.verify_artifact(reference: StoredArtifact) -> ArtifactIntegrityResult

In the checkpoint read/planning boundary:

    CheckpointSelector(source_run_id: str, step: int | LatestComplete, view: Literal["recovery", "model"])
    ResolvedCheckpointSource(... exact artifact/version/digest/model descriptor ...)
    resolve_checkpoint(source: RunDataSource, selector: CheckpointSelector) -> ResolvedCheckpointSource
    CheckpointInspection(... grouped snapshot, readiness, compatibility, bounded summaries ...)
    CheckpointVerification(... checks, failures, downloaded_bytes, deep ...)
    CheckpointDiff(... descriptor changes, added/removed/changed components ...)

In `packages/work`:

    ModelInputRole(name: str, model_seat: str, inference_seats: tuple[str, ...], default: bool = False)
    override_job_model_source(..., role: str | None, source: ResolvedCheckpointSource) -> PlannedJobExecution

Trackio remains the concrete source of direct multipart URLs, artifact version
allocation, run-artifact edges, committed manifests, and S3-compatible storage
verification. dstack remains the execution provider and supplies the stop grace
needed for final drain. TRL and veRL remain private trainer backends; their
native checkpoints are not public framework APIs. Observatory consumes only
provider-neutral tracking read contracts and never imports Trackio directly.

No new dependency is expected in the framework. The Trackio fork may use its
existing `concurrent.futures`, `httpx`, hashing, and direct-upload machinery.
Any proposed dependency addition must be justified in the Decision Log and
locked before implementation proceeds.

Revision note (2026-08-09): Created this production execution plan after
rejecting a separate materialization-job design in favor of checkpoint-scoped
recovery and model views produced by the original training run. The plan also
records the packing boundary, asynchronous publication requirement, generic
train/eval/serve source DX, heavy-transform exception, historical adapter
backfill, and release qualification sequence.

Revision note (2026-08-09): Added the developer-requested checkpoint inspection
surface. The plan now specifies bounded list, detailed show, metadata/deep
verify, and manifest diff commands, their shared pure interpretation layer,
security and cleanup behavior, and acceptance tests.

Revision note (2026-08-09): Implemented Milestone 0. The canonical baseline
and ADR 0015 now make recovery/model view separation, run-scoped inspection,
exact planning selectors, and the no-separate-materialization-job boundary
explicit before runtime contracts are added.

Revision note (2026-08-09): Began Milestone 1 by adding the provider-neutral
checkpoint and model-view contract types, extending tracking protocols with
publication/verification seams, and adding six deterministic tests. Concrete
Trackio publication remains in the next milestone.

Revision note (2026-08-09): Implemented the first runtime seam. TRL save
callbacks create a checkpoint-scoped recovery artifact and, for LoRA/QLoRA, an
adapter-only model view inside the producing run. The Trackio writer drains
bounded background uploads before finalization. The CLI and job definitions
can inspect and select immutable views without scheduling a materialization
job. Release qualification remains open until the Trackio internal publish
workflow succeeds and one continuation plus one evaluation consume a committed
producer artifact.

Revision note (2026-08-09): Published the immutable Trackio post12 wheel and
sdist directly to the stable internal index after the GitHub LAN-runner queue
remained unavailable. Updated the framework dependency, workspace lock, and
OCI runtime constraint lock to the verified post12 wheel digest. The existing
published-image manifest now fails closed until the release candidate rebuilds
job-kind images against that changed lock.

Outcomes & Retrospective
------------------------

The implementation has reached a testable vertical slice, but not a complete
release. The strongest result so far is that recovery and model reuse now share
one producing run and one checkpoint identity while retaining separate artifact
kinds and runtime contracts. The remaining work is evidence-based: publish and
pin the Trackio fork, run the continuation/evaluation acceptance pair, validate
terminal/cancellation behavior on the actual provider, then create and verify
the Posttrain release and update Ambient Agent.
