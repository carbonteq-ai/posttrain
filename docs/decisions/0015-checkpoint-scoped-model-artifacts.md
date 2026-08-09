# ADR 0015 — Checkpoint-scoped recovery and model artifacts

Status: Accepted
Date: 2026-08-09
Deciders: Posttrain framework and AI infrastructure maintainers
Related Plan: `docs/plan/checkpoint-scoped-model-artifacts.md`
Supersedes: None
Superseded By: None

## Context

A training run can save a native checkpoint containing parameters and the
trainer state needed to continue. The current lifecycle treats that directory
as recovery state and generally exposes a terminal model artifact separately.
That leaves a developer without a safe way to evaluate a mid-run checkpoint,
cancel a run while retaining a loadable model, or start a new train, eval, or
serve run from the last failed checkpoint. It also encourages a separate
materialization job even when a LoRA adapter can be projected from files that
already exist in the producing run.

The distinction matters. A `training-checkpoint` contains optimizer,
scheduler, trainer, random-number-generator, and parameter state for exact
resume. A model view (`model-adapter` or `model-weights`) is inference-loadable
and must not expose training-only state. A `job` is a reusable definition;
each execution is a `run`, so inspection belongs under a concrete run while
source selection belongs to `job run` planning. The framework must preserve
these meanings across local and dstack providers, Trackio persistence, and
Observatory lineage.

## Decision

1. The original training run owns publication of selected complete checkpoint
   snapshots. Ordinary checkpoint projection is not a separate provider job.
2. Every published snapshot may expose two typed views linked by one immutable
   `checkpoint_snapshot_id`: a `training-checkpoint` recovery view and a
   `model-adapter` or `model-weights` model view. Only the model view enters
   model lineage and is accepted by train branches, eval, and serve.
3. LoRA and QLoRA model views contain adapter files plus the model-interface
   descriptor and safe tokenizer/renderer metadata. They never copy immutable
   base weights, optimizer, scheduler, trainer, scaler, random state, or
   training arguments. Full-parameter model views require an explicit backend
   capability check and must already be loadable; merge, quantization, export,
   and resharding remain explicit `model.transform` jobs.
4. Packing validates the publication policy, storage configuration, and backend
   capability but cannot materialize learned weights before training reaches a
   checkpoint.
5. Publication uses a bounded asynchronous queue inside the producing run.
   Only a committed manifest, referenced blobs, and run-artifact edge are
   selectable. Success and graceful cancellation drain required work before
   evidence reconciliation; abrupt loss makes the latest already-committed
   view the recovery boundary. Optional model-view failure is visible and does
   not rewrite a cancelled run as success.
6. Checkpoint inspection is a read-only, run-scoped API and CLI:
   `posttrain [--json] run checkpoint list|show|verify|diff RUN_ID`. Exact
   selectors are printed in output. Consumers use `job run --resume-from-run`
   for recovery or `--model-from-run` for a model view; `latest` is resolved at
   planning time and never followed by a worker.
7. Observatory presents the lightweight snapshot summary first and fetches
   detailed manifests lazily. Trackio owns artifact metadata, content-addressed
   blobs, and output edges; dstack owns execution only.

## Consequences

Developers can inspect a run's checkpoints, evaluate a committed intermediate
model, branch training from it, or resume exact trainer state without learning
provider-specific paths. LoRA publications are small and do not duplicate base
model storage. A single producing run is easier to understand in lineage and
avoids a second scheduling and failure boundary.

The producer now owns bounded upload capacity and must report pending or failed
publication explicitly. Checkpoint retention consumes durable storage, so
local checkpoint limits and provider artifact retention remain separate and
must be reconciled with references. Full-parameter and distributed backends
need capability evidence before they can publish model views. Existing
recovery-only runs remain valid but require an explicit, idempotent CPU
backfill before they can be consumed as model views.

## Alternatives Considered

### A separate materialization job for every checkpoint

Rejected for ordinary LoRA projection. It adds provider cost, latency, another
lineage node, and another failure mode without changing parameters. It remains
valid for representation-changing transforms.

### One universal checkpoint artifact accepted everywhere

Rejected because eval and serving must never load optimizer or random state,
and an exact resume must not silently start from an adapter-only model view.
Typed views and explicit source modes make invalid substitutions impossible.

### Materialize learned weights during packing

Rejected because packing happens before training and cannot manufacture future
parameters. It may carry policy and validation only.

### Synchronous upload on every checkpoint

Rejected as the default because object-storage latency would distort training
step time. The framework opts into bounded background publication and a final
drain; Trackio's direct multipart transport remains the transfer primitive.

### Implicit moving `latest` at worker runtime

Rejected because a retry could consume a different checkpoint. Planning resolves
the alias to an exact step, artifact version, digest, base revision, and model
seat before submission.

## Implementation Notes

- Milestone 0 amends the canonical contracts in
  `docs/post-training/03-work-and-evidence.md`, `05-apis.md`, and
  `06-observation-and-lineage.md`.
- Milestone 1 adds provider-neutral snapshot, descriptor, publication-policy,
  and capability contracts under `packages/common`, `packages/train`, and
  `packages/tracking` with deterministic fixtures.
- Milestone 2 adds bounded publication lifecycle support and a qualified
  Trackio release before updating the framework's immutable dependency pin.
- Milestone 3 adds TRL LoRA projection and interruption/cancellation coverage;
  later milestones add source planning, full-parameter backends, Observatory,
  retention/backfill, and live qualification.
- No separate dstack materialization job is introduced for pure projection.
  Any required Trackio or TRL fork change must be committed and published in
  its owning repository before this repository updates a dependency pin.

## Revision History

- 2026-08-09: Initial accepted decision. Reason: establish a developer-facing
  contract for checkpoint inspection, exact recovery, and loadable model views
  before implementing runtime publication.
