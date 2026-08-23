# Make run cleanup, retention, and purge one coherent lifecycle

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

This plan is self-contained. It defines the product meaning and implementation
sequence for terminal execution cleanup, retained run history, and intentional
cross-plane erasure. Local rebuildable job-pack and BuildKit storage is governed
separately by `docs/plan/bounded-job-pack-cache-lifecycle.md`.

## Purpose / Big Picture

A developer should not have to delete a run merely because it finished or
because a workstation is short on space. After this work, normal terminal runs
automatically release their exact provider workspace and local run-scoped image
only after durable evidence has reconciled. Their metrics, traces, checkpoints,
artifacts, lineage, and compact control receipts remain available under an
explicit retention policy. Local rebuildable cache continues to be reclaimed by
`posttrain cache prune` without contacting a provider, registry, or tracking
backend.

`purge` has one narrow purpose: permanently erase a logical run or project and
the remotely and locally stored resources exclusively owned by that deletion
closure, while retaining only a minimal audit tombstone. It is appropriate for
disposable smoke history, corrupt or sensitive evidence, project
decommissioning, and security or legal erasure. It is not routine completion,
archiving, cache management, registry garbage collection, or a way to make an
old but valid run disappear from the default view.

The developer-facing workflow is deliberately small:

    posttrain run cleanup RUN_ID
    posttrain run purge RUN_ID --reason REASON
    posttrain project purge --reason REASON
    posttrain purge show PURGE_ID
    posttrain purge apply PURGE_ID --expect-digest sha256:... --yes

Cleanup is normally automatic; its command remains a retry and diagnosis
surface. Both run and project purge commands are read-only previews that print
the exact deletion closure, shared resources that will be retained, blockers,
warnings, estimated logical bytes, and the immutable plan digest. Apply is the
only mutation. A successful purge leaves a local operator-audit tombstone for
`posttrain run list --include-purged`; Observatory omits purged runs without
retaining a separate erasure record.

This plan changes the frozen product baseline narrowly. The baseline already
defines disposable execution workspaces, immutable evidence, lineage, and
`posttrain run cleanup`, but `docs/post-training/05-apis.md` does not define the
already-implemented public purge commands or a cross-plane tombstone. Before
implementation changes, amend `docs/post-training/03-work-and-evidence.md`,
`docs/post-training/05-apis.md`, and
`docs/post-training/06-observation-and-lineage.md` with the vocabulary and
contracts below. Do not treat current code as authority merely because it
already exposes purge.

## Progress

- [x] (2026-08-23) Read the canonical work/evidence, API, and
  observation/lineage contracts and the current cleanup, purge-planning,
  tracking-lifecycle, CLI, and integration-test surfaces.
- [x] (2026-08-23) Separated four lifecycle meanings: evidence-gated cleanup,
  local rebuildable cache pruning, policy-driven retention, and intentional
  cross-plane purge.
- [x] (2026-08-23) Recorded the target developer experience, ownership rules,
  tombstone boundary, safe retry model, and baseline-amendment requirement in
  this companion plan.
- [x] (2026-08-23) Completed Milestone 1: amended the canonical work/evidence,
  API, and observation/lineage contracts with the cleanup, retention, purge,
  ownership-closure, and privacy-bounded tombstone semantics.
- [x] (2026-08-23) Completed Milestone 2: v2 purge-plan schema requires a
  validated, digest-bound reason; v1 plans remain audit-readable; CLI previews
  require `--reason`; privacy-bounded tombstones persist only per-plane
  outcomes; `standard`/`pinned` retention is resolved from a work package into
  RunSpec, admission, submission, tracking configuration, and purge planning.
- [x] (2026-08-23) Completed Milestone 3: controller and manual reconcile
  clean only after settled reconciliation and before admission release. A live
  Ambient dstack run retained six Trackio artifacts while releasing its exact
  worker workspace; a retry completed the persisted cleanup plan safely.
- [x] (2026-08-23) Completed Milestone 4: reference-aware purge closure is
  planned across all storage planes. A live Ambient disposable local run
  qualified the exact provider, registry, Trackio, and local deletion sequence
  with a digest-bound receipt and completed tombstone.
- [x] (2026-08-23) Completed Milestone 5: CLI audit presentation exposes the
  machine-local tombstone. Observatory intentionally omits purged runs and
  does not require a Trackio tombstone feed.
- [ ] Milestone 6: qualify local and remote behavior with disposable fixtures,
  then release it without purging retained research runs.

## Surprises & Discoveries

- Observation: the existing purge architecture is already cross-plane rather
  than a synonym for local cache pruning.
  Evidence: `packages/execution/src/posttrain/execution/purge.py` defines
  provider, registry, tracking, and local action planes, and
  `apps/cli/src/posttrain_cli/purge_surface.py` supplies adapters for each.

- Observation: normal execution cleanup already enforces most of the right
  safety boundary.
  Evidence: `cleanup_execution()` reconciles retained evidence, writes
  `cleanup-plan.json` before mutation, retains a bounded diagnostic for failed
  startup, targets the exact provider handle and canonical run workspace, and
  returns an existing `cleanup.json` receipt on retry.

- Observation: actual-job images are package resources and can be shared by
  several runs, so run age cannot decide image deletion.
  Evidence: `purge_planner.py` groups selected owners by digest-pinned manifest,
  retains the manifest when an unselected owner exists, and schedules at most
  one manifest deletion after provider cleanup.

- Observation: the canonical observation contract already treats native
  Verifiers output as replay authority and the execution workspace as
  disposable only after durable publication. Purge must therefore erase both
  the queryable projection and its retained native artifact when selected;
  deleting only one would leave misleading partial evidence.
  Evidence: `docs/post-training/06-observation-and-lineage.md`, “Verifiers
  ingest notes” and “Execution workspace”.

- Observation: the public API baseline lists `posttrain run cleanup` but not
  the existing `posttrain run purge`, `posttrain project purge`, or
  `posttrain purge apply` surfaces. Treating them as supported product APIs
  requires a narrow baseline amendment.

- Observation: registry manifest deletion does not guarantee immediate layer
  byte reclamation. Shared blob garbage collection belongs to registry
  infrastructure and must not be reported as bytes reclaimed by a run purge.

## Decision Log

- Decision: reserve `purge` for intentional irreversible erasure, not routine
  lifecycle maintenance or disk pressure.
  Rationale: valid history and lineage have product value; local caches and
  exact terminal workspaces already have safer, narrower collection paths.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: keep `cleanup`, `cache prune`, retention, and `purge` as distinct
  concepts even when they can touch similarly named files.
  Rationale: their authority differs. Cleanup releases exact terminal execution
  resources, cache prune removes local rebuildable material, retention decides
  how valid evidence is kept, and purge erases a logical object across stores.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: the normal path requires no cleanup or retention command.
  Rationale: terminal reconciliation should schedule exact cleanup
  automatically, while project or work-package policy supplies retention
  defaults. The explicit cleanup command remains for retry and diagnosis.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: retention policy may identify purge candidates, but it must not
  silently apply a remote purge in the first release.
  Rationale: age alone cannot prove that evidence, checkpoints, artifacts, or
  images are unneeded. A batch preview removes repetitive discovery work while
  retaining one human authorization boundary for irreversible erasure. An
  explicitly declared future ephemeral policy can be considered only after
  real qualification and a separate baseline decision.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: support `standard` and `pinned` retention classes initially.
  `standard` inherits the project or work-package policy; `pinned` has no
  automatic expiry and must be unpinned or selected explicitly for purge.
  Rationale: these two meanings cover the durable default and intentional
  protection without making every submission answer a lifecycle questionnaire.
  Date/Author: 2026-08-23 / Codex.

- Decision: purge plans operate on an ownership and lineage closure, not a list
  of paths or an age predicate.
  Rationale: a run may produce artifacts consumed by surviving runs and may
  share an actual-job manifest. Closure planning is the only safe place to
  expose cascade consequences and external blockers.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: run purge is blocked by surviving artifact consumers unless the
  user requests a cascade whose complete same-project closure can be proven.
  Project purge is blocked by any consumer outside the selected project.
  Rationale: a dangling lineage edge is worse than retaining the selected run.
  Cross-project erasure requires a separate, explicitly broader plan.
  Date/Author: 2026-08-23 / Codex.

- Decision: delete an actual-job registry manifest only when all known owners
  are selected, registry ownership inventory is complete at preview and apply,
  and the reference is a digest-pinned actual-job image. Never delete shared
  base/kind images or run registry-wide garbage collection.
  Rationale: remote runners need registry publication, local runners may use a
  run-scoped daemon tag, and image layers commonly have owners outside one run.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: require a non-secret purge reason and bind it, the selected IDs,
  actions, dependencies, warnings, and blockers into the immutable plan digest.
  Rationale: the reason is part of the authorization context and must not be
  added after preview. A small controlled reason category plus an optional safe
  note keeps automation possible and audit output understandable.
  Date/Author: 2026-08-23 / Codex.

- Decision: retain a minimal tombstone outside the deleted project/run state.
  It contains logical IDs, purge ID and digest, actor identity when available,
  safe reason, timestamps, requested scope, per-plane outcomes, and blockers.
  It contains no configs, prompts, traces, metrics, artifact payloads,
  checkpoints, signed URLs, credentials, or provider diagnostics.
  Rationale: intentional erasure must remain distinguishable from data loss,
  while the erased information must not survive inside its audit record.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: preserve checkpoint artifacts through the existing Trackio-backed
  artifact lifecycle until a purge explicitly selects their ownership closure.
  Rationale: resume already consumes durable checkpoint artifacts; local build
  contexts and images are neither checkpoint authority nor replay authority.
  Date/Author: 2026-08-23 / Codex and user.

- Decision: make evidence retention an explicit work-package value that resolves
  into `RunSpec`, defaulting to `standard`; support only `pinned` as the first
  opt-out of policy-driven expiry.
  Rationale: this is the real immutable submission/evidence seam. A generic
  worker-manifest dictionary was not on the CLI submission path and would have
  created an inert policy. Explicit run purge may override a root pin with a
  warning; cascade and project purge block pinned descendant runs.
  Date/Author: 2026-08-23 / Codex.

## Outcomes & Retrospective

Milestones 1 through 5 are complete. Routine cleanup is automatic and
evidence-preserving; cache prune remains local and rebuildable; retention keeps
valid history with minimal manual work; purge is a previewed, reasoned,
project-scoped cross-plane erasure with reference protection and a
privacy-bounded tombstone. Disposable Ambient local and dstack qualifications
proved exact cross-plane deletion, retained-evidence gating, and retry-safe
workspace cleanup without purging retained research runs. Milestone 6 now
contains only the final validation and v0.3.21 publication gates.

## Context and Orientation

Posttrain groups evidence as project → work package → run. A run owns one
execution attempt and records immutable consumed and produced artifact edges.
Artifacts may outlive the run workspace and may be consumed by later runs.
Metrics, traces, native replay bundles, reports, model views, and recovery
checkpoints are durable evidence; a provider workspace and local run-scoped
daemon image are execution resources; source objects, assembled contexts, OCI
layouts, and BuildKit layers are local packing/cache resources. These classes
must not share an implicit “old means deletable” policy.

The target state machine is:

    submitted -> running -> terminal -> reconciled -> cleaned -> retained
                                                            |
                                                            v
                                                   purge planned
                                                            |
                                                            v
                                           applying -> purged | partial

`terminal` means the provider has stopped. `reconciled` means provider outcome,
tracking status, and required durable artifacts have crossed the evidence
barrier defined by `packages/execution/src/posttrain/execution/reconciliation.py`.
`cleaned` means exact provider workspace and local run image resources were
released. `retained` is not a new provider state; it describes the evidence
policy after cleanup. `partial` purge means the immutable plan stopped after
some actions and must resume from its journal.

The lifecycle boundaries are:

| Operation | Normal trigger | May delete | Must retain | External calls |
| --- | --- | --- | --- | --- |
| Cleanup | terminal reconciliation | exact provider execution, exact run workspace, run-scoped local daemon tag | tracking evidence, artifacts, lineage, compact reconciliation and cleanup receipts | exact execution provider only |
| Cache prune | budget/free-space pressure or manual apply | local framework-owned rebuildable source, context, layout, dependency and BuildKit cache entries proven unleased | packages, publication receipts, artifacts, checkpoints, active local run images | none; BuildKit adapter only for its owned cache policy |
| Retention | project/work-package policy | nothing by itself in the first release | standard or pinned valid evidence until explicit purge | read-only inventory |
| Purge | explicit run/project erasure | selected provider records, tracking runs, exclusively owned artifacts and actual-job manifests, exact local control state | minimal tombstone and resources with surviving owners | provider, tracking backend, registry, local control store |

The current implementation is split across these locations:

- `packages/execution/src/posttrain/execution/cleanup.py` owns evidence-gated
  cleanup plans and receipts.
- `packages/execution/src/posttrain/execution/purge.py` owns immutable
  cross-plane plans, action dependency ordering, journals, and receipts.
- `packages/execution/src/posttrain/execution/purge_planner.py` discovers run
  consumers and image owners and builds run or project closure.
- `packages/execution/src/posttrain/execution/provider_purge.py`,
  `registry.py`, and `local_purge.py` own exact mutation adapters.
- `packages/tracking/src/posttrain/tracking/lifecycle.py` defines optional
  authenticated tracking deletion plans and receipts.
- `apps/cli/src/posttrain_cli/purge_surface.py` composes control state,
  tracking, registry, and provider adapters into the public preview/apply flow.
- `apps/cli/src/posttrain_cli/commands/run_cmd.py` presents active and purged
  admission history.
- `apps/observatory` presents retained evidence only; it omits purged runs and
  does not read machine-local tombstones.
- `docs/plan/bounded-job-pack-cache-lifecycle.md` separately owns local
  rebuildable packing material and must never call this purge surface.

## Plan of Work

### Milestone 1: amend the canonical lifecycle vocabulary

Amend `docs/post-training/03-work-and-evidence.md` to say that runs are never
overwritten while retained, but may be intentionally purged through an audited
closure. Define cleanup, retention, purge, and tombstone; retain the existing
evidence-reconciliation barrier. Amend `docs/post-training/05-apis.md` with the
preview/apply commands, reason, cascade, and include-purged presentation. Amend
`docs/post-training/06-observation-and-lineage.md` with artifact-consumer
protection, replay-bundle consistency, and the minimal tombstone fields.

Acceptance is a coherent reading order in which a new contributor can answer
what each operation may delete, why an old run remains visible, and why purge
is not a disk cleanup command. Run documentation link and formatting checks and
review the diff before any code change.

### Milestone 2: define provider-neutral retention and tombstone contracts

Add focused contracts under `packages/common` only when they are genuinely
framework-neutral. Keep backend deletion protocols in `packages/tracking` and
execution orchestration in `packages/execution`; `posttrain.common` must not
import Trackio, dstack, Docker, or registry implementations.

Extend the purge plan schema additively to include a required `PurgeReason`
value and optional safe note, request actor, and requested retention override.
Introduce a `PurgeTombstone` whose serialized form contains only the fields in
the Decision Log. Add a one-release reader for v1 plans and receipts if existing
saved plans must remain inspectable; never silently reinterpret or apply a v1
plan under broader semantics. Either apply the exact old plan with its old
adapter contract or require a fresh preview.

Represent initial evidence retention as `standard` or `pinned` in the resolved
run snapshot. Absence means `standard`, so existing projects and runs require
no migration prompt. Keep trace sampling (`full` or `sampled`) separate: it
controls what is recorded, while evidence retention controls how long the
recorded run remains eligible for purge planning.

Acceptance is contract tests proving canonical serialization, digest binding,
secret rejection/redaction, old-plan read compatibility, and no dependency
boundary violations.

### Milestone 3: make exact cleanup automatic and observable

Wire the terminal reconciliation controller to invoke the existing
`cleanup_execution()` only after the same retained-evidence barrier used by the
manual command. Do not weaken successful-output checks to make automation
convenient. Persist scheduling, completion, deferral, and retry state so a CLI
process restart cannot create a second cleanup or lose a needed retry.

For local execution, remove only the Posttrain-owned run-scoped daemon tag and
canonical run workspace. The content-addressed package image can remain in
reusable daemon/BuildKit storage under bounded cache policy. For dstack, release
the exact provider handle and worker workspace after the digest-pinned image
has already been handed off to the registry. Never delete a registry manifest
from cleanup.

The manual `posttrain run cleanup RUN_ID` command remains idempotent and reports
the stored receipt when automation already succeeded. A blocked cleanup prints
the missing evidence or reconciliation disagreement and the next safe command;
it does not suggest purge.

Acceptance is a local and fake-remote terminal run that cleans automatically,
retains queryable evidence and checkpoint artifacts, and returns the same
receipt when cleanup is invoked again.

### Milestone 4: complete reference-aware cross-plane purge planning

Make `purge_planner.py` construct one complete ownership graph from execution
control state and normalized tracking lineage. The graph must include produced
artifact versions, their consumer runs, provider run IDs, native replay
bundles, actual-job manifest owners, exact local paths, and already completed
planes. Unknown inventory is a blocker, not an empty result.

Preview a run purge without cascade as blocked when any produced artifact has a
surviving consumer. With cascade, traverse consumers leaf-first but remain
inside the same project; report any external project consumer as a blocker.
Project purge selects the project's complete known run closure and blocks on
external consumers. A pinned run is a blocker unless explicitly selected and
the preview states that its pin will be overridden.

Tracking preview and apply must agree on exact provider runs and artifact
versions. Native replay bundles and their queryable trace projections are one
logical evidence unit for deletion. Registry preview may select only
digest-pinned actual-job manifests whose complete owner set is inside the
closure. At apply, every adapter revalidates ownership and identity before its
first mutation; a newly discovered owner defers the action and preserves the
remaining journal.

Do not promise physical byte reclamation where a backend retains shared CAS or
registry layers. Report selected logical bytes, backend-reported unique bytes
when trustworthy, and `unknown` otherwise. Registry garbage collection remains
an ai-infra operation outside this plan.

Acceptance is a matrix covering unconsumed and consumed artifacts, cascade,
cross-project consumers, two runs sharing one job image, a new owner appearing
between preview and apply, incomplete inventory, partial backend failure, and
retry of the same digest-bound plan.

### Milestone 5: make tombstones and command language coherent

Persist the tombstone in the machine-scoped purge store before deleting local
run state, update it atomically from the purge journal, and finalize it only
when every selected plane is complete or explicitly already absent. A partial
tombstone names completed, deferred, and failed planes without copying backend
payloads. The purge plan and journal remain outside the selected project tree
so retry survives local deletion.

Update `purge_surface.py` so preview requires `--reason`, shows the selection
closure and shared resources retained, and saves one immutable plan. Keep apply
as `posttrain purge apply PURGE_ID --expect-digest DIGEST --yes`; reject a
different digest, a blocked plan, or missing explicit confirmation before any
adapter mutation. Project-level retention inventory may offer a batch preview,
but it must feed this same planner rather than create a second deletion path.

Update run list/show so default operational views omit purged payload rows,
`--include-purged` labels the retained tombstone as `purged`, and a direct
tombstone view shows safe scope/reason/outcomes. Observatory simply omits
purged runs. Do not preserve a derived
report whose inputs were purged unless it is independently selected and its
remaining lineage is valid.

Acceptance is a disposable purged run that disappears from default operational
and Observatory views, appears as a minimal labeled tombstone in the explicit
CLI audit view, and has no readable metrics/traces/artifacts.

### Milestone 6: qualify the full lifecycle and release it

First run provider-free unit and integration tests. Then create a synthetic
project containing two runs that share one actual-job image, a produced model
or dataset artifact consumed by a third run, a native trace bundle, and a
durable checkpoint. Exercise local and dstack-backed terminal cleanup and prove
all durable evidence remains. Preview blocked, non-cascade, cascade, and project
purges; do not use Ambient research runs as deletion fixtures.

Apply only approved synthetic plans. Verify Trackio/Doris and Observatory no
longer expose purged payloads, the actual-job manifest remains while any owner
survives and is deleted only after its final selected owner, shared registry
layers are not misreported as reclaimed, and the tombstone remains. Capture
plan IDs, digests, provider handles, artifact/version IDs, manifest digests,
logical/unique byte reports, and per-plane receipts in a release-evidence file.

Run the full validation ladder. Review and commit by owning repository. If
generic Trackio lifecycle behavior changes, commit and publish `../trackio`
first, update this repository's immutable pin and `uv.lock`, then publish and
deploy the framework. Deployment qualification must use immutable revisions
and verify live behavior; a passing local suite alone is not a release.

## Concrete Steps

All framework commands run from `/home/hammad/projects/rl` unless stated
otherwise. Begin every milestone by preserving and classifying the dirty tree:

    git status --short
    git -C ../trackio status --short

For Milestone 1, inspect the canonical diff and validate links/whitespace:

    git diff -- docs/post-training/03-work-and-evidence.md \
      docs/post-training/05-apis.md \
      docs/post-training/06-observation-and-lineage.md
    git diff --check

During contract and planner work, run focused suites first:

    uv run pytest packages/execution/tests/test_cleanup.py \
      packages/execution/tests/test_purge.py \
      packages/execution/tests/test_purge_planner.py \
      packages/execution/tests/test_provider_purge.py \
      packages/execution/tests/test_local_purge.py -q
    uv run pytest packages/tracking/tests -q
    uv run pytest apps/cli/tests/test_purge_surface.py -q

Add or extend integration coverage and then run:

    uv run pytest packages/execution/tests/test_purge_integration.py \
      apps/cli/tests/test_purge_surface.py -q
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Expected preview output is structurally equivalent to:

    Purge plan: purge-0123456789abcdef
    Scope: run example-run plus 2 dependent consumers
    Reason: disposable-smoke
    Provider: 3 actions
    Tracking: 3 runs, 4 artifact versions, 12.3 GiB logical
    Registry: 1 actual-job manifest; 2 shared layers not counted as reclaimed
    Local: 6 exact control paths
    Retained: base/kind images; 1 shared artifact used outside selection
    Blockers: 1
    Apply: unavailable until blockers are resolved

No preview command may alter provider, registry, tracking, or local state.

## Validation and Acceptance

The work is accepted only when all of the following observable behaviors hold:

- A successful or failed terminal run automatically cleans its exact execution
  resources only after the applicable evidence barrier. Its metrics, traces,
  artifacts, lineage, and checkpoints remain readable afterward.
- Repeating cleanup returns the stored receipt and performs no broader scan or
  deletion.
- `posttrain cache prune` and its deprecated compatibility alias make no
  provider, registry, or tracking deletion call.
- A purge preview is entirely read-only, requires a safe reason, binds the
  complete closure to a digest, and reports unknown data as unknown or blocked
  rather than zero.
- Run purge without cascade is blocked by a surviving consumer. Cascade is
  leaf-first, same-project, and blocked by external consumers.
- Project purge cannot delete a run or artifact outside the exact project
  closure.
- An actual-job manifest is retained while any unselected owner survives;
  shared base/kind images are never selected.
- A new owner or changed precondition between preview and apply stops or defers
  the exact action before mutation. Retrying resumes the same journal.
- Partial backend failure never produces a false success. Completed actions are
  not repeated, and remaining actions cannot broaden beyond the saved plan.
- The tombstone contains IDs, safe audit context, and plane outcomes only. A
  secret-scan fixture proves prompts, configs, metrics, traces, checkpoint
  metadata, signed URLs, credentials, and diagnostics are absent.
- Default operational views do not present purged runs as live. Explicit CLI
  audit views show `purged`; Observatory is intentionally limited to retained
  evidence.
- A real disposable local and dstack qualification proves remote Trackio and
  registry behavior. No retained Ambient research run is purged for testing.

## Idempotence and Recovery

Cleanup and purge planning are repeatable. Cleanup writes its plan before
mutation and returns its existing receipt after completion. Purge writes an
immutable plan, then a journal entry for each exact action. Apply orders actions
by dependencies, revalidates before mutation, records success or already-absent
outcomes atomically, and stops at the first deferred or failed action. Retry the
same purge ID and digest; never generate a broader plan merely to get past a
partial failure.

If tracking or registry inventory is unavailable, preview blocks and performs
no mutation. If availability is lost during apply, completed actions remain in
the journal and the tombstone reports a partial state. Restore the service and
resume the same plan. If ownership changed, leave the action deferred and
require a fresh preview only after the operator abandons the old plan; never
rewrite an accepted plan in place.

Existing v1 plan/receipt files remain readable for audit. Do not auto-apply an
old plan under new reason or tombstone semantics. Exact provider cleanup and
registry deletion must remain idempotent when a target is already absent.

No rollback can restore erased remote evidence. Qualification therefore uses a
synthetic project and confirms required release evidence has been exported
before apply. The tombstone is not a backup and must never contain recoverable
copies of purged payloads.

## Artifacts and Notes

The implementation must retain compact evidence sufficient to prove behavior:

- baseline amendment diff;
- focused and full validation transcripts;
- synthetic project/run IDs and lineage graph;
- immutable purge plan IDs and digests;
- per-plane preview counts, blockers, warnings, and apply receipts;
- before/after Trackio, Observatory, provider, registry-manifest, and local
  control-state probes;
- logical bytes versus backend-confirmed unique bytes, with unknown values
  stated explicitly;
- secret-scan result over the final tombstone.

Never store credentials, tokens, signed URLs, raw prompts, or user payloads in
the plan, release evidence, tombstone, or test transcript.

## Interfaces and Dependencies

Keep these ownership boundaries at the end of the work:

- `posttrain.common`: provider-neutral identities and retention values only;
  no Trackio, Docker, dstack, or registry imports.
- `posttrain.tracking`: normalized evidence and optional authenticated
  lifecycle administration. Extend `TrackingLifecycleAdmin` only with exact,
  previewable, digest-bound operations.
- `posttrain.execution`: cleanup orchestration, ownership closure, immutable
  purge plans, journals, tombstones, and action execution.
- `posttrain_cli`: composition, authorization input, human/JSON presentation,
  and confirmation. It must not duplicate ownership rules.
- `apps/observatory`: read-only retained-evidence and tombstone presentation;
  it never applies purge.
- ai-infra: registry service garbage collection and host-wide BuildKit policy;
  neither is invoked by a run or project purge.

Preserve and refine these stable interfaces rather than creating a second
deletion stack:

    cleanup_execution(
        service: JobExecutionService,
        store: ExecutionSubmissionStore,
        source: RunDataSource | None,
        run_id: str,
        *,
        diagnostic_limit: int = 500,
    ) -> ExecutionCleanupReceipt

    build_run_purge_plan(
        catalog: PurgeRunCatalog,
        *,
        root_run_id: str,
        cascade: bool = False,
        reason: PurgeReason,
    ) -> PurgePlan

    build_project_purge_plan(
        catalog: PurgeRunCatalog,
        *,
        project_id: str,
        reason: PurgeReason,
    ) -> PurgePlan

    class TrackingLifecycleAdmin(Protocol):
        def plan_run_purge(...) -> TrackingPurgePlan: ...
        def apply_run_purge(...) -> TrackingPurgeReceipt: ...
        def project_delete_plan(...) -> TrackingProjectDeletePlan: ...
        def delete_project(...) -> TrackingProjectDeleteReceipt: ...

The exact `PurgeReason`, `PurgeTombstone`, and retention value location should
be resolved in Milestone 2 by dependency ownership, but their serialized schema
and behavior must follow this plan. Use no new external library unless the
standard library and existing typed contracts cannot provide canonical JSON,
digests, atomic writes, and timezone-aware timestamps.

Revision note (2026-08-23): created this companion lifecycle plan after the
Ambient cache investigation exposed that local rebuildable storage, terminal
execution cleanup, durable evidence retention, and cross-plane erasure had
been discussed as one cleanup problem. The revision assigns each operation one
purpose, keeps the normal developer path automatic, reserves purge for explicit
erasure, and makes shared-reference protection and minimal tombstones release
gates before any further destructive behavior changes.

Revision note (2026-08-23): completed the narrow frozen-baseline amendment.
The canonical documents now distinguish automatic terminal cleanup, local cache
pruning, evidence retention, and explicit cross-plane purge; they define the
consumer/owner blockers and privacy-bounded tombstone before implementation
changes proceed.

Revision note (2026-08-23): began Milestone 2 with the authorization-context
contract. New v2 purge plans require a canonical, non-secret reason and bind it
into their content digest; CLI preview requires that reason; v1 plans remain
readable as legacy audit records without gaining new semantics.

Revision note (2026-08-23): added the tombstone contract and machine-store
persistence. A v2 plan writes an `applying` tombstone before its first action,
then derives only per-plane outcomes from the append-only journal; completed,
deferred, and failed state never copies diagnostics or deleted evidence into the
tombstone. CLI apply refuses legacy v1 plans so an old preview cannot bypass the
new reason/tombstone authorization boundary.

Revision note (2026-08-23): made tombstone state inspectable through
`posttrain purge show`. The command shows safe lifecycle status and per-plane
outcomes beside the immutable plan without contacting a provider or exposing
deleted evidence. Retention remains deliberately unwired until the resolved-run
snapshot is identified; the current worker manifest retention dictionary is
not yet constructed by the CLI and would be a dead policy path.

Revision note (2026-08-23): completed the retention contract at the actual
work-package → RunSpec → admission/submission → tracking path. `pinned` is now
visible to purge planning and protected from broad project/cascade selection.
Also began automatic cleanup: controller reconciliation and `run reconcile`
invoke the existing evidence-gated cleanup before releasing an admission slot;
cleanup failure leaves the run retryable rather than releasing it as complete.

Revision note (2026-08-23): removed the stale unconditional project-purge
blocker. Project purge now saves the inventory-derived cross-plane plan and
uses the existing digest-bound Trackio project-delete executor instead of a
second, disabled path. Live qualification remains required before release.

Revision note (2026-08-23): live-qualified one disposable Ambient local
qualification run (`7eeed416-69ce-4c9e-a500-0883d8ced35a`) using plan
`purge-10a4d29453aa6177`. The guarded apply completed provider, registry,
Trackio, and local actions, then persisted a reason-bound `purged` tombstone.
The qualification also exposed two production-path defects before deletion:
Trackio lifecycle administration omitted the machine trust bundle, and a
single-run preview refreshed every historical provider/Trackio entry. The
adapter now uses machine-owned CA trust and a non-cascade run preview refreshes
and discovers only its selected root; project and cascade previews still use
the complete closure inventory.

Revision note (2026-08-23): live-qualified remote automatic cleanup with
Ambient dstack run `e898b0b8-7f6a-4b7b-b83b-83cb63f67e15`. Reconciliation
proved six retained Trackio artifacts before the cleanup receipt recorded
`provider-managed` and `workspace=removed`; rerunning after the initial
persisted plan completed safely. The CLI audit list now exposes only the
minimal completed tombstone state. The product decision is that Observatory
omits purged runs; it does not consume a machine-local store or require a
Trackio tombstone read API.
