# Bounded job packing with durable receipts and disposable build material

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

This plan is self-contained. It governs project-local package materialization,
actual-job image publication, local-image export, and the BuildKit cache
boundary. It complements `docs/plan/dx-packing-environments-datasets.md`: that
plan defines what belongs in a job package; this plan defines which local bytes
remain after the package has been published and how developers inspect and
reclaim them.

## Purpose / Big Picture

Packing one Posttrain job currently leaves several complete representations of
the same inputs on the submitting workstation: a content-addressed project
source snapshot, a retained assembled build context, sometimes a complete OCI
layout, and BuildKit's own layer cache. In the Ambient Agent project these
trees reached about 199 GiB after an emergency smoke-run cleanup: 20 GiB of
source snapshots, 97 GiB of retained contexts, and 82 GiB of publication data.
The named BuildKit builder had separately occupied about 68 GiB before it was
removed to recover the full filesystem. This is not primarily a Docker-image
problem. It is an ownership and lifetime problem.

The immediate trigger also exposes a packaging flaw. Ambient declares
`source_includes = ["pyproject.toml", "src", "packages", "data"]`. The current
CLI constructs `ImmutableDatasetPackager` with the already-copied project
source snapshot as its source root, so a selected dataset cannot be packaged
unless its project data is first admitted as general application source. A
Reasoning Gym image therefore attempted to copy an unrelated 1.9 GiB Hotpot
graph store and failed before reaching a provider. Cache collection would make
space temporarily but would not correct this repeated overpacking.

After this plan, a developer can run a job without understanding cache
internals. `posttrain job pack` selects only the code, environment resources,
and one or more datasets reachable from that selected job's resolved dataset
seats; reports the planned component sizes before the
expensive build; publishes or reuses one digest-addressed registry image; and
removes temporary assembly bytes when their lease ends. Compact package and
publication records remain available for package diff, replay, submission, and
diagnosis. Checkpoints, LoRA adapters, model weights, reports, and traces remain
in their artifact backend and are never cache-prune candidates.

When developers do need control, one discoverable command family explains it:

    posttrain cache status
    posttrain cache explain <object-or-package-key>
    posttrain cache prune
    posttrain cache prune --apply

The first three operations are read-only by default. `prune` prints exactly
what it would remove, why it is safe, how many bytes would be recovered, and
which active leases or durable records protect the rest. The ordinary packing
path automatically collects only expired, unleased, rebuildable objects when
the configured budget or free-space guard is crossed. It never invokes a
Docker-wide prune.

This work does **not** amend the frozen product baseline. The canonical product
already defines the actual-job OCI image as the immutable distribution unit,
the execution workspace as disposable after durable outputs are published,
and `.posttrain/state/cache` as rebuildable local state. The implementation is
being brought into conformance with those meanings. If implementation reveals
that a package record must become a portable project artifact rather than
machine-local control evidence, pause and amend the canonical baseline before
making that change.

## Progress

- [x] (2026-08-10) Read the canonical work/evidence, framework, API, and
  observation/lineage contracts and confirmed that no baseline amendment is
  required.
- [x] (2026-08-10) Measured the post-cleanup Ambient pack state: approximately
  20 GiB sources, 97 GiB contexts, 82 GiB publications, and 436 MiB dependency
  cache; found abandoned `.job-context-stage-*` and `.job-context-work-*`
  directories.
- [x] (2026-08-10) Traced the source of overpacking from Ambient's
  `[tool.posttrain.pack]` through `execution_planning.py` to
  `ImmutableDatasetPackager(project_root=project_source.package.root)`.
- [x] (2026-08-10) Confirmed that `plan_job_pack()` constructs dataset requests
  only from `DatasetLoadPlan` values in the selected job's prepared seats, but
  that this correct semantic selection is currently defeated by copying the
  project-wide `data/` directory through the generic source snapshot.
- [x] (2026-08-10) Confirmed that ai-infra already owns a bounded,
  BuildKit-native policy for the named `posttrain-builder`; the builder itself
  is currently absent after emergency cleanup and must be recreated and
  qualified before a live canary.
- [x] (2026-08-10) Removed 10 registry-backed local OCI layouts and two
  abandoned staging directories from Ambient after verifying there were no
  active pack/build/training processes. This recovered approximately 42 GiB;
  unpublished layouts, source snapshots, contexts, and durable receipts remain
  protected.
- [x] (2026-08-10) Implemented the first selected-input slice: dataset code
  snapshots and project data inputs now use separate roots, a package plan
  rejects generic source includes that cover catalog-declared local datasets,
  and tests prove zero/one/multi-seat closure and symlink rejection.
- [x] (2026-08-10) Made `posttrain state cache-prune` granular for the migration
  window: only verified registry-backed local layouts and hidden pack staging
  directories are reclaimable; receipts, contexts, source snapshots, and
  unpublished local layouts are reported as protected.
- [x] Milestone 1 local contract: introduce compact package/materialization
  records and move publication receipts out of the rebuildable cache namespace.
  Legacy-state migration and remote-receipt verification remain Milestone 6
  gates.
- [x] (2026-08-10) Added `PackageMaterializationRecord` under the protected
  `packages/materializations/<package-key>.json` application path. It embeds the existing manifest, context
  digest, publication key, and manifest digest; writes are atomic and
  conflicting replacement fails closed. Package history reads records first
  and falls back to legacy contexts during migration.
- [x] (2026-08-10) Separated durable image receipts from rebuildable local
  layouts. Project publishers now default to `.posttrain/state/publications`
  for receipts and `state/cache/pack/local-layouts` for local OCI output;
  custom receipt roots remain supported.
- [x] (2026-08-10) Added expiring `CacheLease` records and wired them through
  context materialization and local OCI builds. Cache pruning protects objects
  with an active lease and tests cover release and expiry behavior.
- [x] (2026-08-10) Added plan-key lookup and receipt-only publication reuse:
  an unchanged published package can be resolved from its compact record and
  verified registry receipt without reconstructing source, dependencies,
  datasets, or an assembled context. A missing or stale remote image falls
  back to the normal materialization path.
- [x] (2026-08-10) Moved new materialization records to protected
  `.posttrain/state/packages/materializations` while retaining a one-release
  reader for legacy `cache/pack/records`; the receipt-only path is covered by
  a test that fails if context materialization is invoked.
- [x] (2026-08-10) Added the developer-facing `posttrain cache status`,
  `posttrain cache explain <selector>`, and `posttrain cache prune` commands;
  the existing `posttrain state cache-prune` remains a compatibility alias.
  Status and explain are read-only, while prune remains dry-run by default.
- [x] (2026-08-10) Completed the local Milestone 2 implementation slice:
  selected dataset inputs are resolved from the project root, generic source
  overlap is rejected before copying, and planning records project/framework
  file and byte estimates plus per-seat selected dataset source estimates (or
  an explicit generated/remote marker).
- [x] (2026-08-10) Completed the local Milestone 3 safety slice: materialized
  contexts and local OCI builds acquire expiring leases, records carry the
  plan key, and published pack paths remove only exact framework-owned retained
  contexts after the publisher returns. Full crash-recovery migration and
  automatic budget collection remain qualification work.
- [x] (2026-08-10) Completed the local Milestone 4 command slice: cache status,
  explain, and dry-run/apply prune share one classification engine and expose
  structured output. Full policy budgets, LRU ordering, and BuildKit telemetry
  remain migration/host qualification work.
- [x] (2026-08-10) Added the versioned machine `[cache]` policy value with
  conservative budget, free-space, age, and failed-debug-retention defaults;
  `posttrain cache status` reports the effective policy without including
  credentials. Enforcement remains gated on the migration inventory.
- [x] (2026-08-10) Marked `posttrain state cache-prune` as a one-release
  deprecated forwarding surface; it uses the same granular planner and no
  longer represents a separate whole-tree cleanup policy.
- [x] (2026-08-10) Completed the local explicit-export slice of Milestone 5:
  `job pack --local --local-output PATH` writes an atomic OCI layout and
  adjacent user-owned receipt outside `.posttrain/state/cache`, and validates
  that receipt before a local export cache hit; internal exports retain lease
  protection. Direct daemon load and terminal tag cleanup remain host/provider
  qualification work.
- [x] (2026-08-10) Ran the new read-only status command against Ambient: it
  reports about 168.8 GB of scoped state, about 505.6 MB immediately
  reclaimable, and about 168.3 GB protected. The large contexts and unpublished
  layouts remain protected because legacy records and ownership have not yet
  been migrated; no apply prune was attempted.
- [x] (2026-08-10) Implemented the provider-neutral direct-daemon contract:
  local execution carries an immutable image digest plus a Posttrain-owned
  daemon tag, while explicit `job pack --local` continues to produce only a
  user-selected OCI export. BuildKit loads single-platform images with
  `type=docker`; receipt reuse first verifies the exact daemon tag, local
  submissions skip registry pull, and terminal local cleanup removes only the
  exact run-scoped tag. Durable submission and
  admission records retain the local tag for restart and reconciliation.
  Focused BuildKit, local-provider, execution, admission, and CLI tests pass.
- [x] (2026-08-10) Recreated and qualified the repository-owned
  `posttrain-builder` through the ai-infra procedure. The named
  `docker-container` builder is running with its retained state volume,
  registry CA, and bounded 60/80 GB/20% GC policy; qualification reported
  `prune_invoked: false` and zero existing records.
- [x] (2026-08-10) Re-ran the Ambient inventory without mutation: 53 legacy
  contexts all contain `package.json`, 80 legacy publication receipts exist,
  and no compact materialization records have been migrated yet. The planner
  reports 168,282,955,310 protected bytes and 505,624,505 rebuildable bytes;
  no prune or migration apply was attempted.
- [x] (2026-08-10) Qualified direct local-daemon publication against the real
  named builder. The first canary exposed two owning-layer contract defects:
  inherited provenance/SBOM attestations made the Docker exporter produce an
  unloadable manifest list, and exporter `name` did not override Bake's target
  tags. The adapter now disables attestations only for ephemeral daemon loads
  and overrides the target tag explicitly; registry publications retain both
  attestations. Focused BuildKit checks pass.
- [x] (2026-08-10) Qualified the worker/provider boundary with Ambient canary
  `ambient-reasoning-gym-local-daemon-canary-20260810-r2`. A first submitted
  attempt proved that provider-only `local_image` transport metadata must not
  enter the versioned worker launch envelope. The corrected r2 image was
  loaded under its run-scoped tag, submitted as local provider
  `pt-44af5a1361b1f73fbbff4252`, accepted by the packaged runtime, initialized
  Trackio run `e352acde44ba4444872fdaebd2906a45`, and is running the two-step
  qualification. Terminal artifacts and tag cleanup remain to be verified.
- [x] (2026-08-10) Ran the first real read-only legacy receipt audit. Of 59
  registry publication receipts, 41 still resolve to the exact recorded
  registry digest and 18 are stale/missing. Forty legacy contexts have at
  least one registry receipt, 28 have at least one remotely verified receipt,
  51 of 53 contexts have some local or remote receipt, and two contexts have
  no receipt. No legacy context, receipt, layout, or artifact was mutated.
- [x] (2026-08-10) Qualified registry publication and fresh-process reuse for
  the current Ambient Reasoning Gym package. Package
  `294be51a3396ba17263f17edca18387cb257cc42c302976d15f69eca4b1d2c6f`
  published as attested digest
  `sha256:d8f45e23d546926f36ae049fba935ec92eb30168596ba3f6a39ad402096c6e05`
  under publication key
  `9edf38728c09c835b8a613c270585eb17caf8f49b9d4ec2b88d3addfbd9fd9c2`.
  The assembled context was absent immediately after publication. A new CLI
  process resolved the remotely verified receipt in 1.537 seconds with
  `cache_hit: true`, created no Buildx history entry, and left the context
  absent.
- [x] (2026-08-10) Captured the post-canary Ambient cache dry-run without
  mutation: 168,805,105,779 bytes observed, 505,624,505 bytes classified as
  rebuildable across five entries, 168,299,481,274 bytes protected across 154
  entries, and zero bytes removed. This remains the before-migration receipt;
  no `--apply` operation is authorized by this dry-run alone.
- [x] (2026-08-10) Diagnosed the terminal failure of local-daemon canary r2 at
  the rollout/actor boundary rather than attributing it to packaging. All 128
  sampled SFT-policy completions ended at the 1,536-token limit, so the OLMo3
  truncation mask correctly left no trainable tokens. The Verifiers bridge now
  derives truncation from the native trace record, records selected/completion
  token counts, and the TRL adapter fails early with an actionable all-masked
  message. Focused train tests, Ruff, and Pyright pass.
- [x] (2026-08-10) Verified the external environment release boundary. The
  current `verifiers-environments` revision
  `3e1582ef3cce8e6d355be3747be0427f700ef865` is present on `origin/main`, its
  Reasoning Gym package tests pass, and RL plus Ambient pin that exact commit.
  That revision includes the prior Math-Python bounded-reasoning fix; no new
  Reasoning Gym prompt contract will be published until the base-policy
  control proves the adapter itself needs one.
- [ ] (2026-08-10) Complete base-policy local canary
  `ambient-reasoning-gym-base-local-canary-20260810-r1`, then decide from its
  native traces whether Reasoning Gym needs a generic bounded-answer prompt.
  If it does, change and qualify `verifiers-environments` first, push an
  immutable commit, and only then advance RL and Ambient pins for this release.
- [x] (2026-08-10) Removed the rollout-batch observation barrier exposed by
  the base-policy canary. The optional observed-bridge extension projects and
  preserves each completed Verifiers trajectory immediately, then submits it
  through one serialized off-event-loop observer lane while Trackio performs
  remote delivery in its background sender. Trainer results remain aligned to
  input order, and older bridge implementations retain batch-complete fallback
  behavior. The train package reports 164 tests passed and 5 skipped; Ruff,
  Pyright, import contracts, and diff validation pass.
- [x] (2026-08-10) Closed the live-trace spool lifecycle. Every completed
  native trace is appended to the run-scoped JSONL before live submission;
  traces accepted by the live observer are excluded from final evidence replay
  while trace-derived metrics still use the complete spool. Batch-only and
  isolated bridges retain replay behavior. The JSONL remains the immutable
  trace-artifact source until terminal evidence reconciliation, after which
  exact-run provider cleanup removes the workspace/container copy. The failed
  base-policy canary was reconciled with one retained Trackio artifact, then
  its exact local container and workspace were removed with a durable cleanup
  receipt.
- [x] (2026-08-10) Published the now-proven Reasoning Gym termination fix as
  `carbonteq-ai/verifiers-environments@ee096746ec3cf28eceffd49f29226e8a8dc7bc31`.
  The environment adds a format-preserving bounded-reasoning system prompt and
  includes it in task identity. Its locked package ladder, repository boundary
  checks, and combined six-wheel installation pass. The RL release candidate,
  Git constraints, catalog, tests, tooling page, and lock now pin the published
  revision; 207 focused framework tests pass with 7 skipped. Ambient remains on
  the prior revision until the new Posttrain release removes its 0.3.5 metadata
  constraint, after which both pins will advance together.
- [ ] Milestone 5: make local execution load directly into the local daemon and
  make explicit local OCI export user-owned rather than silently cached.
- [ ] Milestone 6: migrate the existing Ambient state, correct Ambient source
  selection, recreate the bounded BuildKit builder, and qualify local and
  registry-backed flows.
- [ ] Milestone 7: release, update the Ambient framework pin, and prove that a
  repeated real pack is both bounded and reusable.

## Surprises & Discoveries

- Observation: the largest remaining tree is not BuildKit; it is retained
  framework build contexts under the project.
  Evidence: after smoke cleanup, `.posttrain/state/cache/pack/contexts` is about
  97 GiB while `.posttrain/state/cache/pack/sources` is about 20 GiB. Every
  retained context copies project source and selected package inputs again.

- Observation: local OCI exports are stored below the same directory as small
  remote publication receipts, even though the two have different meanings
  and lifetimes.
  Evidence: `BuildKitJobImagePublisher.publish_local()` writes full layouts to
  `<receipt_root>/local-layouts/<publication_key>`, while `publish()` writes a
  small `<publication_key>.json` receipt. The default `receipt_root` is
  `.posttrain/state/cache/pack/publications`, so the current type boundary is
  not reflected in storage.

- Observation: package comparison depends on retained heavyweight contexts.
  Evidence: `apps/cli/src/posttrain_cli/package_history.py` discovers history
  by scanning `pack/contexts/*/package.json`. Deleting contexts today therefore
  also deletes the developer's ability to run `posttrain job diff`, even though
  the manifest needed for comparison is only a compact JSON value.

- Observation: a remote publication cache hit is checked too late to avoid
  local materialization.
  Evidence: `PlannedJobPackage.pack()` calls `materialize()` before
  `BuildKitJobImagePublisher.publish()` checks its receipt and verifies the
  registry digest. A fully published image can still trigger source snapshot,
  dependency, dataset, and context work on every invocation.

- Observation: the present prune command is safe but too coarse to be useful as
  an automatic policy.
  Evidence: `prune_cache()` classifies the whole known `pack` tree as one
  rebuildable entry. It has no per-object size, age, last-use, lease, package
  reference, publication state, or LRU ordering and can only retain everything
  or delete everything.

- Observation: plain LRU is insufficient for correctness.
  Evidence: two CLI processes may pack the same package concurrently, and a
  provider submission can overlap with cleanup. Age does not prove that a path
  is unused. Cache collection needs explicit leases plus atomic publication;
  LRU should only order already-unreferenced candidates.

- Observation: dataset request selection and package byte selection are two
  separate invariants, and only the first one is currently satisfied.
  Evidence: `plan_job_pack()` filters the selected job's `prepared.seats` for
  `DatasetLoadPlan`, so it does not deliberately enumerate every catalog
  dataset. However, Ambient's broad source include copies every local dataset
  before `ImmutableDatasetPackager.package()` stages the selected requests.
  Manifest correctness therefore does not prove byte-level package relevance.

- Observation: the existing infrastructure policy is already architecturally
  preferable to framework-initiated Buildx pruning.
  Evidence: `/home/hammad/projects/ai-infra/config/buildkit/buildkitd.toml`
  configures native GC for only `posttrain-builder`, with a 60 GB reserved
  working set, an 80 GB collection trigger, a 20% minimum-free guard, and
  cheaper local sources/cache mounts collected before shared image layers.

- Observation: registry-backed local layouts can be removed immediately
  without losing publication authority.
  Evidence: the 10 removed layouts each had a sibling
  `posttrain.job-image-publication-receipt.v1` containing a registry image
  digest; the 11 layouts without such a receipt were retained.

- Observation: the first implementation can enforce dataset relevance before
  any expensive materialization.
  Evidence: Ambient's Reasoning Gym plan now resolves with source includes
  `packages`, `pyproject.toml`, and `src`, no dataset seat, and no generic
  `data/` capture; the focused suite reports 39 passed.

- Observation: package history does not need an assembled context to compare
  job meaning.
  Evidence: `apps/cli/tests/test_package_history.py` resolves a package from
  only its compact record after the context directory is absent, while the
  existing `compare_job_packages` contract consumes the embedded manifest
  payload unchanged.

- Observation: publication receipts and local OCI layouts have independent
  retention requirements.
  Evidence: a custom-root BuildKit test places the local layout below a cache
  directory while its `.local.json` receipt remains under the durable receipt
  directory; no `receipt_root/local-layouts` directory is created.

- Observation: cache inspection can reuse the same classification engine as
  pruning without introducing a second ownership model.
  Evidence: the new status and explain commands serialize `CachePruneReport`
  entries, so protected reasons, active leases, and reclaimable byte counts are
  identical in read-only and apply modes.

- Observation: a compact record can safely bypass source and dataset
  materialization only when its publication receipt is verified again.
  Evidence: `PackageMaterializationStore.resolve(plan_key)` returns the
  manifest and publication identity, while the BuildKit adapter's receipt-only
  resolver rechecks the definition digest and registry manifest digest. A
  record by itself is never accepted as a runnable image.

- Observation: retained contexts are no longer needed after a successful
  publication, including when the caller asks for a local OCI export.
  Evidence: `PlannedJobPackage.pack()` and `pack_local()` now release the
  lease and remove only a 64-hex context below the exact project cache context
  root. Fake or explicitly user-owned paths are left untouched.

- Observation: the existing Ambient state is still too ambiguous for a blanket
  migration or prune.
  Evidence: `posttrain cache status` currently sees roughly 168.8 GB of scoped
  state but can prove only 505.6 MB as immediately reclaimable; contexts and
  local layouts without matching durable publication evidence remain protected.

- Observation: the local host cannot complete live BuildKit qualification yet.
  Evidence: Docker is available and the default builder is running, but
  `docker buildx inspect posttrain-builder` reports that the named builder does
  not exist. The ai-infra qualification scripts are present, so this is an
  infrastructure-owner gate rather than a framework test failure.

- Observation: a daemon-loaded image needs a different lifetime from a user
  OCI export.
  Evidence: local execution now carries a digest-backed image plus a
  run-scoped `posttrain-local:` tag; the tag is persisted with the submission
  and removed only during terminal local cleanup. Remote providers never
  receive or use that tag.

- Observation: a single-platform build can still become a manifest list when
  the target inherits provenance and SBOM attestations.
  Evidence: the real Docker exporter rejected the first Ambient canary with
  `docker exporter does not currently support exporting manifest lists` until
  the daemon-only path disabled both attestations. Registry publication keeps
  them enabled.

- Observation: the Docker exporter `name` option is not the authoritative tag
  when a Bake target already supplies `tags`.
  Evidence: the first successful daemon export loaded
  `registry.lan/carbonteq/posttrain-job:<publication-key>` instead of the
  requested run-scoped `posttrain-local:` tag. Overriding
  `posttrain-job.tags` produced the tag consumed by the local provider.

- Observation: legacy publication receipts are verification evidence, not
  proof that registry bytes still exist.
  Evidence: exact `imagetools inspect` checks found 41 of 59 recorded remote
  digests present and 18 absent. Migration must retain stale receipts as audit
  history while refusing to use them to justify context or layout deletion.

- Observation: the failed SFT-policy Reasoning Gym canary carried valid token
  data; it did not fail because the actor update lost completion tokens.
  Evidence: sampled native traces each contained 1,536 selected completion
  tokens and ended with `finish_reason=length`. With truncated completions
  masked, zero eligible tokens is the correct outcome. The misleading
  `is_truncated=false` projection came from bridge metadata overriding the
  native stop/finish evidence.

- Observation: the external environment revision is already a published input
  to the candidate release, but it is not evidence of a new Reasoning Gym
  behavior change.
  Evidence: `origin/main`, RL constraints/lock/catalogs, and Ambient
  dependencies/catalogs all resolve
  `3e1582ef3cce8e6d355be3747be0427f700ef865`; that commit changes only the
  Math-Python bounded-reasoning prompt. Reasoning Gym remains unchanged at that
  revision and its package contract suite passes.

- Observation: live rollout visibility was delayed in the producer before
  Trackio received anything.
  Evidence: `VerifiersEnvironmentRolloutBridge.run()` awaited one
  `asyncio.gather()` for the complete population and only then projected the
  returned traces; the TRL adapter emitted them in a second post-batch loop.
  Trackio's `Run.log()` already appends locally under its client lock and wakes
  a background remote sender, so adding another network worker in the trainer
  would duplicate ownership.

- Observation: completion-time submission and terminal evidence replay can
  otherwise send the same native trace twice.
  Evidence: the preserved JSONL intentionally contains every rollout, and
  `_publish_bridge_artifacts()` replays every trace returned by `evidence()`.
  The bridge now remembers only trace ids successfully accepted by its live
  observer and filters those ids from trace replay without filtering aggregate
  metrics. A reconstructed isolated bridge has no such in-memory acknowledgments
  and therefore safely replays its preserved traces.

- Observation: the local native-trace file is recovery state and artifact
  source material, not a second long-term evidence store.
  Evidence: base-policy canary `ambient-reasoning-gym-base-local-canary-20260810-r1`
  reconciled to retained Trackio artifact version `v0`; only after that barrier
  did `posttrain run cleanup` remove local provider
  `pt-7838f6ee5d3fd3369fb69aa2` and its exact run workspace. The cleanup receipt
  records `evidence_state=reconciled` and `retained_artifact_count=1`.

- Observation: the base-policy control also reached the actor parity gate with
  no selected training tokens, but it produced a materially varied reward
  population before failing: three native rewards of 1.0, one of 0.01, and
  four of 0.0 across eight traces.
  Evidence: local provider `pt-7838f6ee5d3fd3369fb69aa2` exited 1 and Trackio
  run `15418b50cd3a4da184a84497f849bb1d` retained all eight traces. This means
  the next canary must separate native completion/selection projection from a
  Reasoning Gym prompt-policy decision; reward spread alone does not validate
  actor eligibility.

## Decision Log

- Decision: do not disguise framework truncation fixes as a Reasoning Gym
  package release; advance the external environment pin only for a verified,
  generally reusable environment-contract change.
  Rationale: native trace projection and zero-token actor diagnostics are owned
  by Posttrain, while prompt/termination policy belongs to
  `verifiers-environments`. Publishing an unproven prompt change would alter
  the task distribution and confound the base-versus-SFT control.
  Date/Author: 2026-08-10 / plan author.

- Decision: expose completion-time rollout observation as an optional bridge
  extension and retain the existing batch-return contract.
  Rationale: this gives native Verifiers bridges live evidence without breaking
  third-party bridges or changing the trainer's deterministic input ordering.
  A single off-event-loop submission lane protects Trackio's run-local ordering
  and step state; Trackio remains responsible for retry, buffering, and remote
  delivery.
  Date/Author: 2026-08-10 / plan author.

- Decision: preserve locally first, acknowledge live submission second, and
  reclaim only after terminal evidence reconciliation.
  Rationale: immediate deletion would race Trackio's background artifact
  upload, while unconditional final replay duplicates network and importer
  work. The run-scoped JSONL provides crash recovery and one immutable artifact
  source during execution; the existing exact-run cleanup receipt provides the
  safe deletion boundary after publication.
  Date/Author: 2026-08-10 / plan author.

- Decision: distinguish authority, evidence, cache, and temporary workspace by
  contract instead of treating every local path as cache.
  Rationale: registry image bytes, package meaning, publication verification,
  reusable intermediate objects, and in-flight assembly have different owners
  and recovery consequences. A single recursive prune cannot express those
  differences safely.
  Date/Author: 2026-08-10 / plan author.

- Decision: keep compact package and publication records, but do not keep a
  complete assembled context merely to support history or replay.
  Rationale: `package.json`, its plan key, component digests, the publication
  key, and the verified image digest are sufficient for diff, explanation,
  submission, and registry reuse. Rebuilding from immutable inputs remains the
  fallback when the registry publication is absent.
  Date/Author: 2026-08-10 / plan author.

- Decision: make the transitional prune policy conservative by classifying
  only registry-backed local layouts and abandoned hidden staging directories
  as reclaimable.
  Rationale: compact replacement records and the lease protocol are not yet
  implemented, so deleting ordinary contexts or source snapshots would remove
  useful replay evidence. A verified remote image is the durable replacement
  for its local layout; an unpublished layout has no such recovery authority.
  Date/Author: 2026-08-10 / implementation.

- Decision: expose cache operations as a top-level command family while keeping
  `state cache-prune` as a compatibility alias for one release.
  Rationale: cache ownership is a normal developer workflow, not an opaque
  state migration detail; the alias avoids breaking existing cleanup scripts.
  Date/Author: 2026-08-10 / implementation.

- Decision: store the full existing `JobPackageManifest` payload inside the
  compact record rather than inventing a parallel summary schema.
  Rationale: the manifest already owns package identity and the field-level
  comparison contract. Retaining it avoids divergent package meaning while
  reducing history storage from a complete source/context tree to one small
  JSON value.
  Date/Author: 2026-08-10 / implementation.

- Decision: use expiring file leases rather than process scans for cache safety.
  Rationale: a cache pruner cannot reliably infer whether another process is
  using a path from its command line. A lease has an explicit object key and
  expiry, survives a process crash long enough for recovery, and can be
  inspected without depending on a provider daemon.
  Date/Author: 2026-08-10 / implementation.

- Decision: use a run-scoped daemon tag for local execution instead of sharing
  one publication-wide tag.
  Rationale: tag sharing would require a cross-process reference counter before
  cleanup could safely remove an image. A run-scoped tag makes ownership
  explicit, keeps retries idempotent through the durable submission record, and
  prevents one terminal run from deleting another run's local transport image.
  Date/Author: 2026-08-10 / implementation.

- Decision: a verified remote image does not need a local context through run
  reconciliation.
  Rationale: once the registry digest and publication receipt agree, the
  provider consumes the image by digest. Submission and reconciliation need
  the image reference and execution records, not the Docker build context.
  Retaining contexts until a run terminates would couple package storage to a
  potentially long provider lifetime without adding recovery value.
  Date/Author: 2026-08-10 / plan author.

- Decision: selected project datasets are resolved directly from the project
  root under a bounded dataset-source contract, then copied into the package by
  the dataset packager; they are not admitted through `source_includes`.
  Rationale: application source and data selections are different seats with
  different locks. This makes package contents job-specific and lets unrelated
  data change without invalidating the project-source digest.
  Date/Author: 2026-08-10 / plan author.

- Decision: dataset closure starts from the selected job's resolved dataset
  seats and includes exactly those selections plus their explicitly declared
  assets; it never starts from every dataset in a work package, catalog,
  project directory, or materialization cache.
  Rationale: a job may legitimately bind several dataset seats, so “relevant”
  is a selection closure rather than a single file. Making the closure explicit
  prevents both accidental omission of a second selected seat and accidental
  inclusion of unrelated project data.
  Date/Author: 2026-08-10 / plan author.

- Decision: automatic project-cache collection runs at pack boundaries and on
  budget pressure; no new project cache daemon is required.
  Rationale: packing is the operation that creates and touches these objects,
  so it can maintain leases and make deterministic collection decisions. A
  background daemon would add races and deployment work without an independent
  source of truth. BuildKit retains its own native background GC policy.
  Date/Author: 2026-08-10 / plan author.

- Decision: local execution and explicit local export are different product
  intents.
  Rationale: local execution needs a runnable image in the daemon only for the
  run lifetime; a complete OCI layout is needless duplication. A developer who
  explicitly requests an export owns the named destination, which must not be
  silently removed by cache GC.
  Date/Author: 2026-08-10 / plan author.

- Decision: expose a top-level `posttrain cache` command and retain
  `posttrain state cache-prune` as a deprecated forwarding alias for one release.
  Rationale: developers think in terms of disk/cache, not internal state-layout
  migration. The top-level noun is discoverable, while the alias preserves
  scripts during the compatibility window.
  Date/Author: 2026-08-10 / plan author.

- Decision: the Posttrain CLI may inspect configured BuildKit usage but does
  not mutate or globally prune the builder.
  Rationale: ai-infra owns the named builder, registry trust, GC policy, and
  retention. Framework code owns package semantics and project cache. Keeping
  mutation on the owning side avoids a convenient CLI becoming a destructive
  cross-service operator.
  Date/Author: 2026-08-10 / plan author.

- Decision: make receipt-only reuse an optional publisher capability instead of
  weakening `JobImagePublicationRequest` with a fake staged path.
  Rationale: a local context is required to build but is not part of remote
  publication identity. `JobImageResolutionRequest` makes that distinction
  explicit and lets non-BuildKit publishers fall back to the normal pack path
  until they implement the same verified lookup.
  Date/Author: 2026-08-10 / implementation.

- Decision: remove framework-owned retained contexts after publication rather
  than retaining them as package history.
  Rationale: the compact record and verified registry image are the recovery
  authorities; retaining a full context couples disk usage to run lifetime and
  recreates the original source duplication. Direct `materialize()` callers
  still own the returned context and its lease explicitly.
  Date/Author: 2026-08-10 / implementation.

## Outcomes & Retrospective

The local implementation slice is complete and its focused validation is
green; live migration and provider qualification are still open. Completion
requires all of the following observable results, not only passing unit tests:

1. An Ambient Reasoning Gym pack does not copy the Hotpot dataset or graph
   store and its package identity does not change when unrelated Hotpot bytes
   change.
2. A work package containing jobs that select different datasets produces a
   distinct per-job dataset closure: each package contains all and only the
   dataset seats selected by that job, regardless of what other jobs or catalog
   entries reference.
3. Repeating an unchanged registry-backed pack verifies and reuses the remote
   image without reconstructing a heavyweight context.
4. After a successful new publication, no retained assembled context or
   internal OCI layout remains, while `job diff`, submission, and a fresh CLI
   process can still resolve the package and image.
5. An interrupted build leaves an expired lease and removable staging tree;
   cache status explains both, and safe collection removes them without
   touching a concurrent build.
6. `posttrain cache status` accounts for project-owned bytes and separately
   reports the configured BuildKit builder's policy/availability. It never
   presents BuildKit bytes as project-cache bytes.
7. LoRA adapter checkpoints, full-model artifacts, evaluation reports, rollout
   traces, and execution reconciliation records remain byte-identical across an
   applied cache prune.

Local evidence so far: 1,141 repository tests passed and 21 skipped on
2026-08-10, with Ruff, Pyright, import-boundary checks, and `git diff --check`
passing. The emergency
Ambient cleanup recovered approximately 42 GiB while retaining unpublished
layouts and artifact/checkpoint state. Record measured before/after sizes, pack
wall time, registry reuse wall time, migration receipts, and any design
correction here as the remaining milestones complete.

## Context and Orientation

### The storage classes

The implementation must make this classification explicit. “Durable” below
means required for a normal replay or proof; it does not mean checked into Git.

| Value | Authority and purpose | Lifetime | Owning code | Collection rule |
| --- | --- | --- | --- | --- |
| Registry actual-job image | Runnable package bytes addressed by OCI digest | Registry retention policy | Registry/BuildKit publication adapter | Never deleted by project cache GC |
| Package materialization record | Plan key, package manifest, component digests/sizes, context digest | Compact machine-local history | `posttrain.execution-pack` contract, CLI store | Retain; explicit history policy only |
| Publication receipt | Publication key to verified image digest and build-definition identity | Compact machine-local verification evidence | `posttrain.execution-buildkit` | Retain under compact history policy; never project cache GC; verify remote on reuse |
| Execution records | Submission, provider id, reconciliation, cleanup evidence | Through lifecycle and operational retention | `posttrain.execution` and CLI | Never deleted by cache GC |
| Source, wheel, dependency, and dataset objects | Reusable content-addressed inputs | Bounded LRU cache | Capability packagers | Collect only when unleased and over policy |
| Assembled build context | One builder-readable view over selected objects | One materialization/publication lease | `posttrain.execution-pack` | Remove on success or failure unless explicit debug retention |
| Internal local OCI layout | Transport between BuildKit and local consumer when direct load is unavailable | One local-use lease | `posttrain.execution-buildkit` | Remove after verified import/use |
| Explicit local OCI export | User-requested deliverable at a named path | User-controlled | CLI export command | Outside Posttrain cache; never auto-collected |
| BuildKit layers/cache mounts | Builder acceleration | Bounded native GC policy | ai-infra | Never globally pruned by framework |
| Checkpoints, adapters, weights, reports, traces | Run outputs and resumability/evaluation inputs | Artifact retention policy | Tracking/object-storage backend | Never discovered or deleted by cache GC |

A *plan key* identifies immutable requests known before fetching or building.
A *package key* identifies the fully materialized `JobPackageManifest`, including
resolved environment wheels, runtime dependency locks, and dataset locks. A
*publication key* binds that package to one publication policy/repository. A
*lease* is a small atomic record saying a process currently needs one or more
cache objects or a staging workspace. A lease is not an mtime heuristic.

### Current code paths

`apps/cli/src/posttrain_cli/execution_planning.py` plans, materializes, and
publishes. `PlannedJobPackage.materialize()` creates source snapshots under
`pack/sources`, reusable environment inputs under `pack/cache`, datasets under
`cache/datasets`, and a retained context under `pack/contexts`.

`packages/execution-pack/src/posttrain/execution_pack/service.py` builds the
context in `.job-context-stage-*`, computes `JobPackageManifest`, and atomically
renames it to `contexts/<package_key>`. It correctly cleans its temporary trees
in a `finally`, but a process killed by disk exhaustion or SIGKILL leaves those
trees behind, and the successfully retained destination has no expiry.

`packages/execution-buildkit/src/posttrain_execution_buildkit/job_image.py`
checks small remote receipts and produces full local OCI layouts. It currently
accepts one `receipt_root` for both. The publisher verifies registry bytes by
digest; that verification should remain the remote reuse gate.

`apps/cli/src/posttrain_cli/package_history.py` reads manifests from retained
contexts. It must read compact materialization records before context retention
can change.

`apps/cli/src/posttrain_cli/state_layout.py` safely bounds paths and protects
unknown state, but its cache pruning works at whole-subtree granularity.
`apps/cli/src/posttrain_cli/commands/state.py` exposes the current
`posttrain state cache-prune` command.

Machine-wide settings are loaded from `~/.config/posttrain/config.toml` by
`load_machine_config()` in `apps/cli/src/posttrain_cli/execution_config.py`.
Cache budgets are operational machine policy and belong there, not in tracked
project configuration and not in package identity.

### Repository boundaries

The primary implementation is in `/home/hammad/projects/rl`. The Ambient
migration and live qualification are in `/home/hammad/projects/ambient-agent`.
The existing BuildKit policy and builder qualification are in
`/home/hammad/projects/ai-infra` at inspected commit
`17c678ab6622756c17c8e733d88677e2e1d78f41`; that checkout currently contains
unrelated release and fleet changes, which must be preserved. No ai-infra code
change is planned unless qualification shows the committed GC policy is
insufficient. If one is required, commit and validate it in ai-infra before
depending on it from framework documentation.

The RL checkout also contains unrelated modifications to the checkpoint plan,
Verifiers integration/tests, and release manifest. The Ambient checkout has
uncommitted Reasoning Gym work-package and catalog edits. Implementation must
not overwrite or silently include those changes in a cache-lifecycle commit.

## Plan of Work

### Milestone 1: durable compact package records

Add provider-neutral values in `packages/execution-pack`:

    PackageMaterializationRecord
      schema
      plan_key
      package_key
      context_digest
      manifest
      component_sizes
      created_at

    PackageMaterializationStore
      resolve(plan_key) -> PackageMaterializationRecord | None
      commit(record) -> PackageMaterializationRecord
      list(work_package_id?, job_id?) -> tuple[record, ...]

The record must validate that `manifest.package_key == package_key`, contain
only JSON values, use an atomic mode-0600 write, and reject conflicting records
for an existing plan key. Store it under a protected control path such as
`.posttrain/state/packages/materializations/<plan_key>.json`, not below
`state/cache`. The exact path is an application decision; reusable packages
receive the value/protocol but must not depend on `ProjectLayout`.

Split BuildKit's `receipt_root` into a protected publication-record root and a
cache/temp root. Remote receipt JSON moves to
`.posttrain/state/packages/publications/<publication_key>.json`. A compatibility
reader may import a valid legacy receipt from `cache/pack/publications`, verify
its registry digest, write the new record atomically, and leave deletion to the
migration plan. Full `local-layouts` never move into the protected root.

Rewrite `package_history.py` over the materialization store. `job diff` must
work after `contexts/` is absent. Add `PlannedJobPackage.resolve_published()`:
given its `plan_key` and `publication_plan_key`, it loads the matching
materialization/publication records, validates all identities, and asks the
publisher to verify the remote digest. On success it returns
`PublishedJobImage(cache_hit=True)` without calling `materialize()`. On a
missing, stale, or remotely collected publication it falls through to normal
materialization and rebuild.

Do not make a local record alone sufficient to launch. Remote reuse still
requires registry verification. Do not query Trackio for package cache hits;
job image publication and run evidence are separate concerns.

### Milestone 2: job-specific source and dataset inputs

Make the dataset-selection invariant explicit in `JobPackSpec`: its
`datasets` tuple is the canonical, seat-name-ordered closure of
`DatasetLoadPlan` values reachable from the selected `PreparedWorkPackageJob`
only. Planning must not scan the dataset catalog, sibling jobs in the work
package, the project `data/` directory, or prior materialization cache entries.
If a selected job has no dataset seat, the package contains no `datasets/`
payload. If it has several dataset seats, every selected seat is present once
and no unselected seat is present.

Change dataset packaging so a static project dataset is selected from the
original project root through a bounded source descriptor, not from the
application-code snapshot. The descriptor must normalize a project-relative
path, reject symlinks and escapes, enforce file/count/byte limits, and derive a
digest before materialization. Only the selected dataset and its explicitly
declared assets enter the staged package. “Explicitly declared assets” means
assets named by that dataset selection or its materialization manifest; the
packer must not infer neighboring files or recursively copy a parent data
directory. Its absolute submit-host path never enters `JobPackSpec`, a lock, or
`package.json`.

Keep `[tool.posttrain.pack].source_includes` as the code allowlist specified by
the canonical API. Add validation for known dataset/resource overlap: if a
selected dataset or activation resource also lies under an admitted source
tree, fail planning with an actionable explanation instead of copying it
twice. Do not ban a directory merely because it is named `data`; small package
fixtures or templates may be legitimate code assets. The semantic overlap and
size are the problem, not the spelling.

Add a read-only size inspection to planning. Human output for `job pack` and
JSON output must identify at least project source, framework distributions,
each selected dataset seat with selection id/revision and estimated bytes,
activation resources, cached environment packages, and estimated assembly
bytes. Before copying, show the largest selected roots
when a component crosses its configured budget. The initial default policy is
machine-configurable; choose conservative defaults from measured canaries, not
arbitrary constants in package contracts.

Extend `~/.config/posttrain/config.toml` with a versioned `[cache]` table owned
by a `MachineCachePolicy` value. It should include a total project cache budget,
a minimum host-free-space guard, maximum ages for reusable objects and failed
debug staging, and an optional explicit debug-retention flag. Cache policy is
not hashed into job or package identity.

Update Ambient's `pyproject.toml` only after the new dataset source path passes
tests: remove `data` from `source_includes`. Verify every Ambient catalog
dataset/resource still reaches the package through its owning selection. A
change to unrelated Hotpot data must not change a Reasoning Gym plan or package
key. Also pack two sibling Ambient jobs with different dataset selections and
inspect their OCI contexts: neither package may contain the other's dataset.

### Milestone 3: lease-scoped assembly and crash recovery

Add a provider-neutral cache inventory and lease contract, preferably in
`packages/execution-pack` because it governs pack objects rather than provider
runs. The application store records lease id, process id, machine identity,
creation/heartbeat/expiry time, purpose, and exact owned paths or object keys.
Acquire it before creating a staging tree; publish the tree atomically; release
it in `finally`. A second process using the same immutable object creates or
joins a reference without mutating the first lease.

Do not delete based on PID alone because records can cross containers or stale
PID reuse. A lease is active when its nonexpired heartbeat belongs to the same
machine/process identity. Expired leases become collection candidates only
after their paths pass the same root, symlink, and expected-name checks used by
state pruning.

Change `JobPackService` from “atomically retain one actual-job context” to
“atomically materialize one leased actual-job context.” The publisher consumes
the leased path. After a verified remote publication and committed compact
records, release and remove the context. On ordinary exceptions, remove it and
retain a compact failure diagnostic containing the plan key, selected
component sizes, sanitized builder transcript reference, and error category.
An explicit debug option may retain the failed context with a short expiry and
must print that expiry and byte cost.

Reusable source, wheel, dependency, and dataset objects remain
content-addressed. Record size and last successful use in an inventory index;
do not update every file's mtime. Assembly should use filesystem reflinks when
the source and destination support them, with ordinary bounded copies as the
correctness fallback. Do not use hardlinks if later normalization could mutate
the cached inode.

At the start of pack, recover expired staging leases and ensure the configured
free-space guard. At the end, release the current lease and collect only if the
project cache is over budget. If enough safe bytes cannot be reclaimed, fail
before the expensive copy and print the same candidate/protection information
as `posttrain cache status`.

### Milestone 4: cache inventory and developer experience

Add `apps/cli/src/posttrain_cli/commands/cache.py` and register a top-level
`cache` Typer application:

`posttrain cache status` reports total bytes, policy, free-space guard, and a
breakdown of reusable, leased, expired, protected, and immediately reclaimable
bytes. It also attempts read-only BuildKit inspection for the configured named
builder and labels missing/unavailable builder telemetry rather than failing
the project inventory.

`posttrain cache explain [KEY]` explains why one object/package is retained,
which materialization or lease references it, its size and last-use time, and
whether deleting it would require rebuilding, republishing, or neither. With no
key it explains the storage classes and current largest consumers.

`posttrain cache prune` computes a deterministic mark-and-sweep plan. Roots are
active leases and protected control records; candidates are unreferenced cache
objects ordered by expired staging first, then class priority, then LRU. The
default is dry-run. `--apply` executes the immutable plan with precondition
checks so a new lease or changed inode causes a skip, not a race. Output must
show reclaimed and skipped bytes. `--json` retains the normal CLI structured
output contract.

The command only mutates the selected project's `.posttrain/state/cache`.
BuildKit status tells the developer whether infrastructure GC is configured,
but applying that policy remains the ai-infra procedure. Artifact directories,
Trackio, object storage, Docker images outside the named builder, registry
manifests, and unknown state entries are outside scope.

Keep `posttrain state cache-prune` as a deprecated alias that forwards to the
new planner. It must not retain the existing “delete the entire pack tree”
behavior after the new inventory is available. Document the removal version in
the CLI deprecation warning and release notes.

The normal successful pack summary should be compact and useful, for example:

    Job image reused: registry.lan/ambient/posttrain-job@sha256:...
    Package: 8f31...  source 41 MiB · datasets 1/1, 12 MiB
    Local build material: 0 B retained · project cache 7.2/40 GiB

Do not make every pack print a cleanup essay. Detailed classification belongs
behind `posttrain cache status` and JSON output.

### Milestone 5: local execution versus local export

For a single-platform `local-docker` run, add a BuildKit output that loads the
qualified image directly into the selected local daemon/store. Return a typed
local image handle and bind its lifetime to the execution lease. Remove the
daemon tag/image after terminal reconciliation only when Posttrain created it
and no other active local execution references it. Never remove shared parent
images or unrelated tags.

Change explicit local package export to require or strongly guide a named
destination, for example:

    posttrain job pack WORK_PACKAGE --job JOB --local-output ./job-image.oci

Write to a sibling temporary path, verify `index.json`, then atomically replace
the requested destination. The resulting layout and a compact adjacent receipt
are user-owned outputs outside `.posttrain/state/cache`. Repeating the command
may reuse BuildKit layers, but the CLI must never pretend that a deleted cache
layout is a durable export.

If a platform or daemon cannot use direct load, use a leased internal OCI
layout as transport, verify import, and remove it immediately. Preserve the
existing type rule that a `LocalPublishedJobImage` cannot be submitted to a
remote provider.

### Milestone 6: migration and real qualification

Implement a migration planner for legacy pack state. It inventories each
legacy context, extracts and validates `package.json` into a compact
materialization record, imports valid remote receipts after registry
verification, classifies full local layouts as reclaimable unless they were
explicitly exported, and identifies staging directories left by dead builds.
Dry-run output includes totals and refuses ambiguous/symlinked/dirty entries.

Apply migration in small immutable batches, journaling each decision so an
interruption is retryable. Do not delete a legacy context until its compact
record is committed and any referenced remote publication is verified. Do not
delete a local layout that is the only named output of an unfinished local
operation. Existing execution and artifact records are read-only inputs to
protection, never migration targets.

In ai-infra, from the existing dirty checkout, run the read-only builder plan.
When no build is active, recreate `posttrain-builder` using the repository
procedure and qualify its retained state volume, policy, and registry CA:

    cd /home/hammad/projects/ai-infra
    ./scripts/configure-buildkit-builder plan
    ./scripts/configure-buildkit-builder apply --confirm-restart
    ./scripts/qualify-buildkit-builder

Do not use `docker system prune`, `docker builder prune`, or an unnamed Buildx
prune as part of qualification.

Run three real canaries from Ambient:

1. a Reasoning Gym two-step local 4090 job proving selected data and direct
   local-image lifecycle;
2. a registry-backed pack of the same immutable job proving publication and
   immediate context release; and
3. an unchanged repeated pack from a fresh CLI process proving remote reuse
   without materialization.

Finally run an eval or train package that genuinely selects Hotpot data to
prove that removing global `data` source inclusion did not make declared data
unavailable. Compare manifests and component sizes, not only container exit
status.

### Milestone 7: release and consumer adoption

Commit framework changes by ownership: provider-neutral contracts and tests,
BuildKit adapter and tests, CLI/store/DX and tests, then documentation. If
ai-infra required a code change, commit and qualify it separately before the RL
release records the dependency. Commit Ambient source-selection migration only
after the released framework is available; update its exact Posttrain pin and
lockfile, then repeat the package reuse gate from released wheels.

The release evidence must include exact framework commit/version, package and
publication keys, registry digest, before/after local bytes, BuildKit
qualification receipt, CLI dry-run/applied prune receipts, and proof that model
and checkpoint artifacts were unchanged. Release notes describe the developer
behavior, not internal cache implementation.

## Concrete Steps

Work from `/home/hammad/projects/rl` unless a command names another repository.
Before each milestone, inspect `git status --short` and stage only owned files.

For Milestone 1, add focused tests first in
`packages/execution-pack/tests/`, `packages/execution-buildkit/tests/`, and
`apps/cli/tests/`. Prove conflicting records fail, atomic retries are
idempotent, remote verification is mandatory, and `job diff` works with no
context directory. Then implement the records and compatibility readers.

For Milestone 2, add a fixture project with at least three catalog datasets and
two jobs: one job selects one dataset seat; the other selects two. Include a
large unrelated data tree and a sibling unselected dataset beside a selected
file. Assert that each plan's `DatasetPackRequest` tuple and each staged
`DatasetPackageLock` tuple match the selected job's seats exactly; unselected
bytes are neither read, copied, nor represented in the manifest; no neighboring
file is inferred as an asset; and unrelated changes do not change `plan_key` or
`package_key`. Add overlap and top-contributor diagnostics before changing
Ambient.

For Milestones 3 and 4, use a temporary project state with two processes or a
controlled lease clock. Prove a prune planned before a new lease skips that
object at apply time. Simulate SIGKILL by leaving a staging directory plus an
expired lease, then recover it. Do not use sleeps for lease tests; inject time.

For Milestone 5, retain fake-gateway tests and add Docker integration tests
behind the existing Docker marker. Verify exact tag ownership and ensure a
user-owned export outside state survives every cache operation.

Use the focused ladder during implementation:

    uv run pytest packages/execution-pack/tests -q
    uv run pytest packages/execution-buildkit/tests -q
    uv run pytest apps/cli/tests/test_state_layout.py apps/cli/tests/test_cli.py -q
    uv run ruff check <changed paths>
    uv run pyright <changed packages when supported>
    uv run lint-imports
    git diff --check

Before release, run the repository ladder:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Capture live disk evidence with scoped commands rather than global deletion:

    du -sh .posttrain/state/cache/pack/*
    posttrain cache status --json
    posttrain cache prune --json

The applied command is allowed only after reviewing the dry-run receipt:

    posttrain cache prune --apply --json

## Validation and Acceptance

The feature is accepted when all of these checks pass:

- Contract tests show package and publication records reject mismatched keys,
  unsupported schemas, dirty replacement, symlinks, and group/world-readable
  protected files.
- A registry receipt without a remotely present matching digest is never a
  cache hit and triggers reproducible rebuild or an actionable failure.
- A valid receipt plus materialization record returns an image without invoking
  source materialization, dependency compilation, dataset packaging, or
  BuildKit build.
- Package history and `job diff` read only compact records; removing all
  contexts does not change their result.
- Dataset source resolution stages only selected files/assets and rejects path
  escape, symlink, special file, changed digest, budget overflow, and overlap
  with application source.
- A multi-job/multi-dataset fixture proves dataset closure is computed from the
  selected job only: zero-seat jobs stage zero datasets, one-seat jobs stage one,
  and multi-seat jobs stage every selected seat exactly once. Catalog-only,
  sibling-job, neighboring-directory, and cache-only datasets never enter the
  package.
- Lease tests cover concurrent readers, writer failure, expired recovery, new
  lease between prune plan/apply, and process restart.
- Every destructive cache action is rooted below the exact project cache,
  rejects symlinks, uses expected inode/key preconditions, and skips unknown
  entries.
- Cache status accounts for filesystem bytes within an explained tolerance and
  reports BuildKit as a distinct backend scope.
- Automatic collection occurs only under configured pressure and never blocks
  a successful cache hit on unnecessary cleanup.
- Direct local load and explicit OCI export produce equivalent package
  manifests. The export survives cache prune; transient transport does not.
- The Ambient Reasoning Gym canary excludes unrelated Hotpot bytes. A selected
  Hotpot canary still includes and verifies its own declared data.
- A before/after artifact inventory proves no `training-checkpoint`,
  `model-adapter`, `model-weights`, eval report, trace, or run reconciliation
  record changed.
- After the unchanged repeated registry pack, project-local heavyweight bytes
  remain within policy and no `.job-context-stage-*`, `.job-context-work-*`, or
  internal `local-layouts/<key>` survives without an active lease.

## Idempotence and Recovery

All record writes use temporary files, fsync where the existing state-store
convention requires it, and atomic rename. Repeating a commit with identical
content is a no-op; conflicting content fails closed. Contexts and local
transport layouts are created below unique lease ids, so a retry never mutates
an in-use path.

Migration is plan/apply and journaled. If it stops after writing a compact
record but before deleting a legacy context, rerunning observes the identical
record and resumes deletion. If registry verification fails, preserve the
legacy context and report it; do not infer that local bytes are disposable.

Cache prune separates planning and application. Apply revalidates every
candidate's key, path, type, lease state, and observed size/inode facts. A
changed candidate is skipped and recorded. It is always safe to rerun.

If a released migration must roll back, keep the compatibility readers and old
CLI alias for the stated window. Restoring the prior framework must still see
legacy state left during dry-run; it need not understand new compact records.
Do not restore heavyweight contexts from backup merely to roll back—the
verified registry image is the recovery source.

If BuildKit recreation fails, its ai-infra procedure restores the preceding
configuration and retains its named state volume. Project cache migration does
not depend on builder recreation and may be retried later. If the registry is
unavailable, no remote receipt is newly trusted or discarded.

## Artifacts and Notes

The plan began after an explicit cleanup of names containing `smoke` and the
named BuildKit state on the workstation. That emergency operation recovered
about 134 GiB of free disk but is not the target product workflow. The retained
199 GiB project state is useful migration evidence and should not be manually
blanket-deleted before the migration dry run is captured.

The failed Reasoning Gym pack never reached a provider. Its decisive error was
`no space left on device` while BuildKit copied a Hotpot Kuzu graph path from
the monolithic source context. This is the live acceptance case for Milestone
2, not just a synthetic performance benchmark.

## Interfaces and Dependencies

The final implementation should expose or refine these interfaces without
creating imports from reusable packages into the CLI application:

- `posttrain.execution_pack.PackageMaterializationRecord` and a minimal store
  protocol; the application supplies the project-state implementation.
- `posttrain.execution_pack.CacheLease`, `CacheObject`, `CacheInventory`,
  `CachePrunePlan`, and `CachePolicy` as provider-neutral values where useful.
- `JobPackService` returning a context-managed or explicitly releasable
  materialization rather than an implicitly permanent context path.
- A dataset-source resolver that accepts bounded declared project inputs while
  persisting only relative paths and content locks.
- `BuildKitJobImagePublisher.resolve()` (or an equivalent read path) that
  verifies a committed remote receipt without requiring `staged_context`.
- Separate BuildKit paths/types for protected remote publication records,
  internal transient local layouts, and user-selected export destinations.
- `MachineCachePolicy` loaded from `~/.config/posttrain/config.toml`, with
  packaged safe defaults and no effect on package identity.
- Top-level CLI commands `cache status`, `cache explain`, and `cache prune`, all
  supporting structured output and dry-run-first mutation.

Do not add Trackio, W&B, Docker, BuildKit, dstack, or project layout imports to
`posttrain.common`. `execution-pack` must remain provider-neutral. BuildKit
inspection and publication remain in `execution-buildkit`; the CLI composes
them. No train/eval/serve package may import another capability package to
implement cache policy.

Revision note (2026-08-10): initial plan created from canonical product
contracts, current implementation tracing, live Ambient disk inventory, and
the existing ai-infra BuildKit policy. It replaces the earlier informal “keep
some cache and run LRU” recommendation with class-specific authority,
lease-based safety, selected-data packaging, and an explicit developer-facing
lifecycle.

Revision note (2026-08-10): strengthened dataset relevance from a general
source/data separation goal into a two-part invariant: dataset requests are the
selected job's exact resolved-seat closure, and no unselected dataset bytes may
leak through source snapshots, neighboring paths, sibling jobs, catalogs, or
caches. Added zero/one/multi-seat fixtures and per-seat pack evidence.

Revision note (2026-08-10): recorded the first implementation and disk cleanup
slice. Added a conservative granular `state cache-prune` policy and preserved
unpublished local layouts and heavyweight migration evidence until compact
records and leases are available.

Revision note (2026-08-10): implemented the first compact record path and
legacy-compatible package history reader. Registry receipt relocation and
context lease/reclamation remain later milestones.

Revision note (2026-08-10): separated receipt and local-layout roots and added
  expiring cache leases for materialization and local OCI publication. The next
  work is lease-aware context reclamation and compact-record migration of the
  existing Ambient state.

Revision note (2026-08-10): added the top-level cache status, explain, and
  prune commands with structured output and dry-run-first behavior.

Revision note (2026-08-10): completed the local receipt-reuse and context
reclamation slice. Materialization records now retain the plan key, publishers
can resolve a verified image without a staged context, and successful or
failed planned publication removes only framework-owned retained contexts.
The remaining work is legacy-state migration, local-daemon/export semantics,
BuildKit host qualification, and released Ambient adoption.
