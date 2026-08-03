# Complete the 0.3.0 developer-experience program and deliver safe cross-plane purge

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

This is the closure plan for the DX work that remained open after the
repository, configuration, lifecycle, packing, public-API, and release plans
were started. It is deliberately an umbrella plan: it preserves those six
plans as their detailed design records while imposing one dependency order,
one definition of done, and one evidence packet for the 0.3.0 release.

Posttrain 0.3.0 has now shipped. Its release qualification and the intentional
post-release removal of framework history are completed outcomes, not pending
gates. The remaining release follow-up in this plan is the public, previewed
cross-plane purge workflow that 0.3.0 explicitly did not claim to provide.

## Purpose / Big Picture

A framework maintainer must be able to prepare a project, package an isolated
job, submit it, observe it, reconcile it automatically, and deliberately
remove the job and its outputs when the work is no longer wanted. The same
workflow must work for the framework's Lab project without making Lab a hidden
requirement for normal users. The shipped release proved planning, execution,
evidence, and cleanup; the follow-up must prove that destructive purge applies
only an exact, independently reviewed cross-plane plan.

After this plan, the framework root is a virtual workspace, `apps/lab` owns
the `posttrain-lab` qualification project, and there is no prematurely-created
`posttrain-integration` project. The name is reserved until an independently
owned integration application, not a test fixture, has its own jobs, data
retention policy, and operator. `posttrain project purge` can preview every
resource owned by the opened project across tracking, provider, OCI, and local
state. `posttrain run purge` can plan deletion of one exact terminal run, tells
the user when its produced artifacts are consumed, and only follows the
complete downstream closure when the user explicitly requests a cascade.
Preview and apply are separate commands so an operator or automation applies
the exact reviewed plan rather than recomputing scope at deletion time.

The maintainer demonstrates the follow-up with disposable local and dstack
fixtures, including producer-to-consumer lineage and an isolated project. They
can inspect each immutable plan, apply it interactively or with a bound digest,
resume a deliberately interrupted apply, and retain the machine-scoped receipt
after every target project/run resource is gone. This is evidence of a safe
operator workflow, not merely green unit tests.

## Progress

- [x] (2026-08-01) The six component DX plans were authored and their first
      slices landed. `docs/plan/dx-*-*.md` records the detailed decisions and
      the evidence that already exists.
- [x] (2026-08-01) The framework root was made a virtual workspace and the
      Lab control tree was moved to `apps/lab/.posttrain`; the local release
      staging and OCI packing proof has passed.
- [x] (2026-08-01) A classified cache prune, release metadata drift check,
      `posttrain.env` resolution, and the first public `Project.open()` seam
      were implemented and tested.
- [x] (2026-08-01) A local, committed Trackio prototype adds authenticated
      project deletion preview and purge for SQLite and Doris storage. Its
      focused tests and wheel build passed before its release metadata was
      raised to `0.31.5.post6`.
- [x] (2026-08-01) Milestone 1: the Lab id is `posttrain-lab` in committed
      source. The project manifest, catalog layer, all 25 work packages, the
      Lab package sources and tests, the two legacy qualification launchers
      that still write Trackio, and the cleanup plan's present-tense contracts
      were updated together. `apps/lab` tests pass (76 passed, 2 skipped),
      `project show` reports `posttrain-lab`, and `qualification list` still
      classifies 25 work packages with 11 active gates and no unclassified
      file. The Decision Log records why `posttrain-integration` is not
      created.
- [x] (2026-08-01) Stabilized the recovered Trackio repository, merged post6
      at `82fed847`, published its reproducible wheel to the development index,
      and deployed it through the narrow ai-infra Ansible playbook. Both live
      endpoints report `0.31.5.post6`; an authenticated preview against a
      nonexistent project returned zero owned resources. The later deletion of
      `foundation-models` is recorded in the post-release cleanup outcome below.
- [x] (2026-08-01) Published the exact merged Trackio post6 wheel and sdist to
      the stable Python index, tagged `carbonteq-v0.31.5.post6`, changed the
      framework to consume the indexed wheel, and regenerated the catalog and
      runtime dependency locks. The release manifest is intentionally stale
      until the replacement runtime image publication completes.
- [x] (2026-08-01) Published the 0.3.0 base and all six job-kind images from
      source revision `1030dcae`, including the isolated veRL backend, and
      regenerated the digest-pinned manifest. The base retains both public and
      LAN package-index trust; every kind installed Trackio post6 from the
      stable index. Release consistency and focused image/release tests pass.
- [ ] Implement framework-owned run/project purge planning and execution with
      artifact-consumer closure, machine-scoped plans/receipts, Trackio and OCI
      lifecycle adapters, provider cleanup actions, and local-state removal.
- [ ] Complete the remaining broader configuration, lifecycle, packing, and
      public-service milestones listed below without retroactively making them
      v0.3.0 release blockers.
- [x] (2026-08-01) Configuration milestone 2 is complete: one automatically
      loaded machine config owns shared endpoints, provider locators, storage,
      trust, and registered projects; named mode-0600 sources scope Trackio,
      Hugging Face, Python-index, and dstack credentials. Project init no
      longer writes a machine binding, `posttrain machine init/show/project
      add` provide the operator workflow, and execution-dstack/Ansible—not the
      home config—own worker storage. Ruff, Pyright, the release/lock checks,
      and the full suite pass (894 passed, 17 skipped).
- [x] (2026-08-01) Re-scoped the release gates to the maintainer's selection:
      a SAMPO capsule over Zapier AutomationBench and the existing DAPO capsule
      over GSM8K, one launched through the local provider and one through
      dstack on this machine. Added the
      `automationbench/qwen3.5-0.8b/sampo-mtp-capsule-v1` settings, the
      `automationbench_sampo_qualification.yaml` work package, and its
      `automationbench-sampo` registry entry; both gates now plan. The
      registry classifies 26 work packages with 11 active gates and 15
      candidates.
- [x] (2026-08-01) Repointed the `automationbench-v1` environment source at the
      repository that actually contains it — this one, subdirectory
      `environments/automationbench_v1` — in all three places the identity was
      authored, and deleted the dead third copy. Removed the environment from
      Lab's `gpu-posttrain` extra and `[tool.uv.sources]`, which dropped it and
      `carbonteq-automation-bench` from the workspace lock, and regenerated the
      catalog dependency-lock fingerprint. The full suite passes with 889
      passed and 17 skipped; lock check, ruff, and import contracts are clean.
- [x] (2026-08-01) Pushed `codex/dx-0.3.0`; the environment source revision
      and subsequent framework fixes now name commits available from the
      remote rather than local-only objects.
- [x] (2026-08-01) Qualified the merged AutomationBench fork at revision
      `908db2ab` (75 focused tests, wheel/sdist metadata, and clean Python 3.12
      wheel install), published `carbonteq-automation-bench==1.0.5.post1` to
      the development index, and promoted the identical artifacts to stable.
      The native adapter now selects that exact registry version, its isolated
      lock changed only the maintained-fork entry, and the obsolete consumer
      Git constraint is gone.
- [x] (2026-08-01) Routed the machine-owned, credential-free Python index URL
      and its separately scoped credentials into immutable environment
      dependency compilation. The compiler still ignores ambient index
      variables, runs with `--no-config`, rejects credential-bearing URLs, and
      binds resolved artifact hashes—not credentials—into package identity.
- [x] (2026-08-01) Released v0.3.0 from signed commit `df7e85d`. GitHub CI and
      the clean external consumer passed; the packed Lab data-preparation gate
      retained its dataset artifact, and the managed Qwen 3.5 2B GSM8K gate
      reconciled exit 0 with both Verifiers traces and 2/2 successful bounded
      rollouts. These were the actual release gates; the previously proposed
      SAMPO/DAPO pair remains broader qualification work rather than retroactive
      release evidence.
- [x] (2026-08-02) Completed the operator-authorized one-off post-release
      history removal: deleted five framework Trackio projects, all 99 terminal
      dstack Posttrain submissions, all 25 addressable
      `carbonteq/posttrain-job` manifests, seven stopped local job containers,
      16 local job images, and about 14 GB of stale local state. Preserved shared
      base/kind images, reusable machine cache, and the `ambient-agent`,
      `occupancy-research`, and `ai-infra` projects. The v0.3.0 release notes now
      state both the cleanup policy and that live release evidence was
      intentionally removed after verification.
- [x] (2026-08-02) Reworked the pending purge milestone around the post-release
      state and the final operator DX: content-addressed preview plans,
      separate show/apply commands, exact run selection, dependency closure,
      digest-bound automation, machine-scoped resumable receipts, complete
      cross-plane inventory, and disposable qualification fixtures.
- [x] (2026-08-02) Implemented the provider-neutral first slice in
      `packages/execution`: immutable `PurgeAction`, `PurgePlan`, and
      `PurgeReceipt` contracts plus the mode-0600 machine-scoped `PurgeStore`.
      The focused suite passes (4 tests), Ruff and Pyright are clean. Backend
      adapters and CLI mutation are intentionally still pending.
- [x] (2026-08-02) Added the optional provider-neutral tracking lifecycle
      contracts in `packages/tracking` for exact run/project previews and
      receipts. The execution and tracking package suites pass together (112
      tests), with package-wide Pyright clean. No backend is treated as purge
      capable until it implements these contracts.
- [x] (2026-08-02) Added the first Trackio fork integration slice in sibling
      repository `/home/hammad/projects/trackio`: exact provider-run preview
      and apply endpoints, consumer-aware blockers, SHA-256 preview binding,
      SQLite and Doris storage deletion, and a CAS-retention test. The focused
      Trackio artifact suite passes (10 tests) and changed fork files pass
      Ruff. The follow-up was committed in the sibling repository as
      `946622a` (`feat: add digest-bound run and project purge APIs`), but it
      remains unpublished; the framework pin and deployed images stay on post6
      until that commit is published and qualified.
- [x] (2026-08-02) Added `TrackioLifecycleAdmin` in
      `packages/tracking-trackio`, capability-detected against the new fork
      methods and covered by a digest-bound preview/apply adapter test. The
      adapter maps provider artifacts and receipts into the neutral tracking
      contracts without importing Trackio into `packages/execution`.
- [x] (2026-08-02) Validation for this slice is green: framework execution,
      tracking, and Trackio adapter suites pass (136 tests), framework Pyright
      is clean for the changed packages, and the sibling Trackio unit suite
      passes (385 passed, 4 skipped). Both repositories pass `git diff --check`.
- [x] (2026-08-02) Added the neutral journaled apply engine and strict OCI
      manifest contracts. `apply_purge_plan` resumes completed actions from
      the immutable journal, revalidates immediately before mutation, and
      fails closed on blockers or missing executors. The BuildKit package now
      has a credential-aware Distribution HTTP adapter with exact digest HEAD
      and DELETE semantics; focused execution/registry/BuildKit tests and
      Pyright are clean.
- [x] (2026-08-02) Added the first closure planner and guarded CLI surface:
      exact `run purge`, blocked `project purge`, offline `purge show`, and
      digest/confirmation-gated `purge apply`. CLI previews persist machine
      scoped plans and never mutate. Missing tracking-lineage and provider
      adapters are represented as explicit blockers rather than guessed.
      Focused CLI and execution tests pass (48 tests).
- [x] (2026-08-02) Added safe local-state and execution-provider action
      executors. Local deletion is confined to exact run directories below
      configured state roots and refuses roots/symlinks; provider actions
      revalidate the stored handle and terminal state before calling the
      existing provider cleanup contract. Added the Trackio action bridge so
      each tracking action obtains a fresh consumer-aware, digest-bound
      provider plan before apply.
- [x] (2026-08-02) Added project-level inventory planning and made the CLI
      project preview use it. Empty projects and unmatched runs are explicit
      plan blockers; project apply remains unavailable until the cross-plane
      inventory adapters are connected. The planner suite now covers both
      exact-run closure and project mismatch cases.
- [x] (2026-08-02) Connected the guarded apply surface to the concrete action
      executors: execution-provider cleanup, exact OCI Distribution deletion,
      Trackio digest-bound run deletion, and scoped local-state removal. The
      CLI still fails closed when the installed Trackio client is post6 and
      does not expose the new run-purge API; no dependency pin was advanced.
- [x] (2026-08-02) Extended the Trackio project boundary to return a stable
      digest-bound preview and require that digest for project apply, while
      preserving the legacy endpoint shape for older callers. The neutral
      Trackio adapter now maps project plans/receipts as well as run plans.
      The sibling focused artifact suite passes (11 tests); publication and
      framework pin advancement remain intentionally pending.
- [x] (2026-08-02) Completed the project tracking action path in the neutral
      planner and Trackio adapter: project plans now delete the project only
      after all selected run actions, and the server/client bind project apply
      to the preview digest. Final focused validation for the changed surfaces
      is green (191 framework tests; 386 Trackio unit tests, 4 skipped).
- [x] (2026-08-02) Added adapter coverage for digest-bound Trackio project
      deletion, including logical/storage byte mapping and receipt identity.
      The changed framework suite is now 192 focused tests; Trackio remains
      green at 386 unit tests with 4 known hardware skips.
- [x] (2026-08-02) Added a disposable three-run producer→consumer→leaf
      qualification fixture for the neutral planner and journaled apply. It
      verifies leaf-to-root tracking order, provider/registry/tracking/local
      plane sequencing, an interrupted registry action, and resumable retry
      without widening the immutable plan.
- [x] (2026-08-02) Published the Trackio lifecycle API as
      `carbonteq-trackio==0.31.5.post8` from merged commit `77db6f5c`, tagged
      `carbonteq-v0.31.5.post8`, promoted the identical wheel to the stable
      index, and deployed it to both shared and Doris-candidate services. The
      wheel SHA-256 is
      `9b5ce6df75a6daa40478d3d2d48f4ae8e2c6b8b507d0ca57556786d217fe8d62`;
      both live `/version` endpoints report post8.
- [x] (2026-08-02) Qualified post8 project and run preview/apply against
      disposable SQLite and Doris projects. Both backends rejected stale
      digests with actionable client-visible messages, accepted current
      digests, deleted their two-run projects, and reported the projects
      absent afterward.
- [x] (2026-08-02) Advanced the framework to the stable post8 wheel in commit
      `089f407a`, regenerated its catalog/runtime locks, and published all five
      affected 0.3.0 kind images from the immutable pin. Release consistency,
      85 focused tests with one hardware skip, Ruff, and Pyright pass from a
      clean detached worktree; every published registry digest resolves.
- [x] (2026-08-02) Ran the live three-run cross-plane qualification against
      Doris Trackio, OCI Distribution, the provider cleanup executor, and the
      local-state executor. Non-cascade preview blocked on the producer's
      consumer; cascade selected producer→consumer→leaf. A forced failure at
      `registry:consumer` left the first manifest deleted and the second
      present, then the same `purge-f1d486e43adaeabd` plan resumed. Trackio
      applied leaf→consumer→producer, and all three provider records,
      manifests, workspaces, runs, and the disposable project were absent.
      Durable sanitized evidence is under
      `release-evidence/cross-plane-purge/`.
- [x] (2026-08-03) Published the hash-identical post8 wheel and sdist on the
      existing `carbonteq-v0.31.5.post8` GitHub tag for public CI, then opened
      framework PR #11 for the qualified purge and environment-library slice.
      The public workflow now verifies the mirrored wheel digest and installs
      the remaining committed lock with `uv sync --frozen`; a release-tooling
      regression test binds its URL, filename, version, and SHA-256 to
      `uv.lock`. This is a Python artifact mirror, not a GHCR publication.
- [ ] Complete the non-blocking authoring and release-automation follow-up
      milestones before declaring the entire DX program, rather than merely
      the release, complete.

## Surprises & Discoveries

- Observation: the first package-based SAMPO pack failed closed because the
  native adapter still exposed AutomationBench as a pinned transitive Git
  requirement. `uv pip compile --generate-hashes` correctly emitted that VCS
  requirement without a hash, and the immutable dependency compiler correctly
  rejected it. Publishing the already-prepared maintained fork distribution
  closes the portability gap without weakening hash validation or pretending
  that AutomationBench is provided by the kind image.

- Observation: publishing the maintained fork exposed a missing configuration
  handoff: execution and image publication received the machine Python index,
  but host-side environment dependency compilation did not. An internal
  distribution therefore remained invisible to the compiler despite being a
  valid machine service selection. The explicit compiler gateway binding fixes
  that seam without restoring shell-dependent ambient configuration.

- Observation: private package-index trust must exist inside the framework
  runtime image, independently of BuildKit daemon registry trust.
  Evidence: after the builder was correctly configured to resolve and trust
  `registry.lan`, the base image published successfully, but every kind build
  failed when `uv` fetched the stable Trackio wheel from `pypi.lan` with
  `UnknownIssuer`. The daemon's registry CA does not become part of a build
  stage's Debian certificate store.
  Resolution: runtime publication accepts an explicit machine-owned PEM trust
  bundle. BuildKit mounts it outside the context and appends it to—not replaces
  —the base image's system bundle, so both public and private indexes remain
  verifiable. Only its SHA-256—not the host path—participates in the build
  receipt identity. TLS verification stays enabled.

- Observation: supporting a second locked Python environment in the manifest
  parser was insufficient because the release renderer remained unaware of
  those fields.
  Evidence: the first successful 0.3.0 publication built the veRL backend from
  its exact release lock, then generated a manifest without
  `backend_constraint_lock` or `backend_lock_digest`. The generic consistency
  check passed because the fields were optional, but veRL environment packing
  would have failed.
  Resolution: backend constraint ownership is now a runtime-variant property;
  publication derives and renders it, selective reuse preserves it, and
  manifest validation requires the expected backend lock for veRL and no
  unexpected backend lock for other variants. A renderer round-trip regression
  test covers the complete fields.

- Observation: the retired project id survived in two places that can still
  write to the live tracking server, not only in inert fixtures.
  Evidence: the former Lab qualification launcher
  (`posttrain_lab.qualification.launcher`) built its `RunSpec` and collected
  remote evidence under `foundation-models`, and
  `posttrain_lab.qualification.evidence` defaulted its
  `--trackio-project` to the same id. Either would have recreated the project
  that this plan intends to purge. Both now name `posttrain-lab`. Several
  framework package tests still use `foundation-models` as an arbitrary
  offline fixture string; those are harmless but should become an obviously
  fictional id when their packages are next touched.

- Observation: no veRL job had ever been packed through the public path, on any
  provider. Both selected release gates would have failed identically.
  Evidence: the veRL job-kind image carries two locked Python environments — a
  control venv at `/opt/posttrain/venv` and a backend venv at
  `/opt/posttrain-verl` built from the veRL fork's own release lock — and
  `environment_packager.py` raises `veRL environment packaging requires exact
  backend kind constraints` unless both are supplied. But `PublishedImage`
  carried a single `constraint_lock`, so the release manifest could not publish
  the backend lock, and `_derived_constraint_profiles` therefore always left
  `backend_digest` unset. The only way to satisfy the check was a hand-authored
  `execution.toml` naming a `backend_path`, and neither the machine's existing
  `execution.toml` nor the legacy `scripts/qualification/` launchers ever did:
  `backend_path` appears in no launcher. The historical veRL runs therefore
  never exercised this path.
  Resolution: the backend constraints file was already shipped and generated
  (`containers/posttrain-job-kinds/verl-py313/release/backend-constraints.txt`,
  `uv export`ed from the veRL release lock) — it was simply never bound.
  `PublishedImage` now carries an optional `backend_constraint_lock`,
  `backend_lock_digest`, and `backend_provided_packages`, verified against the
  shipped bytes like the control lock; the manifest declares them for
  `online-rl-verl-py313`; and `_derived_constraint_profiles` binds them. A
  project with no machine configuration at all can now pack a veRL job.

- Observation: the AutomationBench environment was never missing a port. The
  native Verifiers v1 environment already exists, and the base catalog was
  simply pointing at the wrong repository. Nothing needed to be written or
  published to fix it.
  Evidence: `environments/automationbench_v1` is a native `verifiers.v1`
  taskset (`AutomationBenchTaskset(vf.Taskset)`) that reads only the fork's
  data layer — `automationbench.domains`, `.rubric.registry`, `.schema.world`.
  It has been tracked in this repository since tag `v0.2.5` and is on
  `origin/main`. The fork at `carbonteq-ai/AutomationBench` publishes
  `carbonteq-automation-bench` on the legacy 0.1 `StatefulToolEnv` API and has
  no v1 taskset at any revision or branch, confirmed with `gh`: it has one
  merged PR and two branches, neither carrying a port. So `package:
  automationbench-v1` could never have resolved from that repository, and the
  immutable wheel builder's name-mismatch rejection was correct. The framework
  bridge is deliberately a pure `verifiers.v1` consumer; environments present a
  v1 taskset to it, exactly as `gsm8k-v1` does from the Verifiers repository's
  `environments/gsm8k_v1` subdirectory.

- Observation: that one wrong pointer was authored in three places, and one of
  the three was already dead.
  Evidence: the repository/revision pair appeared in
  `packages/catalog/.../base/environments.yaml`, in
  `packages/eval/.../programs/automationbench.py`, and as a bare
  `AUTOMATIONBENCH_REVISION` constant in
  `apps/lab/.../environments/automationbench_grpo.py` that was exported
  through two `__all__` lists and never used to build a source. This is the
  dual-authoring asymmetry from the v0.2.5 critique showing up as a concrete
  correctness bug rather than as an ergonomics complaint.

- Observation: removing the environment from the framework's own workspace
  lock is what the selection contract always implied.
  Evidence: `apps/lab/pyproject.toml` carried `automationbench-v1` in its
  `gpu-posttrain` extra and a `../../environments/automationbench_v1` path
  source, which pulled both the adapter and `carbonteq-automation-bench` into
  the root `uv.lock`. Dropping them removed 30 lock lines, and the environment
  now enters a job only through selected environment packaging from an
  immutable source. The catalog's `dependency_lock_sha256` moved with it and
  was regenerated by `posttrain-release lock-dependencies` rather than
  hand-edited.

- Observation: deleting a Trackio run does not delete its artifact versions;
  that is intentional lineage preservation, not a complete cleanup operation.
  Evidence: the deployed server exposed only `delete_run`; historical cleanup
  removed eleven run rows but left the `foundation-models` project and its
  artifact storage eligible for retention.

- Observation: exact run purge needs a provider-owned second phase after the
  framework computes closure. Trackio's new slice blocks an unselected
  consumer, deletes only versions proven unlinked, and rescans retained
  manifests before unlinking CAS blobs. SQLite and Doris implement the same
  logical result, but provider transaction and restart qualification remains
  open.

- Observation: exposing the CLI before all adapters exist is only safe when the
  incomplete inventory is a hard blocker. The first CLI slice therefore makes
  `run purge` and `project purge` reviewable, but never presents an apply path
  for a plan whose tracking lineage or provider state is unknown. This keeps
  the command useful for diagnosing missing prerequisites without recreating
  the one-off cleanup risk.

- Observation: the first published lifecycle build exposed server-side stale
  digest detail as a generic HTTP 500, and the remote client discarded the
  response body.
  Resolution: post8 maps lifecycle `ValueError` failures to HTTP 400 and
  preserves the JSON error message in `RemoteClient`. Live SQLite and Doris
  checks now receive an actionable stale-digest explanation.

- Observation: public GitHub runners cannot resolve the internal
  `pypi.lan` source recorded for the qualified Trackio wheel. Installing a
  mirrored wheel first was insufficient because an unfrozen `uv sync` still
  refreshed the private registry.
  Resolution: publish hash-identical post8 artifacts on the immutable fork tag,
  verify the wheel before bootstrap, and install the rest of the committed lock
  with `--frozen --no-install-package carbonteq-trackio`. A clean-cache probe
  proved this path without contacting the internal index.

- Observation: the current Trackio project deletion implementation can only
  remove server-owned metadata and local artifact/media bytes. It cannot claim
  to delete a replica independently owned by an object store or a dataset
  integration.
  Evidence: the prototype reports the scoped project data set and documents
  the external-replica boundary; the project is the current storage and
  lineage boundary.

- Observation: `apps/lab/tests/fixtures/remote_gpu_project` has id
  `gpu-qualification`, but it is a Lab fixture, not evidence for a real
  `posttrain-integration` operational project.
  Evidence: it has no independent project manifest, durable provider jobs,
  retention policy, or deployed owner.

- Observation: an older Trackio Git index and current branch ref were found
  empty while finalizing the deletion work. The committed feature object was
  still reachable from the reflog; a dated worktree archive was made before
  repairing the index/ref, and `git fsck` was clean afterward.
  Evidence: `/tmp/trackio-working-tree-20260801-0601.tgz` has SHA-256
  `5d4f38771b99e306861cace5cf491657aa8874a97d76c3bb2e3858c44ad59417`.

- Observation: a fully passing local framework build was not sufficient release
  evidence after old provider, registry, and Trackio records were removed.
  Evidence: v0.3.0 therefore ran fresh packed provider qualifications before
  release. Those accepted records were later intentionally removed by the
  documented post-release cleanup and must not be presented as live evidence.

- Observation: the successful post-release cleanup required a coordinated
  operator procedure because v0.3.0 has no command that can preview the whole
  deletion set before touching any plane.
  Evidence: five Trackio projects, 99 terminal dstack submissions, 25 OCI
  manifests, local containers/images, and about 14 GB of state were removed by
  plane-specific operations. `posttrain run cleanup` could not represent that
  scope because it intentionally preserves evidence.

- Observation: storing a purge receipt below
  `.posttrain/state/executions/<run-id>` would destroy the audit record during
  the operation it records; project purge can remove the enclosing project
  state as well.
  Resolution: immutable purge plans, journals, and receipts live in the
  machine-scoped Posttrain state root, outside every project and run subtree.

- Observation: a plan id derived from the full semantic digest gives repeated
  previews a stable handoff without creating unreviewable plan clutter.
  Evidence: two plans with identical actions and different creation timestamps
  produce the same `purge-<digest-prefix>` id and digest; the focused purge
  tests verify that the persisted plan is reused.

## Decision Log

- Decision: use `posttrain-lab` as the only current Lab/Trackio project id;
  do not create `posttrain-integration` now.
  Rationale: a second empty project makes dashboard/project selection noisier
  and has no independent owner. Reserve the name until an integration app owns
  real long-lived jobs and data.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: keep provider resource release (`posttrain run cleanup`) separate
  from evidence/artifact erasure (`posttrain run purge`).
  Rationale: cleanup is normally safe after evidence is retained; purge is
  destructive and needs a different review, authorization, and receipt.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: every destructive command is plan-first. A non-mutating preview is
  the only behavior of `run purge` and `project purge`. Applying is a separate
  `posttrain purge apply PURGE_ID` operation. An interactive apply requires a
  confirmation; non-interactive apply additionally requires
  `--expect-digest sha256:... --yes`. There is no `--force` and no combined
  preview/apply shortcut.
  Rationale: separating selection from mutation makes the reviewed artifact a
  first-class handoff, prevents apply-time flag drift, and gives automation a
  stable compare-and-apply contract.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: a consumed output blocks a run purge by default. `--cascade`
  calculates the complete downstream consumer closure in the same project,
  shows it as a graph, and may run only when every selected run is terminal and
  reconciled. Provider cleanup is part of the purge plan rather than a
  prerequisite, so no provider record disappears before the operator has seen
  the cross-plane preview. Cross-project consumers, unknown tracking providers,
  live jobs, or missing lineage are hard blocks in the first release.
  Rationale: deleting a producer while leaving consumers creates false
  provenance; silently reaching into another project is more dangerous.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: a project purge is allowed only for an isolated Trackio project
  opened through the normal project layout and after server-authenticated and
  machine-state discovery. The standard command takes no arbitrary project id;
  `--project-root` selects the project as it does elsewhere in the CLI. A sharp
  `--tracking-project NAME --scope tracking-only` escape hatch exists only for
  orphaned legacy Trackio projects, is labeled incomplete cross-plane coverage,
  and never claims provider/OCI/local deletion. Project purge does not
  synthesize a cross-project cascade and does not promise deletion of
  independently owned object-store replicas.
  Rationale: the opened project is the reliable cross-plane ownership boundary;
  making a Trackio name look equivalent would recreate the ambiguity this
  feature is intended to remove.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: exact destructive selectors are mandatory. Run purge accepts one
  full canonical run id and never `--last` or a prefix. A plan receives an
  opaque, content-addressed `purge-<digest-prefix>` id and a full SHA-256 digest
  over canonical action identities; an unchanged repeated preview reuses that
  plan instead of creating clutter. Apply addresses the plan id, not the
  original run/project selector.
  Rationale: convenience selectors are appropriate for reads but unsafe when a
  newer run can appear between review and deletion.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: the purge journal is machine-owned and survives the target. Store
  `plan.json`, append-only `journal.jsonl`, and `receipt.json` below the same
  resolved machine state root that owns admission, under `purges/<purge-id>/`.
  Rationale: run-local and project-local receipts cannot prove deletion after
  their parent state has been removed, while a machine-scoped receipt can also
  resume a partial project purge.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: apply order is provider cleanup, exact unshared OCI manifest
  deletion, Trackio evidence/artifact deletion from leaf consumers toward the
  root producer, and local execution/workspace state last. Every action is
  revalidated immediately before mutation and journaled immediately after it.
  Rationale: this keeps durable evidence available while disposable execution
  and image resources are removed, and preserves the machine-scoped audit trail
  even after local control state is gone.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: land the plan store before any backend adapter or CLI mutation path.
  Rationale: the destructive boundary and recovery record must be testable with
  fakes before Trackio, OCI, dstack, Docker, and confirmation behavior are
  composed around it.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: keep exact run purge implementation split across the neutral
  framework and the Trackio fork. The framework owns selection, dependency
  closure, plan identity, and cross-plane ordering; Trackio owns its
  provider-scoped storage transaction and CAS retention. The fork slice is
  released as post8 from merged commit `77db6f5c` and is qualified against
  both SQLite and Doris.
  Rationale: importing Trackio into `posttrain.execution` would make the
  framework contract provider-specific and would tempt callers to mistake a
  provider preview for a complete purge plan.
  Date/Author: 2026-08-02 / framework maintainer.

- Decision: `carbonteq-trackio` must be released as `0.31.5.post6`, not
  overwritten as `post5`, before deploying the deletion API.
  Rationale: the internal index is non-volatile and `post5` is already tagged
  and deployed; one version must never name two source trees.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: the two release qualification runs are a SAMPO job over Zapier
  AutomationBench and a DAPO job over GSM8K, not the data-preparation and
  managed-evaluation jobs used for the packing proof.
  Rationale: the maintainer selected these two. They are also the stronger
  release signal. Data preparation and evaluation already have complete
  package-closure evidence from the cleanup plan's milestone 2, whereas the
  online-RL path is the only one that exercises a live rollout engine, weight
  synchronization, per-step advantages, and long multi-turn trajectory
  evidence in the same job. AutomationBench is the right home for SAMPO
  specifically, because SAMPO's hierarchical per-step advantage is only
  meaningful over genuinely long multi-turn tool use; the existing
  `multi-turn-alphabet-sort` binding under-tests it.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: the SAMPO/AutomationBench gate runs through the local provider and
  the DAPO/GSM8K gate runs through dstack; both execute on this machine's
  single RTX 4090.
  Rationale: both provider paths must be proven, and this machine is already a
  dstack fleet host, so provider difference is isolated from hardware
  difference. AutomationBench is the least-proven input in the release — its
  environment distribution does not yet build from an immutable published
  source — so it belongs on the provider where a failed pack costs the least.
  GSM8K DAPO has a settled immutable environment and is therefore the right
  workload to prove managed submission, admission, reconciliation, and cleanup.
  The assignment may be swapped only if the swap is recorded here with the
  evidence that motivated it.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: both gates stay `candidate`/`experimental` in the Lab registry
  until their managed runs produce complete retained evidence; promotion to an
  active release tier is an outcome of milestone 8, not a precondition for it.
  Rationale: their recorded replacement conditions already require live
  qualification before promotion. Promoting first would make the registry
  claim evidence that does not exist.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: release readiness is a bounded gate, while the complete DX program
  remains broader. A release may proceed after all release-blocking milestones
  and evidence pass; the remaining authoring/automation milestones remain open
  and must not be described as complete.
  Rationale: this preserves a truthful release decision without treating long
  term product work as an unbounded release blocker.
  Date/Author: 2026-08-01 / framework maintainer.

## Outcomes & Retrospective

Posttrain 0.3.0 shipped and was qualified with real packed provider work. Its
framework-only live history was then deliberately removed after acceptance;
the public release notes accurately warn that the retained release evidence is
no longer available in Trackio. Shared runtime supply, reusable machine cache,
and unrelated application/infrastructure projects were preserved.

The exact-run purge path now has the immutable plan/receipt substrate, concrete
Trackio/OCI/provider/local executors, published post8 dependency, rebuilt
runtime supply, and live interruption/resume qualification. `posttrain run
cleanup` remains correctly evidence-preserving; destructive deletion is a
separate digest-bound workflow. Full normal project purge remains fail-closed
until unmatched cross-plane inventory is connected, while the explicitly
scoped Trackio project deletion primitive is published and qualified. The wider
SAMPO/DAPO qualification and non-blocking authoring work remain open and are not
part of the already completed 0.3.0 release claim.

## Context and Orientation

The primary repository is `/home/hammad/projects/rl` on `main`. It is a `uv`
workspace. `apps/cli` renders the user-facing
`posttrain` command. `packages/execution` owns the durable provider submission
record, reconciliation, and non-destructive cleanup. `packages/tracking`
contains provider-neutral tracking contracts; `packages/tracking-trackio`
adapts Trackio. `apps/observatory` is the read-only inspection service.
`apps/lab` is the framework-maintainer qualification project and must not leak
into public user project dependencies.

An *execution submission* is the durable local record mapping a canonical
Posttrain run id to a provider job id, immutable image, workspace, evidence
source, and required artifact roles. A *reconciliation* compares that provider
record with tracking evidence and says whether it is consistent. An *artifact
consumer closure* is the set of later runs that read an artifact produced by a
selected run, recursively. It is directed producer to consumer. Purging the
closure therefore deletes its leaf consumers first and the root producer last.

The Trackio fork is `/home/hammad/projects/trackio`. Merged commit `77db6f5c`
contains digest-bound project/run preview and purge plus actionable lifecycle
errors and publishes `carbonteq-trackio==0.31.5.post8`.
`/home/hammad/projects/ai-infra` pins that
exact source revision and packages the wheel into the Ansible-managed shared
and candidate services. The live endpoint is `https://trackio.lan`; never
print its write token or credentials in plans, logs, receipts, or command
output.

The source plans remain authoritative for their own detailed contracts:

- `docs/plan/dx-repository-and-qualification-cleanup.md` for repository and
  Lab ownership.
- `docs/plan/dx-configuration-authority.md` for project-owned configuration.
- `docs/plan/dx-run-lifecycle-and-control.md` for controller/reconciliation.
- `docs/plan/dx-packing-environments-datasets.md` for package closure.
- `docs/plan/dx-public-api-and-authoring.md` for public authoring APIs.
- `docs/plan/dx-release-engineering.md` for release generation and receipts.

## Plan of Work

### Milestone 1: make the current Lab identity coherent

Finish the already-tested rename in `apps/lab/.posttrain/project.toml`, all
25 `apps/lab/.posttrain/work_packages/*.yaml`, `apps/lab/src/posttrain_lab/`,
and the related tests. Update `docs/plan/dx-repository-and-qualification-
cleanup.md` so historical `foundation-models` cleanup facts stay historical,
while present-tense contracts require `posttrain-lab`. Add a decision note
stating why `posttrain-integration` does not exist. Run the Lab tests, the
project display command, and the qualification listing before committing this
coherent rename by itself.

### Milestone 2: finish and deploy Trackio project-purge support

In `/home/hammad/projects/trackio`, complete the post6 release metadata,
regenerate `uv.lock`, build a wheel and source distribution, and run the
focused SQLite/server tests plus the full unit suite with the known
host-without-GPU checks converted to explicit skips rather than accepted
failures. The release must include:

- `SQLiteStorage.project_delete_summary()` and `delete_project()` covering the
  project database, sidecars, parquet sidecars, artifacts, and media.
- Equivalent `DorisStorage` operations removing the project rows from every
  project-scoped table and its local bytes.
- authenticated `/get_project_delete_plan` and `/delete_project` endpoints,
  and `RemoteClient.project_delete_plan()` / `delete_project()`.
- documentation that calls this a project boundary deletion and describes the
  external-replica limit.

Tag and publish post6 through `https://pypi.lan/carbonteq/dev/`; verify the
uploaded wheel digest and install it into a clean environment. Change
`ai-infra/scripts/package-trackio`, the Ansible wheel destination, and the
Trackio Dockerfile together to the immutable post6 source commit and filename.
Use a narrowly scoped Ansible control deployment, verify `/version` and the
new endpoint with an authenticated no-op project preview, then inspect the
live service image/package version. Do not delete any project in this
milestone.

### Milestone 3: add safe framework deletion planning

Create `packages/execution/src/posttrain/execution/purge.py`. It must define
immutable plan/receipt types rather than hiding decisions in a CLI callback:

    PurgeMode = Literal["run", "project"]
    PurgePlan(purge_id, mode, project_id, run_ids, root_run_id,
              dependency_edges, tracking_actions, provider_actions,
              registry_actions, local_actions, warnings, blockers,
              digest, created_at)
    PurgeReceipt(purge_id, plan_digest, completed_actions, skipped_actions,
                 failed_action, completed_at)

Add a machine-scoped `PurgeStore` rooted at the resolved admission state root's
`purges/` child. Preview writes mode-0600
`purges/<purge-id>/plan.json`; apply appends one fsync'd event per attempted
action to `journal.jsonl` and atomically writes `receipt.json` only after the
whole plan is complete. The plan digest covers canonical resource identities,
exact local target paths, dependency edges, action order, and preconditions,
including warnings and blockers, but excludes display labels, timestamps, and
secrets. The purge id is derived from a collision-checked digest prefix, so the
same discovery snapshot resolves to the same persisted plan.
Neither run state nor project state owns this store.

Add a small provider-neutral lifecycle protocol in `packages/tracking` for
listing a run's input/output artifacts, finding consumers, planning an exact
run deletion, and applying it. It must be optional: a tracking backend that
does not implement lifecycle deletion makes purge unavailable rather than
performing a partial deletion. Implement it in `packages/tracking-trackio`
using exact Trackio provider run ids and the new authenticated endpoints. Add
a server-side Trackio run-purge primitive if the current project-only API
cannot atomically delete the selected output versions and links. Do not model
this as `delete_run` plus client-side guesses.

This requires a maintained-fork change in `/home/hammad/projects/trackio`.
Implement authenticated run-purge preview/apply for SQLite and Doris, update
the fork's root `CARBONTEQ_FORK.md` and this repository's
`docs/tooling/trackio/README.md`, run both storage conformance suites, commit and
push the fork, publish a new immutable `carbonteq-trackio` version, and only then
update `packages/tracking-trackio/pyproject.toml` and `uv.lock`. Project delete
post6 is already deployed; do not rewrite or republish that version.

Add a provider-neutral `RegistryLifecycleAdmin` contract for checking and
deleting one digest-pinned OCI manifest. Put the concrete OCI Distribution
adapter with the other image/registry integration, not in the CLI callback. It
must parse only `repository@sha256:...` references, use the machine trust
bundle and Docker credential-helper/config resolution without writing secrets,
HEAD the exact digest during preview, and DELETE that same digest during apply.
Never accept tags or a repository prefix as a deletion target. Shared framework
base/kind images are structurally excluded; only actual-job images proven by a
submission/package receipt are eligible.

The planner reads every registered project control store plus the selected
`ExecutionSubmissionStore`, requires terminal reconciled records, verifies that
every selected run's recorded evidence source is Trackio and in the same
project, derives produced artifact versions, and recursively discovers
consumers. Existing `cleanup.json` is an already-completed provider action, not
a prerequisite. Otherwise the plan includes the provider's exact terminal
cleanup action so the preview precedes provider mutation. The default plan has
a blocker for every consumer. With `cascade=True`, it includes every downstream
consumer in the same project and validates the whole closure before creating
actions. Any missing control record, untracked producer, external consumer,
unavailable lifecycle service, live or unreconciled job, incomplete lineage,
tag-only image, or OCI digest shared by a retained submission remains a
blocker.

Project planning uses the opened project by default and inventories all of its
known submissions, tracking runs, provider handles, actual-job image digests,
workspaces, and control state. It renders unmatched resources explicitly. An
unmatched Trackio run is a hard blocker on normal cross-plane project purge
because provider and OCI absence cannot be inferred from missing local state.
The separately named `tracking-only` legacy scope may delete an isolated
Trackio project after its server preview, but its plan and receipt must say that
provider, OCI, and local coverage were not attempted.

Apply actions in a durable journaled order. Revalidate the complete dependency
graph and every exact identity first. Release terminal provider records and
workspaces, then delete exact unshared actual-job manifests, then delete
Trackio artifact versions and run evidence from leaf consumers toward the root
producer, and remove local execution/workspace state last. Journal each success
before continuing. A completed receipt is idempotent. Re-applying an incomplete
plan resumes only actions recorded in that immutable plan whose preconditions
still hold; it never widens the closure or silently substitutes a newer
resource.

### Milestone 4: expose a humane command surface

In `apps/cli/src/posttrain_cli/commands/run_cmd.py`,
`apps/cli/src/posttrain_cli/commands/project_cmd.py`, and a small presentation
module for the machine-scoped purge store, add:

    posttrain run purge RUN_ID
    posttrain run purge RUN_ID --cascade
    posttrain project purge
    posttrain purge show PURGE_ID
    posttrain purge apply PURGE_ID
    posttrain purge apply PURGE_ID --expect-digest sha256:... --yes

The run and project commands only discover, validate, persist, and render a
plan. They never mutate. The run selector must be one full canonical id; purge
does not support `--last` or prefixes. Preview renders a compact dependency
graph followed by a per-plane summary: provider records/workspaces, OCI
repository and digest, Trackio runs/artifact versions/logical and storage bytes,
and local paths/logical bytes. It prints warnings separately from hard blockers,
then the purge id, plan digest, plan path, and the exact `purge apply` command.
Blocked plans remain reviewable through `purge show`, but `purge apply` rejects
them before adapter contact.

`posttrain purge show` can be used for review or handoff without contacting
providers. `posttrain purge apply` loads the immutable plan, reconnects to each
adapter, and revalidates before mutation. On a terminal it prompts with the
purge id, digest prefix, run count, and destructive planes, then requires the
operator to type the complete purge id rather than answer a generic yes/no
question. With `--yes`,
`--expect-digest` is mandatory and must match the complete digest; without a
terminal and without both automation flags, apply fails closed. There is no
`--force`, no selector flags on apply, and no one-command preview/apply shortcut.
JSON mode returns stable plan, blocker, journal, partial-failure, and receipt
envelopes. The existing `posttrain run cleanup` command retains its meaning and
never deletes Trackio evidence.

The default human preview should be recognizable at a glance:

    Purge preview — no changes made
    Target: run 01... (project: disposable-purge-fixture)
    Closure: 3 runs, 2 artifact-consumer edges
    Provider: 3 terminal records/workspaces
    OCI: 3 unshared actual-job manifests
    Trackio: 3 runs, 4 artifact versions, 128 MiB logical
    Local: 3 execution records, 96 MiB logical
    Blockers: none
    Plan: purge-a1b2c3d4e5f60718
    Digest: sha256:a1b2...
    Next: posttrain purge apply purge-a1b2c3d4e5f60718

A blocked preview replaces `Next` with the exact remediation, such as rerunning
with `--cascade` or reconciling a named run. It never prints an apply command for
a plan that cannot be applied.

`posttrain project purge` opens the current `--project-root`, calls authenticated
Trackio and OCI previews, and scans registered project/provider control state.
It refuses an unknown project, an incomplete inventory, cross-project consumers,
or any live/unreconciled run. For orphaned legacy evidence, the explicit
`posttrain project purge --tracking-project NAME --scope tracking-only` form
creates a clearly limited plan after the Trackio server preview. It is not
described as cross-plane success. Do not hand-write HTTP, SQL, registry, or
filesystem deletes in the CLI.

### Milestone 5: complete release-critical configuration and controller work

Complete milestones 2–6 of `dx-configuration-authority.md` and milestones
2–6 of `dx-run-lifecycle-and-control.md` in dependency order. In particular:

- finish the automatically loaded machine config, scoped credentials, and
  project registration at `$XDG_CONFIG_HOME/posttrain`; prove `posttrain.env`
  is authoritative under `env -i`; shell environment
  variables never silently override project configuration;
- make the selected Trackio backend and tracking project a required part of
  admission for remote jobs;
- derive runtime variables from resolved selections rather than copied shell
  strings;
- preserve an immutable provider locator in every submitted run;
- ship the idempotent foreground controller first, then package its systemd
  unit and Ansible role so it reconciles cancellations/completions without a
  manual command; and
- make the controller treat a purge plan/receipt as a terminal durable state,
  never recreate or reconcile deleted evidence.

Tests must cover controller restart, repeated provider responses, a cancelled
job, a completed job, a missing Trackio run, and a completed purge receipt.
The production systemd service must run with a fixed project root/profile, a
minimal environment, structured logs, restart policy, and a safe read-only
health/status command.

### Milestone 6: complete release-critical pack closure and public service

Finish packing milestones 4–7 and public-API milestones 1–4 that affect a
normal project. The command path must use `Project.open()` rather than private
CLI composition for plan, package, provider selection, and purge. Complete
the intent/materialize/publish/launch split, selected transitive overlay
closure, standard `environments/` and `datasets/` discovery, and declared
dataset builders. A packed job must record its project source snapshot,
installed catalog family lock, environment/dataset wheel identities, tracking
endpoint without credentials, and artifact inputs/outputs.

Keep optional family discovery deterministic. A project that requires a
missing family must fail before catalog decoding; the pack identity records the
complete installed family set. Keep the explicit overlay exclusion/migration
story for legacy `layer.yaml`. Do not move all project definitions into Lab or
make public projects install Lab.

### Milestone 7: finish release automation prerequisites

Complete release-engineering milestones 3–4. `posttrain-release` must generate
dependency lock records from their selected maintained forks, stage a
release-neutral tree, build all distributions, and fail if the release manifest
does not match generated metadata. It must capture image build/push receipts
(immutable image digest, source commit, command, build time, package lock) and
assemble a curated release PR. The tag occurs only after the merged commit and
its receipts are re-verified; no image is rebuilt after merge.

For release readiness, also resolve any pinned Trackio requirement to post6,
build the framework release tree against that wheel, and prove a clean consumer
install from the internal index. Do not put credentials in lockfiles, staged
trees, job manifests, or evidence logs.

### Milestone 8: qualify purge with disposable cross-plane fixtures

Create a temporary project through the public `posttrain init` path and run
small, deterministic jobs rather than expensive training. The fixture must
produce three reconciled runs with producer → consumer → consumer artifact
lineage and distinct digest-pinned actual-job images. Exercise at least one
local provider record and one dstack record so both cleanup adapters appear in
the preview. Use a dedicated Trackio project and OCI repository namespace that
are unmistakably disposable; never reuse `posttrain-lab`, release evidence, or
an application project.

First preview the producer without cascade and verify the direct consumer is a
blocker. Preview with cascade and verify the complete three-run closure, exact
leaf-to-root Trackio order, provider actions, three eligible OCI digests, and
local state. Apply the plan interactively. For a second fixture, inject a
failure after one provider or registry action, then re-run the same purge id and
prove it resumes without repeating or widening completed work. In JSON mode,
prove missing and mismatched `--expect-digest` values fail before adapter
contact.

Create another isolated project and preview `posttrain project purge`. Verify
that an injected unmatched Trackio run blocks normal cross-plane apply. Remove
the mismatch through the fixture setup, generate a new plan, apply it, and
verify the Trackio project, provider records, job manifests, workspaces, and
project-local execution state are absent while the machine-scoped receipt
remains. Separately test the `tracking-only` legacy scope against a disposable
Trackio project and verify its receipt explicitly reports the unattempted
planes.

### Milestone 9: follow-up release decision and remaining DX completion

Run the complete source test suite, formatting, static typing, package import
boundaries, release-manifest checker, staged build, clean wheel consumer,
repository ownership checker, and the disposable purge evidence audit. Publish
the Trackio fork change before updating its exact framework pin. Build/push any
changed framework image once, record its digest, and after merge tag the
follow-up release without rebuilding or repushing. Release notes must contrast
evidence-preserving `run cleanup` with destructive plan/apply purge and link the
operator documentation.

Then finish the deliberately non-blocking pieces of public authoring:
registration-extension API, deterministic overlay explain/artifact pin,
schema/generator support, and the curated release PR interface. Update each
component plan's Progress and Outcomes sections, run its acceptance suite, and
only then mark this umbrella plan complete.

## Concrete Steps

All commands below are examples of the authoritative path; update exact
versions/digests after the associated plan changes them. Run them from the
named repository and retain redacted output under the release evidence root.

    cd /home/hammad/projects/rl
    uv run pytest apps/lab/tests -q
    uv run posttrain --project-root apps/lab project show
    uv run --package posttrain-lab posttrain-lab qualification list --project-root apps/lab
    uv run --package posttrain-release posttrain-release repository-check --report-only

Expected Lab output contains `Project: posttrain-lab`, 11 active gates, 14
candidates, and no unclassified work package.

    cd /home/hammad/projects/trackio
    git fsck --full --no-progress
    uv lock
    uv build
    uv run pytest tests/unit -q
    uv run ruff check trackio tests/unit

The environment may lack NVIDIA/Torch optional dependencies. Record those GPU
test skips/failures separately; do not relabel them as deletion failures. The
project-purge focused tests must pass, including the authenticated local-server
test for `RemoteClient.project_delete_plan()` and `delete_project()`.

    cd /home/hammad/projects/ai-infra
    ./scripts/package-trackio
    ./scripts/configure

Use a later narrow control-role playbook when it exists; until then, record the
full-site effect and health checks. Never echo Ansible vault values, Trackio
write tokens, or internal index passwords.

The historical `foundation-models` project was removed by the approved
post-release procedure before this command existed. Use disposable fixtures for
implementation acceptance; do not recreate or target released framework
history merely to demonstrate deletion.

    cd /home/hammad/projects/rl
    uv run posttrain --project-root /tmp/posttrain-purge-fixture project purge
    # expected: dry-run only; per-plane counts/bytes, purge id, digest, no mutation
    uv run posttrain purge show <purge-id>
    # expected: the same immutable actions without provider contact
    uv run posttrain purge apply <purge-id>
    # expected: terminal confirmation naming the digest and destructive planes

For non-interactive acceptance, extract the digest from JSON preview output and
bind apply to it explicitly:

    uv run posttrain --json --project-root /tmp/posttrain-purge-fixture project purge
    uv run posttrain --json purge apply <purge-id> \
      --expect-digest sha256:<complete-digest> --yes

The second command must reject a missing or mismatched digest before contacting
any adapter.

    uv run posttrain run purge <producer-run>
    # expected: blocker naming the direct consumer run and artifact version
    uv run posttrain run purge <producer-run> --cascade
    # expected: provider -> OCI -> leaf-to-root Trackio -> local action order, no mutation
    uv run posttrain purge apply <purge-id> \
      --expect-digest sha256:<complete-digest> --yes
    # expected: machine-scoped durable receipt and no remaining action

## Validation and Acceptance

The purge follow-up is ready only when all statements below have direct, dated
evidence:

- The checked-in Lab project id is `posttrain-lab`; its 25 work packages open,
  its test suite passes, and no maintained Lab surface retains the old id.
- The new Trackio fork release is immutable, published, deployed, healthy, and
  its authenticated run/project preview and purge APIs work against the live
  Doris-backed server without leaking credentials. The published artifact
  contains the same source commit/digest that Ansible deploys; post6 remains
  unchanged as historical project-delete support.
- The framework commands support no-op preview, immutable plan review, blocked
  consumer graph, explicit same-project cascade, digest-bound apply, resumable
  journaling, idempotent receipt, and isolated project purge. They reject
  cross-project, nonterminal, unreconciled, unknown-lineage, tag-only/shared OCI,
  incomplete-inventory, selector-prefix, and non-interactive unbound deletion
  attempts.
- A systemd/Ansible-managed controller uses project-owned config, restarts
  safely, and reconciles provider completion/cancellation without manual ledger
  editing.
- A clean external project can plan, pack, submit, observe, and clean a job
  without importing `posttrain_lab` or sourcing ambient shell configuration.
- The `automationbench-v1` environment wheel builds from an immutable pushed
  commit and subdirectory, the framework workspace lock contains neither it nor
  `carbonteq-automation-bench`, and the distribution-name check is unchanged.
- Disposable fixtures prove local and dstack provider actions, a three-run
  consumer closure, interactive and digest-bound apply, partial-failure resume,
  unmatched-resource blocking, complete project purge, and the explicitly
  limited tracking-only legacy scope. Their machine-scoped receipts remain
  readable after all target state is gone.
- Release commands prove generated metadata, dependency locks, wheels, image
  receipts, indexed dependencies, and the merged release source agree before
  tagging. The final evidence packet includes the commands, versions, digests,
  outcome summaries, and links/paths to redacted receipts.

The broader DX program is complete only after every unchecked milestone in the
six component plans has an acceptance result recorded in that plan and this
document's Progress/Outcomes sections are updated accordingly.

## Idempotence and Recovery

Every preview is safe to repeat and writes an immutable machine-scoped plan.
Apply never recomputes selection from run/project flags. It loads one purge id,
checks the expected digest when supplied, revalidates all identities and closure,
and appends a journal event after every action. Re-running apply reads the
journal/receipt, verifies the same resource identities, and reports
already-completed or already-absent actions rather than deleting a newer object.
A failed plan keeps the evidence necessary to resume; it never widens to a new
dependency graph.

Before any Trackio repair, project purge, registry deletion, or provider job
cleanup, obtain an authenticated preview. Never use bulk shell deletion, raw
SQL, a tag selector, or a wildcard registry command. Do not archive the data a
user explicitly asked to erase unless a separately declared retention policy
requires it; the purge plan itself retains identities and counts, not private
payloads, model bytes, prompts, metrics, or credentials.

If a purge fails, the machine-scoped journal names the last completed and first
failed action. Re-run `posttrain purge apply PURGE_ID`; completed actions are
verified and skipped, and the exact next action resumes. If an external artifact
replica exists, report it as an independently owned retention item rather than
claiming success. If revalidation detects a changed provider handle, OCI digest,
tracking object, local identity, or new consumer, stop and require a new preview;
never edit an immutable plan in place.

## Artifacts and Notes

The follow-up evidence directory should contain only redacted, immutable facts.
The v0.3.0 cleanup inventory remains historical input; new purge qualification
uses a separate path:

    release-evidence/cross-plane-purge/
      source.json                 # framework/fork commits, wheel hashes, deployed versions
      adapters.json               # Trackio, OCI, local, and dstack capability proof
      run-purge.json              # blocker, cascade, plan digest, and receipt proof
      resume.json                 # injected partial failure and same-plan recovery
      project-purge.json          # unmatched blocker and complete project deletion
      tracking-only.json          # explicit unattempted-plane receipt
      cleanup-v0.3.0.json         # historical one-off cleanup inventory and policy
      validation.txt              # exact test/build/check commands and outcomes

Do not store access URLs containing write tokens, provider credentials, raw
prompts containing private task data, or full environment files in this tree.

## Interfaces and Dependencies

The following final contracts must exist and be covered by unit and integration
tests:

    # packages/tracking/src/posttrain/tracking/lifecycle.py
    class TrackingLifecycleAdmin(Protocol):
        def plan_run_purge(self, *, project: str, provider_run_ids: tuple[str, ...]) -> TrackingPurgePlan: ...
        def apply_run_purge(self, plan: TrackingPurgePlan) -> TrackingPurgeReceipt: ...
        def project_delete_plan(self, *, project: str) -> ProjectDeletePlan: ...
        def delete_project(self, plan: ProjectDeletePlan) -> ProjectDeleteReceipt: ...

    # provider-neutral registry lifecycle contract, implemented beside image integration
    class RegistryLifecycleAdmin(Protocol):
        def inspect_manifest(self, reference: RuntimeImageRef) -> RegistryManifestPlan: ...
        def delete_manifest(self, plan: RegistryManifestPlan) -> RegistryManifestReceipt: ...

    # packages/execution/src/posttrain/execution/purge.py
    def plan_execution_purge(
        store: ExecutionSubmissionStore,
        source: RunDataSource,
        root_run_id: str,
        *,
        cascade: bool = False,
    ) -> PurgePlan: ...

    def plan_project_purge(
        layout: ProjectLayout,
        *,
        submission_stores: tuple[ExecutionSubmissionStore, ...],
        lifecycle: TrackingLifecycleAdmin,
        registry: RegistryLifecycleAdmin,
    ) -> PurgePlan: ...

    def apply_execution_purge(
        purge_id: str,
        *,
        purge_store: PurgeStore,
        submission_stores: tuple[ExecutionSubmissionStore, ...],
        lifecycle: TrackingLifecycleAdmin,
        registry: RegistryLifecycleAdmin,
        provider_factory: ExecutionProviderFactory,
        expect_digest: str | None = None,
    ) -> PurgeReceipt: ...

    # apps/cli/src/posttrain_cli/commands/run_cmd.py
    # The CLI is presentation only: it resolves the project service, renders
    # PurgePlan/PurgeReceipt, and never implements cascade traversal itself.

`TrackioLifecycleAdmin` must authenticate only through the configured
credential source, use canonical project and exact provider-run IDs, and reject
a server that lacks the required run-purge API. `PurgeStore` owns durable plans,
journals, and receipts below the machine state root's `purges/` child;
`ExecutionSubmissionStore` is an input and final deletion target, not the audit
owner. Trackio owns tracking metadata/artifact bytes; the execution provider
owns provider job/workspace cleanup; the registry adapter owns exact digest
manifest removal. The CLI owns presentation and confirmation only.

## Revision Notes

- 2026-08-01: Created to consolidate all incomplete work from the current DX,
  Lab, cleanup, deployment, and release discussion. It records the explicit
  decision not to create `posttrain-integration`, separates cleanup from purge,
  and makes fresh managed-job evidence a release gate.
- 2026-08-02: Reconciled the plan with the shipped v0.3.0 release and completed
  one-off framework-history cleanup. Revised purge around a separate
  preview/show/apply workflow, full cross-plane inventory before provider
  cleanup, exact digest binding for automation, machine-scoped resumable audit
  state, opened-project ownership, and an explicitly limited tracking-only
  legacy escape hatch. This replaces the earlier combined `--apply --yes`
  command and run-local receipt design because both could lose the reviewed
  scope or the receipt during the deletion they were meant to control.
