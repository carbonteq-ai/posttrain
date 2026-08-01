# Complete the 0.3.0 developer-experience program and prove a release

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

This is the closure plan for the DX work that remained open after the
repository, configuration, lifecycle, packing, public-API, and release plans
were started. It is deliberately an umbrella plan: it preserves those six
plans as their detailed design records while imposing one dependency order,
one definition of done, and one evidence packet for the 0.3.0 release.

## Purpose / Big Picture

A framework maintainer must be able to prepare a project, package an isolated
job, submit it, observe it, reconcile it automatically, and deliberately
remove the job and its outputs when the work is no longer wanted. The same
workflow must work for the framework's Lab project without making Lab a hidden
requirement for normal users. A release is ready only after that workflow has
passed on real provider jobs and the published components match the exact
source and image receipts that were tested.

After this plan, the framework root is a virtual workspace, `apps/lab` owns
the `posttrain-lab` qualification project, and there is no prematurely-created
`posttrain-integration` project. The name is reserved until an independently
owned integration application, not a test fixture, has its own jobs, data
retention policy, and operator. `posttrain project purge` can preview and
remove a wholly-owned Trackio project. `posttrain run purge` can plan deletion
of one terminal job, tells the user when its produced artifacts are consumed,
and only follows the complete downstream closure when the user explicitly
requests and confirms a cascade.

The maintainer can demonstrate the completed release by running two fresh,
immutable 0.3.0 packages on the managed provider: one data preparation job and
one managed evaluation job. They can inspect Trackio and Observatory during
and after execution, see the reconciler settle the durable state without a
manual ledger repair, then clean up the exact test resources with previewed
receipts. This is evidence of a real deployed system, not a collection of
green unit tests.

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
- [ ] Stabilize the recovered Trackio repository, finish the `post6` release,
      deploy it through Ansible, and obtain a live deletion preview for the
      retired `foundation-models` project before deleting anything further.
- [ ] Implement framework-owned run/project purge planning and execution with
      artifact-consumer closure, receipts, and a Trackio lifecycle adapter.
- [ ] Complete the remaining release-critical configuration, lifecycle,
      packing, and public-service milestones listed below.
- [ ] Run fresh provider packaging/evaluation qualification, verify Trackio,
      Observatory, controller reconciliation, and cleanup receipts, then run
      the 0.3.0 publication and release audit.
- [ ] Complete the non-blocking authoring and release-automation follow-up
      milestones before declaring the entire DX program, rather than merely
      the release, complete.

## Surprises & Discoveries

- Observation: the retired project id survived in two places that can still
  write to the live tracking server, not only in inert fixtures.
  Evidence: `scripts/qualification/run_algorithm_scenario.py` built its
  `RunSpec` and collected remote evidence under `foundation-models`, and
  `scripts/qualification/validate_algorithm_run.py` defaulted its
  `--trackio-project` to the same id. Either would have recreated the project
  that this plan intends to purge. Both now name `posttrain-lab`. Several
  framework package tests still use `foundation-models` as an arbitrary
  offline fixture string; those are harmless but should become an obviously
  fictional id when their packages are next touched.

- Observation: deleting a Trackio run does not delete its artifact versions;
  that is intentional lineage preservation, not a complete cleanup operation.
  Evidence: the deployed server exposed only `delete_run`; historical cleanup
  removed eleven run rows but left the `foundation-models` project and its
  artifact storage eligible for retention.

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

- Observation: a fully passing local framework build is not release evidence
  after old provider, registry, and Trackio records have been removed.
  Evidence: the prior terminal jobs/manifests were intentionally cleaned, so
  the two required managed jobs must be rerun from the release-staged tree.

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
  the default; mutation requires `--apply`, and an interactive terminal also
  requires an explicit confirmation. Automation uses `--apply --yes` only
  after it persists the reviewed plan digest.
  Rationale: a textual prompt alone does not work in automation and a bare
  `--force` conceals what will be lost.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: a consumed output blocks a run purge by default. `--cascade`
  calculates the complete downstream consumer closure in the same project,
  shows it as a graph, and may run only when every selected run is terminal,
  reconciled, and has completed provider cleanup. Cross-project consumers,
  unknown tracking providers, live jobs, or missing lineage are hard blocks in
  0.3.0.
  Rationale: deleting a producer while leaving consumers creates false
  provenance; silently reaching into another project is more dangerous.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: a project purge is allowed only for an isolated Trackio project
  after a server-authenticated preview. It does not synthesize a cross-project
  cascade and it does not promise deletion of remote object-store replicas.
  Rationale: project-level ownership is the only reliable storage boundary
  available today.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: `carbonteq-trackio` must be released as `0.31.5.post6`, not
  overwritten as `post5`, before deploying the deletion API.
  Rationale: the internal index is non-volatile and `post5` is already tagged
  and deployed; one version must never name two source trees.
  Date/Author: 2026-08-01 / framework maintainer.

- Decision: release readiness is a bounded gate, while the complete DX program
  remains broader. A release may proceed after all release-blocking milestones
  and evidence pass; the remaining authoring/automation milestones remain open
  and must not be described as complete.
  Rationale: this preserves a truthful release decision without treating long
  term product work as an unbounded release blocker.
  Date/Author: 2026-08-01 / framework maintainer.

## Outcomes & Retrospective

No release outcome exists yet. At the start of this plan the framework has
strong local proofs and a partially completed 0.3.0 tree, but no fresh managed
provider evidence, no deployed Trackio project-purge API, no framework cascade
purge, and no final release receipt. Update this section after each release
gate and again after the remaining DX program is closed.

## Context and Orientation

The primary repository is `/home/hammad/projects/rl-0.3.0` on branch
`codex/dx-0.3.0`. It is a `uv` workspace. `apps/cli` renders the user-facing
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

The Trackio fork is `/home/hammad/projects/trackio`. Commit `92b7760` contains
the project preview/purge feature; its version metadata is being advanced from
`0.31.5.post5` to `0.31.5.post6`. `/home/hammad/projects/ai-infra` packages a
specific Trackio wheel into the Ansible control role, which currently pins
`post5`; that pin must change only after the post6 wheel, commit, and digest
are final. The live endpoint is `https://trackio.lan`; never print its write
token or credentials in plans, logs, receipts, or command output.

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
    PurgePlan(run_ids, root_run_id, tracking_actions, provider_actions,
              registry_actions, local_actions, blockers, digest, created_at)
    PurgeReceipt(plan_digest, completed_actions, skipped_actions, completed_at)

Add a small provider-neutral lifecycle protocol in `packages/tracking` for
listing a run's input/output artifacts, finding consumers, planning an exact
run deletion, and applying it. It must be optional: a tracking backend that
does not implement lifecycle deletion makes purge unavailable rather than
performing a partial deletion. Implement it in `packages/tracking-trackio`
using exact Trackio provider run ids and the new authenticated endpoints. Add
a server-side Trackio run-purge primitive if the current project-only API
cannot atomically delete the selected output versions and links. Do not model
this as `delete_run` plus client-side guesses.

The planner reads `ExecutionSubmissionStore`, requires terminal reconciled
records and existing `cleanup.json` receipts, verifies that every selected
run's evidence source is Trackio and in the same `project_id`, derives all
produced artifact versions, and recursively discovers consumers. The default
plan has a blocker for every consumer. With `cascade=True`, it includes every
downstream consumer in the same project and validates the whole closure before
creating actions. Any missing run, untracked producer, external consumer,
unavailable Trackio lifecycle service, uncleaned provider job, or shared OCI
image/manifests with another retained submission remains a blocker.

Apply actions in a durable journaled order: provider workspace is already
cleaned; delete leaf consumer artifacts/run evidence first; continue toward the
root producer; delete exact unused registry manifests and local run state only
after Trackio proves the corresponding evidence is gone. A completed receipt
is idempotent. An incomplete receipt resumes only the actions whose precondition
still holds and never widens the closure.

### Milestone 4: expose a humane command surface

In `apps/cli/src/posttrain_cli/commands/run_cmd.py`, add:

    posttrain run purge RUN_ID
    posttrain run purge RUN_ID --cascade
    posttrain run purge RUN_ID --cascade --apply --yes
    posttrain project purge --tracking-project posttrain-lab

The first two commands are dry-run plans. They render a compact graph with
canonical run ids, display names, artifact name/version, resource type, and
the exact reason for every blocker. `--apply` persists the plan first and
requires a terminal confirmation unless `--yes` is supplied; `--yes` is
rejected unless `--apply` is present. JSON mode returns a stable error envelope
and includes the plan digest, action count, blockers, and receipt path. The
existing `posttrain run cleanup` command retains its meaning and never deletes
Trackio evidence.

Add `posttrain project purge` as an explicit project-level command. It calls
the authenticated Trackio preview first, renders run/artifact/version/byte
counts, refuses an unknown or non-isolated project, and only calls delete after
the same confirmation protocol. It does not purge the framework's local state
until the Trackio receipt is durable. This command is the route used for the
retired `foundation-models` project; do not hand-write HTTP or SQL deletes.

### Milestone 5: complete release-critical configuration and controller work

Complete milestones 2–6 of `dx-configuration-authority.md` and milestones
2–6 of `dx-run-lifecycle-and-control.md` in dependency order. In particular:

- finish named site profiles at `$XDG_CONFIG_HOME/posttrain/config.toml` and
  prove `posttrain.env` is authoritative under `env -i`; shell environment
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

### Milestone 8: execute and inspect two fresh managed qualifications

From the release-staged 0.3.0 tree, create isolated Lab package plans for the
data-preparation and managed-GSM8K-evaluation gates. Before submission, record
the immutable framework/Trackio commit, staged wheel SHA-256 values, OCI image
digests, selected provider site profile, and Trackio project `posttrain-lab`.
Submit both jobs through the normal CLI, not ad hoc provider commands.

Use the controller/daemon and `posttrain run show` to observe both executions.
Trackio must contain canonical configurations, lifecycle metrics, artifacts,
and traces where applicable; Observatory must read those persisted records.
Capture provider state, Trackio run id, image digest, logs, reconciliation
result, artifact set, completion status, elapsed time, and cleanup receipt in
the release evidence directory. A job that fails, is cancelled, or does not
publish the required artifacts is a failed release gate, not a reason to reuse
old evidence.

After success, use dry-run `posttrain run purge` on one job and demonstrate
its no-consumer path. Create a tiny dedicated lineage fixture with producer →
consumer → consumer, demonstrate the default blocker and explicit cascade
preview/apply, then destroy only that fixture. Finally preview and, after the
user confirms the rendered byte/count set, delete the now-retired
`foundation-models` Trackio project through `posttrain project purge`. Verify
that `posttrain-lab` still exists and has the two retained release runs.

### Milestone 9: release decision and remaining DX completion

Run the complete source test suite, formatting, static typing, package import
boundaries, release-manifest checker, staged build, clean wheel consumer,
repository ownership checker, Lab list, and the two managed-job evidence audit.
Create the 0.3.0 release PR only when every release blocker is proven. After
merge, re-verify source commit, published wheels, OCI digests, deployed
Trackio/Observatory versions, and live project views before tagging and
publishing release notes.

Then finish the deliberately non-blocking pieces of public authoring:
registration-extension API, deterministic overlay explain/artifact pin,
schema/generator support, and the curated release PR interface. Update each
component plan's Progress and Outcomes sections, run its acceptance suite, and
only then mark this umbrella plan complete.

## Concrete Steps

All commands below are examples of the authoritative path; update exact
versions/digests after the associated plan changes them. Run them from the
named repository and retain redacted output under the release evidence root.

    cd /home/hammad/projects/rl-0.3.0
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

    cd /home/hammad/projects/rl-0.3.0
    uv run posttrain project purge --tracking-project foundation-models
    # expected: dry-run only, counts/bytes and a plan digest, no mutation
    uv run posttrain project purge --tracking-project foundation-models --apply
    # expected: explicit confirmation prompt before the authenticated delete

    uv run posttrain run purge <producer-run>
    # expected: blocker naming the direct consumer run and artifact version
    uv run posttrain run purge <producer-run> --cascade
    # expected: leaf-to-root action order and no mutation
    uv run posttrain run purge <producer-run> --cascade --apply --yes
    # expected: durable receipt and no remaining Trackio/registry/local action

## Validation and Acceptance

The release is ready only when all release-blocking statements below have
direct, dated evidence:

- The checked-in Lab project id is `posttrain-lab`; its 25 work packages open,
  its test suite passes, and no maintained Lab surface retains the old id.
- Trackio post6 is immutable, published, deployed, healthy, and its authenticated
  preview/delete API works against the live Doris-backed server without leaking
  credentials. The published artifact contains the same source commit/digest
  that Ansible deploys.
- The framework commands support no-op preview, blocked consumer graph,
  explicit same-project cascade, idempotent receipt, and isolated project purge.
  They reject cross-project, nonterminal, unreconciled, unknown-lineage, and
  uncleaned-provider deletion attempts.
- A systemd/Ansible-managed controller uses project-owned config, restarts
  safely, and reconciles provider completion/cancellation without manual ledger
  editing.
- A clean external project can plan, pack, submit, observe, and clean a job
  without importing `posttrain_lab` or sourcing ambient shell configuration.
- Two fresh, release-staged Lab provider jobs succeed with exact images,
  Trackio evidence, Observatory visibility, reconciliation, and retained
  artifacts. Their records are distinct from the lineage cleanup fixture.
- Release commands prove generated metadata, dependency locks, wheels, image
  receipts, indexed dependencies, and the merged release source agree before
  tagging. The final evidence packet includes the commands, versions, digests,
  outcome summaries, and links/paths to redacted receipts.

The broader DX program is complete only after every unchecked milestone in the
six component plans has an acceptance result recorded in that plan and this
document's Progress/Outcomes sections are updated accordingly.

## Idempotence and Recovery

Every preview is safe to repeat. Every apply operation writes a plan with a
digest before changing a remote resource, and writes a receipt after each
action. Re-running an applied plan reads the receipt, verifies the same
resource identities, and reports already-absent actions rather than deleting a
newer object. A failed plan keeps the evidence necessary to resume; it never
widens to a new dependency graph.

Before any Trackio repair, project purge, registry deletion, or provider job
cleanup, archive exact inputs and obtain an authenticated preview. Never use
bulk shell deletion, raw SQL, or a wildcard registry command. The current
Trackio working-tree backup is retained until post6 has been published,
deployed, and verified; once the release evidence includes the new immutable
commit/digest, move the backup to the normal recovery retention location or
remove it through the approved cleanup path.

If a project purge fails after Trackio metadata deletion but before local
Posttrain state cleanup, retain the Trackio receipt and rerun the same plan;
the local portion can safely complete. If an external artifact replica exists,
report it as an independently owned retention item rather than claiming
success. If an action detects changed source/image/digest or a new consumer,
stop and require a new preview.

## Artifacts and Notes

The final release evidence directory should contain only redacted, immutable
facts. Its minimal index is:

    release-evidence/0.3.0/
      source.json                 # merged commit, manifest version, wheel hashes
      trackio-post6.json          # fork commit, wheel hash, deployed health/version
      images.json                 # provider image refs and immutable digests
      qualifications.json         # two job ids, outcomes, reconciliations, artifact refs
      observatory.json            # persisted Trackio/Observatory inspection result
      controller.json             # service version, restart and reconciliation proof
      purge-fixture.json          # dry-run blocker/cascade/receipt proof
      cleanup.json                # retired foundation-models preview and final receipt
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

    # packages/execution/src/posttrain/execution/purge.py
    def plan_execution_purge(
        store: ExecutionSubmissionStore,
        source: RunDataSource,
        root_run_id: str,
        *,
        cascade: bool = False,
    ) -> PurgePlan: ...

    def apply_execution_purge(
        plan: PurgePlan,
        *,
        store: ExecutionSubmissionStore,
        lifecycle: TrackingLifecycleAdmin,
    ) -> PurgeReceipt: ...

    # apps/cli/src/posttrain_cli/commands/run_cmd.py
    # The CLI is presentation only: it resolves the project service, renders
    # PurgePlan/PurgeReceipt, and never implements cascade traversal itself.

`TrackioLifecycleAdmin` must authenticate only through the configured
credential source, use canonical project and exact provider-run IDs, and reject
a server that lacks the required post6 API. `ExecutionSubmissionStore` owns the
durable plans/receipts below `.posttrain/state/executions/<run-id>/`; Trackio
owns tracking metadata/artifact bytes; the execution provider owns provider
job/workspace cleanup; the registry adapter owns exact image manifest removal.

## Revision Notes

- 2026-08-01: Created to consolidate all incomplete work from the current DX,
  Lab, cleanup, deployment, and release discussion. It records the explicit
  decision not to create `posttrain-integration`, separates cleanup from purge,
  and makes fresh managed-job evidence a release gate.
