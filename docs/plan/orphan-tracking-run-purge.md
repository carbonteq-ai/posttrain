# Purge orphaned tracking runs and their abandoned admission entries

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

`posttrain run purge <run-id> --reason <slug>` builds a read-only, digest-bound purge plan, and `posttrain purge apply <purge-id>` executes it. Planning discovered candidates only from this machine's submission store (the per-project directory of submission receipts under `.posttrain/state/executions`). A run that exists in Trackio but was launched from a checkout or worktree that has since been removed has no receipt here, so every preview for it failed with the blocker `run '<id>' was not found`. Such runs could never be erased through Posttrain.

After this change an operator can run

    cd apps/lab
    uv run --package posttrain posttrain run purge <run-id> --reason abandoned-run --orphan

and get a plan that deletes that run (and the Trackio artifacts it exclusively owns) from Trackio. When the machine admission ledger still holds an abandoned entry for the run (its owning checkout is gone and the provider execution it names has ended), the same plan also settles that entry, which frees its placement, and deletes the job image the entry recorded when no other run owns it. The plan is produced only when four independent checks prove it is safe, the preview prints the evidence for each check, `purge apply` repeats the checks before deleting, and the retained tombstone records the orphan basis.

Terms used here. An **orphaned tracking run** is a run present in the project's tracking backend (Trackio) with no local submission receipt. The **machine admission ledger** is the cross-project, machine-scoped queue of admitted runs under `POSTTRAIN_ADMISSION_ROOT` (see `packages/execution/src/posttrain/execution/admission.py`); an entry is **settled** when its state is `completed` or `cancelled`. A **tombstone** is the minimal, secret-free audit record a purge leaves behind. An admission entry is **abandoned** when it is in `terminal_pending_evidence` (the provider finished but evidence reconciliation never ran), the owning control store it records no longer exists on disk or holds no submission receipt for the run, and the provider execution it names is terminal or absent when queried now. **Settling** it means moving it to `completed` and removing its own `active_by_key` placement reservation. A **basis** is a new, flat, digest-bound record inside a plan (and copied into its tombstone) that states why a special plan kind is allowed.

## Progress

- [x] (2026-09-23) Read the purge contract in `docs/post-training/03-work-and-evidence.md` ("Terminal cleanup, retention, and purge"), the planner, the store, the CLI surface, the Trackio adapter, and the dstack and local Docker adapters.
- [x] (2026-09-23) Baseline amendment: added the orphaned tracking-run paragraph to `docs/post-training/03-work-and-evidence.md` and the `--orphan` flag to the command list in `docs/post-training/05-apis.md`.
- [x] (2026-09-23) Added an optional digest-bound `basis` to `PurgePlan` and `PurgeTombstone` (`packages/execution/src/posttrain/execution/purge.py`).
- [x] (2026-09-23) Added `OrphanProviderInventory`, `OrphanTrackingRun`, `orphan_run_blockers`, and `build_orphan_run_purge_plan` (`packages/execution/src/posttrain/execution/purge_planner.py`).
- [x] (2026-09-23) Added read-only provider inventories: `DstackExecutionProvider.active_executions_for_run` with a new `active_runs` bridge action, and `LocalDockerExecutionProvider.active_executions_for_run` with a new `active_by_run` Docker action.
- [x] (2026-09-23) Added `TrackioRunActivityLookup` (`packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`).
- [x] (2026-09-23) Wired `--orphan` and `--stale-after-hours` into `posttrain run purge`, orphan rendering, and apply-time revalidation (`apps/cli/src/posttrain_cli/purge_surface.py`, `apps/cli/src/posttrain_cli/commands/run_cmd.py`).
- [x] (2026-09-23) Unit tests with fakes; ruff, pyright, lint-imports, and the affected package suites.
- [x] (2026-09-23) Real read-only previews for `lfm26-grpo20-random-20260912-r2` and `a61d411b-3f91-4681-94c6-3a5cfe394f67` against `https://trackio.carbonteq.com`, project `posttrain-lab`, and dstack project `main`.
- [x] (2026-09-23) Follow-up: abandoned admission entries. Added `ExecutionAdmissionService.settle_orphaned` and `AdmissionEntry.admission_key` (`packages/execution/src/posttrain/execution/admission.py`), `AdmissionSettlePurgeExecutor` (`packages/execution/src/posttrain/execution/admission_purge.py`), `OrphanAdmissionEntry` and the settle/registry branches in `purge_planner.py`, and `_orphan_admission`, `_entry_provider`, and `_LocalPlaneExecutor` in `purge_surface.py`. Amended 03 and 05 again.
- [x] (2026-09-23) Fixed `candidate_catalog` passing the `purges` directory, not its state root, to `_completed_purge_run_ids`, with a test that fails before the fix.
- [x] (2026-09-23) Real read-only previews for the six runs whose entries were abandoned in the ledger; five are unblocked, one is blocked by a shared image.
- [x] (2026-09-23) Second follow-up: a shared job image no longer blocks an orphan plan. It is kept with the same warning a normal purge gives, and the Trackio deletion and the admission settle still proceed. Re-previewed the eval run as unblocked.
- [ ] Apply is deliberately not run by this plan. The operator decides which previewed plans to apply.

## Surprises & Discoveries

- Observation: Every posttrain dstack submission is tagged `posttrain_run_id`, and every local Docker container is labeled `posttrain.run_id`. That makes "no active provider execution references the run id" a real query, not an inference from derived names.
  Evidence: `packages/execution-dstack/src/posttrain_execution_dstack/adapter.py` `_configuration` sets `tags.posttrain_run_id`; a read-only `client.runs.list()` returned `pt-11a37426e8a0ab77701e7db1 running {'posttrain_run_id': 'lfm26-vortex-v3-lr2e4-20260923-r1', ...}`.

- Observation: dstack `runs.list()` without arguments returns at most one page (100) of active runs, and when none are active it returns the single latest run, which may be terminal. The bridge therefore filters terminal statuses itself and reports `complete: false` on a full page, which the adapter turns into an error (and the plan into a blocker).
  Evidence: `dstack.api._public.runs.RunCollection.list` source in the ai-infra dstack environment.

- Observation (first pass, since resolved by the follow-up): `lfm26-grpo20-random-20260912-r2` is not fully orphaned. The machine admission ledger still holds it in `terminal_pending_evidence` with control store `file:///tmp/rl-adaptive-v04/apps/lab/.posttrain/state` (a removed worktree) and dstack run `pt-48b3564ae80357d610c47259`. That entry also makes the registry inventory attribute job image `registry.lan/carbonteq/posttrain-lab/posttrain-job@sha256:8f1dfa40…` to the run. Both orphan checks (a) and (b) therefore block, which was the intended fail-closed outcome at the time: deleting the Trackio run would have left that ledger entry permanently unable to reconcile its evidence. No CLI could release such an entry, because `acknowledge_reconciled` is reached only through `run reconcile`, which needs the gone project's submission store. Six runs were in this state, which motivated the follow-up.
  Evidence: preview output in Artifacts and Notes.

- Observation: Trackio history rows carry a flush `timestamp` that can be later than the event's own `event/occurred_at`; last activity takes the maximum of both, plus host-metric (system history) timestamps and the run's creation and start times.

- Observation: `candidate_catalog` in `apps/cli/src/posttrain_cli/purge_surface.py` called `_completed_purge_run_ids(purge_store.root)`, which appends `purges` a second time, so runs retired by a completed purge were never found there and a purged cancelled admission could reappear as a candidate. Fixed to pass `purge_store.root.parent`; `test_candidate_catalog_skips_runs_retired_by_a_completed_purge` fails before the fix and passes after.

- Observation: all six abandoned entries use dstack and name executions that dstack now reports as `terminated` or `failed`. Their control stores (`/tmp/rl-adaptive-v04`, `/home/hammad/projects/rl-local-async`, `/tmp/rl-evaluation-selection`) no longer exist. For dstack the admission key is `run:<run-id>`, so settling cannot free a placement another run shares.

- Observation: `eval-lfm26-automationbench-heldout20x3-olmo3-random20-lfmrec-20260914-r2` shares its job image (`...posttrain-job@sha256:a9af1c12...`) with six other eval runs. Under the first follow-up's rule ("any other owner blocks") its orphan purge was blocked. The second follow-up aligned orphan purge with normal purge, which keeps a shared image and warns, and that unblocked the run.

- Observation (unrelated, pre-existing): `apps/cli/tests/test_cli.py::test_grpo_plan_is_static_and_selects_online_rl_runtime` fails in the current dirty tree because of the inference-configuration validation work (`enforce_eager` / vLLM continuation messages), not because of purge.

## Decision Log

- Decision: Allow purging orphaned tracking runs from the tracking plane only, as a narrow amendment to the frozen baseline.
  Rationale: runs launched from removed checkouts otherwise cannot be erased at all; restricting the plan to tracking deletion avoids guessing at provider, registry, or local resources the machine cannot attribute.
  Date/Author: 2026-09-23, user approved the amendment; implemented by Claude.

- Decision: Orphan purge is an explicit `--orphan` flag and never a fallback. Without the flag the existing `run '<id>' was not found` blocker is unchanged. With the flag, a run that still has local control state (a submission receipt or a cancelled admission candidate) produces a blocked plan telling the operator to omit `--orphan`, rather than silently switching to the normal path.
  Rationale: the operator asked for a tracking-only erasure; widening it to provider, registry, and local planes without a new request would be surprising. Blocking is the simplest safe choice.
  Date/Author: 2026-09-23, Claude.

- Decision: Provider check (a) queries the machine admission ledger plus live provider inventories. If the run's Trackio configuration records `source_metadata.execution.provider`, only that provider is queried and it must answer; otherwise every configured provider (dstack, local Docker) is queried. Any unsettled ledger entry for the run, any active dstack run tagged with the run id, or any non-exited container labeled with it blocks. An unqueryable inventory blocks.
  Rationale: the tracking configuration does not record the exact dstack run name; it could be derived from the CLI's idempotency-key scheme, but that scheme differs for qualification launches, so a derived name that is "absent" proves nothing. The tag/label query covers every posttrain submission regardless of key scheme.
  Date/Author: 2026-09-23, Claude.

- Decision: Registry check (b) reuses `_registry_image_inventory` (machine admission ledger, registered project stores, terminal admission archive) and blocks both when the run owns an image there and when that inventory reports any gap. An image recorded only in the run's own Trackio configuration is reported as a warning and left untouched.
  Rationale: orphan purge never deletes registry manifests; the check exists to prove there is no machine-side owner record the purge would strand.
  Date/Author: 2026-09-23, Claude.

- Decision: Lineage check (c) uses the Trackio fork's digest-bound run purge plan; any consumer other than the run itself blocks, as do server blockers or a failed request.
  Date/Author: 2026-09-23, Claude.

- Decision: State check (d) accepts terminal statuses (`succeeded`, `failed`, `cancelled`, `lost`), or `running` with last activity older than the threshold (default 24 h, `--stale-after-hours`, minimum 1). Last activity is the newest of creation, start, the last 1000 history rows' timestamps and event times, and all system-metric timestamps. Traces are not consulted.
  Rationale: matches the approved rule ("no metric or event newer than a threshold") while keeping the query bounded.
  Date/Author: 2026-09-23, Claude.

- Decision: Record the orphan basis as an optional `basis` mapping on `PurgePlan` (digest-covered only when present) and copy it into `PurgeTombstone`. Values are restricted to scalars and lists of short strings.
  Rationale: existing plan digests and tombstones are unchanged because the key is omitted when absent; the tombstone stays free of prompts, metrics, configs, tokens, and URLs with credentials.
  Date/Author: 2026-09-23, Claude.

- Decision: Settle an abandoned admission entry to the existing `completed` state (with `settlement: "orphan-purge"` and a one-line `message` stored on the entry) instead of adding a new terminal state.
  Rationale: every ledger reader, the prune/archive path, and `_validate_payload` already treat `completed` as settled; a new state would need changes in all of them. The marker keeps the settlement distinguishable and makes `settle_orphaned` idempotent. Unlike `acknowledge_reconciled`, `settle_orphaned` never pumps the next waiting run, because a purge must not submit work.
  Date/Author: 2026-09-23, Claude (coordinator request).

- Decision: Settle only `terminal_pending_evidence` entries. Other unsettled states (`waiting`, `submitting`, `submission_failed`, `submitted`) still block, and so does an entry with no inspectable control-store locator.
  Date/Author: 2026-09-23, Claude.

- Decision: Delete a registry image in an orphan plan only when the ownership inventory lists the orphan as its sole owner and the image is the one the run's admission entry records. An attributed image from any other owner record still blocks, because that record may mean another project controls the run.
  Rationale: `_registry_image_inventory` merges several sources (ledger, registered project stores, terminal archive) without saying which one attributed an image. Requiring the admission entry's own image keeps the deletion tied to proven-abandoned state. Apply re-runs `_revalidate_registry_ownership` and the registry adapter's digest-bound delete plan, as it does for normal purges.
  Date/Author: 2026-09-23, Claude.

- Decision (second follow-up, supersedes "any other owner blocks"): when the image recorded by the run's own admission entry is also owned by other runs, keep it and warn with the normal-purge wording (`job image '<ref>' retained; referenced by unselected run(s): ...`). The Trackio deletion and the admission settle proceed. An image attributed by any owner record other than the run's admission entry still blocks, whether or not it is shared, because that record may mean another project still controls the run. The basis records kept images under `registry_retained_shared`.
  Rationale: this matches normal purge semantics for shared images, as the coordinator requested; keeping an image is never destructive.
  Date/Author: 2026-09-23, Claude (coordinator request).

- Decision: Do not add a provider cleanup action for orphans. Record the named execution's current state (for example `dstack:pt-48b35... (cancelled, terminated)`) in the basis instead.
  Rationale: the existing `provider.cleanup` action goes through `JobExecutionService`, which needs the submission receipt (the handle, run workspace, and runtime image). For dstack, cleanup launches a worker-side task for an exact workspace path. Rebuilding those inputs without the receipt would be guesswork.
  Date/Author: 2026-09-23, Claude.

- Decision: Apply order is registry, then tracking, then the local settle action, which depends on the tracking deletion. Once the tracking action is journaled complete, CLI-level orphan revalidation is skipped (the run can no longer be read from Trackio); the settle executor still re-checks that the entry has the same state, placement key, and provider id.
  Date/Author: 2026-09-23, Claude.

## Outcomes & Retrospective

The flag, checks, rendering, revalidation, and tombstone basis are implemented and covered by unit tests. The real preview of the stale running orphan `a61d411b-3f91-4681-94c6-3a5cfe394f67` is unblocked (`purge-2fbf21f0466e70c2`). With the follow-up, five of the six runs whose admission entries were abandoned have unblocked previews that delete the Trackio run and the exclusive job image and settle the entry. After the second follow-up, the eval run whose image is shared is also unblocked; its image is kept for the six other runs that use it. Nothing was applied. Remaining: the operator decides which plans to apply.

## Context and Orientation

The purge contract lives in `docs/post-training/03-work-and-evidence.md` under "Terminal cleanup, retention, and purge". Plan, journal, receipt, and tombstone types are in `packages/execution/src/posttrain/execution/purge.py`; `PurgePlan.build` computes a SHA-256 digest over the plan's semantic payload and derives the purge id from it. The provider-neutral planner is `packages/execution/src/posttrain/execution/purge_planner.py`. The CLI composition root is `apps/cli/src/posttrain_cli/purge_surface.py`: `candidate_catalog` builds the local inventory, `_registry_image_inventory` builds machine-wide image ownership, `save_run_preview` saves a plan, `render_plan` prints it, and `apply_saved_plan` revalidates and calls `apply_purge_plan`. The command is registered in `apps/cli/src/posttrain_cli/commands/run_cmd.py` (`run purge`). Trackio access is in `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`; `TrackioLifecycleAdmin.plan_run_purge` returns the fork's lineage-aware purge plan and `TrackioPurgeActionExecutor` applies it. Provider adapters are `packages/execution-dstack/src/posttrain_execution_dstack/adapter.py` (with `sdk_bridge.py`, a standalone script run in dstack's own Python) and `packages/execution-local/src/posttrain_execution_local/adapter.py`. The machine configuration is `~/.config/posttrain/config.toml`; on this workstation it binds dstack project `main` through `/home/hammad/projects/ai-infra/.venv/bin/python` and `/home/hammad/projects/ai-infra/.state/dstack/client.env`, and Trackio at `https://trackio.carbonteq.com`. The Trackio project is the Posttrain project id, `posttrain-lab` for `apps/lab`.

## Plan of Work

First amend the baseline (03 and 05). Then extend `purge.py` with `basis` and `_basis` validation. Add the orphan evidence types and planner functions to `purge_planner.py` and export them from `packages/execution/src/posttrain/execution/__init__.py`. Add `active_runs` to the dstack bridge and `active_executions_for_run`/`inventory_scope` to both provider adapters. Add `TrackioRunActivityLookup` and `TrackioRunActivity` to the Trackio adapter and export them. In `purge_surface.py`, add `_orphan_preview`, `_orphan_local_control_check`, `_collect_orphan`, `_orphan_provider_inventory`, `_inventory_provider`, `_revalidate_orphan`, and `_orphan_lines`; call `_revalidate_orphan` in `apply_saved_plan` after the registry revalidation. Finally add the CLI options. For the follow-up, add `settle_orphaned` to `ExecutionAdmissionService` and expose `admission_key` on `AdmissionEntry`. Add `AdmissionSettlePurgeExecutor` in a new `packages/execution/src/posttrain/execution/admission_purge.py`. In the planner, add `OrphanAdmissionEntry` and move all four checks into `_assess_orphan`, which also decides the registry deletions and whether the entry is settled. In the surface, have `_orphan_provider_inventory` return the run's ledger entry described by `_orphan_admission` (control-store status, plus the named execution's state queried through `_entry_provider`). Route local-plane actions through `_LocalPlaneExecutor` so `local.settle_admission` reaches the settle executor.

## Concrete Steps

From `/home/hammad/projects/rl`:

    uv run ruff check <changed files>
    uv run pyright <changed files>
    uv run lint-imports
    uv run pytest -q packages/execution/tests packages/tracking-trackio/tests packages/execution-dstack/tests packages/execution-local/tests apps/cli/tests

From `/home/hammad/projects/rl/apps/lab` (read-only previews):

    uv run --package posttrain posttrain run purge lfm26-grpo20-random-20260912-r2 --reason abandoned-run --orphan
    uv run --package posttrain posttrain run purge a61d411b-3f91-4681-94c6-3a5cfe394f67 --reason abandoned-run --orphan

## Validation and Acceptance

`uv run lint-imports` reports 9 contracts kept. The affected suites (`packages/execution/tests packages/tracking-trackio/tests packages/execution-dstack/tests packages/execution-local/tests apps/cli/tests`) report 452 passed and 1 failed; the failure is the pre-existing `test_grpo_plan_is_static_and_selects_online_rl_runtime` described above. New tests: `packages/execution/tests/test_orphan_purge_planner.py` (28), 18 orphan and path-fix tests appended to `apps/cli/tests/test_purge_surface.py`, 2 settle tests in `packages/execution/tests/test_admission.py`, 2 each in the dstack and local Docker adapter tests, and 1 in the Trackio adapter tests. The follow-up tests cover: an abandoned entry settled together with its exclusive image; an entry whose owning project still holds the receipt, blocked; an entry whose named provider execution is still running, blocked; an image shared with another run, kept with a warning while the entry is still settled; a shared image not recorded by the admission entry, blocked; apply order with every plane outcome in the tombstone; `settle_orphaned` releasing only its own placement without admitting a waiting run; and the retired-run path fix. They cover a terminal orphan producing only the tracking action, a stale running orphan allowed, a recently active running orphan blocked, an active provider execution blocked, an unsettled admission entry blocked, an attributed registry image blocked, a surviving consumer blocked, a run with a local receipt blocked under `--orphan`, the unchanged `was not found` blocker without the flag, and apply-time revalidation refusing to delete after a provider execution appears.

## Idempotence and Recovery

Previews are read-only apart from saving the immutable plan under the machine purge store; re-running a preview with unchanged evidence reuses the same purge id. A changed last-activity time or inventory produces a new purge id. Apply revalidates every orphan check and refuses if any fails or if the Trackio run identity changed; once the tracking action is journaled as completed, revalidation is skipped so a resumed apply can finish writing its receipt and tombstone.

## Artifacts and Notes

Follow-up previews (2026-09-23, from `apps/lab`, read-only):

    lfm26-grpo20-random-20260912-r2: Blockers: none
      Next: posttrain purge apply purge-b5055e7035e5f418 --expect-digest sha256:b5055e7035e5f418a8275deb438beeb97db953b29caf79f3e074e97277e3f148 --yes
    lfm26-grpo20-post8-vllm-shared-20260910-r1: Blockers: none
      Next: posttrain purge apply purge-c243079921967d89 --expect-digest sha256:c243079921967d89161bd965267f70c95dcdf159b24e8d5335effc661508de72 --yes
    lfm26-gdpo20-deepseek-v41-local-20260911-r1: Blockers: none
      Next: posttrain purge apply purge-03c7f4acb01daebc --expect-digest sha256:03c7f4acb01daebcb749c4576b61a3f985c6e4d2386b9b6f642f90a376f7d440 --yes
    eval-lfm26-automationbench-heldout20x3-olmo3-random20-lfmrec-20260914-r2: Blockers: shares registry image ...@sha256:a9af1c12... with 6 other eval runs
      Next: resolve blockers and create a new preview   (first follow-up; superseded below)
    eval-lfm26-automationbench-heldout20x3-olmo3-random20-lfmrec-20260914-r2 (second follow-up, --reason failed-attempt): Blockers: none
      Warnings: ...; job image '...@sha256:a9af1c12...' retained; referenced by unselected run(s): 6 eval runs
      Next: posttrain purge apply purge-b905c96110acc2a2 --expect-digest sha256:b905c96110acc2a2a2a02eda6a236c1c8536a01d73682753b71fc5f612235512 --yes
    lfm26-grpo20-post7-20260909-r1: Blockers: none (running, stale since 2026-09-09T20:19:02Z)
      Next: posttrain purge apply purge-4ac01e1471c3baee --expect-digest sha256:4ac01e1471c3baee6282019352459dfb02c12d6ed1a592ec110c2e88d50b6f66 --yes
    lfm26-olmo3-30-local-20260911-r1: Blockers: none (running, stale since 2026-09-11T00:48:06Z)
      Next: posttrain purge apply purge-4104f454f00e4683 --expect-digest sha256:4104f454f00e4683fe30931684780a3ccd47d1d3b9ab3f91288f8e606c47c1f1 --yes

First-pass previews:

    === lfm26-grpo20-random-20260912-r2
      (a) provider: recorded provider dstack; inventories checked: machine admission ledger, dstack project 'main'; active executions: machine admission entry terminal_pending_evidence (control store file:///tmp/rl-adaptive-v04/apps/lab/.posttrain/state)
      (b) registry: attributed images: registry.lan/carbonteq/posttrain-lab/posttrain-job@sha256:8f1dfa40...; inventory complete
      (c) lineage: Trackio run 5d3f9814f874451ebbef70feaf4eea73; tracked artifacts: 2; surviving consumers: none
      (d) state: tracking status failed; last activity 2026-09-11T22:37:12.372507+00:00; stale threshold 24 h
    Blockers: ... still attributable to provider execution state ...; ... is attributed registry image ...
    Plan: purge-4376f27eaec2637f

    === a61d411b-3f91-4681-94c6-3a5cfe394f67
      (a) provider: recorded provider none; inventories checked: machine admission ledger, dstack project 'main', local Docker; active executions: none
      (b) registry: attributed images: none; inventory complete
      (c) lineage: Trackio run 4edffa88d9634d8aa11134b9caf6bd6f; tracked artifacts: 0; surviving consumers: none
      (d) state: tracking status running; last activity 2026-09-17T23:27:24.703240+00:00; stale threshold 24 h
    Blockers: none
    Plan: purge-2fbf21f0466e70c2
    Digest: sha256:2fbf21f0466e70c23721b38814b89845313fc0313423752b0c07afd7c1269039

## Interfaces and Dependencies

In `posttrain.execution` (neutral; no provider imports):

    OrphanProviderInventory(checked: tuple[str, ...], active: tuple[str, ...], blockers: tuple[str, ...])
    OrphanTrackingRun(run_id, project_id, evidence_provider, evidence_project, tracking_provider_run_id,
                      tracking_status, last_activity_at, provider, recorded_provider=None, recorded_job_image=None,
                      consumers=(), lineage_complete=True, lineage_blockers=(), tracked_artifacts=0,
                      evidence_retention="standard")
    orphan_run_blockers(orphan, *, registry_image_owners, registry_inventory_blockers, now, stale_after) -> tuple[str, ...]
    build_orphan_run_purge_plan(orphan, *, reason, registry_image_owners, registry_inventory_blockers, now, stale_after) -> PurgePlan
    PurgePlan.basis / PurgeTombstone.basis: Mapping[str, JsonValue] | None
    OrphanAdmissionEntry(state, admission_key, provider, provider_id, control_store, control_store_status,
                         provider_state, provider_native_state=None, provider_query_error=None, job_image=None)
    OrphanTrackingRun.admission: OrphanAdmissionEntry | None
    ExecutionAdmissionService.settle_orphaned(run_id, *, admission_key, provider_id, note) -> bool
    AdmissionEntry.admission_key: str | None
    AdmissionSettlePurgeExecutor(admission)   # handles kind SETTLE_ADMISSION_KIND == "local.settle_admission"

In adapters (loaded by the CLI with `importlib`, so no new import-linter edges):

    posttrain_execution_dstack.DstackExecutionProvider.active_executions_for_run(run_id) -> tuple[str, ...]
    posttrain_execution_local.LocalDockerExecutionProvider.active_executions_for_run(run_id) -> tuple[str, ...]
    posttrain_tracking_trackio.TrackioRunActivityLookup(project, *, server_url).find(run_id) -> TrackioRunActivity | None

Revision note (2026-09-23): initial plan written together with the implementation and real previews.

Revision note (2026-09-23, follow-up): extended orphan plans to settle abandoned admission entries and delete their exclusively owned job image, recorded provider state as evidence instead of cleaning it, fixed the retired-run path bug, and added the six real previews. Requested by the coordinator after the first pass left six runs blocked by abandoned ledger entries.

Revision note (2026-09-23, second follow-up): a shared job image is now kept with a warning instead of blocking, matching normal purge; an image attributed by any source other than the run's admission entry still blocks. Requested by the coordinator.
