# Query prompt-group rewards from indexed trace facts

This ExecPlan is a living document. Maintain it according to `docs/templates/PLAN.md` as the implementation and qualification progress.

## Purpose / Big Picture

On Observatory's Rollouts & rewards page, a group's reward mean, standard deviation, and previous observation must not change when the viewer loads another 100 traces. After this change the browser requests bounded group aggregates from tracking storage, and Observatory compares exact prompt groups across optimizer steps. No read requires streaming every native trace into Observatory or the browser. The owner approved local Doris testing followed by a qualified service rollout and historical fact backfill; neither should change training state or native traces.

## Progress

- [x] (2026-09-23) Identified the loaded-page calculation in `apps/observatory/frontend/src/components/RolloutGroupTable.tsx` and rejected a full-run summary scan as the final design.
- [x] (2026-09-23) Confirmed the pinned Trackio aggregate supports `rollout_step` and `task_type` but lacks exact prompt-group/task identities and second reward moment.
- [x] (2026-09-23) Implemented generic indexed fact dimensions and a second-moment aggregate in `/home/hammad/projects/trackio`; 436 unit tests passed and a new end-to-end test passed against an isolated database on the real Doris host.
- [x] (2026-09-23) Projected exact identities in Posttrain and added the provider-neutral query contract; 121 focused Python tests and 87 frontend tests passed.
- [x] (2026-09-23) Added Observatory run-wide group query and UI, with explicit partial/unavailable states and no loaded-page reward fallback; regenerated API schema and built frontend.
- [x] (2026-09-23) Published Trackio dev25 from immutable commit `bb40b7e333b7f74f4cf6923e3ff6030255ed746d` to `carbonteq/dev`, pinned its hashes in the framework lock, restore-verified retained candidate and production Doris snapshots, migrated both v2 databases to v3, built both indexes, and deployed candidate then production. The public `/version` reports dev25.
- [x] (2026-09-23) Reprojected all 1,219 native traces available at the final checkpoint into indexed v5 facts using idempotent 25-trace pages. The local Observatory table shows step-19 exact group mean/std and a prior observation from older steps while loading only the newest 100 summaries; the protected run remains Running.
- [ ] Reconcile any v4 tail emitted by the still-running old producer after this checkpoint, especially once the run ends. Do not modify the trainer or native traces.

## Surprises & Discoveries

The existing `trace-filters` endpoint calls `trace_summary_population`, which pages through every retained trace. Reusing it would correct pagination semantics but still perform an O(traces) provider read per cache refresh. Trackio already materializes scalar facts and aggregates them in SQLite and Doris, but its grouping dimensions omit exact `posttrain_prompt_group_id` and `example_id`; its scalar operations omit a second moment. Apache Doris schema changes are explicit operator migrations, not automatic startup repairs.

The existing `trackio_candidate` Doris database is schema v2 and contains 6,418 traces, so it is not a disposable test fixture. The new schema and query were instead exercised against `trackio_group_facts_qual_20260923`, an isolated v3 database on the same Doris host. The prior candidate and production backup receipts are stale, and the existing backup-qualification script deletes its test snapshot after restore; neither is a retained pre-migration backup. The live trainer process uses the old producer code, so after a one-time backfill it will continue emitting older fact versions until it ends; a safe tail backfill/checkpoint loop or post-run reconciliation is required.

Fresh retained snapshots replaced the stale receipts before either migration. The first 200-trace backfill page timed out after 162 idempotent writes; the retry used 25-trace checkpointed pages and reached a null next cursor at 1,219 inspected traces. Candidate deployment briefly routed the public URL to the candidate container; the playbook now preserves the production upstream, and read-only queries found no live-run records in the candidate database. Four groups remain incomplete in the indexed population, so the run-wide coverage state is correctly partial rather than a fabricated complete result.

## Decision Log

- Decision: Use Trackio's generic materialized trace-fact query rather than a Posttrain-specific SQL route or an Observatory full-run scan. Rationale: Trackio owns storage/query indexes, while Observatory owns the task-aware previous-step interpretation. Date/Author: 2026-09-23, Codex.
- Decision: A previous observation means the nearest strictly older optimizer step with complete groups for the same exact task identity. If that step contains multiple complete prompt groups, combine their reward moments instead of using arbitrary group-ID ordering. Rationale: this gives stable temporal semantics. Date/Author: 2026-09-23, Codex.
- Decision: Deploy and backfill only after local Doris qualification, a fresh retained backup, immutable fork release, and explicit migration/index verification. Rationale: the owner authorized these operational writes, but the live trainer must remain undisturbed and a v2 Doris service cannot read a v3 schema until cutover. Date/Author: 2026-09-23, Codex.

## Outcomes & Retrospective

The indexed query is deployed and qualified against production Doris. Trackio dev25 is published at `carbonteq-v0.31.5.post14.dev25` and served publicly; the local project-backed Observatory on `127.0.0.1:7871` displays a latest-step group with mean 0.625 and population std dev 0.125, plus the nearest older task observation from step 3 (0.750 / 0.000), while only 100 of 1,219 summaries are loaded. The active run remains Running. Historical backfill is complete only through its final 1,219-trace checkpoint; the old live producer can add v4 facts afterward, so a tail reconciliation remains operational work. No stable-index promotion has been claimed.

## Context and Orientation

`/home/hammad/projects/trackio` is the maintained generic Trackio fork, currently pinned by `packages/tracking-trackio/pyproject.toml` and `uv.lock` in this repository. `trackio/trace_facts.py` validates source facts and queries. `trackio/sqlite_storage.py` and `trackio/doris_storage.py` implement storage aggregates; `trackio/doris_schema.py` owns explicit Doris migrations. In this repository, `packages/environment/src/posttrain/environment/verifiers_evidence.py` projects native Verifiers trace facts, `packages/tracking/src/posttrain/tracking/models.py` defines provider-neutral query types, `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py` implements them, and `apps/observatory` owns the HTTP/UI view. The native trace is replay authority; indexed facts are rebuildable projections, not a second source of truth.

## Plan of Work

First add bounded `prompt_group_id` and `task_id` scalar dimensions, plus a `sum_squares` operation on finite scalar measures, to the fork's fact validation and both storage backends. Materialize indexed group/task/step columns so `GROUP BY` and exact group filters never parse native payload at query time. Extend the Doris schema version with an explicit v2-to-v3 migration; do not run it automatically. Add storage tests proving correct count, reward coverage, mean, and sum of squares for mixed/missing rewards and multiple groups.

Next have the Verifiers fact projector use the recorded `info.posttrain_prompt_group_id` and `info.example_id`, plus optimizer step, without inferring identities from prompt text. Add the same bounded dimensions and operation to the provider-neutral query model. Ensure Trackio adapter forwards it and a fake backend can implement the same logical buckets. Keep the package dependency boundary (`train`, `eval`, and `serve` independent).

Finally, Observatory queries fact buckets for the run. A group is complete only when its trace count equals configured `num_generations` and reward coverage equals that count. Compute population standard deviation as the square root of the nonnegative second moment minus squared mean; tolerate small floating-point cancellation, but reject materially negative variance. Construct a task/step comparison from complete groups only. Return only aggregate rows and provenance/coverage to the UI. Loading another trace page must not change the reward columns. Unavailable provider facts must show unavailable, never silently fall back to loaded summaries.

## Concrete Steps

In `/home/hammad/projects/trackio`, inspect `git status`, current branch/commit, and the fork ledger before edits. Run focused trace-fact tests with the fork's test environment, then the relevant full Trackio tests. Update `CARBONTEQ_FORK.md`. Do not modify the framework pin while the fork is dirty or unpublished.

In `/home/hammad/projects/rl`, run `uv run pytest apps/observatory/tests packages/tracking-trackio/tests packages/environment/tests`, `uv run ruff check .`, `uv run pyright`, `uv run lint-imports`, frontend `npm test` and `npm run build`, and `git diff --check`. The API schema must be regenerated through `posttrain-observatory schema --openapi apps/observatory/openapi.json`; frontend types are regenerated by its build. Record any unrelated baseline failures separately.

## Validation and Acceptance

A fake run with newest four rollouts on the first page and the same task's prior group only on older pages must return the prior mean/std from the aggregate endpoint before any browser “Load more” click. A group missing one rollout or one reward must show unavailable. Two groups for one task in one step must form one prior-step comparison population. Trackio SQLite tests must show an indexed aggregate over the run's facts, and the query path must not call native `traces()` listing. After the exact fork release, Doris migration, service deployment, and approved historical projection backfill, the live Observatory page must show the same result after loading more rows, with provenance naming the source step and coverage.

## Idempotence and Recovery

Schema migrations and fact backfill require an operator-reviewed backup and explicit release sequencing. Fact writes are trace-keyed and receipt-backed, so an interrupted historical page can be retried by external trace ID without changing the native trace. Never infer zero from missing coverage. A failed migration leaves the old server/pin in place; the UI remains on its explicitly loaded-only behavior until the aggregate API is available and qualified.

## Artifacts and Notes

The initial rejected experiment added a full-run summary endpoint via `trace_summary_population`; it was removed before this plan. Local Observatory is now started through `posttrain observatory up` from `apps/lab`, with source `trackio-posttrain-lab` and the new indexed group endpoint. The retained backup receipts are `.state/artifacts/trackio-group-facts/candidate-backup-receipt.json` and `production-backup-receipt.json` in `/home/hammad/projects/ai-infra`; backfill command output is `/tmp/posttrain-group-facts-backfill-20260923.log`.

## Interfaces and Dependencies

Trackio's generic aggregate request will accept `group_by=("prompt_group_id", "task_id", "rollout_step")` and scalar aggregate operations for count, mean, and sum of squares of `task_reward`. The provider-neutral `TraceFactsQuery` mirrors that shape. Observatory owns a separate typed `PromptGroupRewardView` over those buckets. The Trackio fork must be committed, pushed, released, and installed on the Trackio server before Posttrain pins that exact distribution and claims this path works against the deployed provider.

Revision note (2026-09-23): updated after explicit authorization for Doris testing, deployment, and backfill; release and backup gates remain mandatory.
