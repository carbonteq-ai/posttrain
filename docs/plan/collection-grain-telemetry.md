# Report online training per collection, not per optimizer update

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

An online reinforcement-learning run (GRPO, SAMPO and their relatives) alternates two phases: it samples a population of episodes from the current policy, then trains on that population. Until the resolved policy-update engine, one population fed exactly one optimizer update, so "update" and "population" were the same row everywhere. The resolved engine applies one population as several updates (the 2.6B SAMPO run `manifest-steps-26-sampo-100-g16x8-20261006-r10` applies each 128-episode population as 4 updates of 32 episodes). Tracking and the Observatory still assume one population per update: population values such as mean reward are written only at the first update of each population and are blank at the other three, averages "per update" are really averages per population, all 128 episodes are attributed to the first update, and derived figures such as rows per update and rollout share divide a population's work by single updates.

After this change a population is a first-class record called a **collection**. A user can ask the semantic layer for one row per collection (`select step, updates, reward, rollouts_completed from collections`), see each update's collection (`update.collection_step`), and read Observatory charts whose population values are per collection while optimizer values stay per update. Older runs, where each update had its own population, read as one collection per update, so comparisons across old and new runs remain valid.

## Progress

- [x] (2026-10-06 01:20Z) Baseline amendment: `docs/post-training/06-observation-and-lineage.md`, "Envelope" section, defines the collection grain, its identity (`collection_step`, the step of its first update), `train/rl/collection_updates`, and the `collection_step`/`collection_update` attributes on applied updates.
- [x] (2026-10-06 01:35Z) Trainer: the TRL resolved path records `train/rl/collection_updates` and attribute `collection_step` with each collection's metrics (`packages/train/src/posttrain/train/backends/trl/policy_rollouts.py`, `_observe_collection`), and attributes `collection_step` and `collection_update` with each applied update (`packages/train/src/posttrain/train/backends/trl/policy_job.py`, `observe_applied_update`). Tests updated; 1090 passed.
- [x] (2026-10-06 02:40Z) Semantic layer: `collection` entity and `collections` table (`step`, `time`, `updates`, `collection_seconds` and 59 collection measures, including `planned_updates` from `train/rl/collection_updates`); 49 catalog entries moved from `update` to `collection`; `update.collection_step`; `rollout.step` described as the collection step; `rollout_share` and `rows_per_collection` per collection. Note templates `group-policy@4` and `sampo@4` report collections and updates separately. Fixture project `fixture-resolved` (tagged and untagged runs, two collections of two updates); 193 Observatory tests pass. On Doris: r10 reads as collections 1, 5, 9, ... with 4 updates each; the legacy cont100 run reads as 100 collections of 1 update with mean reward 0.3947 (as before).
- [x] (2026-10-06 02:40Z) Algorithm-specific settings (user request): the run snapshot (`packages/work/src/posttrain/work/runner.py`, `_selection_details`) now records `policy_updates` (schedule, execution, objective variant, selections, denominator) when declared and SAMPO's credit and correction settings (discount, step weight, normalization, importance-sampling mode and caps, truncation penalty, masking, admission attempts). Run dimensions `run.prompts_per_collection` (was `prompts_per_update`), `run.update_unit`, `run.update_budget`, `run.update_epochs`, `run.objective_variant`, `run.importance_sampling_mode`, `run.importance_sampling_clip_max`, `run.truncation_penalty`, `run.discount_gamma`, `run.step_advantage_weight`, `run.advantage_normalization`.
- [x] (2026-10-06 03:10Z) Observatory service and frontend: `BackendRuntimeSummary.rollouts_per_update` (veRL `global_batch_size` only) becomes `rollouts_per_collection` (prompt groups × rollouts per prompt, else the global batch); the rollout-setup panel shows prompt groups and rollouts per collection and the `policy_updates` schedule ("32 episodes per update"); active-sampling and SAMPO-credit wording says collection. Collection values were already plotted at their collection's first update, so chart positions are unchanged. OpenAPI snapshot regenerated (only the `collection` entity value is new). 193 Python and 120 frontend tests pass; `npm run build` succeeds.
- [ ] Validation against r10's live data and the full validation ladder.

## Surprises & Discoveries

- Observation: every collection-level value of the resolved TRL path is already written at the collection's first update step, and every native trace of a collection already carries that step as `optimizer_step` (the rollout step). What is missing is only (a) a record of which collection each later update trained on and (b) readers that treat the population as its own row.
  Evidence: `policy_rollouts.py` `_observe_collection` (step `applied + 1`), `observe_trace` attributes; semantic `rollouts.step` is fact `rollout_step` from `optimizer_step`.

## Decision Log

- Decision: identify a collection by the step of its first update (`collection_step`), not by a new ordinal.
  Rationale: it is already the step collection metrics and traces use, it survives checkpoint resume without new saved state (the population's applied-update offset plus one), and for legacy runs it equals the update step, so no migration is needed. An ordinal (1, 2, 3, ...) is derived by readers by ranking collection steps.
  Date/Author: 2026-10-06, Claude.

- Decision: collection-grain catalog measures move from the `update` entity to a new `collection` entity instead of appearing on both.
  Rationale: semantic names are unique across the model, so duplicating would rename one side; and the baseline now forbids per-update views from repeating or dividing collection values. `select reward from updates` therefore stops working; `select reward from collections` works for old and new runs alike. Repository callers are updated in the same change.
  Date/Author: 2026-10-06, Claude.

- Decision: readers derive the collection of an untagged update as the latest collection start at or before it, where a collection start is a step with a collection-grain metric point; an explicit `collection_step` attribute wins when present.
  Rationale: runs recorded before the trainer tags (r10 included) and legacy runs (every update is a start) are read correctly without rewriting stored data.
  Date/Author: 2026-10-06, Claude.

- Decision: a collection start is a step with a marker metric (`train/rl/collection_updates`, `train/rl/reward_mean`, `train/rl/rollouts_attempted`) or with a point of any collection metric the statement reads.
  Rationale: fixed markers keep the collection set stable for update mapping; adding the read metrics keeps legacy population rows whatever columns a statement reads, as the `updates` view already does for its rows. Scanning all 59 collection metrics for every statement would cost Doris a JSON extraction per metric per row.
  Date/Author: 2026-10-06, Claude.

- Decision: record algorithm settings additively in the run snapshot (new keys only, `policy_updates` only when declared) rather than replacing the hand-picked subsets with a full dump.
  Rationale: existing readers keep their keys, existing bindings keep their snapshot digests, and the values that decide how collections become updates and how SAMPO builds credit become queryable.
  Date/Author: 2026-10-06, Claude.

## Outcomes & Retrospective

None yet.

## Context and Orientation

The trainer records metrics through `RunContext.metrics(values, step=..., attributes=...)`; the Trackio adapter (`packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`, `metrics`) stores one row per call in Trackio's `metric_rows` table with the attributes under the key `metric/attributes`. Traces of rollouts go to Trackio's `traces` table with fact columns such as `fact_rollout_step`, `fact_task_id` and `fact_task_reward`.

The Observatory (`apps/observatory`) is the read-only evidence product. Its semantic layer (`apps/observatory/src/posttrain_observatory/semantic_layer/`) declares entities (`framework.py`, `ENTITIES`), dimensions and measures (`framework.py` and `metric_catalog.py`), and compiles user SQL against virtual tables built in `views.py` (`SEMANTIC_TABLES`: `runs`, `updates`, `rollouts`). The `updates` view (`_updates_sql`) stacks every metric point of the measures a statement reads and groups them per run and step. `posttrain query --sql ...` (from `apps/lab`) runs these statements. The Observatory service (`service.py`, `traces.py`, `rollout_time.py`) builds run views and the React frontend (`apps/observatory/frontend/src`, `App.tsx`, `components/EvidenceChart.tsx`, `components/RolloutGroupTable.tsx`) renders them.

A **collection** is one population of episodes sampled at one policy version and applied as one or more optimizer updates. Its **collection step** is the logical step of its first update. **Collection-grain** metrics describe the population (rewards, rollout counts and seconds, admission and active sampling, credit evidence, endings); **update-grain** metrics describe an optimizer update (loss, gradient norm, KL, clipping, step time).

## Plan of Work

Milestone 1 (done) amends the baseline and makes the trainer record collection identity.

Milestone 2 adds the semantic `collection` entity. In `semantic_layer/model.py` the `EntityName` literal gains `collection`. In `metric_catalog.py` the collection-grain entries listed under Context change `entity` to `collection`. In `framework.py` `ENTITIES` gains the collection entity (ordered by `collection.step`), dimensions `collection.step`, `collection.time` and `collection.updates` are added, `update.collection_step` is added, `rollout.step` is described as the collection step, and `rows_per_update` becomes `rows_per_collection` on the collection entity. `rollout_share` cannot mix entities, so it becomes `rollout_seconds / collection_seconds` on the collection entity where `collection_seconds` is the sum of its updates' step times. In `views.py` `SEMANTIC_TABLES` gains `collection: collections`, and `_collections_sql` builds one row per run and collection step from the collection measures' points; `update.collection_step` is the attribute when present, otherwise the latest collection start at or before the update. Tests in `apps/observatory/tests/test_semantic_sql.py` and `conftest.py` gain a resolved run with 2 collections of 2 updates each, tagged and untagged.

Milestone 3 moves Observatory surfaces to collections where they show population values, and Milestone 4 validates against r10 and runs the ladder.

## Concrete Steps

From `/home/hammad/projects/worktrees/rl-perf`:

    uv run --frozen pytest -q packages/train/tests
    uv run --frozen pytest -q apps/observatory/tests
    cd apps/lab && ../../.venv/bin/posttrain query --format csv --sql "select step, updates, reward, rollouts_completed from collections where run_id='manifest-steps-26-sampo-100-g16x8-20261006-r10' order by step"

## Validation and Acceptance

For r10 the collections query returns one row per collection (steps 1, 5, 9, ...) with non-null reward and rollout counts, and `select step, collection_step from updates` maps steps 1 to 4 to collection 1, 5 to 8 to collection 5. For a legacy SAMPO run (`lfm26-sampo-cont100-g30x4-16t-lr6e5-kl1e2-20260930-r1`) the collections query returns 100 rows equal to its former per-update reward values.

## Idempotence and Recovery

All changes are code and documentation; no stored data is rewritten. Reverting the commits restores the previous views.

## Interfaces and Dependencies

`collections` table columns: `run_id`, `step` (the collection step, integer), `time`, `updates` (integer, updates recorded as trained on it), `collection_seconds`, and every measure whose entity is `collection`. `updates` gains `collection_step`. Trainer attributes: `collection_step` (integer) on collection rows and applied-update rows, `collection_update` (1-based integer) on applied-update rows; metric `train/rl/collection_updates`.
