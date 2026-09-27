# Simplify the observation architecture: correct metrics at the source, one metric catalog, SQL as the one query language

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md` and revises two plans that are partly implemented: `docs/plan/semantic-layer.md` and `docs/plan/run-notes.md`.

## Purpose / Big Picture

The semantic layer and run notes work (branch `codex/run-notes`, stacked on `codex/semantic-layer`, pull request 127), but the architecture has more concepts than the problem needs. Writing real reports over the `posttrain-lab` runs showed where. Three things repeat or disagree:

- Metric meaning is declared twice. The Observatory's metric help catalog (`_METRIC_HELP` in `apps/observatory/src/posttrain_observatory/telemetry.py`, about 740 lines of label, description and unit per metric) and the semantic layer's measure declarations (`semantic_layer/framework.py`, about 1,050 lines) each describe the same metrics. A test keeps them from drifting apart.
- There are four ways to write a query: the semantic query form, SQL, a note data-block format, and CLI flags. Two engines answer them: semantic queries aggregate in Python (`semantic_layer/execute.py` plus a formula parser in `formula.py`), and SQL runs in SQLite (`semantic_layer/sql.py`). The same answer can come from two code paths.
- Readers repair data that should be correct when written. An active-sampling trainer writes rollout metrics once per rollout batch, so one update has several points of `train/rl/time/rollout_seconds` and `train/rl/rollouts_*`. Metrics replayed from Verifiers traces when a run finishes are stored at their append position, and their real update is in a point attribute `source_step`. Both tracking adapters compute that logical step but return the provider step, the Observatory projects it again (`service.logical_metric_series`), and the semantic layer adds its own rules for combining points within an update (`Source.within_step`). Before those rules existed, the layer reported that runs spent 66% of each update in rollouts when the truth was 89%.

After this change there is one idea to learn. Every metric is described once in a metric catalog. Metrics are written once per update at the update's step, and the tracking readers return points at logical steps. The semantic layer is a set of named tables (`runs`, `updates`, `rollouts`, `eval_tasks`, `load_levels`, `raw_metrics`) whose columns are catalog names, queried with read-only SQL; the short "measures by dimensions" form still exists for the command line and agents, but it compiles to SQL and shows the SQL it ran. A run note is Markdown with named SQL blocks, views (`chart`, `value`, `table`) that display a block, `{{block.column}}` references and `[[run:id]]` links, which is the evidence.dev model this work set out to copy. The Posttrain side should shrink by roughly half.

To see it working: `posttrain query -m update_seconds:sum,rollout_share --by run.id --runs lfm12-sampo-8gb-20260926-r15` prints the same numbers as today (766.2 seconds, 0.892) and the SQL it compiled; `posttrain query --sql "select run_id, sum(rollout_seconds) from updates group by run_id" --runs lfm12-sampo-8gb-20260926-r15` returns 683.5 without any `--load` option; and the four reports in Artifacts and Notes render with the same values as before, written with SQL blocks.

## Progress

- [x] (2026-09-27) Checkpointed the current state: semantic layer surfaces (0f60c625), canonical run-notes amendment (179fb196), note store (803aff8d), renderer, cards and surfaces (946375d3), Trackio dev28 pin (5473a0a7), web page and report fixes (8f49991c).
- [x] (2026-09-27) Wrote this plan after an architecture review with the user.
- [x] (2026-09-27) Milestone 0: canonical amendment (d26d8aca).
- [x] (2026-09-27) Milestone 1: the TRL trainer writes rollout metrics once per update (`backends/trl/update_totals.py`, flushed by a step-end callback and on failure; per-batch time goes to `train/rl/rollout_batch_seconds`); `posttrain.tracking.logical.logical_series` applies replay authority and the legacy batch rule, and both adapters return it; the Observatory projection, the semantic layer's `within_step` rules and its reader projection are deleted. Real data unchanged: rollout share 0.896/0.895/0.892/0.879 for r13/r14/r15/turns-r1; `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` update 1 has 369.5 s over 96 rollouts.
- [ ] Milestone 2: one metric catalog shared by job views and the semantic layer.
- [x] (2026-09-27) Milestone 3, first attempt (in-memory SQLite engine with column-level loading) parked on branch `codex/sqlite-engine-wip` after the direction changed; see the Decision Log.
- [x] (2026-09-27) Milestone 3a: fork `project_sql` (9b02b7fc, 416950d4, 1c62b0b5 on `codex/run-notes`): 17 unit tests on both SQLite drivers and the Doris path with a recording connection; the real-Doris test is written and skips without `TRACKIO_DORIS_*`. Release as dev29 in progress.
- [x] (2026-09-27) Milestone 3b (c6e8dbe9): `ProjectSql` in `posttrain.tracking`; `semantic_layer/views.py`, `compile.py`, `engine.py`; loaders, in-memory engine, Python aggregation, `formula.py`, and the `eval_task`/`load_level` entities deleted. Tests run on a local Trackio project written by Posttrain's own writer, including updates-view equivalence with `logical_series`.
- [x] (2026-09-27) Milestone 4 (c6e8dbe9, 543e82ca): notes use ```sql <name>``` blocks with `-- runs:` scopes (ids, or filters with any-of lists); templates rewritten (`@2`); cards render with no unresolved references on the local project. The four reports are rewritten in SQL and compile against the model; they run on real data once the shared server has dev29.
- [ ] Milestone 5: re-render the four reports, update both older plans, release 0.4.11, and upgrade the shared Trackio server only after the user confirms.

## Surprises & Discoveries

- Observation: "one value per update" must mean one value per update for each set of tags. Curriculum and active-sampling round metrics are tagged per class or per round (`class_id`, `round_index`) and named as round metrics, so they already conform; only the untagged per-batch writes (rollout counts and time, SAMPO advantage means) broke the rule.
  Evidence: `backends/trl/policy_curriculum.py` writes `train/rl/curriculum/*` with `round_index` and `class_id`; `policy_rollouts.py` wrote `train/rl/episode_advantage_mean` and others once per batch with no tag.

- Observation: both tracking adapters already compute the logical step of a replayed point but return the provider step.
  Evidence: `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py` `_metric_series` computes `logical_step` only to filter by `start_step`/`end_step` and then stores `"step": row.get("step")`; `packages/tracking-wandb/src/posttrain_tracking_wandb/adapter.py` does the same with `posttrain/step`. The Observatory then re-derives it in `service.logical_metric_series`.
- Observation: per-batch rollout metrics are written by one place.
  Evidence: `packages/train/src/posttrain/train/backends/trl/policy_rollouts.py` calls `context.metrics({...}, step=optimizer_step, attributes={"rollout_population_scope": "candidate", "rollout_batch_ordinal": n})` once per rollout batch; with active sampling one optimizer step has several batches (for example 64, 24 and 8 rollouts at update 1 of `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2`). Replayed Verifiers evidence (`packages/train/src/posttrain/train/integrations/verifiers.py`, `_trace_metrics`) is already one point per update.
- Observation: the telemetry definitions are views, not vocabulary. `JobTelemetryDefinition` says which summary fields, charts, health rules and evidence requirements a job kind shows; only `_METRIC_HELP` duplicates the semantic measures.

- Observation: in Doris, metrics are stored one row per logged batch with all values in a JSON string (`metrics.metrics`), configs as a JSON string (`configs.config`), and trace facts as real columns of `traces` (`fact_task_id`, `fact_task_reward`, `fact_is_truncated`, ...). The fork's `query_project` works only on SQLite and refuses Doris.
  Evidence: `trackio/doris_schema.py` (fork `codex/run-notes`); `DorisStorage.query_project` raises "query_project is SQLite-specific".

- Observation: evaluation task results in the Observatory come from the task manifest and repetition slots (first usable attempt per slot), which trace facts do not carry, so an `eval_tasks` SQL view would have been a second, different definition of the same numbers. Evaluation questions in SQL are rollouts grouped by task; the Observatory's evaluation and serving views keep their own computations.
  Evidence: `apps/observatory/src/posttrain_observatory/evaluation_measurement.py` (slots, retries, `usable` attempt).
- Observation: on SQLite a CTE named like its base table (`traces`, `run_notes`) is a circular reference; the fork qualifies SQLite base tables as `main.<table>`. The default Turso engine stores JSON text columns as bytes, so the SQLite stand-ins decode bytes.
  Evidence: fork commit 1c62b0b5; `test_json_paths_follow_doris_quoting`.
- Observation: `max_by(x, step)` returns x at the latest step even where x is null, and update rows exist at every step with an update time, so "first" and "last" use `max_by(x, CASE WHEN x IS NOT NULL THEN step END)`.
- Observation: Doris behaviour this design relies on is untested against a real Doris: quoted JSON path keys (`$."train/rl/entropy"`), `json_extract_*` on nested objects, `unix_timestamp` on ISO 8601 strings with offsets, `max_by`/`min_by` with null order keys, `NULLS LAST`, and the query-timeout session variable. The real-Doris integration test is the release gate for deploying dev29.

## Decision Log

- Decision: metrics are written once per update; per-batch detail, when wanted, goes to its own names (`train/rl/rollout_batch/*`), never to the per-update names.
  Rationale: every consumer (Observatory charts, alerts, summaries, the semantic layer, W&B) then reads one value per update without combining rules. Rules in readers were the source of the rollout-share error.
  Date/Author: 2026-09-27, user agreed to Claude's recommendation.
- Decision: tracking readers return metric points at logical steps with replay authority already applied, through one function in `posttrain.tracking` that both adapters call; the Observatory stops projecting.
  Rationale: the rule is part of the normalized-evidence contract that `posttrain.tracking` owns, and today it lives in three places.
  Date/Author: 2026-09-27, Claude.
- Decision: runs recorded before Milestone 1 keep per-batch points; one legacy rule in `posttrain.tracking` combines points that carry `rollout_batch_ordinal` (sum for counts and seconds, mean for rates), and is removed once no retained run needs it.
  Rationale: tracking history is append-only and cannot be rewritten; one declared compatibility rule is better than rules spread through readers.
  Date/Author: 2026-09-27, Claude.
- Decision: one metric catalog (`apps/observatory/src/posttrain_observatory/metric_catalog.py`) holds, per metric, its raw name, short name, label, description, interpretation, unit, default aggregation and job kinds. Job views keep referencing raw metric names and take help from the catalog; semantic measures are the catalog entries.
  Rationale: removes the duplicated vocabulary without rewriting 2,600 lines of job views.
  Date/Author: 2026-09-27, Claude.
- Decision: SQL over the semantic tables is the one query language and SQLite is the one engine. The short form (`measures`, `by`, `where`, `runs`) compiles to SQL and the result carries the compiled SQL. Metrics are SQL expressions over aggregates (`sum(rollout_seconds) / sum(update_seconds)`). Tables load only the columns the SQL reads, found by preparing the statement against empty tables under an authorizer that records each column read; the `load` option goes away.
  Rationale: one execution path; people and language models already know SQL; column-level loading keeps reads bounded without asking the author which series to fetch.
  Date/Author: 2026-09-27, user agreed to Claude's recommendation.
- Decision: a note's data block is a fenced block whose info string is `sql <name>`; an optional first line `-- runs: <ids, a glob, or dimension=value filters>` scopes it (default: the note's run). Views keep their small `key: value` bodies. The note query format (`measures:`/`by:` blocks) is removed.
  Rationale: this is the evidence.dev model (Markdown plus named SQL); a leading SQL comment keeps the block valid SQL.
  Date/Author: 2026-09-27, Claude.
- Decision: keep notes stored in Trackio as built (fork 0.31.5.post14.dev28), but do not upgrade the shared server until this plan is done, so production migrates once.
  Rationale: storage and its contract are already tested end to end; the simplification does not change them.
  Date/Author: 2026-09-27, Claude recommendation, pending user confirmation for the server upgrade.

- Decision: SQL runs inside Trackio's storage. Doris (the shared server's storage) is the primary engine; a project on local SQLite storage gets the same SQL translated to SQLite, and anything that does not translate is refused with a clear message. The query layer and run notes do not support W&B projects.
  Rationale: user direction ("drop web support and make this primarily for doris with limited support for sqlite"). Querying data where it lives removes per-query copying (loaders, column planning, the run cap, concurrent reads) and gives Doris's SQL (window functions, percentiles, JSON functions) over every run of a project. W&B cannot run SQL.
  Date/Author: 2026-09-27, user decision.
- Decision: the SQL people and notes write is Doris SQL (MySQL flavour). The fork's endpoint accepts Doris SQL and, on SQLite storage, translates it with `sqlglot`; `sqlglot` also validates statements (one SELECT, only the project's logical tables, no qualified names) on both engines.
  Rationale: one dialect for authors; `sqlglot` is the standard parser and translator and avoids hand-written validation.
  Date/Author: 2026-09-27, Claude.
- Decision: the fork exposes generic logical tables per project (`metric_rows`, `run_configs`, `traces` with fact columns, `run_notes`), scoped by wrapping the statement in CTEs that read the base tables filtered by project; Posttrain defines its semantic tables (`runs`, `updates`, `rollouts`, `eval_tasks`) as CTEs over those, generated for only the columns a statement reads.
  Rationale: fork policy (`AGENTS.md`): generic storage and query work goes to Trackio; Posttrain meaning (run summaries from `schema_version` 4 configs, logical steps, catalog metrics) stays in Posttrain.
  Date/Author: 2026-09-27, Claude.
- Decision: the two data rules of Milestone 1 (replay authority at `source_step`, legacy per-batch combination) exist twice: in `posttrain.tracking.logical_series` for the Observatory's run views, and in the generated `updates` view. A test checks that both give the same values on the same recorded rows.
  Rationale: `AGENTS.md`: when two backends implement one contract, test equivalent logical results.
  Date/Author: 2026-09-27, Claude.

## Outcomes & Retrospective

(2026-09-27) Milestones 0 to 4 are done on `codex/run-notes`. Metrics are correct where they are written, one catalog describes every metric, and queries, notes and cards run as Doris SQL inside Trackio's storage (translated on local SQLite). The Posttrain side of the semantic layer and notes went from 4,310 to 3,346 lines, with one engine instead of two, no per-query copying, no run cap, percentiles over rollouts, and every result showing its SQL; the fork gained about 550 lines for project SQL. Remaining: release fork dev29 and pin it; run the real-Doris gate; upgrade the shared Trackio server (Doris schema v4 migration, needs the user's confirmation); render the four reports on real data; release Posttrain 0.4.11.

## Context and Orientation

Posttrain is this repository, a `uv` workspace (`AGENTS.md`). Runs are recorded in a CarbonTeq fork of Trackio (self-hosted, project `posttrain-lab`) or in W&B, and read through `RunDataSource` (`packages/tracking/src/posttrain/tracking/contracts.py`): `metric_series(run_id, names)` returns `MetricSeries` (name and `MetricPoint`s with `value`, `step`, `observed_at`, `attributes`). The Observatory (`apps/observatory`) is the read product; `ObservatoryService` (`service.py`) builds job views from `JobTelemetryDefinition`s in `telemetry.py`. The semantic layer is `apps/observatory/src/posttrain_observatory/semantic_layer/` (model, framework declarations, executor, SQL mode, describe, reader); run notes are `note_render.py`, `run_cards.py` with templates in `note_templates/`, and `run_notes.py`; the CLI adds `posttrain query` (`apps/cli/src/posttrain_cli/commands/query_cmd.py`) and `posttrain run note` / `posttrain run card` (`note_cmd.py`).

Terms. An "update" is one optimizer step of a training run. A "rollout batch" is one group of rollouts generated during an update; active sampling may generate several per update. A "replayed" metric is one the host recomputes from preserved Verifiers traces when a run finishes; it is stored at the next free provider step and carries `observation_source: verifiers` and `source_step: <update>`. "Replay authority" means that when replayed points exist for an update, they replace the live points of the same name for that update. The "logical step" of a point is `source_step` for a replayed point and the recorded step otherwise.

## Plan of Work

Milestone 0 amends the baseline, alone in one commit. In `docs/post-training/06-observation-and-lineage.md` under Metrics/Envelope, state that a training metric has one value per update at that update's step, that per-batch detail uses its own names, and that tracking readers return points at logical steps with replay authority applied. Where computed views are described, state that the semantic layer is a set of named tables queried with read-only SQL and that the short query form compiles to SQL. In the Run notes subsection, state that data blocks are named SQL. In `docs/post-training/05-apis.md`, change `query_semantics(SemanticQuery | SqlQuery)` to say the short form compiles to SQL and results include the SQL; remove `--file`'s mention of a separate format only if it changes.

Milestone 1 makes metrics correct at the source. In `packages/tracking/src/posttrain/tracking/`, add `logical.py` with `logical_series(series: MetricSeries) -> MetricSeries` (move the body of `service.logical_metric_series`, including replay authority and duplicate collapsing) and `LEGACY_ROLLOUT_BATCH_METRICS: Mapping[str, Literal["sum", "mean"]]` applied to points carrying `rollout_batch_ordinal`. Both adapters (`TrackioDataSource._metric_series`, W&B `metric_series`) return `logical_series(...)` of what they read, with steps set to the logical step. The Observatory deletes `logical_metric_series` and its calls, and the semantic layer deletes `Source.within_step`, `_ROUND_SUMS`, `_ROUND_MEANS` and the projection in `semantic_layer/reader.py`. In `policy_rollouts.py`, accumulate the per-batch values in the existing per-step closure and write the per-update names once when the optimizer step's rollouts are complete (sums for counts, seconds and tokens; rates recomputed from the sums, for example tokens per second as total tokens over total seconds); write per-batch values under `train/rl/rollout_batch/<name>` with the ordinal attribute. Tests: a unit test in `packages/tracking/tests` for `logical_series` (replay authority, duplicates, legacy batches); a TRL bridge test that a step with three batches writes one per-update point with the summed values; adapter tests that returned steps are logical. Acceptance on real data: the semantic query `rollout_seconds:sum` for `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` update 1 returns 369.5 (251.7 + 60.9 + 56.9) through the legacy rule, and the Observatory GRPO view shows one point per update.

Milestone 2 creates the metric catalog. Add `metric_catalog.py` with `MetricEntry` (metric, name, label, description, interpretation, unit, aggregation, entity, job_kinds) built from `_METRIC_HELP` and the measure declarations in `semantic_layer/framework.py`, merged so each metric appears once. `telemetry._help_for` reads the catalog; `framework.py` keeps only entities, dimensions and metric formulas, and builds measures from the catalog. The test that every telemetry metric has a measure becomes a test that every telemetry metric has a catalog entry. Acceptance: `uv run pytest apps/observatory/tests` passes, `describe` output is unchanged except for text improvements, and the two old declarations no longer exist.

Milestone 3a is in the Trackio fork (`/home/hammad/projects/trackio-run-notes`, branch `codex/run-notes`, next version `0.31.5.post14.dev29`). `query_project(project, sql, max_rows, timeout_seconds)` works on both engines. It parses the statement with `sqlglot` (dialect `doris`), accepts one `SELECT` (optionally with `WITH`) whose table references are all unqualified names of the logical tables, and wraps it in CTEs that read the base tables filtered by project: `metric_rows(run_id, run_name, step, timestamp, metrics)`, `run_configs(run_id, run_name, config, created_at)`, `traces(run_id, run_name, step, timestamp, trace_type, external_id, fact_* columns)` and `run_notes(...)`. On Doris the wrapped statement runs with a `query_timeout` hint; on SQLite it is translated to SQLite by `sqlglot` and runs read-only under the existing authorizer with a progress-handler timeout. Results are column names and rows, at most `max_rows`. The client gains `Api.query_project`, the server registers the endpoint, and `Api.capabilities()` reports `project_sql`. Tests cover both engines with the fork's existing SQLite stand-in for Doris, plus the real-Doris integration test (`tests/integration/test_doris_storage.py`, which needs `TRACKIO_DORIS_*`).

Milestone 3b is in Posttrain. `posttrain.tracking` gains a protocol `ProjectSql` (`query(sql, *, max_rows) -> (columns, rows)`) that `TrackioDataSource` implements through `Api.query_project`. `semantic_layer/views.py` generates the semantic tables as CTEs over the fork's tables in Doris SQL: `runs` from `run_configs` (Posttrain `schema_version` 4 configs: run id, work package, job kind, stage, settings paths into `resolved_selections`) joined with each run's latest `run/status` row in `metric_rows`; `updates` from `metric_rows`, one column per catalog metric the statement reads, with the logical step and the two data rules; `rollouts` from `traces` fact columns (so percentiles work); `eval_tasks` from evaluation traces grouped by task. Planning which columns a statement reads uses `sqlglot` (`qualify` and column lineage), replacing the SQLite authorizer planner. `compile.py` (from the parked branch) compiles the short form to SQL over these tables. The per-query loading path (`loaders.py`), the in-memory engine, the Python aggregation path and `formula.py` are deleted; `load_levels` waits until serving evidence has a storage-side shape. Tests: generated views run on local Trackio SQLite storage in tests; an equivalence test compares the `updates` view with `logical_series` on the same rows; the four reports render with the same values against the shared server once it runs dev29 (after the user confirms the upgrade).

Milestone 4 moves notes to SQL. In `note_render.py`, a data block is a fence with info string `sql <name>`; its optional first line `-- runs: ...` sets the scope using the same parser as `--runs`. Remove the `data` block key set and the list/mapping value parser; views keep `key: value` lines with scalars and `[a, b]` lists. Rewrite the templates in `note_templates/` as SQL (for example `select first(reward, step) as reward_first, last(reward, step) as reward_last, ... from updates`). Update `render_note_preview`'s MCP description and the web page's query disclosure (it shows `view.query`, now SQL). Tests: renderer tests rewritten with SQL blocks; card tests unchanged in expectation; the four reports render with no unresolved markers and the same values.

Milestone 5 finishes. Re-render the reports, record any change in values, update `docs/plan/semantic-layer.md` and `docs/plan/run-notes.md` with revision notes pointing here, update `docs/tooling/trackio/README.md` for dev28, run the release (0.4.11: images publish locally, release PR, release-candidate workflow, merge, promote, tag), and ask the user before upgrading the shared Trackio server (Doris needs `trackio storage migrate-doris --to 4 --apply --backup-receipt ...` before a dev28 server starts).

## Concrete Steps

From `/home/hammad/projects/rl-perf-guard` on branch `codex/run-notes`:

    uv run pytest -q packages/tracking/tests packages/tracking-trackio/tests packages/tracking-wandb/tests packages/train/tests apps/observatory/tests apps/cli/tests
    uv run ruff check . && uv run pyright && uv run lint-imports && git diff --check
    (cd apps/observatory/frontend && npm test && npm run build)

Real-data checks run from `apps/lab` after `source /home/hammad/projects/rl/scripts/orenv.sh` (never print its values):

    uv run posttrain query -m update_seconds:sum,rollout_share --by run.id --runs lfm12-sampo-8gb-20260926-r15
    uv run posttrain run card lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2

## Validation and Acceptance

The change is accepted when the values in Purpose and in the four reports match what the current implementation produces, the Observatory GRPO view of `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` shows one rollout-time point per update, the deleted modules (`formula.py`, `logical_metric_series`, `within_step`, `SqlQuery.load`, the note query format) are gone, and the Posttrain line count of the semantic layer and notes (today about 5,100) has dropped substantially.

## Idempotence and Recovery

Each milestone is one or more commits on `codex/run-notes`; reverting a milestone's commits restores the previous behaviour. Nothing here writes to the shared Trackio server. The legacy per-batch rule is read-only.

## Artifacts and Notes

The four reports used for acceptance are kept in the session scratchpad and are to be committed with Milestone 5 as examples under `apps/observatory/src/posttrain_observatory/note_templates/examples/`: the 8 GB SAMPO bring-up (anchor `lfm12-sampo-turns-8gb-20260926-r1`), the VORTEX v5 KL run against earlier VORTEX runs (anchor `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2`), the 64K held-out evaluations (anchor `eval-lfm26-heldout-64k-base-20260924-r1`), and cleanup candidates (training runs with fewer than ten updates). The held-out report found that `eval-lfm26-heldout-64k-vortex-v4-step20-20260924-r1` averages 0.694 over the 15 tasks with a valid reward but 0.521 when its 5 fully truncated tasks count as zero, below the base model's 0.592.

## Interfaces and Dependencies

In `packages/tracking/src/posttrain/tracking/logical.py`:

    def logical_series(series: MetricSeries) -> MetricSeries: ...
    LEGACY_ROLLOUT_BATCH_METRICS: Mapping[str, Literal["sum", "mean"]]

In `apps/observatory/src/posttrain_observatory/metric_catalog.py`:

    class MetricEntry(ObservatoryModel): metric: str; name: str; label: str; description: str; interpretation: str | None; unit: str | None; aggregation: Aggregation; entity: EntityName; job_kinds: tuple[str, ...]
    METRIC_CATALOG: tuple[MetricEntry, ...]

In `apps/observatory/src/posttrain_observatory/semantic_layer/`:

    def compile_query(model: SemanticModel, query: SemanticQuery) -> str: ...
    def plan_columns(model: SemanticModel, sql: str) -> dict[str, frozenset[str]]: ...
    async def run_sql(model: SemanticModel, reader: SemanticReader, sql: str, runs: RunScope) -> SemanticResult: ...

No new dependency. SQLite comes with Python.
