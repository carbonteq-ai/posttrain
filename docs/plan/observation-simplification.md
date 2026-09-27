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
- [ ] Milestone 0: amend the canonical baseline (metric envelope: one value per update; readers return logical steps; SQL as the query language; notes cite SQL blocks).
- [ ] Milestone 1: metrics correct at the source (trainer writes per-update values; readers return logical steps through one shared function; one legacy rule for runs already recorded).
- [ ] Milestone 2: one metric catalog shared by job views and the semantic layer.
- [ ] Milestone 3: one query engine (SQLite with column-level loading; the short form compiles to SQL; delete the Python aggregation path and the formula parser).
- [ ] Milestone 4: notes and run cards use SQL blocks; delete the note query format.
- [ ] Milestone 5: re-render the four reports, update both older plans, release 0.4.11, and upgrade the shared Trackio server only after the user confirms.

## Surprises & Discoveries

- Observation: both tracking adapters already compute the logical step of a replayed point but return the provider step.
  Evidence: `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py` `_metric_series` computes `logical_step` only to filter by `start_step`/`end_step` and then stores `"step": row.get("step")`; `packages/tracking-wandb/src/posttrain_tracking_wandb/adapter.py` does the same with `posttrain/step`. The Observatory then re-derives it in `service.logical_metric_series`.
- Observation: per-batch rollout metrics are written by one place.
  Evidence: `packages/train/src/posttrain/train/backends/trl/policy_rollouts.py` calls `context.metrics({...}, step=optimizer_step, attributes={"rollout_population_scope": "candidate", "rollout_batch_ordinal": n})` once per rollout batch; with active sampling one optimizer step has several batches (for example 64, 24 and 8 rollouts at update 1 of `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2`). Replayed Verifiers evidence (`packages/train/src/posttrain/train/integrations/verifiers.py`, `_trace_metrics`) is already one point per update.
- Observation: the telemetry definitions are views, not vocabulary. `JobTelemetryDefinition` says which summary fields, charts, health rules and evidence requirements a job kind shows; only `_METRIC_HELP` duplicates the semantic measures.

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

## Outcomes & Retrospective

Not started.

## Context and Orientation

Posttrain is this repository, a `uv` workspace (`AGENTS.md`). Runs are recorded in a CarbonTeq fork of Trackio (self-hosted, project `posttrain-lab`) or in W&B, and read through `RunDataSource` (`packages/tracking/src/posttrain/tracking/contracts.py`): `metric_series(run_id, names)` returns `MetricSeries` (name and `MetricPoint`s with `value`, `step`, `observed_at`, `attributes`). The Observatory (`apps/observatory`) is the read product; `ObservatoryService` (`service.py`) builds job views from `JobTelemetryDefinition`s in `telemetry.py`. The semantic layer is `apps/observatory/src/posttrain_observatory/semantic_layer/` (model, framework declarations, executor, SQL mode, describe, reader); run notes are `note_render.py`, `run_cards.py` with templates in `note_templates/`, and `run_notes.py`; the CLI adds `posttrain query` (`apps/cli/src/posttrain_cli/commands/query_cmd.py`) and `posttrain run note` / `posttrain run card` (`note_cmd.py`).

Terms. An "update" is one optimizer step of a training run. A "rollout batch" is one group of rollouts generated during an update; active sampling may generate several per update. A "replayed" metric is one the host recomputes from preserved Verifiers traces when a run finishes; it is stored at the next free provider step and carries `observation_source: verifiers` and `source_step: <update>`. "Replay authority" means that when replayed points exist for an update, they replace the live points of the same name for that update. The "logical step" of a point is `source_step` for a replayed point and the recorded step otherwise.

## Plan of Work

Milestone 0 amends the baseline, alone in one commit. In `docs/post-training/06-observation-and-lineage.md` under Metrics/Envelope, state that a training metric has one value per update at that update's step, that per-batch detail uses its own names, and that tracking readers return points at logical steps with replay authority applied. Where computed views are described, state that the semantic layer is a set of named tables queried with read-only SQL and that the short query form compiles to SQL. In the Run notes subsection, state that data blocks are named SQL. In `docs/post-training/05-apis.md`, change `query_semantics(SemanticQuery | SqlQuery)` to say the short form compiles to SQL and results include the SQL; remove `--file`'s mention of a separate format only if it changes.

Milestone 1 makes metrics correct at the source. In `packages/tracking/src/posttrain/tracking/`, add `logical.py` with `logical_series(series: MetricSeries) -> MetricSeries` (move the body of `service.logical_metric_series`, including replay authority and duplicate collapsing) and `LEGACY_ROLLOUT_BATCH_METRICS: Mapping[str, Literal["sum", "mean"]]` applied to points carrying `rollout_batch_ordinal`. Both adapters (`TrackioDataSource._metric_series`, W&B `metric_series`) return `logical_series(...)` of what they read, with steps set to the logical step. The Observatory deletes `logical_metric_series` and its calls, and the semantic layer deletes `Source.within_step`, `_ROUND_SUMS`, `_ROUND_MEANS` and the projection in `semantic_layer/reader.py`. In `policy_rollouts.py`, accumulate the per-batch values in the existing per-step closure and write the per-update names once when the optimizer step's rollouts are complete (sums for counts, seconds and tokens; rates recomputed from the sums, for example tokens per second as total tokens over total seconds); write per-batch values under `train/rl/rollout_batch/<name>` with the ordinal attribute. Tests: a unit test in `packages/tracking/tests` for `logical_series` (replay authority, duplicates, legacy batches); a TRL bridge test that a step with three batches writes one per-update point with the summed values; adapter tests that returned steps are logical. Acceptance on real data: the semantic query `rollout_seconds:sum` for `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` update 1 returns 369.5 (251.7 + 60.9 + 56.9) through the legacy rule, and the Observatory GRPO view shows one point per update.

Milestone 2 creates the metric catalog. Add `metric_catalog.py` with `MetricEntry` (metric, name, label, description, interpretation, unit, aggregation, entity, job_kinds) built from `_METRIC_HELP` and the measure declarations in `semantic_layer/framework.py`, merged so each metric appears once. `telemetry._help_for` reads the catalog; `framework.py` keeps only entities, dimensions and metric formulas, and builds measures from the catalog. The test that every telemetry metric has a measure becomes a test that every telemetry metric has a catalog entry. Acceptance: `uv run pytest apps/observatory/tests` passes, `describe` output is unchanged except for text improvements, and the two old declarations no longer exist.

Milestone 3 makes SQLite the one engine. `sql.py` becomes the executor: `plan_columns(sql) -> dict[str, set[str]]` prepares the statement against empty tables created from the model under an authorizer that records `SQLITE_READ` (table, column) pairs; `build_database` loads only those columns for the runs in scope; aggregates `first(value, order)`, `last(value, order)` and `stddev(value)` are registered with `create_aggregate`; the `rollouts` table keeps `<measure>_sum`, `_count` and `_sum_squares` columns, and a view-free convention compiles rollout means as `sum(x_sum) / sum(x_count)`. `query.py` gains `compile_query(model, SemanticQuery) -> str` producing `SELECT <dims>, <aggregates> FROM <grain table> JOIN runs ... WHERE ... GROUP BY ... ORDER BY ... LIMIT`; metric formulas in the model are SQL expressions. `SemanticResult` gains `sql: str`. Delete the Python aggregation path in `execute.py` (keep run resolution and loaders), `formula.py`, and `SqlQuery.load`. Tests: every existing semantic-layer test keeps its expected values through the compiled path; a test that `plan_columns` finds the columns of a join, a `*`, and a CTE; a test that loading reads only planned series. Acceptance: the commands in Purpose print the same values as before, and `posttrain query ... --format json` includes the SQL.

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
