# Add a semantic layer for querying runs: entities, dimensions and measures defined once per job type

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Answering a question about runs today means knowing raw metric names and writing code against the tracking reader. "How did entropy evolve per update in the KL run compared with the 1e-4 run?" requires knowing the series is called `train/rl/entropy`; "what share of each update did rollout take in the 8 GB SAMPO runs?" requires fetching `train/rl/time/rollout_seconds` and `train/step_time_seconds` for each run and dividing; "reward by task domain for an evaluation" requires the trace-fact aggregation API. Each Observatory view and each script re-derives these meanings in its own way.

After this change there is a semantic layer: a model, declared once per job type, of what can be asked about runs. It names entities (the grains of data, such as a run, one training update, one rollout, one evaluation task, one serving load level), dimensions (fields to group and filter by, such as `run.work_package`, `run.learning_rate`, `update.step`, `rollout.task_type`), measures (numbers with a default aggregation, such as `reward`, `entropy`, `update_seconds`, `pass_rate`, `throughput`) and a few metrics (named formulas over measures, such as `rollout_share`). Each measure records where its values come from (a Trackio metric series, a trace fact, an evaluation or serving view field, or a recorded setting), so `entropy` means `train/rl/entropy` in exactly one place.

A query names measures, dimensions and filters, for example

    measures: [entropy, kl]
    by: [run, update.step]
    where: {run.id: [lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2, lfm26-vortex-v5-150-dspark-opt-20260926-r1]}

and returns a table. The layer compiles it into only the reads it needs, joins them at the requested grain and aggregates. For anything the query form cannot express, a raw SQL mode runs a read-only `SELECT` over the same clean tables (`runs`, `updates`, `rollouts`, `eval_tasks`, `load_levels`) whose columns carry semantic names. A `describe` call returns what is available for a set of runs, with descriptions and units, so a person or a language model can write correct queries without guessing names.

The layer is exposed through the Observatory service, HTTP (`POST /api/v1/semantic/query`), the Observatory's MCP server (tools `describe_semantics` and `query_semantics`, so an agent can answer questions directly) and the CLI (`posttrain query`). The run-notes plan (`docs/plan/run-notes.md`) builds on it: a note's data blocks are semantic queries, and its charts, values and tables display their results. This plan adds no write path and does not change the Observatory's read-only contract.

To see it working: `posttrain query --measures update_seconds:mean --by run --where run.id=lfm12-sampo-turns-8gb-20260926-r1` prints one row with 195.5 (the mean of its three updates, 261.9, 138.0 and 186.7 seconds).

## Progress

- [x] (2026-09-27 03:00Z) Surveyed the existing pieces this plan builds on (Context and Orientation).
- [ ] Milestone 0: amend the canonical baseline to name the semantic layer as the definition of computed views and to list its query surfaces.
- [ ] Milestone 1: semantic model types and the framework declarations for the job types in use.
- [ ] Milestone 2: query compiler and executor (semantic query to reads to SQLite to result).
- [ ] Milestone 3: raw SQL mode.
- [ ] Milestone 4: `describe`.
- [ ] Milestone 5: surfaces: Observatory service, HTTP, MCP, CLI.
- [ ] Milestone 6: acceptance on real runs in `posttrain-lab`.

## Surprises & Discoveries

- Observation: the Observatory already holds most of the vocabulary, scattered across views.
  Evidence: `apps/observatory/src/posttrain_observatory/telemetry.py` declares one `JobTelemetryDefinition` per job kind (`GRPO_TELEMETRY`, `SAMPO_TELEMETRY`, `GENERAL_EVAL_TELEMETRY`, `SERVE_BENCHMARK_TELEMETRY`, ...) with summary fields (a metric and a reducer `last|min|max|mean|sum`), charts and a metric help catalog (`_METRIC_HELP`: label, description, interpretation, unit). Trace facts already support a small semantic query (`TraceFactsQuery`: group by `task_id`, `prompt_group_id`, `rollout_step`, ...; aggregate `task_reward`, token counts, `tool_calls`, ... with `mean|sum|sum_squares|count`).

## Decision Log

- Decision: build a semantic layer (entities, dimensions, measures, metrics declared per job type, one query form, a raw SQL mode, `describe`) and have run notes query it.
  Rationale: user direction ("design good data apis based on job types and our capabilities", "imagine creating a semantic layer which can be used to make queries", "we can even include raw sql mode"). Declaring meanings once removes raw metric names from every consumer, gives agents a catalog to query against, and lets notes, views and scripts agree on what `reward` means.
  Date/Author: 2026-09-27, user direction; design by Claude.
- Decision: the layer lives in the Observatory (`apps/observatory/src/posttrain_observatory/semantic_layer/`) and reads only through `RunDataSource` and existing Observatory view builders.
  Rationale: the Observatory is the job-aware read product (`AGENTS.md`); it must not import `trackio` or `wandb` (import-linter), and reading through the protocol keeps the layer provider-neutral.
  Date/Author: 2026-09-27, Claude.
- Decision: both query modes execute in a fresh in-memory SQLite database per query, populated only with the scoped data; SQL is limited to reading by SQLite's authorizer, bounded by a progress-handler time limit and row caps.
  Rationale: SQLite is in the Python standard library (no new dependency), supports window functions and `FILTER`, and its authorizer callback can deny everything except reads; user SQL never reaches the Trackio database.
  Date/Author: 2026-09-27, Claude.
- Decision: rollout-level measures support only the aggregations trace facts can compute exactly (`count`, `sum`, `mean`, `stddev` from `sum_squares`); no percentiles at rollout grain.
  Rationale: rollouts are aggregated by the tracking backend (`aggregate_trace_facts`), not downloaded; downloading traces to compute percentiles would be slow and is not needed for the first questions.
  Date/Author: 2026-09-27, Claude.
- Decision: project-defined measures and metrics are out of scope for this plan; the framework declarations are the only vocabulary at first.
  Rationale: fewer moving parts until the model has been used; a project overlay can be added later the same way catalogs overlay.
  Date/Author: 2026-09-27, Claude.

## Outcomes & Retrospective

Not started.

## Context and Orientation

Posttrain is this repository, a Python 3.12 `uv` workspace (`AGENTS.md`). Runs are recorded in a CarbonTeq fork of Trackio (self-hosted, project `posttrain-lab`) and read through the provider-neutral protocol `RunDataSource` in `packages/tracking/src/posttrain/tracking/contracts.py`: `list_runs(RunQuery)` returns `RunSummary` rows (run id, work package id, job kind, status, start and finish times); `get_run(run_id)` returns `RunDetail` with `resolved_inputs` (the resolved selection per job seat: `model`, `environment`, `settings`, `training`, `rollout_inference`, ...), `source_metadata`, `metric_names` and events; `metric_series(run_id, names)` returns step-indexed series; `aggregate_trace_facts(run_id, TraceFactsQuery)` groups rollout facts by dimensions (`model`, `task_type`, `task_id`, `prompt_group_id`, `rollout_step`, `is_truncated`, `has_error`) and aggregates measures (`model_input_tokens`, `model_output_tokens`, `thinking_tokens`, `tool_calls`, `model_calls`, `trace_latency_ms`, `task_reward`, `algorithm_reward`, reward components) with `mean`, `sum`, `sum_squares` or `count`. Sources report `TrackingCapabilities`, including `trace_facts` (`available`, `unsupported`, `unavailable`). The Trackio implementation is `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`.

The Observatory (`apps/observatory`) is the read product over those sources. `ObservatoryService` (`apps/observatory/src/posttrain_observatory/service.py`) builds job-aware views: `get_run_view`, `get_metric_series` (a `MetricSeriesQuery` accepts at most 12 names and at most 2,000 points per series, downsampling longer ones), `compare_runs`, `get_trace_evaluation_view` (per-task evaluation results `EvaluationTaskResult`: planned and valid repetitions, failures, truncations, mean reward, success frequency, facets), `get_serving_capacity_view` (per-load-level `ServingOperatingPoint`: concurrency, context tokens, requests attempted, completed and failed, token counts, measurement seconds, aggregate output tokens per second, failure rate), `get_rollout_time` and `get_prompt_group_rewards`. `telemetry.py` holds the per-job-kind `JobTelemetryDefinition` registry (`telemetry_registry`, `DEFAULT_TELEMETRY_DEFINITIONS`) and the metric help catalog. `packages/advisor/src/posttrain/advisor/snapshot.py` extracts seats from a run's resolved inputs deterministically (`settings_seat`, `training_seat`, `environment_seat`, ...). `semantic.py` in the Observatory is unrelated: it produces optional language-model summaries; this plan's package is named `semantic_layer` to avoid confusion. The Observatory's HTTP routes are in `http.py`, its MCP tools in `mcp.py` (all read-only), and checked-in contract snapshots `apps/observatory/openapi.json` and `apps/observatory/mcp-schema.json` must be regenerated when they change. The CLI (`apps/cli`) builds an `ObservatoryService` for `posttrain run show` through `project_observatory_settings` and `create_service` in `apps/cli/src/posttrain_cli/commands/run_cmd.py`.

Terms. An entity is a kind of row: `run` (one per run), `update` (one per run and logged training step), `rollout` (one rollout episode, only ever aggregated), `eval_task` (one per evaluation run and task), `load_level` (one per serving benchmark run and concurrency level). A dimension is a named field of an entity used to group or filter; dimensions of `run` can be used with any entity because every row belongs to a run. A measure is a named number belonging to an entity, with a unit, a description, a default aggregation and the aggregations allowed. A metric is a named formula over aggregated measures of one entity. The grain of a query is the entity whose rows it groups, decided by its `by` dimensions (the most detailed entity named, or `run` when only run dimensions appear).

## Plan of Work

Milestone 0 amends the canonical baseline first (`AGENTS.md`). In `docs/post-training/06-observation-and-lineage.md`, where computed views are described ("Mean reward, success rate, truncation rate by slice are computed views"), add that computed views are defined by the semantic layer: entities, dimensions and measures declared per job type, each with its source, queried through one read-only query form or read-only SQL over the same entities. In `docs/post-training/05-apis.md`, add `describe_semantics` and `query_semantics` to the Observatory API listing (service, HTTP, MCP) and `posttrain query` to the CLI list. Commit this alone.

Milestone 1 adds the model. In `apps/observatory/src/posttrain_observatory/semantic_layer/model.py`, define `Entity` (name, description, parent entity or none, key dimensions), `Dimension` (name such as `run.learning_rate`, entity, type `string|integer|number|time|boolean`, description, source), `Measure` (name, entity, label, description, unit, source, default aggregation, allowed aggregations, the job kinds that provide it), `Metric` (name, entity, label, description, unit, formula) and `SemanticModel` (all of the above, validated: names unique, every dimension and measure belongs to a declared entity, every formula references measures of its own entity). A source is one of: `run_field` (a `RunSummary` field), `setting` (a path into resolved inputs, read with the advisor seat helpers, for example the settings seat's `loop.learning_rate`), `metric_series` (a Trackio metric name), `trace_fact` (a trace-fact dimension or measure), `eval_view` (an `EvaluationTaskResult` field), `serving_view` (a `ServingOperatingPoint` field), or `derived` (computed at load, such as `update.time` from the step timestamp).

In `apps/observatory/src/posttrain_observatory/semantic_layer/framework.py`, declare the framework model. The `run` entity gets dimensions `run.id`, `run.work_package`, `run.job_kind`, `run.status`, `run.started_at`, `run.finished_at`, `run.model`, `run.parent_run`, `run.parent_step`, `run.environment`, `run.algorithm`, `run.learning_rate`, `run.kl_beta`, `run.prompts_per_update`, `run.rollouts_per_prompt`, `run.max_updates`, `run.lora_rank`, `run.training_binding`, `run.inference_binding`, and the measure `run.duration_seconds`. The `update` entity (for the group-policy job kinds in `telemetry.GROUP_POLICY_JOB_KINDS` and the other training kinds) gets dimensions `update.step` and `update.time` and measures `reward`, `reward_std`, `entropy`, `kl`, `grad_norm`, `loss`, `update_seconds`, `rollout_seconds`, `actor_seconds`, `truncation_rate`, `zero_spread_share`, `generated_rows`, `generation_rounds`, `correction_clamp_share`, `speculative_acceptance`, and for SAMPO `episode_advantage`, `turn_advantage`, `anchor_group_size` and `step_reward_share` (one minus `train/rl/sparse_reward_projection_fraction`); each maps to the metric names the telemetry definitions already use. The `rollout` entity (training and evaluation kinds, when trace facts are available) gets dimensions `rollout.task`, `rollout.task_type`, `rollout.prompt_group`, `rollout.step`, `rollout.truncated`, `rollout.failed` and `rollout.model` and measures `rollouts`, `rollout_reward`, `input_tokens`, `output_tokens`, `thinking_tokens`, `tool_calls`, `model_calls` and `rollout_latency_ms`. The `eval_task` entity gets dimensions `eval_task.task` and `eval_task.facet.<name>` (one per declared facet) and measures `task_reward`, `success_rate`, `valid_repetitions`, `execution_failures` and `truncations`. The `load_level` entity gets dimensions `load_level.concurrency` and `load_level.context_tokens` and measures `throughput` (aggregate output tokens per second), `failure_rate`, `completed_requests`, `output_tokens_mean` and `output_tokens_p95`. Metrics: `rollout_share = sum(rollout_seconds) / sum(update_seconds)` and `actor_share` on `update`; `pass_rate = mean(success_rate)` weighted as the evaluation measurement defines on `eval_task`. Descriptions and units come from `_METRIC_HELP` where a metric has help. A test asserts that every metric named in a `JobTelemetryDefinition` summary field or chart for these job kinds is the source of some measure, so the two vocabularies cannot drift.

Milestone 2 adds the query compiler and executor in `semantic_layer/query.py` and `semantic_layer/execute.py`. `SemanticQuery` has `measures` (names of measures or metrics, each optionally `name:aggregation`), `by` (dimensions), `where` (a mapping from dimension to a value, a list of values meaning "any of", or a comparison string such as `">= 10"`; string values may use `*` wildcards), `runs` (shorthand for `where` on run dimensions: explicit run ids, or a mapping of run dimensions), `order_by`, `limit` (default 1,000 rows). Execution resolves runs first: `list_runs` with the filters it can push down (work package, job kind, status), then `get_run` for runs that need setting dimensions (cached per query), then applies the remaining run filters; more than 50 runs is an error that asks for a narrower filter (a setting, not a constant). It then decides the grain, checks that every measure is provided by the job kinds of the selected runs (a run that lacks a measure contributes nulls and is listed in `unavailable` with the reason; a measure no selected run provides is an error), and fetches only what is needed: metric series in batches of at most 12 names at up to 2,000 points for `update` measures, one trace-fact aggregation per run grouped by the requested rollout dimensions for `rollout` measures, the evaluation or serving view for `eval_task` or `load_level` measures, and resolved inputs for setting dimensions. The fetched data is written into a fresh in-memory SQLite database as tables `runs`, `updates`, `rollouts`, `eval_tasks` and `load_levels` with semantic column names, and the query is compiled into one generated `SELECT` with `GROUP BY` over the `by` dimensions and the chosen aggregations. The result is `SemanticResult`: `columns` (name, kind `dimension|measure|metric`, type, unit, label), `rows`, `grain`, `sources` (for each measure, the source names and runs read), `downsampled` (true when any series was downsampled) and `unavailable`.

Milestone 3 adds raw SQL mode in `semantic_layer/sql.py`. A `SqlQuery` has `sql` (one `SELECT`, optionally with `WITH`), `runs` (the same scope as above, required) and `load` (which entities and which of their measures to load; default: `runs` plus every entity the SQL text names, with all measures the selected runs provide). The same tables are loaded, then the SQL runs under an SQLite authorizer that allows only `SQLITE_SELECT`, `SQLITE_READ` and `SQLITE_FUNCTION` and denies everything else (`ATTACH`, `PRAGMA`, writes, temporary objects), a progress handler that aborts after a configurable time (default 5 seconds), and a row cap (default 10,000). Errors are returned as the SQLite message with the query position. The result has the same `SemanticResult` shape with column kinds `value`.

Milestone 4 adds `describe` in `semantic_layer/describe.py`: given runs (the same scope) or job kinds, return the entities, dimensions, measures and metrics that are available for them, with descriptions, units, allowed aggregations and which runs provide each; and the SQL table schemas. This is what agents call first.

Milestone 5 exposes the layer. `ObservatoryService` gains `describe_semantics(scope)` and `query_semantics(query: SemanticQuery | SqlQuery)`. `http.py` adds `GET /api/v1/semantic/model` (the whole framework model), `POST /api/v1/semantic/describe` and `POST /api/v1/semantic/query`. `mcp.py` adds read tools `describe_semantics` and `query_semantics`, whose descriptions tell an agent to describe before querying. The CLI adds `posttrain query` in a new `apps/cli/src/posttrain_cli/commands/query_cmd.py`: `--measures`, `--by`, `--where key=value` (repeatable), `--runs`, `--sql`, `--file <query.yaml>`, `--format table|csv|json`, and `posttrain query describe [--runs ...]`. Regenerate `openapi.json` and `mcp-schema.json`.

Milestone 6 is acceptance on the real project, described under Validation and Acceptance.

## Concrete Steps

From `/home/hammad/projects/rl-perf-guard` (or a fresh worktree of `main` on a new branch):

    uv sync --all-packages --locked --python 3.13 --group dev --extra trl --extra verifiers
    uv run pytest -q apps/observatory/tests -k semantic
    uv run ruff check . && uv run pyright && uv run lint-imports && uv run pytest
    uv run posttrain query describe --runs lfm12-sampo-turns-8gb-20260926-r1
    uv run posttrain query --measures update_seconds:mean --by run --where run.id=lfm12-sampo-turns-8gb-20260926-r1

Expected output of the last command:

    run.id                              update_seconds (mean, s)
    lfm12-sampo-turns-8gb-20260926-r1   195.5

## Validation and Acceptance

Unit tests use a fake `RunDataSource` with fixed runs and series. They must show: the model validator rejects duplicate names and formulas across entities; a query on `update` measures fetches only the named series (the fake records calls); grain selection picks `update` for `by: [run, update.step]` and `run` for `by: [run]`; a measure one run lacks yields nulls and an `unavailable` entry; a measure no run provides is an error; more than 50 runs is an error; rollout measures call `aggregate_trace_facts` with the mapped group-by dimensions and never download traces; SQL mode returns rows for a `SELECT` with a window function and rejects `ATTACH`, `PRAGMA`, `INSERT`, `CREATE TEMP TABLE` and a query that runs past the time limit; the telemetry coverage test passes.

On the real project, these queries must return values that match Trackio (checked by reading the same series directly): mean `update_seconds` by run for `lfm12-sampo-turns-8gb-20260926-r1` is 195.5 and for `lfm12-sampo-8gb-20260926-r15` is 255.4; `entropy` and `kl` by `run, update.step` for `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` reproduce its logged series; `rollout_share` by run for the 8 GB SAMPO runs r13, r14 and r15 is above 0.8; `task_reward` and `success_rate` by `eval_task.task` for `eval-lfm26-heldout-64k-base-20260924-r1` reproduce the Observatory evaluation view; `rollouts` and `rollout_reward` by `rollout.truncated` for `lfm12-screen-8k-20260926-r2` show 33 truncated rollouts (its trace facts were recorded under fact calculator v6, before context-overflow rollouts counted as truncated); `throughput` by `load_level.concurrency` for a serving benchmark run reproduces its capacity view. The MCP tool `query_semantics` returns the same table as the CLI for one of these queries.

## Idempotence and Recovery

Everything in this plan is read-only and stateless per query: each query builds and discards its own in-memory database. Re-running any command is safe. There is no migration and no deployment step beyond releasing the Observatory as usual.

## Artifacts and Notes

The first consumers are the run notes (`docs/plan/run-notes.md`), whose data blocks are semantic queries or SQL, and agents answering questions through MCP. Moving the Observatory's own views and `JobTelemetryDefinition` onto the semantic model is a later step, not part of this plan.

## Interfaces and Dependencies

In `apps/observatory/src/posttrain_observatory/semantic_layer/model.py`:

    type Aggregation = Literal["last", "first", "min", "max", "mean", "sum", "count", "stddev"]
    type SourceKind = Literal["run_field", "setting", "metric_series", "trace_fact", "eval_view", "serving_view", "derived"]

    class Source(ObservatoryModel): kind: SourceKind; name: str
    class Entity(ObservatoryModel): name: str; description: str; parent: str | None
    class Dimension(ObservatoryModel): name: str; entity: str; type: Literal["string", "integer", "number", "time", "boolean"]; description: str; source: Source
    class Measure(ObservatoryModel): name: str; entity: str; label: str; description: str; unit: str | None; source: Source; aggregation: Aggregation; allowed: tuple[Aggregation, ...]; job_kinds: tuple[str, ...]
    class Metric(ObservatoryModel): name: str; entity: str; label: str; description: str; unit: str | None; formula: str
    class SemanticModel(ObservatoryModel): entities: tuple[Entity, ...]; dimensions: tuple[Dimension, ...]; measures: tuple[Measure, ...]; metrics: tuple[Metric, ...]

In `semantic_layer/query.py`:

    class SemanticQuery(ObservatoryModel):
        measures: tuple[str, ...]
        by: tuple[str, ...] = ()
        where: dict[str, JsonValue] = {}
        runs: tuple[str, ...] | dict[str, JsonValue] | None = None
        order_by: tuple[str, ...] = ()
        limit: int = 1000

    class SqlQuery(ObservatoryModel):
        sql: str
        runs: tuple[str, ...] | dict[str, JsonValue]
        load: dict[str, tuple[str, ...]] | None = None

    class SemanticResult(ObservatoryModel):
        columns: tuple[ResultColumn, ...]
        rows: tuple[tuple[JsonValue, ...], ...]
        grain: str
        sources: dict[str, tuple[str, ...]]
        downsampled: bool
        unavailable: tuple[str, ...]

In `service.py`:

    async def describe_semantics(self, scope: SemanticScope) -> SemanticDescription: ...
    async def query_semantics(self, query: SemanticQuery | SqlQuery) -> SemanticResult: ...

Dependencies: Python's standard `sqlite3` only; no new packages.
