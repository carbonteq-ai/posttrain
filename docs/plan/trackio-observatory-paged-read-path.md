# Bound Trackio reads consumed by Observatory

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current as the implementation changes.

## Purpose / Big Picture

Observatory should be able to open a project, inspect a run, chart selected metrics, and browse evaluation traces without downloading the entire Trackio project or run into the web process. Today the Trackio client exposes whole-history reads, and the post-training adapter compensates by downloading all traces and then filtering and paginating in Python. That makes a small dashboard request proportional to the oldest run's stored telemetry, which is especially damaging for Verifiers rollouts.

This change establishes a bounded read path. Trackio will accept explicit page bounds for run discovery, metric history, and system history; its SQLite and Doris stores will apply those bounds in the database; the Trackio client will expose them without changing the existing unbounded convenience methods; and the post-training adapter will pass filters before paging. Observatory will therefore receive only the rows needed for the requested view. The result is observable through regression tests that assert provider calls are bounded and through a live API probe against the configured Trackio server when credentials are available.

The work spans two repositories with separate ownership. Generic storage and API changes belong to `/home/hammad/projects/trackio`. The normalized adapter and Observatory-specific behavior belong to `/home/hammad/projects/rl`. Dependency pins will not be changed in this slice until the Trackio fork change is committed and published.

## Progress

- [x] (2026-08-07) Read the canonical observation/API contracts and the plan template.
- [x] (2026-08-07) Traced the current Trackio client, server, SQLite/Doris storage, adapter, and Observatory trace/metric call paths.
- [x] (2026-08-07) Add bounded history/system-history and trace-count contracts to the Trackio fork.
- [x] (2026-08-07) Make SQLite and Doris apply `LIMIT`/`OFFSET` before materializing rows, while preserving existing convenience behavior.
- [x] (2026-08-07) Make the Trackio adapter page traces with the requested Verifiers filter, use a bounded list payload, and avoid unbounded `get_run` trace reads.
- [x] (2026-08-07) Add provider-side metric/system key projection and remove duplicate `get_run` reads when resolving job views.
- [x] (2026-08-07) Add regression tests in both repositories and run package-specific validation.
- [x] (2026-08-07) Exercise the live remote API and record that the installed server is post10 and does not expose the new contract.
- [x] (2026-08-07) Start a local Trackio process from the working tree against the live Doris FE through an authenticated, loopback-only SSH tunnel; point local Observatory at that Trackio process.
- [x] (2026-08-07) Exercise the local Doris-backed path in the in-app browser across run-list, SFT, evaluation, serving, GRPO, trace-summary, and trace-detail surfaces.
- [x] (2026-08-07) Bound run-detail event reads and derive the metric catalog from the Trackio run summary instead of scanning the full history.
- [x] (2026-08-07) Make the Observatory run shell paint from the high-level run list before the selected view payload resolves, and keep system metrics, evaluation, and generic metric workspaces behind their section boundaries. Artifact/config payload splitting remains a follow-up because the current view contract still carries those fields for compatibility.
- [x] (2026-08-07) Stop opening the first trace detail automatically; trace-list summaries are fetched when the trace section opens and the full transcript is fetched only after an explicit trace selection.
- [x] (2026-08-07) Isolate the remaining large-run delay to Trackio client/server revision skew: starting the working-tree Trackio server was insufficient while Observatory still imported the installed post10 client and therefore materialized every trace to compute `trace_count`.
- [x] (2026-08-07) Relaunch the local Observatory process with the same Trackio working tree on its import path and remeasure the large RTX PRO GRPO overview and trace tab independently.
- [ ] Decide whether to publish/pin the Trackio fork; no release or production switch is authorized by this plan alone.
- [x] (2026-08-07) Split trace population aggregates from the paged trace table. Optimization jobs use their recorded `train/rl/*` population aggregates and fetch only one 100-row summary page; evaluation jobs can request aggregate-only evidence independently.
- [x] (2026-08-07) Add cursor pagination, explicit load-more behavior, and exact-detail click-through to the Observatory HTTP, MCP, service, and frontend contracts.
- [x] (2026-08-07) Reuse the bounded run metadata already loaded for Overview for trace pages and detail reads, with a 60-second expiry for live runs and a five-minute expiry for terminal runs.

## Surprises & Discoveries

- Observation: `TrackioDataSource.traces` calls `provider_run.traces(limit=1000, ...)` repeatedly until the entire run is loaded, then filters trace type and slices in memory.
  Evidence: `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`, `TrackioDataSource.traces`.
- Observation: Trackio's existing history `max_points` is not a database bound. Both SQLite and Doris select every row and subsample a Python list afterward.
  Evidence: `sqlite_storage.py::_fetch_metric_logs_with_cursor` and `doris_storage.py::get_logs`.
- Observation: run listing loads all provider runs and whole-project config/lifecycle maps before applying `RunQuery.limit`; the apparent constant-cost API is only constant in request count, not in bytes or server work.
  Evidence: `trackio/api.py::Runs._load_runs`, `TrackioDataSource._list_runs`.
- Observation: the existing provider-neutral `TraceQuery` already has a cursor and limit, so the first adapter fix can preserve the public Observatory contract while correcting the provider call order.

- Observation: a trace list and a trace detail have different payload requirements. Trackio stores large transcript/tool-call bodies in the same trace row as rewards and outcome fields.
  Evidence: `TraceRecord` consumers in `apps/observatory/src/posttrain_observatory/traces.py` and Trackio's `traces` table columns `messages`, `metadata`, and `payload`.
- Observation: ordinary post-training trace kinds are encoded as physical Trackio type `trackio` and carry their semantic kind in metadata; only Verifiers can currently be filtered by the physical `trace_type` SQL predicate.
  Evidence: `TrackioDataSource._normalize_trace_record` and the adapter writer path.
- Observation: the live server reports `0.31.5.post10`; a GET page probe returned HTTP 405, so the new optional parameters are not yet deployed.
  Evidence: `curl https://trackio.lan/version` returned version `0.31.5.post10`.
- Observation: job views have distinct evidence shapes: supervised/RL jobs primarily need declared metric series; evaluation and GRPO/SAMPO/distill jobs additionally scan Verifiers summaries; serving benchmarks scan inference summaries and metrics; generic views need a metric catalog and only user-selected series.
  Evidence: `telemetry.py` job definitions, `service.py::get_run_view_response`, and `serving_capacity.py::_request_evidence`.
- Observation: the service fetched the same `RunDetail` twice for normal job and generic views before building the view.
  Evidence: `get_run_view_response` loaded detail and `_metric_job_view`/`_generic_view` loaded it again.
- Observation: the first real GRPO run view is still dominated by downstream evaluation aggregation even after trace list pages are summary-shaped. On the local Doris-backed path, the 5,248-trace run rendered in approximately 48 seconds; a 256-trace GRPO run rendered in approximately 7 seconds. The trace-evaluation scan alone took about 17 seconds for its 5,000-record safety bound.
  Evidence: browser timings and a direct `trace_evaluation_view` timing against the local Trackio/Doris process on 2026-08-07.
- Observation: SFT and serving views are much cheaper because they do not scan thousands of Verifiers traces: browser selection-to-render was approximately 3.1 seconds for each. The generic `eval.model` view took approximately 27 seconds in the first browser pass, reflecting run-detail/provider round trips rather than a specialized trace projection.
  Evidence: in-app browser selection timings against `http://127.0.0.1:7861`.
- Observation: the local Observatory process was initially pointed at `https://trackio.lan`; its API calls stalled while the dashboard HTML remained available. Repointing it to local Trackio at `http://127.0.0.1:7863`, with Trackio using Doris through the loopback tunnel, made readiness and run-list requests responsive.
  Evidence: `/health/ready`, local Trackio `get_all_projects`/`get_runs_for_project`, and browser render after the restart.
- Observation: the first local implementation still issued a full trace-detail read immediately after the trace population arrived, even though the user had not selected a trace. Removing that eager read leaves the trace tab summary-only until click-through.
  Evidence: `apps/observatory/frontend/src/App.tsx::openSection` and the exact-detail `api.trace` callback passed to `TraceView`.
- Observation: the current `RunView` contract includes artifacts and resolved configuration alongside the overview metric projection, so those fields are still transferred during the background overview request even though the dedicated tabs are not opened. Splitting them requires an additive tab payload contract rather than silently making existing fields nullable.
  Evidence: `apps/observatory/src/posttrain_observatory/models.py::RunView`, `service.py::_metric_job_view`, and the `ArtifactsView`/`ConfigView` consumers in `App.tsx`.
- Observation: a Trackio server process and an Observatory process can point at the same Doris data while using different Trackio Python revisions. In that state the old client has no `Run.trace_count`; the adapter compatibility branch calls `len(provider_run.traces())`, downloading the complete 16,767-trace run before the overview can render. The measured `get_run` time was approximately 55.7 seconds, while the bounded metric-series and artifact reads took approximately 0.26 seconds and 0.13 seconds respectively.
  Evidence: live profiling of `TrackioDataSource.get_run` against the RTX PRO GRPO run, plus inspection of the installed post10 `Run` interface and the Trackio working-tree interface on 2026-08-07.
- Observation: after Observatory imported the same Trackio working tree as the local server, the large GRPO overview measured approximately 2.1 seconds and 630 KB over HTTP; a cold browser selection rendered in approximately 4 seconds. The prior 52-62 second behavior therefore came from the unbounded compatibility fallback, not the overview metric history query.
  Evidence: direct `/view?mode=auto` timing and in-app browser selection against `http://127.0.0.1:7861` on 2026-08-07.
- Observation: the dedicated rollout view remains oversized even on the corrected client. It scans its 5,000-record safety bound, produces 512 slice rows, and returns all 5,000 trace summaries in one 11.2 MB response; a warm direct request took approximately 11.1 seconds. Virtualized rendering limits DOM nodes but does not reduce Doris work, Python projection, JSON serialization, or transfer size.
  Evidence: direct `/traces-evaluation` timing and response-shape inspection for the RTX PRO GRPO run on 2026-08-07.
- Observation: GRPO/DAPO already records population-level reward, variance, truncation, throughput, and update evidence as `train/rl/*` metrics. Recomputing evaluation slices across thousands of training rollouts duplicated Overview evidence, generated one slice per dataset example, and made an optimization investigation tab behave like an evaluation report.
  Evidence: the GRPO telemetry definition, the 512 generated slice rows from the large run, and the canonical rule that bounded `train/rl/*` aggregates back dashboards while native traces remain replay authority.
- Observation: after separating the contracts, the large GRPO run's first 100-row trace page measured approximately 0.21 seconds and 219 KB after Overview populated the short-lived metadata context; the next 100-row page measured approximately 0.31 seconds and 220 KB. The old response was approximately 11.2 MB for 5,000 rows.
  Evidence: direct HTTP timings for `/traces?limit=100` and `/traces?limit=100&cursor=100` against the live local Doris-backed run on 2026-08-07.
- Observation: an aggregate-only evaluation response removes raw trace rows from browser transfer but still costs approximately 12 seconds and 227 KB on this 16,767-rollout training run because Observatory computes 5,000-record evaluation-style slices. Optimization jobs no longer issue that request. A future large evaluation population may still justify provider-native aggregate projection, but it is no longer on the GRPO/DAPO page-load path.
  Evidence: direct `/traces-evaluation?include_traces=false` timing and the browser/server request log showing no evaluation request when opening the optimization trace tab.

## Decision Log

- Decision: Keep current convenience methods backward compatible and add optional page arguments rather than silently changing `Run.history()` or `Run.system_history()` semantics.
  Rationale: external Trackio consumers may rely on those methods returning all rows; Observatory needs an explicit bounded path.
  Date/Author: 2026-08-07 / Codex.
- Decision: Use deterministic timestamp-plus-storage-id ordering for history pages and the existing numeric offset cursor for traces in this slice.
  Rationale: traces already support provider-side `limit`/`offset`; history rows currently expose no public cursor identity, and a bounded offset API is the smallest compatible repair. Keyset cursors remain a follow-up if production measurements show deep-page latency.
  Date/Author: 2026-08-07 / Codex.
- Decision: Do not redesign the frozen post-training observation models or move generic storage logic into Observatory.
  Rationale: Trackio owns provider pagination; Observatory owns normalized views and should stay backend-neutral.
  Date/Author: 2026-08-07 / Codex.
- Decision: Make trace list reads summary-shaped and reserve full transcript/tool-call payloads for an exact trace-detail read.
  Rationale: the list and evaluation views need reward/outcome/task evidence, while a click-through needs the complete record. Sending full payloads for every list row is the main source of pressure.
  Date/Author: 2026-08-07 / Codex.
- Decision: Keep job-specific projection logic in Observatory telemetry definitions, but make the provider read seam generic: declared metric names become provider-side `keys`, and trace pages remain summary-shaped for evaluation/serving readers.
  Rationale: different jobs need different evidence, but the transport must not fetch unrelated metric keys or full trace bodies for any of them.
  Date/Author: 2026-08-07 / Codex.
- Decision: Treat `RunDetail` as metadata plus a bounded event sample. Read the metric catalog from `Run.summary()` and at most 256 event-key rows; do not use full history to discover names. Full trace payloads remain an exact-detail operation.
  Rationale: the run page needs catalog/lifecycle context before it can render, while full metric and transcript bodies belong to their dedicated views.
  Date/Author: 2026-08-07 / Codex.
- Decision: Render the run identity from `RunItem` first and let the selected section request its payload independently. Do not make the sidebar wait for the job projection, and do not fetch a full trace until the user selects one.
  Rationale: run identity is small and common to every job family; metric series, system samples, rollouts, artifacts, and configuration have different costs and should not block or overfetch one another.
  Date/Author: 2026-08-07 / Codex.
- Decision: Keep artifact/config fields in the compatibility view for this slice, but record a separate tab-payload seam as the next API change. Do not claim they are provider-lazy until `/artifacts` and `/config` (or equivalent capability-scoped endpoints) are wired through the frontend.
  Rationale: the current models make those fields required and several job views use them for lineage/context; an additive endpoint avoids a partial response that looks complete but is not.
  Date/Author: 2026-08-07 / Codex.
- Decision: Treat the Trackio read client and server as one versioned capability boundary. A local or production deployment is not qualified merely because the new server is running; Observatory must import the matching client and prove `Run.trace_count` and bounded history arguments before serving large runs.
  Rationale: the compatibility fallback is semantically correct but changes an O(1) metadata read into an O(number of traces) transfer without an externally visible warning.
  Date/Author: 2026-08-07 / Codex.
- Decision: Split the trace-tab contract into provider-side aggregate evidence and a small paged summary table, while preserving exact trace detail as a third click-through request. Do not lower the 5,000 safety bound and present a smaller sample as if it were a complete population.
  Rationale: pagination fixes transfer and browser work; provider-side aggregation fixes database and Python work. Both are required to keep whole-population statistics honest and make the first table page fast.
  Date/Author: 2026-08-07 / Codex.
- Decision: For `train.grpo`, `train.sampo`, and `train.distill`, treat the operation's recorded scalar metrics as the population aggregate contract and the paged traces as investigation evidence. Do not invoke `TraceEvaluationView` merely to open an optimization trace tab.
  Rationale: training already emits bounded full-population aggregates; evaluation slice semantics and success predicates belong to evaluation jobs. This preserves honest population metrics while avoiding an unnecessary trace scan.
  Date/Author: 2026-08-07 / Codex.
- Decision: Keep aggregate-only evaluation and paged summary reads as separate API calls. An evaluation can progressively render its first trace page while whole-population aggregates resolve, and a training job can omit the evaluation request entirely.
  Rationale: page browsing, whole-population interpretation, and exact transcript inspection have different cost and payload shapes and should not block one another.
  Date/Author: 2026-08-07 / Codex.

## Outcomes & Retrospective

The local Trackio fork now supports bounded history/system reads, provider-side key projection, SQL trace counts, summary-shaped trace pages, and exact trace payload reads. Observatory's job view path reuses the already-loaded `RunDetail`; declared telemetry keys are passed down by the Trackio adapter, and run-detail metadata no longer scans unbounded history. The frontend now paints a high-level run shell before the selected projection resolves, keeps section payloads lazy, and only fetches a complete trace after explicit selection. Focused Trackio, adapter, and Observatory tests pass. A real local Trackio process connected to the deployed Doris FE through a loopback tunnel was exercised in the browser.

The large-run timeout has now been reproduced and removed from the local preview by making Observatory import the same Trackio working tree as the local Trackio server. The old client/server mismatch remains a production release gate: the deployed post10 pair lacks the bounded contract, and there is no published immutable Trackio revision or updated framework pin.

The trace-tab gate is complete for optimization jobs. GRPO/DAPO no longer runs the 5,000-record evaluation projection. It reads full-population learning evidence from the run's recorded scalar metrics, requests 100 summary rows at a time, merges pages explicitly in the UI, and requests one complete transcript only after selection. Against the 16,767-rollout RTX PRO run, a warmed first page was approximately 0.21 seconds and 219 KB and the next page was approximately 0.31 seconds and 220 KB, versus the former approximately 11.1-second, 11.2 MB response. Aggregate-only evaluation remains independently available and honest; very large evaluation jobs may still need a provider-native aggregation endpoint if live measurements show the current bounded scan is too slow.

## Context and Orientation

The framework's read contract is `RunDataSource` in `packages/tracking/src/posttrain/tracking/contracts.py`. It exposes `list_runs`, `get_run`, `metric_series`, `traces`, and `artifacts` to job-aware views. `RunQuery` and `TraceQuery` are provider-neutral models in `packages/tracking/src/posttrain/tracking/models.py`; the latter already represents a page using `cursor` and `limit`.

The generic Trackio client is in `/home/hammad/projects/trackio/trackio/api.py`; its `Run.history`, `Run.system_history`, `Run.traces`, and `Runs._load_runs` methods translate Python calls into Trackio HTTP endpoints or local storage calls. `Run.traces(include_payload=False)` is the summary path; `include_payload=True` is reserved for one detail record. The server functions are in `/home/hammad/projects/trackio/trackio/server.py`. SQLite and Doris implementations are in `sqlite_storage.py` and `doris_storage.py`. Trackio's trace SQL already applies `LIMIT` and `OFFSET`, but its metric and system log readers materialize all rows before sampling.

The post-training adapter is `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`. It turns Trackio rows into strict framework models. `apps/observatory/src/posttrain_observatory/service.py` and `traces.py` consume that adapter and calculate views; they must not know whether the source is Trackio or another backend.

“Bounded” means a request supplies a maximum number of rows and the database applies that maximum before decoding JSON or constructing Python objects. “Provider-side filtering” means the filter, such as `trace_type`, is sent to Trackio's SQL endpoint before the page is selected.

## Plan of Work

First, add optional `limit`, `offset`, and `keys` parameters to Trackio history/system-history endpoints and storage methods. The default remains the current full-history behavior. SQLite queries will order by timestamp and the stable integer row id, then append SQL `LIMIT` and `OFFSET`; Doris will use timestamp and event id with the same ordering. The storage helpers will decode only the returned page and project requested metric keys before the response leaves Trackio. Server and client wrappers will validate non-negative offsets and bounded positive limits.

Second, add a page-aware run listing path. The first implementation may expose `limit` and `offset` on `/get_runs_for_project` and retain the existing full-project config/lifecycle endpoints for compatibility. The adapter will use the page of provider run records and fetch only the matching metadata if the Trackio server supports a run-id filter; otherwise it will use the current maps as a compatibility fallback and record that fallback in tests. This keeps the first production rollout safe while making the primary path bounded.

Third, repair `TrackioDataSource.traces` so each request sends `trace_type` when it maps to Trackio's physical Verifiers type, `limit`, and the current offset to Trackio and fetches only one summary-shaped page. `TrackioDataSource.get_trace` sends an exact external-ID search with `include_payload=True` for click-through detail. The Observatory trace utility uses that method when present and retains paged fallback for other sources.

Fourth, reduce `TrackioDataSource.get_run` and `metric_series` pressure. `get_run` uses the SQL count endpoint rather than materializing trace payloads. The exact-detail path is separate and returns one complete trace. Metric-series reads will use bounded history pages and retain the existing normalized output; Observatory's `max_points` and step window remain the user-facing controls.

Finally, add regression tests around provider call arguments, storage SQL page sizes, and backward compatibility. Run local tests, then make a read-only live request to `/version` and a small page request on the configured server. Publishing the Trackio fork and updating `uv.lock` is a separate release step after these tests pass.

## Concrete Steps

Run all commands from `/home/hammad/projects/trackio` for Trackio changes and `/home/hammad/projects/rl` for framework changes. Before editing, capture `git status --short` in each repository and do not modify unrelated files.

Use `apply_patch` for hand-authored changes. Start with the Trackio tests covering `Run.history`, `Run.system_history`, `get_runs_for_project`, and local server calls. Add tests that call a page with `limit=2, offset=2` and verify the fake server receives those values. Add storage tests with more rows than the limit and verify only the requested rows are decoded.

Then add adapter tests that use a fake provider run whose `traces` method records keyword arguments. A request for `TraceQuery(trace_type="verifiers", cursor="4", limit=2)` must produce one provider call with `trace_type="verifiers", offset=4, limit=2, include_payload=False`, return summary-shaped items, and set the next cursor only when the page is full. A detail request must produce one exact-ID call with `include_payload=True` and preserve the full payload.

Run the smallest tests first, then the repository checks described below. If the remote server is still post10 and does not implement a new endpoint, the old behavior must remain available through compatibility defaults; do not point a production Observatory instance at an untested server.

## Validation and Acceptance

From `/home/hammad/projects/trackio`, run the focused Trackio unit tests for API and storage reads. They must pass, including new page-bound tests. From `/home/hammad/projects/rl`, run the focused tracking adapter and Observatory service tests. The new trace test must fail against the old implementation because it observes repeated `limit=1000` calls and pass after the change.

The behavioral acceptance is: a trace list request makes exactly one provider call for one summary page; its response contains reward/outcome fields but no transcript body; clicking a trace makes one exact-ID provider call and returns the complete payload; a history page causes SQL to return no more than the requested rows; and existing no-argument `Run.history()`/`Run.system_history()` calls still return the complete logical history. A live compatibility probe should show HTTP 200 for `/version`; if the deployed server lacks the new optional parameters, the probe is a release blocker rather than a silent compatibility claim.

After focused tests pass, run `uv run ruff check` and the relevant `uv run pytest` selections in both repositories. In `/home/hammad/projects/rl`, also run `uv run lint-imports` if imports changed. Full lockfile/package updates are not part of this implementation slice.

## Idempotence and Recovery

All changes are additive API parameters with defaults matching the old behavior, so tests and local servers can be restarted safely. If a storage query fails on an older schema, preserve the existing no-table handling and fall back to the old unbounded method only for that backend, with a test and explicit warning. Do not delete Trackio data, caches, or artifacts while implementing this plan. If a live server rejects a new parameter, leave production unchanged and publish the fork only after the server release and framework pin are coordinated.

## Artifacts and Notes

The durable artifact is this plan plus focused regression tests in the owning repositories. Validation evidence from this slice:

    /home/hammad/projects/trackio: focused API/storage tests passed (31 tests in the latest API/storage run).
    /home/hammad/projects/rl: 29 Trackio adapter tests and 45 Observatory service/product tests passed together (74 tests); the final adapter change still passes the same 74-case selection.
    pyright on the changed adapter and trace projection: 0 errors.
    live /version: {"version":"0.31.5.post10","api_version":1,"api_transport":"http",...}
    live GET page probe: HTTP 405, so the new optional read arguments are not deployed.
    local Doris-backed browser evidence (2026-08-07): run list API ~0.24 s; SFT ~3.1 s; serving benchmark ~3.1 s; 256-trace GRPO run ~6.7 s plus ~8.2 s to open the trace population; one trace detail ~3.1 s; the 5,248-trace GRPO run ~48 s to render its full run view.

The live response is intentionally abbreviated and contains no credentials.

## Interfaces and Dependencies

The final Trackio client interface must preserve `Run.history(keys=None, scalar_only=False)` and add optional page arguments without changing defaults; similarly for `Run.system_history`. `Run.traces` must accept `include_payload`, and `Run.trace_count` must be an aggregate read. Server endpoints `/get_run_history`, `/get_system_logs`, `/get_traces`, `/get_trace_count`, and `/get_runs_for_project` must accept validated optional bounds. `TrackioDataSource.traces` must continue returning `posttrain.tracking.TracePage` and use `TraceQuery`'s cursor/limit; its optional `get_trace` method is the exact-detail seam used by Observatory. SQLite and Doris remain interchangeable implementations of the same logical results. The framework package continues to depend on the immutable Trackio fork pin; that pin is updated only in a later release step after the fork commit is available.

Plan revision note (2026-08-07): initial plan created after tracing the unbounded Trackio-to-Observatory read path; scope deliberately separates generic Trackio pagination from framework adapter behavior. Updated the plan after clarifying that trace list pages must be summary-only while click-through detail loads one complete trace.
