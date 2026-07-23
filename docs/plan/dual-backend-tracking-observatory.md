# Deliver dual-backend tracking and a job-aware observatory

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. The
canonical product authority is `docs/post-training/README.md` and the six
documents it indexes. This plan introduces a narrow amendment to their current
Trackio-first observation wording; it does not change the project → work package
→ run-of-job hierarchy or the meaning of training, evaluation, serving, traces,
and artifact lineage.

## Purpose / Big Picture

After this work, an ML expert can run the same SFT, DPO, GRPO, or evaluation job
with either Trackio or Weights & Biases (W&B) as the tracking backend and inspect
the result through one dedicated post-training read product called Observatory.
Observatory replaces the parallel reports package and owns Python analysis,
report exports, HTTP, MCP, and the frontend over one query/intelligence service.
It shows the metrics, health conditions, traces, and artifacts relevant to each
job kind instead of presenting a generic list of logged keys.

The first slice is demonstrated without a GPU by writing a synthetic run through
each backend, reading it back through that backend, and comparing the normalized
logical views. It is also demonstrated by opening the observatory, selecting an
SFT or evaluation fixture run, and seeing the job-specific summary and charts,
then calling the corresponding MCP tools and receiving the same summary fields
and alerts. Trackio is local-first. The W&B end-to-end reader test requires a
real synced W&B test project and credentials because W&B's stable Public API
does not query unsynced offline run directories.

## Progress

- [x] (2026-07-22 01:26Z) Surveyed the canonical observation contract, current
  `Observer`, Trackio adapter, report queries, Trackio fork APIs/MCP/frontend,
  W&B writer/Public API semantics, and adjacent repository layout.
- [x] (2026-07-22 01:26Z) Added root `AGENTS.md` with canonical documentation,
  planning, repository, dependency, and validation guidance.
- [x] (2026-07-22 01:45Z) Amended the frozen post-training observation
  documents narrowly for a provider-neutral tracking backend and shared
  job-aware views while retaining Trackio as the local default.
- [x] (2026-07-22 01:45Z) Implemented `posttrain-tracking` lifecycle contracts,
  normalized strict read models, SFT/DPO/GRPO/eval telemetry definitions, and
  the job-aware view service; migrated the lab `RunSpec` boundary to the new
  provider-neutral artifact reference with a compatibility re-export.
- [x] (2026-07-22 02:10Z) Improved the public query surface in the adjacent
  Trackio fork, committed and pushed it at
  `f52d5c34dac45f803f44fdf6fac21658810afe3b`, pinned that revision, and
  implemented the Trackio writer and normalized reader adapter with Turso and
  SQLite conformance coverage.
- [x] (2026-07-22 02:57Z) Completed the W&B writer/reader and shared
  conformance slice. Real Cloud coverage now proves nested config, logical
  steps, events, standard and Verifiers traces, input/output artifact lineage,
  success/failure/partial/cancelled/unsupported outcomes, and idempotent finish.
- [x] (2026-07-22 02:57Z) Migrated lab tracked execution to a selected
  `TrackingBackend`, retained Trackio as the CLI default, added W&B selection,
  finalized canonical outcomes on successful, failed, and interrupted exits,
  and exercised synthetic jobs through both adapters.
- [x] (2026-07-22 03:32Z) Created the `posttrain-observatory` workspace
  distribution, moved strict telemetry definitions, job-aware view models, and
  projection service ownership out of tracking, and enforced the provider-SDK
  boundary with import-linter.
- [x] (2026-07-22 04:44Z) Added the detailed downstream product plan at
  `docs/plan/observatory-product-implementation.md`, including strict
  job-specific views, a deterministic generic metrics fallback, optional
  grounded LLM semantic summaries, unified Verifiers traces/evaluation,
  frontend behavior, transport contracts, and OCI acceptance.
- [ ] Fold `packages/reports` behavior into the dedicated
  Observatory product, move job-aware query/view ownership out of tracking,
  migrate callers and tests, then remove the reports distribution entirely.
- [ ] Implement the observatory API, MCP tools, and minimal Svelte run list and
  job-aware run detail.
- [ ] Package Observatory as one multi-stage OCI image, add local Compose,
  publish version/schema metadata, and leave Helm plus the thin client wheel as
  explicit follow-on distribution artifacts.
- [ ] Complete package, cross-backend, frontend, MCP, import-boundary, and
  end-to-end validation; record evidence and retrospective here.

## Surprises & Discoveries

- Observation: The write path is already partially abstract. Reusable
  operations emit through `packages/common/src/posttrain/common/execution.py`
  `Observer`, and `apps/lab/src/posttrain_lab/tracking/trackio_observer.py`
  performs Trackio translation. The read path is not abstract because
  `packages/reports/src/posttrain/reports/query.py` imports Trackio storage and
  `work_packages.py` queries physical tables.
  Evidence: `Observer` contains event, metric, metric-batch, trace, and artifact
  methods, while report queries select directly from `configs`, `metrics`, and
  artifact-link tables.

- Observation: W&B's physical history step is append-oriented. It accepts the
  current or next history step and does not provide arbitrary retroactive writes.
  The framework therefore cannot define its logical training step by copying a
  provider's internal row number.
  Evidence: the W&B logging contract documents consecutive current/next steps;
  the existing framework permits an explicit optional step on each metric.

- Observation: W&B offline mode is useful for writer tests but its stable
  `wandb.Api` reader addresses synced entity/project/run records. Parsing W&B's
  local offline files would introduce another private-storage dependency.
  Evidence: W&B's Public API exposes config, state, history, and artifacts by
  run path, while offline mode saves data for later synchronization.

- Observation: The Trackio fork already provides local server functions and MCP
  tools for raw project, run, metric, system, log, alert, and snapshot queries,
  but `trackio.Api` only exposes runs, config, alerts, and mutations. The generic
  public Python read API should be completed in Trackio rather than copied into
  this repository as more physical SQL.
  Evidence: sibling repository files `trackio/api.py`, `trackio/server.py`, and
  `trackio/mcp_setup.py` at commit
  `02351d871050bf4b3505c7371239c698b710ec83`.

- Observation: Trackio and W&B both already expose generic MCP capabilities.
  Those tools remain useful for provider debugging, but they do not share the
  framework's SFT/DPO/GRPO/eval projections. The observatory MCP server must be
  domain-aware and backend-neutral rather than proxy either provider MCP.
  Evidence: Trackio registers raw storage-oriented tools in
  `trackio/mcp_setup.py`; W&B offers a hosted and local provider-specific MCP
  server.

- Observation: Trackio's complete unit directory contains five explicitly
  hardware-dependent GPU tests that do not skip when NVML, PyTorch, or a usable
  GPU is absent. The provider/API work does not touch GPU collection.
  Evidence: the complete run passed 323 tests and skipped 2, while the five
  failures were all in `tests/unit/test_gpu_hardware.py`; excluding that module
  passed the same 323 tests with 2 skips.

- Observation: W&B flattens nested history dictionaries into dotted keys in
  Public API history. Requesting the original dictionary key can return no rows.
  Evidence: `event/attributes={"phase": "train"}` was returned as
  `event/attributes.phase`; the reader now reconstructs strict logical
  attributes from dotted keys.

- Observation: W&B may deduplicate an artifact with the same physical name and
  content, omitting the second run's expected output edge.
  Evidence: a producer and consumer both logging `training-qwen-adapter` with
  identical bytes left the consumer with only a used-artifact edge. W&B
  physical names are now run-scoped while `logical_name` remains stable in
  metadata.

- Observation: Adding Observatory as a workspace distribution changes the
  committed `uv.lock` digest embedded in every training selection, even though
  it does not change a trainer dependency directly.
  Evidence: the cross-package suite initially failed only the reproducibility
  guard; updating all four catalog entries to the new lock digest restored 50
  passing non-network tests.

## Decision Log

- Decision: Implement Trackio and W&B writers and readers in the first slice,
  then run one shared conformance suite against each.
  Rationale: a second real backend exposes semantic leaks in run identity,
  grouping, steps, status, nested configuration, artifacts, and failure paths
  that a fake adapter cannot reveal.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Keep the existing low-level `Observer` and `RunContext` in
  `posttrain.common`; put run lifecycle, reader protocols, normalized view
  models, telemetry definitions, and query intelligence in the new
  `posttrain.tracking` package.
  Rationale: capability packages already compile against the small observation
  protocol. Moving it would create churn without improving the backend boundary.
  Date/Author: 2026-07-22 / Codex.

- Decision: Framework `run_id`, `work_package_id`, job kind, logical artifact
  identity, logical metric step, and canonical outcome are authoritative.
  Provider run IDs, groups, row steps, states, and artifact versions are storage
  metadata translated by adapters.
  Rationale: normalized views must not change meaning when the provider changes.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Persist the canonical logical step as data. The W&B adapter will use
  a dedicated `posttrain/step` field and configure job metric series to use it as
  their axis; readers ignore provider row numbers when reconstructing
  `MetricPoint.step`. Explicit logical steps must be non-negative and
  nondecreasing within a run.
  Rationale: W&B's append-only history rows and Trackio's explicit step support
  otherwise produce observably different series.
  Date/Author: 2026-07-22 / Codex.

- Decision: In the first slice, the W&B adapter stores framework traces in an
  append-only JSONL staging file and publishes it as a typed output artifact at
  finish. The W&B reader reconstructs normalized traces from that artifact.
  Trackio retains live native `Trace` and `VerifiersTrace` records. The
  normalized reader advertises whether live traces are available.
  Rationale: the first W&B scope remains runs, config, metrics, events,
  artifacts, and finish/fail while preserving post-finish trace evidence needed
  by the common logical model. Adopting W&B Weave is a separate product choice.
  Date/Author: 2026-07-22 / Codex.

- Decision: `JobTelemetryDefinition` and all projection logic live beside the
  query service in `apps/observatory`, not in the web application frontend or
  the raw tracking-contract package.
  Rationale: the frontend, MCP tools, Python API, and comparisons must render the
  same summary fields, health rules, missing-evidence semantics, and chart
  definitions.
  Date/Author: 2026-07-22 / user.

- Decision: Build a custom backend-neutral observatory rather than extending
  Trackio's bundled frontend with post-training nouns. Generic Trackio query API
  improvements remain in the Trackio fork.
  Rationale: a Trackio-hosted post-training UI could not render W&B runs through
  the same service and would couple product views to one provider.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Observatory is the single read and analysis product. Retire
  `packages/reports` rather than retaining it as a compatibility peer, and move
  telemetry definitions plus job-aware query intelligence from tracking into
  Observatory. Tracking retains lifecycle and normalized raw evidence
  contracts.
  Rationale: reports and a thin Observatory overlap around the same ML-expert
  questions and would create competing Python APIs and view ownership. One
  product keeps Python, report exports, HTTP, MCP, and frontend interpretations
  aligned.
  Date/Author: 2026-07-22 / user and Codex. See ADR 0012.

- Decision: Distribute Observatory OCI-first as one centrally hosted or
  self-hosted product image containing the service, frontend, HTTP, MCP, report
  exports, and both provider readers. Use Compose locally, add an OCI Helm chart
  for Kubernetes, and publish a separate thin client wheel later rather than
  asking teams to install the server as a Python library.
  Rationale: deployment consumers need a reproducible product and stable remote
  contract; developer consumers need a small client without provider SDKs or
  duplicated analysis logic. One image also keeps frontend and API schemas on
  the same release.
  Date/Author: 2026-07-22 / user and Codex. See ADR 0012.

- Decision: After both adapters passed the shared conformance contract, record
  Observatory ownership and distribution in ADR 0012.
  Rationale: the second implementation had shaped the backend-neutral boundary,
  so the read-product and OCI-first distribution choices were ready to become
  durable architecture rather than remaining plan-only assumptions.
  Date/Author: 2026-07-22 / user and Codex.

## Outcomes & Retrospective

Milestones 1 through 4 and the first Observatory foundation step are complete.
The canonical documentation now treats
Trackio and W&B as adapters behind one tracking boundary without weakening the
existing run, trace, artifact, or lineage semantics. `posttrain.tracking` now
provides immutable lifecycle contracts, strict API/read models, shared telemetry
definitions for all five planned job kinds, and projection-shaped run views,
alerts, deltas, and comparisons. The lab consumes the canonical `RunSpec` and
provider-neutral stored artifact reference while retaining its current import
surface for callers. ADR 0012 now establishes Observatory as the single read
product and defines one OCI image as its primary central or self-hosted
distribution unit.

The contract and lab integration suites each pass seven tests, and focused Ruff
and Pyright validation pass. The real Trackio adapter now proves idempotent
finalization, logical-step enforcement, standard and Verifiers traces, physical
artifact mappings, input materialization, success/failure outcomes, and
read-after-write behavior under both Turso and SQLite. The first credentialed
W&B Cloud run now proves the basic success path through the SDK writer and
Public API reader, including metric history and trace-artifact reconstruction.
Milestone 4 is now complete. A shared canonical fixture is exercised by both
adapters and checked against the same normalized projection. Credentialed W&B
Cloud coverage includes the complete artifact chain and all terminal outcome
states. Lab execution now depends on `TrackingBackend`, treats interruption as
cancelled, avoids placing raw exception messages in tracking data, and selects
Trackio or W&B explicitly while keeping untracked execution available.
`posttrain-observatory` now owns its own strict Pydantic product contracts,
telemetry registry, run views, alerts, deltas, comparisons, service, and first
CLI entrypoint. Tracking retains only provider-neutral lifecycle and raw
evidence contracts. The verified foundation passes 50 non-network package and
integration tests with 5 network tests deselected, Ruff, focused Pyright, and
all 8 import-linter contracts. Report migration, delivery adapters, frontend,
and OCI packaging remain incomplete.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. It is a Python 3.12 `uv`
workspace whose members are `apps/*` and `packages/*`. A work package is a
project-local declaration of bindings and enabled jobs for one stage. A run is
one observed execution of one job definition inside that work package. A
tracking backend stores the evidence emitted by a run. A normalized view is a
provider-neutral Pydantic model returned after reading and interpreting that
evidence. MCP is a protocol through which an LLM can call typed tools.

The current canonical execution identity is
`apps/lab/src/posttrain_lab/execution.py::RunSpec`. The current operation-facing
observation contract is
`packages/common/src/posttrain/common/execution.py::Observer`, injected through
`RunContext`. `apps/lab/src/posttrain_lab/execution.py::execute_run_tracked`
starts and finalizes the host-selected provider-neutral `TrackingBackend`.
`packages/reports/src/posttrain/reports/work_packages.py` still validates the
stored configuration and builds a work-package view through raw Trackio SQL
from `packages/reports/src/posttrain/reports/query.py`; that is the remaining
Milestone 5 migration.

Create these workspace distributions:

- `packages/tracking`, distribution `posttrain-tracking`, import
  `posttrain.tracking`. It owns `RunSpec`, `RunOutcome`, `TrackingBackend`,
  `TrackedRun`, `RunDataSource`, normalized raw evidence models, and shared
  backend-conformance contracts.
- `packages/tracking-trackio`, distribution `posttrain-tracking-trackio`, import
  `posttrain_tracking_trackio`. It is the only workspace package that imports
  Trackio and implements Trackio writer and reader adapters.
- `packages/tracking-wandb`, distribution `posttrain-tracking-wandb`, import
  `posttrain_tracking_wandb`. It is the only workspace package that imports W&B
  and implements W&B writer and reader adapters.
- `apps/observatory`, distribution `posttrain-observatory`, import
  `posttrain_observatory`. It is the dedicated read product and owns job
  telemetry definitions, `RunViewService`, work-package/stage/lineage analysis,
  alerts, comparisons, report exports, Python API, FastAPI, FastMCP, CLI wiring,
  and `frontend/`, a Svelte 5/Vite static web application. Every surface calls
  the same application service.

The related Trackio source is the sibling repository
`/home/hammad/projects/trackio`. Its implementation branch is
`codex/trackio-read-api` at commit
`f52d5c34dac45f803f44fdf6fac21658810afe3b`, based on
`feature/turso-verifiers-ui`, with CarbonTeq `origin` and Trackio `upstream`.
The tested SHA is pinned in `packages/tracking-trackio/pyproject.toml` and
`uv.lock`. The sibling TRL fork at
`/home/hammad/projects/trl` is evidence for trainer metric behavior but requires
no changes in this plan. W&B is consumed as a released SDK, not a local source
repository.

Use current compatible dependency lines when implementation begins and commit
exact lock resolutions. The researched starting bounds are `wandb>=0.28.1,<0.29`,
`fastapi>=0.139.2,<0.140`, and `mcp[cli]>=1.28.1,<2`; avoid the MCP 2 prerelease
line in this slice. Add Node 24 to `mise.toml`, commit the observatory
`package-lock.json`, and pin the chosen Svelte 5/Vite versions through that
lockfile.

## Plan of Work

### Milestone 1: Amend the canonical observation boundary

Add a dated, narrow observation-backend amendment to
`docs/post-training/README.md`. Update only the affected passages in
`docs/post-training/04-framework.md`, `05-apis.md`, and
`06-observation-and-lineage.md`. Preserve the existing ontology and metric,
trace, and artifact semantics. Replace statements that make Trackio the product
boundary with a tracking-backend contract having Trackio and W&B adapters.
Define normalized job-aware views as the read boundary and state that one shared
telemetry definition drives Python, frontend, and MCP projections. Continue to
describe Trackio as the default local backend and its native Verifiers trace
support as richer than the first W&B trace mapping. Do not revise older
architecture documents or ADRs during this milestone.

This milestone is complete when the canonical documents consistently permit
both backends, still forbid capability packages from importing either backend,
and still define native Verifiers traces as replay authority. Run link checks if
available and `git diff --check`.

### Milestone 2: Establish canonical tracking and job-view contracts

Create `packages/tracking/pyproject.toml` and
`packages/tracking/src/posttrain/tracking/`. Move `RunSpec` and its artifact input
value from `apps/lab` into this package, leaving a temporary re-export so current
lab callers continue to work during the migration. Replace
`TrackioArtifactRef` at this boundary with a provider-neutral immutable stored
artifact reference containing provider, namespace, logical name, immutable
version, optional digest, and opaque provider metadata. Keep Hub and local
artifact values in `posttrain.common`; do not put provider SDK objects into any
canonical model.

Define frozen internal lifecycle contracts and strict Pydantic read/API models.
The required protocol shape is:

    class TrackingBackend(Protocol):
        def start_run(self, spec: RunSpec) -> TrackedRun: ...

    class TrackedRun(Observer, Protocol):
        def materialize_inputs(self, inputs, root: Path) -> Mapping[str, LocalArtifactRef]: ...
        def finish(self, outcome: RunOutcome) -> None: ...

    class RunDataSource(Protocol):
        async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]: ...
        async def get_run(self, run_id: str) -> RunDetail: ...
        async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]: ...
        async def traces(self, run_id: str, query: TraceQuery) -> TracePage: ...
        async def artifacts(self, run_id: str) -> ArtifactSet: ...

`RunOutcome` supports `succeeded`, `partial`, `failed`, `cancelled`, and
`unsupported`, plus started/finished timestamps and a safe typed error summary.
`finish` is idempotent: the same outcome can be retried, while a conflicting
second outcome raises a contract error. The logical metric step is non-negative
and nondecreasing. Provider-specific capabilities, including live trace reads,
are returned as data-source metadata rather than silently changing behavior.

As an interim contract exercise, define `JobTelemetryDefinition` with a stable schema version, job kind, display
name, summary fields and reducers, chart series, health rules, comparison keys,
trace sections, and artifact roles. Add definitions for `train.sft`,
`train.dpo`, `train.grpo`, `eval.general`, and `eval.domain` using metric names
from `docs/post-training/06-observation-and-lineage.md`. A missing required value
becomes `missing` evidence, never numeric zero. Implement `RunViewService` with
`get_job_telemetry_schema`, `get_run_view`, `get_run_alerts`, `get_run_delta`,
and `compare_runs`. `get_run_delta` returns changed summary fields, alert
additions/removals, status, and at most the configured tips for key series; it
does not return every new metric row.

These product-facing types move into Observatory in Milestone 5 after both
provider adapters exercise the normalized evidence boundary. Add package tests that validate all telemetry definitions, reject unknown
fields, exercise missing/incomparable evidence, prove stable projection output,
and prove that cursors produce compact deterministic deltas.

### Milestone 3: Complete Trackio's generic public read API and adapter

In `/home/hammad/projects/trackio`, create a focused branch from the currently
pinned fork commit. Extend `trackio/api.py` with public run summary, history or
metric-series, trace, input/output artifact, and capability methods by delegating
to existing storage/server functions. Add stable return dictionaries or typed
values without exposing SQL table names. Preserve the existing CLI, server, MCP,
Turso, SQLite fallback, frontend, and Verifiers behavior. Add unit tests beside
the existing API/storage/server tests for config, explicit steps, standard and
Verifiers traces, artifact direction, empty runs, and missing identifiers. Run
the Trackio unit suite in default Turso mode and the focused compatibility suite
with `TRACKIO_DATABASE_ENGINE=sqlite`.

Commit the generic Trackio change before updating this repository. In
`packages/tracking-trackio`, pin that exact fork commit and implement
`TrackioBackend`, `TrackioTrackedRun`, and `TrackioDataSource`. Move the current
translation logic out of `apps/lab` without changing emitted names. Map
`work_package_id` to Trackio group for navigation but always persist and read the
canonical field. Map provider artifact versions into stored artifact references,
and use the canonical `run_id` in config rather than treating the Trackio
storage ID or display name as identity.

Trackio adapter tests use a temporary project database and the real Trackio SDK.
They write a common fixture containing nested resolved selections, two metric
steps, an event, a standard trace, a Verifiers trace, one input artifact, one
output artifact, and success or failure outcome, then read it through
`TrackioDataSource`.

### Milestone 4: Implement the W&B adapter and dual-backend conformance

Create `packages/tracking-wandb` with `wandb>=0.28.1,<0.29`. Implement
`WandbBackend` using explicit entity, project, canonical run ID, display name,
group, job type, nested config, mode, and base URL settings. Store
`work_package_id` in config and also map it to W&B group. Store job kind in
config and map it to W&B job type. Never require a W&B-generated name or ID to
reconstruct framework identity.

Every W&B history record emitted by the adapter includes a provider-independent
observation sequence and, when present, `posttrain/step`. Configure job metric
series against the logical step and do not treat W&B `_step` as the framework
step. Events use namespaced history fields and retain occurrence timestamps and
JSON-safe attributes. On finish, write canonical outcome fields to W&B summary,
flush artifacts, and call `finish(exit_code=0)` for success or a nonzero exit
code for failure. Preserve partial, cancelled, and unsupported as canonical
summary outcomes even when W&B's native state vocabulary is coarser.

Use W&B input/output artifact relationships. Store standard and Verifiers trace
observations in typed JSONL output artifacts for this slice, including external
ID, type, payload, and attributes. The reader uses `wandb.Api` to list and fetch
runs, `scan_history` for metrics/events, and public artifact relationships for
lineage and trace bundles. Do not parse private W&B offline files. Expose that
live traces are unavailable while a W&B run is active.

Put the shared conformance fixtures and assertions in
`packages/tracking/tests/conformance/`. Run them once with the real local Trackio
harness and once with a real synced W&B harness. Assert equivalent normalized
run identity, nested config, logical steps, events, final outcome, artifact
directions and logical versions, and post-finish traces. Cover success, explicit
failure, interruption without normal finish, and idempotent finish. Provider
metadata may differ and is excluded from logical equality.

The W&B network suite is marked `network`, skips clearly when credentials are
absent, and is a mandatory release gate. It uses `WANDB_API_KEY`, `WANDB_ENTITY`,
optional `WANDB_BASE_URL`, and `POSTTRAIN_WANDB_TEST_PROJECT`. Give each test run
a unique ID and a `posttrain-conformance` tag. Do not delete external runs
automatically; document a separate explicit cleanup command or retention policy.

### Milestone 5: Establish Observatory and retire reports

Refactor `apps/lab/src/posttrain_lab/execution.py` so tracked execution accepts a
configured `TrackingBackend`, calls `start_run`, injects the returned tracked
run as the operation observer, materializes inputs through it, and finalizes a
canonical outcome on every exit path. Preserve `execute_run` for untracked
execution. Select `trackio` or `wandb` through explicit CLI/config wiring; Trackio
remains the local default. Do not make trainer callbacks async. Backend writers
may buffer or use background workers internally, and finalization must wait for
durability.

Create the Observatory Python product structure first. Port the useful behavior
from `packages/reports` behind `RunDataSource`, including work-package queries
and materialized exports, without porting Trackio SQL. Move
`JobTelemetryDefinition`, the telemetry registry, `RunViewService`, and
job-aware view models from `packages/tracking` into Observatory while leaving
raw evidence models and reader protocols in tracking. Migrate all callers and
package-specific tests, replace `trackio-query` with an Observatory query/export
command, then delete the `posttrain-reports` distribution and
`packages/reports` directory. Do not retain a parallel compatibility product.

Add import-linter contracts proving that only the two backend packages import
their provider SDK and that Observatory and capability packages do not. Add an
Observatory ownership test proving its Python, report-export, HTTP, and MCP
adapters resolve views through the same application service.

Exercise one synthetic job through lab execution with each backend. A GPU is not
needed: the operation emits representative observations and a small artifact.
Existing package-specific train/eval/serve tests continue using recording or
null observers.

### Milestone 6: Complete Observatory API, MCP, and frontend

Extend the Observatory product established in Milestone 5 with one composition
function that selects a `RunDataSource`, loads the telemetry registry, and
returns its application service. FastAPI routes and MCP tools are adapters over
this service. Implement these HTTP operations and equivalent MCP tools:

    get_job_telemetry_schema(job_kind)
    get_run_view(run_id)
    get_run_alerts(run_id)
    get_run_delta(run_id, cursor, fields=None, include=("summary", "alerts"))
    compare_runs(run_ids, view="summary")

Also expose a filtered run-list route for the web application. All responses use
the same strict Pydantic models. The cursor is an opaque service-generated value;
consumers do not interpret provider timestamps or row IDs. MCP defaults to
projection-shaped summaries and alerts so an LLM does not receive the raw
telemetry firehose. Support stdio for local Codex integration and Streamable
HTTP for the observatory service using stable `mcp[cli]>=1.28.1,<2`.

Create `apps/observatory/frontend` as a Svelte 5/Vite TypeScript application.
The first page lists runs with provider, job kind, work package, status, start
time, duration, and fired-alert count. The run-detail page renders the summary
cards, charts, health conditions, trace section, and artifact links declared by
the selected `JobTelemetryDefinition`; it visibly marks missing and incomparable
evidence. It does not contain separate hard-coded SFT/DPO/GRPO metric lists.
Generate API types from the OpenAPI schema or maintain one checked conversion
step so frontend types cannot drift from Pydantic models.

Add API tests, an in-process MCP client test for every tool, frontend unit tests
for schema-driven rendering, and a Playwright smoke that starts the API with a
fixture data source, opens the run list, selects a run, and verifies summary,
alert, chart, and missing-evidence rendering. Capture one final screenshot as
validation evidence after the UI is complete.

## Concrete Steps

Work from `/home/hammad/projects/rl` unless a command explicitly changes
repository.

First validate documentation and create the new workspace packages:

    git diff --check
    uv sync --all-packages --python 3.12
    uv run pytest packages/tracking/tests -q

During Trackio work, use the sibling repository and do not update the application
pin until the fork tests pass and a commit exists:

    cd /home/hammad/projects/trackio
    git status --short --branch
    uv sync --extra dev --extra spaces
    uv run pytest tests/unit -q
    TRACKIO_DATABASE_ENGINE=sqlite uv run pytest tests/unit/test_sqlite_storage.py tests/unit/test_local_server_api.py -q

Return to the application, update the exact Trackio commit pin, lock, and run the
adapter suite:

    cd /home/hammad/projects/rl
    uv lock
    uv sync --all-packages --locked --python 3.12
    uv run pytest packages/tracking-trackio/tests -q

Run local W&B adapter tests without external credentials, then the required real
integration with credentials supplied by the operator:

    uv run pytest packages/tracking-wandb/tests -m "not network" -q
    WANDB_API_KEY=... WANDB_ENTITY=... POSTTRAIN_WANDB_TEST_PROJECT=posttrain-conformance \
      uv run pytest packages/tracking-wandb/tests -m network -q

The expected network transcript includes one created W&B run per conformance
case, successful Public API lookup, and logical-view equality assertions. A
credential failure must say which environment variable is missing rather than
falling back to a fake or offline reader.

Run observatory services against Trackio fixture data:

    uv run --package posttrain-observatory posttrain-observatory api --backend trackio --project posttrain-observatory-fixture
    uv run --package posttrain-observatory posttrain-observatory mcp --backend trackio --project posttrain-observatory-fixture --transport stdio

In another terminal, build and test the frontend:

    cd /home/hammad/projects/rl/apps/observatory/frontend
    npm ci
    npm run test
    npm run build

At the end of each milestone, run focused tests and update `Progress`,
`Surprises & Discoveries`, and `Decision Log`. Before completion, run the full
validation ladder:

    cd /home/hammad/projects/rl
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Record exact test counts and any skipped GPU, Docker, network, or credentialed
tests in `Artifacts and Notes` rather than describing unavailable coverage as
passing.

## Validation and Acceptance

The feature is accepted only when all of the following behavior is observable.

A common synthetic run written through Trackio and the same run fixture written
through W&B can each be read back into logically equal `RunDetail`, metric
series, events, artifact edges, final outcome, and post-finish traces. Tests prove
both successful and failed finalization and show that repeated identical finish
calls do not corrupt either run. Metric charts use the explicit logical step and
remain equivalent even though W&B internal history rows advance for events.

`apps/lab` can select `trackio` or `wandb` without a train, eval, serve, or data
package importing either SDK. `uv run lint-imports` enforces the boundary.
The `posttrain-reports` distribution and `packages/reports` directory no longer
exist; their retained behaviors are tested through Observatory against
provider-neutral data sources.

For each of SFT, DPO, GRPO, general eval, and domain eval, the telemetry schema
validates and produces a typed run view. Missing evidence is labeled `missing`,
failed runs expose safe errors and fired health rules, and comparisons reject or
mark incompatible runs instead of manufacturing values.

The web run-detail and `get_run_view` MCP tool return the same summary field
names and values for a fixture run. `get_run_delta` returns only status, changed
summary fields, alert differences, and configured series tips. A test asserts
that unrelated raw metrics are absent. `compare_runs` returns a table-shaped
projection under one telemetry schema.

The Svelte production build completes. The Playwright smoke opens the local
observatory, selects a run, and displays job-specific cards, at least one chart,
an alert or healthy state, and artifact/trace navigation. A screenshot is saved
under a documented validation-artifact path and linked from the completed plan.

The real W&B network conformance command passes against a synced test project.
Trackio adapter integration passes using its real local database. Provider fakes
alone are insufficient for completion.

## Idempotence and Recovery

Documentation and package creation are additive and safe to repeat. Keep the
existing Trackio execution and report entrypoints working until their new
adapters pass conformance; remove compatibility paths only in Milestone 5 after
all callers migrate.

Trackio fork work occurs on its own branch. If its public API change fails, leave
the application pinned to the old commit and revise this plan with the missing
capability; never point the application at an uncommitted sibling checkout. If a
new Trackio pin causes lock or regression failures, restore only that dependency
line and regenerate `uv.lock`; do not reset unrelated working-tree changes.

W&B tests create uniquely named, tagged runs. They do not delete external state
automatically. If a test stops before finish, keep the run as crash-path evidence
and record its URL/ID in the test output. Cleanup is an explicit operator action
against the dedicated conformance project. Never print or store `WANDB_API_KEY`.

`TrackedRun.finish` is retry-safe. Adapter staging files live in the run's
temporary workspace; on failure, preserve only the paths needed to diagnose a
failed artifact/trace upload and document the retry command. Do not make that
workspace a second run registry.

Frontend installation uses `npm ci`, which is repeatable from the committed
lockfile. Generated OpenAPI/frontend types must be produced by one documented
command and checked for a clean diff in validation.

## Artifacts and Notes

Initial repository evidence:

    rl workspace: /home/hammad/projects/rl
    canonical docs: docs/post-training/README.md and 01 through 06
    Trackio source: /home/hammad/projects/trackio
    Trackio base branch: feature/turso-verifiers-ui
    Trackio implementation branch: codex/trackio-read-api
    Trackio pin: f52d5c34dac45f803f44fdf6fac21658810afe3b
    TRL source: /home/hammad/projects/trl (no planned edits)

Current code anchors:

    packages/common/src/posttrain/common/execution.py::Observer
    packages/tracking/src/posttrain/tracking/contracts.py::RunSpec
    apps/lab/src/posttrain_lab/execution.py::execute_run_tracked
    apps/lab/src/posttrain_lab/tracking/trackio_observer.py::TrackioObserver
    packages/reports/src/posttrain/reports/query.py::query_project
    packages/reports/src/posttrain/reports/work_packages.py::work_package_view

W&B 0.28.1 supports explicit run ID, name, nested config, group, job type,
online/offline mode, resume policy, metric/media logging, input/output artifacts,
and finish with an exit code. Its Public API exposes run config, state, history,
summary, and artifact relationships. The plan deliberately uses those stable
surfaces and does not parse W&B offline storage or adopt Weave in the first
slice.

Trackio already supports custom frontends and raw MCP tools. The custom
observatory remains separate because it must query both providers and render
post-training job semantics. Generic Trackio query improvements are contributed
to the Trackio fork so `posttrain-tracking-trackio` does not add new raw SQL.

Append concise test transcripts, run IDs/URLs without secrets, the final
observatory screenshot path, and provider capability differences here as work
proceeds.

Milestone 2 validation (2026-07-22 01:45Z):

    uv run --package posttrain-tracking pytest packages/tracking/tests -q
    7 passed in 0.08s

    uv run --package posttrain-lab pytest apps/lab/tests/test_execution.py -q
    7 passed in 0.44s

    uv run ruff check packages/tracking <focused lab paths>
    All checks passed

    uv run pyright packages/tracking <focused lab paths>
    0 errors, 0 warnings, 0 informations

    git diff --check
    clean

Milestone 3 validation (2026-07-22 02:10Z):

    /home/hammad/projects/trackio: uv run pytest tests/unit -q -k 'not gpu_hardware'
    323 passed, 2 skipped, 5 deselected

    /home/hammad/projects/trackio: focused API/trace/artifact suite
    47 passed, 1 skipped

    /home/hammad/projects/trackio: focused public API tests, Turso and SQLite
    2 passed in each engine

    uv run pytest packages/tracking-trackio/tests -q
    2 passed under Turso; 2 passed with TRACKIO_DATABASE_ENGINE=sqlite

    Trackio complete unit run without deselection
    323 passed, 2 skipped, 5 GPU-environment failures

Milestone 4 partial validation (2026-07-22 02:38Z):

    W&B Cloud entity/project: carbonteq/posttrain-conformance
    W&B SDK: 0.28.1
    run: pt-06118c4538be4c07a1d41199560cc8a9
    URL: https://wandb.ai/carbonteq/posttrain-conformance/runs/pt-06118c4538be4c07a1d41199560cc8a9
    credentialed writer-to-reader network test: 1 passed in 20.96s

    uv run --group dev pytest packages/tracking/tests packages/tracking-trackio/tests packages/tracking-wandb/tests -m 'not network' -q
    11 passed, 1 deselected in 1.49s

    focused Ruff validation
    All checks passed

    focused Pyright validation
    0 errors, 0 warnings, 0 informations

Milestone 4 completion and lab migration validation (2026-07-22 02:57Z):

    W&B Cloud entity/project: carbonteq/posttrain-conformance
    W&B SDK: 0.28.1
    comprehensive consumer run: pt-767ca99ff1c84a9ca96bfecdd5cdfa59
    URL: https://wandb.ai/carbonteq/posttrain-conformance/runs/pt-767ca99ff1c84a9ca96bfecdd5cdfa59

    credentialed W&B network suite
    5 passed in 87.42s

    uv run pytest packages/tracking/tests packages/tracking-trackio/tests packages/tracking-wandb/tests apps/lab/tests -m 'not network' -q
    49 passed, 5 deselected in 1.23s

    uv run lint-imports
    7 contracts kept, 0 broken

    uv run ruff check .
    All checks passed

    focused Pyright validation for all changed tracking and lab paths
    0 errors, 0 warnings, 0 informations

    full-workspace Pyright validation
    19 missing-import errors from optional GPU/data extras not installed in the
    base environment; no changed-path type errors

    git diff --check
    clean

## Interfaces and Dependencies

At the end of the plan, provider-neutral imports are available from
`posttrain.tracking`:

    RunSpec
    RunOutcome
    TrackingBackend
    TrackedRun
    RunDataSource
    RunQuery
    RunSummary
    RunDetail
    MetricPoint
    MetricSeries
    TraceQuery
    TracePage
    ArtifactSet
    TrackingCapabilities

Observatory exports the product-facing read surface from
`posttrain_observatory`:

    JobTelemetryDefinition
    RunView
    RunAlert
    RunDelta
    RunComparison
    RunViewService

The Trackio distribution exports `TrackioBackend` and `TrackioDataSource`. The
W&B distribution exports `WandbBackend`, `WandbDataSource`, and an explicit
settings model containing entity, project, base URL, and mode. Observatory
imports these only in its composition root; Python queries, report exports,
routes, MCP functions, and frontend payloads depend on its application service
or provider-neutral raw evidence models.

Use Pydantic 2 strict, frozen, `extra="forbid"` models at persisted, HTTP, MCP,
and provider-read boundaries. Retain small frozen dataclasses and protocols for
trusted in-process emission where they already work. All timestamps are
timezone-aware UTC. All config and attributes are JSON-safe and exclude secrets.

Dependency ownership is:

    posttrain-tracking         -> posttrain-common, pydantic
    tracking-trackio          -> posttrain-tracking, pinned CarbonTeq Trackio
    tracking-wandb            -> posttrain-tracking, wandb>=0.28.1,<0.29
    posttrain-observatory      -> posttrain-tracking, both adapters,
                                  fastapi>=0.139.2,<0.140,
                                  mcp[cli]>=1.28.1,<2
    observatory frontend       -> Svelte 5, Vite, TypeScript, test/build tools

No capability package imports a provider SDK. Exact Python and JavaScript
resolutions live in `uv.lock` and `apps/observatory/frontend/package-lock.json`.
Git dependencies use immutable commits.

Revision note (2026-07-22): Created this plan after the user narrowed the
observability direction to dual Trackio/W&B backends, shared job telemetry
definitions, a custom frontend, and domain-aware MCP tools. The plan records
W&B step/read constraints and the multi-repository Trackio contribution order so
implementation can proceed without relying on the preceding conversation.

Revision note (2026-07-22 01:45Z): Completed the canonical documentation
amendment and provider-neutral tracking/view foundation, migrated the lab
`RunSpec` boundary, and recorded focused validation. Trackio adapter work is the
next milestone.

Revision note (2026-07-22 02:10Z): Added and pinned Trackio's public run read
surface, implemented the Trackio backend/data source, and recorded real
write/read conformance in both storage engines. W&B adapter parity is next.

Revision note (2026-07-22 02:38Z): Authenticated against W&B Cloud, corrected
the configured entity slug, and proved the first real SDK write/Public API
readback with metric history and trace-artifact reconstruction. Milestone 4
remains in progress for failure and artifact-lineage parity.

Revision note (2026-07-22 02:57Z): Completed shared Trackio/W&B logical
conformance, including a real W&B artifact producer/consumer chain and all
terminal outcomes. Migrated lab execution and CLI selection to the backend
contract, recorded the W&B flattening and artifact-deduplication findings, and
left report migration as the remaining Milestone 5 work.

Revision note (2026-07-22 03:05Z): Replaced the overlapping reports-plus-thin-
Observatory architecture with one dedicated Observatory read product. ADR 0012
records retirement of `posttrain-reports`, relocation of job-aware query/view
ownership from tracking into Observatory, and one shared service for Python,
report exports, HTTP, MCP, and frontend consumers.

Revision note (2026-07-22 03:18Z): Recorded the OCI-first distribution model:
one image with both readers and all product surfaces, central or self-hosted
deployment, Compose first, Helm when Kubernetes is introduced, immutable
release identity, server-side secrets, and a separate future client wheel.

Revision note (2026-07-22 04:44Z): Split the remaining Observatory product work
into the self-contained execution plan at
`docs/plan/observatory-product-implementation.md`. That plan is authoritative
for generic fallback behavior, job view resolution, unified trace/evaluation
UX, frontend implementation, transports, and packaging while this file remains
the umbrella history for dual-backend tracking.

Revision note (2026-07-22 04:55Z): The detailed Observatory plan now treats LLM
semantic summaries as an optional, provider-neutral analysis layer over both
job-specific and generic evidence. Generated claims are explicit, cited, and
non-authoritative; deterministic views and health rules remain unchanged.
