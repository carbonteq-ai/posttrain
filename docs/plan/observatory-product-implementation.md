# Build the Observatory product from provider evidence to job-aware analysis

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. It is the
detailed continuation of `docs/plan/dual-backend-tracking-observatory.md` after
the Trackio and W&B reader/writer contracts and the initial Observatory Python
package were established. Product authority remains
`docs/post-training/README.md` and the six documents it indexes. Durable
ownership and distribution decisions are in
`docs/decisions/0012-observatory-read-product.md`. Visual direction and
exploratory screens are in
`docs/design/observatory/moodboard/README.md`.

## Purpose / Big Picture

After this plan is complete, an ML expert can open one light-themed Observatory
product, find runs across configured Trackio and W&B sources, and move from a
project or work package into the evidence that matters for the run's job kind.
SFT, DPO, GRPO, general evaluation, and domain evaluation receive curated
summaries, charts, alerts, comparisons, trace sections, and artifact roles from
versioned job definitions. Every run, including an unknown or newly introduced
job kind, also has a generic evidence view that lists its metric namespaces,
plots explicitly selected series, and exposes status, events, traces, artifacts,
resolved inputs, and source metadata without assigning deterministic domain
meaning. On either view, the expert may explicitly request a grounded semantic
summary through the provider-neutral API. That summary uses an LLM to explain
observed evidence, identify possible relationships, and suggest what to inspect
next while citing the exact metrics, traces, alerts, and artifacts supplied to
it. The first frontend deliberately does not expose this control until its
placement and researcher workflow are resolved; measured values and
deterministic job health remain the product surface.

The default run page is a focused operational brief. It leads with status,
health, the most important job-specific figure, and the evidence needed to
investigate it. Notebook-like sections continue down that same page; there is no
second competing notebook product. Evaluation and Verifiers traces occupy one
surface: evaluation is a population projection over atomic trace records, and
any aggregate anomaly can be filtered down to the exact rollout, rubric,
reward components, transcript, tool calls, flags, and artifacts that explain
it.

The result is observable without a GPU. Start Observatory against a fixture
source, open the run list, and inspect one SFT run, one GRPO run, one evaluation
run, and one unknown job. The first three resolve to job-specific views. The
unknown job resolves safely to the generic metrics workspace. The same logical
responses are available through Python, HTTP, MCP, CLI/export, and the frontend.

## Progress

- [x] (2026-07-22 04:44Z) Established provider-neutral tracking contracts,
  real Trackio and W&B adapters, shared conformance coverage, and backend
  selection in lab execution under the umbrella plan.
- [x] (2026-07-22 04:44Z) Created `apps/observatory`, moved the strict telemetry
  registry and projection service into it, and added focused service tests.
- [x] (2026-07-22 04:44Z) Accepted ADR 0012: Observatory is the single read
  product, `packages/reports` will be retired, and one OCI image is the primary
  distribution.
- [x] (2026-07-22 04:44Z) Captured the Hex-inspired light visual direction and
  four Observatory workflow explorations under
  `docs/design/observatory/moodboard/`.
- [x] (2026-07-22 04:55Z) Refined the view boundary to include optional,
  grounded LLM semantic summaries without weakening deterministic job and
  generic evidence contracts.
- [x] (2026-07-22 06:04Z) Ported work-package, stage, lineage, and typed JSON
  export behavior from
  `packages/reports` behind provider-neutral data sources, then remove the
  reports package and raw SQL entrypoint.
- [x] (2026-07-22 06:04Z) Added the multi-source registry, stable run locator, view resolver,
  discriminated product models, and generic metric/evidence view.
- [x] (2026-07-22 06:04Z) Completed job-specific SFT, DPO, GRPO, general-eval, and domain-eval
  projectors and comparison contracts.
- [x] (2026-07-22 06:04Z) Added the unified Verifiers trace/evaluation index, filters, population
  summaries, trace detail, and explicit completeness semantics.
- [x] (2026-07-22 06:04Z) Added optional, provider-neutral LLM semantic summaries for job-specific,
  generic, evaluation, and comparison evidence with citations, provenance,
  redaction, bounded context, and deterministic fallback behavior.
- [x] (2026-07-22 06:04Z) Exposed the single application service through versioned HTTP, compact MCP
  tools, Python, CLI, and report exports.
- [x] (2026-07-22 06:04Z) Replaced the discarded Svelte prototype with the accepted React 19,
  Tailwind CSS 4, TypeScript, and Vite frontend; visually verify the run list,
  focused job views, generic metrics, dedicated system metrics,
  traces/evaluation, lineage, and alerts.
- [x] (2026-07-22 06:04Z) Ran a visual robustness audit against the selected run-detail and
  trace/evaluation mocks, fix every P0/P1/P2 mismatch, and retain screenshots
  plus `design-qa.md` as evidence.
- [x] (2026-07-22 06:04Z) Ran a separate job-by-job usability audit for SFT, DPO, GRPO,
  evaluation, unknown jobs, traces, and system telemetry; fix the high-impact
  findings and retain the captured flow evidence.
- [x] (2026-07-22 06:04Z) Added the non-root, read-only OCI build, local Compose
  composition, production configuration guardrails, release metadata, and
  operational health endpoints.
- [ ] Complete local, cross-backend, browser, MCP, API, accessibility, Docker,
  and credentialed W&B validation and record exact evidence here.
- [x] (2026-07-22 06:50Z) Reworked the React evidence hierarchy against the
  accepted images: self-hosted Inter and Instrument Serif, coherent analytical
  surfaces, compact segmented chart controls, compact Radix filter popovers,
  reduced label sizes, chart-local selected-step context, and a dedicated
  input-to-output lineage rail.
- [x] (2026-07-22 06:50Z) Executed a real schema-v4 Qwen 3.5 2B SFT QLoRA run
  and a real schema-v4 DPO QLoRA run in Trackio project
  `posttrain-observatory-v4-20260722`; the successful logical run IDs are
  `ebff1388-74bf-4c0e-ad0d-597cdb157242` and
  `efade9e3-ea7b-4556-b93a-9b3361e8ceb0`.
- [x] (2026-07-22 06:50Z) Switched the local OCI composition from fixture data
  to a read-only snapshot of the real Trackio project and verified that
  Observatory resolves the real schema-v4 runs, artifacts, selections, and
  normalized system telemetry without giving the product write access to the
  host evidence store.
- [x] (2026-07-22 07:25Z) Replaced the artifact-only detail page with a
  run-centered lineage projection that visibly connects recorded consumed
  artifacts to the run and the run to durable outputs, while preserving a
  provider/version/digest ledger and explicit missing-edge states.
- [x] (2026-07-22 07:32Z) Corrected the global form-control font reset that was
  overriding component typography, then reduced the run evidence navigation to
  an 11-pixel, 36-pixel-high compact tier and verified computed styles in the
  Codex in-app Browser.
- [x] (2026-07-22 11:38Z) Replaced the generic multi-axis metric plot with a
  searchable, bounded workspace of independent metric cards; added the public
  Trackio system-history read surface; and verified that Observatory queries
  five existing host samples for the real DPO run and clips them to the run's
  recorded start/finish interval without adding a system-metric writer.
- [x] (2026-07-22 12:33Z) Made trace/evaluation navigation capability-driven
  from `JobTelemetryDefinition.trace_sections`: SFT and DPO no longer expose
  the section, while GRPO, distillation, evaluation, and generic runs with
  recorded traces retain it.
- [x] (2026-07-22 12:38Z) Replaced the raw run-configuration page with a
  job-aware inspector over the existing provider-neutral selection envelopes:
  SFT, DPO, GRPO, and evaluation runs receive schema-specific groups, source
  provenance is separate, unknown selections remain visible, and the complete
  server-redacted JSON remains available as a secondary audit disclosure.
- [x] (2026-07-22 12:53Z) Corrected the Observatory navigation hierarchy:
  one project is now an explicit selector scope, the sidebar expands work
  packages as the grouping above their runs, and selecting a package opens a
  provider-neutral package page with job-kind groups, run outcomes, lineage,
  and an explicit missing decision-record state. Also reduced search focus to
  one visible focus boundary.
- [x] (2026-07-22 13:00Z) Added a compact, job-aware algorithm-settings block
  between training inputs and produced evidence in the run lineage rail. SFT,
  DPO, GRPO, and distillation now project only their method-defining data and
  optimization values from resolved selections; future runs also retain the
  full resolved LoRA, QLoRA, full-update, or QAT plan needed to show adapter
  and quantization parameters without backend inference.
- [x] (2026-07-22 13:14Z) Added validated descriptions to `WorkPackage` and
  `JobDefinition`, snapshotted both into every new run, projected them through
  provider-neutral work-package views, and rendered package purpose plus
  versioned definition purpose in the run and work-package interfaces.
- [x] (2026-07-22 13:20Z) Reorganized project navigation by canonical workflow
  stage (`screen`, `train`, `qualify`), sorted packages and runs by most recent
  start within each stage, and exposed compact, exact run times. Removed the
  redundant curated/provider ornament and deferred generated interpretation
  from the run overview while retaining its provider-neutral API.
- [x] (2026-07-22 13:25Z) Reworked the run lineage canvas to show snapshotted
  base-model, training-dataset, and validation-dataset selections as “Run
  inputs,” while retaining recorded consumed artifacts as a distinct immutable
  edge subsection. Reduced status badges to the compact interface type scale.
- [x] (2026-07-22 13:34Z) Reduced the System metrics run window to compact
  heading metadata, promoted the actual telemetry summary to the first
  analytical surface, and moved each chart's metric definitions inline with
  its title instead of spending a separate header row.
- [x] (2026-07-22 14:30Z) Passed the real W&B cloud conformance gate (`5
  passed`) and produced successful schema-v4 W&B SFT and DPO runs in
  `posttrain-observatory-v4-20260722`. The SFT run is
  `b16ddad6-f331-4e8e-9885-92bbfabfc670`; the one-backpropagation DPO run is
  `a22ad70f-5f30-4dbf-ba05-ad9a716f67ac`. Failed DPO attempts remain visible
  and correctly finalized as evidence of resource and data-boundary failures.
- [x] (2026-07-22 14:30Z) Added simultaneous Trackio and W&B source
  configuration plus an explicit Observatory backend selector. Project,
  work-package, run, and view queries remain source-qualified; the live product
  shows seven Trackio runs and five W&B runs without merging package contents.
- [x] (2026-07-22 14:30Z) Normalized W&B's existing system-history stream into
  the canonical `system/*` projection without adding a writer. Live browser
  verification shows four host samples for the W&B SFT run and two for the W&B
  DPO run, bounded by their recorded start and finish times.
- [x] (2026-07-22 14:30Z) Closed three live-only defects: W&B history cache
  writes now use the container's writable `/tmp`, slower prior view requests
  cannot replace the current run, and consumed adapter artifacts no longer
  appear under produced evidence. The production container and Codex in-app
  Browser were used for the final SFT, DPO, system, and lineage checks.
- [x] (2026-07-22 14:42Z) Completed the implementation validation gate: Ruff,
  Pyright, all eight import contracts, `git diff --check`, the React test/build
  (`12 passed`), the credentialed W&B conformance suite (`5 passed`), and the
  full Python suite (`195 passed`, `5` credential/network tests skipped by the
  default marker policy). Aligned all training catalog lock fingerprints with
  the current immutable `uv.lock` digest.
- [x] (2026-07-23 11:32Z) Added provider-neutral runtime phase boundaries to
  `RunContext`, instrumented TRL SFT/DPO/GRPO/distillation, veRL process
  execution, Verifiers evaluation, and lab operation boundaries, and projected
  existing Trackio/W&B host samples into non-overlapping phase windows.
  Observatory now returns per-phase duration, sample count, mean/minimum/peak
  system metrics, explicit coverage issues, and an SVG VRAM timeline. The
  implementation does not depend on W&B Weave and does not add a second system
  telemetry writer.
- [x] (2026-07-23 11:41Z) Tightened SFT and DPO phase ownership: both now keep
  rendering, population profiling, and dataset conversion inside
  `data_preparation`; every SFT validation invocation, including evaluations
  triggered from inside `trainer.train`, emits a nested `evaluation` interval
  so its host samples are not attributed to `actor_update`. DPO retains no
  validation phase until its request contract supports held-out preference
  validation.
- [x] (2026-07-23 11:57Z) Persisted complete, revisioned execution-target
  context with every work-package run; normalized complete and historical
  partial targets in Observatory; and changed the phase surface into a
  capacity-first view plus an overlapping interval timeline. Declared
  aggregate VRAM now provides the capacity denominator, while the observed
  peak remains a separate measured value.

## Surprises & Discoveries

- Observation: the current `ObservatoryService.get_run_view` raises a `KeyError`
  for an unregistered job kind, so an otherwise valid run cannot be inspected.
  Evidence: `apps/observatory/src/posttrain_observatory/service.py` calls
  `get_job_telemetry_schema` before loading metrics and has no fallback path.

- Observation: the current normalized reader already exposes the minimum safe
  generic evidence boundary: `RunDetail.metric_names`, `metric_series`, events,
  paged traces, artifacts, and capabilities.
  Evidence: `packages/tracking/src/posttrain/tracking/contracts.py::RunDataSource`
  and `packages/tracking/src/posttrain/tracking/models.py` contain no
  Observatory-specific interpretation.

- Observation: `packages/reports` still derives work-package status, metric
  names, and lineage by reading Trackio SQLite tables directly.
  Evidence: `packages/reports/src/posttrain/reports/query.py` and
  `work_packages.py` import Trackio storage and query physical tables. Porting
  this SQL would violate ADR 0012 and make W&B runs second-class.

- Observation: Verifiers records are persisted as raw, versioned trace payloads
  and already carry useful stable selection attributes, but Observatory has no
  projector or population aggregation layer.
  Evidence: `packages/eval/.../verifiers/adapter.py` emits `trace_type="verifiers"`
  with the complete record, while `TraceRecord.payload` deliberately remains a
  provider-neutral JSON object.

- Observation: the generated run, comparison, and traces/evaluation concepts
  share one shell and evidence vocabulary. The remaining design ambiguity is
  whether the focused brief or analytical notebook is the default run page.
  Evidence: `docs/design/observatory/moodboard/README.md` records both concepts
  and warns against shipping them as peer pages.

- Observation: deterministic job definitions and LLM interpretation solve
  different problems. Definitions provide repeatable metric meaning and health
  rules; an LLM can connect recorded evidence into a useful narrative even when
  the job kind is unknown.
  Evidence: the product already has normalized metric, trace, event, artifact,
  and configuration evidence from which a bounded, cited analysis bundle can be
  constructed without changing tracking storage.

- Observation: building the exact pinned Trackio fork from source requires both
  Git and Node/npm because its Hatch build hook produces the bundled dashboard.
  Evidence: the first two OCI builds failed first without Git and then without
  npm; the final multi-stage build supplies both only in `python-builder` and
  copies the resulting virtual environment into a smaller runtime image.

- Observation: visual and usability review found product-meaning defects that
  unit tests did not: the missing selected-step inspector, the wrong DPO default
  chart, narrow page overflow, and a trace label that conflated errors with
  verifier outcomes.
  Evidence: the before/after captures and findings are retained in
  `apps/observatory/frontend/design-qa.md` and `usability-audit.md`.

- Observation: loading a real provider run exposed two fixture-hidden gaps.
  The SFT and DPO metric names emitted by current TRL differed from the initial
  telemetry definitions, and Trackio stores auto-captured GPU/CPU samples in a
  dedicated system table whose names are not part of run metric history.
  Evidence: the schema-v4 runs showed real `train/rewards/*` metrics while the
  initial system page was unavailable despite five stored GPU samples.

- Observation: Observatory's Trackio adapter accepted a remote server URL for
  ordinary metrics but bypassed that server for system metrics and opened
  `SQLiteStorage` in the Observatory process. In the two-container composition
  that local database is intentionally absent, so real host telemetry appeared
  missing even though Trackio held five DPO samples.
  Evidence: the real `train.dpo-efade9e3` interval is
  `2026-07-22T06:47:10.837759Z` through `06:47:33.900562Z`; the corrected remote
  query returns five samples between `06:47:10.947487Z` and
  `06:47:31.367367Z`.

- Observation: the first DPO QLoRA attempt at the inherited 512-token maximum
  exceeded the 8 GiB execution target. A 192-token workaround rejected the
  selected preference example; renderer inspection showed that it requires 434
  tokens. The bounded 448-token setting completed two backpropagation steps in
  7.594 seconds with 0.3472 aggregate train loss.
  Evidence: failed runs `ff93efa6-8e61-4f40-9c6f-4e897fc2fdfd` and
  `f454dd52-b4e3-4c62-98cc-1fd4c0e40448` remain visible beside successful run
  `efade9e3-ea7b-4556-b93a-9b3361e8ceb0`.

- Observation: W&B Public API run history uses a local cache even for read-only
  queries. The first production-container view failed because `scan_history`
  tried to create `/home/observatory/.cache` on the read-only root filesystem.
  Evidence: the live ASGI error reported `failed to create runhistory cache
  directory`; setting `WANDB_CACHE_DIR` and `XDG_CACHE_HOME` under the existing
  `/tmp` tmpfs restored the view without weakening the read-only image.

- Observation: W&B records GPU, CPU, process-memory, and elapsed-time samples in
  a distinct `history(stream="system")` stream. Those samples do not appear in
  `scan_history()` or the canonical run summary, so the initial W&B System
  metrics page appeared empty.
  Evidence: the real SFT run returned four native system rows and the DPO run
  returned two; the normalized view now exposes GPU utilization, allocated GPU
  memory, CPU utilization, process RSS, and wall time from those rows.

- Observation: live cloud latency exposed a frontend request-order race that
  fixture tests did not cover. Selecting SFT after DPO could update the run
  header immediately and then let a slower prior DPO response replace the SFT
  detail payload. The same browser pass showed that the overview's produced
  evidence list did not filter out input artifact edges.
  Evidence: the Codex in-app Browser briefly showed an SFT header with DPO
  rewards, settings, and lineage. A request-sequence guard and explicit
  input/output artifact split now have dedicated React regression tests.

- Observation: the real W&B DPO example cannot be shortened arbitrarily. The
  rendered pair needs 434 tokens; a 256-token run was rejected by the data
  guard, while a two-step 448-token run exhausted the 8 GiB target on its
  second update. One 448-token backpropagation completed and published full
  lineage and output artifacts.
  Evidence: successful run `a22ad70f-5f30-4dbf-ba05-ad9a716f67ac` records
  `max_length=448`, `max_steps=1`, loss `0.693147`, and gradient norm `36.75`.

- Observation: neither the pinned Trackio fork nor the core W&B experiment
  API exposes the provider-neutral runtime event object Observatory needs.
  Both adapters can nevertheless round-trip timestamped event name,
  occurrence, and attributes through ordinary run history.
  Evidence: `packages/tracking-trackio/.../adapter.py` and
  `packages/tracking-wandb/.../adapter.py` already serialize `EventObservation`
  under `event/*`; the shared conformance suite compares the reconstructed
  logical event rather than provider storage.

- Observation: nested phases such as an actor update containing rollout or
  teacher-scoring work cannot be aggregated by independently summing every
  interval. That double-counts both duration and host samples.
  Evidence: the Observatory projector cuts the run at every phase boundary,
  selects the most specific active phase for each segment, and the GRPO fixture
  test assigns six host samples exactly once across model loading, actor
  update, rollout, and artifact export.

- Observation: the common `ExecutionTarget` already contains device class,
  per-device memory, placement, and host constraints, but SFT, DPO, GRPO, and
  distillation snapshots previously retained only the binding's `target_id`.
  Reopening the current catalog from Observatory would not reconstruct the
  immutable target revision used by a historical run.
  Evidence: work-package resolution now writes a versioned
  `execution_targets` envelope that deduplicates targets while retaining every
  referencing role. Observatory preserves old target IDs as partial context
  and does not infer capacity from names such as `local-cuda-8gb`.

- Observation: the plan's schema-export command had drifted from the actual
  CLI; `schema` accepts `--openapi` and `--mcp` directly and has no `export`
  subcommand.
  Evidence: the stale command failed at argument parsing before any file write;
  the corrected command regenerated both checked schemas successfully.

## Decision Log

- Decision: Represent runtime phase boundaries with provider-neutral
  `runtime_phase_started`, `runtime_phase_completed`, and
  `runtime_phase_failed` events carrying `phase` and `phase_id`. Store those
  through each backend's existing event/history path. Do not add W&B Weave or
  any provider-specific span dependency.
  Rationale: phase correlation is required equally for Trackio and W&B, while
  Weave would create a second W&B-only data product and leak its call semantics
  into the common execution contract.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Keep system telemetry collection unchanged. Observatory joins
  existing timestamped `system/*` samples to phase intervals at read time,
  assigns nested intervals to the most specific active phase, and computes
  aggregates from raw samples. Runs without phase events retain the ordinary
  System metrics view and show phase analysis as unavailable.
  Rationale: the tracking providers already sample host telemetry; a dedicated
  writer would duplicate evidence and introduce synchronization errors.
  Historical runs must remain truthful rather than receiving inferred phase
  labels.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Use the run's immutable execution-target snapshot as the
  denominator for phase capacity analysis. Keep declared capacity and observed
  allocated-memory peak as separate fields; show capacity as ambiguous when
  role-specific targets disagree, and unavailable when an older snapshot lacks
  bytes. Render raw overlapping phase intervals in the debugging timeline
  while retaining non-overlapping segments for sample attribution.
  Rationale: target configuration answers what hardware the run was intended
  to fit, provider system history answers what was observed, and neither can
  substitute for the other. Keeping both prevents a run's own peak from being
  mislabeled as 100 percent capacity and makes nested phase overlap inspectable
  without double-counting host samples.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Configure every Observatory evidence source explicitly and let the
  user choose one backend at a time. Run locators, project scope, work-package
  queries, and view requests retain `source_id`; identical project and package
  names from different providers are never implicitly merged.
  Rationale: backend choice is part of evidence provenance, and hiding it in
  provider return order makes cross-backend validation and operational failures
  difficult to diagnose.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Read W&B's native system stream and normalize it to the existing
  canonical `system/*` names at query time. Do not add a system-metric writer or
  copy native W&B names into Observatory views.
  Rationale: W&B already captures the required host evidence, while the shared
  System metrics view needs provider-neutral names and the run's canonical
  start/finish boundary.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Represent stored artifacts from any tracking provider with the
  common `StoredArtifactRef`; keep provider-specific compatibility references
  only at existing boundaries. DPO may therefore consume the exact W&B SFT
  adapter and publish W&B lineage without importing W&B types into train or
  common contracts.
  Rationale: real cross-job lineage is where a Trackio-shaped artifact contract
  would leak most visibly into the supposedly neutral execution model.
  Date/Author: 2026-07-22 / Codex.

- Decision: Present resolved model and dataset selections in the lineage canvas
  as `Run inputs`, with `Recorded input artifacts` nested separately. Never
  promote a resolved selection into an artifact edge when the provider did not
  record one.
  Rationale: researchers need the effective model and data context beside the
  run, including for historical runs with incomplete artifact edges, but the
  canonical lineage contract reserves lineage for immutable consumed/produced
  artifact edges. The split makes both facts visible without weakening
  provenance semantics.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Order the project sidebar by canonical workflow stage, then by the
  most recently started package and run within each stage. Show recorded start
  time on both package and run summaries. Keep semantic-summary APIs available,
  but remove the generated-interpretation control and the redundant
  curated/provider badge from the first run overview.
  Rationale: workflow position is more stable and useful than provider return
  order, while recency makes retries discoverable inside that position. The
  overview should spend its limited hierarchy on measured training evidence;
  semantic analysis needs a deliberate researcher workflow before returning to
  the interface.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Author work-package and job-definition descriptions on their
  versioned contracts and copy them into every run snapshot. Observatory reads
  only the snapshotted text and leaves descriptions explicitly missing for
  historical runs that predate the fields.
  Rationale: package and implementation purpose are necessary interpretation
  context, but resolving mutable catalog or code text at read time would make
  historical evidence change meaning after it was produced.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Treat project as the active Observatory scope and work package as
  the persistent run-grouping and navigation unit. A work-package page
  summarizes job-kind groups, run outcomes, and artifact lineage, but does not
  infer a package conclusion from successful executions when the owner,
  decision question, and conclusion are absent from the read projection.
  Rationale: the canonical hierarchy is project → work package → run; a flat
  cross-project run list obscures the decision boundary and makes retries look
  like unrelated work.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Resolve run views with `mode=auto|job|generic`. `auto` selects a
  registered job-specific projector and otherwise returns the generic view.
  `generic` is available for every run. `job` returns a typed unavailable error
  when no definition exists.
  Rationale: known work should be interpreted consistently, while new or
  provider-specific jobs must remain inspectable without pretending that raw
  metric names carry known semantics.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: The deterministic generic projector may group metric names, show
  scalar/latest values, and plot explicitly selected numeric series. It may not
  create health rules, semantic labels, comparison keys, recommendations, or
  success criteria. Optional LLM analysis is a separate, explicitly generated
  response and may offer cited interpretations or hypotheses, but it does not
  mutate the generic view or become a telemetry definition.
  Rationale: the generic evidence browser must remain repeatable, while a
  clearly labeled and grounded analysis layer can still help experts understand
  novel jobs without silently turning probabilistic output into product truth.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Support semantic summaries for both registered and generic jobs
  behind a provider-neutral `SemanticSummaryProvider`. Generation is explicit,
  bounded, redacted, and non-authoritative. Every claim must cite evidence IDs
  from the supplied bundle and be labeled `observation`, `inference`, or
  `hypothesis`; invalid citations reject the result. Model, endpoint class,
  prompt version, job-view schema version, evidence fingerprint, generation
  time, and completeness are returned as provenance.
  Rationale: LLMs are useful for synthesis, especially over unfamiliar metric
  sets and large trace populations, but experts must be able to distinguish a
  measured fact from a model interpretation and reproduce the input snapshot.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Do not generate semantic summaries automatically during run-list,
  run-view, alert, or polling requests. Generate on demand, cache by the complete
  evidence and model fingerprint, and expose `disabled`, `unavailable`,
  `generating`, `ready`, `stale`, and `failed` states. The product remains fully
  functional with semantic analysis disabled.
  Rationale: explicit generation controls cost, latency, data disclosure, and
  nondeterminism, while preventing a monitoring refresh loop from repeatedly
  invoking a model.
  Date/Author: 2026-07-22 / Codex.

- Decision: Use a discriminated union for product responses rather than a
  mostly-optional universal `RunView` model. The first view variants are
  `job.metrics`, `job.evaluation`, and `generic`.
  Rationale: strict variants keep HTTP, MCP, TypeScript, and Python consumers
  honest about which fields exist and prevent a generic fallback from looking
  like an incomplete job-specific view.
  Date/Author: 2026-07-22 / Codex.

- Decision: Keep pure, versioned `JobViewDefinition` data beside projector
  code. SFT, DPO, and GRPO use the shared metric projector; general and domain
  evaluation use a Verifiers evaluation projector. A registry maps exact job
  kinds to a definition and projector.
  Rationale: most job views are data-driven, while evaluation needs trace
  population behavior that cannot be expressed as a list of metric names.
  Date/Author: 2026-07-22 / Codex.

- Decision: Treat a Verifiers trace as the atomic evaluation record. Compute
  reward distributions, pass rates, slice summaries, failures, truncations,
  tool-use distributions, and rubric breakdowns from a complete trace
  population. Surface an explicit `complete|partial|unavailable` population
  state and never present a partial scan as a complete evaluation.
  Rationale: canonical observation guidance says to persist values once at the
  lowest trustworthy grain and compute higher-level views. This also makes an
  aggregate anomaly directly traceable to its evidence.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Implement a generic trace projector for all trace types and a
  narrow `VerifiersV1TraceProjector` inside Observatory. The projector validates
  the fields it consumes, retains the raw record, and degrades to generic trace
  detail when a record version is unknown or malformed. Do not import the full
  evaluation engine or provider SDK into product logic.
  Rationale: Observatory needs Verifiers semantics without coupling its server
  image to training/evaluation runtime dependencies or making one malformed
  record break the run page.
  Date/Author: 2026-07-22 / Codex.

- Decision: The focused run brief is the default desktop run page. Its lower
  evidence sections use the analytical-notebook flow, but there is no separate
  notebook route in the first release.
  Rationale: monitoring requires a decisive first screen; deeper investigation
  still benefits from a documented, sequential evidence narrative.
  Date/Author: 2026-07-22 / Codex, resolving the mood-board open decision for
  this implementation plan.

- Decision: Introduce a configured `source_id` distinct from provider name and
  canonical `run_id`. Every UI/API run reference resolves through
  `(source_id, run_id)`, and cross-source comparisons retain both values.
  Rationale: one Observatory may read multiple Trackio projects or W&B
  namespaces, and canonical run IDs alone are not guaranteed globally unique.
  Date/Author: 2026-07-22 / Codex.

- Decision: Remove arbitrary SQL from the product. The replacement CLI and MCP
  surfaces expose typed run, work-package, metric, trace, lineage, and export
  queries only.
  Rationale: raw Trackio SQL is provider-specific, difficult to secure, and
  impossible to offer consistently for W&B.
  Date/Author: 2026-07-22 / user and Codex via ADR 0012.

- Decision: Keep Observatory read-only. Notes shown in explorations remain
  out of the first release unless a separate owned annotation store is
  approved; no tracking backend is used as an accidental product database.
  Rationale: execution decisions and mutations are outside Observatory's
  accepted boundary, and provider parity would otherwise be lost.
  Date/Author: 2026-07-22 / Codex.

- Decision: Desktop is the primary surface. Responsive work must keep run
  status, alerts, compact summaries, and trace links usable on narrow screens,
  but comparison tables and dense trace investigation may require desktop.
  Rationale: the ML-expert workflows depend on coordinated charts, evidence
  tables, and inspectors that do not remain legible when compressed.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Use React 19 and Tailwind CSS 4 for the Observatory frontend and
  remove the earlier Svelte implementation completely. Use Apache ECharts core
  directly for dense metric, distribution, and scatter plots; use TanStack
  Table with TanStack Virtual for sortable, filterable, bounded trace
  populations. Do not add a framework-specific chart wrapper.
  Rationale: React and Tailwind match the intended team distribution stack.
  ECharts supports Canvas rendering and coordinated scientific interactions at
  the data densities expected from training and system telemetry, while the
  headless TanStack pair gives the trace workspace accessible semantic markup
  and controlled virtualization without imposing a generic admin-dashboard
  visual system.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Treat `System metrics` and `Traces & evaluation` as first-class
  run sections, not generic metric leftovers. System metrics are a cross-job
  projection of canonical `system/*` and `tracking/*` series with utilization,
  memory, throughput, and ingest-health groupings. Trace views adapt to the run:
  GRPO shows rollout evidence, evaluation shows trace-derived evaluation, and
  jobs without traces show a truthful unavailable state.
  Rationale: an ML expert needs to distinguish algorithm behavior, runtime
  saturation, and example-level failures without manually rediscovering metric
  namespaces. Keeping all projections server-defined also preserves the shared
  UI/MCP/Python semantics.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: Use self-hosted Inter Variable for dense interface text and
  Instrument Serif for editorial headings and evidence values. Use Radix
  Popover for compact filter controls while keeping the visual layer in
  Tailwind and the shared semantic tokens.
  Rationale: the accepted images depend on a deliberate editorial-versus-UI
  typographic contrast and compact, accessible controls; relying on undeclared
  system fonts and native selects caused visible cross-platform drift.
  Date/Author: 2026-07-22 / user and Codex.

- Decision: The local read-only OCI composition copies only the selected
  Trackio SQLite database and WAL sidecar from the read-only host mount into a
  tmpfs snapshot at startup. It does not mount the 4.5 GiB evidence store
  writable or duplicate artifact payloads.
  Rationale: the current Trackio query API opens a process lock and configures
  SQLite pragmas even for reads. A bounded snapshot preserves Observatory's
  read-only security boundary and is sufficient for run, metric, trace, and
  artifact-lineage metadata. Restarting the local composition refreshes the
  snapshot.
  Date/Author: 2026-07-22 / Codex.

- Decision: Supersede the local-snapshot implementation detail above.
  Observatory never opens Trackio storage directly. The Trackio service owns
  the read-only evidence mount and exposes both ordinary and system history;
  Observatory queries that provider API and includes only samples whose
  timestamps fall inside the canonical run start/finish interval. Do not add a
  second writer or synthesize start/finish samples for this projection.
  Rationale: provider storage belongs behind the tracking reader, existing
  per-run host samples are already sufficient, and time-window clipping avoids
  attributing activity outside the selected run.
  Date/Author: 2026-07-22 / user and Codex.

## Outcomes & Retrospective

The first complete Observatory product slice now runs as one React/Tailwind
frontend and provider-neutral Python service. It serves curated SFT, DPO, GRPO,
and evaluation views; a truthful unknown-job metric selector; first-class
system telemetry; trace-derived evaluation with example detail; lineage; and
explicit fixture-backed semantic summaries through the same HTTP, MCP, Python,
CLI, and export models. The legacy reports package has been removed.

The visual and usability gates materially improved the implementation. The
focused brief now carries selected-step context inside the analytical surface
and uses the right rail for input-to-output lineage. DPO opens on preference
reward evidence, native trace selects have become compact accessible popovers,
narrow screens contain their own navigation, and trace errors remain distinct
from verifier outcomes. The final visual gate used real schema-v4 Trackio and
W&B SFT, DPO, system, and lineage evidence; deterministic fixtures remain only
for trace, GRPO, evaluation, and unknown-job journeys that those training runs
do not contain.
The audits and Codex in-app Browser captures are preserved beside the frontend.
The artifact surface now renders the baseline's actual lineage primitive—run
consumed/produced artifact edges—instead of presenting a flat inventory under a
lineage label. System telemetry now also answers which runtime phase was active
when host memory and utilization were observed. The phase analysis is derived
from explicit provider-neutral boundaries and existing provider samples, so it
preserves backend parity and leaves historical runs visibly incomplete instead
of guessing. New runs additionally retain their exact execution-target
revision. The System metrics page uses that declared hardware context for a
capacity fit view and keeps a separate raw interval timeline for overlap and
debugging; historical runs that retained only a target ID stay visibly partial.

The OCI image builds from pinned Python and JavaScript locks, runs as UID 10001
with a read-only filesystem and dropped capabilities, and is healthy at
`127.0.0.1:7861`. The credentialed W&B conformance, real training, system
history, and browser gates now pass. Remaining release work is deliberately
narrower: exercise provider-backed high-volume traces, complete the comparison
and alert frontend journeys, and run a dedicated keyboard/screen-reader pass.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. It is a Python 3.12 `uv`
workspace. `apps/observatory` is the read product. `packages/tracking` defines
provider-neutral evidence models and the `RunDataSource` protocol.
`packages/tracking-trackio` and `packages/tracking-wandb` translate Trackio and
W&B storage into that protocol. Capability packages such as train and eval emit
evidence but do not query Observatory.

The current Observatory package contains four modules:

    apps/observatory/src/posttrain_observatory/models.py
    apps/observatory/src/posttrain_observatory/telemetry.py
    apps/observatory/src/posttrain_observatory/service.py
    apps/observatory/src/posttrain_observatory/cli.py

`telemetry.py` defines SFT, DPO, GRPO, general-eval, and domain-eval metric
schemas. `service.py` builds summaries, charts, alerts, deltas, and same-kind
comparisons. It is already async on the read side and uses strict frozen
Pydantic models. Preserve those properties while splitting the package by
responsibility.

A **job-specific view** is an opinionated projection registered for an exact
`job_kind`. It may name important metrics, compute documented reducers, apply
health rules, declare comparison fields, interpret known trace schemas, and
assign artifact roles. A **generic view** is a safe evidence inspector that
works without a registered job definition. It exposes names and recorded
values, but its deterministic fields do not infer meaning. A **semantic
summary** is an optional LLM-generated interpretation over a finite,
fingerprinted evidence bundle. Its claims cite bundle items and clearly separate
observations from inferences and hypotheses. A **projector** is application code
that turns raw normalized evidence into one strict product view. A **source** is
one configured Trackio or W&B reader connection, identified by a stable
operator-chosen `source_id` such as `trackio-local` or `wandb-research`.

The five first-class job kinds are:

- `train.sft`: loss, learning rate, gradient norm, throughput, optimization and
  performance charts, missing/non-finite loss alerts, and trained artifacts.
- `train.dpo`: loss, reward margin, chosen/rejected rewards, preference charts,
  negative-margin and non-finite alerts, and trained artifacts.
- `train.grpo`: reward mean/std, KL, entropy, clip fraction, optimization and
  policy charts, rollout traces, and trained artifacts.
- `eval.general` and `eval.domain`: run counters plus a unified Verifiers
  traces/evaluation view, slice summaries, rubric/reward breakdowns, trace-sync
  health, and evaluation artifacts.

Other current or future kinds—including serve benchmarks, data preparation,
model transforms, and team-specific jobs—open in generic mode until an explicit
definition and projector are registered. Known jobs also expose a Metrics tab
that invokes generic mode so experts can inspect series omitted from the
curated default.

## View Resolution and Product Behavior

The application service receives a `RunLocator(source_id, run_id)` and a view
mode. It first loads the source and `RunDetail`. In `auto` mode it looks up the
exact `job_kind` in `JobViewRegistry`. A match invokes that projector; no match
invokes `GenericRunProjector` and returns a short fallback reason. `job` mode
requires a match. `generic` bypasses the job registry. The resolver never
matches a kind by provider name, fuzzy text, or metric-name heuristics.

The generic view always includes the located run summary, resolved inputs and
source metadata after redaction, events, artifacts, trace types/count and reader
capabilities. It groups `RunDetail.metric_names` by the prefix before the first
slash and returns a deterministic metric catalog. Series are loaded only for
explicit names supplied by the consumer, with a small validated maximum. The
HTTP and MCP defaults return metric names and latest tips, not every point.
Series requests accept a logical-step or time window and a maximum number of
points. When downsampling is necessary, bucketed min/max sampling preserves
extrema and the response says that it was downsampled. Raw export remains a
separate explicit operation.

The frontend route for a run has one shell and these sections:

    Overview                 auto job-specific view or generic fallback
    Metrics                  generic metric catalog and selected series
    Traces & evaluation      trace explorer; evaluation projection when supported
    Artifacts & lineage      consumed and produced edges
    Run config               redacted resolved inputs and source revision

Evaluation is not a peer navigation product. For eval runs, `Overview` includes
the curated evaluation headline and `Traces & evaluation` is the main evidence
workspace. For GRPO it shows rollout evidence without claiming evaluation
completeness. For trace types with no registered projector it shows a generic
table and strict JSON detail.

Semantic analysis is a companion API to these deterministic views, not a fourth
view mode. An API consumer chooses a scope—current run overview, selected
generic metrics, evaluation population, one trace, or a comparison—and requests
a summary. Frontend invocation is deferred until that researcher workflow has a
clear placement. Observatory first constructs a `SemanticEvidenceBundle` from the same
application models already visible to that user. Each bundle item receives a
stable local evidence ID, such as `metric:train/loss:window-200-400`,
`alert:loss_non_finite`, or `trace:rollout-17:reward`. The LLM sees only the
bounded, redacted bundle and the registered job vocabulary when one exists. A
generic bundle does not pretend that a metric name has a known unit or favorable
direction; the generated response may propose an interpretation only as an
inference or hypothesis.

The product validates the structured model response before returning it. Every
claim must cite at least one supplied evidence ID, all cited IDs must exist, and
the response cannot add alerts, alter status, or write decisions. A cached
summary is marked stale when the run evidence fingerprint, selection window,
job-view schema, prompt, or model changes. If no provider is configured, the
button and APIs report `disabled`; all deterministic views continue to work.

## Plan of Work

### Milestone 1: Create the product query layer and retire reports

Split the current package without changing behavior first. Create
`apps/observatory/src/posttrain_observatory/domain/` for strict view models,
`application/` for services and projectors, `definitions/` for versioned job
definitions, `composition.py` for provider construction, `transports/` for
HTTP/MCP/CLI, and `exports/` for materialized outputs. Keep public re-exports in
`posttrain_observatory/__init__.py` small and intentional.

Create `domain/locations.py` with `RunLocator`, `LocatedRunSummary`, and
`SourceSummary`. Create `application/sources.py` with `RunSourceRegistry`, which
owns a mapping from stable `source_id` to `RunDataSource`, resolves a locator,
and merges filtered run lists without hiding the source. It must bound
concurrency, isolate source failures, and return typed source-health results so
one unavailable backend does not erase healthy sources.

Port the useful models from
`packages/reports/src/posttrain/reports/work_packages.py` into
`domain/work_packages.py` as strict Pydantic models. Implement
`application/work_packages.py` exclusively through `RunSourceRegistry`,
`RunDataSource.list_runs`, `get_run`, and `artifacts`. Preserve project,
work-package, stage/job grouping, status, metric-name discovery, and input/output
lineage. A work-package query may span configured sources and must retain each
run locator. Do not copy `_CanonicalRunConfig`, Trackio table names, or SQL.

Add JSON and CSV exporters that consume product models. Replace `trackio-query`
with typed commands such as `posttrain-observatory runs`, `run`,
`work-package`, and `export`. Once migrated tests pass, delete
`packages/reports`, remove its test path and dependency declarations, and amend
import-linter so `posttrain.reports` is no longer a root capability package.

This milestone is complete when the same fixture work package can be built from
a fake source and real Trackio adapter without SQL, and no `posttrain.reports`
distribution or arbitrary query command remains.

### Milestone 2: Implement strict view resolution and generic evidence

Replace the current universal `RunView` with strict variants in
`domain/views.py`. Preserve common value types such as `SummaryValue`,
`ChartView`, `RunAlert`, and evidence states, then define a discriminated union:

    type RunView = Annotated[
        MetricJobRunView | EvaluationRunView | GenericRunView,
        Field(discriminator="view_kind"),
    ]

`MetricJobRunView.view_kind` is `job.metrics`,
`EvaluationRunView.view_kind` is `job.evaluation`, and
`GenericRunView.view_kind` is `generic`. Every variant contains a schema
version, `RunLocator`, the normalized run summary, capabilities, and explicit
evidence states. Add `RunViewResponse` with requested mode, resolved mode, and
an optional fallback reason. Add `domain/semantic.py` with strict models for
`SemanticEvidenceItem`, `SemanticEvidenceBundle`, `SemanticClaim`,
`SemanticSummary`, and provenance. Keep semantic output outside the `RunView`
union so deterministic responses remain stable and cacheable independently.

Move and rename `JobTelemetryDefinition` to `JobViewDefinition` under
`definitions/models.py`; add a view schema version and trace projection ID while
retaining summary, chart, health, comparison, delta-tip, and artifact-role
definitions. Create `application/registry.py` with `JobViewRegistry` and a
`RunViewProjector` protocol. Registration rejects duplicate job kinds and
schema-version conflicts at startup.

Create `application/generic.py`. It builds `GenericRunView`, a grouped
`MetricCatalog`, selected `MetricSeriesView` values, generic trace-type
summaries, events, artifacts, and redacted config. Metric selection is exact,
deduplicated, limited, and checked against `RunDetail.metric_names`. Unknown
names return a typed validation error. Do not infer units or favorable
direction. Introduce one shared `RedactionPolicy` that recursively removes
configured secret-key patterns before config, event attributes, or raw trace
fields reach any transport.

Create `application/service.py` as the only public query/intelligence service.
Move the existing alert, reducer, cursor, delta, and comparison behavior into
focused helpers, then add `get_run_view(locator, mode, metrics)`,
`list_run_metrics`, and `get_metric_series`. Keep cursor payloads versioned and
opaque. Add fixture tests for an unknown job, a known job forced to generic
mode, exact metric selection, redaction, source collisions, missing metrics,
downsampling metadata, and partial source failures.

This milestone is complete when an unknown `custom.team_job` run opens in
generic mode, charts an explicitly selected `custom/quality` series, and returns
no invented alerts or comparison keys.

### Milestone 3: Complete job-specific training and comparison views

Create versioned definitions in `definitions/jobs/sft.py`, `dpo.py`, and
`grpo.py`. Migrate the current metric names and rules without changing their
meaning, then extend only from metrics guaranteed by the canonical observation
document. Create `application/projectors/metric_job.py` as the shared projector.
It must request only declared series, preserve missing evidence, apply reducers
and health rules deterministically, and expose artifact roles without changing
the stored artifact identity.

The SFT default figure combines loss with a clearly separated learning-rate
axis and provides gradient norm and tokens-per-second as supporting evidence.
The DPO default figure separates loss from chosen/rejected rewards and makes
reward margin the primary preference signal. The GRPO default figure leads with
mean reward and relates it to KL, entropy, and clip fraction without implying
that higher is always better. System metrics remain available in generic mode
unless a later job definition promotes them explicitly.

Rework comparison into a projector-owned operation. Runs are comparable only
when the exact job kind and view schema version match and any job-specific
compatibility fields declared by the definition agree. Preserve
`available|missing|incomparable` per value. Cross-source comparisons are allowed
because locators retain source identity. Recommendations, Pareto membership, or
promotion actions are not computed in this milestone; the UI may explain
observed trade-offs but cannot decide for the operator.

Add golden logical-model tests for SFT, DPO, and GRPO, including failed,
partial, missing-series, non-finite, and cross-source runs. Execute the same
view fixtures through Trackio and W&B readers and compare normalized product
models after excluding provider metadata.

This milestone is complete when the web-independent service returns stable,
job-specific views and comparisons for all three training kinds while their
unrelated metrics remain accessible only through generic mode.

### Milestone 4: Build unified Verifiers traces and evaluation

Create `domain/traces.py` with `TracePopulationState`, `TraceSummary`,
`TraceFilter`, `TraceDetail`, `RewardComponent`, `RubricOutcome`,
`EvaluationSlice`, and `TraceEvaluationView`. Filters cover trace type,
environment/task slice, verifier outcome, reward range, tool-call count,
truncation, error state, and exact external ID. Unsupported fields remain absent
rather than receiving guessed values.

Create `application/traces/projectors.py` with a generic `TraceProjector`
protocol and `VerifiersV1TraceProjector`. The Verifiers projector validates the
narrow record fields it reads—ID, nodes/messages, reward, reward components,
task/example metadata, truncation/error state, tool calls, token/latency fields,
and rubric results—and retains the original JSON behind an explicit raw-detail
flag. A validation failure becomes a generic trace with a projection warning;
it does not fail the entire population.

Create `application/traces/population.py` to page through `RunDataSource.traces`
until exhaustion, cancellation, or an explicit safety limit. The result records
the number scanned, expected counters when available, next cursor, reader live
capability, trace-sync metric, and `complete|partial|unavailable`. Compute eval
aggregates only when their denominator and completeness are visible. Use a
bounded in-process cache keyed by source, run, trace type, and a source
fingerprint; it is a rebuildable performance cache, never evidence authority.
Invalidate live Trackio entries on a short TTL. W&B post-finish trace artifacts
may use a longer immutable TTL.

Create `application/projectors/evaluation.py` for `eval.general` and
`eval.domain`. It combines run counters with the Verifiers population, computes
reward distribution, completion/failure/truncation counts, pass rate when a
pass rubric exists, task/environment slices, reward components, tool-use and
latency summaries, and trace-sync health. GRPO uses the same trace service for
rollout browsing but retains `job.metrics` as its overview type.

Add service operations `get_trace_evaluation_view` and `get_trace_detail`.
Population responses are projection-shaped and paginated; one trace detail may
include its transcript and raw record after redaction. Add tests for complete
and partial populations, malformed Verifiers records, missing denominators,
slice filtering, reward/rubric aggregation, cursor stability, live Trackio
traces, post-finish W&B trace artifacts, and a direct aggregate-to-selected-
trace drill-down.

This milestone is complete when selecting an underperforming evaluation slice
filters the trace list and opens one rollout whose reward components and rubric
explain the aggregate, while partial trace synchronization is visibly labeled.

### Milestone 5: Add grounded semantic summaries

Create `application/semantic/evidence.py` to build a bounded
`SemanticEvidenceBundle` from an already-authorized product model, never by
querying a provider directly. Support scopes for a job-specific run overview,
explicitly selected generic metrics, evaluation population, one trace, and a
same-schema run comparison. Use deterministic reducers before model invocation:
latest/min/max/mean and slope for selected numeric windows, fired alerts,
evaluation completeness and aggregate values, named artifact roles, and compact
event or trace excerpts. Include missing-evidence and partial-population states.
Assign a stable evidence ID to every supplied fact and compute a fingerprint
over the canonical redacted bundle.

Create `application/semantic/prompts.py` with versioned prompt templates. A
registered job bundle includes the job definition's labels, units, favorable
directions, health rules, and comparison vocabulary. A generic bundle includes
only recorded names and values plus explicit instructions that meaning is
unknown. The output schema contains a short overview, cited findings, possible
causes, suggested next evidence to inspect, limitations, and provenance. Each
finding is classified as `observation`, `inference`, or `hypothesis`; the model
cannot emit a deterministic alert, success criterion, promotion decision, or
configuration mutation.

Define an async `SemanticSummaryProvider` protocol in
`application/semantic/provider.py`. Ship a disabled provider, a deterministic
fake for tests, and one `OpenAICompatibleSemanticSummaryProvider` under
`integrations/semantic/`. The remote adapter uses the existing async HTTP stack
against a configured OpenAI-compatible endpoint, requests strict JSON, applies
timeouts and bounded retries, and validates the response with Pydantic. Keep
endpoint URL, API key environment-variable name, model ID, maximum input/output
tokens, timeout, and concurrency as server settings. The adapter is replaceable;
domain and transport modules do not import its client details.

Create `application/semantic/service.py`. It authorizes and redacts evidence,
builds the bundle, invokes the configured provider, rejects unknown or missing
citations, and returns a typed result. Cache successful results by source/run,
scope, evidence fingerprint, prompt version, job-view schema version, provider
kind, and model ID. The first release uses only a bounded in-process cache; it
does not write generated text into Trackio, W&B, or a new database. Concurrent
identical requests share one in-flight generation. Cancellation stops waiting
without corrupting the cache. Failures produce safe typed states and never make
the underlying run view unavailable.

Add tests for registered and unknown jobs, generic metric selection, evaluation
and trace scopes, comparisons, partial evidence, prompt-injection-shaped config
and trace text, redaction, evidence-budget truncation, valid citations, invented
citations, malformed JSON, timeouts, cancellation, cache hits, staleness, and a
disabled provider. The fixture provider must be deterministic; no network call
is part of the default test suite.

This milestone is complete when an SFT run and an unknown custom run both
produce clearly labeled, cited semantic summaries; the SFT summary uses its
registered vocabulary, the generic summary marks meaning as inferred, and an
invented citation is rejected without affecting either deterministic view.

### Milestone 6: Expose one versioned service through HTTP, MCP, CLI, and exports

Add FastAPI and the stable MCP Python SDK bounds already recorded by the
umbrella plan, then resolve exact versions in `uv.lock`. Create
`transports/http.py` with `create_http_app(service, settings)` and a versioned
`/api/v1` router. Create `transports/mcp.py` from the same service. Do not call
HTTP from in-process MCP and do not duplicate calculations in a route or tool.

The first HTTP surface is:

    GET  /health/live
    GET  /health/ready
    GET  /version
    GET  /api/v1/sources
    GET  /api/v1/runs
    GET  /api/v1/runs/{run_key}/view?mode=auto|job|generic
    GET  /api/v1/runs/{run_key}/metrics
    POST /api/v1/runs/{run_key}/metric-series
    GET  /api/v1/runs/{run_key}/alerts
    GET  /api/v1/runs/{run_key}/delta
    POST /api/v1/runs/compare
    GET  /api/v1/runs/{run_key}/traces-evaluation
    GET  /api/v1/runs/{run_key}/traces/{trace_id}
    POST /api/v1/runs/{run_key}/semantic-summary
    POST /api/v1/runs/semantic-comparison-summary
    GET  /api/v1/work-packages/{work_package_key}
    GET  /api/v1/job-kinds
    GET  /api/v1/job-kinds/{job_kind}
    POST /api/v1/exports

`run_key` is an opaque, versioned URL-safe encoding of `RunLocator`; consumers
must not construct it manually. List responses provide it. Error responses are
strict Pydantic models with a stable code, safe message, and request ID.

Expose MCP tools named `list_runs`, `get_run_view`, `list_run_metrics`,
`get_metric_series`, `get_run_alerts`, `get_run_delta`, `compare_runs`,
`get_trace_evaluation_view`, `get_trace_detail`, `get_work_package_view`, and
`get_job_view_schema`. Add explicit `summarize_run` and
`summarize_run_comparison` tools; neither is called by another read tool.
Defaults are compact: `get_run_view` returns curated
summary/alerts and chart tips; metric series require explicit names and bounded
points; trace populations omit transcripts; `get_trace_detail` returns one
trace. Raw metric or trace export is always explicit. Support Streamable HTTP
at `/mcp` and local stdio through the CLI.

Extend `cli.py` with `serve`, `mcp`, `runs`, `run`, `work-package`, `summarize`,
`schema`, and `export`. `summarize` requires an explicit locator and scope and
prints the same typed semantic response as HTTP and MCP. Add contract tests
proving Python, HTTP, MCP, CLI JSON, and exporters serialize the same application
model. Generate and commit an OpenAPI artifact for frontend type generation,
and add a clean-diff test for generated schemas.

This milestone is complete when one fixture run returns the same summary keys,
evidence states, alerts, and locator through all product surfaces and no
transport imports a provider SDK.

### Milestone 7: Build the light desktop-first frontend

Pin Node 24.18.0 in `mise.toml`. Create `apps/observatory/frontend` as a React 19,
Tailwind CSS 4, Vite, and TypeScript application with a committed
`package-lock.json`. Generate API types from the committed OpenAPI document.
Use Apache ECharts core directly for line, histogram, scatter, brush,
reference-line, and data-zoom behavior so the product does not depend on an
unowned React chart wrapper. Use TanStack Table and TanStack Virtual for the
trace population and one proven icon library. Resolve and lock exact frontend
versions during this milestone; do not use floating runtime CDN dependencies.

Create design tokens under `frontend/src/lib/styles/` and component primitives
under `frontend/src/lib/components/`. Follow the mood board: warm off-white
light theme, fine dividers, restrained accent colors, clean plot canvases,
compact controls, and selective editorial typography. Create original,
licensed Observatory texture and technical-line assets rather than copying the
Hex source. Decorative texture may appear in quiet shell/header areas but never
behind charts, tables, or trace transcripts. Self-host any font and record its
license; the product must render without public font/CDN access.

Implement routes for the run list, work package, run detail, comparison, and
alerts. The run detail uses the focused brief as its first viewport and the
shared section navigation described above. Add dedicated `System metrics` and
`Traces & evaluation` sections. The former consumes a cross-job server
projection of canonical `system/*` and `tracking/*` evidence; the latter uses
the trace/evaluation projection and opens an individual trace inspector.
Components render from the
discriminated API response. They do not contain their own SFT/DPO/GRPO metric
lists. The generic Metrics view provides namespace search, explicit series
selection, axis compatibility warnings, step/time windows, raw/latest values,
and downsampling disclosure. The traces/evaluation view links population charts
to a paginated trace table and selected-trace inspector. Add a semantic-analysis
panel that starts in an ungenerated state, explains which evidence scope will be
shared, and requires an explicit action. Render claim type, inline evidence
links, limitations, model/prompt provenance, generation time, completeness, and
staleness. Clicking a citation scrolls to or opens the exact metric window,
alert, trace field, or artifact. Never visually merge generated claims into the
deterministic alert list.

At widths below the desktop breakpoint, collapse navigation and stack summary
and alert content. Preserve status, key metrics, alert review, and trace links.
Show a clear “Open on desktop for comparison” state rather than compressing
dense comparison tables into illegibility. Keyboard focus, contrast, reduced
motion, table semantics, chart text alternatives, and non-color evidence states
are release requirements.

Add Vitest component tests, accessibility checks, and browser flows at
1440x1024 plus one narrow status/alert viewport. Validate job-specific SFT,
DPO, GRPO, evaluation, unknown/generic, missing evidence, provider failure, and
partial trace-sync states. Save final screenshots under
`artifacts/observatory/validation/` and link them in this plan. Compare the
implementation visually against the selected mood-board direction before
acceptance. After visual QA passes, run a separate usability audit across SFT,
DPO, GRPO, eval, generic, trace-detail, and system-metric investigation flows.
Visual fidelity and workflow usefulness are separate release gates.

This milestone is complete when a human can navigate from run list to a
job-specific brief, switch the same run to generic metrics, drill from an eval
slice into one trace, and compare same-kind runs without frontend-owned metric
semantics. The same user can explicitly generate a cited analysis, inspect its
provenance, and return to the referenced evidence.

### Milestone 8: Compose, secure, package, and operate the product

Create strict settings in `settings.py` for a list of named Trackio/W&B read
sources, service limits, redaction patterns, cache TTLs, bind address, public
base URL, CORS origins, authentication mode, and optional semantic-summary
provider. Tracking and model-provider credentials are read
only by the server from environment variables or mounted secrets. They never
appear in OpenAPI, frontend configuration, MCP tool arguments, logs, exports,
or run config views.

Local Compose may run without authentication only on loopback. A non-loopback
or production deployment must configure an approved authenticated ingress or
an Observatory authorization policy before readiness succeeds. Keep identity
provider choice outside this plan if the organization has not selected one, but
do not silently ship an unauthenticated remote service. Apply the same principal
and source/project authorization checks to HTTP, Streamable HTTP MCP, exports,
and artifact links.

Semantic analysis is disabled by default. Enabling it requires an endpoint,
model ID, secret environment-variable name, source/project allowlist, evidence
and token budgets, timeout, and maximum concurrency. Redaction happens before
the remote call. Treat all run content as untrusted data, delimit it from prompt
instructions, and never give the summarizer tools or credentials. Record safe
audit metadata—principal, run locator, scope, evidence fingerprint, provider
kind, model ID, prompt version, latency, token usage when returned, and
outcome—but never prompt bodies, model responses, secrets, or raw transcripts.

Add structured logs with request/source/run correlation IDs, source health,
query duration, cache outcomes, trace-scan counts, and safe error codes. Add
readiness checks for configured sources without making one optional source
failure take down every healthy source. Expose product build version, OpenAPI
schema version, job-definition versions, source revision, and frontend revision
from `/version`.

Build the frontend and Python service in separate stages of one multi-stage
Dockerfile. The final OCI image contains compiled static assets, both reader
adapters, CA certificates, and the non-root service runtime—no Node toolchain,
test data, provider credentials, or build caches. Add a Compose file with local
Trackio and optional W&B configuration examples. Pin production image versions
or digests and do not publish mutable `latest` as a deployment instruction.

Add Docker health, startup, graceful shutdown, source failure, static asset,
HTTP, and Streamable HTTP MCP smoke tests. Generate an SBOM and vulnerability
scan report in release CI. Helm and the thin remote client wheel remain follow-
on artifacts unless this plan is explicitly revised to include them.

This milestone is complete when the same image starts locally through Compose,
serves the frontend/API/MCP, reads a Trackio fixture, reports a disabled or
healthy W&B source explicitly, exposes semantic analysis as disabled unless
configured, and refuses insecure production configuration.

### Milestone 9: End-to-end acceptance and release evidence

Create deterministic Observatory fixtures for SFT, DPO, GRPO, general eval,
domain eval, and an unknown custom job. Each includes enough metrics, events,
traces, artifacts, missing evidence, and alerts to exercise the intended view.
Write them through both real adapters where supported and read them only through
the product service. Add recorded semantic-provider fixtures for job-specific,
generic, evaluation, trace, and comparison scopes; keep live model calls in a
separate credentialed smoke marker.

Run focused package tests after each milestone and the full validation ladder
at completion. Run the credentialed W&B suite against the dedicated test
project. Start the OCI image and execute the Playwright and MCP smoke tests
against it. Record exact pass/skip counts, W&B run IDs/URLs without credentials,
image digest, OpenAPI checksum, MCP schema checksum, screenshots, Lighthouse or
accessibility results, and known provider capability differences in `Artifacts
and Notes`.

Release only when unknown jobs degrade to generic mode, registered jobs never
silently degrade because of projector bugs, deterministic generic mode never
invents semantics, generated interpretations are cited and visibly distinct,
partial trace populations remain visibly partial, Trackio and W&B produce
equivalent logical views for common evidence, and no provider secret or
unredacted sensitive key is present in client-visible data or model requests.

## Concrete Steps

Work from `/home/hammad/projects/rl` unless a command says otherwise. At the
start of each milestone, confirm the dirty worktree and preserve unrelated user
changes:

    git status --short
    git diff --check

Validate the existing foundation before restructuring:

    uv sync --all-packages --locked --python 3.12
    uv run pytest apps/observatory/tests packages/tracking/tests \
      packages/tracking-trackio/tests packages/tracking-wandb/tests \
      -m "not network" -q
    uv run ruff check apps/observatory packages/tracking \
      packages/tracking-trackio packages/tracking-wandb
    uv run pyright apps/observatory packages/tracking
    uv run lint-imports

After report migration, prove removal rather than leaving a shadow product:

    uv run pytest apps/observatory/tests -q
    test ! -d packages/reports
    ! rg -n "posttrain\.reports|posttrain-reports|trackio-query" \
      apps packages pyproject.toml docs --glob '!docs/plan/*.md' \
      --glob '!docs/decisions/*.md'

After HTTP and MCP implementation, generate schemas and require a clean diff:

    uv run --package posttrain-observatory posttrain-observatory schema \
      --openapi apps/observatory/openapi.json \
      --mcp apps/observatory/mcp-schema.json
    uv run pytest apps/observatory/tests/api apps/observatory/tests/mcp -q
    git diff --exit-code -- apps/observatory/openapi.json \
      apps/observatory/mcp-schema.json

Validate semantic analysis without network access, then run the optional live
provider smoke test only when its dedicated credentials are present:

    uv run pytest apps/observatory/tests/semantic -m "not network" -q
    POSTTRAIN_OBSERVATORY_LLM_API_KEY=... \
      POSTTRAIN_OBSERVATORY_LLM_BASE_URL=... \
      POSTTRAIN_OBSERVATORY_LLM_MODEL=... \
      uv run pytest apps/observatory/tests/semantic -m network -q

Set up and validate the frontend from its own directory:

    mise install
    cd /home/hammad/projects/rl/apps/observatory/frontend
    npm ci
    npm run generate:api
    npm run check
    npm run test
    npm run build
    npm run test:e2e

Return to the repository and validate the packaged product:

    cd /home/hammad/projects/rl
    docker compose -f apps/observatory/compose.yaml build observatory
    docker compose -f apps/observatory/compose.yaml up -d
    uv run pytest apps/observatory/tests/e2e -m docker -q
    docker compose -f apps/observatory/compose.yaml down

The credentialed release gate uses environment variables already documented by
the tracking adapter and must never echo the key:

    WANDB_API_KEY=... WANDB_ENTITY=... \
      POSTTRAIN_WANDB_TEST_PROJECT=posttrain-conformance \
      uv run pytest packages/tracking-wandb/tests \
      apps/observatory/tests/integration -m network -q

Run the full repository ladder last:

    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

If full Pyright still requires optional GPU/data extras, record that environment
gap and run the exact changed-path Pyright command as a required gate. Do not
describe optional or credentialed coverage as passing when it was skipped.

## Validation and Acceptance

The implementation is accepted when all behavior below can be observed.

Starting Observatory with a fixture source shows a run list containing source,
provider, project, work package, stage, job kind, status, start time, duration,
and alert count. Two configured sources with the same canonical run ID produce
two distinct run keys and remain navigable.

Opening SFT, DPO, and GRPO runs in `auto` mode returns `job.metrics` with the
registered schema version, expected summary fields, charts, health rules,
artifact roles, and missing-evidence states. Opening a general or domain eval
returns `job.evaluation`. Opening `custom.team_job` returns `generic` with a
fallback reason, metric catalog, events, traces, artifacts, and config. Forcing
generic mode on SFT exposes its raw metric catalog without job alerts. Forcing
job mode on the custom run returns a stable unavailable error rather than HTTP
500.

The generic metric workspace plots only requested names, rejects unknown names,
discloses downsampling, preserves extrema, and does not label a metric good,
bad, loss, reward, or throughput unless its registered job definition does so.
Metric, event, config, trace, and artifact values are redacted consistently
before Python model serialization.

With semantic analysis disabled, every deterministic route and screen remains
usable and the summary operation returns a typed `disabled` response. With the
fixture provider enabled, an SFT summary uses registered loss/optimization
vocabulary, while a `custom.team_job` summary treats metric meaning as inferred.
Every returned finding cites supplied evidence; citation links resolve to the
exact visible evidence. A response with an invented citation is rejected. The
UI never renders generated claims as alerts, measured status, or acceptance
decisions, and a changed evidence fingerprint marks the previous result stale.

The eval page reports exactly how many traces were expected, scanned, and
included. A partial sync is labeled partial. Selecting an underperforming slice
filters the trace table. Selecting one trace exposes its transcript, verifier
outcomes, reward components, tool calls, token/latency metadata, flags, and
artifact links. Recomputing the displayed aggregate from selected fixture
traces produces the same value. A malformed or unknown trace version remains
available through generic detail with a projection warning.

Same-kind comparisons retain source identity and per-cell evidence state.
Different job kinds or schema versions are incomparable with an explicit
reason. The product does not persist or execute promote/reject decisions.

Python, HTTP, MCP, CLI JSON, frontend, and exports agree on view kind, schema
version, summary keys, evidence states, alert IDs, and run locator. MCP calls
remain compact by default and never return all metric points or all trace
transcripts without explicit bounded selection.

The desktop UI visibly matches the accepted mood-board direction: light warm
surfaces, restrained texture outside evidence canvases, compact controls,
scientific charts, thin dividers, and no generic card wall. Keyboard navigation,
focus visibility, contrast, missing/incomparable states, chart text summaries,
and reduced motion pass automated and manual checks. Narrow viewports retain
status and alerts and clearly redirect dense comparison work to desktop.

The final OCI image runs as non-root, serves frontend/API/MCP from one version,
publishes build/schema metadata, contains no credentials, and passes local
Trackio plus credentialed W&B release gates. A production-mode instance cannot
become ready when exposed remotely without the configured authentication
boundary.

## Idempotence and Recovery

Make additive moves before deletions. Keep report tests running while their
behavior is ported; delete `packages/reports` only after equivalent Observatory
tests pass. If a caller remains, migrate it rather than adding logic to an
import shim. Never reset the dirty worktree or discard unrelated edits.

View models and routes are versioned. During an internal model migration, update
Python tests, OpenAPI, MCP schema, generated TypeScript, fixtures, and frontend
in one milestone. If schema generation leaves a diff, review and commit the
intentional change; do not hand-edit generated TypeScript.

Source failures are isolated. A failed optional W&B source can be retried after
credentials or network recover without rebuilding the product. The trace cache
is disposable and can be cleared safely because provider evidence remains the
authority. Cursors are opaque and may become invalid after a schema version
change; return a typed stale-cursor response that tells the client to restart
the query.

Semantic-summary caches are also disposable. Retrying a failed request rebuilds
the same redacted bundle; concurrent identical retries share one provider call.
Changing prompt version, model ID, source revision, evidence window, or job-view
schema produces a new key instead of overwriting the old result. Disabling the
provider immediately prevents new outbound calls and leaves deterministic
observability intact.

Frontend installs use `npm ci`. OCI builds are multi-stage and repeatable from
`uv.lock` and `package-lock.json`. Compose teardown must not delete provider
data unless an explicit test-volume cleanup command is requested. External W&B
test runs are uniquely tagged and are not deleted automatically.

## Artifacts and Notes

Current implementation anchors:

    apps/observatory/src/posttrain_observatory/models.py
    apps/observatory/src/posttrain_observatory/telemetry.py
    apps/observatory/src/posttrain_observatory/service.py
    apps/observatory/src/posttrain_observatory/cli.py
    apps/observatory/tests/test_service.py
    packages/tracking/src/posttrain/tracking/contracts.py
    packages/tracking/src/posttrain/tracking/models.py
    packages/reports/src/posttrain/reports/query.py
    packages/reports/src/posttrain/reports/work_packages.py
    packages/reports/tests/test_reports.py

Design anchors:

    docs/design/observatory/moodboard/hex-homepage-light-texture-reference.png
    docs/design/observatory/moodboard/observatory-focused-run-brief.png
    docs/design/observatory/moodboard/observatory-analytical-notebook.png
    docs/design/observatory/moodboard/observatory-experiment-comparison.png
    docs/design/observatory/moodboard/observatory-traces-and-evaluation.png

Append exact validation transcripts, fixture IDs, external run URLs without
secrets, image digest, OpenAPI/MCP checksums, accessibility results, and final
screenshot links here as implementation proceeds. For live semantic smoke
tests, record only provider kind, model ID, prompt version, evidence fingerprint,
latency, token counts when available, and pass/fail outcome—not prompts, raw
responses, secrets, or sensitive evidence.

Implementation evidence from 2026-07-22:

- Observatory Python tests: 15 passed; Ruff and Pyright passed.
- Full local repository suite excluding explicit network, Docker, and GPU
  markers: 155 passed, 5 deselected. Root Ruff, Pyright, import contracts, and
  `git diff --check` passed.
- Frontend: TypeScript check passed, Vitest 1 passed, production Vite build
  passed, and `npm audit --omit=dev --audit-level=high` reported no production
  vulnerabilities.
- Visual audit: `apps/observatory/frontend/design-qa.md` with captures under
  `artifacts/observatory/validation/visual-audit/`.
- Usability audit: `apps/observatory/frontend/usability-audit.md` with captures
  under `artifacts/observatory/validation/usability-audit/`.
- OCI image: `sha256:45d2d78904081b21c15e274050ab29815f215932296933dba73b6bc955877ba1`,
  507576083 bytes, healthy as UID 10001.
- OpenAPI SHA-256:
  `2396be0b047289a5523121297b7e387ad7ce1ca549f52d337ccd5645929062b6`.
- MCP schema SHA-256:
  `26a3c52e0cdd5147537a346b940ca4cecdd82f8cc00cc3767c344da536bbe4e2`.
- Execution-target and phase-capacity slice, 2026-07-23: 82 Observatory/lab
  tests passed; the full repository suite passed with 259 tests and 5
  credential/network tests skipped by policy. Root Ruff, Pyright, all eight
  import contracts, `git diff --check`, frontend TypeScript, 13 Vitest tests,
  and the production Vite build passed. The live fixture API reported a
  24 GiB declared capacity, 15.3 GB observed peak, one complete target, and
  four raw phase intervals. The Codex in-app Browser declined the localhost
  reload under its URL safety policy, so no browser screenshot is claimed for
  this slice.

## Interfaces and Dependencies

Across Milestones 2 through 6, `posttrain_observatory` exports or internally
owns interfaces equivalent to:

    class RunLocator(ObservatoryModel):
        source_id: str
        run_id: str

    class RunSourceRegistry:
        async def list_runs(self, query: RunQuery) -> tuple[LocatedRunSummary, ...]: ...
        def resolve(self, locator: RunLocator) -> RunDataSource: ...

    class RunViewProjector(Protocol):
        async def build(self, context: RunEvidenceContext) -> RunView: ...
        async def compare(self, contexts: tuple[RunEvidenceContext, ...]) -> RunComparison: ...

    class JobViewRegistry:
        def register(self, definition: JobViewDefinition, projector: RunViewProjector) -> None: ...
        def resolve(self, job_kind: str) -> RegisteredJobView | None: ...

    class ObservatoryService:
        async def get_run_view(
            self,
            locator: RunLocator,
            mode: Literal["auto", "job", "generic"] = "auto",
            metrics: tuple[str, ...] = (),
        ) -> RunViewResponse: ...
        async def list_run_metrics(self, locator: RunLocator) -> MetricCatalog: ...
        async def get_metric_series(self, locator: RunLocator, query: MetricSeriesQuery) -> MetricSeriesSet: ...
        async def get_trace_evaluation_view(self, locator: RunLocator, query: TraceEvaluationQuery) -> TraceEvaluationView: ...
        async def get_trace_detail(self, locator: RunLocator, trace_id: str) -> TraceDetail: ...
        async def compare_runs(self, locators: tuple[RunLocator, ...]) -> RunComparison: ...
        async def summarize_run(self, locator: RunLocator, request: SemanticSummaryRequest) -> SemanticSummaryResult: ...
        async def summarize_comparison(self, request: SemanticComparisonSummaryRequest) -> SemanticSummaryResult: ...

At the end of Milestone 5, semantic analysis owns interfaces equivalent to:

    class SemanticSummaryProvider(Protocol):
        async def summarize(
            self,
            bundle: SemanticEvidenceBundle,
            request: SemanticGenerationRequest,
        ) -> SemanticProviderResponse: ...

    class SemanticAnalysisService:
        async def summarize_run(
            self,
            locator: RunLocator,
            request: SemanticSummaryRequest,
            principal: ObservatoryPrincipal,
        ) -> SemanticSummaryResult: ...

`SemanticEvidenceBundle` contains only redacted, authorized, bounded evidence
items with stable IDs and a fingerprint. `SemanticSummaryResult` is a strict
status union for `disabled`, `unavailable`, `ready`, `stale`, and `failed`.
`ready` and `stale` carry a `SemanticSummary`; each `SemanticClaim` has a claim
kind, text, one or more valid evidence citations, and optional calibrated
confidence wording. Confidence is not a numeric probability unless the provider
can supply a documented calibrated value.

Keep `RunDataSource` and raw provider-neutral models in `posttrain.tracking`.
Only `composition.py` imports `posttrain_tracking_trackio` and
`posttrain_tracking_wandb`; application, domain, transport, export, and frontend
code do not import provider SDKs. Import-linter continues to forbid `trackio`
and `wandb` under `posttrain_observatory` while allowing the adapter packages
themselves as composition dependencies.

Only `integrations/semantic/` imports or implements the remote model HTTP
contract. The semantic service consumes application view models and cannot call
a `RunDataSource` directly. Transports call `ObservatoryService`; the frontend
never receives model credentials or constructs model prompts.

Python runtime dependencies are Pydantic 2, the existing tracking packages,
FastAPI within the compatible line recorded by the umbrella plan, the stable
MCP Python SDK below its next major version, and the selected production ASGI
server. Semantic analysis uses the workspace's async `httpx` line for its first
OpenAI-compatible remote adapter rather than binding domain code to one model
vendor SDK. Record the supported request/response contract in adapter tests.
Add dependencies only when their milestone begins and commit exact resolutions
in `uv.lock`. The frontend uses React 19, Tailwind CSS 4, Vite, TypeScript,
ECharts core, TanStack Table, TanStack Virtual, generated OpenAPI types, Vitest,
Playwright, and accessibility tooling, with exact resolutions in
`package-lock.json`. Node 24.18.0 is pinned in `mise.toml` and the OCI frontend
builder.

No generic view or frontend component may import a job definition. No job
definition may import a transport. No HTTP route, MCP tool, CLI command, or
exporter may query a `RunDataSource` directly; they call `ObservatoryService`.

Revision note (2026-07-22 04:44Z): Created the dedicated Observatory product
plan after the user required both job-kind-specific views and a generic metrics
view for other work. The plan resolves fallback semantics, makes generic mode
available on every run without inferred meaning, unifies Verifiers traces and
evaluation, selects the focused brief as the default run page, and expands the
umbrella plan into executable query, service, frontend, security, packaging,
and validation milestones.

Revision note (2026-07-22 04:55Z): Revised the deterministic generic-view
boundary after the user identified LLM semantic summaries as a valid product
capability. Added a separate, optional grounded-analysis layer for job-specific
and generic runs, evaluations, traces, and comparisons; required evidence
citations, provenance, redaction, explicit generation, cache fingerprints,
provider abstraction, security limits, UI distinction, transport operations,
and offline/live validation. Deterministic projections and alerts remain the
authoritative observable state.

Revision note (2026-07-22 05:39Z): Replaced the provisional Svelte frontend
direction with the user-selected React 19 and Tailwind CSS 4 stack. Added
Apache ECharts plus TanStack Table/Virtual as the bounded visualization and
trace investigation stack, promoted trace and system telemetry into dedicated
run sections, and split visual fidelity and job-by-job usability into separate
evidence-backed release audits.

Revision note (2026-07-22 06:04Z): Completed the React/Tailwind product slice,
recorded the visual and job-by-job usability audits, added the first-class
system and trace projections, and verified the locked non-root OCI/Compose
runtime. The remaining release gate is credentialed and scale validation plus
the unfinished comparison and alert frontend journeys.

Revision note (2026-07-22 06:20Z): Added a semantic light-theme token layer and
rebalanced the application typography after browser review found compressed
metadata and inconsistent jumps between labels, body copy, and display values.
Re-verified the SFT and trace workspaces plus the 375 by 812 narrow layout.
