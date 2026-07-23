# Make DPO evidence requirements enforceable and backend-neutral

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this change, an ML researcher can open a DPO run in Observatory and see whether the run captured enough evidence to diagnose preference learning, optimization stability, data quality, and runtime behavior. Missing evidence is no longer represented only by blank cards: Observatory evaluates a versioned job contract and classifies the run as complete, partial, or insufficient. The contract is provider-neutral, so a future veRL DPO adapter must produce the same logical evidence as the current TRL adapter even when its native metric names differ.

This work does not change the frozen post-training product hierarchy or add veRL as a supported DPO executor. It extends the existing job-aware observation implementation under the meanings already established by `docs/post-training/06-observation-and-lineage.md`.

## Progress

- [x] (2026-07-22 10:52Z) Re-read the canonical observation baseline, inspected the current DPO adapter, telemetry schema, service projection, frontend, and dirty worktree.
- [x] (2026-07-22 11:04Z) Defined strict required, conditional, and diagnostic evidence requirements plus product-facing completeness models.
- [x] (2026-07-22 11:07Z) Evaluated DPO completeness from normalized provider metrics and resolved run inputs, including explicit inactive conditional states.
- [x] (2026-07-22 11:09Z) Extended TRL DPO data profiling with deterministic length p95 values and minimum max-length headroom.
- [x] (2026-07-22 11:13Z) Presented compact DPO evidence completeness and the leading missing group in the React product and generated API types.
- [x] (2026-07-22 11:18Z) Added package-specific Python and frontend tests, ran the focused validation ladder, rebuilt Compose, and inspected the real Trackio DPO run in the Codex in-app browser.

## Surprises & Discoveries

- Observation: the TRL callback already records every numeric trainer log under `train/*`, so loss, rewards, log-probabilities, entropy, gradient norm, and learning rate do not need duplicate DPO-specific emission code.
  Evidence: `packages/train/src/posttrain/train/backends/trl/common.py` maps every finite numeric log entry and derives clipping plus effective token throughput.

- Observation: current preference rendering rejects over-limit examples instead of truncating them.
  Evidence: `packages/train/src/posttrain/train/rendering.py::render_preferences` raises when prompt plus completion length exceeds `max_length`. DPO therefore needs an over-limit rejection/preflight counter rather than an SFT-style truncation rate until truncating preference rendering is deliberately introduced.

- Observation: `apps/observatory/openapi.json` is a committed generated boundary; the frontend build consumes it but does not refresh it from the Python application.
  Evidence: `posttrain-observatory schema --openapi apps/observatory/openapi.json` was required before `npm run generate:api` could expose the new completeness models to TypeScript.

- Observation: the view route's generic `dict[str, object]` return annotation hid the product models from OpenAPI, and the tracking package's recursive `JsonValue` schema was legal OpenAPI but generated an invalid self-referential TypeScript property.
  Evidence: the route now returns `RunViewResponse`, the schema retains arbitrary JSON as an intentionally opaque value, and a transport test asserts both the completeness reference and non-recursive JSON representation.

- Observation: the existing real DPO smoke run predates static pair-profile and derived runtime evidence.
  Evidence: Observatory classified `train.dpo-efade9e3` as `insufficient`, with four of six required groups available and the rendered-pair population identified as the leading missing group.

## Decision Log

- Decision: make Observatory's `JobTelemetryDefinition` the backend-neutral evidence contract and keep provider mapping inside backend adapters.
  Rationale: the researcher-facing questions and completeness rules must not inherit TRL or veRL storage names. Observatory already owns job meaning while tracking owns raw evidence.
  Date/Author: 2026-07-22 / Codex

- Decision: use three requirement levels: required, conditional, and diagnostic. Conditional requirements declare an explicit activation condition and become completeness-gating when that condition is active.
  Rationale: a single optional flag hides important distinctions between unavailable evidence and genuinely inapplicable evidence.
  Date/Author: 2026-07-22 / Codex

- Decision: do not classify absent validation as successful research evidence. A DPO run may be operationally complete without a validation selection, but it is research-ready only when the validation requirement is active and satisfied.
  Rationale: forward-only held-out preference validation answers generalization; training accuracy alone can be memorized, especially in smoke runs.
  Date/Author: 2026-07-22 / Codex

## Outcomes & Retrospective

The slice is complete. DPO now has a provider-neutral, versionable evidence contract shared by Python, HTTP, MCP, and React. The service distinguishes required, active conditional, inactive conditional, and diagnostic evidence; computes `complete`, `partial`, or `insufficient`; and does not call a run research-ready without held-out preference evidence. TRL now records the additional pair-length tail measurements that it can derive without another model pass.

Focused validation passed with 19 Python tests and 4 frontend tests. Ruff, Pyright, all eight import contracts, the frontend production build, and `git diff --check` passed. The live packaged OpenAPI schema was also checked through the in-app browser and exposes `RunView.completeness` without a recursive JSON schema. The build retains the existing non-blocking Vite warning for one approximately 503 KB chart-renderer chunk. The real Trackio DPO run remains honestly insufficient because it predates the new pair-profile and runtime metrics; a new DPO run is required to demonstrate a complete operational record. A future veRL adapter must map its native outputs into these canonical metrics, but no veRL execution dependency or provider-specific meaning entered Observatory.

## Context and Orientation

`packages/train/src/posttrain/train/backends/trl/common.py` converts TRL logs into provider-neutral `RunContext` metrics. `packages/train/src/posttrain/train/backends/trl/dpo.py` renders preference pairs, emits immutable data characteristics, and runs the pinned TRL DPO trainer. `apps/observatory/src/posttrain_observatory/telemetry.py` defines the job-aware DPO fields, charts, help text, alerts, and comparison keys. `apps/observatory/src/posttrain_observatory/service.py` reads normalized metrics from any `RunDataSource` and constructs the view returned to Python, HTTP, MCP, and React. `apps/observatory/frontend/src/App.tsx` renders that shared view.

A required metric must exist for every successful DPO run. A conditional metric is mandatory when its configuration or recorded population activates the condition, such as gradient clipping being enabled or a validation dataset being selected. A diagnostic metric remains queryable and explainable but does not prevent a run from being operationally complete.

## Plan of Work

Add strict requirement definitions to `apps/observatory/src/posttrain_observatory/telemetry.py`. Requirements group related metrics by researcher question and identify their level and activation condition. Add immutable completeness response models to `apps/observatory/src/posttrain_observatory/models.py`. Update `ObservatoryService._metric_job_view` to load every required metric, evaluate active conditions from resolved inputs and metric values, and return deterministic requirement states, counts, a completeness state, and a research-readiness flag.

Extend the DPO telemetry definition so its required groups cover the objective, preference ordering, policy movement, stability, rendered data, and runtime. Mark held-out preference metrics, clipping evidence, and source-score evidence as conditional. Keep distributed, quantized-update, and packing conditions in the shared condition vocabulary for job definitions that actually require their evidence; do not invent DPO-specific requirements without a canonical metric producer. Keep entropy, token accuracy, and raw logits diagnostic unless their computation is explicitly selected. Missing active requirements create typed alerts; inactive conditions show `not_applicable` rather than `missing`.

Enhance `packages/train/src/posttrain/train/backends/trl/dpo.py` only for evidence available without additional model work. Record preference-length quantiles and explicit maximum-length headroom alongside the existing means. Do not invent a truncation rate because current preference rendering rejects over-limit examples.

Update the frontend API types and DPO overview to show a compact evidence-completeness strip with status, required/conditional counts, research readiness, and the most important missing reason. Keep the full requirement list available in the same DPO evidence section without duplicating metric meaning in frontend-only code.

## Concrete Steps

From `/home/hammad/projects/rl`, edit only the DPO/train and Observatory files named above using additive changes. Run focused tests first:

    uv run pytest packages/train/tests/test_dpo_observability.py apps/observatory/tests -q
    cd apps/observatory/frontend && npm test -- --run && npm run build

Then run boundary and type validation from the repository root:

    uv run ruff check packages/train/src/posttrain/train/backends/trl/dpo.py packages/train/tests/test_dpo_observability.py apps/observatory/src apps/observatory/tests
    uv run pyright packages/train/src/posttrain/train apps/observatory/src apps/observatory/tests
    uv run lint-imports
    git diff --check

Rebuild the existing Observatory Compose application and inspect a real DPO run through the Codex in-app browser. The current real run may be classified as incomplete because it predates the new static data metrics; this is expected and proves missing evidence is preserved honestly.

## Validation and Acceptance

The Python service test must prove that a DPO run with every required group is complete, a run missing a required metric is insufficient, an active conditional requirement with missing evidence is partial, and an inactive condition is `not_applicable`. The schema test must reject conditional requirements without a condition and reject required requirements that incorrectly declare one.

The train test must prove exact length-profile values over deterministic rendered pairs. Frontend tests must prove that the DPO page renders completeness state, counts, research-readiness status, and a missing reason from the server response rather than recomputing the contract.

In the live product, selecting the real DPO run must show the same charts and lineage as before plus a compact evidence assessment. Existing metrics remain available, and missing newly introduced evidence is clearly identified instead of being synthesized.

## Idempotence and Recovery

All changes are additive and safe to rerun. Tests use in-memory sources and temporary directories. Rebuilding the Observatory container does not mutate tracking evidence. If the frontend or service change fails, revert only the files named in this plan; do not reset the repository because the worktree contains extensive accepted changes from other slices.

## Artifacts and Notes

- Live real-run screenshot: `docs/design/observatory/audit/dpo/05-dpo-evidence-completeness.jpg`.
- Focused Python result: `19 passed, 1 warning`; the warning is the pre-existing Starlette `TestClient` deprecation.
- Frontend result: `4 passed`; TypeScript checking and the Vite production build passed.
- Boundary result: Ruff passed, Pyright reported zero errors, and all eight import-linter contracts were kept.
- Live transport result: packaged `/openapi.json` contains `EvidenceCompleteness`, references it from `RunView`, and represents arbitrary JSON without a recursive TypeScript-breaking schema.

## Interfaces and Dependencies

`JobTelemetryDefinition` will expose a tuple of strict `EvidenceRequirementDefinition` values. `RunView` and `EvaluationRunView` will expose an `EvidenceCompleteness` object. No new dependency is required: Pydantic already validates Observatory contracts, and train already depends on Pydantic. Tracking backends remain unaware of DPO semantics.

Revision note (2026-07-22): created this self-contained plan after the user approved enforceable required, conditional, and diagnostic DPO evidence levels and requested implementation.

Revision note (2026-07-22): completed the DPO contract, provider-neutral completeness projection, TRL pair profiling, typed generated API boundary, React presentation, focused tests, and real Trackio browser verification. Recorded the expected legacy-run evidence gaps and left veRL execution as a future adapter concern.
