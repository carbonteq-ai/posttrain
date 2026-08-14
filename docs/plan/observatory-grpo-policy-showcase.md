# Keep GRPO policy evidence in one divided chart

This ExecPlan is a living document and must be maintained according to
`docs/templates/PLAN.md` and `.agents/PLAN.md`.

## Purpose / Big Picture

The GRPO Overview keeps reward, policy objective, exploration, divergence, and
clipping measurements in one Policy optimization chart. The renderer divides
incompatible numeric scales into synchronized internal panels, so a user can
inspect one coherent surface and one selected logical step without separate
chart cards.

## Progress

- [x] (2026-08-13 14:22Z) Reproduced the six-scale GRPO chart on the active
  Ambient Agent run and traced it to the `optimization` telemetry chart.
- [x] (2026-08-13 19:30Z) Reverted the exploratory panel-response and
  showcase-card implementation after visual review showed the original
  single-chart, internal-panel composition was clearer.
- [x] (2026-08-13 19:30Z) Regenerated API types and validated Python service,
  HTTP, frontend rendering, production build, linting, and diff hygiene.
- [x] (2026-08-13 19:45Z) Kept policy loss and entropy in one internal panel
  with independent axes, then revalidate the local GRPO run.
- [x] (2026-08-13 19:50Z) Added dedicated legend-to-panel spacing and covered
  the small-multiple panel origin and final x-axis label with frontend tests.
- [x] (2026-08-13 19:55Z) Moved conditional DAPO and OLMo3 sampling views to
  the end of the GRPO tab order.
- [x] (2026-08-13 20:00Z) Replaced the rejected run-wide rollout summary cards
  with a trace-derived Rollout behavior panel inside the main Policy
  optimization chart. It shows per-step averages on synchronized axes.

## Surprises & Discoveries

- Observation: `EvidenceChart` intentionally turns three or more incompatible
  scales into synchronized small multiples.
  Evidence: `EvidenceChart.tsx` calculates `useSmallMultiples` from the number
  of scale groups.

## Decision Log

- Decision: retain one `optimization` chart and the existing internal-panel
  rendering rather than exposing an API-level panel contract.
  Rationale: visual review showed that the common outer frame, legend, and
  selected logical step make the related GRPO evidence easier to read.
  Date/Author: 2026-08-13 / Codex
- Decision: policy loss and entropy share one policy-update panel but retain
  distinct axes.
  Rationale: they answer the same update-control question, while their values
  have incompatible magnitudes; a shared ruler would hide the loss trajectory.
  Date/Author: 2026-08-13 / Codex
- Decision: derive rollout behavior only from an explicit trace
  `optimizer_step` attribute.
  Rationale: a trace-page position can change with provider pagination and is
  not evidence of when the rollout informed the optimizer. Traces without an
  explicit step stay visible as unattributed coverage rather than being
  assigned heuristically.
  Date/Author: 2026-08-13 / Codex

## Outcomes & Retrospective

Implemented and validated. `train.grpo` keeps its one Policy optimization
chart, and the temporary three-card presentation plus its public panel fields
were removed. The active local Ambient Agent run now uses the stable divided
chart renderer again, with policy loss and policy entropy in a shared
dual-axis policy-update panel. Its Rollout behavior group is a fourth
synchronized panel in that same chart: it aggregates provider-recorded total
completion tokens and tool calls for each optimizer step, adding exact thinking
tokens only when the selected run actually retains them. Thinking is a subset
of completion tokens, never an independently additive output total.

Historical Verifiers v2 traces record their training attribution under
`payload.run.step`, while current traces use `optimizer_step`. The behavior
projection accepts both explicit schemas (and only accepts `run.step` for a
record marked `type: train`), so older completed GRPO runs are not silently
excluded from the chart.

## Context and Orientation

`apps/observatory/src/posttrain_observatory/telemetry.py` owns job-aware chart
definitions. `service.py` projects provider series into `ChartView` values.
`App.tsx` renders the selected chart, while `EvidenceChart.tsx` divides a chart
into synchronized internal panels when its metric series need incompatible axes.

## Plan of Work

Keep the existing `ChartDefinition.metrics` and `ChartView.series` interfaces.
The Policy optimization definition continues to combine reward, objective,
entropy, KL, and clipping metric names. The generic renderer owns numerical
scale separation. Its semantic panel rule keeps the standard
`train/rl/policy_loss` and `train/rl/entropy` pair together with separate axes;
no GRPO-specific response metadata is added.

The GRPO chart definition now orders the shared policy-health evidence before
the DAPO dynamic-sampling and OLMo3 active-sampling extensions. The local API
was restarted and the active run verified the resulting visible tab order.

The Policy optimization view additionally loads one bounded per-step trace
projection after its run shell is visible. `rollout_behavior_view` groups only
traces with `optimizer_step`, exposes attributable and unattributed coverage,
and never invents a zero when a provider did not retain a scalar. When a
historical Qwen3.5 trace retains generated token IDs and the sampled mask,
Observatory counts sampled tokens before the native `</think>` marker. This is
an exact count from the original generation, not a character-based estimate.
The dedicated full-payload scan is cached for 60 seconds on live runs and five
minutes for terminal runs. On the active run, all 1,384 traces are attributed
across steps 26–33; provider usage omits `reasoning_tokens`, but retained trace
IDs support an exact thinking-token series wherever the original generated IDs
and sampled mask are present.

## Concrete Steps

Run from `/home/hammad/projects/rl`:

    uv run pytest apps/observatory/tests/test_service.py apps/observatory/tests/test_product_service.py apps/observatory/tests/test_http.py -q
    uv run posttrain-observatory schema --openapi apps/observatory/openapi.json

Run from `/home/hammad/projects/rl/apps/observatory/frontend`:

    npm run generate:api
    npm test -- --run src/App.test.tsx src/components/EvidenceChart.test.tsx
    npm run build

Then run from the repository root:

    uv run ruff check apps/observatory
    git diff --check

## Validation and Acceptance

The GRPO service test proves that the one optimization chart contains the
combined reward, policy, and clipping metrics. The trace projection test proves
that per-step means use recorded `optimizer_step` values and leave missing-step
traces unattributed. The renderer test proves that three incompatible scale
families become synchronized internal panels. On the active local GRPO run,
Policy optimization shows its divided plots plus a trace-derived behavior chart
with eight points for steps 26–33.

## Idempotence and Recovery

The change is read-only. Regenerating the OpenAPI file and frontend types is
repeatable. No tracking data is modified.

## Artifacts and Notes

The motivating route is the active Ambient Agent GRPO run at
`/runs/WyJhbWJpZW50LWFnZW50IiwiYW1iaWVudC1rMWEtb2xtbzMtZnVsbDc3NDktY29tcGxldGlvbjIwNDgtcmVzdW1lMTAwc3RlcC1wb3N0dHJhaW4wMzEzLTIwMjYwODEyLXIxIl0`.

## Interfaces and Dependencies

`ChartDefinition` retains `metrics`, and `ChartView` retains `series`.
`EvidenceChart` derives synchronized panels from compatible numeric scale
families already present in that one chart response. The narrow
`/rollout-behavior` read endpoint returns `RolloutBehaviorView`, avoiding an
expensive trace population read in the standard run-view response.

Revision note (2026-08-13): visual review rejected the separate showcase-card
experiment in favor of the existing one-chart internal-panel composition.
