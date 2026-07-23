# Observatory job-view usability audit

Date: 2026-07-22

Status: healthy for real schema-v4 SFT/DPO evidence and deterministic trace,
GRPO, evaluation, and unknown-job flows. Scale and credentialed provider
validation remain separate release gates.

## Audited journeys

1. **SFT optimization - healthy.** The page asks whether loss is improving
   without unstable gradients or falling throughput. Loss and learning rate
   lead, performance has a separate chart tab, supporting learning-rate,
   gradient-norm, and token-throughput values stay compact, and selecting a
   point exposes the exact step. The final layout keeps step context beside the
   chart and uses the right rail for resolved base model, dataset, QLoRA
   binding, and output artifacts. Evidence:
   `artifacts/observatory/validation/visual-audit/13-real-v4-sft-overview.png`.
2. **DPO preference learning - healthy after correction.** The first pass led
   with loss, which did not answer the core preference question. The default
   chart now leads with chosen reward, rejected reward, and reward margin; loss
   remains available as a secondary tab. A real two-step QLoRA run proves the
   metric mapping and produced-artifact lineage. Evidence:
   `artifacts/observatory/validation/visual-audit/12-real-v4-dpo-overview.png`.
3. **GRPO policy optimization - healthy.** Reward and policy-control evidence
   are separated into useful views, while the run keeps a direct path to
   rollout traces rather than treating traces as an evaluation-only feature.
   Evidence: `artifacts/observatory/validation/usability-audit/03-grpo-overview.jpg`.
4. **Evaluation and traces - healthy.** Aggregate reward, pass rate, slice
   performance, tool-use correlation, filters, sortable trace population,
   transcript, latency, tokens, and reward components coexist on one surface.
   The page states that evaluation is derived from the trace population and
   distinguishes sync completeness, ingestion errors, truncation, and verifier
   outcomes. Evidence:
   Compact popovers replace native selects, and choosing `calendar` reduces the
   population from 12 to 8 records. Evidence:
   `artifacts/observatory/validation/visual-audit/15-trace-filter-popover.png`.
5. **System telemetry - healthy.** A dedicated section groups accelerator and
   host utilization, memory pressure, runtime throughput, and observer/trace
   health. Missing canonical series remain explicit instead of becoming zero.
   The final view reads normalized GPU, VRAM, CPU, and wall-time series from the
   real Trackio system table; unavailable process RSS and dropped-trace series
   remain explicit. Evidence:
   `artifacts/observatory/validation/visual-audit/14-real-v4-sft-system-metrics.png`.
6. **Unknown job - healthy after correction.** The generic workspace does not
   infer metric meaning. It presents available names, requires explicit series
   selection, limits the chart to three series, and discloses point bounds and
   downsampling. Evidence:
   `artifacts/observatory/validation/usability-audit/06-generic-metric-selection.jpg`.
7. **Narrow status check - healthy.** Project, work package, run, status, and
   evidence navigation remain reachable without page-level overflow. Dense
   analysis correctly remains desktop-first. Evidence:
   `artifacts/observatory/validation/visual-audit/05-narrow-sft-overview-fixed.jpg`.

## What works especially well

- Job definitions choose the default question, summary values, and charts, so
  SFT, DPO, and GRPO do not become differently labeled copies of one dashboard.
- Generic metrics, system telemetry, and trace/evaluation have distinct
  responsibilities and remain available through the same run shell.
- Trace aggregates lead directly to exact examples and the selected example
  keeps both human-readable transcript data and quantitative evidence visible.
- Generated interpretation is explicit, on demand, cited, and visually
  separate from deterministic status and alerts.

## Remaining release risks

- The fixture population proves the interaction model, not table behavior at
  production trace counts. A provider-backed high-volume trace fixture remains
  necessary for performance validation.
- Browser semantics expose headings, landmarks, labels, table roles, and chart
  text alternatives, but this pass is not a substitute for a full screen-reader
  and keyboard-only session.
- Comparison, alert triage, and cross-provider source-failure journeys are
  represented in the service contract but are not yet complete frontend flows.
- Real Trackio SFT/DPO evidence is now exercised. W&B and other job/provider
  combinations may still expose missing evidence combinations.

## Decision

The implemented run, trace/evaluation, system, and generic views are useful for
their intended first-slice questions. The final assessment is grounded in real
Trackio SFT/DPO runs where those runs provide evidence and bounded fixtures only
where a training run cannot represent the journey. The remaining items are
explicit release validation and later product breadth, not blockers for the
React/Tailwind run-detail slice.
