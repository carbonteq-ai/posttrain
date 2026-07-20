# ADR 0003 — Backend-neutral evaluation data

## Status

Superseded by [ADR 0004](./0004-lifecycle-driven-mvp-platform.md). Retained as prototype history.

## Context

The lab needs to compare base models, training checkpoints, quantized descendants, and team-owned inference optimizations. Measurements may come from vLLM, SGLang, Triton, LightEval, Verifiers, or future tools. Organizing performance benchmarks separately from quality evaluation creates overlapping concepts and makes cross-model reporting difficult.

## Decision

- Treat capability, task, and serving measurements as evaluation suites under one evaluation system.
- Identify every evaluated weight set through a model-artifact catalog entry with family and parent lineage.
- Describe execution independently through an inference profile with backend, implementation identity, and backend-specific settings.
- Store immutable results using shared model/runtime/suite/environment dimensions and canonical metrics.
- Preserve tool-native output under `backend_metrics` rather than forcing every backend field into the common schema.
- Represent weight-changing optimizations as descendant model artifacts and runtime-only optimizations as implementation revisions.
- Generate resolved plans and results; do not require a separate user-maintained experiment file.

## Consequences

- Reports can compare different model families, checkpoints, runtimes, and optimization revisions from one result collection.
- vLLM-specific flags do not leak into SGLang or Triton records.
- Backend adapters must map useful measurements into canonical metric names while retaining their native data.
- The catalog adds identity and lineage metadata but does not become a model-weight registry.
- Schema versions must be revised deliberately as real backend data reveals missing fields.

## Alternatives considered

### Separate benchmark and evaluation trees

Rejected because serving benchmarks and quality evaluations share model identity, inference configuration, provenance, and reporting requirements.

### Store only raw backend output

Rejected because comparisons would require backend-specific parsing every time and would be fragile across tool versions.

### Force all backend data into one fixed metric schema

Rejected because it would discard useful engine-specific measurements and make new backend integration unnecessarily disruptive.

## Implementation notes

- `catalog/models/` contains lightweight YAML references to Hub or local artifacts.
- `configs/inference/` contains runtime profiles; arbitrary backend settings remain nested below `settings`.
- `evaluations/suites/` contains capability, task, and serving definitions.
- `evaluations/results/<id>/result.yaml` is the normalized observation; optional raw files may live beside it.
- `optimizations/inference/` is available for team-owned code and compatibility documentation.
- Schema version `1` is intentionally small and should evolve from measured needs rather than anticipated integrations.

## Revision history

- 2026-07-20: Superseded as current platform guidance by ADRs 0004 and 0006;
  the target uses Trackio runs, metrics, traces, artifacts, and computed views
  instead of a shared observation-index schema.
- 2026-07-19: Initial decision.
