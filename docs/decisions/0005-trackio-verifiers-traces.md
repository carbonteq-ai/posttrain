# ADR 0005 — Add Verifiers traces through a compatible Trackio fork


> **STALE — pending reconciliation (2026-07-21).**
> Canonical design: [docs/post-training/](../post-training/README.md). Gap list: [RECONCILIATION.md](../architecture/RECONCILIATION.md).

## Status

Accepted.

## Context

ADR 0004 selected Trackio for run metrics, artifacts, and lineage while keeping
framework-native outputs authoritative. Evaluation now needs live, searchable
rollout observability without flattening Verifiers' versioned trace graph into
scalar metrics or introducing a separate trace database.

Upstream Trackio supports conversational traces but does not preserve the full
Verifiers v1 record, its native identity, or its schema version. Storing only an
evaluation directory artifact preserves fidelity but prevents live filtering
and inspection. Building a new Rust/Doris observability service would expand
the MVP before the ingestion and query contract has matured.

## Decision

- Maintain an additive fork at `carbonteq-ai/trackio`, with
  `gradio-app/trackio` configured as its upstream remote.
- Pin the workspace to an immutable fork commit while retaining the `trackio`
  package name and Python imports.
- Add `trackio.VerifiersTrace` with `_type: trackio.verifiers_trace`. It contains
  the native trace ID and schema version, the complete JSON-safe Verifiers
  record, a final-branch message projection, and bounded searchable metadata.
- Keep `trackio.Trace` and its UI unchanged. Add a parallel Verifiers rollout
  UI that understands branches, rewards, environment metrics, model calls,
  tools, phase timing, and errors, and links every rollout to its producing run.
  Do not render the complete native payload. Join the selected rollout with its
  producing run's aggregate `eval/*` metrics while keeping rollout results and
  evaluation results conceptually separate.
- Add `trace_type`, `external_id`, `schema_version`, and `payload` to trace
  persistence and Parquet interchange. Make
  `(run_id, trace_type, external_id)` unique for retry-safe ingestion.
- Index projected messages and selected metadata, never the complete native
  payload.
- Keep native Verifiers `traces.jsonl`, resolved configuration, and logs as the
  authoritative evaluation artifact.
- Tail completed JSONL records during execution, validate them with the pinned
  Verifiers wire schema, and send small batches to Trackio. Retry failed batches
  during finalization. A partial telemetry copy is recorded in scalar sync
  metrics but does not invalidate a successful evaluation.
- Keep recorded scalar metrics separate from high-cardinality trace rows. Compute
  deterministic evaluation summaries on read according to ADR 0006; any
  persisted summary is a versioned, rebuildable materialization rather than the
  result authority.
- Use Turso through `pyturso` as the default embedded SQL metadata engine while
  retaining `TRACKIO_DATABASE_ENGINE=sqlite` as a compatibility fallback.
  Preserve Trackio's database-per-project, offline-first contract; a Hugging
  Face Space is not required.
- Keep artifact bytes and native evaluation directories in Trackio's existing
  file/artifact storage. Turso owns queryable metadata, not object storage.

## Consequences

- Evaluation owners retain replayable native output while platform users gain
  live and queryable rollout traces.
- Resumed evaluations and uncertain batch retries cannot duplicate a rollout in
  the same Trackio run.
- Existing Trackio clients, APIs, artifacts, standard traces, SQLite-compatible
  databases, Parquet exports, and standard UI views remain compatible.
- Verifiers debugging no longer depends on forcing a rollout graph into the
  conversational trace view. The separate view can evolve without coupling the
  standard Trace product to Verifiers' schema.
- The fork now owns a database-driver compatibility boundary and a specialized
  frontend surface, increasing its upstream rebase and test obligations.
- The platform carries a small upstream-maintenance obligation. Every fork
  release records its upstream base commit and runs upstream compatibility tests.
- The Trackio trace copy may be partial during outages; its sync metrics and the
  native artifact make that state explicit.

## Alternatives Considered

### Store traces only as artifacts

Rejected because artifact-only storage is authoritative but cannot support live
search, trace filtering, or future observability consumers.

### Flatten Verifiers records into Trackio metrics

Rejected because message graphs, model calls, tools, errors, and task state are
high-cardinality records and would pollute the scalar metric model.

### Build a custom trace service now

Deferred. Rust, Tokio, Doris, object-storage redesign, and a custom frontend may
be reconsidered after the API and workload are proven by the compatible fork.

### Reuse only the upstream conversational trace UI

Rejected. A final-branch projection is useful for compatibility and search, but
it hides alternate branches, scoring, environment metrics, calls, tools, phase
timing, and stop/error state—the evidence needed to debug Verifiers evaluations.

### Move artifacts into Turso

Rejected. Turso is the SQL metadata engine. Large model, media, and evaluation
payloads remain in the artifact/file layer and can later use dedicated object
storage without changing trace or run schemas.

## Implementation Notes

- Fork maintenance: [Trackio tooling](../tooling/trackio/README.md).
- Runtime ownership: `packages/eval/src/eval/trace_sync.py`.
- Tracking boundary: `packages/common/src/common/tracking.py`.
- Observability model: [Observability](../architecture/observability.md).
- Job, run, standard-trace, and computed-metric model:
  [ADR 0006](./0006-trackio-observation-model.md).
- Initial upstream base: `438cb28d2c82c7b7d42431e45d5677a8cc90eb77`.
- Initial fork pin: `a79040fd9cecbb5881cda8d4c1961a55aeb7600f`.
- Current Turso/UI pin: `02351d871050bf4b3505c7371239c698b710ec83`.

## Revision History

- 2026-07-20: Replaced obsolete “facts” wording with Trackio metric terminology
  and aligned the extension with Trackio's observation-only responsibility.
- 2026-07-20: Accepted the additive Trackio fork and native-authority sync model.
- 2026-07-20: Selected Turso as the default local SQL engine and a dedicated,
  branch-aware Verifiers rollout UI linked to producing runs and jobs.
- 2026-07-20: Added run-level evaluation results beside the selected rollout and
  kept pass/fail policy environment- or job-owned.
- 2026-07-20: Aligned evaluation summaries with ADR 0006's computed-metric
  model and retained only irreducible scalar measurements as recorded metrics.
