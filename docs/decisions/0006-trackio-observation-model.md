# ADR 0006 — Use Trackio only as the observability and evidence layer

## Status

Accepted.

## Context

The prototype records executions in both local run directories and Trackio.
Earlier architecture drafts also made Trackio groups approximate a platform job
hierarchy and required each model branch to become a new job. That blurred
source intent, execution control, observability, and artifact lineage.

The platform now uses code-defined jobs and actions. One action invocation can
call several typed package operations, and one job can contain repeated attempts or
sibling model branches. Trackio already provides the durable surfaces needed to
observe these executions: runs, configuration, metrics, standard LLM traces,
artifacts, and run-artifact relationships. The CarbonTeq fork adds specialized
Verifiers traces and their UI.

## Decision

- Treat Trackio as a pure observability layer. It records execution evidence but
  never defines or executes jobs, actions, profiles, dispatch, scheduling,
  retries, branching, acceptance policy, or promotion.
- Use one Trackio project namespace, `lab`, for the MVP. A Trackio project is a
  storage/query namespace, not a product workflow entity.
- Require every Trackio run recorded by this lab to carry a code-defined
  `job_id`. Use that ID as Trackio
  group and as searchable configuration.
- Record `job_module`, `action_id`, and optional `invocation_id`. These correlate
  a run to source and sibling runs but do not introduce new stored hierarchy
  entities.
- Define a run as one typed execution request or attempt. Keep `run_kind` as a
  configuration field, never as Trackio group.
- Permit multiple run kinds, attempts, checkpoints, and artifact branches in
  one job. Create a new job only for a distinct objective, owner, or lifecycle.
- Make Trackio the only durable run, metric, trace, artifact, and observed
  lineage store. Use temporary execution workspaces only for framework I/O,
  recovery, and artifact staging.
- Keep Python source and packages authoritative for jobs, profiles, configs,
  and behavior. Record an immutable JSON-safe resolved snapshot and source
  revision on each run.
- Use trace and run as the raw metric grains. Compute job/model summaries from
  selected populations through versioned calculators in `packages/reports`.
- Do not create direct job- or model-metric write APIs. Allow rebuildable
  materializations only with calculator and population provenance.
- Use `trackio.Trace` for ordinary inference requests, with optional external
  identity for idempotent ingestion.
- Use `trackio.VerifiersTrace` for native graph-shaped Verifiers rollouts and
  provide a dedicated structured rollout UI.
- Link every trace to exactly one run. Store request/rollout-specific values on
  the trace and shared execution context on the run.
- Use consumed/produced artifact edges as observed model/data lineage. Trackio
  records those edges but does not decide which artifacts to create, consume,
  alias, or promote.

## Consequences

- Git answers what the team intended; Trackio answers what actually ran.
- Every execution remains searchable by job/action/invocation without Trackio
  becoming a workflow database.
- A complete fine-tuning objective can keep related SFT, DPO, GRPO, evaluation,
  and serving runs together while exact branches remain visible through model
  artifact lineage.
- Independent serving enablement or general-evaluation maintenance can use
  separate jobs because their objectives and ownership differ.
- Metrics, traces, artifacts, and status have one durable write path.
- Reports can evolve calculation definitions without rewriting raw evidence.
- Ordinary inference reuses the standard trace ecosystem; Verifiers retains a
  specialized lossless representation and UI.
- Current local run bundles, optional jobs, run-kind grouping, persisted
  trace-derived summaries, and direct physical-SQLite report coupling are
  implementation gaps.

## Alternatives Considered

### Let Trackio define platform jobs

Rejected because source behavior, thresholds, actions, and branch decisions are
versioned Python concerns. Trackio group is sufficient for correlation.

### Keep jobs optional

Rejected because even a one-run model onboarding or benchmark activity needs an
explicit owner and purpose. A lightweight code-defined job is sufficient.

### Create a new job for every artifact branch

Rejected because branch lineage and objective ownership are independent. The
artifact graph provides exact branching; job boundaries describe workstreams.

### Retain the filesystem as a second run authority

Rejected because duplicate configuration, status, and result stores require
reconciliation. Files exist only for temporary framework execution or retained
artifacts.

### Persist every aggregate

Rejected because means, percentiles, rates, totals, and Pareto membership depend
on populations and calculator definitions. Raw observations plus versioned
views preserve provenance.

### Flatten Verifiers rollouts into standard traces

Rejected because message graphs, alternate branches, environment state,
rewards, tools, phases, and stop semantics would be lost.

## Implementation Notes

- Detailed boundary: [Trackio observability architecture](../architecture/trackio.md).
- General guidance: [Observability](../architecture/observability.md).
- Verifiers extension: [ADR 0005](./0005-trackio-verifiers-traces.md).
- `packages/common` supplies the write adapter; `packages/reports` supplies the
  read-only query and computation boundary.
- The CarbonTeq fork uses embedded Turso by default with stdlib SQLite fallback.
- The standard Trace external ID and numeric trace-metric projection remain
  required fork work.

## Revision History

- 2026-07-20: Made Trackio purely observational, mapped code-defined
  job/action/invocation context onto runs, allowed artifact branches within one
  job, and made a single `lab` project the MVP evidence namespace.
- 2026-07-20: Defined trace/run recording grains, computed views, idempotent inference and Verifiers traces, Turso storage, and temporary workspaces.
