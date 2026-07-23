# Observability


> **STALE — pending reconciliation (2026-07-21).**
> Canonical design: [docs/post-training/](../post-training/README.md).
> Do not treat this document as the current product contract. Gap list: [RECONCILIATION.md](./RECONCILIATION.md).

Status: target MVP architecture  
Last revised: 2026-07-20

## Purpose

Observability explains an execution without turning the observability system
into the platform's control plane. Trackio records evidence; Python jobs and
reusable train/eval/serve packages define and execute behavior.

The complete storage model is in [Trackio observability architecture](./trackio.md).

## Boundary

```text
Git/Python authority                 Trackio authority
--------------------                 -----------------
job objective and actions            run identity and status
profiles and typed configs            resolved config snapshot
package operation and adapter behavior metrics and system telemetry
branch and promotion decisions        traces
acceptance thresholds                 artifacts and observed lineage
```

The same value may appear on both sides only for provenance: Trackio records
the resolved value and source revision used by a run, but does not become its
definition registry.

## Common observation context

For every operation observed by this lab, the host context supplies:

- `job_id`, `job_module`, and `action_id`;
- mandatory `invocation_id` correlating attempts submitted together;
- run kind and implementation ID/package/version;
- source repository commit and dirty digest;
- exact model and consumed artifact identities;
- resolved package-operation inputs and internal config snapshot;
- environment, workload, sampling, hardware, and software context;
- produced artifacts, final status, and failure information.

`posttrain.common` owns the framework-neutral envelope and observer protocol;
`posttrain_lab` owns the Trackio write adapter. The reusable
`posttrain.train`, `posttrain.eval`, and `posttrain.serve` APIs emit through an
execution context and do not import Trackio.

## Run kinds

- `model-onboarding`;
- `serving-benchmark`;
- `general-eval`;
- `domain-eval`;
- `sft`;
- `dpo`;
- `grpo`;
- `model-transformation`;
- `comparison-report`.

Every run recorded by this lab belongs to a job, but the job remains a Python
source definition. Trackio group is only the observable carrier of `job_id`.

## Measurement grain

Trace and run are recording scopes. Job and model are computed scopes.

| Scope | Record or compute |
| --- | --- |
| Trace | Record one request/rollout plus selected numeric metrics and indexed attributes. |
| Run | Record direct trainer/backend/system series and irreducible counters. |
| Job | Compute comparisons over explicitly eligible runs. |
| Model/lineage | Compute cumulative evidence and regression views. |

Record a measurement once at the lowest trustworthy grain. Do not persist p95,
mean, rate, total, or Pareto membership as another source of truth when it can
be calculated from retained observations.

## Training observability

Direct run metrics include step, epoch, learning rate, loss, wall time, GPU
telemetry, checkpoint events, and technique-native values. Representative
samples or rollout traces are retained according to the package operation and
caller data policy.

Recovery checkpoints remain workspace state. Selected outputs become Trackio
model artifacts with consumed/produced links.

## Evaluation observability

Verifiers `traces.jsonl` is the full-fidelity replay authority. The eval adapter
streams completed rollouts as idempotent `trackio.VerifiersTrace` records and
attaches the complete native directory as an artifact.

The Trackio Verifiers UI provides a dedicated rollout view with:

- message-graph branch navigation;
- rewards and environment metrics;
- model calls, tool calls, token usage, and timing;
- phase spans;
- stop, truncation, and error state;
- producing run and job context.

The UI uses the stored structured record but does not expose a raw payload dump.
Evaluation summaries such as mean reward, success rate, and truncation rate are
computed by `posttrain_observatory` from the trace population.

## Serving observability

Every measured inference request uses standard `trackio.Trace`. It records
messages, workload identity, timing, token counts, finish state, and errors.

Direct run metrics retain GPU/system series, cache and scheduler counters,
speculative acceptance, and kernel-specific measurements. Throughput and
latency distributions are computed from measured request traces plus direct
population counters.

One benchmark result emits its run-level scalar set as one metric batch. Metric
names are columns in that observation, not synthetic sequential steps. Request
traces remain separate high-cardinality observations linked to the same run.

Warmup requests are explicitly labeled or excluded. Production-style tracing
must declare sampling and redaction policy.

## Artifacts and logs

Attach or reference only outputs needed for lineage, replay, recovery, or
diagnosis:

- selected model outputs;
- native evaluation bundles;
- dataset manifests or immutable external references;
- serving benchmark bundles;
- comparison reports;
- useful framework logs.

Temporary execution workspaces are not a durable run store. They are cleaned
after successful finalization unless an explicit recovery policy retains them.

## Reports

`packages/reports` is the only shared read boundary. It provides versioned
calculators and frontend-facing trace, run, job, and model-lineage views.
Materialized values are rebuildable and carry their calculator and population
provenance. A durable human decision may be attached as a report artifact.

## Status vocabulary

Use:

- `complete`;
- `partial`;
- `failed`;
- `cancelled`;
- `unsupported`;
- `stale`;
- `not_run`.

`unsupported`, `stale`, and `not_run` are evidence states, not failed scores.

## Security and scale

- Never store secrets, tokens, or signed URLs in resolved snapshots.
- Keep high-cardinality records in trace storage, not scalar metric tables.
- Record model/judge/provider usage where available.
- Apply job/domain retention and redaction policy to prompts and user data.
- Preserve diagnostic error context without indiscriminate payload copying.

## Revision history

- 2026-07-20: Made the Trackio context a host-provided integration around
  reusable train/eval/serve package operations rather than a runner requirement.
- 2026-07-20: Made Trackio exclusively responsible for recorded observability,
  moved intent and behavior to code, and retained computed views in reports.
- 2026-07-20: Defined Verifiers and inference trace products, trace/run grains, and Turso-backed Trackio storage.
