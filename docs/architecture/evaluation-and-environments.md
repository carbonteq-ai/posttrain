# Evaluation and environments

Status: target MVP architecture  
Last revised: 2026-07-20

## Purpose

General and domain evaluation use the same Verifiers substrate while retaining
independent package ownership and code-based composition.

## Verifiers boundary

The platform adopts the current Verifiers architecture directly:

```text
TaskData + Task behavior -> Taskset
Taskset + Harness + Runtime -> Environment -> Trace
```

It does not add another task-row schema, environment base class, reward
interface, or registry.

Every environment is an independently installable package with its own code,
dependencies, tests, and version. Owners publish and consume qualified
Environment Hub references. The platform records the resolved package version
used by each execution.

## Code-first evaluation operations

`packages/eval` exposes a reusable public operation API and evaluation programs:

```python
GENERAL_SMOKE = EvaluationProgram(
    id="general.smoke",
    environments=(
        EnvironmentRef("team/instruction-following", version="1.1.0"),
        EnvironmentRef("team/long-context-needle", version="2.0.1"),
    ),
)

result = model_eval.evaluate(model=model, program=GENERAL_SMOKE)
```

An evaluation program selects environment references and supplies default
runtime, rollout, and sampling policy. It never copies environment data or
scoring logic. A job may create an explicit copy with task counts, subsets, or
sampling overrides.

TOML is useful for upstream Verifiers CLI interoperability and can instantiate
a typed config. It is not the platform's job or workflow language.

## General versus domain evaluation

| Concern | General evaluation | Domain evaluation |
| --- | --- | --- |
| Purpose | characterize raw models and broad regressions | measure one product/domain objective |
| Reuse | across models, checkpoints, and jobs | across jobs sharing domain semantics |
| Curation | reusable programs in `packages/eval` | selected by job/domain code |
| Environment source | standalone Verifiers package | standalone Verifiers package |
| Thresholds | reference ranges or job-selected policy | job-owned acceptance policy |

The distinction is curation and intent, not a different package API or storage
schema.

## Environment creation and reuse

A job may expose a missing domain capability:

```text
job identifies missing scenarios
  -> domain team authors standalone Verifiers package
  -> job uses a local editable reference while iterating
  -> owner publishes a version
  -> job pins/imports the published reference
  -> evaluation and compatible online RL reuse it
```

The source does not become a permanent `jobs/<id>/environments` tree. General
environments such as agency benchmarks or long-context tests can be onboarded
in their own maintenance job and then selected by all later jobs.

## Raw-model onboarding

A model-onboarding job can evaluate any new instruction-tuned or reasoning
model without waiting for a product fine-tuning job:

1. add or import the model profile;
2. choose a general program or category subset;
3. call evaluation operations against a compatible endpoint/runtime;
4. run only missing or deliberately refreshed combinations;
5. record Verifiers traces and artifacts in Trackio;
6. expose the accumulated evidence through reports.

The onboarding job is still code and can contain many models and runs. Trackio
only observes its executions.

## Checkpoint evaluation

A post-training job can call domain and general-regression operations for any
exact checkpoint artifact. The artifact may be evaluated several times under
different environments, seeds, providers, or runtime settings without creating
a new job.

Comparability requires exact model, environment version, harness/runtime,
sampling, and provider/judge context. Reports reject or clearly separate
incompatible populations.

## Context and generation budgets

Keep three values distinct:

```text
model native context limit
serve endpoint exposed context
environment response allowance
```

The eval adapter validates that the endpoint does not exceed the model limit
and that each response allowance fits inside the served context. Task-shaped
output ceilings are preferable to one global small cap. Truncation and
`finish_reason=length` are recorded on traces and must be considered when
comparing capability.

## Execution and evidence

`packages/eval`:

- resolves and runs the qualified environment package;
- validates model/endpoint compatibility;
- emits resolved operation inputs through the optional host observation context;
- emits native rollouts and the native evaluation directory through that
  context;
- returns typed execution outputs.

In this lab, the host context converts those emissions to idempotent
`VerifiersTrace` records and a Trackio artifact. Another project can supply a
different observer. The environment owns task semantics. Trackio owns only the
observations the lab host records.
`packages/reports` owns aggregate calculations and comparison views. The job
owns selection, thresholds, and what to do next.

## Serving evaluation remains separate

Serving benchmarks measure TTFT, decode latency, throughput, concurrency,
memory, cache behavior, and failures. They are `posttrain.serve` operations.
Verifiers evaluations measure model behavior and are `posttrain.eval`
operations.

They may share a model artifact, endpoint, action invocation, Trackio job, and
report, but one package never absorbs the other.

## MVP sequence

1. Expose a typed, directly reusable `eval.evaluate` operation and program API.
2. Run one qualified Verifiers environment package.
3. Define a reusable general-smoke program.
4. Prove trace/native-artifact emission with a no-op/local observer and the lab's Trackio observer.
5. Create one independently published domain environment.
6. Exercise both from model-onboarding and post-training job actions.

## Revision history

- 2026-07-20: Replaced YAML eval/job composition with typed programs and
  operations, required model onboarding to use a code-defined job, and made
  Trackio observation-only.
- 2026-07-20: Defined task-shaped generation budgets and Verifiers trace synchronization.
- 2026-07-19: Established independently published Verifiers environments and shared general/domain execution.
