# Evaluation and environments

Status: implemented evaluation vertical slice; GPU qualification in progress
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

`packages/eval` exposes a reusable public operation API and evaluation programs.
The canonical implementation files are:

- `packages/eval/src/posttrain/eval/requests.py` for program, target, budget,
  and request contracts;
- `packages/eval/src/posttrain/eval/api.py` for the public operation;
- `packages/eval/src/posttrain/eval/backends/verifiers/` for the pinned v1
  adapter and trace synchronization;
- `packages/eval/src/posttrain/eval/programs/` for reusable code-defined
  selections.

```python
GENERAL_SMOKE = EvaluationProgram(
    id="general-smoke-v1",
    kind="general",
    environments=(
        EnvironmentProgram(...),
    ),
)

result = evaluate(
    context,
    EvaluationRequest(
        model=model,
        target=EvaluationTarget("http://127.0.0.1:8000/v1", served_model),
        program=GENERAL_SMOKE,
        environment_id="math-gsm8k",
        context_window=8192,
        budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
    ),
)
```

An evaluation program selects native environment factories and supplies default
task, rollout, concurrency, and sampling policy. It never copies environment
data or scoring logic. `EvaluationBudget` selects a smaller invocation subset
without cloning or mutating the reusable program.

One `evaluate` call runs exactly one environment cell. A general program may
contain several categories, but each category gets its own run attempt,
configuration, trace population, and native artifact. Reports can compose those
runs later without making an opaque multi-environment run.

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

## Qualified MVP environments

| Program | Intent | Environment package |
| --- | --- | --- |
| `GENERAL_SMOKE` | math, instruction following, code execution, multi-turn state | pinned upstream `gsm8k-v1`, `reverse-text-v1`, `code-golf-v1`, and `alphabet-sort-v1` |
| `AGENTIC_SMOKE` | foundational cross-application tool use | local `automationbench-v1`, simple domain |
| `AUTOMATIONBENCH_PUBLIC` | full domain evaluation | the same `automationbench-v1` package across sales, marketing, operations, support, finance, and HR |

`environments/automationbench_v1` is a real native-v1 port rather than a call
through Verifiers' legacy environment bridge. It reuses Zapier AutomationBench
1.0.5 task builders, simulated SaaS world, API routes, and assertion registry at
commit `a321764ace3cfbe42289e6a13abef2f0f4f56fad`. The port owns typed v1
`TaskData`, per-rollout state, an MCP API toolset, and trace projection.

AutomationBench declares Python 3.13 while the GPU workspace currently uses
Python 3.12. The environment therefore has an independent uv project and lock.
This is dependency isolation, not a separate evaluation model: a 3.13
environment worker can evaluate the same OpenAI-compatible endpoint and emit
the same Verifiers trace schema.

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

Conversation rendering is not redefined by an environment. The foundation
model profile owns chat-template selection, reasoning modes, roles, and native
tool-call grammar. A serving profile maps those facts to backend parsers. The
Verifiers harness supplies messages and MCP tools through the endpoint's
OpenAI-compatible protocol, so Qwen and LFM keep their own rendered formats
without model-specific branches in `posttrain.eval`.

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
The run stores synchronization counters because they are direct execution
health observations. It does not store mean reward, pass rate, or other
cross-trace capability summaries. `packages/reports` owns those aggregate
calculations and comparison views. The job
owns selection, thresholds, and what to do next.

## Serving evaluation remains separate

Serving benchmarks measure TTFT, decode latency, throughput, concurrency,
memory, cache behavior, and failures. They are `posttrain.serve` operations.
Verifiers evaluations measure model behavior and are `posttrain.eval`
operations.

They may share a model artifact, endpoint, action invocation, Trackio job, and
report, but one package never absorbs the other.

## MVP sequence

1. [x] Expose a typed, directly reusable `eval.evaluate` operation and program API.
2. [x] Consume pinned upstream GSM8K through native Verifiers v1.
3. [x] Define reusable general, agentic-smoke, and AutomationBench domain programs.
4. [x] Prove trace/native-artifact emission with local and Trackio observer adapters.
5. [x] Create and package the native-v1 AutomationBench port.
6. [x] Qualify real GPU GSM8K runs for both foundation profiles.
7. [x] Qualify the AutomationBench package and MCP/scoring lifecycle against a live endpoint in Python 3.13.
8. [ ] Integrate the isolated environment process with the lab observer so live trace streaming and artifact promotion are automatic.
9. [ ] Reuse the same task packages for checkpoint evaluation and GRPO.

## Revision history

- 2026-07-20: Implemented one-cell endpoint-neutral evaluation, typed subset
  budgets, code-defined general/agentic/domain programs, and a native Verifiers
  v1 AutomationBench 1.0.5 port with isolated Python 3.13 dependencies.
- 2026-07-20: Qualified Qwen and LFM GSM8K Trackio runs plus the native
  AutomationBench Python 3.13 MCP and final-state scoring lifecycle on live GPU
  inference; retained isolated-worker observation as the next integration.
- 2026-07-20: Replaced YAML eval/job composition with typed programs and
  operations, required model onboarding to use a code-defined job, and made
  Trackio observation-only.
- 2026-07-20: Defined task-shaped generation budgets and Verifiers trace synchronization.
- 2026-07-19: Established independently published Verifiers environments and shared general/domain execution.
