# ADR 0009 — Native Verifiers environment packages

## Status

Accepted.

## Context

General and domain evaluation both need executable environments, not a second
benchmark abstraction. Verifiers v1 already defines the task, taskset, toolset,
state, reward, metric, and trace contracts needed by the evaluation engine.
Environment authors must be able to publish and version those contracts without
depending on the post-training monorepo.

The upstream GSM8K v1 environment already implements the required native
Verifiers contract and can be consumed directly. Zapier AutomationBench 1.0.5
contains authoritative datasets, simulated SaaS APIs, world schemas, and
assertion scoring, but its published environment targets the previous
Verifiers API and requires Python 3.13. The main GPU workspace currently targets
Python 3.12. Importing the legacy bridge into the reusable evaluation engine
would couple every evaluation to two incompatible environment APIs and obscure
which trace contract is authoritative.

## Decision

- Treat independently versioned Verifiers v1 packages as the reusable unit for
  both general and domain environments.
- Consume the upstream `gsm8k-v1` package directly for the initial reasoning
  evaluation.
- Maintain `automationbench-v1` as a native Verifiers v1 adapter package under
  `environments/automationbench_v1`.
- Reuse AutomationBench's pinned task builders, datasets, simulated API world,
  tools, and assertion scorers. The adapter owns only the Verifiers v1 task,
  state, toolset, trace projection, and package boundary.
- Preserve per-rollout mutable world state in typed Verifiers state. Store
  dense partial credit as the reward, strict completion and assertion counts as
  trace metrics, and assertion details plus the final world in trace info.
- Keep `automationbench-v1` outside the root uv workspace and give it an
  independent Python 3.13 lock. Execute it in an isolated worker while exposing
  the same OpenAI-compatible model endpoint and Verifiers trace schema used by
  other environments.
- Do not add a legacy Verifiers compatibility layer to `posttrain.eval`.

## Consequences

- General and domain evaluation use one execution model; their difference is
  program curation and product intent, not engine architecture.
- Environment teams can release GSM8K-like or AutomationBench-like packages on
  their own cadence and jobs can pin those releases.
- AutomationBench retains upstream scoring semantics without copying its
  dataset or verification rules into platform code.
- The Python 3.13 environment needs an isolated worker integration before the
  lab can execute it from the Python 3.12 composition process in one command.
- Cross-environment reports can query the same trace fields, while detailed
  task-specific evidence remains in each trace's typed metadata.
- Updating either pinned upstream repository requires its own compatibility and
  scoring regression run before changing the environment package lock.

## Alternatives Considered

### Port the AutomationBench dataset and scoring into `posttrain.eval`

Rejected because it would make reusable evaluation infrastructure own one
domain's data and verification semantics, prevent independent publication, and
create a second copy of the benchmark.

### Run the published legacy AutomationBench environment through a bridge

Rejected because it would make a transitional Verifiers API part of the new
platform contract and produce a second trace lifecycle that jobs and
observability would have to understand.

### Move the complete GPU workspace to Python 3.13 immediately

Rejected for the MVP because the pinned CUDA, vLLM, TRL, and model stacks are
already validated on Python 3.12. An environment package should not force an
unrelated serving and training runtime migration.

### Vendor AutomationBench into this repository

Rejected because the upstream project is already versioned and publishable.
Pinning its immutable commit gives reproducibility without taking ownership of
all of its source.

## Implementation Notes

- `posttrain.eval` composes environments through Verifiers v1 and remains
  independent of model servers and Trackio.
- `posttrain.eval.programs.general` defines reusable general-evaluation cells;
  `posttrain.eval.programs.agentic` references the independently installable
  AutomationBench package.
- The AutomationBench package pins upstream AutomationBench commit
  `a321764ace3cfbe42289e6a13abef2f0f4f56fad` and Verifiers commit
  `284a868d6a9022109b749710672a0460e8a996d4` in its own `uv.lock`.
- The package test suite covers task loading, upstream dense and strict scoring,
  API discovery and mutation, and final trace evidence.
- Native `traces.jsonl` remains the authoritative evaluation artifact. The lab
  observer synchronizes completed traces into Trackio for querying.

## Revision History

- 2026-07-20: Established independently publishable native Verifiers v1
  environments and the isolated Python 3.13 AutomationBench adapter boundary.
