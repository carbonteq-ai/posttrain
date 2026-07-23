# Post-Training Framework Lab

Reference workspace for developing and validating a reusable Post-Training
Framework for model onboarding, inference, evaluation, and staged training.

The framework definition and contracts are established before the repository is
reconciled with the [target architecture](./docs/architecture.md). Prototype
catalogs, configuration trees, task callbacks, normalized evaluation stores,
and framework-specific CLIs are not retained as compatibility interfaces.

## Workspace

```text
packages/
  common/    framework-neutral identities, selections, artifacts, and execution contracts
  catalog/   versioned base catalog assets, project discovery, and overlay loading
  data/      canonical supervised/preference/rollout data and adapters
  train/     reusable training operations with internal framework adapters
  eval/      reusable evaluation operations and programs
  serve/     reusable serving operations, profiles, and optimizations
  tracking/  provider-neutral run lifecycle and evidence reads
  work/      reusable recipe/work-package loading, preflight, and execution

apps/
  cli/          primary `posttrain` project, catalog, and work-package command host
  lab/          reference work-package execution host
  observatory/  read-only analysis, API, MCP, frontend, and report product

environments/ independently versioned domain environment packages
docs/        lifecycle, architecture, decisions, and tooling guidance
.posttrain/  tracked project configuration plus ignored local state
```

`data`, `train`, `eval`, `serve`, and `tracking` are reusable package APIs.
`posttrain-catalog` distributes the framework base selections and resolves
project overlays. The lab is a reference host rather than a required import
path. Trackio and W&B are replaceable tracking adapters; Observatory reads
their normalized evidence through the shared tracking contracts.

## Setup

The source repository is public at
[`carbonteq-ai/posttrain`](https://github.com/carbonteq-ai/posttrain). Until
the framework packages are published to PyPI, team environments should use an
immutable GitHub release or clone an exact tag:

```bash
git clone --branch <release-tag> --depth 1 \
  https://github.com/carbonteq-ai/posttrain.git
cd posttrain
```

```bash
mise install
uv sync --all-packages --locked --python 3.12
```

Backend and Verifiers dependency variants are installed only when working on that engine:

```bash
uv sync --package posttrain-serve --extra vllm --python 3.12
uv sync --package posttrain-eval --extra verifiers --python 3.12
uv sync --package posttrain-train --extra trl --python 3.12
```

Use the checked-in lockfile for executable evidence. Training bindings record
its SHA-256. Materialized weight quantization has a separately solved and
locked environment under `tools/quantization`; do not force LLM Compressor
into the TRL environment merely to reduce the number of lockfiles.

## Project layout

Portable projects use one control directory:

```text
.posttrain/
  project.toml       tracked project identity and path policy
  catalog/           tracked project catalog overlays
  work_packages/     tracked screen, train, and qualify compositions
  state/             ignored scratch, recovery, cache, and provider-local state
```

Framework base selections are package resources in `posttrain-catalog`; a
project does not copy them into its overlay. Durable run artifacts are
published through the selected tracking/artifact backend. They are not inferred
from `.posttrain/state/` directory structure.

`posttrain.catalog.discover_project` checks an explicit project root,
`POSTTRAIN_PROJECT_ROOT`, then searches upward for
`.posttrain/project.toml`. See `tests/consumer/fixture` for a minimal independent
project.

The primary command host is installed by the `posttrain` distribution. From a
tagged source checkout, run its workspace build directly:

```bash
uv run --package posttrain posttrain version
uv run --package posttrain posttrain doctor
uv run --package posttrain posttrain project show
uv run --package posttrain posttrain catalog validate
uv run --package posttrain posttrain work-package validate foundation_screen.yaml
```

Initialize a separate project with:

```bash
uv run --package posttrain posttrain init ../my-posttrain-project \
  --project-id my-posttrain-project
```

Initialization creates tracked project configuration and an empty valid catalog
overlay while `.posttrain/state/` remains ignored. It refuses to overwrite an
existing project manifest.

## Portability acceptance

The external-consumer test builds framework wheels, installs them into a fresh
environment, copies a fixture repository outside this workspace, resolves a
packaged base selection plus project overlay, executes a deterministic CPU work
package, records its terminal run through real local Trackio storage, and reads
the same run through Observatory:

```bash
uv run pytest -q tests/consumer
```

This is a source-build acceptance path, not a claim that public packages have
already been released. See
[Release and consumption](./docs/release-and-consumption.md) for the package
graph, remote-project workflow, and remaining release gates.

## Current executable surface

The Qwen foundation screen is the reference vertical: packaged base catalog
selections resolve into a `.posttrain/work_packages` definition, invoke
`serve.benchmark` through `RunContext`, and publish durable evidence through the
selected backend.

The canonical qualification composition resolves an `EvaluationPlan` and
`EnvironmentBinding`, then calls `eval.general` or `eval.domain`. Native
Verifiers traces remain the evaluation authority; incomplete observer sync is
reported as `partial` rather than converted into a score.

```bash
uv run --package posttrain-lab posttrain-lab foundation-qwen-smoke --tracked
uv run --package posttrain-lab posttrain-lab foundation-lfm-gsm8k --tracked
uv run --package posttrain-lab posttrain-lab gsm8k-qwen-sft-smoke --tracked
```

The reusable inference suite contains concurrency 1/2/4/8, but the current
RTX 3070 Ti execution policy stops at 4. Its 32K cells select the model's
TurboQuant K8V4 serve variant automatically.

Start with [01 · Workflow](./docs/post-training/01-workflow.md), then
[02 · Primitives](./docs/post-training/02-primitives.md),
[03 · Work and Evidence](./docs/post-training/03-work-and-evidence.md),
[04 · Framework](./docs/post-training/04-framework.md), and
[05 · APIs](./docs/post-training/05-apis.md), and
[06 · Observation](./docs/post-training/06-observation-and-lineage.md).
The [architecture](./docs/architecture.md) is reconciled only after that
baseline is accepted.
