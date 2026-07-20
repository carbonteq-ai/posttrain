# Post-training lab

Reusable uv workspace for model onboarding, inference, evaluation, and staged post-training.

The repository is being rebuilt around the [target architecture](./docs/architecture.md). Prototype catalogs, configuration trees, task callbacks, normalized evaluation stores, and framework-specific CLIs have been removed rather than retained as compatibility interfaces.

## Workspace

```text
packages/
  common/    lightweight job, request, identity, and observability contracts
  train/     reusable training operations with internal framework adapters
  eval/      reusable evaluation operations and programs
  serve/     reusable serving operations, profiles, and optimizations
  reports/   read-only Trackio queries and report-facing data access

profiles/
  models/    typed foundation and intentionally promoted derived targets

jobs/        code-based objectives, actions, and decisions
benchmarks/  versioned inference suites and canonical prompt corpora
docs/        lifecycle, architecture, decisions, and tooling guidance
```

`train`, `eval`, and `serve` are reusable package APIs for this lab and other
projects; framework runners/adapters stay internal. Their definitions live with
the package that validates and executes them. Verifiers environments are independently published packages and do not
live in a repository-owned environment catalog. Trackio is purely the durable
observability layer; Git/Python defines jobs, profiles, and behavior.

## Setup

```bash
cd /home/hammad/projects/rl
mise install
uv sync --all-packages --python 3.12
```

Backend and Verifiers dependency variants are installed only when working on that engine:

```bash
uv sync --package serve --extra vllm --python 3.12
uv sync --package serve --extra sglang --python 3.12
uv sync --package eval --extra verifiers --python 3.12
uv sync --package train --extra vllm --python 3.12
```

## Current executable surface

Profile resolution, typed Trackio runs, raw reporting queries, and the first
offline vLLM benchmark are available now:

```bash
uv run --package common profile-resolve models <profile-id>
uv run --package common profile-resolve train <config-id>
uv run --package reports trackio-query lab \
  "SELECT run_id, run_name, config FROM configs ORDER BY id DESC LIMIT 10"
uv run --package serve --extra vllm serve-benchmark lfm2.5-1.2b-thinking
uv run --package serve --extra vllm \
  serve-benchmark-suite lfm2.5-1.2b-thinking --dry-run
```

The first Verifiers general-eval slice is also executable:

```bash
uv run --package eval --extra verifiers eval-suite lfm2.5-1.2b-thinking --dry-run
uv run --package eval --extra verifiers eval-suite qwen3.5-2b --reasoning-mode thinking
```

The current executable surface predates the final code-first target. The next
refactor moves backend definitions beside `packages/serve`, replaces the generic
YAML resolver with typed definitions, and removes the local run registry in
favor of Trackio plus temporary execution workspaces.

The reusable inference suite contains concurrency 1/2/4/8, but the current
RTX 3070 Ti execution policy stops at 4. Its 32K cells select the model's
TurboQuant K8V4 serve variant automatically.

Start with the [post-training lifecycle](./docs/functional/finetuning-lifecycle.md), [architecture](./docs/architecture.md), and [profile model](./docs/architecture/profiles-and-model-variants.md).
