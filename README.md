# Post-training lab

Reusable uv workspace for model onboarding, inference, evaluation, and staged post-training.

The repository is being rebuilt around the [target architecture](./docs/architecture.md). Prototype catalogs, configuration trees, task callbacks, normalized evaluation stores, and framework-specific CLIs have been removed rather than retained as compatibility interfaces.

## Workspace

```text
packages/
  common/    framework-neutral identities, models, artifacts, and execution contracts
  train/     reusable training operations with internal framework adapters
  eval/      reusable evaluation operations and programs
  serve/     reusable serving operations, profiles, and optimizations
  reports/   read-only Trackio queries and report-facing data access

apps/
  lab/       code-defined jobs, execution host, and Trackio observer

environments/ independently versioned domain environment packages
docs/        lifecycle, architecture, decisions, and tooling guidance
```

`train`, `eval`, and `serve` are reusable package APIs for this lab and other
projects; framework runners/adapters stay internal. Their definitions live with
the package that validates and executes them. Inference workloads and prompt
corpora live in `packages/serve`; Verifiers environments are independently
published packages and do not live in a repository-owned environment catalog.
Trackio is purely the durable observability layer; Git/Python defines jobs,
profiles, and behavior.

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

Profiles and jobs are typed Python definitions in their owning packages. Runs
use temporary execution workspaces and publish durable evidence to Trackio.
Representative commands are:

```bash
uv run --package posttrain-reports trackio-query posttrain-platform \
  "SELECT run_id, run_name, config FROM configs ORDER BY id DESC LIMIT 10"
uv run --package posttrain-lab posttrain-lab foundation-qwen-smoke --tracked
uv run --package posttrain-lab posttrain-lab foundation-lfm-gsm8k --tracked
uv run --package posttrain-lab posttrain-lab gsm8k-qwen-sft-smoke --tracked
```

The reusable inference suite contains concurrency 1/2/4/8, but the current
RTX 3070 Ti execution policy stops at 4. Its 32K cells select the model's
TurboQuant K8V4 serve variant automatically.

Start with the [post-training lifecycle](./docs/functional/finetuning-lifecycle.md), [architecture](./docs/architecture.md), and [profile model](./docs/architecture/profiles-and-model-variants.md).
