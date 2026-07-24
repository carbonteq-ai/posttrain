# Posttrain

Posttrain is CarbonTeq's framework for taking a base model through screening,
training, and qualification. It gives projects a shared way to select models,
data, environments, inference engines, training methods, evaluation plans, and
execution targets while keeping project policy in the project repository.

The framework is built around three stages:

| Stage | Question | Typical work |
| --- | --- | --- |
| Screen | Is this model and runtime combination worth pursuing? | Capability checks, inference benchmarks, dataset validation |
| Train | Can we produce a better model variant? | SFT, preference training, GRPO, distillation, quantization |
| Qualify | Is the resulting variant ready for its intended use? | General and domain evaluation, performance checks, evidence review |

A **work package** captures one decision-making unit across those stages. It
contains ordered jobs, resolved catalog selections, and the evidence needed to
understand or reproduce a run.

## Quickstart

Posttrain currently ships to the team as a versioned GitHub Release wheelhouse.
Python 3.12, [`uv`](https://docs.astral.sh/uv/), and the GitHub CLI are required.

Download and install one exact release:

```bash
gh release download <release-tag> \
  --repo carbonteq-ai/posttrain \
  --pattern 'posttrain-wheelhouse-*.tar.gz'

mkdir posttrain-wheelhouse
tar -xzf posttrain-wheelhouse-*.tar.gz -C posttrain-wheelhouse

uv venv --python 3.12
uv pip install \
  --python .venv/bin/python \
  --constraint posttrain-wheelhouse/github-constraints.txt \
  --find-links posttrain-wheelhouse \
  posttrain
```

Initialize and install an SFT starter. Point uv at the same wheelhouse so the
generated project can resolve the unpublished team release:

```bash
POSTTRAIN="$(pwd)/.venv/bin/posttrain"
WHEELHOUSE="$(pwd)/posttrain-wheelhouse"

UV_FIND_LINKS="$WHEELHOUSE" \
UV_CONSTRAINT="$WHEELHOUSE/github-constraints.txt" \
"$POSTTRAIN" init my-model-project --template sft --project-id my-model-project
cd my-model-project

.venv/bin/posttrain doctor
.venv/bin/posttrain dataset validate datasets/posttrain-sft-smoke@1
.venv/bin/posttrain work-package validate sft.yaml
# CUDA release gate:
.venv/bin/posttrain work-package run sft.yaml --job train
.venv/bin/posttrain observatory up
```

`posttrain init` writes the project package and `.posttrain/` configuration,
creates the project-local `.venv`, and installs the selected extras. There is
no separate Posttrain sync command. The wheelhouse constraints file pins the
CarbonTeq forks to immutable Git commits.

## Project structure

`posttrain init --template sft` creates an installable project:

```text
my-model-project/
  pyproject.toml
  uv.lock
  .posttrain/
    project.toml       project identity and path policy
    catalog/
      settings.yaml    visible project-owned training settings
    work_packages/
      sft.yaml         standard train/trl-sft@1 composition
    state/             ignored caches, scratch, recovery, and local provider state
```

Commit `project.toml`, catalog overlays, work packages, `pyproject.toml`, and
`uv.lock`. Do not commit `.posttrain/state/`, credentials, downloaded model
weights, or machine-local caches.

The framework base catalog is included in `posttrain-catalog`; projects add only
their own selections and overrides. Project discovery checks an explicit
`--project-root`, then `POSTTRAIN_PROJECT_ROOT`, then searches upward for
`.posttrain/project.toml`.

## Define and validate work

A project catalog layer declares its files:

```yaml
# .posttrain/catalog/project/layer.yaml
schema_version: 1
layer_id: my-model-project-v1
files:
  - targets.yaml
```

The selection file contains project-specific execution policy:

```yaml
# .posttrain/catalog/project/targets.yaml
target:
  targets/local-cpu:
    revision: "1"
    device_class: cpu
    placement:
      world_size: 1
```

A work package references catalog selections instead of copying their values:

```yaml
# .posttrain/work_packages/cpu_check.yaml
project_id: my-model-project
work_package_id: screen/cpu-check
stage: screen
description: Validate the local CPU execution target.
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats:
    target: target
  jobs:
    - id: validate
      kind: data.prepare
      definition: data/cpu-check@1
bindings:
  target:
    type: ref
    family: target
    id: targets/local-cpu
enabled_optional_jobs: []
metadata:
  question: Can this project resolve and execute its local configuration?
```

Validate the composed catalog and work package before execution:

```bash
posttrain doctor
posttrain catalog list
posttrain catalog show target targets/local-cpu
posttrain work-package validate .posttrain/work_packages/cpu_check.yaml
```

The primary CLI owns initialization, diagnostics, catalog inspection,
materialization, work-package validation/execution, and local Observatory
bring-up. Standard SFT, DPO, GRPO, DAPO, SAMPO, distillation, serve, evaluation,
and model-transform definitions come from `posttrain.jobs`; projects do not
need a host or `posttrain-lab` on the common path. An optional project entry may
add unshipped definitions without redefining standard ids.

## Choose capabilities

Install only the packages a project needs from the same release wheelhouse:

| Package | Use it for |
| --- | --- |
| `posttrain-data` | Dataset preparation and canonical training data |
| `posttrain-train` | Backend-neutral training operations |
| `posttrain-eval` | Evaluation plans and Verifiers environments |
| `posttrain-serve` | Serving and inference benchmarks |
| `posttrain-tracking-trackio` | Default local tracking backend |
| `posttrain-tracking-wandb` | W&B tracking backend |
| `posttrain-observatory` | Read-only evidence queries, reports, HTTP, MCP, and UI |
| `posttrain-jobs` | Standard cross-capability job definitions and project runtime |
| `posttrain-lab` | Framework qualification scenarios and backend release gates |

Backend-specific extras are opt-in. For example:

```bash
uv pip install \
  --python .venv/bin/python \
  --constraint posttrain-wheelhouse/github-constraints.txt \
  --find-links posttrain-wheelhouse \
  'posttrain-train[trl]' \
  'posttrain-eval[verifiers]' \
  'posttrain-serve[vllm]' \
  posttrain-tracking-trackio
```

Training, evaluation, serving, and tracking remain separate reusable
capabilities. A project or host composes them through framework contracts;
those packages do not import one another.

## Run the qualification suite

Install `posttrain-lab` when developing the framework or adapting one of its
qualification workflows. Product projects should use `posttrain init` instead.

```bash
uv pip install \
  --python .venv/bin/python \
  --constraint posttrain-wheelhouse/github-constraints.txt \
  --find-links posttrain-wheelhouse \
  posttrain-lab posttrain-observatory

.venv/bin/posttrain-lab foundation-qwen-smoke --tracked
```

From a framework checkout, YAML qualification packages also run through the
primary CLI with no `--host` (repo-root `.posttrain/project.toml` entry):

```bash
uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/foundation_screen.yaml
```

`--tracked` uses local Trackio by default. W&B is available through
`--tracking-backend wandb` when its credentials and project settings are
provided. Runs expose provider-neutral metrics, events, traces, artifacts, and
outcomes to Observatory.

## Framework development

Clone an exact tag when contributing to the framework itself:

```bash
git clone --branch <release-tag> --depth 1 \
  https://github.com/carbonteq-ai/posttrain.git
cd posttrain

mise install
uv sync --all-packages --locked --python 3.12
uv run --package posttrain posttrain doctor
uv run pytest -q tests/consumer
```

That installs the framework, applications, and development tools without the
large GPU backends. Select one workspace profile when working on training:

```bash
# Transformers + the pinned CarbonTeq TRL fork: SFT, DPO, and trainer tests
uv sync --all-packages --extra gpu-train --locked --python 3.12

# TRL + vLLM + Verifiers + AutomationBench: GRPO, DAPO, SAMPO, distillation
uv sync --all-packages --extra gpu-posttrain --locked --python 3.12
```

Use `uv run --no-sync ...` for backend-specific commands after selecting one
of these profiles so uv does not synchronize the environment back to the core
dependency set.

The complete agentic profile is Linux/NVIDIA-specific and uses the CUDA 13
PyTorch index selected by the workspace lock. veRL intentionally lives in a
separate environment because its Transformers/vLLM stack conflicts with the
TRL environment. See [Developer environment setup](./docs/tooling/mise-uv/setup-environment.md)
for prerequisites, verification commands, Hugging Face access, CUDA library
setup, and the isolated veRL boundary.

The repository is a Python 3.12 `uv` workspace:

```text
packages/       reusable contracts, capabilities, catalog, tracking, and composition
apps/cli/       primary `posttrain` command
apps/lab/       reference execution host
apps/observatory/ read-only evidence product
environments/   independently versioned domain environment packages
.posttrain/     this repository's project overlays and work packages
```

Run the full validation ladder before submitting a change:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run lint-imports
uv run pytest
git diff --check
```

## Learn the framework

Read the documentation progressively:

1. [Workflow](./docs/post-training/01-workflow.md) — screen, train, and qualify.
2. [Primitives](./docs/post-training/02-primitives.md) — reproducible
   selections.
3. [Work and evidence](./docs/post-training/03-work-and-evidence.md) — projects,
   work packages, jobs, runs, and views.
4. [Framework](./docs/post-training/04-framework.md) — package ownership and
   extension boundaries.
5. [APIs](./docs/post-training/05-apis.md) — public names and contracts.
6. [Observation and lineage](./docs/post-training/06-observation-and-lineage.md)
   — metrics, traces, artifacts, and provenance.

For release installation, remote-server use, package ordering, and known
release gaps, see [Release and consumption](./docs/release-and-consumption.md).
For the project-author journey and configuration ownership, see
[Developer experience](./docs/developer-experience.md).
The smallest independent project is
[`tests/consumer/fixture`](./tests/consumer/fixture).
