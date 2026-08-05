# Posttrain

Posttrain is CarbonTeq's framework for taking a base model through screening,
training, and qualification — with every decision versioned, every run
reproducible, and every result traceable to exactly what produced it.

## Why it exists

Most post-training efforts fail as process, not as algorithms. A model gets
fine-tuned with a pile of scripts; three weeks later nobody can say which
dataset revision, which chat template, which inference settings, or which
dependency set produced the eval numbers in the report. Comparisons quietly
become invalid because one input drifted between runs. "Is this quantized
model actually derived from that checkpoint?" is answered from memory, not
evidence.

Posttrain makes those questions mechanically answerable. Every input to a run
— model, dataset, environment, inference engine, training method, execution
target — is a versioned selection in a layered catalog. Your project never
copies configuration; it *references* catalog selections, and the framework
resolves, plans, and packs them into an immutable job before a GPU or provider
is ever contacted. If any input changes after planning, packing fails instead
of silently running something else. Every run records its resolved identity
and its evidence — metrics, traces, produced and consumed artifacts — so
lineage is a property the system enforces, not tribal knowledge.

The result: raw-model comparison, SFT, preference training, GRPO, evaluation,
serving qualification, and quantization all become one auditable workflow that
a team can repeat, instead of a folder of scripts only one person understands.

## How it works

The framework is built around three stages:

| Stage | Question | Typical work |
| --- | --- | --- |
| Screen | Is this model and runtime combination worth pursuing? | Capability checks, inference benchmarks, dataset validation |
| Train | Can we produce a better model variant? | SFT, preference training, GRPO, distillation, quantization |
| Qualify | Is the resulting variant ready for its intended use? | General and domain evaluation, performance checks, evidence review |

A **work package** captures one decision-making unit across those stages: an
ordered list of jobs, the catalog selections they bind, and the evidence
needed to understand or reproduce the outcome. You author it as YAML, validate
it statically, and execute its jobs locally or on a remote GPU provider. The
same resolved identities travel from an early screening run through training,
qualification, artifact handoff, and cleanup — backend adapters (TRL, veRL,
vLLM, Verifiers) are replaceable; the project-owned workflow and its lineage
are not.

Execution is deliberately split into three steps, and each is a command:

1. **Plan** (`posttrain job plan`) resolves every selection — no provider, no
   GPU, no ML backend contacted.
2. **Pack** (`posttrain job pack`) turns the plan into a content-addressed OCI
   image containing source, wheels, dataset artifacts, and dependency locks.
   Packing fails if any input drifted after planning.
3. **Run** (`posttrain job run`, or `posttrain work-package run`) submits the
   packed job. A durable run lifecycle (`posttrain run status|wait|logs|cancel|
   retry-submit|reconcile`) and a reconciling controller keep it crash-safe
   after your shell exits.

`posttrain job diff` explains why two packed jobs have different identities —
the "why did this run differ" question has a command, not a meeting.

All evidence lands in provider-neutral tracking (local Trackio by default,
W&B optional) and is explored through **Observatory**, a read-only product
showing metrics, traces, comparisons, and lineage without changing the meaning
of earlier runs.

New vocabulary in this README — work package, catalog, selection, binding,
seat, evidence, screen, qualification — is defined in the
[glossary](./docs/glossary.md).

## Quickstart

Posttrain currently ships to the team as a versioned GitHub Release
wheelhouse. Python 3.13, [`uv`](https://docs.astral.sh/uv/), and the GitHub
CLI are required. On the internal network you can install from `pypi.lan`
instead — both paths, plus CA trust and machine configuration, are covered in
the [installation guide](./docs/install.md).

Download and install one exact release:

```bash
gh release download <release-tag> \
  --repo carbonteq-ai/posttrain \
  --pattern 'posttrain-wheelhouse-*.tar.gz'

mkdir posttrain-wheelhouse
tar -xzf posttrain-wheelhouse-*.tar.gz -C posttrain-wheelhouse

uv venv --python 3.13
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
.venv/bin/posttrain dataset materialize datasets/posttrain-sft-smoke@1
.venv/bin/posttrain work-package validate sft.yaml
# CUDA release gate:
.venv/bin/posttrain work-package run sft.yaml --job train
.venv/bin/posttrain observatory up
```

`posttrain init` writes the project package and `.posttrain/` configuration,
creates the project-local `.venv`, and installs the selected extras. There is
no separate Posttrain sync command. The wheelhouse constraints file pins the
CarbonTeq forks to immutable Git commits.

This exact sequence is exercised in CI by
[`tests/consumer/test_wheel_project.py`](./tests/consumer/test_wheel_project.py);
if the quickstart and that test ever disagree, the test is right.

For the full first-day walkthrough — machine configuration, credentials,
local Docker and dstack execution, and passing one job's model into the next —
continue with [Getting started](./docs/getting-started.md).

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

The framework base catalog is included in `posttrain-catalog`; projects add
only their own selections and overrides. Project discovery checks an explicit
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
bring-up. Standard SFT, DPO, GRPO, DAPO, SAMPO, distillation, serve,
evaluation, and model-transform definitions come from `posttrain.jobs`;
projects do not need a host or `posttrain-lab` on the common path. An optional
project entry may add unshipped definitions without redefining standard ids.

The smallest complete independent project is
[`tests/consumer/fixture`](./tests/consumer/fixture) — a project overlay, one
work package, a custom job definition, and evidence read-back in a handful of
files.

## What you can build

| Workflow | Included capabilities |
| --- | --- |
| Project setup | Installable SFT and GRPO starters, project-local catalogs and work packages, and machine-level configuration for providers, tracking, registries, package indexes, storage, trust, and credentials |
| Data and environments | Reproducible supervised and preference datasets, serving workloads, project-owned Verifiers packages, deterministic subsets, and immutable source and builder identities |
| Training | SFT, DPO, GRPO, DAPO, SAMPO, and on-policy distillation through TRL or maintained veRL profiles, with full-parameter and adapter-based updates where supported |
| Serving | vLLM smoke tests and capacity sweeps with latency, throughput, memory, KV-cache, MTP, eligibility, and Pareto evidence |
| Evaluation | General and domain evaluation against local Verifiers environments or an OpenAI-compatible policy endpoint, with explicit success criteria and native traces |
| Model production | AWQ and RTN W4A16 model transformation, immutable model variants, and artifact handoff between work packages |
| Execution | Provider-free planning, immutable OCI packaging, local-container and dstack execution, shared GPU admission, queue inspection, logs, wait, cancel, retry, and recovery |
| Operations | Continuous job reconciliation, evidence-preserving cleanup, digest-confirmed purge, and read-only evidence exploration through Observatory |

### Supported jobs

| Capability | Job kind |
| --- | --- |
| Prepare supervised or preference data | `data.prepare` |
| Supervised fine-tuning | `train.sft` |
| Direct preference optimization | `train.dpo` |
| GRPO and DAPO | `train.grpo` |
| Multi-turn SAMPO | `train.sampo` |
| On-policy distillation | `train.distill` |
| Serving smoke and capacity tests | `serve.smoke`, `serve.benchmark` |
| General and domain evaluation | `eval.general`, `eval.domain` |
| AWQ or RTN quantization | `model.transform` |

### Models and environments

The base catalog includes Qwen 3.5 0.8B, 2B, and 4B; LFM 2.5 1.2B Thinking;
and Gemma 4 E2B, E4B, 12B Unified, and 31B. Available training, serving, tool
use, and acceleration profiles vary by model size.

Six versioned Verifiers environments are available as independently
installable packages:

| Environment | Use |
| --- | --- |
| `gsm8k-v1` | Grade-school mathematical reasoning |
| `automationbench-v1` | Multi-turn tool use over AutomationBench Simple tasks |
| `mmlu-pro-v1` | Knowledge and reasoning across 14 categories |
| `ifeval-v1` | Verifiable instruction following |
| `reasoning-gym-v1` | Procedural reasoning across ten generators |
| `math-python-v1` | Competition mathematics with Python tools and symbolic checking |

`posttrain environment new` scaffolds a project-owned environment package;
`posttrain environment add local` binds it into the project catalog.

Each run records the exact environment package, task population, model,
inference settings, success criteria, and native traces used to produce its
results. Observatory presents coverage, pass rate, rewards, latency,
distributions, facets, compound breakdowns, tool-aware traces, comparisons,
and lineage without changing the meaning of earlier runs.

### The execution surface

The prose above summarized the lifecycle; these are the actual commands:

| Area | Commands |
| --- | --- |
| Project | `posttrain init`, `doctor`, `catalog list/show/validate`, `work-package validate/run`, `project show/purge` |
| Data | `posttrain dataset materialize`, `dataset add hf\|jsonl\|nemo`, `workload materialize/verify` |
| Jobs | `posttrain job plan`, `job pack`, `job run`, `job diff` |
| Runs | `posttrain run list/queue/status/wait/logs/cancel/retry-submit/reconcile/cleanup/purge/show`, `run recover-cancelled-tracking` |
| Machine | `posttrain machine init/show`, `machine project add`, `posttrain workers`, `posttrain state migrate/cache-prune` |
| Images | `posttrain runtime images list/verify/mirror/build` |
| Environments | `posttrain environment new`, `environment add local` |
| Evidence | `posttrain observatory up` |
| Controller | `posttrain controller run [--once]`, `controller status` |

`posttrain controller run` continuously reconciles queued and active jobs
beyond the submitting shell. It submits work when capacity becomes available,
delivers cancellation requests, checks the provider recorded at submission,
finalizes tracking evidence, releases settled GPU admissions, and writes
recovery receipts. Run it under your service supervisor for unattended
operation. Posttrain v0.3 does not include service installation, `controller
enable` or `controller disable`, a systemd unit, or an Ansible role.

### Reliability and performance

- Plans and packages include the complete selected catalog, source, dataset,
  environment, runtime, and dependency identity. Packing fails if those inputs
  change after planning.
- Job images contain their environment wheels and dependency locks before
  submission, so workers do not install or upgrade packages at startup.
- Cancellation, reconciliation, and admission release are safe to retry after
  an interrupted client or controller process.
- Verifiers rollout groups run concurrently, distillation scores the exact
  student response tokens, and serving profiles expose batching, KV-cache,
  native or paired-assistant MTP, and speculative-acceptance measurements.
- Evaluation keeps reward, configured success, errors, truncations, missing
  signals, and coverage separate. Partial traces remain visible while a run is
  active.
- Release artifacts are built from committed source, installed in a clean
  consumer, checked against immutable image digests, and exercised through a
  packed dstack GPU canary before promotion.

### Current support boundaries

- Online RL is synchronous; Posttrain does not currently provide asynchronous
  learner and rollout execution.
- Gemma 4 qualifications are bounded, text-only profiles. They do not establish
  multimodal training or full native-context support.
- Standard KV is the qualified default for hybrid Qwen training paths;
  TurboQuant K8V4 long-context use remains experimental.
- The six environment packages have provider-backed activation and execution
  evidence, but that does not mean every full catalog population has completed.
  See the [v0.3 release notes](./docs/releases/v0.3.md) for the exact coverage.

See the [v0.3 release notes](./docs/releases/v0.3.md) for release-specific
capabilities and qualification, the [CHANGELOG](./CHANGELOG.md) for individual
versions, and the [product baseline](./docs/post-training/README.md) for the
public contracts.

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

These are the packages meant for direct installation. The workspace contains
further internal packages (contracts, catalog, execution providers, packing);
they arrive as dependencies and are not installed by name.

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

## Screen serving capacity

Serving capacity is measured as one bounded concurrency sweep per model,
inference binding, workload, and execution target. Product constraints stay in
the project brief; backend search settings stay in the inference binding. The
benchmark records direct request/run evidence and a versioned
`serving-result.json`, while Observatory derives throughput, latency
percentiles, eligibility, and the cross-contender Pareto frontier.

The repository example uses the qualified `general-serving-v1` population: 128
deterministically selected GSM8K reasoning prompts, HumanEval code prompts, and
reviewed first-party chat, extraction, structured-output, and tool-use
messages. It fixes decode work at 128 output tokens and never treats systems
throughput as task correctness.

The lab application is itself a Posttrain project
(`apps/lab/.posttrain/project.toml`), so its qualification packages run from
that directory in a framework checkout:

```bash
cd apps/lab
uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/foundation_screen.yaml

# GPU execution gate
uv run --no-sync --package posttrain-lab posttrain-lab foundation-qwen-smoke --tracked

# Read the run and work-package evidence
uv run --package posttrain posttrain observatory up
```

The Overview shows every concurrency point, response length and request
coverage, TTFT/TPOT, memory and KV-cache pressure, runtime settings, and the
constraint-relative operating point. The work-package view keeps eligible,
constrained, failed, unsaturated, and incomparable contenders visible; only
eligible results with an identical requirements/workload/corpus/target
comparison basis can enter the strict Pareto set.

For the small real-vLLM release gate, set
`POSTTRAIN_RUN_SERVE_GPU_INTEGRATION=1` and optionally choose
`POSTTRAIN_SERVE_GPU_VARIANT=standard|mtp|turboquant|mtp-turboquant`, then run
`uv run --no-sync pytest packages/serve/tests/test_vllm_capacity_integration.py -q`.

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
primary CLI using the lab project (`apps/lab/.posttrain/project.toml`):

```bash
cd apps/lab
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
uv sync --all-packages --locked --python 3.13
uv run --package posttrain posttrain doctor
uv run pytest -q tests/consumer
```

That installs the framework, applications, and development tools without the
large GPU backends. Select one workspace profile when working on training:

```bash
# Transformers + the pinned CarbonTeq TRL fork: SFT, DPO, and trainer tests
uv sync --all-packages --extra gpu-train --locked --python 3.13

# TRL + vLLM + Verifiers + AutomationBench: GRPO, DAPO, SAMPO, distillation
uv sync --all-packages --extra gpu-posttrain --locked --python 3.13
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

The repository is a Python 3.13 `uv` workspace:

```text
packages/       reusable contracts, capabilities, catalog, tracking, and composition
apps/cli/       primary `posttrain` command
apps/lab/       reference project and qualification suite
apps/observatory/ read-only evidence product
environments/   independently versioned domain environment packages
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

## Where to go next

Pick the door that matches what you are doing:

- **I want to train a model with Posttrain** →
  [Installation](./docs/install.md), then
  [Getting started](./docs/getting-started.md).
- **I want to work on the framework itself** →
  [Contributing](./docs/contributing.md) and
  [Developer environment setup](./docs/tooling/mise-uv/setup-environment.md).
- **I am cutting or auditing a release** →
  [Release engineering](./docs/release-engineering.md) and
  [Publishing](./docs/publishing.md).

To learn the concepts behind the framework, read the baseline documents in
order:

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

The [glossary](./docs/glossary.md) defines every term of art in one place.
For the project-author journey and configuration ownership, see
[Developer experience](./docs/developer-experience.md). The full docs map is
at [docs/README.md](./docs/README.md).
