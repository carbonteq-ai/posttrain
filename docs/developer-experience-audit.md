# Developer experience audit

> **Superseded (2026-07-23).** Use [developer-experience.md](./developer-experience.md)
> for vocabulary, golden path, lab role, and the `WorkPackageHost*` rename map.
> This audit remains as historical diagnosis only; do not treat its “host SDK”
> recommendations as product direction.

This audit evaluates Posttrain from the perspective of a developer who has a
model, data, and a use case and wants to build an SFT, preference-training,
GRPO, distillation, or qualification workflow. It evaluates the installed
framework, not the convenience of working inside this monorepo.

## Executive assessment

Posttrain has a strong internal model and a credible evidence boundary, but it
is not yet a self-sufficient fine-tuning developer product.

Today it is best described as a framework construction kit:

- The reusable contracts, selections, job kinds, tracking, and lineage model
  are substantial.
- A project can be discovered, its catalog can be composed, and a work-package
  job can execute from installed wheels through an explicit host.
- A new developer still has to understand catalog schemas, framework types,
  job-definition factories, backend operations, and tracking composition before
  their first fine-tuning run.

The largest gap is not another training algorithm. It is the missing path from
"I have a model and dataset" to "I ran one tracked fine-tuning job."

## Who the current experience serves

Three roles are currently mixed together:

| Role | Goal | Current fit |
| --- | --- | --- |
| Fine-tuning developer | Configure a model, data, technique, and target; run and compare results | Weak |
| Project host author | Bind project data sources, job definitions, backends, tracking, and runtime policy | Emerging |
| Framework maintainer | Add selection types, adapters, recipes, telemetry, and backend support | Strongest |

The public documentation explains the framework-maintainer model well. It does
not yet hide that model for a fine-tuning developer.

## Current journey

### 1. Installation

The GitHub wheelhouse is reproducible and checksum-protected, but installation
requires downloading an archive, unpacking it, carrying a constraints file, and
manually selecting component packages. The resulting `uv pip install` does not
by itself add a durable framework requirement to the consuming project's
`pyproject.toml` and `uv.lock`.

This is acceptable release transport, but not yet a natural project dependency
experience.

### 2. Project initialization

`posttrain init` creates:

```text
.posttrain/
  project.toml
  catalog/layer.yaml
  work_packages/README.md
  .gitignore
```

The generated catalog is empty and the README says only to add work-package
YAML. It does not create:

- A project Python package.
- A host factory.
- A dataset adapter.
- A runnable CPU example.
- An SFT, GRPO, or evaluation template.
- A dependency declaration.

Initialization proves the layout, but does not lead to a successful first run.

### 3. Model, data, and runtime configuration

Models, targets, inference bindings, workloads, environments, evaluation
plans, and training settings have versioned catalog concepts. This is a good
reproducibility foundation.

Authoring is difficult:

- Catalog files are handwritten YAML with no `posttrain add`, schema export,
  completion, or guided generator.
- The current catalog decoder does not provide a declarative project dataset
  source. SFT and DPO data sources are commonly constructed in Python by the
  lab host.
- Training requires multiple related selections: model, settings, training
  binding, target, renderer, and sometimes rollout inference. Errors are
  individually correct but the system does not guide a developer toward a
  compatible set.
- The base catalog is useful as reference material, but discovering which
  selections form a supported recipe requires reading YAML and implementation
  code.

### 4. Work-package authoring

Work packages make project intent, selections, and job extent explicit. They
are a useful durable artifact.

The developer must currently write both:

1. A YAML work package that names recipe jobs and selection bindings.
2. Python host code that maps every job-definition ID to typed operations and
   selects tracking, scratch, and backend behavior.

The explicit host boundary is architecturally correct. The usability problem is
that there is no small host SDK or standard first-party definition bundle, so
developers must imitate `posttrain-lab`.

### 5. Validation and planning

`doctor`, catalog validation, composition validation, and host-backed job
preflight catch real structural errors before GPU execution.

They do not yet answer the operational questions a developer needs before an
expensive run:

- Are the selected extras installed?
- Can the model and dataset revisions be accessed?
- Is the renderer compatible with the model?
- Does the target GPU satisfy memory and placement constraints?
- Which job will run, which artifacts will it consume, and where will evidence
  be written?
- What is estimated to download or materialize?

There is no resolved execution-plan view.

### 6. Execution and retries

Execution must be job-level. A work package is a decision and evidence grouping,
not a retry unit. The developing CLI now requires:

```bash
posttrain work-package run PACKAGE.yaml \
  --job JOB_ID \
  --host PROJECT_MODULE:FACTORY
```

This correctly creates one run for one selected job. Remaining gaps include
resume from a recovery checkpoint, retry metadata, cancellation, status
inspection, and a scheduler/remote-executor boundary.

### 7. Artifacts and pipeline continuation

The framework correctly distinguishes recovery checkpoints from immutable
descendant artifacts and records artifact lineage through tracking.

The project-author experience is incomplete. After an SFT job produces an
adapter, there is no guided command to:

- Inspect produced artifact identities.
- Materialize or nominate one output.
- Add that exact descendant to a project catalog overlay.
- Bind it into the next GRPO or qualification package.

This makes an SFT-to-GRPO "pipeline" conceptually sound but operationally
manual.

### 8. Evidence and comparison

This is the strongest developer-facing area. Runs have stable project,
work-package, job-kind, selection, metric, event, trace, artifact, and outcome
contracts. Trackio and W&B are adapters, and Observatory provides normalized
read views.

The missing bridge is discoverability from the primary CLI. A developer should
not have to know provider IDs and separate Observatory environment variables
just to answer "what happened to my run?"

### 9. Remote execution

The release can be installed on a remote GPU, and a reproducible qualification
gate is being added. A project developer still performs remote execution
manually. There is no execution-target adapter that stages a locked project,
starts one job remotely, streams its status, and returns the logical run ID.

Remote execution should remain a host concern, but the host contract needs a
standard interface.

## Maturity score

| Journey area | Score | Reason |
| --- | ---: | --- |
| Reproducible concepts and contracts | 4/5 | Clear identities, selections, work, run, and evidence boundaries |
| Release transport | 3/5 | Immutable and verified, but cumbersome to add and lock |
| First project setup | 1/5 | Valid skeleton, no runnable fine-tuning path |
| Model/runtime selection | 2/5 | Strong schema, manual authoring and compatibility discovery |
| Dataset onboarding | 1/5 | No declarative project dataset golden path |
| Work authoring | 2/5 | Durable YAML plus too much required Python host glue |
| Preflight | 2/5 | Structural validation exists; operational planning is missing |
| Job execution | 3/5 | Typed runner and job-level CLI exist on the release branch |
| Artifact continuation | 2/5 | Lineage contract exists; next-branch workflow is manual |
| Evidence and analysis | 4/5 | Strong normalized tracking and Observatory model |
| Retry, resume, and remote operation | 1/5 | Mostly host-specific procedures |
| Learning material | 2/5 | Deep technical docs, no end-to-end project tutorial |

## Highest-leverage opportunities

### P0: Deliver one runnable project template

Add:

```bash
posttrain init support-agent --template sft
```

The template should include a Python package, `pyproject.toml`, locked
dependencies, project host, tiny local/Hugging Face dataset adapter, catalog
overlay, CPU data-validation job, bounded GPU SFT job, Trackio configuration,
and qualification package. CI must execute the same generated project from
installed wheels.

One complete SFT path is more valuable than several additional isolated
operations.

### P0: Make datasets first-class project selections

Define declarative source bindings for Hugging Face, local JSONL/Parquet, and
project Python factories. Separate source identity from the materialized
dataset snapshot. Add `posttrain dataset inspect` and `posttrain dataset
validate` so schema, renderer, partition, and leakage checks happen before
training.

### P0: Publish a small host SDK and first-party job bundle

Fine-tuning projects should configure standard SFT, DPO, GRPO, distillation,
evaluation, and serving definitions without recreating request translation.
Host code should mainly select:

- Which standard definitions are enabled.
- Dataset/environment factories.
- Tracking backend.
- Local or remote executor.
- Secret and scratch policy.

The host remains explicit, but ordinary projects should need tens of lines, not
copies of the lab application.

### P0: Finish job-level run control

Keep work packages as groupings and jobs as execution units. Add:

- `posttrain job plan PACKAGE --job ID`.
- `posttrain job run PACKAGE --job ID`.
- `posttrain run show RUN_ID`.
- `posttrain run retry RUN_ID` with explicit lineage.
- Backend-specific resume through a selected recovery checkpoint.

Do not make "run the entire YAML again" the recovery path.

### P1: Add a resolved plan and compatibility preflight

Before GPU allocation, print the exact model revision, dataset snapshot,
renderer, backend versions, target, estimated downloads, artifact inputs,
tracking destination, and missing credentials or extras. Support deterministic
JSON for CI and schedulers.

### P1: Make artifact continuation explicit and guided

After a training run, show produced descendants and generate the project-overlay
entry or next-package binding without automatically promoting anything. The
human still decides which artifact to retain.

### P1: Unify run discovery

Add primary-CLI commands that resolve the configured tracking source and open
or query Observatory:

```bash
posttrain run show <run-id>
posttrain work-package status <package-id>
posttrain observatory open
```

Provider details should remain available but should not be required for the
common path.

### P1: Make project installation lockable

Until PyPI publication, provide a generated project dependency file or uv
configuration that records the GitHub release tag, wheelhouse hash, fork
constraints, and selected extras. The remote server should be able to reproduce
the environment with one project-owned command.

### P2: Standardize remote executors

Define a host executor protocol for local process, SSH, and later scheduler
backends. The logical `RunSpec` and evidence contract should stay identical.
Remote staging, status, cancellation, and log transport should not leak into
training operations.

## Target experience

The following is a product direction, not a claim about current commands:

```bash
posttrain init support-agent --template sft
cd support-agent
uv sync --locked

posttrain dataset add hf \
  --repo carbonteq/support-conversations \
  --revision <immutable-revision>
posttrain dataset validate datasets/support-sft

posttrain job plan .posttrain/work_packages/sft.yaml --job train
posttrain job run .posttrain/work_packages/sft.yaml --job train
posttrain run show <run-id>

posttrain artifact inspect <artifact-id>
posttrain work-package create qualify --from-artifact <artifact-id>
```

The commands may evolve, but the experience should preserve these properties:

- A generated project runs before the developer learns framework internals.
- Every expensive action is planned and preflighted.
- One command executes one job and creates one run.
- Artifacts, not directory order, connect fine-tuning stages.
- The project owns policy; the framework supplies standard capabilities.
- The same locked project works locally, in CI, and on a remote GPU.

## Recommended next milestone

Build and test an external `sft-starter` fixture generated by `posttrain init
--template sft`. It should take a tiny public or local dataset through data
validation, one bounded training job, artifact publication, and Observatory
readback. Use that fixture to drive the dataset selection, host SDK, job plan,
job run, artifact continuation, and upgrade experiences.

Do not expand to more techniques until this single path is understandable and
reproducible without reading `apps/lab`.
