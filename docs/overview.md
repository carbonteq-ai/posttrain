# Post-training project overview

This repository develops and validates a post-training workflow and, for teams
that repeat it, a reusable framework — using small open models on a consumer
GPU. The local lab is a reference project; it is not the product boundary.
Terms of art used below are defined in the [glossary](./glossary.md).

## Product goal

Make evidence-backed finetune production repeatable: raw-model comparison,
general and domain evaluation, SFT, DPO, and GRPO stages, model lineage,
serving qualification, observability, and explicit decisions.

Start with [01 · Workflow](./post-training/01-workflow.md), then
[02 · Primitives](./post-training/02-primitives.md),
[03 · Work and Evidence](./post-training/03-work-and-evidence.md),
[04 · Framework](./post-training/04-framework.md), and
[05 · APIs](./post-training/05-apis.md), and
[06 · Observation](./post-training/06-observation-and-lineage.md).

## Architecture

```text
published Verifiers environments
              │
              ▼
typed definitions ──► code-based jobs ──► train/eval/serve packages
              │
              ▼
        execution and model artifacts
              │
              ▼
       Trackio observations + lineage
```

| Surface | Responsibility |
| --- | --- |
| `packages/common` | Lightweight job/action/invocation, model, artifact, execution-context, and observation protocols |
| `packages/data` | Trainer-neutral supervised/preference data contracts and external-format adapters |
| `packages/train` | Reusable SFT/DPO/RL operations with internal TRL or future adapters |
| `packages/eval` | Reusable evaluation operations/programs with internal Verifiers integration |
| `packages/serve` | Reusable serving/benchmark operations with an internal vLLM adapter |
| `packages/common/posttrain/common/profiles` | Reusable typed foundation-model entry points |
| `apps/lab/.../jobs` | Code-based objectives, actions, job-local policy, and decisions |
| Trackio | Pure observability: runs, snapshots, metrics, traces, artifacts, and observed lineage |

See [Post-training platform architecture](./architecture.md) and [Architecture documents](./architecture/README.md).

## Current implementation state

See the [v0.3 release notes](./releases/v0.3.md) for shipped capabilities,
qualification coverage, and support boundaries, and the
[CHANGELOG](../CHANGELOG.md) for individual versions.

## Design principles

1. Compose upstream libraries rather than recreating their frameworks.
2. Make `train`, `eval`, and `serve` reusable across projects and keep their
   internal framework adapters independently replaceable.
3. Treat model profiles as starting points and checkpoints/adapters as lineage artifacts.
4. Publish Verifiers environments independently of jobs and engine code.
5. Keep Trackio observational: capture resolved inputs and native evidence for every execution without placing behavior there.
6. Keep 8 GB constraints explicit; use short-context and parameter-efficient training where required.

## Setup

To install a released version, see [install.md](./install.md). For a
framework checkout (Python 3.13 `uv` workspace), see
[contributing.md](./contributing.md) and
[tooling/mise-uv/setup-environment.md](./tooling/mise-uv/setup-environment.md).
