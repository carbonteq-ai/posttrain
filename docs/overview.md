# Post-training project overview

This repository develops and validates a post-training workflow and, for teams
that repeat it, a reusable framework — using small open models on a consumer
GPU. The local lab is a reference host; it is not the product boundary.

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

The common execution contracts, code-defined lab host, Trackio observation
adapter, typed foundation/serve profiles, packaged benchmark data, workload
matrix, and vLLM benchmark operation are implemented. The legacy YAML eval and
training paths are still being replaced and are not compatibility contracts.

The next vertical slices will add:

1. endpoint-neutral, code-defined Verifiers evaluation operations and programs;
2. independently packaged Verifiers environments;
3. renderer-aware SFT and DPO operations;
4. a Verifiers-to-TRL GRPO operation;
5. one end-to-end job proving model lineage and reevaluation.

## Design principles

1. Compose upstream libraries rather than recreating their frameworks.
2. Make `train`, `eval`, and `serve` reusable across projects and keep their
   internal framework adapters independently replaceable.
3. Treat model profiles as starting points and checkpoints/adapters as lineage artifacts.
4. Publish Verifiers environments independently of jobs and engine code.
5. Keep Trackio observational: capture resolved inputs and native evidence for every execution without placing behavior there.
6. Keep 8 GB constraints explicit; use short-context and parameter-efficient training where required.

## Setup

```bash
cd /home/hammad/projects/rl
mise install
uv sync --all-packages --python 3.12
uv run --package posttrain-lab posttrain-lab noop --tracked
```
