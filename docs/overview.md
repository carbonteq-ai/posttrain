# Project overview

Local post-training lab for small open models on a consumer GPU. The project uses a generic uv workspace so model, training, evaluation, serving, and environment work can evolve independently.

## Product goal

Support the complete path from raw-model comparison through repeated SFT, DPO, and GRPO stages while retaining model lineage, evaluation evidence, serving observations, and human decisions.

The detailed product journey is in [Post-training job lifecycle](./functional/finetuning-lifecycle.md).

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
| `packages/common` | Lightweight Job/action/request, identity, artifact-reference, provenance, and Trackio-write contracts |
| `packages/train` | Reusable SFT/DPO/RL operations with internal TRL or future adapters |
| `packages/eval` | Reusable evaluation operations/programs with internal Verifiers integration |
| `packages/serve` | Reusable serving/benchmark operations with internal vLLM/SGLang adapters |
| `profiles/models` | Reusable typed model entry points |
| `jobs/` | Code-based objectives, actions, job-local policy, and decisions |
| Trackio | Pure observability: runs, snapshots, metrics, traces, artifacts, and observed lineage |

See [Post-training platform architecture](./architecture.md) and [Architecture documents](./architecture/README.md).

## Current implementation state

The current uv workspace has the domain packages and an earlier YAML profile/run
implementation. That implementation is evidence for the refactor, not a
compatibility contract. The target replaces it with typed Python definitions,
code-based jobs, and temporary execution workspaces.

The next vertical slice will add:

1. the code-first Job/action/request SDK;
2. typed model and engine definitions for the selected base models;
3. vLLM/SGLang baseline, MTP, and TurboQuant serving configs;
4. independently packaged Verifiers environments;
5. one end-to-end code-based job proving multiple runs, artifact lineage, and reevaluation.

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
uv run --package common profile-resolve --help
```
