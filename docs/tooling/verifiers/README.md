# Verifiers (library)

Notes **about** the Verifiers dependency. Product contracts live in
[docs/post-training/](../../post-training/README.md) — especially
[02 · eval evidence](../../post-training/02-primitives.md#verifiers-backed-eval-evidence)
and [06 · ingest](../../post-training/06-observation-and-lineage.md#verifiers-ingest-notes).

## Install / pin

- Via **`packages/eval` `verifiers` extra**:
  `uv sync --package eval --extra verifiers --python 3.12`
- Also pulls **`prime`** CLI (Environments Hub: `prime env install` / `prime eval run`)
- Workspace pins upstream `main` at commit
  `284a868d6a9022109b749710672a0460e8a996d4` (2026-07-19), not the older `0.2.0`
  release — API under `verifiers.v1` (legacy env API still present but not our
  contract)

When advancing Verifiers: pin a reviewed commit, refresh `uv.lock`, run a
taskset smoke eval, record the new SHA here. No unpinned moving branch.

## What Verifiers v1 exposes (eval / RL)

Not a suite API. Composition:

```text
TaskData + Task (@reward / @metric / tools / …)
  -> Taskset.load()
Taskset + Harness + EnvConfig limits
  -> Environment
EvalConfig (+ model client, sampling, num_tasks, rollouts, …)
  -> run_eval -> list[Trace]  (traces.jsonl + config.toml)
```

- Env packages are independently installable plugins (Hub `org/name` or local);
  they export a Taskset subclass via `__all__`
- Same environment model for **evaluation** and **online RL** rollouts
- `ProgramResult` is a harness concept — not an evaluation-plan type

## Our integration contract

| Ours | Verifiers |
| --- | --- |
| `EnvironmentBinding` / env catalog entry | Published package → `EnvConfig` (taskset id + harness + params) |
| `EvaluationPlan` | Which cells/budgets/slices/aggregation — **framework**, not Verifiers |
| `eval.general` / `eval.domain` (target) | One cell → `EvalConfig` + `run_eval`; `EvaluationBudget` may bound the task count and request Verifiers' fixed-seed shuffle |
| Evidence | **Save native traces**; project aggregates; do not replace scoring |

Authoritative evidence = native Verifiers bundle (`traces.jsonl`, resolved
config, logs). Observer stores `VerifiersTrace` projections; `eval/*` metrics
and reports **extract** from traces. Partial sync ≠ invented zeros.

Prototype path today: `posttrain.eval.evaluate` →
`backends/verifiers/adapter.py` (`EnvConfig` factory → `EvalConfig` →
`run_eval` → synchronizer + `verifiers-evaluation` artifact).

## Practice notes

- Prefer the same versioned taskset for held-out eval and online RL when
  semantics match
- Track the open
  [environment-data packaging feedback](../../feedback/verifiers-environment-data-packaging.md)
  when an environment needs package-owned or externally staged task resources
- Keep training reward **weights** in training settings; reward **meanings** in
  the env Task
- Resolve and record the **inference binding** (endpoint + engine limits); the
  eval adapter forwards context limits into Verifiers as `max_total_tokens`
- Use `EvaluationBudget(num_tasks=...)` for a cheap invocation-scoped subset;
  add `shuffle=True` for Verifiers' reproducible fixed-seed sample. The
  effective `head` or `verifiers-fixed-shuffle` policy is recorded in run
  evidence. Environment activations still own semantic splits, categories, and
  balancing; the framework does not copy task rows or invent task IDs.
- Inspect truncation / `finish_reason=length` rates — length-capped scores are
  weak capability evidence
- Source-data cards stay under [datasets/](../../datasets/); env implementation
  stays in env packages (e.g. `environments/automationbench_v1`)
- `packages/train` dataset pin can conflict with Hub envs — switch sync when
  needed

Architecture docs under `docs/architecture/` are stale pending reconcile; do
not treat them as overriding the post-training baseline.
