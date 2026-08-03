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

CarbonTeq's maintained Verifiers v1 environment packs live in the separate
framework-neutral [verifiers-environments repository](https://github.com/carbonteq-ai/verifiers-environments).
The current framework integration uses published commit
`017ac72f543f79f48400cbb4cb641d6df4c3adfa` and keeps each package independently
installable:

| Package | Taskset | Source data / generator revision |
| --- | --- | --- |
| `gsm8k-v1` | `gsm8k-v1` | `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` |
| `automationbench-v1` | `automationbench-v1` | CarbonTeq AutomationBench fork pinned by its package lock |
| `mmlu-pro-v1` | `mmlu-pro-v1` | `TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| `ifeval-v1` | `ifeval-v1` | `google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84` |
| `reasoning-gym-v1` | `reasoning-gym-v1` | `open-thought/reasoning-gym@49b07130b3fcd12f2d064bba7c43869543a0e7e7` |
| `math-python-v1` | `math-python-v1` | `DigitalLearningGmbH/MATH-lighteval@0530c78699ea5e8eb5530600900e1f328b48acad` |

The four-pack `general-capability-balanced-v1` catalog evaluation covers
MMLU-Pro, IFEval, Reasoning Gym, and Math Python. The Lab overlay adds six
bounded one-cell qualification work packages so each native trace can be
inspected independently. The Math Python image is published in the CarbonTeq
OCI registry at
`registry.lan/carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`.
Its lifecycle cleanup gate remains; the current overlay uses the package's
subprocess path and must not be treated as a sandbox-isolation claim.

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
