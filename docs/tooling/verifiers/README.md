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
`b7bcb591facfcd2b073802f6d7496b24ab9c479e` and keeps each package independently
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

## Online-RL cold-start selection

Use correctness rewards whose parsing contract is explicit. The Reasoning Gym
package supports a compatibility `native` mode and a training-oriented
`boxed_exact` mode. `boxed_exact` requires one final `\\boxed{...}` answer,
passes only the extracted value to the selected generator's native verifier,
and maps native partial scores to exact zero or one. This prevents incidental
oracle text and response length from becoming hidden reward components.

For the initial 2B reasoning qualification, prefer the existing Math Python
taskset over a single easy procedural generator. Select MATH Levels 2-4,
deterministically balance by problem type, generate eight completions per
prompt, and measure the current policy before training. Retain a stratum only
when its measured pass@8 is between 10% and 90%, its exact rewards provide
two-sided group signal, and its truncation rate passes the campaign guard.
Level labels and problem types are filtered before deterministic balancing, so
selection does not depend on a fragile contiguous Hub row range.

Reasoning Gym remains useful as a deterministic coverage probe across
arithmetic, algebra, number theory, representation, symbolic logic, and calendar
reasoning. Do not infer training suitability from generator diversity alone;
promote only the measured difficulty strata. Larger sources such as Skywork
OR1 follow the same rule and require a pinned environment adapter plus
policy-specific difficulty measurement before joining the campaign.

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

## Facets and compound breakdowns

This reporting contract is available to every Verifiers environment. The
environment package emits task metadata; its `EnvironmentBinding` promotes
stable fields to independently filterable facets. An `EvaluationPlan` may then
select a two-dimensional breakdown for a particular environment. Observatory
reads the resolved, versioned run contract. It does not infer combinations from
task names or from the current catalog.

For example, a math environment can expose two native fields:

```yaml
environments:
  math-python-release:
    # source, activation, execution limits, and signals omitted
    observation:
      primary_metric: math_reward
      pass_rate_metric: symbolic_correctness
      facets:
        - field: problem_type
          dimension: problem_type
          label: Problem type
        - field: level
          dimension: difficulty
          label: Difficulty
```

The evaluation plan chooses how those dimensions should be combined for this
run:

```yaml
evaluations:
  math-release-v1:
    revision: "1"
    kind: general
    environments: [math-python-release]
    success:
      math-python-release:
        id: symbolic-correctness
        label: Symbolically correct
        source: {namespace: metric, name: symbolic_correctness}
        predicate: {operator: eq, value: 1}
    breakdowns:
      math-python-release:
        - id: problem-type-by-difficulty
          label: Problem type × difficulty
          dimensions: [problem_type, difficulty]
          presentation: matrix
          multi_value: reject
          missing: exclude
```

The same mechanism can represent instruction family by complexity, generator
by difficulty, domain by workflow type, or any other pair of meaningful native
facets. Add the breakdown only after the environment emits both source fields;
do not parse presentation labels or synthesize dimensions from task IDs.

Current policy is deliberately explicit:

- `dimensions` contains exactly two distinct facet dimension ids declared by
  the selected environment binding.
- `multi_value: reject` is the safe default. `cross` includes a trace in the
  Cartesian product of its values and can make group counts exceed the trace
  count, so use it only when that reporting meaning is intended.
- `missing: exclude` keeps incomplete traces out of the matrix and reports the
  excluded count. `bucket` retains them under a visible missing-value group.
- The stored task identity remains structured. Labels such as
  `Algebra · Level 4` are UI presentation, so each dimension remains usable for
  filtering and future views.
- Changing facets, the success predicate, or a breakdown requires a new binding
  or plan revision. Existing schema-v1/v2 runs keep their original meaning;
  compound breakdowns appear only when snapshotted in a schema-v3 run.

Validate the catalog, then inspect the detached plan before packing or running:

```console
uv run --package posttrain posttrain --project-root apps/lab catalog validate
uv run --package posttrain posttrain --project-root apps/lab --json \
  job plan apps/lab/.posttrain/work_packages/<qualification>.yaml \
  --job evaluate > /tmp/posttrain-eval-plan.json
jq '.resolved_inputs.evaluation.plan.breakdowns' /tmp/posttrain-eval-plan.json
```

The product-level ownership and historical-evidence rules remain authoritative
in [05 · APIs](../../post-training/05-apis.md#environmentbinding) and
[06 · observation and lineage](../../post-training/06-observation-and-lineage.md#eval-metrics).

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
- Native online-RL bridges should implement the optional observed-rollout
  extension so each completed trajectory is preserved and submitted to the
  run observer immediately. Posttrain serializes local observer submission off
  the rollout event loop; Trackio owns background remote delivery and retry.
  Bridges that implement only the batch-return contract remain compatible but
  expose traces only after the complete batch returns.
- Source-data cards stay under [datasets/](../../datasets/); environment
  implementations stay in the external `carbonteq-ai/verifiers-environments`
  packages (for example its `environments/automationbench_v1` subdirectory)
- `packages/train` dataset pin can conflict with Hub envs — switch sync when
  needed

Architecture docs under `docs/architecture/` are stale pending reconcile; do
not treat them as overriding the post-training baseline.
