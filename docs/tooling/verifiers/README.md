# Verifiers (library)

Notes **about** the Verifiers dependency. Product contracts live in
[docs/post-training/](../../post-training/README.md) — especially
[02 · eval evidence](../../post-training/02-primitives.md#verifiers-backed-eval-evidence)
and [06 · ingest](../../post-training/06-observation-and-lineage.md#verifiers-ingest-notes).

## Install / pin

The selected independently maintained CarbonTeq distribution is
`carbonteq-ai/verifiers@1f6793f7d46e8a650a54b2a585193b4010578fa6`, based on
upstream main commit `27bbd216df0af719a43705866b2cf6139bcc95de` and retaining
the CarbonTeq host-client, selected-template, and cancellation seams. It adds
optional host-client injection through native serving/interception. The fork
ledger is `CARBONTEQ_FORK.md`; three real local harness lifecycle cases and the
complete upstream v1 suite pass.
Framework modern integration now runs AutomationBench tools, retains original
policy tokens, carries distinct explicit SAMPO turn rewards, and publishes
native Episode replay artifacts. A real loopback HTTP evaluation integration
also passes native nested-config activation, episode accounting and trace-view
synchronization. The failed R10 cloud attempt proved that the old upstream pin
cannot inject the already-loaded policy; runtime-image publication and the
corrected cloud retry remain open.

The selected synchronization commit is pushed on
`codex/carbonteq-verifiers-latest`. Its complete v1 suite passes, with only
credential-dependent Prime cases skipped. Verifiers is consumed directly by
immutable Git revision rather than as a private-index wheel; the Posttrain
manifests and runtime locks therefore constitute its development selection.

Prime-RL demonstrates the intended asynchronous ownership boundary. Verifiers
executes environments and carries the neutral episode `PolicySpan`; the
training orchestrator stamps that span and owns scheduling barriers, policy
publication, staleness admission, group advantages, and learner coordination.
Posttrain follows the same split for its TRL asynchronous prototype rather than
putting trainer state or update policy inside Verifiers.

Selected commit `1f6793f7d46e8a650a54b2a585193b4010578fa6` additionally lets a native
`TrainClientConfig` carry the exact selected chat template and fences the
shared renderer cache by that template. This is required for LFM because the
framework's versioned package template intentionally differs from the model
artifact's bundled template when serializing historical structured tool calls.
The fork config/cache tests and a framework worker-versus-direct LFM token and
attribution parity test pass. The workspace manifests, base catalog, and
candidate control-runtime inputs select this immutable Git revision. The last
published veRL backend image retains its prior immutable dependency closure
until a replacement image passes publication gates. Full native episode
execution, multi-turn continuation, and immutable image qualification remain
release gates.

The same selected commit lets `EnvClient.run` accept a caller-owned request ID
and exposes an acknowledged `EnvClient.cancel`. Posttrain uses an opaque digest
of the full logical episode identity for that wire ID. This closes the previous
fire-and-forget cancellation gap: a failed or timed-out acknowledgment poisons
the collection instead of permitting a weight update with uncertain live work.
The real ZMQ request/response contract and deterministic fixed-pool adapter
lifecycle tests pass; real spawned environment-worker qualification remains
open.

The selected commit also makes a served task lossless across the native worker
boundary: `EnvClient.run` carries the task's per-instance validated config as
well as its data. This is required for tasksets such as AutomationBench that
derive a row-specific concrete-tool allowlist during loading. The server still
accepts legacy data-only requests and falls back to the static catalog config.
Its training client additionally registers a safe `lfm2` parser for the
model's special-token-delimited Python call list; parsing uses literal values
only and never executes sampled code. Posttrain selects this parser from the
versioned LFM conversation contract. The LFM tool-cycle bridge preserves the
exact sampled prefix when an inference engine strips the stop token, then adds
only the protocol close/newline scaffold and new tool observations. This keeps
the second assistant turn on one policy-token branch. Deterministic fork and
consumer tests cover task-config preservation, legacy fallback, exact LFM
rendering, structured tool-call recovery, a successful real AutomationBench
tool execution, and the resulting two-turn linear token history. A real GPU
canary with tool execution remains the next qualification gate.

CarbonTeq Verifiers is not maintained as a temporary patch awaiting upstream
acceptance. It is the supported environment, harness, episode, trace and scorer
runtime for Posttrain. Prime Intellect Verifiers remains an upstream source to
review and synchronize deliberately; upstreaming a CarbonTeq capability is
optional and is never a release gate. Every synchronization must preserve a
reviewed delta ledger, pass fork and consumer compatibility suites, publish an
immutable CarbonTeq revision, and update Posttrain pins only after qualification.

The published external environment revision is
`carbonteq-ai/verifiers-environments@b14dfe0ba9d60184f36d78786a543242fabfb765`
(development branch `codex/verifiers-latest-support`). It migrates
`Task.toolsets(config)` and `Toolset.register` and includes
an optional native AutomationBench episode-judge plugin. The plugin uses a
supplied hosted endpoint, retains bounded attempts and scorer identity, and
emits seven independent whole-episode components without adding quality to the
benchmark's scalar reward. Version 0.4 has one rubric and one wire schema; the
former selectable turn/prefix contract was removed after exact production
request replay showed its turn prompt could be paired with the episode schema.
Twenty-one AutomationBench tests and its clean Pyright/Ruff checks pass on the
latest Verifiers contract. The real framework tool integration also passes.
Mixed-version replay and installed GPU runtime release gates remain open.

- Via **`packages/eval` `verifiers` extra**:
  `uv sync --package eval --extra verifiers --python 3.12`
- Also pulls **`prime`** CLI (Environments Hub: `prime env install` / `prime eval run`)
- Workspace pins the maintained fork commit above; its API remains under
  `verifiers.v1` and defaults to upstream client resolution when no host factory
  is supplied.

Latest Verifiers moves harness ownership beneath `agent` in `EnvConfig`
(`agent.harness`, `agent.runtime`, `agent.timeout`, and `agent.max_turns`) and
uses MCP 2. Posttrain catalogs and Observatory's MCP host have been migrated to
those contracts. Native replay serialization explicitly disables the library's
four-decimal display rounding so retained policy log probabilities remain exact.

When advancing Verifiers: review upstream and CarbonTeq divergence by behavior,
pin a reviewed CarbonTeq commit, refresh `uv.lock`, run taskset and Posttrain
train/eval integration smokes, and record the new SHA here. No unpinned moving
branch and no assumption that upstream is the product authority.

CarbonTeq's maintained Verifiers v1 environment packs live in the separate
framework-neutral [verifiers-environments repository](https://github.com/carbonteq-ai/verifiers-environments).
The current framework integration uses published commit
`b14dfe0ba9d60184f36d78786a543242fabfb765` and keeps each package independently
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

### Native hosted judges (unpublished migration candidate)

`posttrain.jobs.bind_native_judges(context, environment, requests)` binds a
mapping of existing native judge names to `ServeLaunchRequest` selections. The
context manager starts endpoints and closes every started endpoint on normal
exit, training failure, or later endpoint startup failure. It does not interpret
rubrics, aggregate scores or load a model inside an environment plugin.

The declarative native location is `taskset.task.judges`. Each managed entry has
a unique `name` (or plugin `id`), a model identity and the exact sampling mapping
from its inference selection. Mismatches fail before serving. Explicit distinct
ports and capacity are supplied by the composition host. Authentication is passed
through a temporary environment-variable reference, never a stored key value.
The environment's recovery identity retains model/renderer, engine, sampling and
target selections; changing a judge behind an unchanged environment package
revision cannot silently resume a checkpoint.

For standard job composition, `structured_rl_definition("gdpo",
judge_inference_seats={"quality": ("judge_inference", 8123)})` adds a required
inference seat without rubric-specific fields. `sampo_definition(turn_rewards=True,
judge_inference_seats=...)` also requires an explicit `reward_projection` with
`turn_reward_key` and a declaration of terminal-outcome inclusion. Existing
sparse SAMPO definitions retain their selection contract.

The external AutomationBench example plugin emits named turn scores and error
turn IDs. `prefix` assessment sees context only through the rated turn;
`retrospective` sees the full trajectory. Both are versioned scorer settings.
Missing, invalid, timed-out, abstained and inapplicable assessments are not zero
rewards. The example deliberately has no mutable score cache: reassessment uses
a new annotation namespace/trace, while checkpoints use immutable reward-contract
validation. Native Episode JSONL is the replay authority; derived trace JSONL
keeps exact sampled tokens/logprobs and excludes observation tokens from credit.

The live migration exposed nested native-validator mutation; activation now
passes a detached JSON tree so the saved selection remains serializable and
stable across repeated activation. The five-step installed-package runner is
`scripts/qualification/automationbench_native_judges.py`. See the active plan
for failed attempts, candidate provenance, calibration and publication gates;
this section does not declare the new runtime release-qualified.

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
