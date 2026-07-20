# Thread handoff

Handoff for work in `/home/hammad/projects/rl`.

Read first: [post-training lifecycle](./functional/finetuning-lifecycle.md), [target architecture](./architecture.md), [profiles and model variants](./architecture/profiles-and-model-variants.md), and [ADR 0004](./decisions/0004-lifecycle-driven-mvp-platform.md).

## Current state (2026-07-20)

The repository is a clean architecture foundation rather than a compatibility-preserving migration:

```text
packages/common    shared profile/run contracts
packages/train     empty TRL engine boundary
packages/eval      executable Verifiers v1 evaluation boundary
packages/serve     executable vLLM benchmark and SGLang boundary
packages/reports   read-only Trackio query boundary
profiles/          legacy YAML model, train, eval, and serve starting points
jobs/              placeholder for code-based job definitions
```

This is the implementation state, not the final contract. The canonical target
is now code-first: `train`, `eval`, and `serve` become reusable public packages
for this lab and other projects; framework runners/adapters remain internal.
Model and engine profiles become typed Python definitions, and jobs expose
ordinary Python actions that invoke those package operations. Trackio is purely
observability and never defines or executes those objects.

Removed intentionally:

- prototype `apps/` package layout;
- local model catalog and executable `configs/` tree;
- task-specific callback package;
- normalized `evaluations/` result store and LightEval orchestration;
- separate inference-optimization directory;
- old train/eval/serve CLIs and their schema-bound tests.

The removed files were placed in the desktop trash and are recoverable, but they are not part of the target design.

## Implemented foundation

- uv workspace membership is `packages/*` only;
- all workspace packages have independent manifests;
- heavy vLLM, SGLang, and Verifiers variants are optional package extras;
- `profiles/{models,train,eval,serve}` and `jobs/` are present;
- `common.profiles.ProfileResolver` supports one-parent inheritance, deep mapping merge, cycle detection, and model-profile validation;
- `profile-resolve` exposes resolution and validation through the CLI.
- `common.tracking.TrackedRun` records typed Trackio runs plus local recovery bundles;
- the workspace pins the CarbonTeq Trackio fork by immutable commit; its
  additive `VerifiersTrace` type stores full native records with queryable
  transcript/metadata projections and retry-safe external identities;
- `packages/eval` tails completed Verifiers JSONL records during execution,
  validates and synchronizes them in small batches, retries during finalization,
  and records partial synchronization without invalidating successful evals;
- `serve-benchmark` runs a profile-resolved offline vLLM benchmark;
- the CUDA 13.0 runtime/compiler chain is pinned and the serve package creates a validated standard toolkit view for FlashInfer JIT;
- `trackio-query` provides read-only raw SQL access for future reports;
- official pinned profiles exist for Qwen3.5-2B and LFM2.5-1.2B-Thinking.
- `benchmarks/inference` contains a 96-cell controlled suite over four workload
  shapes, six context windows (through 32K), and concurrency 1/2/4/8;
- `serve-benchmark-suite` plans or executes one typed Trackio run per matrix
  cell, with this machine explicitly limited to concurrency 1/2/4;
- model profiles declare tokenizer-native reasoning controls, and representative
  prompts remain canonical message data;
- 32K cells automatically use TurboQuant K8V4 through model-specific serve
  variants;
- the serve package registers a guarded vLLM general plugin for the vLLM 0.25.1
  hybrid TurboQuant quantization-marker defect so spawned workers are repaired.
- model profiles now keep immutable weight/capability facts separate from
  backend-native serving settings and task-shaped evaluation budgets;
- general eval resolves a 32K TurboQuant endpoint profile, forwards the total
  context limit to Verifiers, and logs call/rollout truncation rates;
- Qwen has explicit standard, TurboQuant K8V4, and native-MTP vLLM variants.

## Validated GPU slice

Trackio run `lfm2.5-1.2b-vllm-smoke-8` completed on the RTX 3070 Ti. It used
vLLM 0.25.1, PyTorch 2.11.0+cu130, FlashAttention 2, FlashInfer sampling,
compiled graphs, chunked prefill, and asynchronous scheduling. The measured
one-request smoke result was 177.46 output tokens/s with 6.66 GiB peak device
memory. Treat it as path validation, not as a concurrency or production
capacity result.

Three controlled cells also completed: 32K/c1 short interactive with
TurboQuant (172.67 output tok/s), 4K/c4 short interactive (591.98 output tok/s),
and a real 24,576-token prefill at 32K/c1 with TurboQuant (1.83 s mean TTFT,
8,616.93 input tok/s, 44.88 output tok/s). The reusable suite still includes c8,
but it must be run on the larger machine.

Trackio contains both output artifacts:

- `lfm2.5-1.2b-vllm-smoke-8-serving-benchmark`;
- `lfm2.5-1.2b-vllm-smoke-8-run-bundle`.

## Commands

```bash
cd /home/hammad/projects/rl
uv sync --all-packages --python 3.12
uv run --package common profile-resolve --help
uv run --package common python -m unittest discover -s tests
uv run --package reports trackio-query lab \
  "SELECT run_id, run_name, config FROM configs ORDER BY id DESC LIMIT 10"
```

Optional engine variants:

```bash
uv sync --package serve --extra vllm --python 3.12
uv sync --package serve --extra sglang --python 3.12
uv sync --package eval --extra verifiers --python 3.12
uv sync --package train --extra vllm --python 3.12
uv run --package serve --extra vllm serve-benchmark lfm2.5-1.2b-thinking
uv run --package serve --extra vllm \
  serve-benchmark-suite lfm2.5-1.2b-thinking --dry-run
uv run --package eval --extra verifiers \
  eval-suite lfm2.5-1.2b-thinking --dry-run
```

## Next vertical slice

1. Define stable, directly usable operation/result APIs for `train`, `eval`, and
   `serve`, plus the optional observation-context protocol.
2. Move vLLM definitions behind `serve.benchmark`, prove it from a standalone
   script, and then consume the same operation from a code-defined onboarding job.
3. Replace the durable local run bundle with a temporary workspace while
   preserving Trackio as the sole evidence store.
4. Emit standard request traces and add stable report queries before running
   the matched Qwen/LFM comparison.
5. Migrate Verifiers behind `eval.evaluate`, then add SFT, DPO, and GRPO behind
   the public `train` API with internal TRL adapters.

Do not reintroduce the deleted catalog/result/task abstractions as compatibility layers.

## Revision history

- 2026-07-20: Recorded the code-first target, package-owned profiles, typed job
  actions/requests, and pure Trackio observability boundary while preserving the
  current implementation inventory above.
- 2026-07-20: Clarified that train/eval/serve packages—not framework runners—are
  the reusable units across projects.
