# Training, evaluation, and serving package boundaries

Status: target MVP architecture  
Last revised: 2026-07-20

## Purpose

`posttrain.train`, `posttrain.eval`, and `posttrain.serve` are reusable Python packages for many kinds of
projects. This lab's jobs are consumers of those packages; their internal
framework runners are not the shared architecture.

## Public package contract

Each package exposes domain operations, typed configuration, typed results, and
optional instrumentation hooks:

| Package | Stable public operations | Initial internal integration |
| --- | --- | --- |
| `posttrain.train` | `sft`, `dpo`, `grpo`, checkpoint selection/resume | TRL |
| `posttrain.eval` | `evaluate`, program execution, checkpoint comparison inputs | Verifiers |
| `posttrain.serve` | `launch`, `generate`, `probe`, `benchmark` | vLLM |

The public operation is stable across implementations. The concrete config may
remain framework-specific when the framework exposes genuinely different
behavior.

```python
result = train.sft(model=base, data=dataset, config=TRL_SFT_QLORA)
report = model_eval.evaluate(model=result.model, program=GENERAL_SMOKE)
bench = serve.benchmark(model=result.model, profile=VLLM_BASELINE)
```

These calls can be made by another project without constructing a lab Job or a
Trackio run.

## Execution context

Operations accept an optional host-supplied context with narrow capabilities:

- cancellation and temporary workspace;
- progress/events;
- metric and system-telemetry emission;
- trace emission;
- artifact input/output callbacks.

The package remains usable with a local/no-op context. The lab supplies a
Trackio-backed context and adds job/action/invocation provenance around the
operation. Trackio is therefore an integration choice of this platform rather
than a mandatory dependency of the reusable package contract.

## Internal adapters

Each package may define private protocols for testing and framework replacement:

```text
posttrain.train.sft -> TRL adapter or future trainer adapter
posttrain.eval.evaluate -> Verifiers adapter
posttrain.serve.benchmark -> vLLM adapter
```

Those protocols organize package internals. They are not the objects job or
project authors are expected to reuse directly, version independently, or wire
together.

## Replacing TRL

A new training framework is added inside `packages/train`:

1. add its typed config and internal adapter;
2. map the public operation lifecycle and typed result;
3. map checkpoints, artifacts, metrics, and failures;
4. add capability and compatibility tests;
5. publish a new `posttrain-train` package version.

Consumers continue calling `train.sft`, `train.dpo`, or `train.rl`. They may
choose the new concrete config, but do not replace the reusable package with a
runner object.

## Training operations

A training operation consumes an exact model/profile, data or environment
inputs, typed technique config, and recovery/output policy. It returns a typed
result containing selected outputs and checkpoint/recovery information.

| Operation | Package-owned behavior | Caller supplies |
| --- | --- | --- |
| SFT | validation, execution lifecycle, PEFT/checkpoint semantics | model, data, typed config |
| DPO | preference validation, execution lifecycle, result semantics | model, preference data, typed config |
| RL | supported environment/rollout lifecycle, result semantics | model, environment, typed config |

Environment-driven RL uses the same native Verifiers episode model as
evaluation. The public `train.grpo` operation and its result remain the reuse
boundary; neither a TRL callback nor a Verifiers runner is exposed as the
cross-project API.

### GRPO rollout and Verifiers bridge

GRPO keeps policy generation, environment execution, trainer translation, and
observation separate:

```text
Verifiers task + harness + runtime
  -> OnlineRLBridge.run(batch, policy_generator)
  -> native Verifiers episode
       -> policy client requests one or more model turns
       -> PolicyGenerator uses the already-loaded TRL policy
       -> tools, user simulation, stops, finalization, and scoring stay native
  -> final Verifiers trace branch
  -> prompt IDs + completion IDs + sampling logprobs + env mask + reward
  -> private TRL rollout adapter
  -> GRPO loss and optimizer step

same trace
  -> ExecutionContext.trace(...)
  -> injected observer
  -> Trackio queryable trace copy

native traces.jsonl
  -> required evaluation-traces artifact
```

`GRPORequest` accepts one backend-neutral `OnlineRLBridge`. The bridge owns a
task-neutral rollout dataset, native episode execution, scored-trajectory
projection, and native evidence finalization. It receives a `PolicyGenerator`
from the selected trainer adapter rather than constructing or loading a model.
This inversion lets Verifiers drive as many model, tool, and user turns as the
native environment requires while the trainer retains sole ownership of policy
weights and optimization.

The reusable contracts live in `posttrain.train.online_rl`:

| Contract | Owns | Does not own |
| --- | --- | --- |
| `PolicySampling` | max output tokens, temperature, top-p shared by environment and generator | backend-only engine tuning |
| `PolicyTurnRequest` / `PolicyTurnResult` | messages, tools, exact token IDs, token attribution, logprobs, finish reason | environment state, rewards, trainer tensors |
| `PolicyGenerator` | one model turn against an already-loaded policy | task selection, runtime, scoring, observation |
| `RolloutBatch` | trainer step, model identity, aligned stable example IDs | model-visible prompt metadata |
| `TrainingRollout` | exact trainer sequence, sampled-token mask, reward, truncation, trace observation | framework-specific tensors |
| `OnlineRLBridge` | native episodes and trajectory projection | TRL, vLLM, Trackio |

`posttrain.train.integrations.verifiers.VerifiersOnlineRLBridge` implements the
bridge with native Verifiers `Environment.episode(...)`. Its policy client maps
native model turns onto `PolicyGenerator`, then derives the training sequence
from the final trace branch. Tokens produced by the model are `True` in
`env_mask`; tool, harness, simulator, and template tokens remain in the
sequence for context but are masked from policy loss. Native termination and
error state is authoritative instead of being guessed from the final token.

`posttrain.train.backends.trl.online_rl.TrlPolicyGenerator` implements policy
generation with the trainer's already-loaded Transformers or colocated-vLLM
representation and the model-family renderer. The private TRL rollout adapter
converts `TrainingRollout` values into `rollout_func` output. The pinned TRL fork
passes aligned raw dataset rows into that function, preserving stable task
identity without adding hidden fields to model-visible prompts.

An environment package owns task loading, tools, runtime requirements, rewards,
and optional native reward enrichment. It imports neither TRL nor Trackio. The
lab job composes that package with `VerifiersOnlineRLBridge`, a training
profile, and an observer-backed execution context.

The MVP supports linear single-branch chat trajectories, including multiple
model turns and tools. A native trace with zero or multiple terminal branches is
rejected explicitly because one trainer example cannot silently choose among
several trainable branches. Multimodal sidecars and trainer-side parallel
episode scheduling remain separate extensions to the same contracts.

The GRPO rollout profile explicitly selects one of two execution modes:

- Transformers generation uses the autograd model directly and therefore has
  one policy-weight representation. It is the constrained-hardware fallback.
- colocated vLLM creates an optimized inference representation inside the TRL
  process. For QLoRA, it retains the quantized base and reloads only the active
  LoRA adapter before rollout. Sleep level 1 moves the immutable quantized base
  to CPU and discards the KV cache before optimization, then restores the base
  without an unsupported bitsandbytes checkpoint reload. This is not literally
  a single representation, but it prevents a separate Verifiers model and
  bounds concurrent GPU residency.

Text-only training does not imply that the Hub checkpoint has a text-only
top-level namespace. Qwen3.5 is distributed as a composite checkpoint, so
vLLM retains its native composite loader while `language_model_only` omits the
vision tower without changing the text execution path.

Full-weight synchronization is unsafe for a QLoRA policy: a merged
`Linear4bit` parameter still exposes packed quantized storage, not the dense
weight expected by vLLM's ordinary update loader. The rollout profile therefore
selects native LoRA synchronization. TRL writes the current adapter in PEFT's
standard format and vLLM reloads the same dynamic adapter ID in place. The
generic full-weight namespace option remains available for non-quantized model
layouts. Jobs and Verifiers environments are unaware of either mechanism.

Every TRL adapter runs inside the same trainer lifecycle boundary. After model
publication and recovery-checkpoint discovery—or after an exception—the
boundary calls Accelerate's `end_training()` so trackers and distributed
process groups are closed explicitly rather than left to interpreter shutdown.

For compatible Qwen profiles, colocated vLLM may additionally enable native MTP.
Engine-level speculative configuration is part of the rollout profile and is
passed through the pinned TRL fork. Verifiers always scores the completions from
that selected rollout engine; it never starts a second inference server.

## Evaluation operations

`eval.evaluate` consumes a model/endpoint, a reusable program or environment
reference, and typed execution settings. `packages/eval` owns environment
loading, execution lifecycle, trace/native-bundle emission, and public result
semantics. The environment owns task behavior and scoring.

General and domain evaluation reuse the same package operation. They differ in
program selection and policy, not in framework or observability architecture.

## Serving operations

`packages/serve` owns public model-serving and measurement semantics while its
adapters own backend translation. Reusable profiles may include:

- dtype and load format;
- context exposed by the endpoint;
- weight and KV-cache quantization;
- memory targets, scheduling, and caching;
- prefill/decode controls;
- MTP or draft-model speculation;
- TurboQuant and custom kernels;
- tensor, pipeline, data, or expert parallelism.

The model profile records native MTP capability. A serve profile records how a
backend uses it. Runtime-only optimization creates new evidence, not a new model
artifact.

## Why vLLM can appear in training

vLLM may support standalone serving, evaluation endpoints, or rollout
generation. Shared compatibility utilities can remain inside or below the
owning packages, but each package presents its own public lifecycle:

- `serve` owns standalone endpoint/benchmark behavior;
- `eval` owns evaluation consumption of an endpoint;
- `train` owns rollout generation used by its operation.

Consumers should not pass one package's full internal adapter object into
another package.

## Inference benchmark data

`posttrain.serve.benchmarks` owns reusable workload inputs with the benchmark
operation that interprets them:

- suites define token shapes, configured context, concurrency, warmup, and
  repetitions;
- corpora contain canonical messages;
- model profiles identify chat templates and reasoning modes;
- `serve.benchmark` renders and executes them through its selected backend;
- the host observer can persist one run per matrix cell and one trace per
  measured request.

The checked-in suite may contain concurrency 1, 2, 4, and 8 while the current
machine selects at most 4. A 32K cell may select a compatible TurboQuant K8
profile as explicit host policy.

Serving benchmarks measure system behavior. Evaluations measure model behavior.
They may share a model artifact and endpoint but remain separate package calls
and observation run kinds.

## Dependency isolation

Public modules stay lightweight; concrete integrations use extras:

```text
posttrain-train[trl]
posttrain-train[trl,trl-vllm]
posttrain-eval[verifiers]
posttrain-serve[vllm]
```

Other projects can install only the packages and implementations they require.

## Trackio integration in this lab

The lab's job runtime wraps a public package operation with a Trackio-backed
context:

1. resolve job/action/invocation and source provenance;
2. start an observable run;
3. call the public `posttrain.train`, `posttrain.eval`, or `posttrain.serve` operation;
4. stream observations through the context;
5. link selected input/output artifacts;
6. finalize the run and return the package's typed result.

The package never asks Trackio what to execute next. Job code and human policy
make that decision.

## MVP sequence

1. Define the public operation/result surface for each reusable package.
2. Define the optional execution/observation context protocol in `common`.
3. Adapt the existing vLLM benchmark behind `serve.benchmark`.
4. Adapt Verifiers execution behind `eval.evaluate`.
5. Add SFT and DPO behind `posttrain.train`; then define the supported GRPO boundary.
6. Prove direct use from a small standalone script with no Job and no Trackio.
7. Prove lab-job use of the same operations with Trackio observation enabled.

## Revision history

- 2026-07-20: Replaced the public runner-centric model with reusable
  `train`/`eval`/`serve` operation APIs, internal framework adapters, and an
  optional host-injected execution/observation context.
- 2026-07-20: Defined code-based jobs, implementation-owned typed config, benchmark workloads, and Trackio observation.
