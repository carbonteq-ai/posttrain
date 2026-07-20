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

Complex Verifiers environments may require a different internal training
integration than a simple TRL reward callback. The public `train.rl` operation
and its result remain the reuse boundary.

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
