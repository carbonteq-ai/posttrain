# Optimizing GRPO on a single GPU

Status: operational guidance with a partially completed 8 GiB qualification
Last revised: 2026-07-23

This guide explains how to make Group Relative Policy Optimization (GRPO)
practical on one GPU without silently turning it into a different experiment.
It covers workload definition, phase-aware memory budgeting, rollout and
training controls, failure diagnosis, measurements, and release gates for the
framework's TRL and veRL backends.

The current 8 GiB work proves that a Qwen 3.5 0.8B, tool-using GRPO workload can
cross the large-vocabulary scoring boundary after chunked projection and fused
loss are enabled. It does **not** yet prove that the selected three-update
workload completes through both TRL and veRL. Keep that distinction visible in
plans, dashboards, and handoffs.

## What is being optimized

A GRPO optimizer update is not one forward and backward pass. For each prompt,
the policy generates a group of completions, the environment scores those
completions, the trainer computes group-relative advantages, and the actor
replays the sampled tokens through one or more scoring and backward
microbatches before stepping the optimizer.

For the current matched study:

- two prompt groups are selected per optimizer update;
- eight generations are sampled per prompt;
- each optimizer update therefore produces 16 trajectories;
- physical actor batch size is one;
- gradient accumulation has 16 slices;
- three optimizer updates would produce 48 trajectories if all updates
  complete.

Reducing the number of generations, completion limit, task complexity, or
optimizer updates changes the workload. Those controls are valid for creating a
smaller profile, but they are not evidence that an unchanged workload was
optimized.

Use these terms consistently:

- **Algorithm batch**: all prompt-generation samples that define one GRPO
  update.
- **Physical microbatch**: samples resident during one actor scoring/backward
  slice.
- **Gradient accumulation**: the number of physical slices whose gradients are
  combined before the optimizer step.
- **Rollout phase**: generation and environment interaction.
- **Correction/scoring phase**: recomputation of token log-probabilities,
  entropies, reference values, and importance ratios.
- **Backward phase**: differentiable GRPO loss and gradient construction.
- **Optimizer update**: the parameter mutation after all required slices.

## The single-GPU memory model

Peak memory is controlled by the largest phase, not by the sum of every
configured capacity. A colocated engine can release or CPU-back rollout
allocations before backward, but only if its lifecycle is explicit and
validated.

The important allocations are:

| Allocation | Typical lifetime | Main controls |
| --- | --- | --- |
| Actor weights and adapters | Whole process | model size, precision, LoRA/QLoRA |
| Optimizer states and gradients | Training phases | update method, trainable modules, optimizer |
| vLLM rollout representation | Rollout, possibly CPU-backed during sleep | weight-sync mode, sleep level |
| KV cache | Rollout | cache bytes, cache dtype, context, concurrency |
| CUDA graph private pools | Process lifetime after capture | eager execution |
| Hidden states and activations | Scoring/backward | sequence length, microbatching, checkpointing |
| Vocabulary logits | Scoring/backward | vocabulary size, token count, projection chunking, fused loss |
| Environment state and traces | Host memory, with serialization pressure | concurrency, trace size, retention |

For a dense LM head, a full logits tensor grows approximately with:

    token positions × vocabulary size × logits bytes

This term can dominate even for a sub-billion-parameter model with a large
vocabulary. Reducing actor microbatch size does not help if one remaining
sequence still projects all token positions into the full vocabulary at once.

### Phase sharing must be observed

Do not assume that `sleep_during_optimization` frees enough memory merely
because it is selected. Record memory at these boundaries:

1. actor loaded;
2. rollout engine initialized;
3. KV cache allocated;
4. generation peak;
5. rollout engine asleep and cache released;
6. old-policy/reference scoring peak;
7. differentiable loss peak;
8. backward peak;
9. optimizer step;
10. rollout engine awake after synchronization.

The phase labels matter. A single global peak says that a run was tight but not
which control should change.

## Preserve the experiment before tuning

Write a comparison manifest before changing memory or performance controls.
The following values must match between baseline and optimized runs unless the
comparison explicitly names the difference:

- model identity, immutable revision, precision, and renderer;
- environment package and revision;
- environment category, sampling seed, and bounded task policy;
- prompt groups per update and generations per group;
- maximum prompt and completion lengths;
- sampling temperature and top-p;
- GRPO beta, loss type, clipping, reward scaling, and importance-sampling
  settings;
- update method, LoRA rank, alpha, and target modules;
- optimizer, learning rate, scheduler, and optimizer-update count;
- reference-model behavior;
- MTP configuration and KV-cache dtype;
- software revisions and dependency-lock digest.

Backend, physical microbatch schedule, projection chunk size, fused-kernel
selection, and memory-offload controls may differ in an optimization study.
State those differences in the run configuration and comparison view.

Use an optimizer-update limit, not raw dataloader iteration or rollout count, as
the stopping contract. Verify the backend translation with a test because TRL
and veRL use different native batch and loop concepts.

## Optimization order

Apply controls in the following order. Each step preserves more of the
experiment than the steps below it and makes failures easier to interpret.

### 1. Make the run reproducible

Resolve every catalog reference before launch and record the immutable result:
model, environment, training binding, inference binding, job definition,
backend source revision, and dependency lock. Preserve the generated native
traces and partial logs even when the process fails.

Validate model-specific LoRA targets against the loaded module names before
starting vLLM. A regular expression copied from another wrapper can match
nothing and waste an otherwise expensive launch.

### 2. Separate algorithm batch from physical batch

Keep the GRPO group and global algorithm batch unchanged while lowering the
physical actor microbatch. Increase gradient accumulation so the optimizer sees
the same number of samples.

For the current 16-trajectory update:

    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    global_batch_size: 16

This changes scheduling, not the group-relative reward calculation. Confirm
that all 16 trajectories contribute before the optimizer step and that metrics
are reduced over the algorithm batch rather than reported from the final
microbatch only.

### 3. Bound rollout memory explicitly

On a constrained GPU, use an explicit KV-cache byte budget. Percentage-based
automatic reservation reacts to transient free memory and can consume the
headroom required by later phases.

The budget must still represent at least one maximum-length request. vLLM
reports the minimum cache required when it cannot. Treat that message as a
hard lower bound for the selected model, cache dtype, and context.

Also bound:

- `max_model_len`;
- `max_num_batched_tokens`;
- `max_num_seqs`;
- rollout concurrency;
- prompt and completion limits.

Use eager execution when CUDA graph private pools make the training phase
unviable. Enable sleep/wake and explicit cache release for colocated rollouts.
After wake-up, verify that the synchronized adapter or weights—not a stale
policy version—produce the next trajectories.

### 4. Use parameter-efficient updates deliberately

LoRA usually offers the best initial single-GPU trade-off:

- the base remains fixed;
- gradients and optimizer states cover only adapters;
- adapter-only synchronization avoids copying dense actor weights into vLLM.

Target fewer projections only when the selection is a declared training
profile, not an undocumented emergency change. The current constrained Qwen
profile targets `o_proj` and `down_proj`; that choice must remain matched across
backends when comparing them.

QLoRA reduces the actor base footprint further, but it adds compatibility and
kernel constraints. Qualify the complete lifecycle: quantized actor load,
forward/backward, adapter update, vLLM synchronization, sleep/wake, checkpoint,
and export. Do not assume that a quantized training representation can be
copied into an inference engine as dense weights.

### 5. Chunk old-policy and reference projection

Old-policy and reference scoring do not require one full vocabulary-logits
tensor. The CarbonTeq TRL fork exposes:

    logits_chunk_size: 128

The trainer flattens token positions, projects bounded chunks through the same
LM head, and reconstructs token-aligned log-probabilities and entropies. The
chunked and unchunked paths have a numerical-equivalence regression test.

Chunk size is a memory-versus-launch-overhead control:

- smaller chunks lower projection peak memory but invoke the LM head more
  often;
- larger chunks improve arithmetic efficiency but approach the original peak;
- `None` keeps the unchunked path.

Start at 128 token positions for an 8 GiB, large-vocabulary model, measure the
scoring phase, and increase only when headroom and throughput justify it.

### 6. Bound the differentiable loss

Projection chunking solves the non-differentiable old/reference scoring
allocation. The train loss can still recreate a full differentiable logits
tensor. For the current TRL profile:

    use_liger_kernel: true

Liger's fused GRPO loss keeps the differentiable projection and loss
memory-bounded. Treat numerical equivalence, gradient finiteness, and actual
parameter change as release gates. Kernel enablement alone is not proof of an
optimizer update.

### 7. Add activation and optimizer controls if needed

If the run still fails after vocabulary projection is bounded:

- enable gradient checkpointing;
- verify that attention uses a memory-efficient supported implementation;
- offload optimizer state before offloading frequently accessed parameters;
- reduce the trainable-module set as an explicit profile;
- use QLoRA only after its lifecycle is qualified.

Measure the throughput penalty. CPU or NVMe offload can change an OOM into a
run dominated by transfers.

### 8. Evaluate MTP as an accelerator

Multi-token prediction (MTP) uses the model's native draft head during rollout.
It does not add an auxiliary MTP training objective.

Always record:

- configured speculative-token count;
- drafted tokens;
- accepted tokens;
- acceptance rate;
- accepted length;
- rollout tokens per second;
- rollout phase duration;
- memory delta versus a matched non-MTP run.

Acceptance is an intermediate mechanism metric. High acceptance can still lose
end-to-end time through draft overhead, synchronization, or constrained
batching. Compare MTP on/off with the same policy revision, prompts, samples,
context, and cache dtype.

### 9. Treat TurboQuant as a qualified cache mode

TurboQuant changes rollout KV-cache storage, not actor-weight precision. It may
increase capacity or reduce memory, but those gains do not establish output
correctness.

A qualified cache mode needs:

- deterministic short-generation comparison against normal KV;
- recall or retrieval probes at every supported context length;
- matched task-reward and truncation evidence;
- cache capacity and peak-use measurements;
- throughput and memory comparison.

The current Qwen 3.5 K8V4 probe increased cache-token capacity by about 2.67
times but failed beginning-of-context recall at 8K, 16K, 24K, and 32.7K while
normal KV passed. Therefore K8V4 remains configuration-supported but
quality-unqualified for this model. Do not combine it with MTP until each
feature passes independently.

### 10. Change workload bounds last

If the original experiment still cannot fit, create a new versioned profile.
The most effective workload changes are:

- fewer generations per prompt;
- fewer prompt groups per update;
- shorter prompt or completion limits;
- a smaller context window;
- fewer tool turns;
- a smaller model.

Name the new profile and do not compare its wall time directly with the
original as if only the runtime changed.

## Current 8 GiB case study

The current study runs on an RTX 3070 Ti with 7.63 GiB reported CUDA capacity.
Its intended workload is Zapier AutomationBench v1, Qwen 3.5 0.8B, thinking
enabled, native MTP-1, two prompt groups, eight generations per group, and
three optimizer updates.

The active constrained selections are:

| Control | Selected value |
| --- | --- |
| Actor update | BF16 LoRA, rank 8, alpha 16 |
| Trainable modules | `o_proj`, `down_proj` |
| Algorithm batch | 16 trajectories per update |
| Physical actor batch | 1 |
| Gradient accumulation | 16 |
| Maximum prompt | 2,048 tokens |
| Maximum completion | 6,144 tokens |
| Engine context | 8,192 tokens |
| KV-cache budget | 201,326,592 bytes (192 MiB) |
| vLLM execution | eager, sleep during optimization |
| Speculation | native MTP, one draft token |
| Old/reference projection | 128 token positions per chunk |
| Differentiable loss | Liger fused GRPO loss |
| Tracking evidence | Trackio plus native Verifiers JSONL |

These AutomationBench values are project-owned experiment policy. The reusable
TRL fork owns only the generic projection control and runtime behavior.

### Failure ladder

| Attempt | Observation | Conclusion |
| --- | --- | --- |
| Initial launch | LoRA target expression matched no modules | Validate targets against the loaded model before engine startup |
| Generation translation | TRL generation batch did not match the declared algorithm batch | Translate prompts × generations explicitly and test it |
| Automatic cache | vLLM attempted a 496 MiB allocation with only 301 MiB free | Percentage reservation consumed training headroom |
| 64 MiB cache | vLLM required about 0.16 GiB for one 8,192-token request and estimated only 1,088 tokens | A cache cap below one declared request is invalid |
| 192 MiB cache, full scoring batch | Scoring requested another 1.43 GiB after 6m29s | Physical batching alone did not bound projection |
| Batch-one scoring | One vocabulary projection requested 3.54 GiB after 9m20s | The `[tokens, vocabulary]` tensor was the dominant allocation |
| Chunked projection plus fused loss | Sixteen real traces completed and the run crossed the prior scoring boundary | Both non-differentiable scoring and differentiable loss needed bounded paths |

The optimized attempt created Trackio run `train.grpo-494bbf38` and preserved
16 native AutomationBench traces in a 1,786,242-byte JSONL file. All 16 traces
completed without recorded native errors; one received positive partial
credit. Host telemetry reached full GPU utilization with roughly 1.9 GiB
headroom during the phase that previously failed.

The process was interrupted before its first completed optimizer update.
Therefore the evidence establishes memory-boundary progress, not a successful
three-update training benchmark. There is no valid TRL-versus-veRL speed
conclusion from this attempt.

## Backend-specific considerations

### TRL

TRL runs the actor and colocated vLLM inside one process. This makes phase
sharing and adapter synchronization direct, but CUDA allocator state and
vLLM's process-lifetime pools affect later training phases.

The framework currently pins CarbonTeq TRL commit
`5c50c69f2d9b25dc2ce729d030f7cabb144d8431`. The relevant generic controls are:

- native MTP and per-generation speculative counters;
- guarded vLLM engine options and cache dtype;
- adapter-only synchronization for LoRA/QLoRA;
- compatible rollout sleep/wake behavior;
- bounded log-probability and entropy projection through
  `logits_chunk_size`.

### veRL

veRL uses an isolated runtime and Ray-managed workers. Compare its actual
placement, FSDP mode, offload settings, rollout engine, and synchronization
schedule with TRL rather than assuming equivalent topology from one visible
GPU.

The veRL adapter must preserve its structured metric sidecar and replay it into
the framework observer. A single-GPU profile should make these values explicit:

- actor, rollout, and reference placement;
- parameter and optimizer offload;
- physical microbatch and dynamic batching;
- response-length and token-budget controls;
- rollout-engine cache budget;
- sleep/wake and weight synchronization;
- Ray startup and teardown time.

Do not call veRL slower or faster until both backends complete the same optimizer
updates with comparable realized completion tokens and valid traces. Startup,
rollout, scoring/backward, checkpoint, and total wall time should be reported
separately.

## Measurements required for every optimization run

### Correctness and learning

- optimizer updates completed;
- reward mean and standard deviation;
- fraction of groups with zero reward variance;
- policy loss;
- gradient norm and non-finite-gradient count;
- KL when selected;
- clip fraction and importance ratios when selected;
- proof that trainable parameters changed;
- checkpoint and adapter integrity.

### Rollout population

- prompt groups attempted;
- generations attempted, completed, failed, truncated, and unscorable;
- completion tokens;
- turns and tool calls for agent environments;
- environment reward components;
- exact policy version associated with each trace.

### Performance

- process startup;
- model and rollout-engine initialization;
- rollout model time and environment-harness time;
- old/reference scoring;
- differentiable loss and backward;
- optimizer step;
- checkpoint and artifact synchronization;
- total wall time;
- completion tokens per second and effective update tokens per second.

### Memory and runtime

- peak allocated and reserved GPU memory by phase;
- free GPU memory at each phase transition;
- KV-cache capacity and peak use;
- host memory and offload transfer volume;
- GPU utilization;
- CUDA graph private-pool memory;
- MTP drafted and accepted tokens when enabled.

Normalize metrics at logical optimizer step. Keep high-cardinality task,
transcript, tool-call, and reward-component data in native traces rather than
flattening it into metric keys.

## Benchmark procedure

1. Record the immutable comparison manifest.
2. Start from a clean GPU and record idle memory.
3. Execute a one-update diagnostic to validate construction, rollout, reward,
   scoring, backward, optimizer step, checkpoint, and finalization.
4. Confirm that expected trajectories, native traces, metrics, and artifacts
   agree.
5. Execute three optimizer updates.
6. Repeat with the second backend under the same logical workload.
7. Repeat the candidate optimization at least once to distinguish warm-cache
   or compilation effects.
8. Compare medians and phase breakdowns; do not use one total duration without
   realized token counts.
9. Inspect at least one successful, failed, and truncated trajectory when those
   populations exist.
10. Preserve the run manifest, logs, metrics, traces, checkpoints, environment
    revision, and backend revision together.

For stochastic generation, exact text need not match across backends. The
configuration, task-selection policy, seed, prompt population, generation
count, and bounds must match. Report realized completion tokens, truncation,
reward population, and errors so the remaining stochastic difference is
visible.

## Failure playbook

### Failure before model construction

Check catalog resolution, immutable source revisions, environment import,
renderer compatibility, and LoRA target matches. Do not start GPU allocation
until these preflight checks pass.

### OOM while initializing vLLM

Check the reported minimum KV cache for one maximum-length request. Use an
explicit cache budget, lower concurrency, enable eager mode, and verify that
the selected context is intentional.

### OOM after rollout sleep

Measure allocations after cache release. Persistent CUDA graph pools or a
non-released inference representation may be the cause. Verify sleep level,
free-cache behavior, and adapter synchronization.

### OOM during old/reference scoring

If the allocation scales with vocabulary size, lower
`logits_chunk_size`. If it scales with activation length, lower physical
microbatch, enable checkpointing, or use sequence-aware batching.

### OOM during differentiable loss or backward

Enable the fused GRPO loss, confirm the LM head is handled as intended, reduce
the physical microbatch, and then consider checkpointing or optimizer offload.

### Run completes trajectories but no optimizer step

Treat it as an interrupted or partial run. Check reward-group variance,
gradient creation, accumulation count, callback step semantics, and stop
conditions. Trace completion is not optimizer completion.

### High MTP acceptance but slower rollout

Compare drafted-token overhead, batch size, realized completion length,
environment time, and synchronization. Remove MTP if end-to-end rollout
throughput does not improve.

### More TurboQuant capacity but worse behavior

Fail the quality gate. Capacity and speed do not override deterministic
generation or long-context recall regressions.

## Recovery and idempotence

Every attempt uses a new run id and output directory. Never overwrite a failed
attempt merely to make a dashboard look successful.

On interruption, retain:

- resolved configuration and dependency digests;
- stdout/stderr and phase timings;
- native Verifiers traces;
- native backend metric sidecars;
- partial checkpoints;
- allocator/OOM diagnostics;
- tracking run identity and final partial/interrupted outcome.

Resume from a checkpoint only when the checkpoint records the actor revision,
optimizer and scheduler state, completed logical step, environment selection,
and compatible backend revision. Otherwise start a new run and link it as a
retry.

## Release gates

A single-GPU profile is **qualified** only when:

- the exact selected workload completes its declared optimizer updates;
- every update has a valid rollout population and group denominator;
- at least one update demonstrates a non-zero, finite parameter change unless
  zero gradient is explained by measured zero reward variance;
- adapter, checkpoint, summary, and native traces are preserved;
- peak memory has positive safety headroom in every phase;
- a retry produces consistent phase behavior;
- Observatory can read the same logical evidence from the selected tracking
  backend.

The current 8 GiB AutomationBench profile has crossed the scoring-memory gate
but remains **candidate** because:

- the optimized TRL attempt stopped before optimizer update one;
- three TRL updates have not completed;
- the matched three-update veRL run has not completed;
- no valid matched runtime comparison exists;
- TurboQuant K8V4 remains quality-unqualified for Qwen 3.5.

## Source locations

- Project workload:
  `.posttrain/work_packages/automationbench_zapier_grpo.yaml`
- GRPO and runtime selections:
  `packages/catalog/src/posttrain/catalog/base/training.yaml`,
  `packages/catalog/src/posttrain/catalog/base/inference.yaml`, and
  `packages/catalog/src/posttrain/catalog/base/environments.yaml`
- TRL adapter:
  `packages/train/src/posttrain/train/backends/trl/grpo.py`
- veRL adapter:
  `packages/train/src/posttrain/train/backends/verl/`
- Research and Observatory execution plan:
  `docs/plan/grpo-research-observability.md`
- Framework TRL selection and qualification:
  `docs/tooling/trl/README.md`
- TRL fork delta:
  `../trl/CARBONTEQ_FORK.md`
- Current attempt evidence:
  `artifacts/grpo-benchmark-20260723/`
