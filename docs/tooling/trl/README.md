# TRL

TRL is the execution library behind `packages/train`.

The rebuilt `train` package will expose reusable SFT, DPO, and RL operations.
TRL is an internal adapter selected by a typed TRL config, not the object other
projects must compose directly. PEFT/QLoRA, checkpoint behavior, public result
semantics, and instrumentation hooks belong to the `train` package boundary.
The lab's injected observation context maps those hooks to Trackio. Datasets,
rewards, and Verifiers environment implementations remain independently owned.

The production framework installs the immutable internal release
`trl==1.9.2.post1`. The Policy Prism constrained-IWOPD feature branch instead
pins the pushed candidate `trl==1.9.2.post5` directly at
`carbonteq-ai/trl@b2dcbd0050f17383f97093d226b227d4b25acd75` until its independent
internal-index promotion is complete. The fork is based on upstream TRL 1.9.2 and
keeps the trainer runtime compatible with `datasets 4.6.1` so Verifiers v1 and
TRL can share one environment. Project-specific job policy and environments
remain outside the fork.
Its entropy metrics also preserve chunked-memory behavior for non-contiguous
sequence slices, which is required for DPO on large-vocabulary models.
The fork also exposes colocated vLLM's engine-level speculative configuration
through `GRPOConfig`. This lets a typed Qwen rollout profile enable native MTP
without bypassing TRL's weight synchronization, importance-sampling correction,
or sleep lifecycle. The generic change was merged in
[`carbonteq-ai/trl#5`](https://github.com/carbonteq-ai/trl/pull/5).
It also accepts non-conflicting colocated engine arguments while protecting
TRL-controlled weight-sync and lifecycle options. Text-only runs of multimodal
models use this to skip an irrelevant dummy vision profiling pass. This was
merged in [`carbonteq-ai/trl#6`](https://github.com/carbonteq-ai/trl/pull/6).
The bounded-wave follow-up additionally accepts validated `max_num_seqs` and
`max_num_batched_tokens` engine caps. This lets a 256/512-completion logical
batch execute as multiple resident-32 rollout waves instead of allowing TRL's
derived generation batch to overcommit KV cache. The behavior is included in
`trl==1.9.2.post1` and still requires qualification for each selected
model/runtime profile. It also bounds the actual colocated `LLM.generate`
request list to the configured
resident sequence cap; engine caps alone do not prevent vLLM from queueing a
large logical batch in one call.

The post3 candidate additionally renders the same semantic messages through
the teacher's native template while replaying the exact student completion
ids. Its replay processor computes XGrammar-constrained student/teacher token
log probabilities before forcing each exact token; Posttrain recomputes the
current-student denominator under the same grammar in bounded position chunks.
Schema, completion, request, allowed-set, and chat-template fingerprints fail
closed before an optimizer update. This path is qualified only after the two
Policy Prism live canaries recorded in the experiment plan.
Composite vLLM implementations may retain a namespace around a text-only
training model. The fork therefore exposes an explicit weight-name prefix at
the synchronization boundary instead of placing model-name rewrites in a job.
The generic change was merged in
[`carbonteq-ai/trl#7`](https://github.com/carbonteq-ai/trl/pull/7).
For PEFT QLoRA, the fork also exposes native LoRA synchronization. It leaves
vLLM's quantized base untouched, exports only the current adapter, and reloads
that adapter through vLLM's dynamic-LoRA API. This avoids treating packed
4-bit parameter storage as a dense weight tensor. The generic change was
merged in [`carbonteq-ai/trl#8`](https://github.com/carbonteq-ai/trl/pull/8).
Because level-2 sleep discards the immutable base and vLLM cannot reload a
bitsandbytes checkpoint in place, native-LoRA mode uses level-1 sleep. This
CPU-backs the quantized base while still releasing its GPU allocation; full
weight synchronization retains level 2. The lifecycle correction was merged
in [`carbonteq-ai/trl#9`](https://github.com/carbonteq-ai/trl/pull/9).

### LoRA policy parity and OLMo 3 OlmoRL

The release fixes a namespace gap between these two
generic seams. A Qwen3.5 PEFT actor exports keys below
`base_model.model.model...`, while its text-only vLLM rollout model owns the
same modules below `base_model.model.language_model.model...`. Full-weight
synchronization already applied the inference binding's weight-name prefix;
native-LoRA synchronization rejected that prefix and could therefore load an
adapter without attaching it to the modules used for generation.

The release applies `weight_name_prefix` to only the disposable adapter
export used to refresh vLLM. The actor checkpoint remains in PEFT's native
namespace. A second generic guard compares sampler and actor log-probabilities
over the first training rollout's selected tokens and fails before optimizer
step one when the globally weighted mean absolute delta exceeds `0.05`.
Importance-sampling correction remains active for small numerical drift; it is
no longer allowed to conceal a structurally different rollout policy.

All affected Qwen3.5 native-LoRA inference bindings select:

    weight_sync_mode: lora
    weight_name_prefix: language_model.

The package also exposes `Olmo3GRPOConfig`, a model-agnostic implementation of
the combined OlmoRL objective: zero-gradient filtering with bounded active
refill, token-level DAPO normalization, no KL loss, `0.2/0.272` clipping,
token-level TIS capped at `2.0`, and mean-only group advantages. Its active
sampler requests only the synchronized number of missing rows instead of
generating another full DAPO candidate batch. The trainer path is released;
Posttrain exposes it as `GRPOSettings.algorithm = "olmo3"` and includes the
`qwen3.5-2b/olmo3-grpo-smoke-v1` catalog profile. The typed settings reject
changes to the recipe-defining objective, while batch shape, rollout lengths,
learning schedule, and bounded refill attempts remain profile-owned. The veRL
adapter rejects this selection until it has equivalent active-refill and TIS
semantics. Selected model/runtime profiles still require a live GPU canary.

Framework recovery is independent of the online-RL algorithm. SFT, DPO,
GRPO/DAPO, SAMPO, and on-policy distillation all accept an explicitly
materialized `training-checkpoint`. LoRA and QLoRA checkpoints contain the
adapter plus trainer, optimizer, scheduler, and RNG state and are rejected if
they duplicate immutable base-model weights. On failure or cancellation, the
latest complete checkpoint is committed before the worker workspace is
released. A new `posttrain job run --resume-from-run RUN_ID` invocation uses a
fresh run identity and requires exactly one checkpoint output from the source
run. Recovery may lose work after the last configured checkpoint; interruption
before the first checkpoint has no safe resume point.
The current release exposes that same generic `VLLMGeneration`
synchronization choice through experimental `IWOPDConfig`. This is required
when an on-policy distillation student
uses a PEFT update: full synchronization attempts to merge and push the
adapter-shaped parameter set, while native LoRA synchronization keeps vLLM's
base immutable and refreshes the adapter. It is part of the framework's
immutable TRL pin, but still requires the retained ten-backward-pass
distillation qualification before the overall release can be called complete.
Every prior real training attempt died on step one with
`FloatingPointError: grad_norm=nan`. The root cause was not numerics: the base
`Trainer` counts `num_items_in_batch` from the raw, pre-generation dataloader
batch, whose labels are prompt-only for on-policy rows (the completion does
not exist until generation fills the buffer). That always counted to zero for
a fully on-policy accumulation window, so the divergence loss divided a finite
JSD sum by zero. The fork's `IWOPDTrainer` now recomputes the count
from the post-generation buffered labels and stamps it onto every
micro-slice — the same pattern `GRPOTrainer` already used for its own
post-generation count — and `compute_loss` prefers that value over the
Trainer-level parameter. See `CARBONTEQ_FORK.md` in the fork for the full
rationale and the added regression test.
The released configuration and trainer wiring pass their focused tests. An
order-dependent CPU failure was traced to a vLLM-generation test leaking
distributed-launch environment variables: a later GRPO test initialized NCCL,
and the distillation test inherited that process group. The test module now
restores `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`
after every case. The exact failing three-test order and the broader
vLLM-generation/GRPO/distillation selection pass after the isolation fix;
production trainer behavior was not changed. The complete repository release
gate that previously failed now reports 153 passed and 60 skipped with
distillation executing after vLLM and GRPO in the same interpreter. It leaves
no process group, CUDA context, distributed environment, Ray process, or GPU
client behind. The live ten-backward-pass IW-OPD run remains a qualification
gate for the selected production profile.
DPO kernel choice is model-specific and recorded as `dpo_loss_kernel`. Liger's
fused DPO loss can reduce projection memory for moderate vocabularies, but its
current backward path creates a full FP32 LM-head gradient even when that head
is frozen. The Qwen3.5 profile therefore uses the Torch loss with expandable
CUDA allocator segments; LFM2.5 uses Liger. This is a measured profile choice,
not a universal backend default.
See [ADR 0007](../../decisions/0007-trl-vllm-025-fork.md) for the provenance and
upgrade policy.

For GRPO, the fork additionally exposes `logits_chunk_size`. It bounds the
number of flattened token positions projected through the LM head at once
during old-policy and reference-policy scoring, then reconstructs the same
token-aligned log-probabilities and entropies. The focused fork regression
compares chunked and unchunked numerical results. This control does not bound
the differentiable train loss by itself; the current constrained profile pairs
it with `use_liger_kernel=true`.

## SAMPO support

The release adds two generic runtime seams used by `train.sampo`:
bounded retained-group dynamic sampling and finite token-aligned advantages
returned by `rollout_func`. TRL still computes rewards and group variance for
filtering and evidence, while the framework-owned rollout layer supplies the
hierarchical episode/turn advantage used by the loss. The adapter selects one
sequence-level geometric-mean importance ratio and the standard clipped
PyTorch loss. Liger is rejected because it does not accept these precomputed
advantages.

This support is selected by the immutable workspace pin. The current veRL
adapter rejects SAMPO because its GSPO kernel does not supply the required
hierarchical GiGPO estimator. A real multi-turn GPU qualification is still
required before SAMPO is described as quality-qualified.

The fork's colocated vLLM path has been exercised on the local RTX 3070 Ti with
a 0.5B Qwen smoke through engine creation, CUDA graph capture, weight sync,
generation, and token-logprob extraction. That compatibility smoke does not
replace SFT, DPO, or GRPO acceptance for the two foundation profiles.

## Native MTP and TurboQuant rollouts

The release standardizes both controls through the same
backend-neutral inference binding used by veRL:

    engine:
      mode: colocate
      max_model_len: 32768
      kv_cache_dtype: auto
      speculative_config:
        method: mtp
        num_speculative_tokens: 1

`speculative_config` enables a compatible Qwen model's native MTP head for
rollout acceleration. It does not add an MTP loss or train the draft head.
The adapter rejects non-MTP methods, non-positive draft counts, models which do
not declare MTP, and trainer-side speculative settings in external-server mode
before constructing a trainer.

For colocated GRPO and on-policy distillation, TRL forwards the speculative
configuration without bypassing weight synchronization or sleep/wake. vLLM's
process-lifetime counters are captured before the rollout engine sleeps,
converted to per-generation deltas, and logged under the same normalized names
as veRL:

- `rollout/spec_num_drafts`
- `rollout/spec_num_draft_tokens`
- `rollout/spec_num_accepted_tokens`
- `rollout/spec_accept_rate`
- `rollout/spec_accept_length`

TurboQuant uses the same binding with
`kv_cache_dtype: turboquant_k8v4`. The private TRL adapter forwards the cache
dtype, selects an FP16 rollout copy on the local Ampere target, and applies the
narrow vLLM 0.25.1 cache-marker guard only if that build still reports no
TurboQuant quantization mode. TurboQuant affects rollout KV-cache storage, not
QLoRA actor weights.

This is an experimental configuration surface, not a Qwen 3.5 quality claim.
The existing matched probe increased cache-token capacity by about 2.67 times,
but K8V4 failed the beginning-of-context recall check at 8K, 16K, 24K, and
32.7K where normal KV passed. Therefore the first real TRL MTP GRPO and
distillation qualifications must use normal KV. K8V4 becomes supported for
Qwen 3.5 only after it passes deterministic short-generation and 32K recall
comparisons against normal KV. Combining MTP and K8V4 is a later, separate
qualification.

### Qwen 3.5 0.8B MTP GRPO qualification

Run `artifacts/automationbench-trl-qwen35-08b-mtp-qualification-09` completed
four original AutomationBench trajectories across two optimizer steps on the
local RTX 3070 Ti. It used a 32,768-token engine window, native MTP-1, BF16
LoRA, vLLM sleep mode, eager rollout execution, and a 640 MiB explicit KV
cache. The effective GRPO batch was two generations, executed as physical
microbatch one with two gradient-accumulation slices. This is still one GRPO
group per optimizer update; accumulation changes memory scheduling, not the
algorithm batch.

The first step had non-zero gradient norm `0.1378`, reward mean `0.25`, reward
standard deviation `0.3536`, and 84.35% MTP draft-token acceptance. The second
post-synchronization rollout had 87.49% acceptance and completed its second
backward/optimizer cycle. Its gradient was zero because both sampled rewards
were identical, which is expected GRPO behavior; the exported adapter was
already changed by step one. Checkpoint 2, the final adapter, four native
traces, and the training summary were all preserved.

Run `artifacts/automationbench-trl-qwen35-08b-mtp-qualification-10` requalified
the final step-total observability bridge after replacing TRL's default metric
averaging. Its two complete trajectories recorded 237 draft tokens, 204
accepted tokens, 86.08% weighted acceptance, gradient norm `0.1332`, reward
standard deviation `0.3536`, a recovery checkpoint, and changed LoRA-B
weights.

The failed attempts are also operational evidence. A 256 MiB KV allocation
cannot represent one 32K MTP request; vLLM reports a 0.49 GiB minimum. CUDA
graph capture leaves about 1.05 GiB in private pools after rollout sleep on
this device, and a physical actor batch of two then OOMs during backward.
Therefore the qualified 8 GiB profile uses `kv_cache_memory_bytes=671088640`,
`enforce_eager=true`, physical microbatch one, and gradient accumulation two.
Do not lower the context target or call a constructor-only run a substitute.

The exact qualification entrypoint is
`uv run posttrain job run` against the matching AutomationBench work package.
On-policy distillation accepts the same MTP and engine settings in code and
unit coverage, but still requires its own real student-rollout plus
teacher-score GPU qualification before it is marked released.

### Current 8 GiB, large-group optimization

The current matched benchmark increases the workload to two AutomationBench
prompt groups, eight generations per group, and three intended optimizer
updates with an 8,192-token engine window. Its TRL selection uses:

- physical actor batch one and gradient accumulation 16;
- a 192 MiB explicit KV-cache budget;
- native MTP-1 with eager colocated vLLM and rollout sleep;
- LoRA rank 8 on `o_proj` and `down_proj`;
- `logits_chunk_size=128`;
- Liger's fused GRPO loss.

The failure sequence isolated two different vocabulary-projection allocations.
Full-batch old-policy scoring requested another 1.43 GiB, while batch-one
scoring still attempted a 3.54 GiB full logits tensor. Chunked projection plus
the fused differentiable loss crossed both boundaries. Trackio run
`train.grpo-494bbf38` preserved 16 completed native AutomationBench traces
totaling 1,786,242 bytes, but the process was interrupted before optimizer
update one completed.

This is memory-boundary evidence, not a qualified three-update profile and not
a valid TRL-versus-veRL timing result. The detailed operating sequence,
failure ladder, benchmark method, and release gates are maintained in
[Optimizing GRPO on a single GPU](../../techniques/grpo/single-gpu-optimization.md).

The fork implementation is published and selected immutably. The maintained
delta, source/test surfaces, constraints, and rebase procedure live in the
fork's root `CARBONTEQ_FORK.md`.

The code-defined `posttrain-lab` entrypoint now composes typed training requests
with job-owned data. Reusable trainers remain callable directly from Python.
The generic `VerifiersOnlineRLBridge` runs native Verifiers episodes through a
policy client backed by TRL's already-loaded generator. It returns aligned
token IDs, sampling logprobs, environment masks, rewards, and native traces;
the private TRL adapter converts those values into its custom rollout contract
and records traces through the execution context. Verifiers does not initialize
a model. Transformers and colocated-vLLM generation remain explicit
training-profile choices rather than behavior hidden in job code.
