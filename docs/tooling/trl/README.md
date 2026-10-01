# TRL

## Sampled KL boundary and matched kernel qualification

Published source candidate `5d4f9ad3c5f5d51b1ea50b827d82fdd379231dbf`
uses a tenth-order k3 series through absolute delta 0.25. The finer independent
typed-input sweep closes 268 failures and passes all 3880 checks; 92 focused
SAMPO tests pass. Matched Qwen, LFM and tiny Gemma4 BF16/FP16 loss kernels
pass 12 updates and 24 microbatch comparisons. TRL applies those updates;
native veRL engines and production wheel/runtime adoption remain open.
See [matched-kernel evidence](../../research/evidence/correctness-matrix/matched-kernel-results.md).


## Preference divergence numerical qualification

Published source-only candidate
`416b8978053d56bf4bc8674748c652f464999fac` on `codex/sampo-local-credit`
uses stable expm1 for forward-KL and alpha-divergence DPO scores. Matched
BF16 Qwen0.8B/LFM1.2B controls reproduce 1.9–2.3% parameter-gradient errors
near alpha=1 before correction and exact agreement on all six repaired
preference steps. The independent 306-case grid improves from 69 to 13 failures;
remaining strict value gates are retained. All 66 expanded BF16 model-step
parameter checks match; four SPPO-hard values miss the original tolerance.
Twenty-five focused regressions and four native divergence training tests pass.
These are numerical checks, not full recipe or held-out learning qualification.
FP64 is reference-only. Runtime wheel pins and default objectives are unchanged.
See [preference branch evidence](../../research/evidence/correctness-matrix/preference-branches-and-divergences.md).

## Near-zero KL and native Trainer numerical qualification

Source-only follow-on on `codex/sampo-local-credit` evaluates sampled k3 with
published source `9f0825046ae3509a6be804d74a93fb89d5dc695e`. It uses
a stable series near zero, preserving the selected estimator, beta, and
bias-correction setting. Decimal80 reproduces 22/28 value/gradient failures
before correction; all 28 pass after. The focused fork slice passes 82 tests.
Matched native Trainer runs on Qwen0.8B and LFM1.2B in BF16/FP32 complete three
updates each with supplied multi-turn traces. All 24 microbatch derivative
checks pass after correction; three prior BF16 parameter-gradient misses
disappear. Tool/context tokens have zero direct loss gradient while remaining
visible as conditioning context. Fresh Verifiers collection, task learning,
fused/distributed paths, and wheel/runtime qualification remain open.
Evidence: `docs/research/evidence/correctness-matrix/native-multiturn-results.md`.

## Uno full-policy and native LoRA-policy refresh candidate

The post9 candidate supports both CUDA-IPC full-policy refresh
and native policy-LoRA refresh. For Uno with LoRA training, target and seed
rows use the current policy adapter. Draft-noise rows use an atomically rebuilt
rank-concatenated adapter whose delta is exactly policy plus Uno.

This composition is required because vLLM selects one adapter per token row.
It preserves native LoRA precision instead of folding small updates into the
bfloat16 base. Posttrain selects `lora` for LoRA update plans, `full` for
full-parameter plans, and rejects mismatches before allocation. QLoRA is not
covered.

The K2 live gate passed on RTX PRO at learning rate `1e-4`: trainable delta norm
`0.1214345`, finite loss `2.5226655`, 32/32 finite sampled logprobs, stable
completion tokens, and maximum post-update logprob movement `0.0655067` across
policy versions `0` and `1`. The secondary full-policy gate remains open; this
candidate is published to GitHub and selected by the development pin.

## GDPO/CAPO follow-on qualification

`scripts/qualification/structured_trl_lifecycle.py` exercises installed TRL
1.12.0.post2 on an immutable tiny Qwen fixture with masked precomputed GDPO/CAPO
advantages and beta=0.1. Both complete two finite nonzero-gradient CUDA updates,
resume checkpoint 1 to matching uninterrupted weights, and generate from the
export. This is deterministic full-parameter fixture evidence, not live Verifiers,
judge, LoRA, vLLM or pilot-model qualification. Main pins remain unchanged.

Selected candidate: `1.12.0.post12`, tag `carbonteq-v1.12.0.post12`, release
commit `c4d0db051a7839fe1b1d587185fac33ba88c784f` (branch
`codex/fp16-loss-fp32`, feature commit `9aa833ae`, on the post11 release
commit). It is post11 plus float16-safe GRPO and RLOO: float16 logits are
scored in float32 on GRPO's full and chunked-logits paths and RLOO's, and the
GRPO loss takes float16 log-probabilities and entropies to float32 before its
KL, importance-ratio and masked-sum arithmetic. bfloat16 and float32 behavior
is unchanged, and so is the `vllm` extra. Wheel SHA-256
`7fb9b9c3805e65cb064a1789060d40f4ebe3b31833840f1f439c51241de0029d`, sdist
`81d60f035864a36d4c65b429504c58bce47beed3f9a530abec0c2dd08b4142ce`; both are on
`carbonteq/dev` and the GitHub prerelease. Posttrain's float32 trainer subclass
and LM-head upcast stay in place and are no-ops on post12
(`packages/train/tests/test_trl_precision.py` checks they leave the loss
unchanged). The catalog lock `trl-fork@current` is
`f18308d1af3fbfe72c9be17344762ae83dbbeaeb23a6d736767f166c763aed80`.

Previously selected: `1.12.0.post11`, tag `carbonteq-v1.12.0.post11` (GitHub
prerelease https://github.com/carbonteq-ai/trl/releases/tag/carbonteq-v1.12.0.post11,
asset digests verified against the hashes below). Posttrain retained-asset
workflow run https://github.com/carbonteq-ai/posttrain/actions/runs/36434036656
published the same bytes to `carbonteq/dev`, which serves both exact hashes; the
lock resolves them from there. Stable promotion and GPU qualification remain
open. Fork branch
`codex/active-sampling-oversample`, release commit
`3135b502d69956200d1c030a514470c351d2ee9f` (feature commits `0b428cd6` and
`2393648d`), on top of post10. It adds active-sampling oversampling:
`GRPOConfig.active_sampling_oversample` adds that many prompt groups to the
first round and `active_sampling_oversample_refill` adds that many to each
refill round, never exceeding the first round. The update still keeps the first
target groups with reward spread in candidate order and discards the surplus.
Both default to `0`, which is byte-for-byte post10 behavior. Local build from
the release commit (`SOURCE_DATE_EPOCH=1790604110 uv build --python 3.13` on a
`git archive` export; the wheel rebuilds to identical bytes, the sdist does
not): wheel SHA-256
`7fcea40a21239ae57d22333aa612af939cf8e44a72443681ad4900a697e5a5b2`; sdist
`bb3cdec95d3562054b4ad599a8ce8975232f466c7a593171f8798372c40c7785`. Publish
those exact files. The previous pin was post10.

Post11 also adds `peft_reference` to `GRPOConfig` and `RLOOConfig`. Upstream
gives a run that continues a trained adapter a frozen copy of that adapter as
its KL reference, so every `--model-from-run` restart re-anchored the penalty
to an already drifted policy. `peft_reference="base"` creates no copy and
scores the reference with adapters disabled, the base model. Posttrain's GRPO
and SAMPO settings take `kl_reference: base | start`, default `base`. The TRL
backend passes `peft_reference="base"` only for a continued adapter with
`beta > 0` and `kl_reference: base`; everywhere else TRL's default already gives
the intended reference (a fresh LoRA adapter starts at zero, so disabling it is
the base model, and a full-parameter run from the foundation loads the
foundation as `ref_model`). The TRL backend loads no other starting form for
training: a full-parameter update from an adapter or a `full-finetuned`
checkpoint is rejected when the model loads. `kl_reference` is part of the
SAMPO reward-contract digest; `start` hashes like settings written before the
field existed, so older checkpoints resume with `kl_reference: start`, and a
resume that would switch the reference is refused. `posttrain job plan` prints
the reference and runs record `kl_reference` (resolved: `off`, `base` or
`start`) and `kl_reference_setting`. GDPO and CAPO have no setting and keep TRL's
default. The veRL backend continues an adapter on its foundation weights and
computes the LoRA reference with the adapter disabled (the base model), and
otherwise loads the reference from the starting checkpoint; job planning and
the veRL launcher reject `kl_reference: start` for a continued adapter and
`kl_reference: base` for any other non-foundation starting model (see
`docs/tooling/verl/README.md`).

Posttrain exposes it as GRPO (OLMo 3) and SAMPO settings
`active_sampling: {max_candidate_batches: N, oversample: K1, oversample_refill:
K2}`, counted in prompt groups. `oversample` and `oversample_refill` are passed
to TRL only when non-zero, so exact refill keeps working on post10; a non-zero
value with an earlier TRL fails before training with the required version. The
adaptive-curriculum sampler in `policy_curriculum.py` applies the same round
sizes. The first round, `(num_prompts_per_step + oversample) *
num_generations` episodes, is the largest round and must fit every rollout
concurrency limit: the rollout engine's `max_num_seqs` (TRL's default is one
generation batch), the environment's `max_concurrent` and, with native workers,
`env_workers * episodes_per_worker`. `posttrain job plan` and trainer start
reject an oversampled first round that exceeds any of them, naming each limit.
The candidate reservation (`num_prompts_per_step * max_candidate_batches`) must
hold `num_prompts_per_step + oversample` and still fit the environment's tasks.
Oversampling is excluded from the SAMPO reward-contract digest: it changes
rollout cost and wall time, not rewards, credit or how an update is assembled,
so existing checkpoints keep their digest and a resumed run may turn it on.
With an adaptive curriculum it does change which tasks later rounds select,
because extra groups add evidence before a refill is chosen; that is a sampling
choice recorded in the run attributes, not a learning-semantics change.

Previous candidate: `1.12.0.post10`, release commit
`4950b99d457faacbec856cbd5305732e7b3cf7b0`, tag `carbonteq-v1.12.0.post10`.
It makes the single-process GRPO actor update cheaper for long agentic
episodes: micro-batches are scored at their own real extent instead of the
generation batch's padding, decoder layers can be compiled individually
(`compile_decoder_layers`), checkpointing can be limited to long micro-batches
(`gradient_checkpointing_min_tokens`), and the vLLM importance-sampling ratio
can come from the training forward (`importance_sampling_from_training_logps`).
Training binding `training/lfm2.5-2.6b-trl-lora-automationbench-local-g64-w8@2`
selects compile and the training-forward ratio and keeps every episode
checkpointed. With DSpark rollout, VORTEX v5 LFM2.5-2.6B updates fell from an
823-second average (actor update 373 seconds) to 276 seconds (actor update 114
seconds, all of it forward and backward; the optimizer step is negligible), in
runs `lfm26-v5-opt-ab-dspark-20260926-r1` and
`lfm26-v5-opt-breakdown-20260926-r1`. Its `vllm` extra pins CarbonTeq vLLM
`carbonteq-v0.29.1.dev4`. Wheel SHA-256:
`96710f9cb530a9adae0de2cc0f37f7d77e0f8d0c9f9dc022489f62ba10d08bfd`; sdist:
`d46df4912ef0b647ebf54b11ddf4359406bba97033c33334d018e17ae7191837`.

Previous candidate: `1.12.0.post9`, release commit
`3b7a582e011a32de74d1f2b6e572b794360fbfd7`, tag
`carbonteq-v1.12.0.post9`. It publishes native Uno policy-LoRA refresh,
MoE-safe chunked scoring, cache/speculation observability, and the
continuous-batched colocated vLLM
request path, session-owned LoRA refresh, and async parity-probe routing on top
of the post6 async lifecycle and post5 complete-group admission behavior. It
consumes retained source-row
identities before reward calculation, validates complete groups, and keeps
single-process GRPO accumulation normalized over admitted samples. OLMo active
sampling accepts partial and empty candidate rounds within its existing bound.
Padding exists only in trainer tensors after scoring; no synthetic rewards or
episodes are created. Partial distributed, multimodal, fixed alternate-loss,
fused-loss, entropy-bonus and auxiliary-loss cases remain unqualified/rejected.
The focused post9 gate passes 33 tests across async session, cache/speculation
metrics, Uno policy refresh, and MoE scoring. Wheel SHA-256:
`c5d204da587a9a45dd279f1407d0e20816344075d001da9fe03f6dd4879fa710`;
sdist: `146e46b37f6795a206d684e286e137edb1eec673b658d0168d45c603870156fa`.
Posttrain retained-asset workflow `35292803386` passed development-index
publication, exact-byte readback, and clean installation.
Live RTX PRO qualification remains pending. Harness optimization is deferred.

The post8 candidate connects `GRPOConfig.vllm_request_mode`
to a lazily owned colocated `AsyncLLM`. Posttrain selects it only together with
an explicit bounded `rollout_execution` topology; `batch` remains the legacy
default. During each synchronous GRPO/OLMo collection round, native Verifiers
workers may submit independent model turns to vLLM's continuous scheduler.
Posttrain then closes admission, drains all requests, suspends inference, and
only then returns control for the optimizer update. The initial supported
shape is single-process LoRA/QLoRA. Deterministic TRL and Posttrain tests pass;
a changed-weight GPU update remains required before stable promotion.

The async rollout lifecycle is published in post6 from consolidated fork branch
`codex/posttrain-v04-dev`; ledger follow-up commit
`3ab670f3611f381b373b7e97267879a4afde37ff` records its development
publication. The current pushed development head is
`1deb8d0191b8a2a70be615f2348d16936bb33dc3`; functional commit
`e3f49dc796d1013ff735bc383103ca552be51a34` adds the unpadded
hybrid-model parity correction described below and is not yet a published
distribution. It includes corrected
learner-consumption acknowledgement and native-agent qualification. Against
the selected vLLM 0.25.1 runtime, its
bounded Qwen 0.5B gate passes independent request completion, explicit abort,
sampled-token logprobs, drain, staged weights/KV-cache wake, sleep, and clean
shutdown on the local RTX 3070 Ti. The first attempt found that restoring only
weights leaves vLLM scheduling paused and that shutting down while allocations
remain asleep produces a CuMem cleanup error; both lifecycle transitions are
covered by the fork tests and the repeated real-engine gate. Changed actor
weight synchronization and actor/sampler parity now also pass in a distinct-host
GPU topology. The passing implementation was
`2308ab41aeedc082e154734f33cfe44809b1fdea`: an RTX 3070 Ti actor transferred a
changed normalization tensor over the production NCCL client to an RTX PRO
6000 vLLM server. Base selected-token log-prob delta was `0.0022419691`; after
transfer it was exactly `0.0`, while the server value moved by
`10.6749088764`. The live failure-boundary gate now injects an invalid native
`finish_weight_update` into the real trainer synchronization path and proves
the server keeps version 0 authoritative with unchanged selected-token
log-probability. It is a single-rank control-path result only; repeated NCCL
group initialization on a long-lived vLLM server remains an unqualified
distributed failure-propagation case. A controlled asynchronous
Verifiers-to-learner optimizer update now passes locally; real environment
serving, a 2B learner update, an immutable candidate package, and
checkpoint/resume remain open. See
`docs/plan/async-continuous-rollout-workers.md` for the exact command and gates.

The local candidate also adds an acknowledged model-request drain to native
async GRPO and async distillation. Before weight transfer, the trainer closes
group and model-request admission and waits for every already-admitted request
to finish. Tool-running episodes remain alive and wait before their next model
turn. After vLLM resumes, the trainer publishes the new policy version and
reopens admission. This is a trainer lifecycle seam; Verifiers still owns
environment execution and exact sampled evidence, while stale-sample policy
and importance correction remain trainer-owned. Deterministic fork tests pass
for ordering and the cross-process drain handshake. A transfer-failure regression additionally
proves that the prior model version remains authoritative and inference is
neither resumed nor reopened when weight publication fails.

The candidate includes `scripts/qualify_async_vllm_changed_weight.py` and two
dstack task descriptions for the native NCCL actor/sampler parity gate. The
passing server used immutable runtime image
`registry.carbonteq.com/carbonteq/posttrain-kind-online-rl-trl-py312@sha256:8230413ea572158e59e3f4099b218474d339869fb3eb1676ebaf23e35d35d03d`.
Its slim runtime has no CUDA compiler, so the task explicitly selects vLLM's
native sampler. This is already required for `processed_logprobs` because the
FlashInfer sampler cannot return post-top-k/top-p log probabilities; FlashInfer
attention and NCCL transfer remain enabled. Rollout-only sampling performance
with FlashInfer remains a separate benchmark, not a reason to invalidate the
correctness gate.

The same candidate adds optional recovery hooks for custom rollout workers.
TRL's collator transports one group identity per sample, but acknowledgement
occurs only when `training_step` receives that batch. This distinction matters
because dataloader prefetch can collate a later group that the learner never
uses. TRL saves or restores worker-declared JSON scheduling metadata with the
trainer checkpoint; it does not serialize live queues, environments, requests,
or processes. Posttrain uses this seam to keep generated/enqueued work distinct
from learner-consumed work and fails closed on a partially consumed
relative-reward group. Whole-group batching and a real resume must still be
qualified before the mode can be selected.

The original native `AsyncRolloutWorker` also passed a local tool-calling
throughput gate with Qwen3.5-2B on one RTX 3070 Ti. Sixteen groups of four
produced 64/64 successful tool calls and zero tool failures. Holding the
workload and sampling controls constant, concurrency 32 completed in 18.7395
seconds (3.4152 samples/s), compared with 46.7045 seconds (1.3703 samples/s)
at concurrency one, a 2.49x speedup. The gate found and fixed standalone
Accelerate-logger coupling and a false failure on normal Python 3.13 task
cancellation. It is rollout evidence, not a 2B optimizer-update qualification.

A composed 2.6B LFM canary on the local 8 GiB target is intentionally rejected.
TRL sleep mode releases vLLM during optimization, but actor and vLLM weights
coexist during rollout; two BF16 policy copies have a 10.02 GiB weight-only
floor before KV cache, activations, adapter state, and workspaces. Run
`lfm26-local-lifecycle-unpadded-20260909` confirmed this at startup when the
loaded actor left only 1.66/7.63 GiB free. Posttrain now rejects that provable
cross-seat conflict before packaging. The composed changed-weight gate selects
the 96 GiB RTX PRO target; 8 GiB lifecycle qualification must use a smaller
policy rather than a startup-only offload workaround.

Posttrain now has an unselected run-scoped token gateway for this candidate.
It forwards native Verifiers requests to the trainer-owned vLLM server while
gating model turns around the fork's two-phase weight publication. It records
the served version of each request by Verifiers trace session and persists the
resulting span in the native episode before projection. This avoids deriving
provenance from wall-clock completion or asking Verifiers to own trainer state.
Deterministic two-turn and shared-upstream-failure tests pass; changed-weight
actor/sampler parity passes, while the corresponding live failure path remains
a release gate.

Previous candidate: `1.12.0.post4`, commit
`19e6c89a18617f1bd6e6385212705a67f5434962`, supersedes post3 below without
overwriting its immutable assets. Post4 initializes the bounded diagnostic
configuration on the trainer and includes the post3 change that bounds the one-time actor/vLLM raw
policy-parity probe to selected rows and a 4,096-token prompt-plus-completion
window without changing rollout or optimizer-update sequences. Its retained
wheel SHA-256 is `53214b7a6a58114d542ba319df519fb7942ef4a1e906727c6a866ee06244ecf8`;
sdist SHA-256 is `27d8849a72e6c3aae4612630e469645e0aa6450817c04e6a0afd33ef23a7dd5b`.
Development publication workflow `34220248742` passed exact-byte readback and
clean installation. Live GRPO/OLMo GPU qualification remains open.

Post2 repairs IW-OPD checkpoint skipping
losing Accelerate device placement and batch repetition. The installed wheel
passes native CUDA two-update training, matching checkpoint resume, and
exported-model generation with accumulation 1 and 2. The clean framework
candidate contains reproducible `scripts/qualification/trl_retained_fork_lifecycle.py`
and JSON receipts. Development publisher: `34007394648`; vLLM/runtime-image
qualification and stable/main adoption remain open.

TRL is the execution library behind `packages/train`.

## 2026-09-06 upstream audit: candidate, not a pin update

The user subsequently prioritized the full upgrade. The retained fork is now
integrated on v1.12.0, committed and pushed as
`6a5532e2f51e4e1cdc8a891582514a50f68a775a`, and released as candidate
`carbonteq-v1.12.0.post1`. Its wheel SHA-256 is
`cf242fafdfe476b7b8a250b300d6cbd52f502410a4053f4a9bac3366287727c5`;
sdist SHA-256 is `1e7bae5ee846972be7763e66dbd0b1149d99fb44c51125e2499adf4773d2e127`.
Validation: 140 focused source checks, 45 installed-wheel checks, and 26
framework adapter checks pass. Upstream Liger >=0.8.2 normalization replaces
the older compatibility adaptation; full-weight wake restores the trained
actor. Posttrain Actions `34006220244` publishes the retained bytes to dev.
Stable consumer adoption remains gated on runtime-image/GPU qualification.

The selected source remains `69cf80a7319079ec5523841553467e119ebc1cec`
(`1.9.2.post11`). Upstream v1.12.0 is available, but is an accidental duplicate
of v1.11.0 according to its release notes. The full vLLM lifecycle/API migration
must retain our exact-token IW-OPD, bounded admission, precomputed advantages,
raw actor/sampler parity, LoRA synchronization, and memory-bounded projection
before a version change is admitted.

An isolated `/home/hammad/projects/trl-gdpo-capo` candidate fixes Liger's
microbatch versus generation-window token normalization using the retained
Liger 0.8.0 API. Nineteen CPU tests pass, including native Liger loss/gradients,
positive-beta KL, and two-rank Gloo parity. The first 18 tests produce 14
failures against the installed pin; the unchanged GRPO cases pass. This is
source regression evidence, not a GPU optimizer or release qualification.
The fork ledger records the adaptation and upstream replacement procedure.

The framework explicitly preserves `use_bias_correction_kl=False`; adopting
the newer upstream default would change KL gradients, not just configuration.
See `docs/plan/gdpo-capo-dual-backend-support.md` for the inventory and gates.

The rebuilt `train` package will expose reusable SFT, DPO, and RL operations.
TRL is an internal adapter selected by a typed TRL config, not the object other
projects must compose directly. PEFT/QLoRA, checkpoint behavior, public result
semantics, and instrumentation hooks belong to the `train` package boundary.
The lab's injected observation context maps those hooks to Trackio. Datasets,
rewards, and Verifiers environment implementations remain independently owned.

The current framework selection resolves `trl==1.9.2.post11` from
`carbonteq/stable`,
built from `carbonteq-ai/trl` commit
`69cf80a7319079ec5523841553467e119ebc1cec`. Its prerelease tag,
`carbonteq-v1.9.2.post11`, plus wheel and source hashes are recorded in
`packages/train/pyproject.toml`. Posttrain's retained-asset publisher verified
the release hashes, uploaded the exact bytes to the development index, and
retained a clean-install receipt in workflow `31653581435`. The one-update GPU
canary succeeded, and workflow `31655218728` promoted the same retained bytes
server-side and verified stable readback plus a clean install. The fork is based on upstream TRL 1.9.2 and keeps
the trainer runtime compatible with `datasets 4.6.1` so Verifiers v1 and TRL
can share one environment. Project-specific job policy and environments remain
outside the fork.
The current candidate repairs complete IW-OPD behavior-policy sampling and
rejects non-finite required student, teacher, or behavior-policy log
probabilities at the IW-OPD loss boundary with an affected-token count.
`IWOPDConfig` now declares min-p, repetition penalty, and additional generation
arguments; `IWOPDTrainer` forwards them consistently to local transformers and
vLLM generation. Posttrain's integration test constructs the real pinned config
from its adapter arguments, while the isolated veRL agent-loop test proves that
the same complete `PolicySampling` value reaches the veRL vLLM server request.
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
`trl==1.9.2.post11` and still requires qualification for each selected
model/runtime profile. It also bounds the actual colocated `LLM.generate`
request list to the configured
resident sequence cap; engine caps alone do not prevent vLLM from queueing a
large logical batch in one call.
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

The first Reasoning Gym cold-start qualification exposed a false form of that
gate: it compared vLLM's post-processor sampling log probabilities with raw
actor log probabilities. Temperature, top-p, top-k, repetition, and presence
controls made the mean absolute delta `0.095867`, so the gate failed even
though those controls were the selected behavior policy. The local TRL
candidate now teacher-forces a bounded prompt/completion probe through vLLM
prompt-logprob collection and compares it with raw actor values at temperature
one. Processed sampled log probabilities remain the independent TIS signal.
The candidate also records raw parity mean, maximum, and token count.

The LFM2.5 AutomationBench R9 qualification then exposed a second false-parity
path. The bounded vLLM rows were unpadded, while actor probe prompts and
completions were padded independently across heterogeneous tasks. That made the
actor input nearly twice the configured per-row bound and injected thousands of
leading pad tokens into LFM's recurrent/convolutional stack. Exact retained
trace replay measured `0.00658` mean selected-token delta without padding and
`0.06207` with 2,700 leading pad tokens. The fork now scores the one-time actor
probe row by row without padding. The LFM rollout bindings also omit
`weight_name_prefix`: native LFM vLLM modules use `model.layers...`, unlike the
Qwen3.5 composite `language_model.model.layers...` namespace.
The corrected native LFM adapter was also exercised with 166 deliberately
nonzero LoRA-B tensors: vLLM showed a `0.00910` mean selected-token adapter
effect and remained within `0.00894` mean actor/vLLM delta. A separate local
AsyncLLM probe completed two policy rounds around cancellation, drain, sleep,
and wake. These are source-level component gates; one composed optimizer update
and post-update rollout remain required before immutable release qualification.

This repair is in immutable `trl==1.9.2.post11` bytes, tagged at
`carbonteq-v1.9.2.post11`. Posttrain's candidate consumes it from
`carbonteq/stable` after the one-update actor/LoRA artifact audit and
byte-identical promotion.

Post11 also repairs IW-OPD's loss denominator for fully on-policy batches. The
base Trainer counts labels before rollout generation and therefore supplies
zero items for a prompt-only raw batch. IW-OPD now recomputes the global count
from buffered completion labels and carries it into the loss, matching the
existing post-generation normalization seam in TRL's other online trainers.

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
released. When the host cancels a GRPO, DAPO, OLMo 3, SAMPO, GDPO, or CAPO run,
the online-RL adapter first saves and publishes a checkpoint of the last
completed optimizer update if it is newer than the last periodic one; a
cancellation that arrives during an optimizer step is delivered after that
update completes (bounded at 60 seconds), and the run's `cancel_checkpoint`
event records the saved step or why nothing was saved
(`docs/plan/cancel-checkpoint.md`). `train/cancel_checkpoint_step` carries the
saved step as its value and is recorded at the step of the interrupted update
(the event's `cancelled_update`), because that update's rollout may already
have logged metrics at that step and logical steps never decrease. A new
`posttrain job run --resume-from-run RUN_ID` invocation uses a fresh run
identity and requires exactly one checkpoint output from the source run. SFT,
DPO, and distillation recovery may still lose work after the last configured
checkpoint, and a failure other than cancellation keeps the last periodic
checkpoint; interruption before the first checkpoint and before any completed
update has no safe resume point.
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
compares chunked and unchunked numerical results. For MoE policies, the
backbone pass also retains router logits and computes the same load-balancing
auxiliary loss, so chunking does not require disabling the router objective.
This control does not bound the differentiable train loss by itself; profiles
that do not use the ordinary Torch loss still pair it with an independently
qualified fused-loss implementation.

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

## Trainer and rollout precision

The trainer and the colocated vLLM sampler compute the same policy's token
log-probabilities with different kernels, and rounding in their compute dtype
makes the two disagree; the truncated importance-sampling (IS) correction then
absorbs the difference. Three selections choose the precision on each side
(resolved by `posttrain.train.precision` and shown by `posttrain job plan` on
its `Precision:` line):

    training binding backend_options:
      training_precision: fp16   # default bf16
      fp16_initial_loss_scale: 1024  # fp16 only; default 1024 (PyTorch's own default is 65536)
      logits_float32: true       # default false; always on (and false rejected) with fp16
    rollout inference binding engine:
      dtype: float16             # bfloat16 | float16 | float32; default: checkpoint

`training_precision: fp16` loads the frozen base in float16, sets the
Transformers `fp16` flag (float16 autocast with a dynamic loss scaler), and
requires a LoRA update: PEFT keeps the adapter in float32 over a float16 base,
fresh or resumed, so the scaler steps float32 master weights. The backend
checks that before training and records `train/loss_scale`,
`train/optimizer_step_skipped` and `train/optimizer_steps_skipped` after every
optimizer step; one skipped step discards one rollout batch. The gradient norm
of a skipped step is infinite by construction and is not treated as a failure.
`fp16_initial_loss_scale` is the scaler's starting scale. Starting at
PyTorch's 65536, the Qwen3.5-0.8B fp16 arm overflowed and skipped updates 1-5
and 7 while the scale backed off to 1024, and every skipped step discards a
rollout batch; the framework therefore starts at 1024. Growth (doubling after
2000 finite steps) is unchanged.
`logits_float32` casts the language-model head's output to float32 before the
log-softmax (the logits are still produced in the model dtype, as vLLM's
sampler also receives them); it cannot be combined with the fused Liger loss.
Float16 training always does this, and its trainer also casts the per-token
log-probabilities and entropies to float32, so every loss term (the k3 KL
`exp(ref - logp)`, token and sequence importance ratios, vLLM importance
weights, clipped policy term and masked sums) is computed in float32. TRL
1.12.0.post11 and earlier compute those terms in the dtype of their
log-probabilities (post12 computes them in float32 itself), and
its chunked-logits path calls the LM head outside Accelerate's autocast, so
under float16 they were float16: `exp` of a log-ratio above about 11 is
infinite in float16, and on the masked tool and environment tokens of
multi-turn completions (log-ratios the policy never bounded) infinity times the
zero mask made the loss NaN. The LFM2.5-2.6B SAMPO fp16 canary skipped every
update that way. `logits_float32: false` is rejected with fp16. The fused Liger
loss computes its logits and loss in float32 itself.
Both options are online-RL only (GRPO, SAMPO, GDPO, CAPO); SFT, DPO and
distillation reject them.

`engine.dtype` is forwarded to vLLM. A TurboQuant KV cache still implies
float16 and rejects any other explicit dtype. vLLM's chunked Gated-DeltaNet
kernel rejects float32, so the advisor fails a Qwen3.5 binding with
`dtype: float32` at plan time (`VLLM_FLOAT32_UNSUPPORTED_FOR_GATED_DELTANET`).
A float16 trainer with a non-float16 sampler is a warning
(`FP16_TRAINER_WITH_NON_FP16_ROLLOUT`), and a float16 sampler on a bf16
checkpoint is only reported (`VLLM_FLOAT16_ON_BF16_CHECKPOINT`) when the trainer
is not also float16. The fork itself is unchanged; see
`docs/plan/fp16-training-precision.md` for the offline mismatch matrix and the
training-arm evidence.

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

### Selected-logprob offset stability source candidate

Published source4020c122e4ba2147829ecc0bbaddf6b566a0c8b5 repairs
selective_log_softmax by normalizing with batch-row log_softmax before gathering.
Absolute FP32 selected-logit minus logsumexp loses its small correction at large
common offsets and corrupts derivatives; equal logits1e8 returned0 with gradient
[0,-1] instead of-log2 and[.5,-.5]. Preserve half behavior, dtype and top-K indices.
Focused TestSelectiveLogSoftmax passes28 CPU/CUDA cases after6/8 new CPU
regressions fail on the original source; Ruff/diff checks pass. Larger-context
FP32 backward memory/throughput, native TRL optimizer equivalence and production
assets remain open. This is not a wheel/pin update or evidence of task-quality
causation. Generic regressions and rebase obligations live in CARBONTEQ_FORK.md.
