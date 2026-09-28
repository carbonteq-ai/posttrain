# veRL training backend

## 0.9.0.post8 (selected)

Tag `carbonteq-v0.9.0.post8` (release commit
`ef1c37715fa75de5973ae5b3c398383cd7e0093d`, branch
`codex/agent-loop-config-defaults`, asset receipt `be582879`) is post7 plus one
config fix. Since post2 the agent-loop manager read
`rollout.agent.num_cpus_per_worker`, `max_concurrent_episodes` and
`max_concurrent_episodes_per_worker` from the struct trainer config, but
`rollout.yaml` never declared them; Posttrain passes them only with
`rollout_execution` (accepted on no published fork revision), so every veRL
job failed before its first rollout with `ConfigAttributeError`. Found by the
0.4.12 qualification `q0412h-verl-qwen08b-bf16-r1`. Post8 declares them with
the dataclass defaults (1.0, null, null: one reserved Ray CPU per worker and
unbounded fan-out); the fork test
`test_default_trainer_config_declares_every_agent_loop_field` fails on post7.
Wheel SHA-256 `9da6fb77815d66aa424ab71adfed4862be5291dd12f7a15df4da81d8ffcc0d08`,
sdist SHA-256 `ea34395aea7121f45e43045161830245e4239b0dd08495c9bf2a8f3438725cb1`.
Published: GitHub release
<https://github.com/carbonteq-ai/verl/releases/tag/carbonteq-v0.9.0.post8> and
Posttrain run <https://github.com/carbonteq-ai/posttrain/actions/runs/36489415680>.
`_FORK_NATIVE_NAME_REVISIONS` records post8 and its receipt. Relocking the
`online-rl-verl-py313` kind changed only veRL.

## 0.9.0.post7

Tag `carbonteq-v0.9.0.post7` (release commit
`6069abe14e2b3d27c89815a6502b849f15124e12`, branch `codex/vortex-lora-sync`,
asset receipt `07ecac23`) is post6 plus the LoRA weight-sync fix. veRL named
the synced LoRA tensors with the model's full HF-to-vLLM mapper, whose stacked
maps rename `q_proj`/`k_proj`/`v_proj` to `qkv_proj` and LFM2's `w1`/`w3` to
`w13`; the constituents collapsed onto one name, the last one won, and LoRA on
fused layers crashed or silently loaded the wrong weights. Post7 uses vLLM's
rename-only mapper, as vLLM's own adapter loader does. Post5 and post6 have the
bug. Wheel SHA-256
`9786ec44fbdba367791d8e9a4c58a895955639c79c4b9dd0e3a403a05b5e29c5`, sdist
SHA-256 `3b0259523476d67aff9282b2b3c775c081bd866c04d369ad7a5eda8af6607088`
(`carbonteq/dev` serves both). Published: GitHub release
<https://github.com/carbonteq-ai/verl/releases/tag/carbonteq-v0.9.0.post7> and
Posttrain run <https://github.com/carbonteq-ai/posttrain/actions/runs/36483660287>.

Post6's additions (round-based active sampling, candidate-batch DAPO, a prompt
selector, TRL-mode GRPO statistics and excluded rows, admission retries, the
`sequence_clip` loss, the rollout-correction lower clamp and log-ratio bound,
a linear LR schedule, extra SAMPO metrics) are opt-in, and 0.4.12's adapter
selects none of them. Composing veRL's Hydra config for every override set the
0.4.12 train tests generate (35 sets: GRPO with and without LoRA, DAPO group
filtering, GDPO, CAPO, OLMo 3, FP16 and BF16 training and rollouts, adapter
continuation, distillation, TurboQuant KV cache; veRL SAMPO stays rejected)
under post5 and under post7 differs only by
the new keys at their defaults (`algorithm.active_sampling.enable=false`,
`filter_groups.candidate_batches=false`, `grpo_std_epsilon=null`,
`grpo_std_scope=group`, `exclude_flagged_rows=false`,
`rollout_correction.rollout_is_clip_min=null`,
`rollout_is_log_ratio_bound=20.0` (the previous fixed bound),
`data.prompt_selector.class_path=null`,
`trainer.v1.sampler.failed_group_attempts=0`); with those defaults the trainer
takes the post5 code paths. The one runtime change is the LoRA-sync mapper.
`_FORK_NATIVE_NAME_REVISIONS` records post7 and its receipt, so GDPO and CAPO
(`token_clip`, `k3_unclipped`) are accepted on it. Post8 supersedes it in 0.4.12.

## 0.9.0.post5 and the VORTEX port (in progress)

Tag `carbonteq-v0.9.0.post5` (release commit
`9fd6e7a31396ba33a29233cc869ab05b0a9e5a80`, branch `codex/vortex`, pushed) is
post4 plus the `token_clip` policy loss (asymmetric PPO token clipping without
veRL's dual clip or log-ratio clamp) and the `k3_unclipped` KL estimator.
Retained wheel SHA-256
`c16a2ad14d1bd60947229bde41fae99955a4fcb7ab3f892020ef0c24e35dd158` (byte
reproducible), sdist SHA-256
`3c9e17c2d04a798eae1836e8dd82f64480bcbe4d8ee94441028625b2da71df57`. Published:
GitHub release
<https://github.com/carbonteq-ai/verl/releases/tag/carbonteq-v0.9.0.post5> and
Posttrain run <https://github.com/carbonteq-ai/posttrain/actions/runs/36445711207>
(`carbonteq/dev` serves both hashes). `codex/release-0.4.12` pins it for the
`online-rl-verl-py313` kind together with the framework's Verifiers `cdd2ec76`,
`carbonteq-renderers` 0.1.12.post1.dev2 and the Qwen3.5 fast-path kernels
(`fla-core`, `causal-conv1d`); the regenerated backend lock and constraints
digests are in `verl-py313/profile.toml`.

Post5 fixes a live bug: the adapter selects both names for GDPO and CAPO, and
post3/post4 register neither, so those runs failed at veRL's first actor
update. The worker now records which fork commits register each fork-only name
(`_FORK_NATIVE_NAME_REVISIONS` in `backends/verl/worker.py`) and rejects a
clean checkout at any other revision before veRL starts. A strict-xfail test
(`test_pinned_verl_fork_registers_every_native_name_posttrain_requests`) fails
as soon as the job kind pins a fork with the names, so the pin commit must
remove its marker; `test_verl_fork_native_names.py` checks the record against
an installed veRL.

With post5 the adapter maps `algorithm: olmo3` natively: `loss_agg_mode=token-mean`,
`policy_loss.loss_mode=token_clip`, clip 0.2/0.272,
`algorithm.norm_adv_by_std_in_grpo=false`, decoupled rollout correction
`rollout_is=token`, `rollout_is_threshold=2.0`, and `kl_loss_type=k3_unclipped`
for a nonzero `beta` (the LoRA reference is the base model, as in TRL). A CPU
parity test feeds one fixed batch (both clip bounds, the correction cap,
dual-clip and KL-clamp regions, a truncated rollout) through TRL post11's
`GRPOTrainer._compute_loss` and veRL's `ppo_loss`; advantages, correction
weights, loss and gradient agree to float64 round-off. `truncation_penalty` is
applied in the veRL agent loop by the same shaping function as the TRL path and
is accepted for GRPO and DAPO. OLMo 3 itself stays rejected on veRL until
active sampling is ported (plan [verl-vortex-port.md](../../plan/verl-vortex-port.md),
Phase 2).

## Round-based active sampling (unreleased candidate)

Fork branch `codex/vortex-active-sampling` commit
`6c7295cd411c4d3973ddc206e43816560c842336` (post5 plus this delta, not
pushed) adds `algorithm.active_sampling` to the synchronous V1 trainer with
TRL post11's round semantics: round one dispatches the batch plus
`oversample` prompt groups; each later round dispatches only the missing
groups plus `oversample_refill`, capped at round one and at the remaining
pool of `max_candidate_batches * train_batch_size`; a round completes before
its groups are judged (reward spread of `seq_reward`, the shaped reward);
failed groups are rejected; the first target groups in dispatch order form
the batch. Metrics use TRL's `active_sampling/...` names and values, mapped to
the same `train/rl/active_sampling_*` names. The trainer rejects an oversampled
first round larger than the engines' `max_num_seqs`, the agent episode limit
or worker slots; the launcher and `posttrain job plan` apply the same guard.

The adapter maps `ActiveGroupSampling` for OLMo 3 and accepts OLMo 3 only on
a fork revision recorded with `active_sampling` (or a dirty candidate
checkout), so the selected post5 still rejects it. A side-by-side test against
TRL's real `_prepare_active_sampling_inputs` agrees on 19 scenarios. The 8 GB
GPU check needs a release candidate and kind image containing this commit.

## Adaptive curriculum through the prompt selector (unreleased candidate)

Fork commit `24920b395f8571f8f5be6b9d8469737f2355dcc9` adds
`data.prompt_selector`: a selector chooses the dataset rows of every dispatch
(`initial_batch`, or each numbered `active_sampling_refill` round), observes
finished groups' `seq_reward` values in dispatch order, and saves/loads its
state in each `global_step_*` folder. Posttrain's
`PosttrainCurriculumSelector` (`backends/verl/curriculum.py`) runs the same
`AdaptiveCurriculumRuntime` as the TRL path; its events and metrics are
journaled to `verl-curriculum-events.jsonl` and replayed by the parent, and
the final state is published as `adaptive-curriculum-state`. The snapshot
`adaptive-curriculum-state.json` sits in each veRL checkpoint folder and is also
copied to `curriculum-checkpoints/step-<N>/`, which the launcher publishes as the
TRL path's per-checkpoint view (`checkpoint-<N>/curriculum`), so
`--curriculum-checkpoint-step` warm starts work from veRL runs. The
adapter accepts the curriculum with GRPO and OLMo 3 (not DAPO) on revisions
recorded with `prompt_selector`.

## TRL-equivalent GRPO settings (unreleased candidate)

Fork commit `2607b91d3cccc9d73aae924734b5104bf8cfb590` lets veRL reproduce every
GRPO setting Posttrain selects: sampler correction in all four TRL modes with
lower and upper bounds and exact log ratios; TRL's advantage scaling (group or
batch std + 1e-4, or none) with masked truncated completions excluded as TRL's
NaN rewards are; row exclusion after advantages; TRL's group admission (retry
with the same prompt, then drop, loss normalized over real rows); and the
linear LR schedule. GRPO and DAPO on veRL now use `token_clip` and
`k3_unclipped`, so their loss matches TRL's too. This changes veRL GRPO runs:
their settings always declared these semantics, but earlier veRL runs used
dual clipping, a clamped KL, no sampler correction, veRL's 1e-6 std epsilon
and refill-with-new-prompts on failures. Two DAPO combinations
were first kept TRL-only (curriculum with DAPO, and batch advantage scaling
with DAPO dynamic sampling); fork commit
`ce8e0430018204b03c009b72bfba3b58968696c7` then added TRL's candidate-batch DAPO
dynamic sampling (`algorithm.filter_groups.candidate_batches`), so no GRPO
setting is rejected on veRL. Runs record `verl_semantics: trl-parity-v1`;
`CHANGELOG.md` lists every difference from earlier veRL runs. CPU parity: 49
tests against TRL post11's real code (plan Artifacts).

## SAMPO on veRL (unreleased candidate)

Fork commit `4d37a18bc492f0f4f9c224285603740ef4a2ba54` adds `sequence_clip`
(TRL's `importance_sampling_level="sequence"` objective: one ratio per row with
gradient through the mean log ratio) and SAMPO hierarchy evidence metrics. The
adapter now accepts SAMPO: fork SAMPO estimator, `sequence_clip`,
`seq-mean-token-mean`, clip 0.003/0.004, `k3_unclipped` KL, token or sequence
sampler correction truncated at the selected cap, round-based active sampling,
optional curriculum and truncation penalty; the TRL-equivalent settings
above (masked truncated completions, admission retries, every correction mode
and bound) apply to SAMPO too. This replaces the historical GSPO mapping
described under "SAMPO operating configuration" below, whose runs predate the
rejection that preceded this port.

## LFM2.5 on veRL (unreleased candidate)

LFM2.5 (hybrid short-convolution and attention blocks, tied input and output
embeddings) needs no fork source change: Transformers' `Lfm2ForCausalLM`
trains under FSDP2 with `use_remove_padding=false`, PEFT `all-linear` selects
the attention `q_proj`/`k_proj`/`v_proj`/`out_proj`, short-convolution
`in_proj`/`out_proj` and feed-forward `w1`/`w2`/`w3` projections (never the
tied `lm_head`), and the CarbonTeq vLLM's `Lfm2ForCausalLM` accepts those
LoRA names through its packed-module mapping. Fork commit `7d850ef5` adds a
CPU regression test that exports such an adapter through `verl.model_merger`
and reloads it with identical logits.

The Posttrain side was the renderer. The veRL agent loop used the default
renderer for every non-Qwen family, ignored the model's package chat template
and never recovered LFM2.5's Python call lists (`<|tool_call_start|>[...]`),
so LFM2.5 would have seen different prompts and tool calls than on TRL. The
launcher now resolves the renderer with the TRL backend's own function
(`renderer_config_spec`: family config, reasoning-mode template arguments,
package chat template, tool-call protocol) and the agent loop rebuilds it.
Bindings for LFM2.5 on veRL use no fused-kernel or Qwen-specific Hydra
overrides; `attention_implementation: sdpa` is fine. The GPU qualification
(two updates of LFM2.5-1.2B on the 8 GB card) waits for the post6 image.

## FP16 metrics (0.9.0.post4, contained in post5)

Branch `codex/precision-fp16` tags `carbonteq-v0.9.0.post4` at immutable
release commit `54124edfb8d0b73694696400cf07a76a14d9be65`: post3 plus
`actor/loss_scale` and `actor/optimizer_step_skipped` from the FSDP engine's
fp16 `ShardedGradScaler` and the rollout-versus-actor log-probability gap
(`training/rollout_logp_diff_{mean,p99,max}`,
`training/rollout_seq_logp_diff_abs_mean`); no training or dependency change.
Wheel SHA-256
`1e5e5a50c14ec486019421ca06de03fbbc24010f850f5e66f2d731469a8f8eb0`;
sdist SHA-256
`8620646c250e85a0dee984d360a4c102e97a4776d5711accea8eb66a743445e5`.
Post4 is published; post5, post7 and post8 contain it; post8 is the 0.4.12 selection.

## Rollout-execution development candidate

Branch `codex/verl-rollout-execution` publishes `0.9.0.post3` from immutable
release commit `18338a0efbd6f103378d2861f4a078ad243db455`. Post3 retains
the post2 behavior and pins its `vllm` extra to CarbonTeq vLLM commit
`fbbba6698b2f8a912b94705cfc09eb4fd7243716`. This dependency release does not
claim native K2 training support in veRL. Post2 added opt-in Ray agent-worker episode limits, explicit CPU
reservations, and complete failed-group replacement in the V1 TransferQueue
path. The framework accepts a private
`TrainingBinding.backend_options.rollout_execution` mapping with
`env_workers`, `episodes_per_worker`, and `worker_native_threads`; it validates
their product against the Verifiers bridge's global `max_concurrent` and maps
the values to native veRL settings. Native overrides cannot replace those
owned settings.

The candidate also has a deterministic native partial-rollout gate. It proves
that an aborted assistant generation resumes from the original prompt plus its
retained generated prefix, spends only the remaining token budget, preserves
token-aligned behavior log probabilities exactly once, and reports the served
policy span across versions. Nine native continuation tests pass; the existing
real AutomationBench bridge test separately proves one emitted tool action is
executed once before the next model turn. This is CPU protocol evidence, not a
changed-weight GPU qualification.

The candidate wheel SHA-256 is
`55db59d956084c27941a787952c1d66fcdbd4f7c1d71c9ddd1f20c283af7186f`;
the sdist SHA-256 is
`75caad0671f3f3f70241b049ff6b3fb937cee34a4bdcadd74230b1c46bc4bb31`.
Posttrain workflow `35292934219` passed development-index publication,
exact-byte readback, and clean installation. The development profile selects post3, but
runtime-image reconstruction and a real GPU collection/update gate are still
required before stable promotion. Renderer construction is now
binding-driven for Qwen and LFM, but the existing Qwen-only GPU qualification
policy remains in force.

## GDPO/CAPO follow-on implementation (unpublished)

Algorithm qualification uses deterministic synthetic turn inputs separately
from LLM-judge calibration. The qualification-only runner is
`scripts/qualification/automationbench_deterministic.py`; its fixture wheel is
installed into the isolated interpreter and its digest is retained in selection
and native evidence. Native AutomationBench success is not overwritten.
The first live attempt exposed unset `min_p` being forwarded as `None` to vLLM;
the framework adapter now omits unset options and preserves explicit zero values.
Startup telemetry now labels GDPO/CAPO correctly instead of SAMPO. These fixes
pass the 51 adapter tests; they do not yet close the five-update live-run gates.

Both deterministic veRL runs subsequently completed five updates:
`gdpo-deterministic-d` and `capo-deterministic-a`. Independent Decimal mean/min/max
advantage audits and native Episode token/mask/logprob parity pass. CAPO retained
one rejected max-turns group before replacement. These are not resume or release
certifications. Strict export reload found the merger's leaf-name target
inference could attach untrained vision layers. The unpublished fork now saves
exact module paths, with 17 merger tests passing; original exports are retained
and fresh exports are regenerated from unchanged final checkpoints. Follow
Revision 11 of the execution plan for exact evidence and remaining gates.

The native turn-reward bridge also transports complete selected turn scores into
SAMPO metadata. An explicitly selected scorer is included in its checkpoint
recovery digest; the unselected sparse-terminal path stays unchanged.
The candidate V1 trainer accepts a recovery-only structured contract without
overriding SAMPO admission policy. Framework backend/recovery/SAMPO tests:
61 passed on 2026-09-06. Live veRL turn-reward qualification remains open.

The v0.9.0 candidate checkout now contains explicit structured-reward profiles,
V1 raw evidence transport, bounded invalid-group replacement, ordinary token
clipping and unclipped k3 KL. Legacy loss profiles are unchanged. Checkpoints
retain a selected reward-contract digest and reject missing/changed identities.
The adapter maps versioned projection settings into this contract; runtime
qualification and publication of a new fork version remain open. Do not point
stable pins at this dirty source checkout or claim these changes exist in post1.

## 2026-09-06 upstream audit: candidate, not a pin update

The user subsequently prioritized the full upgrade. The retained fork is now
integrated on v0.9.0, committed and pushed as
`cec7e74c361bb973b641db8dfbb75a5544c33139`, and released as candidate
`carbonteq-v0.9.0.post1`. Its wheel SHA-256 is
`3d66ac6b78848ef591dd6e4be4d324fcac12c27247a05ad1c14dcf9fb370256e`;
sdist SHA-256 is `6da77c10e5d37399c655b15e3ed78d520be2ca648ed763d43ccb7404dfe53d72`.
Validation: 151 focused CPU source checks and 63 installed-wheel checks pass.
SAMPO/V1 transport, bounded refill, dense teacher alignment, 3-D position
repair, QLoRA and speculative metrics remain; equivalent upstream fixes replace
the old maintained deltas. Posttrain Actions `34006221953` publishes retained
bytes to dev. Stable consumer adoption remains gated on runtime-image/GPU
qualification; the harness's Ray 2.49.2 is not the runtime's Ray 2.56.1.

The selected source remains `808923d487aa2c524fda02cf5289110541b4221f`.
Upstream v0.9.0 is available at `483b8a009ba3a97563edee3a19887e4862b8094a`,
but a direct switch would lose SAMPO and its V1 identity/metadata transport.
Dense teacher alignment, repaired 3-D position IDs, bounded group refill,
LoRA synchronization/checkpoints, and distillation normalization require parity
review when porting to that stable base.

An isolated `/home/hammad/projects/verl-gdpo-capo` candidate includes the
REINFORCE++ observation-credit fix from upstream: masked tool observations no
longer erase credit for earlier actions. Twenty-eight native CPU core/regression
tests pass, including non-unit discount factors and float32/float64. This does
not change CAPO's direct-token-credit objective. The dirty `verl-upstream`
checkout is untouched; the candidate is uncommitted and unpublished, and the
runtime-image/GPU gates remain open. Its fork ledger and
`docs/plan/gdpo-capo-dual-backend-support.md` record the exact changes.

The framework exposes veRL as the general versioned training backend product
`verl@<version-or-revision>`. The public operation names and requests remain
`train.grpo` / `GRPORequest` and `train.distill` /
`OnPolicyDistillationRequest`; callers do not use veRL-specific job types.

## Trainer and rollout precision

`backend_options.training_precision: fp16` on a veRL training binding selects
veRL's own FP16 path (upstream verl-project/verl#4036 and #6150, both in the
published fork 18338a0e): the worker adds
`+actor_rollout_ref.{actor,ref}.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}`,
and the V1 trainer's `FSDPEngine` then computes in float16 over float32 master
weights with a `ShardedGradScaler(growth_interval=400)` that skips overflowing
steps. veRL builds that scaler in its Ray actors without a configurable start,
so for `fp16_initial_loss_scale` (default 1024) the worker also passes
`++ray_kwargs.ray_init.runtime_env.env_vars.POSTTRAIN_FP16_INITIAL_LOSS_SCALE`
and the Ray `worker_process_setup_hook`
`posttrain.train.backends.verl.loss_scale_hook.configure_initial_loss_scale`,
which makes the value the constructor's default `init_scale` in every Ray
worker. The bf16 default adds no override. The rollout `engine.dtype` becomes
`actor_rollout_ref.rollout.dtype` (default `bfloat16`; float16 for a TurboQuant
KV cache). veRL logs neither the loss scale nor skipped steps; Posttrain records
a step whose gradient norm is infinite under fp16 as
`train/optimizer_step_skipped`. veRL's rollout-versus-actor mismatch metrics are
probability differences (`train/rl/sampling_prob_delta_*`). GPU qualification
(`qwen08b_gsm8k_verl_grpo_precision_{bf16,fp16}_local.yaml`) waits for the
kind rebuild described below; see `docs/plan/fp16-training-precision.md`.

## Training settings on veRL

The veRL backend runs a selection exactly as the TRL backend would, or rejects
it at `posttrain work-package plan` and again when the launch plan is built
(`posttrain.train.backend_support`). Training loop, as of 0.4.12:

| Setting | veRL mapping |
| --- | --- |
| `lr_scheduler_type: constant` | `actor.optim.lr_scheduler_type=constant`, `lr_warmup_steps=0` (Transformers' `constant` never warms up) |
| `lr_scheduler_type: constant_with_warmup` | `constant` with `lr_warmup_steps = ceil(max_steps * warmup_ratio)`, the TRL value |
| `lr_scheduler_type: linear` (the default) | rejected: veRL 0.9.0.post4 schedules only constant (after linear warmup) and cosine |
| `seed` | `data.seed` (prompt order), `rollout.seed`, `actor.data_loader_seed`, `actor/ref.fsdp_config.seed` |
| `logging_steps` | must be 1; veRL logs every update |
| `per_device_batch_size` x `gradient_accumulation_steps` | must equal prompt groups x generations; one optimizer step per update (`ppo_mini_batch_size` = prompt groups) in micro-batches of `per_device_batch_size` rows per device (`ppo_micro_batch_size_per_gpu` and the log-prob micro-batches); the rows must split evenly over the devices |

Weight decay is 0.0, the Transformers default the TRL backend trains with
(veRL's own default is 0.01). Before 0.4.12 veRL ignored the schedule, seed,
logging cadence and batch split: it always ran a constant rate, micro-batches
of one row, weight decay 0.01 and an unseeded prompt order, while the lab veRL
settings left the schedule at the `linear` default. Those settings now state
`lr_scheduler_type: constant`, what they ran.

GRPO settings veRL does not receive are rejected unless left at their
defaults: `adaptive_curriculum`, `active_sampling`, `advantage_scaling`
(`group`), `importance_sampling_mode`/`_clip_min`/`_clip_max`
(`sequence_truncate`, 0.1, 3.0) and `max_admission_attempts` (3). The OLMo 3
recipe (and with it `active_sampling`) stays rejected until veRL has active
sampling; `truncation_penalty` is applied for GRPO and DAPO.

## Current support and qualification boundary

The current adapter accepts the **Qwen 3.5** and, pending its GPU
qualification with the post6 image, **LFM2.5** model families, for:

- GRPO with fresh trajectories owned and scored by a Verifiers environment.
- On-policy distillation in which a Qwen 3.5 teacher scores the exact token ids
  generated by a Qwen 3.5 student on those trajectories.

The backend name is intentionally not Qwen-specific. Other model families must
be added to this qualification matrix only after an end-to-end GPU test covers
model loading, rollout, optimization, actor-to-vLLM weight synchronization,
checkpoint recovery, export, and native Verifiers trace preservation. The
adapter rejects unqualified families before starting Ray.

This first slice accepts full-parameter and LoRA updates. QLoRA and
quantization-aware updates are not qualified.

A GRPO run can continue a trained LoRA adapter (`--model-from-run` of an
adapter view). The launcher records the adapter's foundation as the model's
`base`; the worker loads that foundation as `actor_rollout_ref.model.path` and
attaches the adapter with the fork's `actor_rollout_ref.model.lora_adapter_path`
(upstream veRL: the FSDP engine calls `PeftModel.from_pretrained(...,
is_trainable=True)`). The synchronous trainer's initial weight sync then gives
the vLLM rollout that adapter before the first collection, and the adapter
keeps training. With LoRA, veRL's KL reference is the actor with the adapter
disabled, the base model, so `kl_reference: base` holds; veRL cannot keep a
frozen copy of the starting adapter, so `kl_reference: start` with `beta > 0`
is rejected at planning and launch. The adapter's `adapter_config.json` rank
must equal the binding's LoRA rank, because vLLM sizes its LoRA slots from the
binding. Any other non-foundation starting model (for example
`full-finetuned`) takes its reference from the starting checkpoint, so there
`kl_reference: base` is rejected and `start` is required. No fork change was
needed; published post3 (`18338a0e`) already carries `lora_adapter_path`.

GPU qualification of this path is blocked by the published `online-rl-verl-py313`
kind image, not by the continuation code. Packing
`qwen08b_verl_grpo_2_adapter_continuation.yaml` (Qwen 3.5 0.8B, local RTX 3070 Ti)
cannot package any Verifiers environment for that kind. Its control
environment locks Verifiers `cdd2ec76` and its backend environment
`b71ade0a`, so gsm8k-v1 0.3.0 (`cdd2ec76`) fails the backend resolution with
conflicting Verifiers URLs. gsm8k-v1 0.2.0 (`b71ade0a`) and the unpinned
alphabet-sort environment resolve, but the kind declares neither
`provided_packages` nor `backend_provided_packages`. The backend compile
therefore emits `verifiers @ git+...` without a hash, and environment
packaging rejects it ("every compiled dependency must have a sha256 hash"). The
release tooling derives provided packages only from
`profiles/<variant>.txt`, which the veRL kind lacks. The fix is a veRL kind
rebuild whose backend selects the framework's Verifiers revision and declares
Verifiers as provided in both roles; then run the package fresh and again with
`--model-from-run` of its adapter.

| Technique | Accepted family | Validation status |
| --- | --- | --- |
| GRPO | Qwen 3.5 | Qwen 3.5 0.8B ordinary BF16 LoRA qualified locally on GPU |
| On-policy distillation | Qwen 3.5 student and teacher | Two-step GPU execution and retained artifacts qualified; required telemetry gate open |
| GRPO, SAMPO | LFM2.5 | CPU renderer and export tests; two-update 8 GB GPU run pending the post6 image |

The complete backend release is therefore not yet production-qualified.

## Runtime isolation

veRL is not installed into the main repository environment. The current TRL
lock and the inspected veRL/Qwen 3.5 stack require different Transformers and
vLLM versions. Shared execution values use normalized `runtime` names, while
veRL-native launch settings use `backend_options`:

- `backend_options.python_executable`: an absolute path to the isolated environment's
  Python executable.
- `backend_options.working_directory`: an absolute path to the veRL fork checkout.
- `backend_options.source_revision`: the complete 40-character Git commit that
  must match the checkout's `HEAD`.
- `backend_options.dependency_lock_sha256`: the digest of the isolated environment lock
  used for lineage. This is recorded by the public operation even though the
  worker does not interpret it.
- `runtime.global_batch_size`: prompt groups multiplied by generations, as
  required by the public GRPO and distillation request contracts.
- `runtime.nodes` and `runtime.devices_per_node`: topology shared by training
  backends.
- `runtime.parameter_offload` and `runtime.optimizer_offload`: portable offload
  intent translated into veRL's actor/FSDP settings.

Additional trainer-native Hydra settings may be supplied through
`backend_options.hydra_overrides`. They cannot replace selected model, data,
agent-loop, or artifact paths. veRL's internal `ppo_mini_batch_size` is derived
from GRPO prompt groups and is not a public setting.

The isolated environment must contain the pinned veRL fork, its compatible
Ray/Transformers/vLLM stack, this workspace's `posttrain-common`,
`posttrain-data`, and `posttrain-train` packages, the pinned Verifiers
environment packages, `renderers` with Qwen 3.5 support, `datasets`, and
`huggingface-hub`.

For a multi-node Ray run, the run workspace, portable bridge snapshot, and
Verifiers trace path must be mounted at the same absolute paths on every node.
The binding's `runtime.nodes` and `runtime.devices_per_node` must multiply to
the training target's world size. Distillation may separately provide
`runtime.teacher_nodes` and `runtime.teacher_devices_per_node`; the teacher
tensor/data/expert parallel product must exactly partition its target world
size.

The launcher writes `posttrain-verl-launch.json`, a portable Verifiers bridge
snapshot, the materialized rollout parquet, the agent-loop configuration, and
native logs inside the run's trainer directory. It invokes the isolated worker
without a shell. The worker verifies the fork commit before downloading pinned
model snapshots or starting Ray.

## Environment and token contract

`PosttrainVerifiersAgentLoop` adapts veRL's already-loaded rollout server to the
framework's `PolicyGenerator`. The existing Verifiers bridge continues to own
episode sequencing, tools, rewards, truncation, and native traces. The agent
loop returns the exact prompt ids, completion ids, per-token rollout log
probabilities, environment response mask, scalar reward, and trace identity to
veRL. Multi-process trace appends use an operating-system file lock.

For distillation, veRL's native teacher server uses the already-validated
student/teacher tokenizer fingerprint equality and the `k1` sampled-token loss.
This slice sets `use_task_rewards=false` and uses veRL's policy-gradient form
of the sampled-token `k1` objective. The gradient is therefore taken through
the current student log probability while Verifiers still owns and preserves
the trajectory evidence.

## Single-GPU phase lifecycle

On one GPU, rollout and optimization time-share device memory. Do not interpret
vLLM's `gpu_memory_utilization` as a permanent partition reserved during
backward. The qualified synchronous lifecycle is:

```text
offload actor and optimizer
  -> wake vLLM weights and KV cache
  -> generate the complete rollout batch
  -> sleep vLLM and discard KV cache
  -> restore actor and optimizer
  -> compute actor logprobs, advantages, backward, and optimizer step
  -> stage the LoRA adapter on CPU and offload actor state
  -> wake vLLM weights
  -> synchronize the adapter
  -> wake the KV cache for the next rollout batch
```

`free_cache_engine=true` enables the rollout sleep boundary. Adapter mode uses
sleep level 1 so vLLM can retain its immutable base backup in host RAM while
releasing GPU weights and KV cache. The fork stages the updated adapter before
waking vLLM weights, avoiding overlap between the trainable actor and rollout
model during synchronization.

## Qualification checkpoint and 32K target

Run `artifacts/automationbench-verl-qwen35-08b-qualification-23` used the
following profile on an RTX 3070 Ti:

| Area | Qualified value |
| --- | --- |
| Model/update | Qwen 3.5 0.8B, BF16 LoRA rank 8 over language `o_proj` and `down_proj` |
| Work | 2 prompts, 8 generations per prompt, 2 optimizer steps |
| Environment | AutomationBench `limited_zapier`, 50 turns, 1800-second rollout timeout |
| Sequence capacity | 8192 model length, 2048 prompt cap, 6144 completion cap, 512 tokens per turn |
| vLLM | `gpu_memory_utilization=0.70`, `max_num_seqs=4`, `max_num_batched_tokens=8192`, eager mode, sleep enabled |
| Actor | FSDP2, parameter and optimizer offload, gradient checkpointing, Torch fused PPO head |
| Entropy | dense-FSDP chunking enabled, chunk size 256, FSDP entropy compilation disabled |
| Worker concurrency | 2 agent-loop workers, 2 reward workers, verifier/tool work asynchronous on CPU |

This 8192-token run is an integration and memory-lifecycle checkpoint. It does
**not** satisfy the product requirement that the backend support at least a
32768-token context window. The next GRPO qualification must use this rollout
target:

| Area | Required 32K target |
| --- | --- |
| Model/update | Qwen 3.5 0.8B, ordinary BF16 LoRA; no QLoRA requirement |
| Sequence capacity | 32768 model length, 8192 initial-prompt cap, 24576 trajectory/completion cap |
| KV cache | Start with normal FP16 KV for the release run; K8V4 is capacity-tested but currently fails the Qwen 3.5 matched recall gate |
| Scheduler | chunked prefill, `max_num_batched_tokens=4096`, initially `max_num_seqs=4` |
| Correctness | no speculative decoding in the first TurboQuant qualification |
| Training gate | complete rollout group, old logprobs, advantages, backward, optimizer step, adapter sync, and a later post-update rollout |
| Long-context gate | at least one separate rollout reaches beyond the old 8192-token boundary and preserves a correct Verifiers trace |

`max_num_batched_tokens=4096` is intentionally smaller than
`max_model_len=32768`. Chunked prefill bounds one scheduler iteration's prefill
work without reducing the length of an individual sequence.

TurboQuant K8V4 was evaluated because it compresses the rollout
engine's keys and values while leaving the BF16 model and LoRA training
representation unchanged. Qwen 3.5 0.8B has six full-attention layers among its
24 text layers; the remainder use linear attention. Its raw BF16 full-attention
KV payload is approximately 384 MiB for one fully occupied 32K sequence before
allocator and hybrid-cache alignment overhead. K8V4 reduces that variable
payload. In the vLLM 0.25.1 probe it exposed 776,722 cache tokens versus
291,328 for the matched normal-cache control, approximately 2.67 times the
capacity. The normal cache already reported 8.89 concurrent 32768-token
sequences at the selected 0.70 budget, so K8V4 is not necessary for the initial
four-sequence release profile.

K8V4 is not currently quality-qualified for Qwen 3.5 0.8B. The FP16 rollout
control recalled a code placed at the beginning of prompts with 8192, 16384,
24576, and 32700 input tokens. K8V4 failed all four matched cases, even though
it successfully generated short answers. This is a model/backend compatibility
failure, not an out-of-memory result. The 32K GRPO gate must therefore use the
normal FP16 KV cache unless a corrected vLLM build passes the same matrix.

The optimization has a strict boundary: it helps vLLM rollout memory only. It
does not compress actor activations, vocabulary projection work, gradients, or
optimizer state during backward. The fused PPO head, gradient checkpointing,
entropy chunking, phase sleep/offload, and adapter-first synchronization remain
required. A 32K engine initialization alone is therefore not qualification.

The 0.70 budget is rollout-phase capacity. In the qualified run, vLLM used a
3.01 GiB KV cache and total GPU usage reached about 7.2 GiB during generation.
After sleep, its worker retained only about 268 MiB of CUDA process state while
the actor ran. The fused actor peaked at 3.5327 GiB allocated.

`max_num_seqs` controls ready sequences scheduled by vLLM; it is not the number
of rollouts. Verifier environments and tool servers remain asynchronous in host
RAM. When one environment waits on a tool, vLLM can serve another ready turn.
The local host had roughly 40 GiB available RAM, so GPU scheduling—not verifier
process memory—was the relevant concurrency constraint.

## Native MTP rollout acceleration

The veRL adapter accepts native MTP through the existing backend-neutral
`InferenceBinding.engine.speculative_config` selection:

```yaml
engine:
  dtype: float16
  kv_cache_dtype: auto
  gpu_memory_utilization: 0.65
  max_model_len: 32768
  max_num_batched_tokens: 4096
  max_num_seqs: 4
  speculative_config:
    method: mtp
    num_speculative_tokens: 1
```

The adapter translates this to `actor_rollout_ref.model.mtp.enable=true`,
`enable_rollout=true`, `enable_train=false`, and veRL's native method/token
fields. Other speculative methods are rejected in this first veRL slice. The
canonical method is `mtp`; the older model-family-specific aliases are not
stored in framework bindings.

This is rollout acceleration, not MTP-loss training. The latter changes the
optimization objective and current veRL documentation supports it only through
Megatron/Megatron-Bridge. The Qwen 3.5 FSDP2 LoRA backend intentionally leaves
`enable_train=false`. Even so, `mtp.enable=true` loads the checkpoint's native
auxiliary head into the actor, so a full backward memory gate remains required.

Qwen 3.5 0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17` contains one native MTP layer and
the corresponding tensors. On the RTX 3070 Ti, vLLM 0.25.1 MTP-1:

- produced token-identical deterministic short outputs to the normal control;
- recalled the beginning-of-context code at 8192, 16384, 24576, and 32700
  input tokens;
- accepted 133 of 150 short-workload drafts (88.67%) and 30 of 32 long-context
  drafts (93.75%);
- reported capacity for 8.43 concurrent 32768-token requests;
- completed level-1 sleep plus separate weight and KV-cache wake;
- used about 501 MiB more GPU memory than the matched non-MTP rollout.

The measured short-probe generation time improved from 3.71 to 3.27 seconds,
but that includes JIT effects and is not a throughput claim. At a 0.70 rollout
budget MTP left only about 200 MiB free; the local candidate uses 0.65 and still
has enough cache capacity for four 32K sequences.

vLLM 0.25.1 exposes aggregate speculative counters through its metrics
snapshot, but does not populate the per-request
`request_spec_decode_stats` field that veRL currently checks. The standalone
probe records aggregate acceptance. Production qualification additionally
requires normalized run-level acceptance metrics or a verified replacement
for the missing per-request bridge.

The full lifecycle gate passed in
`artifacts/automationbench-verl-qwen35-08b-mtp-qualification-05`: two complete
Verifiers GRPO steps, 32 trajectories, non-zero gradients, checkpoint and
adapter export, adapter synchronization after both updates, and a second
post-update MTP rollout. The real colocated profile required
`gpu_memory_utilization=0.55` and 4096-token chunked prefill; 0.65 OOMed during
four-sequence prefill because the actor worker retained additional CUDA state.

Normalized acceptance telemetry is now available for synchronous GRPO. The
fork snapshots vLLM's aggregate Prometheus counters after each completed
rollout batch and before sleep, computes deltas from the previous snapshot,
and emits `rollout/spec_num_drafts`, `rollout/spec_num_draft_tokens`,
`rollout/spec_num_accepted_tokens`, `rollout/spec_accept_rate`, and
`rollout/spec_accept_length`. Qualification run `-06` measured 79.03%
acceptance before the first update and 81.67% after adapter synchronization.
These are batch-level metrics; vLLM 0.25.1 still does not expose per-request
attribution. The run evidence is retained in the tracked run/artifact backend,
not in a checked-out local `artifacts/` directory.

## Required memory optimizations

The qualified profile depends on all of the following behaviors:

1. **Phase-based sleep and offload.** vLLM releases weights and KV cache before
   actor computation; FSDP offloads actor parameters and optimizer state before
   rollout.
2. **Adapter-first synchronization.** The fork collects the LoRA adapter on CPU
   before waking rollout weights. Waking first exceeded the 8 GiB device during
   post-update synchronization.
3. **Dense-input chunked entropy.** veRL's FSDP
   `use_remove_padding=False` branch previously bypassed the configured chunked
   entropy function and attempted a multi-gigabyte softmax allocation.
4. **Qwen 3.5 fused PPO head.** `model.use_fused_kernels=true` with
   `fused_kernel_options.impl_backend=torch` computes vocabulary projection,
   selected-token logprobs, and entropy in token chunks. The unfused actor
   materialized full sequence-by-vocabulary logits and exhausted 8 GiB during
   backward.
5. **Disable nested FSDP entropy compilation.** Public
   `actor.use_torch_compile=false` does not disable
   `actor.fsdp_config.use_torch_compile`. Set the nested backend override
   explicitly; otherwise varying sequence lengths trigger repeated entropy
   recompilation and retain avoidable CUDA memory.
6. **Match timeout to queueing.** Turn count, token capacity, active vLLM
   sequences, and rollout wall-clock timeout must be tuned together. A
   300-second timeout with 16 rollouts and one active sequence caused later
   rollouts to time out before their first token.

These are backend implementation details, not new public GRPO concepts. The
portable selections remain model/context/turn/timeout/offload/concurrency
values. veRL-only fused-kernel, entropy, checkpoint-bucket, and worker settings
belong in `TrainingBinding.backend_options.hydra_overrides`.

The completed 8K qualification used the same actor and lifecycle settings shown
below. The rollout block is updated to the required 32K release target:

```yaml
rollout_binding:
  engine:
    max_model_len: 32768
    max_num_batched_tokens: 4096
    max_num_seqs: 4
    gpu_memory_utilization: 0.70
    enable_chunked_prefill: true
    dtype: float16
    kv_cache_dtype: auto

actor_rollout_ref:
  model:
    use_remove_padding: false
    enable_gradient_checkpointing: true
    use_fused_kernels: true
    fused_kernel_options:
      impl_backend: torch
    lora_rank: 8
    lora_alpha: 16
    target_modules: ".*language_model.*[.](o_proj|down_proj)$"
  actor:
    strategy: fsdp2
    entropy_from_logits_with_chunking: true
    entropy_from_logits_chunk_size: 256
    fsdp_config:
      param_offload: true
      optimizer_offload: true
      use_torch_compile: false
  rollout:
    gpu_memory_utilization: 0.70
    max_model_len: 32768
    max_num_batched_tokens: 4096
    max_num_seqs: 4
    dtype: float16
    free_cache_engine: true
    enable_sleep_mode: true
    engine_kwargs:
      vllm:
        kv_cache_dtype: auto
    agent:
      num_workers: 2
    checkpoint_engine:
      update_weights_bucket_megabytes: 16
reward:
  num_workers: 2
data:
  dataloader_num_workers: 1
```

veRL still calls the private actor fields `ppo_mini_batch_size` and
`ppo_micro_batch_size_per_gpu` for GRPO. The adapter derives them from prompt
groups and per-device batching; they are not exposed as public PPO concepts.

## AutomationBench environment profile

The pinned Zapier benchmark defaults to the `zapier` meta-tool interface:
`search_tools` discovers actions and `execute_tool` invokes one. The original
port accidentally made optional API mode the default. The environment package
now keeps three explicit modes:

- `zapier`: upstream-compatible discovery and execution meta-tools;
- `limited_zapier`: only the concrete task-required actions, used by the 0.8B
  qualification as a small-model curriculum;
- `api`: the separate REST-like search/fetch/base64 surface.

Tool mode is an environment selection, not a veRL override. The qualified run
uses the original task prompts unchanged, a non-thinking Qwen 3.5 renderer,
text-only vLLM initialization, and exact concrete tools for tasks 194 and 198.
It does not inject tool guidance into the prompt. Native Verifiers traces remain
the replay authority for rewards, tool calls, turn count, errors, and world
state.

The 50-turn and 32768-token values are capacities, not targets. The completed
checkpoint had an 8192-token capacity, and its trajectories used one to three
sampled turns and 120-1360 completion tokens.
Keeping realistic capacity still matters because tool responses expand later
prompts and because future tasks may require longer interaction.

## Diagnostic history

| Symptom | Cause | Resolution |
| --- | --- | --- |
| vLLM cannot allocate one 8192-token cache block | rollout budget was on the minimum boundary | use a measured rollout-phase budget rather than a tiny static reservation |
| zero trainable branches after several minutes | queued rollouts hit the old 300-second harness timeout | expose and raise the environment rollout timeout |
| 1.6-2.3 GiB allocation in old-logprob entropy | dense FSDP ignored entropy chunking | fork fix plus GPU regression test |
| 2 MiB failure with only 43-45 MiB free in actor update | unfused full-vocabulary logits consumed the device | enable Qwen 3.5 Torch fused PPO head |
| OOM while waking vLLM after optimizer step | adapter collection happened after rollout weights woke | stage adapter before weight wake |
| BF16 K8V4 warmup fails on Ampere | Triton cannot cast BF16 keys directly to FP8 in this stack | use an FP16 rollout copy; actor and LoRA remain BF16 |
| K8V4 misses beginning-of-context code while normal KV recalls it | Qwen 3.5 hybrid-model quality regression in the vLLM 0.25.1 TurboQuant path | keep K8V4 experimental and run the 32K GRPO gate with normal FP16 KV |

Do not reduce the required 32768 context merely to make a training run pass.
Qualification must retain a complete rollout batch, backward pass, optimizer
step, adapter synchronization, and a later rollout using updated weights.

## Fork-owned changes

The maintained CarbonTeq fork is published at immutable SAMPO candidate
revision
`553280b88afe4e7fbc4aefeff27bbf0a22e7c048` on
`carbonteq-ai/verl:main`. It is based on upstream veRL
commit `a35908ca3c9632859c58d6a2855d858918ae21dc`, contains the previously
published Qwen 3.5 QLoRA and runtime-counter work, and adds the SAMPO
extension. The SAMPO implementation itself is commit
`8a718e5be7a107587f63967336ece333a5c160e1`.

A published runtime-delta candidate has been reconstructed from that exact
published revision as commit
`8aa0b356d462568a92dedab642bba54aae37475d` on branch
`codex/runtime-release-qwen35`. It adds:

- compatible `orjson` and Transformers `<5.15` dependencies plus the exact
  CarbonTeq vLLM maintenance commit selected by the separately locked runtime;
- `verl/workers/engine_workers.py`: stage LoRA adapter tensors before waking
  colocated rollout weights;
- `verl/workers/engine/fsdp/transformer_impl.py`: honor chunked entropy for
  dense no-remove-padding inputs;
- `verl/models/transformers/qwen3_5.py`: retain attention dispatch through the
  generated FSDP2 wrapper;
- `verl/trainer/ppo/metric_utils.py`: retain the raw response-token total;
- the synchronous vLLM/trainer boundary: emit step-local MTP counters instead
  of process-lifetime totals.
- the core V1 replay buffer: enforce
  `algorithm.filter_groups.max_num_gen_batches` as a bounded candidate budget
  for DAPO and SAMPO instead of depending on an external recipe entrypoint;
- the V1 advantage path: retain an explicit SAMPO prompt-group identity, turn
  spans, anchor-state keys, and step rewards when reconstructing the optimizer
  batch from TransferQueue. Replay-key parsing remains a compatibility fallback,
  not the grouping contract.

The current V1 SAMPO release delta is published at
`5433d1c297870207c335108dcaac42f8da6f59bd` on branch
`codex/sampo-v1-metadata`.

The dense-distillation parent is published at
`c3f49b9117b882fa888e25e4a771461e13167848` on branch
`codex/distill-dense-teacher-logprobs`. In addition to the preceding maintained
delta, it aligns dense and jagged teacher log probabilities to response tokens
without reading unused TensorDict backing storage and treats fully masked
synthetic padding microbatches as zero-contribution batches. The paired runtime
kind preinstalls the Verifiers harness prerequisites so parallel rollout
workers never race while mutating the container package database.

The current immutable CarbonTeq veRL release candidate is `0.9.0.dev2`, source
commit `808923d487aa2c524fda02cf5289110541b4221f`, and tag
`carbonteq-v0.9.0.dev2`. Its wheel and source distribution were built manually
in the fork, published byte-for-byte to `carbonteq/dev` by Posttrain workflow
`31649999160`, and read back from that index. In addition to the dev1 release
delta, it preserves repaired three-dimensional position IDs through nested
TensorDict minibatch selection. The candidate has passed the focused Python
3.13 CPU suite, wheel install, focused Ruff/diff checks, and the bounded GPU
canary `verl-dev2-sampling-canary-20260813`. That run completed one optimizer
step and retained the adapter, summary, trace-sync receipt, and two native
traces while resolving the selected non-default sampling policy. Workflow
`31655220576` then promoted the same bytes server-side to `carbonteq/stable`
and verified their hashes and a clean install.

The previous dev1 matching kind image is retained as historical evidence at
`registry.lan/carbonteq/posttrain-kind-online-rl-verl-py313:0.9.0.dev1`.
Its OCI index digest is
`sha256:5684281f0f85ab741a156f1a92e11062a0e910f58aad03a488e8a2188db2421a`
(linux/amd64 manifest
`sha256:766ba146f5b1aa97d8557ca07b957e7c2ae3c57e786ec3e77ef84e88babdde07`).
Registry readback verified the Posttrain source revision
`c70e79ec9f07aa2076fe2534efea3bd998242987`, the veRL source revision
`a6fe39c22719ec981ed8544ad8feffd59995cc13`, and lock digest
`df7769f306ef8606fed37fd89c3fa2589ac72f39a45e4743816368a1acffe69f`.
The image passed its real Docker/Bake import smoke before publication.

The published SAMPO revision adds:

- `verl/trainer/ppo/core_algos.py` and
  `verl/trainer/ppo/ray_trainer.py`: register SAMPO's GiGPO-style hierarchical
  advantage estimator, validate token-aligned turn metadata, and combine the
  result with the existing GSPO actor loss.

The current delta passed 68 focused Python 3.13 CPU tests with 2 expected
skips, plus focused Ruff and diff checks. It contains no `runtime/`
environment, `sitecustomize`, or vLLM monkey-patch. The separately maintained
CarbonTeq vLLM fork owns the TurboQuant cache-reshape correction. TurboQuant
remains unqualified. Its DAPO/SAMPO-only and combined MTP matrix is deferred
from this release and must not be inferred from baseline or MTP-only results.

The fork's root `CARBONTEQ_FORK.md` is the maintained delta ledger. The
published revision is independently GPU-qualified for the baseline two-step
SAMPO configuration described below. MTP and TurboQuant variants remain
separate qualification gates.

### SAMPO operating configuration

The framework's typed SAMPO manifest maps to:

```text
algorithm.adv_estimator=sampo
algorithm.sampo.discount_gamma=<SAMPOSettings.discount_gamma>
algorithm.sampo.step_advantage_weight=<SAMPOSettings.step_advantage_weight>
algorithm.sampo.advantage_normalization=<mean|mean_std>
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
```

The Verifiers agent loop emits `sampo_prompt_group_id`, `sampo_turn_lengths`,
`sampo_turn_spans`, `sampo_anchor_state_keys`, and `sampo_step_rewards`. The
fork aligns the stable per-turn lengths to its materialized response mask and computes
episode-relative and anchor-relative advantages on the driver. The fork's core
V1 replay buffer supplies bounded replacement sampling for reward-constant
groups. A project training binding needs only the veRL interpreter, worktree,
and source revision; it no longer carries a second recipe checkout. Select
`backend_options.source_revision` as
`5433d1c297870207c335108dcaac42f8da6f59bd`.

The first dstack SAMPO attempt with the locked runtime reached advantage
computation on the RTX 4090 without the earlier CUDA illegal-memory-access
failure, then exposed a V1 TransferQueue omission: the agent loop had emitted
the SAMPO metadata, but `_compute_advantage()` had not selected it. The fork
regression above closes that source defect.

Run `verl-sampo-4090-extra-fields-20260729` retained all
eight expected Verifiers traces and reached the estimator, proving that rollout
completion and metadata propagation were complete. It then exposed a second
V1 boundary defect: SAMPO grouped by the materialized trajectory `uid` rather
than a stable prompt identity. A later key-parser run retained eight traces but
produced incomplete `[2, 2, 4]` groups. Those traces contained two dataset
`example_id` values with four trajectories each. Fork revision
`722595be16f9ac839d8f9c34efdb6bbff788b3ad` therefore consumes an explicit
`sampo_prompt_group_id` emitted from the dataset example and reconstructs
absolute optimizer spans from per-turn policy-token lengths plus the
materialized response mask. Its regressions cover opaque TransferQueue keys
and masks containing environment-token gaps.

Baseline qualification run `verl-sampo-96gb-vllm-fork-20260730` exercised
immutable veRL revision `5433d1c297870207c335108dcaac42f8da6f59bd` with the
CarbonTeq vLLM fork on the 96 GB RTX PRO worker. It completed two optimizer
steps with eager rollout, retained 28 native Verifiers traces, and published
the selected LoRA adapter, recovery checkpoint, retention manifest, and
summary. The run emitted episode- and turn-level advantages, a mean anchor
group size of 4, and sparse-reward projection metrics at both steps. Provider
exit code 0 reconciled with succeeded tracking and no missing artifact roles.

### Fleet scheduling

Revision-2 qualification selections use
`targets/carbonteq-cuda-24gb-plus`: one CUDA GPU with at least 24 GB and no
hostname constraint. dstack may therefore place the bounded 0.8B veRL jobs and
the colocated 2B teacher-score distillation job on either the RTX 4090 worker
or the RTX PRO worker. The original revision-1 selections remain pinned to the
96 GB worker so previously retained evidence keeps its exact target meaning.
Jobs whose measured peak or context requirement exceeds 24 GB must continue to
select a larger target rather than relying on a preferred hostname.

The bounded replacement behavior adapts the Apache-2.0 DAPO semantics from
`verl-project/verl-recipe` commit
`230ee612279d552a4f34ecbfab931c213abd514d`; the maintained implementation and
tests now live in the CarbonTeq veRL fork.

This composition follows the official ARL-Arena SAMPO extension at
`a25a2a229c85431b421ac785fa5f375a99b2072a`: hierarchical GiGPO advantages
plus GSPO policy loss. The framework retains one complete Verifiers trajectory
per optimizer row, so the sequence ratio covers all sampled policy tokens while
turn spans receive their own anchor-relative credit.

## Status and release gate

The adapter, deterministic translation, backend dispatch, Qwen 3.5 preflight,
portable Verifiers bridge, and custom agent-loop code have CPU unit coverage.
Qwen 3.5 0.8B GRPO's 8K lifecycle is GPU-qualified by a two-step
AutomationBench run on an 8 GiB device. It preserved 32 native traces and
produced reward variance in every
prompt group, completed tool-using multi-turn trajectories, reported gradient
norms `0.2127731889` and `0.2282306403`, synchronized updated weights before the
second rollout batch, and exported 30 changed LoRA B tensors containing 245,760
non-zero elements. Focused validation passed 19 framework/environment tests and
7 veRL lifecycle/FSDP tests.

Production qualification still requires the 32K GRPO gate using the qualified
normal FP16 KV path and complete required distillation telemetry. The two-step
distillation GPU execution and artifact gate is complete. The baseline
multi-turn SAMPO GPU gate is complete. MTP-only, TurboQuant-only, and combined
MTP plus TurboQuant DAPO/SAMPO configurations remain explicitly unqualified.
TurboQuant-only and combined configurations are deferred from this release;
MTP-only DAPO and SAMPO remain active release gates.
Do not describe the backend as production-qualified until both commands below
have been represented as catalog/work-package selections and completed:

1. A Qwen 3.5 GRPO run configured for a 32768-token context with normal FP16 KV
   cache that performs at least one optimizer step and then
   generates with synchronized updated weights. **The equivalent 8K lifecycle
   gate is complete; 32K remains open.**
2. A Qwen 3.5 student/teacher distillation run that performs at least one
   optimizer step over the exact student-generated token ids. **Run
   `verl-distill-shared-pool-retentionfix-20260730` completed two steps and
   retained 16 traces plus the adapter, summary, and retention manifest.
   Projection of required `scored_tokens` and `teacher_failures` telemetry
   remains open.**

Both runs must produce an exported model or LoRA adapter, a recovery checkpoint,
a normalized training summary, and native Verifiers traces. The execution plan
is [verl-qwen35-grpo-distillation.md](../../plan/verl-qwen35-grpo-distillation.md).

## References

- [vLLM release notes: TurboQuant hybrid-model and uniform-quantization support](https://github.com/vllm-project/vllm/releases)
- [TurboQuant paper, ICLR 2026](https://openreview.net/pdf/86df3c70aa9b7035c407e886e8238951a5d6ec23.pdf)
- [vLLM TurboQuant follow-up tracker](https://github.com/vllm-project/vllm/issues/40069)
