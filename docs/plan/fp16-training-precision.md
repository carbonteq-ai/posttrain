# FP16 trainer and rollout precision for TRL online RL

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Online reinforcement learning (RL) in this repository trains a policy with the TRL trainer while a colocated vLLM engine samples the rollouts. Both compute the probability of every sampled token, but with different kernels, and the rounding of their compute dtype makes the two log-probabilities disagree. In our bfloat16 (bf16) runs the mean per-token disagreement (TRL's `sampling/sampling_logp_difference/mean`, normalized to `train/rl/sampling_logp_delta_mean`) is 0.016 to 0.022 on LFM2.5 AutomationBench, and the truncated importance-sampling (IS) ratio, which corrects for the disagreement, hits its cap of 2.0 on every update. Qi et al. 2025, "Defeating the Training-Inference Mismatch via FP16" (arXiv 2510.26788), show that running both sides in float16 (fp16), with float32 master weights and dynamic loss scaling in the trainer, cuts this mismatch by roughly an order of magnitude and prevents the late collapse they observe in bf16, including for a LoRA GRPO run with token-level IS.

After this change a job can select the trainer precision (`backend_options.training_precision: bf16 | fp16` on a TRL training binding), the rollout precision (`engine.dtype: bfloat16 | float16 | float32` on the rollout inference binding), and float32 log-probabilities in the trainer (`backend_options.logits_float32`). Defaults reproduce the previous behaviour exactly. `posttrain job plan` prints the resolved precision, float16 runs record the loss scale and every skipped optimizer step, every online-RL run records the 99th percentile and the sequence-level size of the sampler-trainer gap, and the advisor rejects combinations that cannot run (for example vLLM float32 on Qwen3.5). An offline harness measures the full sampler-by-trainer precision matrix without training.

## Progress

- [x] (2026-09-28 09:20Z) Read AGENTS.md, the research notes, the TRL backend (`packages/train/src/posttrain/train/backends/trl/`), and the fork source at `../trl` commit 4950b99d (read only).
- [x] (2026-09-28 09:40Z) Switched the study model from LFM2.5-1.2B to Qwen3.5-0.8B at the user's request (cheaper; the goal is to build and qualify the support, not learning quality). No LFM2.5 measurements were taken in this branch.
- [x] (2026-09-28 10:05Z) Offline matrix, vLLM side: samples at bfloat16 and float16 on the posttrain-local image; vLLM float32 fails for Qwen3.5 (see Surprises); float32 reference sampled with Transformers generate instead.
- [x] (2026-09-28 10:30Z) Offline matrix, trainer side: Transformers scoring at bf16, fp16 and fp32 with hooks on all 383 module outputs; vLLM fp16 eager pass with hooks on all 393 module outputs (decode and prefill). No inf or NaN anywhere; fp16 preflight clean.
- [x] (2026-09-28 11:10Z) Added the fp32-LM-head variants (vLLM `hf_overrides.head_dtype=float32`, trainer fp32 head matmul) and a temperature 0.5 row.
- [x] (2026-09-28 12:30Z) Implemented `posttrain.train.precision`, binding validation, TRL wiring (load dtype, `fp16` trainer flag, loss-scale monitor, float32 logits hook, dtype forwarding), gap metrics, advisor rules, `job plan` output, and tests.
- [x] (2026-09-28 12:50Z) Added the two training-arm work packages and catalog entries; both resolve with `posttrain job plan`.
- [x] (2026-09-28 13:00Z) Validation ladder: ruff, format, lint-imports, full pytest (2020 passed, 30 skipped; 2022 after the workstation preparation), pyright unchanged against the baseline, `git diff --check`.
- [ ] Training arms 1 and 4 (below). Blocked by request of the lead: the trainer image lacks the flash-linear-attention and causal-conv1d kernels, so the Qwen3.5 trainer runs Transformers' torch fallbacks while vLLM runs the Triton kernels. Another agent is adding the kernels to the runtime image; run the arms on that image.
- [ ] Re-run the offline matrix trainer side on the image with the kernels (the numbers below use the torch fallback, which is what the trainer used until now) and confirm the rewritten harness scripts reproduce them.
- [ ] Record arm results, decide, and write Outcomes.
- [x] (2026-09-28 15:40Z) Workstation preparation (not run): the harness in `scripts/qualification/precision_mismatch/` takes the model, revision, prompt source, chat template, generation budget and sampling as arguments, adds a top-p-renormalized trainer column and a one-command driver (`run_matrix.sh`); `export_engine.py` and `bench_rollout_engine.py` replay a rollout binding's exact vLLM engine at a chosen dtype; LFM2.5-2.6B prompts render from retained AutomationBench SAMPO traces.
- [x] (2026-09-28 15:40Z) Read the vLLM fork's SM120 and rollout kernels for float16 support (table in Artifacts); added `VLLM_FLOAT16_BF16_ONLY_KERNEL` (warning) and `VLLM_FLOAT32_KERNEL_UNSUPPORTED` (error).
- [x] (2026-09-28 15:40Z) The sampler-gap p99 and sequence statistics are now pooled over the whole update (they were averaged per call, which is per micro-batch when the importance ratio comes from the training forward, as in the 2.6B runs).
- [x] (2026-09-28 15:40Z) LFM2.5-2.6B canary work packages (fp16 and its bf16 twin, five updates from the fixed-tools continuation) plan cleanly; runbook below.
- [x] (2026-09-28 17:10Z) Reproduced the Qwen3.5-0.8B cells with the rewritten `gen_vllm.py`/`score_hf.py` on the kernel image `posttrain-kind-online-rl-trl-py312:qwen-kernels-local-cc1d` (fla-core 0.5.2, causal-conv1d 1.7.0; branch codex/qwen-fast-kernels). vLLM samples are bit-identical to the committed ones (64/64 completions, log-probabilities equal); with the fast trainer kernels bf16/bf16 is 0.0134/0.124 and fp16/fp16 0.0020/0.016 (mean/p99), against 0.0137/0.124 and 0.0019/0.015 with the torch fallback: a 6.7x cut instead of 7.2x, same conclusion. No "fast path is not available" warning.
- [ ] Training arms on the kernel image: waiting for the published kind images with fla-core and causal-conv1d (see Decision Log, 2026-09-28, lead); then launch from a clean worktree merging both branches. `posttrain job run` builds the actual-job image only on a registry-hosted, digest-pinned kind image whose lock matches the published manifest (see Surprises); the kernel image exists only in the local Docker daemon.
- [ ] Workstation session: preflight, throughput, canaries (runbook below), when the 100-update continuation has finished.

## Surprises & Discoveries

- Observation: vLLM cannot run Qwen3.5 in float32. The chunked Gated-DeltaNet prefill kernel asserts at the first prefill; FlashInfer's GDN backend is not available on sm86 either.
  Evidence (vllm 0.29.1.dev4, RTX 3070 Ti): `AssertionError: ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.` from `vllm/third_party/flash_linear_attention/ops/chunk.py`. The advisor now fails such a binding at plan time (`VLLM_FLOAT32_UNSUPPORTED_FOR_GATED_DELTANET`). Arm 3 of the paper's ablation (bf16 trainer, fp32 vLLM) is therefore impossible for Qwen3.5 on this stack; it was dropped by the user in any case.

- Observation: Qwen3.5-0.8B has small activations and runs cleanly in float16 on both sides, despite public reports of NaN logits for bf16-native Gated-DeltaNet models forced to fp16.
  Evidence: largest finite |module output| over 64 prompts (22-25K generated tokens, up to 3.6K-token contexts) is 82.5 (`model.layers.4.linear_attn.in_proj_qkv`, token 0) in bf16, fp16 and fp32 alike, against the fp16 maximum of 65,504 (about 800x headroom); zero inf or NaN in any of 383 Transformers or 393 vLLM hooked outputs (vLLM decode and prefill), zero non-finite sampler log-probabilities. vLLM keeps the Gated-DeltaNet recurrent state in float32 (`mamba_ssm_dtype: float32` in the checkpoint config).

- Observation: upcasting model-dtype logits to float32 before the log-softmax (the literal "fp32 logits" fix) changes almost nothing; an fp32 LM head matmul on both sides removes about a third of the bf16 gap; fp16 on both sides removes about 86%.
  Evidence (trainer bf16, sampler vLLM bf16, mean per-token |gap|): log-softmax in bf16 0.0138; logits upcast to fp32 0.0137; fp32 head on both sides 0.0095; fp16 on both sides 0.0019. The rounding of the log-probability itself is small because most sampled tokens have log-probabilities near zero, where bf16 spacing is fine; the logits are already rounded to bf16 before any upcast.

- Observation: a mixed configuration buys almost nothing; the side left in bf16 dominates.
  Evidence: vLLM fp16 with a bf16 trainer 0.0133; vLLM bf16 with an fp16 trainer 0.0112; both fp16 0.0019. This matches Qi et al.'s claim that both engines must change.

- Observation: in vLLM float16 the fused CUDA Gated-DeltaNet decode kernel is not used (it requires bf16), yet generation throughput is unchanged on this workload.
  Evidence: log line "Falling back to the Triton GDN decode path: the fused CUDA kernel requires a BF16 GDN model"; 64 prompts with CUDA graphs: bf16 1,681 tok/s, fp16 1,700 tok/s (T=1.0); bf16 1,672 tok/s, fp16 1,699 tok/s (T=0.5).

- Observation: the Transformers trainer runs Qwen3.5's Gated-DeltaNet and short convolution with torch fallbacks in the current runtime image ("The fast path is not available ... Falling back to torch implementation"), because flash-linear-attention and causal-conv1d are not installed, while vLLM uses Triton kernels. The fallback computes the recurrence in float32 internally (`torch_chunk_gated_delta_rule` casts q, k, v, beta and g to float32), so it is slow but numerically conservative. The offline trainer numbers were measured with this fallback; actor step time in the arms should be compared on the same image for both arms.

- Observation: the colocated TRL rollout ignored `engine.dtype` before this change (only a TurboQuant KV cache forced float16), and the recorded `rollout_precision` attribute took an explicit dtype or the checkpoint precision, so TurboQuant runs were recorded as bf16 although vLLM ran float16. No existing colocated TRL rollout binding sets `dtype` (catalog scan: only veRL TurboQuant rollouts do, with float16), so forwarding it changes no recorded run.

- Observation: plain GRPO settings default to sequence-level truncated IS with bounds 0.1 to 3.0; the arms select token-level truncation at cap 2.0 to match the paper's GRPO-Token-TIS LoRA case and our production runs.

- Observation: `pyright` over the whole repository reports 196 errors on the unmodified base commit in this environment (mostly "unknown import symbol" across apps and packages); this change adds none (the error list is identical before and after).

- Observation: vLLM's processed log-probabilities are renormalized over the top-k/top-p nucleus, and TRL's trainer log-probabilities are not. With top_p 0.95 (the LFM2.5 AutomationBench runs) every sampled token's sampler log-probability exceeds the trainer's by the log of the nucleus mass, independently of precision; the production 0.016 to 0.022 "gap" includes this offset.
  Evidence: vllm fork f09e4479 `vllm/v1/sample/ops/topk_topp_sampler.py` computes `logits.log_softmax(...)` after `apply_top_k_top_p` when `logprobs_mode == "processed_logprobs"` (TRL sets that mode); the fork's `_get_per_token_logps_and_entropies` divides by temperature only. The harness's `nucleus_logps` column (trainer logits with the same mask) separates the offset from precision; the Qwen3.5-0.8B matrix used top_p 1.0, so it is unaffected.

- Observation: rendering a retained AutomationBench trace with the LFM2.5-2.6B tokenizer's chat template reproduces the recorded prompt length to within about 1 to 8% (for example 3,247 against 3,213 tokens at the first turn, 7,396 against 8,013 at a later one), not token for token: the production renderer (`lfm2.5-tools-thinking@2`, carbonteq-renderers) formats tools and earlier reasoning differently. The prompts are real rollout states of the environment in the model's own format, which is what the precision matrix needs; they are not the exact tokens the rollout saw.

- Observation: in the fork's rollout kernels, float16 only leaves bfloat16-tuned code in batch-invariant GEMMs on SM120 (tuned shape table and generic rule are bf16-only; float16 falls back to the untuned default persistent configuration), in Qwen3.5's fused CUDA Gated-DeltaNet decode (bf16-only; Triton fallback), and in DSpark's NVFP4 gathered-bias kernel (bf16-only; used only for an NVFP4-quantized drafter). The c144 LFM2.5-2.6B rollout engine (FlashAttention 2, DSpark, LoRA, no batch invariance) uses none of them, so float16 should cost no kernel there; float32 is impossible on it (FlashAttention 2 and the Punica LoRA shrink kernel reject float32).

- Observation: the local provider cannot be pointed at a kind image that exists only in the local Docker daemon, and a merged worktree cannot pack either. Every kind image is a `RuntimeImageRef`, which requires an `@sha256:` registry digest (`packages/execution/src/posttrain/execution/contracts.py`); `ensure_kind_image_ready` inspects it in its registry; the actual-job build runs `FROM ${POSTTRAIN_KIND_IMAGE}` on the docker-container builder `posttrain-builder`, which pulls from registries; and the kind lock must match the published manifest. A local integration worktree merging codex/precision-fp16 and codex/qwen-fast-kernels (`/home/hammad/projects/rl-precision-integration`, merge a1edaaf4, not pushed) fails at `posttrain job pack` with `job-kind constraint profile changed after configuration load: online-rl-trl-py312`, by design ("a lock edited without republishing cannot reach this point"). The designed route for a candidate runtime is a project-scoped `[registry]` table in `apps/lab/.posttrain/state/execution.toml` (git-ignored, mode 0600) with `kind_images.online-rl-trl-py312 = "<registry>/...@sha256:..."` and `constraint_profiles.online-rl-trl-py312` pointing at the kernels branch lock, which needs the image in a registry.
  Evidence: `posttrain job pack .posttrain/work_packages/qwen08b_gsm8k_grpo_precision_bf16_local.yaml` in the integration worktree: `Error: job-kind constraint profile changed after configuration load: online-rl-trl-py312`; `docker images --digests` shows no repository digest for the kernel image.

## Decision Log

- Decision: Study model Qwen3.5-0.8B on GSM8K GRPO instead of LFM2.5-1.2B AutomationBench SAMPO.
  Rationale: requested by the user; cheaper on the 8 GB card and single-turn, so arms measure precision rather than environment variance. Qwen3.5's Gated-DeltaNet layers make the fp16 preflight a primary result.
  Date/Author: 2026-09-28, user via lead.

- Decision: Training arms are arm 1 (trainer bf16, vLLM bf16) and arm 4 (trainer fp16, vLLM fp16) of Qi et al. 2025, Section 4.4, Figure 5, run longer rather than all four arms. The offline matrix covers the mixed and float32 combinations without training.
  Rationale: user decision. In the paper arm 1 is the baseline that collapses after about 600 steps; arm 4 is the recommendation (lowest mismatch, no speed loss). Arm 2 (bf16 trainer, fp16 vLLM) and arm 3 (bf16 trainer, fp32 vLLM, "fully stable but about 3x slower") are represented by offline rows; arm 3 cannot run for Qwen3.5 in vLLM at all.
  Date/Author: 2026-09-28, user via lead.

- Decision: `training_precision` values are `bf16` and `fp16`, default `bf16`, and fp16 requires a LoRA update on the TRL backend; any other backend rejects a non-default value when the catalog loads.
  Rationale: dynamic loss scaling must step float32 master weights. PEFT keeps LoRA adapters in float32 over a float16 base (verified for fresh adapters, resumed adapters with `is_trainable=True`, and TRL's frozen `ref` copy); full-parameter updates would step float16 weights and QLoRA computes in bf16. veRL has its own precision configuration and does not read this option, so accepting it there would misstate what ran.
  Date/Author: 2026-09-28, Claude.

- Decision: `logits_float32` is an opt-in option (default false) that casts the LM head's output to float32 before the log-softmax, as the task specified; it is not the default.
  Rationale: offline it changes the bf16 gap by under 1% (0.0138 to 0.0137) and the fp16 gap not at all, so it does not justify changing the semantics of existing runs (IS ratios and the logged entropy would shift slightly). It makes the trainer follow vLLM's sampler pipeline exactly (model-dtype logits taken to float32 before the log-softmax) and is cheap with `logits_chunk_size`. The measured lever is an fp32 LM head matmul on both sides (-31%), which needs the TRL fork's `cast_lm_head_to_fp32` plus vLLM `hf_overrides.head_dtype`; that is a follow-up (see Outcomes).
  Date/Author: 2026-09-28, Claude.

- Decision: `engine.dtype` accepts `bfloat16`, `float16`, `float32` for TRL rollouts and is forwarded to vLLM; a TurboQuant KV cache keeps implying float16 and rejects any other explicit dtype. The advisor's `TRL_ROLLOUT_DTYPE_NOT_APPLIED` warning is removed (the value is now applied), `VLLM_FLOAT16_ON_BF16_CHECKPOINT` no longer fires when the TRL trainer is also fp16 (unified fp16), and two findings are added: `FP16_TRAINER_WITH_NON_FP16_ROLLOUT` (warning) and `VLLM_FLOAT32_UNSUPPORTED_FOR_GATED_DELTANET` (error, any vLLM binding of a qwen3.5 model). `TRL_PRECISION_UNQUALIFIED_FOR_JOB` (error) rejects the trainer options on non-online-RL jobs. The serving calculator still does not suggest a dtype for colocated TRL rollouts, because the rollout dtype is chosen together with the trainer precision.
  Rationale: fail at plan time with a precise reason instead of at rollout start; keep existing plans' findings unchanged.
  Date/Author: 2026-09-28, Claude.

- Decision: A skipped optimizer step's infinite gradient norm is dropped from that step's metrics instead of failing the run; any non-finite gradient norm on a step that was not skipped still fails it.
  Rationale: Transformers logs the unscaled gradient norm, which is infinite exactly when the loss scaler skips the step. The GRPO metric normalizer rejects non-finite values, so without this every early scaler overflow would abort the job. The skip itself is recorded as `train/optimizer_step_skipped` and an `optimizer_step_skipped` event.
  Date/Author: 2026-09-28, Claude.

- Decision: Add `train/rl/sampling_logp_delta_p99` and `train/rl/sampling_sequence_logp_delta_abs_mean` to every TRL online-RL run through the Posttrain trainer subclass (`policy_telemetry.actor_update_trainer_type`), and `train/loss_scale`, `train/optimizer_step_skipped`, `train/optimizer_steps_skipped` to fp16 runs. The canonical baseline is not amended: `docs/post-training/06-observation-and-lineage.md` lists the GRPO metrics as a non-exhaustive "includes" set, and these follow the existing `train/rl/sampling_logp_delta_*` names.
  Rationale: TRL reports only the mean and maximum gap; the tail and the sequence-level sum are what the IS cap and SAMPO's sequence IS act on.
  Date/Author: 2026-09-28, Claude.

- Decision: No change to the TRL fork in this branch.
  Rationale: everything needed is reachable from Posttrain's config and trainer subclass. Fork follow-ups are listed in Outcomes.
  Date/Author: 2026-09-28, Claude.

- Decision: Do not start training arms until the runtime image has flash-linear-attention and causal-conv1d.
  Rationale: requested by the lead; with torch fallbacks in the trainer and Triton kernels in vLLM the bf16-versus-fp16 comparison is confounded by kernel differences and the actor step is slow.
  Date/Author: 2026-09-28, lead.

- Decision: Measure rollout throughput with a script that replays the rollout binding's exact engine (`export_engine.py` uses the TRL backend's own `vllm_rollout_options`; `bench_rollout_engine.py` builds the same `LLM` with LoRA, DSpark, FlashAttention 2, KV cache and `max_num_seqs` 144, and samples 24 real prompts x 6), not a `serve.benchmark` work package.
  Rationale: `serve.benchmark` runs synthetic token-shape workloads through the serving engine configuration, which has no LoRA requests or colocated rollout keys, and random prompts would misstate DSpark acceptance; the rollout speed that matters is the collection with real AutomationBench states.
  Date/Author: 2026-09-28, Claude.

- Decision: The LFM2.5-2.6B canaries are two new work packages with new catalog ids (settings cut to five updates with the continuation's five-update warmup, an fp16 copy of the g144-w12 training binding and an fp16 copy of the c144 rollout binding); the bf16 twin reuses the continuation's training and rollout bindings unchanged. Both start from the same continuation checkpoint with `--model-from-run ... --model-checkpoint-step`, run the same day, one after the other.
  Rationale: identical data, settings and starting adapter isolate precision; a same-day twin controls for environment and service drift, which a comparison with the continuation's own first updates would not.
  Date/Author: 2026-09-28, lead and Claude.

- Decision: Pool the gap statistics per logged update instead of appending one value per call.
  Rationale: with `importance_sampling_from_training_logps` (the 2.6B bindings) TRL computes the importance ratio per micro-batch of one episode, so the per-call p99 was an average of per-episode percentiles.
  Date/Author: 2026-09-28, Claude.

- Decision: Do not run the training arms on the kernel image through any of the four local workarounds considered: (1) push the local kernel image to a scratch registry.lan repository and select it with a project `[registry]` override; (2) pull `registry:2` for a throwaway loopback registry and do the same; (3) pack on the published kind image and overlay the fast-kernel packages onto the `posttrain-local:<tag>` job image before `job run` (the image labels would misstate the software and the tag could be reused by later packs); (4) run both arms now on the published image with the torch fallback kernels. Instead the kernels agent publishes causal-conv1d and the kernel job-kind images properly (user-authorized); once the published images exist and codex/qwen-fast-kernels has adopted causal-conv1d, run arms 1 and 4 on the published image from a clean worktree merging both branches. No GPU work until then.
  Rationale: provenance stays exact (the run records the published kind digest it actually ran on), and nothing is pushed or downloaded outside the release path.
  Date/Author: 2026-09-28, lead.

## Outcomes & Retrospective

Interim (2026-09-28, before the training arms). The precision options, metrics, advisor rules and plan display are implemented and tested with defaults unchanged. Offline on Qwen3.5-0.8B, fp16 on both sides cuts the mean per-token sampler-trainer gap 7.2x (0.0137 to 0.0019 at T=1.0; 8.3x at T=0.5), the 99th percentile 8x (0.124 to 0.015) and the mean absolute sequence log-ratio 6.5x (0.39 to 0.06), with a clean fp16 preflight. The fp32-logit upcast alone closes under 1% of the bf16 gap; an fp32 LM head on both sides closes about 31%.

Follow-ups: (1) TRL fork: an option to compute the trainer LM head with float32 accumulation without a float32 weight copy (`torch.mm(..., out_dtype=torch.float32)`), forwarded to vLLM as `hf_overrides.head_dtype`, and immune to autocast (fork `cast_lm_head_to_fp32` keeps a float32 copy of the tied embedding and does not forward the head dtype to vLLM); (2) TRL fork: return float32 log-probabilities from `selective_log_softmax` for low-precision logits; (3) gather the new gap statistics across ranks for multi-GPU runs (they are per-rank today; the qualified runs are single-GPU); (4) veRL: map `training_precision` to its FSDP mixed-precision settings if fp16 is adopted there.

## Context and Orientation

The framework is a Python 3.12+ `uv` workspace. Training capabilities live in `packages/train` (import name `posttrain.train`). The TRL backend is private under `packages/train/src/posttrain/train/backends/trl/`: `common.py` loads the model (`load_trainable_model`), builds Transformers trainer arguments (`trainer_arguments`) and translates rollout-engine selections into vLLM constructor arguments (`vllm_rollout_options`); `policy_config.py` builds the TRL `GRPOConfig` for online RL (GRPO, SAMPO, GDPO, CAPO); `policy_optimization.py` runs it; `policy_telemetry.py` holds the Posttrain subclass of TRL's `GRPOTrainer` that adds telemetry. TRL itself is the CarbonTeq fork pinned at `trl==1.12.0.post10` (source commit 4950b99d in `../trl`).

A training binding (`TrainingBinding` in `packages/train/src/posttrain/train/bindings.py`) selects the backend, renderer, parameter update (LoRA, QLoRA or full) and free-form `backend_options`. A rollout inference binding (`InferenceBinding` in `packages/common/src/posttrain/common/selections.py`) selects the vLLM engine options (`engine`) and sampling. Catalog YAML under `packages/catalog/src/posttrain/catalog/base/` and `apps/lab/.posttrain/catalog/` declares both; work packages under `apps/lab/.posttrain/work_packages/` combine them. The settings advisor (`packages/advisor`) checks a resolved job's selections and reports findings; an `error` finding fails `posttrain job plan` unless the binding acknowledges it.

Terms. bf16 (bfloat16) has 8 exponent bits and 7 mantissa bits: the float32 range with coarse precision. fp16 (float16) has 5 exponent bits and 10 mantissa bits: 8x finer precision but a maximum of 65,504, so large activations can overflow to infinity. Dynamic loss scaling multiplies the loss by a large factor before the backward pass so small float16 gradients do not round to zero, divides the gradients by it before the optimizer step, and skips the step and halves the factor when a gradient overflows. Master weights are the float32 copies the optimizer updates; with LoRA they are the adapter matrices. The sampler-trainer gap is |log p_vLLM(token) - log p_trainer(token)| for each sampled token. Truncated IS multiplies each token's loss by min(p_trainer/p_vLLM, cap). Gated-DeltaNet is Qwen3.5's linear-attention layer (three of every four layers), a recurrence with a matrix state.

## Plan of Work

Milestone 1 (done), offline mismatch matrix and fp16 preflight. `scripts/qualification/precision_mismatch/` holds a harness that runs inside the posttrain-local runtime image: `prep.py` builds 48 GSM8K prompts and 16 AutomationBench agent contexts rendered with the Qwen3.5 chat template (reasoning off); `gen_vllm.py` samples them with vLLM at one dtype (optionally with an fp32 head, another temperature, or eager hooks); `score_hf.py` scores every sample set with Transformers at one dtype with hooks on every module, or samples the fp32 reference with Transformers generate; `analyze.py` prints the matrix.

Milestone 2 (done), precision selections. `packages/train/src/posttrain/train/precision.py` resolves `training_precision`, `logits_float32` and the rollout dtype into `ResolvedPrecision`. `TrainingBinding.__post_init__` validates them. `common.py` loads the base in float16 for fp16, sets `fp16`/`bf16` in the trainer arguments and forwards `engine.dtype`. `precision_runtime.py` provides the loss-scale monitor and callback, the float32 logits hook and the float32 trainable-parameter check; `policy_optimization.py` wires them. `policy_telemetry.py` adds the gap tail and sequence metrics; `grpo_observations.py` maps them. SFT, DPO and distillation reject non-default precision. The advisor rules and `posttrain job plan` output are updated.

Milestone 3 (pending), training arms on the local 8 GB card, one at a time, from a clean detached worktree of the committed branch: `apps/lab/.posttrain/work_packages/qwen08b_gsm8k_grpo_precision_bf16_local.yaml` (arm 1) and `..._fp16_local.yaml` (arm 4). Both use settings `qwen3.5-0.8b/gsm8k-grpo-precision-v1` (100 updates of 4 prompts x 8 generations, LoRA r8/alpha 16 all-linear, learning rate 4e-5 constant, beta 0.005, token-truncated IS at cap 2.0, seed 1729, shuffled GSM8K train prompts, T=1.0, top_p=1.0, 384 completion tokens) and differ only in `training_precision` and `engine.dtype`. Pass criteria for arm 4: clean preflight (met offline), mean gap at least 3x below arm 1, skipped steps under 2% after step 10, no NaN, and reward at matched steps not worse than arm 1 beyond noise. Short runs mainly show mismatch and speed; the paper's bf16 collapse appeared after hundreds of steps.

## Concrete Steps

Offline matrix (working directory: repository root; output outside /tmp; one GPU job at a time; check `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty first):

    export IMAGE=<posttrain runtime image with vLLM> OUT=$HOME/precision-mismatch/qwen08 TRACES=<trace JSON dir>
    cd scripts/qualification/precision_mismatch
    MODEL=Qwen/Qwen3.5-0.8B REVISION=2fc06364715b967f1860aea9cf38778875588b17 \
    PREP_ARGS="--source gsm8k --num-prompts 48 --max-tokens 1024 --template-kwargs {\"enable_thinking\":false}" \
    VLLM_ARGS="--language-model-only" HOOK_ARGS="--gpu-memory-utilization 0.45 --max-num-batched-tokens 512" \
    ./run_matrix.sh

The committed Qwen numbers also used 16 agent contexts appended with `prep.py --source traces --append` (AutomationBench traces rendered by the Qwen3.5 template, 1,500 to 3,072 prompt tokens, 512 generated tokens), a temperature 0.5 pair (`gen_vllm.py ... --temperature 0.5 --tag bfloat16t05`) and an fp32-head row (`--head32 --tag bfloat16h32`). Individual steps run as `./run.sh <script> <arguments>`; every script documents its arguments with `--help`.

Plan the arms:

    cd apps/lab
    uv run posttrain job plan .posttrain/work_packages/qwen08b_gsm8k_grpo_precision_fp16_local.yaml

    Precision: trainer fp16 (base weights float16, dynamic loss scaling; log-probs from float16 logits); rollout vLLM float16 (binding)

Run an arm (after the image with the Gated-DeltaNet kernels is available; from a clean detached worktree of the committed branch because packing includes the working tree):

    git worktree add --detach ../rl-precision-run codex/precision-fp16
    cd ../rl-precision-run/apps/lab
    uv run posttrain job run .posttrain/work_packages/qwen08b_gsm8k_grpo_precision_bf16_local.yaml --provider local

## Workstation runbook

Run this only when the RTX PRO 6000 workstation is free: no active Posttrain run holds it (`uv run posttrain workers` from `apps/lab`) and `nvidia-smi --query-compute-apps=pid --format=csv,noheader` on it prints nothing. One GPU job at a time; each step below finishes before the next starts. Keep outputs outside /tmp (below, `~/precision` on the workstation). Estimated time: preflight 1 to 2 hours, throughput 30 minutes, each canary about as long as five continuation updates plus startup.

Step 0, runtime and inputs. On the workstation pick the runtime image of the latest LFM2.5-2.6B SAMPO job, so vLLM (fork f09e4479 with the SM120 kernels), Transformers and TRL match training:

    docker images --filter label=org.carbonteq.posttrain.job-kind=train.sampo --format '{{.Repository}}:{{.Tag}} {{.CreatedAt}}' | head -3
    export IMAGE=<newest of those>  OUT=~/precision/lfm26  TRACES=~/precision/traces
    mkdir -p $OUT $TRACES

Copy the harness (`scripts/qualification/precision_mismatch/` from the branch head) to `~/precision/harness/`, and AutomationBench SAMPO traces (Observatory trace JSON, `/api/v1/runs/<run>/traces/<id>`; 211 from the continuation r1 are in the session scratchpad `trace_audit/details/`) to `$TRACES`. The Hugging Face cache must hold `LiquidAI/LFM2.5-2.6B` at 654f9463 (and the DSpark drafter at 458cedab for step 2); otherwise set `HF_HUB_OFFLINE=0` for the first run.

Step 1, preflight (offline matrix, fp16 overflow check):

    cd ~/precision/harness
    MODEL=LiquidAI/LFM2.5-2.6B REVISION=654f9463ce32b05d0429d76fe1f580b27d4c1ac0 \
    PREP_ARGS="--source traces --num-prompts 64 --max-tokens 4096 --min-prompt-tokens 2048 --max-prompt-tokens 20480" \
    TEMP=0.5 TOP_P=0.95 \
    VLLM_ARGS="--max-model-len 24576 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85" \
    HOOK_ARGS="--max-num-batched-tokens 4096 --gpu-memory-utilization 0.6" \
    ./run_matrix.sh

It writes `$OUT/report.txt` and `report.json`: the matrix (vLLM bf16/fp16/fp32 rows by Transformers bf16/fp16/fp32 columns, four trainer log-probability variants), per-sampler throughput, any dtype vLLM rejected (`failed_<dtype>.json`; the fp32 row then comes from Transformers generate), and the largest activation and inf/NaN counts per module for the Transformers scorers and vLLM fp16 decode and prefill. Pass: no inf or NaN in any fp16 module output or sampler log-probability; largest fp16 activation below 8,188 (8x headroom); fp16/fp16 mean gap (`nucleus_logps` column) at least 3x below bf16/bf16. Record the `fp32_logps` minus `nucleus_logps` difference as the top-p offset. Fail: stop, record the module, layer and token position from `report.json` (`first_bad_pos`), and do not run the fp16 canary.

Step 2, rollout throughput with the c144 engine. On the machine with the repository, export the engine and copy the JSON to `$OUT`:

    uv run python scripts/qualification/precision_mismatch/export_engine.py --project apps/lab \
      --inference inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c144-4k-t05@1 \
      --training training/lfm2.5-2.6b-trl-lora-automationbench-local-g144-w12@1 --out engine_c144.json

Copy the continuation's adapter at the canary's start step (`uv run posttrain run checkpoint show lfm26-sampo-cont40-fixed-tools-20260928-r1 <step>` names the model view) to `$OUT/adapter`. Then on the workstation, bf16 first:

    for dtype in bfloat16 float16; do
      ./run.sh bench_rollout_engine.py --engine /work/engine_c144.json --dtype $dtype --prompts /work/prompts.json \
        --num-prompts 24 --samples-per-prompt 6 --lora /work/adapter --out /work/bench_$dtype.json
    done

Pass: float16 `generated_tokens_per_second` at least 0.9x bfloat16, similar DSpark acceptance (`spec_acceptance_rate`) and prefix-cache hit rate, zero non-finite log-probabilities. float32 is not attempted: FlashAttention 2 and the Punica LoRA kernel reject it (`posttrain job plan` reports `VLLM_FLOAT32_KERNEL_UNSUPPORTED`).

Step 3, canaries, from a clean detached worktree of the branch head on the submitting machine (packing includes the working tree). Choose `<step>`, a checkpoint of the continuation that exists (`uv run posttrain run checkpoint list lfm26-sampo-cont40-fixed-tools-20260928-r1`; checkpoints are every 10 updates, the last three retained), and use the same step for both:

    git worktree add --detach ../rl-precision-canary codex/precision-fp16 && cd ../rl-precision-canary/apps/lab
    uv run posttrain job plan .posttrain/work_packages/lfm26_automationbench_sampo_turns_precision_bf16_canary.yaml
    uv run posttrain job run .posttrain/work_packages/lfm26_automationbench_sampo_turns_precision_bf16_canary.yaml \
      --provider dstack --model-from-run lfm26-sampo-cont40-fixed-tools-20260928-r1 --model-checkpoint-step <step>
    uv run posttrain run wait <bf16 run id>
    uv run posttrain job run .posttrain/work_packages/lfm26_automationbench_sampo_turns_precision_fp16_canary.yaml \
      --provider dstack --model-from-run lfm26-sampo-cont40-fixed-tools-20260928-r1 --model-checkpoint-step <step>
    uv run posttrain run wait <fp16 run id>

`job plan` prints `Precision: trainer fp16 (base weights float16, dynamic loss scaling; ...); rollout vLLM float16 (binding)` for the fp16 canary and the bf16 line for the twin. Stop a run with `uv run posttrain run cancel <run id>`; the provider stops it and tracking finalizes.

Per update, compare the two runs in Observatory: `train/rl/sampling_logp_delta_mean`, `..._p99`, `..._max`, `train/rl/sampling_sequence_logp_delta_abs_mean`, `train/rl/importance_sampling_ratio_clamped_fraction` and `..._max`, `train/grad_norm`, `train/rl/entropy`, `train/rl/kl`, `train/rl/reward_mean`, `train/rl/rollout_tokens_per_second`, `train/rl/time/rollout_seconds`, `train/rl/time/actor_update_seconds`, `train/peak_gpu_memory_gib`, and for fp16 `train/loss_scale`, `train/optimizer_step_skipped`, `train/optimizer_steps_skipped` (and any `optimizer_step_skipped` events). Pass for fp16: finishes with no NaN; at most one skipped step and never two in a row; mean gap at least 3x below the twin's at every update, or, when step 1 showed the top-p offset (`fp32_logps` minus `nucleus_logps`) to be a large part of the bf16 gap, the gap minus that offset at least 3x below the twin's (the trainer metric includes the offset, which precision cannot remove); clamped fraction below the twin's; reward, grad norm, entropy and KL within the twin's range; rollout tokens per second within 10% and actor update time within 15% of the twin's; peak memory not above the twin's by more than 10%. Any NaN, repeated skips or a collapsing loss scale is a fail: cancel and record it.

Record in each run's notes (`uv run posttrain run note add <run id>`): git head, image, start checkpoint step, the preflight verdict and `report.txt` numbers, the throughput pair, the per-update table of the metrics above, the pass/fail verdict and anything unusual. Then update this plan's Progress, Outcomes and Artifacts.

## Validation and Acceptance

From the repository root:

    uv sync --all-packages --group dev --extra trl --python 3.13 --locked
    uv run ruff check . && uv run ruff format --check .
    uv run lint-imports                     # 9 kept, 0 broken
    uv run pytest                           # 2022 passed, 30 skipped (CUDA hidden)
    uv run pyright                          # identical error list to the base commit
    git diff --check

The new tests are `packages/train/tests/test_trl_precision.py` (defaults unchanged, fp16 flags, dtype forwarding, TurboQuant rule, binding validation, loss-scale monitor, skipped-step gradient norm, gap statistics, and a real PEFT float16-base model showing float32 adapters on the fresh, resumed and `ref` paths and a successful scaled optimizer step) and additions to `packages/advisor/tests/test_compatibility_rules.py` and `test_performance_rules.py`.

## Idempotence and Recovery

The offline harness writes only under `$OUT` and can be re-run; each step overwrites its own output file. Work packages and catalog entries are additive; removing the two work packages, `apps/lab/.posttrain/catalog/precision-qualification.yaml`, its `layer.yaml` line and the two gates in `apps/lab/src/posttrain_lab/qualification/gates.toml` (with the counts in `apps/lab/tests/test_qualification_gates.py`) reverts the experiment. A failed arm leaves a recovery checkpoint (every 50 updates); rerun the same work package to repeat it with the same seed.

## Artifacts and Notes

Offline matrix, Qwen3.5-0.8B, 48 GSM8K + 16 agent prompts, T=1.0, top_p=1.0, 21.8K to 25.0K sampled tokens per sampler. Cells: mean / p99 / max per-token |gap|, then mean |sequence sum of gaps|. Trainer log-probabilities from logits upcast to float32. Rows: sampler; columns: Transformers trainer dtype.

    sampler \ trainer        bf16                        fp16                        fp32
    vLLM bf16                0.0137/0.124/0.43 seq 0.39  0.0112/0.090/0.31 seq 0.32  0.0111/0.088/0.34 seq 0.31
    vLLM fp16                0.0133/0.101/0.26 seq 0.33  0.0019/0.015/0.12 seq 0.06  0.0015/0.011/0.02 seq 0.04
    HF fp32 (generate)       0.0122/0.095/0.29 seq 0.39  0.0016/0.012/0.04 seq 0.05  0 (reference)
    vLLM fp32                not runnable (Gated-DeltaNet kernel rejects float32)

    T=0.5: bf16/bf16 0.0115/0.209/0.82 seq 0.50; fp16/fp16 0.0014/0.027/0.07 seq 0.06
    fp32 LM head on both sides (bf16 otherwise): 0.0095/0.088/0.38 seq 0.29
    bf16/bf16 with the trainer log-softmax in bf16 (TRL today): 0.0138/0.129/0.43 seq 0.39
    vLLM fp16 eager prefill vs Transformers fp32 on the bf16 samples: 0.0015/0.012/0.037

Scripts and raw outputs of the first run: session scratchpad `precision/qwen08/` (not retained in the repository); the harness in `scripts/qualification/precision_mismatch/` is the tidied, model-agnostic version of those scripts.

Float16, bfloat16 and float32 support of the vLLM fork's kernels (carbonteq-v0.29.1.dev4, commit f09e4479, read from `~/projects/vllm-sm120` with `git show`; paged FA4 kernel from `sm120-paged-attention` 99a6fe0a). "Used by c144" means the LFM2.5-2.6B rollout binding `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c144-4k-t05@1`.

    kernel                                         fp16     bf16   fp32   used by c144  fp16 falls back to
    SM120 paged FA4 (SM120_FA4 backend)            yes      yes    no     no            -
    FlashAttention 2 (FLASH_ATTN)                  yes      yes    no     yes           -
    Triton unified attention, batch-invariant
      split-KV (fork)                              yes      yes    yes    no            -
    batch-invariant persistent GEMM, SM120 tuned
      shape table and generic rule                 untuned  tuned  yes    no            default persistent config
    invariant CUDA RMSNorm (fork)                  yes      yes    yes    no            -
    GDN chunked prefill (FLA Triton)               yes      yes    no     no (Qwen3.5)  -
    GDN fused CUDA decode                          no       yes    no     no (Qwen3.5)  Triton decode kernel
    LFM2 short convolution (Triton causal_conv1d)  yes      yes    yes    yes           -
    DSpark drafter, unquantized                    yes      yes    yes    yes           -
    DSpark NVFP4 gathered Markov bias (Triton)     no       yes    no     only if NVFP4  dequantize-and-scatter
    Punica LoRA shrink/expand (Triton)             yes      yes    no     yes           - (fp32 inputs rejected)
    LM head with hf_overrides head_dtype=float32   yes      yes    -      no            -

## Interfaces and Dependencies

In `packages/train/src/posttrain/train/precision.py`:

    type TrainingPrecision = Literal["bf16", "fp16"]
    def training_precision(backend_options) -> TrainingPrecision
    def logits_float32(backend_options) -> bool
    def rollout_dtype(engine) -> tuple[RolloutDtype | None, Literal["binding", "turboquant", "checkpoint"]]
    def resolve_precision(backend_options, engine | None, weight_precision) -> ResolvedPrecision

In `packages/train/src/posttrain/train/backends/trl/precision_runtime.py`: `LossScaleMonitor` (`observe(optimizer, global_step)`, `finite_grad_norm(normalizer)`), `loss_scale_callback_type(imports, monitor)`, `upcast_logits_to_float32(model)`, `require_float32_trainable_parameters(model)`, `require_default_precision(backend_options, technique)`.

`trainer_arguments(loop, output_dir, *, precision="bf16")` and `load_trainable_model(..., model_dtype="bfloat16" | "float16" | "float32")` in `common.py`. Metric names: `train/loss_scale`, `train/optimizer_step_skipped`, `train/optimizer_steps_skipped`, `train/rl/sampling_logp_delta_p99`, `train/rl/sampling_sequence_logp_delta_abs_mean`. Run attributes on `grpo_runtime_resolved`: `rollout_precision`, `rollout_precision_source`, `training_precision`, `training_model_load_dtype`, `training_loss_scaling`, `logits_float32`.

Revision note (2026-09-28): created with the offline results and the implementation; training arms pending the runtime image with Gated-DeltaNet kernels.
Revision note (2026-09-28, later): added the model-agnostic harness, the rollout-engine benchmark, the SM120 kernel dtype table and findings, the LFM2.5-2.6B canary work packages and the workstation runbook; recorded the top-p offset between vLLM processed log-probabilities and TRL's trainer log-probabilities.
