# LFM2.5-2.6B SAMPO from the base model in fp16 with a rank-16 adapter

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`. It prepares one training run and does not change the frozen product baseline under `docs/post-training/`, the framework code or any fork.

## Purpose / Big Picture

The best LFM2.5-2.6B AutomationBench runs so far are SAMPO (per-turn credit from live environment progress) from the base model (`lfm2.5-2.6b/automationbench-sampo-turns-150-lr5e-5-kl5e-3-local-v1`, run "SAMPO r2") and two continuations of it (`docs/plan/lfm26-mistake-penalty-resume.md`). Each continuation restarted its KL penalty at its own starting adapter, so the policy's distance from the base model was never bounded over the whole lineage. This plan prepares a single fresh run that keeps the continuation's data, budgets, penalties and 24 x 6 update shape, anchors the KL penalty to the base model for all 100 updates, uses a rank-16 LoRA adapter at a lower learning rate, trains and samples in float16 (fp16) to shrink the sampler-trainer log-probability gap, and oversamples prompt groups so an update needs fewer rollout waves.

After this change `posttrain job plan` resolves `apps/lab/.posttrain/work_packages/lfm26_automationbench_sampo_turns_100_r16_fp16_os4r5_local_v1.yaml` for the RTX PRO 6000 workstation (`targets/carbonteq-rtx-pro-6000-96gb`, dstack provider). Nothing is launched by this plan.

Terms. SAMPO is the multi-turn policy-optimization algorithm this repository implements as `train.sampo`; each assistant turn's advantage adds a discounted per-turn reward (discount gamma 0.95) to the episode advantage. LoRA is a low-rank adapter trained on top of frozen base weights; its update is scaled by alpha / rank. Active sampling regenerates prompt groups whose attempts all earned the same reward (no learning signal) until an update has 24 groups with reward spread; oversampling generates extra groups up front (`oversample`) and in each refill (`oversample_refill`) and discards surplus. KL reference is the policy the KL penalty measures distance from: `base` is the foundation model, `start` the checkpoint the run started from.

## Progress

- [x] (2026-09-29) Selections and work package added; `posttrain job plan` resolves it (with the release head's pending veRL manifest check relaxed in a scratch wrapper, see Surprises); the work package is registered as the candidate gate `lfm26-sampo-turns-100-r16-fp16-os4r5` in `apps/lab/src/posttrain_lab/qualification/gates.toml` (the inventory counts in `apps/lab/tests/test_qualification_gates.py` rise by one); `apps/lab/tests` and `packages/train/tests` pass (732 passed, 13 skipped); ruff clean.
- [ ] Launch on the workstation from a clean detached worktree of the pushed commit, after the 0.4.12 runtime images are republished and the fp16 canary qualification in `docs/plan/fp16-training-precision.md` has passed.
- [ ] Record results in Outcomes.

## Surprises & Discoveries

- Observation: the code's concurrency guard checks only the first round, `(num_prompts_per_step + oversample) x num_generations`, here (24 + 4) x 6 = 168, against the smallest of vLLM `max_num_seqs`, environment `max_concurrent` and `env_workers x episodes_per_worker` (all 180). Refill rounds are capped in the trainer at the first round's size (`min(missing + oversample_refill x generations, target + oversample x generations)` in `packages/train/src/posttrain/train/backends/trl/policy_curriculum.py`), so no round exceeds 168. The conservative bound (24 + max(4, 5)) x 6 = 174 also fits 180.
  Evidence: with `oversample: 7` the plan fails with "active_sampling oversample 7 needs 186 concurrent episodes ... max_num_seqs is 180; environment max_concurrent is 180; ... 12 x 15 = 180".
- Observation: on release head 602725cf every `posttrain job plan` fails with "installed runtime image manifest is unusable: kinds.online-rl-verl-py313: backend runtime identity differs from its shipped profile", because that commit changed the Verifiers lock and the runtime images await republishing. The existing fp16 canary package fails the same way. The plan above was produced by calling the CLI with `load_manifest(verify_variants=False)` from a scratch script outside the repository.
- Observation: with oversampling, the adaptive active-sampling loop raises "adaptive active sampling exhausted its bounded candidate capacity" if five rounds leave more than four groups missing, instead of ending the update short; the candidate pool is 24 x 6 = 144 groups and `max_candidate_batches` cannot exceed 6 on the 160-task mix. The lineage needed at most 5 rounds without oversampling, so this is unlikely.

## Decision Log

- Decision (user, 2026-09-29): start from the base model with a fresh LoRA adapter (no `--model-from-run`), `kl_reference: base`, beta 0.02, 100 updates.
  Rationale: the drift finding: both continuations used a moving KL anchor (their own starting adapter), so KL restarted near 0 at each resume while the policy kept moving away from the base. A base anchor bounds the whole run. Multi-turn agentic RL works use a KL coefficient around 0.01 (GiGPO; arXiv 2510.01132); 0.02 is the user's slightly firmer brake for a run that must not drift.
- Decision (user): LoRA rank 16, alpha 32, all linear layers, learning rate 2e-5 constant after 5 warmup updates (`warmup_ratio: 0.05`).
  Rationale: "LoRA Without Regret" (Thinking Machines, 2025) finds that LoRA's best learning rate is about 10 to 15 times full fine-tuning's, that with a fixed alpha of 32 and the 1/r scaling the optimal rate is roughly independent of rank, that RL needs very little capacity (even rank 1 matches full fine-tuning), and that LoRA must cover all layers, MLP included. kalomaze's rl-lora-ddd notes reach the same conclusions for RL with LoRA. The lineage's alpha/rank of 8/4 = 2 multiplies updates by 2; 32/16 = 2 keeps that multiplier, so 2e-5 is a 2.5x smaller step than 5e-5, chosen to reduce drift. Rank 16 adds no meaningful memory (tens of MB of float32 adapter and optimizer state).
- Decision (user): trainer fp16 with dynamic loss scaling from 1024 and float32 logits, rollout vLLM float16.
  Rationale: Qi et al. 2025, "Defeating the Training-Inference Mismatch via FP16" (arXiv 2510.26788): fp16 on both sides cuts the sampler-trainer mismatch by about an order of magnitude and avoids late bf16 collapse, including LoRA GRPO with token-level truncated importance sampling. Our LFM2.5-2.6B preflight measured the mean gap at 0.0037 in fp16 against 0.0098 in bf16. `logits_float32` is implied by fp16 since the float32 loss fix; it is stated explicitly for the reader.
- Decision (user): active sampling `oversample: 4`, `oversample_refill: 5`, `max_candidate_batches: 6`; environment clone `automationbench-lfm26-sampo-turns-v3-c180` (`max_concurrent: 180`), environment workers 12 x 15, vLLM `max_num_seqs: 180`.
  Rationale: the first round covers the usual shortfall of zero-spread groups so most updates finish in one wave; every limit is at 180 so each round (at most 168 episodes) runs at once. 6 is the most candidate batches the 160-task mix supports at 24 prompts.
- Decision: KV cache 48 GiB (`kv_cache_memory_bytes: 51539607552`), `gpu_memory_utilization: 0.68`.
  Rationale: c144 used 41 GiB (about 1.6M tokens, 11K per episode at 144); 48 GiB keeps about 11K tokens per episode at 168 (the largest round) and about 10.4K at 180, above the lineage's measured 8.4K average. Utilization rises by the extra 7 GiB (0.60 to 0.68, about 65 GB of the 96 GB card). With vLLM sleeping during the optimizer step the bf16 canary peaked at 61 GiB in total; adding 7 GiB of cache gives about 68 GiB, leaving roughly 25 GiB of headroom.
- Decision: checkpoint every 20 updates and keep all five (`checkpoint_limit: 5`), and a 24-hour job limit (`timeout_seconds: 86400`).
  Rationale: every checkpoint can be evaluated held-out; the 24-hour limit is the user's setting, since 100 updates with rounds of up to 168 episodes may exceed the lineage's 12-hour limit.
- Decision: the remaining settings are the g24x6 continuation's unchanged: truncation penalty 0.1, mistake penalty 0.02 per mistake capped at 0.1 (from the v3 environment), temperature 0.5 and top_p 0.95, 4,096-token replies, 20,480-token prompts, 24,576-token context, 12 turns, yield-first curriculum seed 172846, step weight 1.0, token-level truncated importance sampling at cap 2.0, reward projection `reward/automationbench-turn-progress@2`, and the trainer efficiency options (compiled decoder layers, importance ratio from the training forward, logits chunks of 128, gradient checkpointing).

## Outcomes & Retrospective

Not launched yet.

## Context and Orientation

New selections: the environment `automationbench-lfm26-sampo-turns-v3-c180` in `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` (a YAML merge of `automationbench-lfm26-sampo-turns-v3`, which now carries the anchor `&automationbench_sampo_turns_v3`); the settings `lfm2.5-2.6b/automationbench-sampo-turns-100-g24x6-r16-lr2e-5-kl2e-2-t05-os4r5-v1`, the training binding `training/lfm2.5-2.6b-trl-lora-automationbench-local-g144-w12-r16-fp16@1` and the rollout binding `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c180-4k-t05-fp16@1` in `apps/lab/.posttrain/catalog/precision-qualification.yaml`, beside the fp16 canary selections they derive from. Every work package must be classified in `apps/lab/src/posttrain_lab/qualification/gates.toml`, so the run is registered there as an experimental candidate.

## Concrete Steps

From `apps/lab` in a clean detached worktree of the pushed commit (runtime images republished):

    uv run --no-sync posttrain job plan .posttrain/work_packages/lfm26_automationbench_sampo_turns_100_r16_fp16_os4r5_local_v1.yaml --explain

Expected key lines:

    Precision: trainer fp16 (base weights float16, dynamic loss scaling from 1024; log-probs from float32 logits); rollout vLLM float16 (binding)
    KL reference: base model (kl_reference: base, beta 0.02)

The launch itself (`posttrain job run ... --provider dstack --target targets/carbonteq-rtx-pro-6000-96gb`) is the lead's decision and is not part of this plan.

## Validation and Acceptance

`uv run --no-sync pytest -q apps/lab/tests packages/train/tests` passes and `uv run --no-sync ruff check .` is clean. During the run, watch `train/peak_gpu_memory_gib` (expect under about 70 GiB), `train/loss_scale` and `train/optimizer_steps_skipped` (at most one or two early skips), `train/rl/kl` against the base, and `train/rl/active_sampling_round_*` for the number of waves per update.

## Idempotence and Recovery

The change only adds catalog entries and one work package; removing them restores the previous state. A failed launch can be retried with a new run id.

## Interfaces and Dependencies

No code or dependency changes. Backend `trl@1.12.0.post11` with dependency lock `3806d424...` and vLLM `vllm@0.29.1.dev4` with the DSpark drafter `LiquidAI/LFM2.5-2.6B-DSpark@458cedab`, as in the fp16 canary.
