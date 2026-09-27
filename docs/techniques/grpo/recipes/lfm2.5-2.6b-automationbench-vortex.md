# LFM2.5-2.6B LoRA on AutomationBench with VORTEX: what we tried

**Status:** living record, started 2026-09-26. **Authority:** subordinate to
`docs/post-training/`. Numbers come from Trackio runs in project `posttrain-lab`;
run IDs are given so each claim can be re-read.

This page records the learning rates, algorithm settings and efficiency changes
tried while training LiquidAI LFM2.5-2.6B on the AutomationBench training mix with
the VORTEX curriculum, what each did, and the rules we now follow.

## Setup these results apply to

- **Model and update:** `models/lfm2.5-2.6b@bf16`, LoRA rank 4, alpha 8, all linear
  layers, no dropout. One process on one RTX PRO 6000 (96 GB), vLLM colocated.
- **Environment:** `automationbench-lfm26-train-mix-v7` (Verifiers). Reward is
  AutomationBench partial credit: the share of task assertions passed.
- **Algorithm:** OLMo 3 GRPO with active sampling. VORTEX chooses which tasks to
  sample (yield-first curriculum) and refills groups whose rewards are all equal.
  One optimizer step per generation batch, so the policy ratio is always 1.
- **Episode budgets (v5):** 12 turns, 4096 tokens per reply, 24,576 tokens of
  context, temperature 0.8, top-p 0.95.

## Learning rates

LoRA needs a larger learning rate than full fine-tuning: a LoRA update scales with
learning rate × alpha / rank, and only the small adapter moves. Tinker's LoRA RL
recipes use 1e-5 to 4e-5 at alpha 32, which is 4e-5 to 1.6e-4 at our alpha 8 (catalog
comment on `vortex-20-local-v3`, commit 583f21e7). Our first runs (v1, v2) used a
normal fine-tuning rate, 1e-5. The sweep then deliberately started high, at 2e-4 in
v3, and stepped down: 1e-4 in v5, then 5e-5 with a KL penalty.

Reward and entropy below are averages of the first and last 10 updates of each run
(Trackio `updates`; reward includes the 0.2 truncation penalty from v5 on).
Truncation is the share of truncated replies over the last 10 updates.

| Learning rate | Runs (updates) | Entropy | Truncation rate | Reward mean | Outcome |
|---|---|---|---|---|---|
| 1e-5 (full-model rate) | v1 `lfm26-olmo3-adaptive-oversample10x4-12k-20260915-r1` (1-20), v2 `lfm26-vortex-v2-agentic-20260923-r1` (1-20); also `lfm26-olmo3-random20-20260912-r3`, `lfm26-olmo3-adaptive20-20260912-r4`, `lfm26-adaptive-grpo-50-c32-20260920f` (1-50) | flat: 0.174→0.171 (v1), 0.173→0.166 (v2) | 30% (v1), 19% (v2) | 0.366→0.316 (v1), 0.299→0.402 (v2) | Stable but barely learns in 20-50 updates. Held-out (64K, update 20): v1 0.555, v2 0.595 against base 0.592 |
| 2e-4 | v3 `lfm26-vortex-v3-lr2e4-20260923-r1` (1-20), v4 `lfm26-vortex-yield-first-v4-8k-20260923-r1` (1-20) | 0.186→0.255 (v3), 0.178→0.242 (v4) | 42% (v3), 43% (v4) | 0.335→0.299 (v3), 0.339→0.315 (v4) | Too high: entropy and truncation climb within 20 updates. Held-out (64K, update 20): v3 0.616, v4 0.488, which is measurably below base |
| 1e-4 | `lfm26-vortex-v5-yield-first-64-20260925-r4` (1-20), `lfm26-vortex-v5-100-from-r4-step20-20260925-r2` (21-41), `lfm26-vortex-v5-150-dspark-opt-20260926-r1` (41-65) | 0.17→0.23 (20), →0.64 (41), →0.88 (54), →5.6 (65) | 0.1-0.25 until 56, then →0.84 | 0.35-0.45 until 56, then <0 | Learns, but entropy drifts upward from the first update and runs away after update 54 |
| 5e-5 + KL 0.005 | `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` (1-150). Run `-r1` was cancelled after 2 updates: the trainer silently ran without KL (see rule 4) | 0.18→0.29 (60), →0.40 (130), →1.24 (150) | 8-25% of completions | 0.28→0.43 by update 40, then flat around 0.4 | Stable for longer, but KL to the base model kept doubling about every 20 updates (0.029 at 60, 0.24 at 150) while reward stopped improving. Held-out (20 unseen tasks × 3): base 0.589, update 100 0.595, 130 0.599, 150 0.569, so none beat base |

### What the 1e-4 collapse looked like

Traces from `lfm26-vortex-v5-150-dspark-opt-20260926-r1`:

| Updates | Entropy | Episodes truncated | Turns (mean) | Thinking share of output | Gradient norm |
|---|---|---|---|---|---|
| 41-50 | 0.58-0.80 | 7% | 6.4 | 67% | 0.003-0.005 |
| 51-56 | 0.67-1.33 | 6% | 6.2 | 64% | 0.003-0.009 |
| 57-60 | 1.6-3.0 | 34% | 4.6 | 73% | 0.010-0.019 |
| 61-66 | 3.8-5.6 | 83% | 2.8 | 91% | 0.020-0.025 |

- Every truncation came from one reply hitting the 4096-token cap, not from the
  turn or context budget. By update 61 those replies were random multilingual
  tokens, not reasoning.
- As more groups became all-failing, VORTEX generated more rows to fill 64 useful
  ones (72 → 204) and each update slowed (≈280 s → 965 s). At update 66 it gave up
  after 10 rounds with 52 rows.
- The vLLM importance-sampling ratio stayed at 0.97-0.98 and was never clamped, so
  the collapse was not a rollout/trainer mismatch. The same entropy drift is
  present on the original trainer without DSpark (runs r4 and r2), so neither
  change caused it.

### Rules

No update-20 checkpoint of v1-v4 beat the base model on held-out tasks: v1-v3
are within noise (each paired 95% bootstrap interval of the difference includes
zero) and v4 is worse (−0.200 to −0.020;
`docs/plan/vortex-v3-v4-matched-heldout-evaluation.md`).

1. **Do not reuse full-model learning rates for LoRA.** At rank 4 / alpha 8, 1e-5
   left the policy nearly unchanged over 20-50 updates; published runs train for
   hundreds to thousands of steps, so this does not show 1e-5 fails over a full run.
2. **Do not go above 5e-5 here; published LoRA RL points to about 1e-5.**
   2e-4 destabilised within 20 updates, 1e-4 within about 55, and 5e-5 drifted
   without gaining reward after update 40. See Research basis: the published
   LoRA optimum is about 10× the full fine-tuning norm of 1e-6.
3. **Watch entropy from the first update.** Its slope predicts collapse long
   before reward does. At 1e-4 it tripled in 41 updates while reward still looked
   healthy. Stop or lower the rate when entropy passes about twice its early level.
4. **If a run uses a KL penalty, check it is applied.** Whether to use one is
   open (see Research basis). OLMo 3
   originally fixed `beta` at 0. TRL's `Olmo3GRPOConfig` still rejects `beta`, so
   Posttrain sets it on the built config. Confirm a run logs `train/rl/kl` from
   update 1; without it the KL penalty is not in the loss. Resuming from a LoRA adapter, TRL keeps a frozen copy of
   that adapter as the reference; a fresh adapter uses the base model.
5. **Start again from a checkpoint instead of an exact resume when changing the
   rate.** An exact resume restores the scheduler state, which can keep the old
   rate. Use `--model-from-run` with `--curriculum-from-run` at the same step.

## Research basis

The rates and penalty above were chosen from our own short runs, not from the
literature. This section records what published RL work on language models uses,
so each setting can be defended or changed. Values are as stated in each paper;
"none" means the paper sets the term to zero.

### What published runs use

| Work | Model, update | RL learning rate, schedule | KL to reference | Entropy and stability notes |
|---|---|---|---|---|
| DeepSeekMath GRPO ([2402.03300](https://arxiv.org/abs/2402.03300)) | 7B, full | 1e-6 | 0.04 | Origin of GRPO |
| DeepSeek-R1 ([2501.12948](https://arxiv.org/abs/2501.12948)) | 671B MoE, full | 3e-6 | 0.001 | Clip ratio called crucial |
| DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)) | 32B, full | 1e-6 constant, 20-step warmup | none | Clip-higher (0.2/0.28) against entropy collapse; sample-level loss made entropy and length rise unhealthily |
| Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)) | 1.5-7B, full | 1e-6 constant | none | Verifiable rewards remove the distribution-shift concern |
| Skywork-OR1 ([2505.22312](https://arxiv.org/abs/2505.22312)) | 7-32B, full | 1e-6 constant | none (0.001 pulled the policy back and hurt later stages) | Fast entropy collapse predicts worse tests; clip-high 0.28 made entropy rise sharply; adaptive entropy target 0.2 |
| ProRL ([2505.24864](https://arxiv.org/abs/2505.24864)) | 1.5B, full | 2e-6 constant | kept, with periodic reference and optimizer resets when validation stalls | KL was the more stable fix than temperature or clip-higher alone; late instability in long runs |
| OLMo 3 ([2512.13961](https://arxiv.org/abs/2512.13961)) | 7-32B, full | 1e-6 to 2e-6 constant | none; KL to the reference still grows gradually | Clip 0.2/0.272, 450-2,300 steps |
| Tülu 3 PPO ([2411.15124](https://arxiv.org/abs/2411.15124)) | 8-405B, full | 3e-7 (8B) to 1e-7, linear decay | 0.01-0.1 swept, 0.05 final | Lower KL penalty led to over-optimization; best checkpoint chosen every 100 steps |
| ScaleRL ([2510.13786](https://arxiv.org/abs/2510.13786)) | 8B and 17B×16 MoE, full | 5e-7 constant, 100-step warmup | none | A 10-15% truncation rate typically destabilized training; entropy alone did not predict performance |
| Practitioner's guide to multi-turn agentic RL ([2510.01132](https://arxiv.org/abs/2510.01132)) | 1.5-8B, full | 1e-6 (GRPO); higher rates trained faster in a sweep | 0.001 default; above 0.001 more stable, 0.01 best | GRPO and RLOO collapsed at 1.5B on the hard task |
| GiGPO ([2505.10978](https://arxiv.org/abs/2505.10978)) | 1.5-7B, full | 1e-6 | 0.01 (ALFWorld, WebShop), 0.001 (search) | Step-level credit for multi-turn agents |
| RAGEN ([2504.20073](https://arxiv.org/abs/2504.20073)) | 0.5-3B | full; LoRA variant at 10× the rate | 0.001, removed in the stable variant | "Echo trap": reward spread and entropy move before reward collapses; gradient-norm spikes mark the point of no return |
| Turn-PPO ([2512.17008](https://arxiv.org/abs/2512.17008)) | 1.7-7B, full | 1e-6 actor | 0.001; removing KL did not stop multi-turn GRPO crashes | Multi-turn GRPO collapses abruptly; a critic fixed it |
| LoRA Without Regret ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/lora/)) | 8B, LoRA | about 10× the full fine-tuning optimum (15× under 100 steps), constant | – | RL matches full fine-tuning even at rank 1 |
| Tina ([2504.15777](https://arxiv.org/abs/2504.15777)) | 1.5B, LoRA r32 α128 | 1e-6 cosine with warmup | – | Ranks 8-32 all strong |
| Entropy mechanism ([2505.22617](https://arxiv.org/abs/2505.22617)) | 0.5-32B | – | a plain KL-to-reference penalty degraded performance | Entropy bonuses are highly coefficient-sensitive |

### What this says about our settings

- **Learning rate.** Full fine-tuning RL uses 1e-6 almost everywhere (range
  1e-7 to 5e-6). LoRA's optimum is about 10× that, so about 1e-5, and
  LoRA Without Regret found it barely depends on rank. Our 5e-5 is five times
  that and our 1e-4 and 2e-4 ten to twenty times. We judged 1e-5 as "barely
  learns" after 20-50 updates, but the runs above train for 500-2,300 steps: a
  low rate needs more updates, not a higher rate. Our rules 1 and 2 therefore
  traded stability for speed without evidence that 1e-5 fails over a full run.
- **Schedule.** GRPO-family work uses a constant rate, often with a 20-100 step
  linear warmup. Decay is not the norm: Tülu 3 (PPO) decays linearly and Tina
  uses cosine, and the only direct comparison found (DRG-Sapphire,
  [2505.21908](https://arxiv.org/abs/2505.21908)) saw similar results. Learning-rate
  decay is not the missing fix; a warmup is cheap and standard.
- **KL penalty.** Most recent work drops it (DAPO, Dr. GRPO, Skywork-OR1, OLMo 3,
  ScaleRL, and multi-turn agents such as SkyRL, SimpleTIR and DeepSWE). Where it
  is kept, values are 0.001-0.01 for multi-turn agents and up to 0.04-0.1 in
  GRPO's original and PPO recipes. Our 0.005 had no source. At 5e-5 the penalty
  added about 0.0012 to a loss that moves by ±0.01, too weak to hold the policy.
  KL to the reference grows in every run, with or without a penalty (OLMo 3);
  the warning sign is growth without reward, as in our KL run after update 40.
- **Entropy.** Published work mostly fights entropy collapse (falling). Our
  failure is the opposite: entropy rising until replies turn to noise. Clip-higher,
  the usual remedy, does nothing here: one optimizer step per batch keeps the
  policy ratio at 1, so clipping never triggers (clip fraction 0%). Rising
  entropy is reported with entropy bonuses, too-high clip bounds, sample-level
  losses and training on negative or truncated samples.
- **Truncation.** ScaleRL links instability to 10-15% truncated completions. Our
  healthy updates ran at 8-25%, and the 1e-4 collapse went to 84%. Truncation
  is a better-supported lever than KL here.
- **Checkpoint choice.** Tülu 3, ProRL and Search-R1 pick checkpoints by held-out
  evaluation. Our held-out results (update 130 best, 150 below base) show the last
  checkpoint is not the one to keep.

### Next experiments this supports

1. LoRA learning rate 1e-5 (10× the full fine-tuning norm), constant with a
   10-update linear warmup, run long enough to judge (at least 150 updates).
2. KL either off, relying on held-out checkpoint selection, or at 0.01, the value
   the multi-turn guide found best; not 0.005.
3. Keep truncation under 10%: raise the per-reply budget or change how truncated
   rollouts enter the loss (masking them is contested: DeepSWE and SkyRL mask,
   SWE-Master found masking caused collapse).
4. Evaluate held-out tasks every 10-20 updates and keep the best checkpoint.

## Other settings tried

| Setting | Values tried | Result |
|---|---|---|
| Prompt groups × rollouts per update | 8×4, 10×4, 16×4 | 16×4 (64 rollouts) uses the RTX PRO well. SAMPO's paper uses groups of 8; not yet tried. |
| Truncation penalty | none, 0.2 | 0.2 ranks a truncated rollout below an equally scored finished one before active sampling measures spread. It also adds negative gradient when the policy degrades. |
| Curriculum | random, adaptive quota, yield-first | Choosing tasks by their earlier pass rates cut groups whose rollouts all score the same from 52% (random, `lfm26-olmo3-random20-20260912-r3`) to 15% (adaptive, `lfm26-olmo3-adaptive20-20260912-r4`), and sampling rounds per update from 5.6 to 2.1. Yield-first (exploration share 0.2) took v3's 30% to 20% in v4 and runs at 9-10% from v5 on, where the environment fix also made the always-failing HR tasks solvable. It is the current default. |
| Per-reply output budget | 4096, 8192 (v4) | v4 raised it to 8192 because v3's truncated replies often stopped at exactly 4096 tokens. The traces showed the model spending the extra budget on thinking and then producing garbage, so budget was not the constraint; v5 went back to 4096 and added the truncation penalty. Healthy episodes hit 4096 5-6% of the time. |
| Turn budget | 12 | Healthy episodes averaged 6.3 turns; 0-3% used all 12. |
| Context | 24,576 | 3% of healthy episodes exceeded 23K tokens. |

## Efficiency changes

Measured per 64-rollout update on the RTX PRO 6000. The baseline is run
`lfm26-vortex-v5-100-from-r4-step20-20260925-r2`: 823 s per update, of which
rollout 334 s and actor update 373 s (mean of 21 updates).

| Change | Where | Effect | Status |
|---|---|---|---|
| Score each micro-batch at its own length instead of the generation batch's padding | TRL 1.12.0.post10 | Micro-batches were padded to ~25K tokens for ~9.7K real ones. Actor update 373 → 126 s | Adopted |
| Compile each decoder layer (`compile_decoder_layers`) | TRL post10, binding `g64-w8@2` | 0.961 → 0.672 s per 9.7K-token episode, same peak memory | Adopted |
| Importance-sampling ratio from the training forward (`importance_sampling_from_training_logps`) | TRL post10, binding `g64-w8@2` | Removes a separate no-grad pass (0.301 s per episode) | Adopted |
| Checkpointing only above 14,336 tokens (`gradient_checkpointing_min_tokens`) | TRL post10 | Saves ~0.2 s per episode but raises peak memory from ~9 GB to ~53 GB | Not adopted |
| No gradient checkpointing | benchmark | Faster, but needs 46-82 GB | Not adopted |
| Liger kernels | benchmark | No LFM2 integration in Liger | Not adopted |
| DSpark speculative decoding at c64 (`c64-4k@2`, 13.75 GiB KV cache, 9 draft tokens) | vLLM `carbonteq-v0.29.1.dev4` | Rollout 281 → 160 s per update (4.13 → 2.00 s per row), 26-33% draft acceptance, ~3.3-4 tokens per step | Adopted |
| Session-aware prefix-cache eviction | vLLM dev4 | Small gain in replay | Kept, not a lever |
| Actor-update timing split | Posttrain 0.4.8 | Forward/backward 113.5 s; optimizer step ~0 s | Diagnostic |

Together: **823 s → 276-292 s per update** (runs
`lfm26-v5-opt-ab-dspark-20260926-r1`, `lfm26-v5-opt-breakdown-20260926-r1`,
`lfm26-vortex-v5-150-dspark-opt-20260926-r1`). The first update of a run takes
about 550 s because of compilation and warm-up.

Open: the live forward/backward costs about 1.78 s per episode against 0.67 s in
the single-length benchmark. Mixed episode lengths, allocator churn or compile
recompiles are the candidates.

## Next

- The experiments listed under Research basis: LoRA rate 1e-5 with warmup, KL
  off or 0.01, truncation under 10%, held-out checkpoint selection.
- SAMPO with environment-derived turn rewards (assertion progress minus 0.05 per
  failed tool call) at the KL run's settings is running as
  `lfm26-sampo-turns-150-lr5e5-kl5e3-20260927-r2`. Over updates 101-121 it drifted
  less than the KL run (entropy 0.28 against 0.33, KL 0.051 against 0.077), but by
  update 141 entropy had reached 0.56 and KL 0.235. Its checkpoints at 100, 130 and
  150 go on the same held-out suite (`heldout-matched-64k-v3`) against base 0.591;
  compare them in Observatory's Evals views. Anchor-state step advantages match
  41% of turns in VORTEX traces, 50% with IDs stripped.
- Run notes in Observatory (KL run: "How this run came about") hold the full run
  history, including which reasons were not recorded at the time.
