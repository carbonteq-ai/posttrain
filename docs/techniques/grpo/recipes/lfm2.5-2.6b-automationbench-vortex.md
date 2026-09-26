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
recipes use 1e-5 to 4e-5 at alpha 32, which is 4e-5 to 1.6e-4 at our alpha 8 (see
`docs/plan/agentic-workload-inference-optimization.md`). Our first runs used the
full-model rate 1e-5.

| Learning rate | Runs (updates) | Entropy | Truncation rate | Reward mean | Outcome |
|---|---|---|---|---|---|
| 1e-5 (full-model rate) | `lfm26-olmo3-random20-20260912-r3` (1-20), `lfm26-olmo3-adaptive20-20260912-r4` (1-20), `lfm26-adaptive-grpo-50-c32-20260920f` (1-50), `lfm26-vortex-v2-agentic-20260923-r1` (1-20) and other 1e-5 arms | flat, 0.15-0.25 | flat or noisy | flat: 0.39→0.43, 0.43→0.34, 0.28→0.37 | Stable but barely learns: the policy hardly moves in 20-50 updates |
| 2e-4 | `lfm26-vortex-v3-lr2e4-20260923-r1` (1-20), `lfm26-vortex-yield-first-v4-8k-20260923-r1` (1-20) | 0.18→0.30 in 20 updates | 0.20→0.55, 0.13→0.48 | 0.39→0.25, 0.29→0.33 | Too high: entropy and truncation climb within 20 updates |
| 1e-4 | `lfm26-vortex-v5-yield-first-64-20260925-r4` (1-20), `lfm26-vortex-v5-100-from-r4-step20-20260925-r2` (21-41), `lfm26-vortex-v5-150-dspark-opt-20260926-r1` (41-65) | 0.17→0.23 (20), →0.64 (41), →0.88 (54), →5.6 (65) | 0.1-0.25 until 56, then →0.84 | 0.35-0.45 until 56, then <0 | Learns, but entropy drifts upward from the first update and runs away after update 54 |
| 5e-5 + KL 0.005 | queued: `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2`, 150 updates from the base model. Run `-r1` was cancelled after 2 updates: the trainer silently ran without KL (see rule 4) | – | – | – | Pending |

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

1. **Do not reuse full-model learning rates for LoRA.** At rank 4 / alpha 8, 1e-5
   left the policy nearly unchanged.
2. **Start LoRA RL at 3e-5 to 5e-5 here, not 1e-4 or 2e-4.** 2e-4 destabilised
   within 20 updates; 1e-4 within about 55.
3. **Watch entropy from the first update.** Its slope predicts collapse long
   before reward does. At 1e-4 it tripled in 41 updates while reward still looked
   healthy. Stop or lower the rate when entropy passes about twice its early level.
4. **Anchor long runs with a KL penalty, and check it is applied.** OLMo 3
   originally fixed `beta` at 0. TRL's `Olmo3GRPOConfig` still rejects `beta`, so
   Posttrain sets it on the built config. Confirm a run logs `train/rl/kl` from
   update 1; without it the KL penalty is not in the loss. Resuming from a LoRA adapter, TRL keeps a frozen copy of
   that adapter as the reference; a fresh adapter uses the base model.
5. **Start again from a checkpoint instead of an exact resume when changing the
   rate.** An exact resume restores the scheduler state, which can keep the old
   rate. Use `--model-from-run` with `--curriculum-from-run` at the same step.

## Other settings tried

| Setting | Values tried | Result |
|---|---|---|
| Prompt groups × rollouts per update | 8×4, 10×4, 16×4 | 16×4 (64 rollouts) uses the RTX PRO well. SAMPO's paper uses groups of 8; not yet tried. |
| Truncation penalty | none, 0.2 | 0.2 ranks a truncated rollout below an equally scored finished one before active sampling measures spread. It also adds negative gradient when the policy degrades. |
| Curriculum | random, adaptive quota, yield-first | Yield-first (exploration share 0.2) is the current default. |
| Per-reply output budget | 4096 | Healthy episodes hit it 5-6% of the time; raising it does not prevent collapse. |
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

- Learning rate 5e-5 with KL 0.005 for 150 updates from the base model. Record
  its results in the learning-rate table above.
- SAMPO with environment-derived turn rewards (assertion progress and failed tool
  calls) is implemented and runs on the 8 GB GPU with LFM2.5-1.2B
  (`lfm12-sampo-turns-8gb-20260926-r1`); a 2.6B comparison against this VORTEX
  control is the next experiment. Anchor-state step advantages match 41% of
  turns in these traces, 50% with IDs stripped.
