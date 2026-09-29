# LoRA for RL: settings and evidence

**Status:** living record, started 2026-09-29. **Authority:** subordinate to
`docs/post-training/`; this page records evidence and does not choose settings for
any run. It is model-independent: our own runs appear as evidence, not as
defaults.

This page collects what is known about training LoRA adapters with online
policy-gradient RL (GRPO, SAMPO, GiGPO and relatives) and separates three kinds
of evidence that are often mixed:

- **LoRA-RL:** experiments that train LoRA adapters with RL.
- **Full-FT RL:** RL with every weight trainable. Settings that do not depend on
  the parametrization (KL coefficient, temperature, batch shape, filtering,
  precision) usually carry over; learning rates do not.
- **SFT only:** supervised fine-tuning results. They are marked **does not
  transfer** unless an RL experiment confirms them.

Numbers are quoted from the primary text with a section, table or figure. A claim
marked **UNVERIFIED** could not be confirmed in the primary text (for example,
values that appear only in a plot, or in a paper version we did not read). Our run
IDs refer to Trackio project `posttrain-lab`.

## How a LoRA step is sized

LoRA replaces a weight `W` with `W + (α/r)·B·A`, where `A` is initialised randomly
(PEFT: uniform with scale `1/sqrt(d_in)`), `B` is zero and `r` is the rank.
LoRA Without Regret ("Optimal learning rate and rank" and "Parametrization
invariances") shows that with Adam the first updates have the same expected size
for every rank: each of the `r` rank-1 terms contributes the same expected update,
and the `1/r` prefactor cancels the sum. The quantity that sets the initial update
scale is `α · init_A · LR_B`. With PEFT's initialisation fixed, that is
**α × learning rate, independent of rank**. Two consequences:

- Comparing α/r ratios between runs of different rank is the wrong unit. Rank 16
  with α 32 is not "the same multiplier" as rank 4 with α 8; at the same learning
  rate its initial step is four times larger.
- At a fixed α, the optimal learning rate is approximately independent of rank
  ("The 1/r scaling factor makes the optimal learning rate approximately
  independent of rank"). The blog adds that optimal LR "has some
  rank-dependence in the longer-training regime" (SFT, Figure 2).

Limits of the conversion: it is exact only for the initial updates, while `B` is
near zero. With one learning rate for `A` and `B`, changing α at a fixed α × LR
also changes how fast `A` moves relative to its initial scale (the blog's second
invariant, `init_A / LR_A`), a second-order effect. rsLoRA scales by `α/sqrt(r)`
instead of `α/r`, so an rsLoRA α is not comparable with a standard one (at rank
16, rsLoRA α 64 equals standard α 256).

### Unit conversion

Each row is one α × LR value written as the learning rate that produces it at
each α.

| α × LR | LR at α 8 | LR at α 16 | LR at α 32 | LR at α 64 |
|---|---|---|---|---|
| 8.0e-5 | 1e-5 | 5e-6 | 2.5e-6 | 1.25e-6 |
| 1.6e-4 | 2e-5 | 1e-5 | 5e-6 | 2.5e-6 |
| 3.2e-4 | 4e-5 | 2e-5 | 1e-5 | 5e-6 |
| 4.0e-4 | 5e-5 | 2.5e-5 | 1.25e-5 | 6.25e-6 |
| 6.4e-4 | 8e-5 | 4e-5 | 2e-5 | 1e-5 |
| 8.0e-4 | 1e-4 | 5e-5 | 2.5e-5 | 1.25e-5 |
| 1.6e-3 | 2e-4 | 1e-4 | 5e-5 | 2.5e-5 |

### Our runs and published configurations in these units

| Run or source | Rank / α / LR | α × LR | What happened |
|---|---|---|---|
| v1, v2 (`lfm26-olmo3-adaptive-oversample10x4-12k-20260915-r1`, `lfm26-vortex-v2-agentic-20260923-r1`) | 4 / 8 / 1e-5 | 8.0e-5 | Stable but barely moved in 20 updates |
| SAMPO r2 (`lfm26-sampo-turns-150-lr5e5-kl5e3-20260927-r2`), KL run (`lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2`) | 4 / 8 / 5e-5, KL 0.005 to base | 4.0e-4 | SAMPO r2 stable to about update 120 and the best held-out checkpoint so far; both drifted after about update 130 |
| v5 (`lfm26-vortex-v5-yield-first-64-20260925-r4` and its continuations) | 4 / 8 / 1e-4, no KL | 8.0e-4 | Entropy 0.18 → 0.64 by update 41, runaway after 54 |
| v3 (`lfm26-vortex-v3-lr2e4-20260923-r1`) | 4 / 8 / 2e-4 | 1.6e-3 | Entropy and truncation rose within 20 updates |
| Next run's candidate settings | 16 / 32 / 2e-5 | 6.4e-4 | Not run. 1.6× SAMPO r2's step and 0.8× the drifting 1e-4 runs, not a smaller step |
| Tinker cookbook RL recipes (`math_rl`, `code_rl`, `verifiers_rl`, multi-turn `harbor_rl`) | 32 / 32 (α UNVERIFIED) / 1e-5 | ≈3.2e-4 | Defaults, not ablated. α 32 is inferred from the blog ("We use α=32") and the `lora_alpha: int = 32` default of `get_lora_lr_over_full_finetune_lr` in `tinker_cookbook/hyperparam_utils.py`; the service's α is not in the cookbook |
| verl-agent LoRA configs (`examples/gigpo_trainer/run_alfworld_lora.sh`, `run_webshop_lora.sh`) | 64 / 64 / 3e-6, no KL | 1.9e-4 | Repository config, multi-turn, not ablated |
| Tina (Table 5) | 32 / 128 / 1e-6, cosine | 1.3e-4 | Single-turn math |
| Defeating the Training-Inference Mismatch via FP16 (§5.2) | 32 / 64 / 4e-5 | 2.6e-3 | Single-turn math on Qwen2.5-Math-1.5B; bf16 collapsed after about 600 steps, fp16 stable |
| kalomaze, rl-lora-ddd | 1-64 / rsLoRA α 64 / 1e-5 | 2.6e-3 at rank 16 (rsLoRA) | Single- and multi-turn toy tasks and math; higher ranks also got larger steps |

Published multi-turn LoRA-RL defaults therefore sit near **α × LR ≈ 2e-4 to
3.2e-4**, just below SAMPO r2's 4.0e-4. The single-turn math configurations run
up to ten times higher.

## Rank, alpha and target modules

**(a) LoRA-RL.**
- LoRA Without Regret ("Reinforcement learning"): "LoRA fully matches the
  learning performance of FullFT when running policy gradient algorithms for
  reinforcement learning, even with ranks as low as 1." Setting: single-turn math
  only. Llama-3.1-8B base on MATH and GSM8K (Figure 6) and Qwen3-8b-base on
  DeepMath with 8,192-token samples (the figures after Figure 6); α 32;
  policy gradient with importance sampling and group-mean centering. KL is not
  mentioned (UNVERIFIED whether any was used). The RL learning rates appear only
  in the figures (UNVERIFIED). Capacity argument ("How much capacity is
  needed"): MATH is about 10,000 problems × 32 samples = 320,000 bits, while
  rank-1 Llama-3.1-8B has 3M parameters.
- TinyLoRA ("Learning to Reason in 13 Parameters"): Qwen2.5-7B-Instruct reaches
  91% on GSM8K with 13 trained parameters (abstract; §6.2: 76% baseline, 95% at
  120 parameters; SFT reaches only 83% and 84%). On MATH (Table 2,
  Qwen2.5-3B-Instruct) capacity does matter: average 37.1 with 16 parameters,
  46.1 with 504, 48.0 with full fine-tuning. "Our findings are limited to math
  datasets" (Limitations). Qwen needs about 10× fewer parameters than LLaMA for the
  same result.
- Tina (§4.3, Table 4, one run per cell, DeepSeek-R1-Distill-Qwen-1.5B): ranks 4,
  8, 16, 32 and 64 averaged 47.72, 47.89, 48.92, 48.47 and 46.95. Tina adapts
  attention only (Table 5: query, key, value, dense) and still works.
- kalomaze, rl-lora-ddd (prime-rl, Qwen3-4B-Instruct-2507 and
  DeepSeek-R1-Distill-Qwen-7B, all attention and MLP projections): on the
  multi-turn alphabet-sort task rank 1 was "sufficient". On unscramble and
  acereason-math, ranks 1 and 4 "aren't nearly as close" to 16 and 64 after 100
  steps. **Confound:** all ranks used rsLoRA with a constant α 64, which by the
  sizing rule above gives higher ranks a larger initial step, so rank and step
  size are not separated.
- The Path Not Taken (§5.2, Figure 10): DS-Qwen-1.5B on DeepMath, ranks 8/32/64 and
  LR 1e-4/5e-5/1e-5 for 200 steps. PiSSA (principal-direction LoRA) gave no gain
  over plain LoRA and "often collapses early" at the higher rates.
- Multi-turn configurations, none ablated: verl-agent rank 64, Tinker
  `harbor_rl` rank 32, SkyRL-Agent MemAgent rank 128 (§5).

**(b) Full-FT RL.** RL fine-tunes small subnetworks (Mukherjee et al., abstract
and Table 2): full-FT RL changes only 5-30% of parameters, but the updates are
99.2-99.8% of full rank, spread over nearly all matrices. A low-rank adapter is
therefore an approximation of a full-FT RL update, one that works in practice at
matched learning rate (LoRA Without Regret).

**(c) SFT only, does not transfer.** LoRA Without Regret's rank-dependent SFT loss
gaps (Tulu3, OpenThoughts3) and its target-module table on Llama-3.1-8B and
Qwen3-30B-A3B are SFT. The same post did repeat the module comparison with RL on
MATH: "Attention-only LoRA underperforms MLP-only LoRA (which performs similarly
to MLP+attention) in these settings as well" ("Layers Where LoRA Is Applied"), so
that one result does transfer.

**(d) Our evidence.** No rank ablation yet; every run so far used rank 4 α 8,
all linear layers. For LFM2.5-2.6B, rank 1 across all linear layers is about 1.5M
parameters (our estimate from `config.json` shapes), against 14,400 episodes in a
100-update run of 24 × 6. For hybrid convolution/attention models, no source
covers whether the convolution projections (`in_proj`, `out_proj`) should be
adapted (UNVERIFIED either way).

**(e) Recommendation.** Any rank from 4 to 32 is defensible for RL. Choose rank for
memory and serving, then set the learning rate in α × LR units; rank is not a
drift lever. Adapt all linear layers, MLP included. **Confidence: moderate.** All
rank ablations are single-turn math or toy tasks, and the only short-run evidence
that rank matters is confounded.

## Learning rate and schedule

**(a) LoRA-RL.**
- LoRA Without Regret ("Optimal learning rates for LoRA vs. FullFT"): "the optimal
  LR for LoRA is consistently 10x the one used for FullFT in the same
  application, for both supervised learning and reinforcement learning." The
  fitted SFT multiplier was 9.8. Schedule: "constant learning rate schedule (no
  warmup or cooldown)".
- Tinker cookbook RL recipes default to 1e-5 at rank 32 (see above), including the
  multi-turn `harbor_rl` recipe (10 turns, 8,192 tokens, group 4, 8 groups per
  batch, temperature 1.0, KL 0).
- Tina (§4.3, Table 4): LR 5e-6, 1e-6 and 5e-7 at α 128 averaged 47.87, 48.47 and
  47.91; the default is 1e-6, cosine with a minimum LR and warmup ratio 0.1
  (Table 5).
- TinyLoRA (§5.1) swept 1e-7 to 2e-4 and kept the best per update size (per-size
  optima UNVERIFIED; not given in the text).
- FP16 paper (§5.2): LoRA at 4e-5, "slightly larger" than its full-FT rate.

**(b) Full-FT RL.** Multi-turn agents use 1e-6: ARLArena Table 6 (all four
tasks), the GiGPO repository (`run_alfworld.sh`), SkyRL-Agent (§4.2), Golubev et
al. (App. C), and the Practitioner's Guide (App. C GRPO default; its PPO sweep
winner, Table 13, is actor 1e-6). Single-turn reasoning uses 5e-7 to 2e-6
(ScaleRL App. 5e-7 with 100 warmup steps; DAPO 1e-6 with 20 warmup steps;
Dr. GRPO 1e-6 constant, Table 6; ProRL 2e-6 constant, §3.2; OLMo 3 1e-6 to 2e-6
constant, Table 49). Applying the 10× rule to 1e-6 gives **1e-5 at α 32**
(α × LR 3.2e-4).

**(c) SFT only, does not transfer.**
- The "15x over the FullFT for short runs" multiplier ("Learning rates in short
  and long runs") is SFT, and "based on anecdotal evidence" for runs under about
  100 steps.
- The Tinker `get_lr` formula (`5e-5 × 10 × (2000 / hidden_size)^exponent`,
  about 5e-4 for a 2048-wide model) was fitted on SFT sweeps over Tulu3. Tinker's
  own RL recipes use 1e-5, fifty times lower. Do not size RL learning rates from
  it.
- The implicit schedule of LoRA's zero-initialised `B` (effective learning rate
  rising as `B` grows) was measured on SFT (Tulu3, OpenThoughts). It does mean a
  fresh adapter starts with small effective steps whatever the schedule.

**(d) Our evidence (α 8, rank 4; α × LR in parentheses).** 1e-5 (8e-5) barely
moved in 20-50 updates. 5e-5 (4e-4) with KL 0.005 to the base model was stable to
about update 120-130 in both the KL run and SAMPO r2. 1e-4 (8e-4) without KL
drifted by updates 41-54. 2e-4 (1.6e-3) destabilised within 20. In α 32 units,
5e-5 at α 8 is 1.25e-5, at the published optimum rather than five times above it.

**(e) Recommendation.** Size learning rates in α × LR. Stay near the published
multi-turn value and our stable value, 3.2e-4 to 4e-4 (1e-5 to 1.25e-5 at α 32),
unless an ablation supports more. Constant schedule; a short warmup is harmless
and not evidence-driven. **Confidence: moderate.** The 10× rule is stated for RL
but was measured on single-turn math, and our stable point comes from one model.

## KL penalty

**(a) LoRA-RL.** Tinker's RL recipes default to `kl_penalty_coef = 0.0`; when set,
`incorporate_kl_penalty` in `tinker_cookbook/rl/metrics.py` adds
`coef × (avg_kl − per_token_kl)` to the advantages (in the reward, k1 estimator,
reference = base model). verl-agent's LoRA configs set `use_kl_loss=False` and
`use_kl_in_reward=False`. LoRA Without Regret does not mention KL (UNVERIFIED).
No LoRA-RL ablation of KL was found.

**(b) Full-FT RL.**
- **Anchor.** RL's Razor (abstract; Figures 3 and 11): forgetting is predicted by the KL
  between the fine-tuned and base policy on the new task (quadratic fit R² 0.96 on
  ParityMNIST, 0.71 on the LLM experiments), and on-policy RL is implicitly biased
  toward KL-minimal solutions even without a penalty. ProRL (§2.3.1, §3.3) resets
  the reference model to *allow* drift: resets restore stability "but also
  facilitate greater policy divergence from the base model"; it resets when
  validation stalls, together with the optimizer. Kimi k1.5 (§2.3.2) uses the
  current policy as the reference at every iteration (mirror descent) and resets
  the optimizer each time. A moving reference is a tool for more exploration, not
  a drift brake.
- **Coefficient, multi-turn agents.**
  - 0.01: ARLArena Table 6 (k3, mean-std normalised advantages, temperature 1.0),
    the GiGPO repository (`kl_loss_coef=0.01`, `kl_loss_type=low_var_kl`, in the
    loss) and the Practitioner's Guide sweep winner (App. A, Table 13: 0.01 above
    0.005 and 0.001).
  - 0.001: the Practitioner's Guide defaults (App. C) and RAGEN's vanilla StarPO
    (App. C.2, k1). RAGEN's stable variant StarPO-S removes it (§4.2).
  - Evidence for keeping it: ARLArena Table 2, adding KL k3 gave +18.10 success
    rate on ALFWorld (GRPO).
  - Evidence against larger values: ARLArena Table 4, KL 0.05 cut CISPO's success
    from 54.42 to 38.46 (and raised SAPO's from 25.16 to 48.05), "overly
    constrains training".
- **Coefficient, single-turn reasoning:** usually 0 (DAPO §2.3; Dr. GRPO Table 6;
  OLMo 3 Table 49; ScaleRL; the FP16 paper, Table 3). The Entropy
  Mechanism paper (§4.1, Figure 10): a reference KL "achieves stable entropy values"
  but "leads to a degradation in performance", in a setting whose problem was
  entropy collapse, not growth.
- **Estimator and placement.** k3 in the loss (TRL's default; ARLArena; GiGPO's
  `low_var_kl`, a clamped k3) versus k1 in the reward (RAGEN, Tinker). No
  head-to-head comparison was found (UNVERIFIED which is better).
- **Advantage scale.** The coefficients above were tuned with advantages divided
  by the group standard deviation (ARLArena and GiGPO: `mean_std_norm`). With
  mean-only normalisation (our `advantage_normalization: mean`), the
  policy-gradient term is σ times smaller for the same rewards, so a given β is
  about **1/σ times stronger** relative to the policy gradient. With partial
  credit rewards and a within-group σ around 0.1-0.2, β 0.02 acts roughly like
  0.1-0.2 would under std normalisation. Measure σ before transferring a
  published β.

**(c) SFT only.** Not applicable.

**(d) Our evidence.**
- Base-anchored β 0.005 at α × LR 4e-4 held entropy near 0.3 until about update
  60 and near 0.4 until 130 (KL run), then entropy reached 1.24 and KL 0.24 by
  update 150. SAMPO r2 drifted less (entropy 0.28 against 0.33 and KL 0.051
  against 0.077 over updates 101-121) and its update-120 checkpoint scored 0.624
  and 0.619 held out (temperature 0.1 and 0.5, suite
  `automationbench-lfm26-heldout-mix-v4`) against base 0.617 and 0.607.
- No KL at 8e-4 drifted by updates 41-54 (v5).
- Continuations anchored to their own starting adapter drifted:
  `lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1` (β 0.01; entropy
  0.18 → 0.30, KL to its start 0.001 → 0.048 in 43 updates; held out 0.585 and
  0.555 at updates 20 and 40 against its start's 0.624) and
  `lfm26-sampo-cont40-fixed-tools-20260928-r1` (entropy 0.31 → 0.42 → 0.59 → 0.72
  per ten updates, 0.91 at update 38; KL to its start 0.005 → 0.23).

**(e) Recommendation.** Anchor to the base model (**strong**). Keep KL on when
the policy gradient has no other trust region (see pitfalls), k3 in the loss
(**moderate**, no ablation). Coefficient 0.005-0.02 under mean-only
normalisation, after checking σ (**weak to moderate**). Add an automatic stop on
KL to the base or on entropy, and choose checkpoints on held-out tasks.

## Sampling temperature and top-p

**(a) LoRA-RL.** Tinker recipes 1.0. The FP16 paper's LoRA run used its standard
rollout settings (1.0 and top-p 1.0 in Table 3 for the large runs; the LoRA run's
settings are UNVERIFIED). No LoRA-specific temperature study was found.

**(b) Full-FT RL.**
- Training at 1.0 with top-p 1.0 is the norm: ARLArena Table 6 (training 1.0,
  validation 0.6-0.7), OLMo 3 Table 49, Dr. GRPO Table 6, the FP16 paper Table 3,
  Golubev et al. (App. C). ProRL uses 1.2 (§3.2) to delay entropy collapse.
- The only multi-turn sweep, the Practitioner's Guide (App. A, Figure 2 and
  Table 13), found "optimal performance occurs between 0.7 and 1.0", with 0.7
  best.
- POLARIS (§2): choose the training temperature per model at the point where
  accuracy starts to fall while diversity is highest (0.7 for
  DeepSeek-R1-Distill-Qwen-7B, 1.4 for Qwen3-4B), above the recommended
  decoding temperature, and raise it in later stages.
- **Top-p.** Golubev et al. (§5.2): after a vLLM upgrade enabled top-k and min-p,
  performance "degrade[d] after 5–10 training iterations" because trajectories
  came from a truncated distribution; they sample at 1.0 with every filter off
  "to ensure unbiased sampling" (App. C).
- No source supports matching the training temperature to the evaluation
  temperature.

**(c) SFT only.** Not applicable.

**(d) Our evidence.** SAMPO r2 trained at 0.8 and top-p 0.95. Its checkpoints are
the best held out at both 0.1 and 0.5. On the older suite the base model scored
0.589 at 0.1, 0.621 at 0.5 and 0.596 at 0.8. The continuations at 0.5 got worse,
but they also changed the anchor, batch and penalties, so the temperature effect
is not isolated. The TRL fork divides logits by the temperature before scoring,
so entropy and KL are measured on the tempered distribution: values from runs at
different temperatures are not comparable.

**(e) Recommendation.** Train at 0.7-1.0 with top-p 1.0 (**moderate**). A
temperature below the model's useful exploration range has no published support.

## Group size and batch

**(a) LoRA-RL.** No LoRA-RL batch study was found. Configurations: Tinker
`harbor_rl` 8 groups × 4, `verifiers_rl` 32 × 8, `math_rl` 100 × 4; verl-agent
16 × 8; Tina 4 generations, batch 32.

**(b) Full-FT RL.**
- ScaleRL (§5): larger batches reach a higher asymptote ("moving to batch size of
  2048 prompts both stabilized training"). With the total batch fixed, 8, 16, 24
  and 32 generations per prompt left "fitted scaling curves essentially
  unchanged".
- RAGEN Table 1 (0.5B, symbolic tasks, fixed batch): 4 responses per prompt
  generalised best.
- Multi-turn defaults are 16 prompts × 8 (ARLArena Table 6, GiGPO).
- Step-level credit (GiGPO, SAMPO) also needs attempts that share a state, which
  larger groups provide.

**(c) SFT only, does not transfer.** LoRA Without Regret's large-batch penalty
("Batch size effects", Figure 3: "LoRA is less tolerant of large batch sizes than
full fine-tuning", independent of rank) was measured on a 10,000-example
OpenThoughts3 SFT subset. No RL test exists (UNVERIFIED for RL).

**(d) Our evidence.** Anchor-state step advantages matched 41% of turns in VORTEX
traces at group 4 (50% with IDs stripped). 16 × 4 and 24 × 6 were both used;
24 × 6 has not been compared at matched settings.

**(e) Recommendation.** 6-8 samples per prompt and at least 16 prompts per
update for step-level credit (**weak to moderate**).

## Active sampling and oversampling

**(a) LoRA-RL.** None specific.

**(b) Full-FT RL.** Dropping groups whose rewards are all equal and refilling
helps. DAPO (§3.2 and the ablation table): AIME 42 → 50 when dynamic sampling was
added. OLMo 3 (Figure 26): "greatly reduced loss variance". ScaleRL (§3.2,
Figure 6a): the effective-batch (zero-variance filtering) variant "performs
better asymptotically". RAGEN (§4.2, Figure 5): keeping the 50% most variable
rollouts avoided collapse in PPO on FrozenLake. ARLArena Table 2: DAPO's retry
cap 2 → 3 gave +22.15 success rate, and Table 3 reports dynamic filtering gains.
No paper ablates how many extra groups to sample up front; that is a throughput
choice.

**(c) SFT only.** Not applicable.

**(d) Our evidence.** Choosing tasks by earlier pass rates cut zero-variance
groups from 52% (`lfm26-olmo3-random20-20260912-r3`) to 15%
(`lfm26-olmo3-adaptive20-20260912-r4`) and sampling rounds per update from 5.6 to
2.1; yield-first ran at 9-10% from v5 on.

**(e) Recommendation.** Keep active sampling (**strong**). Size oversampling for
throughput (**weak**, no quality evidence). If surplus groups are discarded in
completion order, the kept batch favours short episodes: discard by prompt
order or at random.

## Precision

**(a) LoRA-RL.** FP16 paper §5.2 and Figure 1(h): LoRA rank 32, α 64, LR 4e-5 on
Qwen2.5-Math-1.5B with token-level truncated importance sampling; "BF16-based
LoRA training collapses after roughly 600 steps, whereas FP16 maintains stable
training throughout." TinyLoRA (Figure 4) found fp32 adapters better than bf16 and
fp16 at a fixed byte budget, which only matters for tiny adapters.

**(b) Full-FT RL.** FP16 paper Figure 2: the sequence-level log-probability ratio
between sampler and trainer, KL 7.64 in bf16 against 0.32 in fp16; fp16 on both
sides stabilised every algorithm it tested (§4). ScaleRL (§3.2, Figure 5b): an
fp32 LM head on generator and trainer raised the asymptotic reward from 0.52 to
0.61.

**(c) SFT only.** Not applicable.

**(d) Our evidence** (`docs/plan/fp16-training-precision.md`).
- Offline on Qwen3.5-0.8B: fp16 on both sides cut the mean per-token gap 7.2×
  (0.0137 → 0.0019). An fp32 LM head on both sides removed about 31% of the bf16
  gap; upcasting bf16 logits to fp32 removed under 1%.
- LFM2.5-2.6B preflight: mean gap 0.0037 against 0.0098, p99 0.047 against 0.187.
- Canary `prec-lfm26-canary-fp16-20260929-r3` against
  `prec-lfm26-canary-bf16-20260929-r2` (five updates each): gap mean 0.0067
  against 0.0133, p99 0.052 against 0.198, loss scale 1024 throughout, no skipped
  step. The first fp16 canary failed (see pitfalls).

**(e) Recommendation.** fp16 on both trainer and sampler, with loss scaling and
float32 loss arithmetic (**moderate to strong**). The published bf16 collapses
came after hundreds of steps, so a 100-update run gains mainly a cleaner
importance ratio.

## Pitfalls found in our system

- **Moving KL anchor on continuations.** A continuation started from an adapter
  used that adapter as its KL reference, so KL restarted near zero at each resume
  while the policy kept moving away from the base model (the two continuations
  above). Fixed by `kl_reference: base`, the default since release 0.4.12
  (`docs/plan/active-sampling-oversample.md`).
- **fp16 KL overflow in a half-precision loss.** TRL 1.12.0.post11 computed the
  loss in the dtype of the log-probabilities. In fp16 the k3 term
  `exp(ref − logp)` overflowed (beyond ln 65504 ≈ 11.09) on masked multi-turn
  tool tokens, `inf × 0` gave NaN, and the loss scaler skipped every update of the
  first LFM2.5-2.6B fp16 canary. Fixed by scoring fp16 logits in float32 and
  casting log-probabilities and entropies to float32 before the loss
  (`float32_logprob_trainer_type`, then TRL `1.12.0.post12`).
- **top_p < 1 adds a constant sampler/trainer log-prob gap.** vLLM's processed
  log-probabilities are renormalised over the nucleus and TRL's trainer
  log-probabilities are not, so each sampled token's sampler log-probability
  exceeds the trainer's by the log of the nucleus mass. With top-p 0.95 this
  offset was part of the production 0.016-0.022 "gap", and it biases truncated
  importance sampling whatever the precision.
- **One optimizer step per batch means clipping never fires.** With
  `num_iterations = 1` and one optimizer step per rollout batch, TRL sets the old
  log-probabilities to the current forward pass, the PPO ratio is exactly 1, and
  the clip fraction is 0 on every update. Clip-higher and SAMPO's sequence-level
  clipping (which ARLArena's Finding 1 calls critical for stability) do nothing.
  The KL penalty and the learning rate are then the only brakes on drift.
- **veRL before 0.4.13 ignored the binding's temperature and top_p.** veRL's
  rollout defaults (1.0, 1.0, top-k −1) overrode every agent-loop episode, and the
  actor and reference scaled logits by 1.0 (commit `8bdaead1`). TRL-versus-veRL
  comparisons made before that commit were not like for like.

## Open questions and the ablations that would settle them

1. **Learning rate at a new rank.** At rank 16 α 32, compare 1e-5 with 2e-5
   (α × LR 3.2e-4 and 6.4e-4) for 40 updates: KL and entropy slopes and held-out
   score at update 40.
2. **Does rank matter at matched α × LR?** Rank 4 against 16 at the same α and LR
   for about 20 updates. LoRA Without Regret predicts nearly identical early
   curves.
3. **Temperature and top-p.** 0.8 with top-p 1.0 against 0.5 with top-p 0.95 at
   otherwise equal settings. Compare the clamped importance-weight fraction and
   held-out score at both evaluation temperatures.
4. **KL coefficient with a base anchor under mean-only normalisation.** 0.005
   against 0.02, after measuring the within-group reward σ.
5. **KL estimator and placement.** k3 in the loss against k1 in the reward, with
   everything else equal. No published comparison was found.
6. **Convolution projections in hybrid models.** All linear layers against all
   except the convolution `in_proj` and `out_proj`.
7. **Oversampling discard order.** A code check, not a run.

## Sources

Read for this page (2026-09-29), from the arXiv PDF unless noted:

- Thinking Machines, LoRA Without Regret (blog):
  <https://thinkingmachines.ai/blog/lora/>
- Tinker cookbook (`tinker_cookbook/hyperparam_utils.py`,
  `tinker_cookbook/rl/metrics.py`, `recipes/{math_rl,code_rl,verifiers_rl,harbor_rl}/train.py`):
  <https://github.com/thinking-machines-lab/tinker-cookbook>
- Tina: Tiny Reasoning Models via LoRA: <https://arxiv.org/abs/2504.15777>
- Learning to Reason in 13 Parameters (TinyLoRA): <https://arxiv.org/abs/2602.04118>
- Reinforcement Learning Finetunes Small Subnetworks in Large Language Models:
  <https://arxiv.org/abs/2505.11711>
- The Path Not Taken: RLVR Provably Learns Off the Principals:
  <https://arxiv.org/abs/2511.08567>
- kalomaze, RL Learning with LoRA: A Diverse Deep Dive (blog):
  <https://kalomaze.bearblog.dev/rl-lora-ddd/>
- A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning:
  <https://arxiv.org/abs/2510.01132>
- RAGEN: <https://arxiv.org/abs/2504.20073>
- Group-in-Group Policy Optimization (GiGPO): <https://arxiv.org/abs/2505.10978>;
  configurations in verl-agent `examples/`:
  <https://github.com/langfengQ/verl-agent/tree/master/examples>
- ARLArena (SAMPO): <https://arxiv.org/abs/2602.21534>
- SkyRL-Agent: <https://arxiv.org/abs/2511.16108>
- Golubev et al., Training Long-Context, Multi-Turn Software Engineering Agents
  with Reinforcement Learning: <https://arxiv.org/abs/2508.03501>
- RL's Razor: <https://arxiv.org/abs/2509.04259>
- ProRL: <https://arxiv.org/abs/2505.24864>
- Kimi k1.5: <https://arxiv.org/abs/2501.12599>
- DAPO: <https://arxiv.org/abs/2503.14476>
- Understanding R1-Zero-Like Training (Dr. GRPO): <https://arxiv.org/abs/2503.20783>
- The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models:
  <https://arxiv.org/abs/2505.22617>
- POLARIS (blog): <https://hkunlp.github.io/blog/2025/Polaris/>
- OLMo 3: <https://arxiv.org/abs/2512.13961>
- The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL):
  <https://arxiv.org/abs/2510.13786>
- Defeating the Training-Inference Mismatch via FP16:
  <https://arxiv.org/abs/2510.26788>

Not read, so not used: ARTIST, Agent-R1, SWE-RL.
