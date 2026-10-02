# Excluded-token arithmetic and sampled distillation: measured follow-on

2026-09-30. Two further numerical failures are reproduced and repaired in the
source candidate. Both real models retain finite, nonzero adapter updates.
Correct gradients do not establish useful training: the LFM Thinking model
cannot produce its final answer under the 32-token diagnostic budget.

## Independent policy-loss comparison

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/policy_loss_grid.py` derives Python float64 loss and gradient
references outside TRL. The 240-case cross-product covers GRPO, DAPO, BNPO,
DR-GRPO, and LUSPO; token/sequence ratios; scalar/token credit; KL beta 0/0.04;
both KL correction choices; and signed ratio shifts. Unequal lengths and
excluded tool/padding tokens are included. All 240 match within 1e-10.

Eight additional excluded-behavior cases bring the loss grid to 248.
Five separate accumulation comparisons are retained in the same artifact.
Logical-batch versus microbatch accumulation matches exactly for GRPO, DAPO,
DR-GRPO, and LUSPO. BNPO uses a microbatch-local valid-token denominator, so its
unequal-length gradient differs from a single global batch (maximum difference
0.190867). This is expected for that exposed internal normalization, not a
newly confirmed public algorithm bug.

For the excluded-behavior cases, missing old-policy scores and a 112-nat gap
were injected only on masked positions. Six cases returned nonfinite loss or
gradient before correction. Masking after exponentiation cannot undo `inf * 0`
or `NaN * 0`. GRPO now neutralizes excluded log ratios before exponentiation;
all 248 cases pass. It does not clamp valid action ratios.

The real BF16 Qwen canary with missing excluded scores and LFM canary with an
extreme excluded gap both retain the expected [-0.5, 0, 0.5] score gradient,
finite parameter gradients, and approximately 1e-4 maximum adapter movement.
Peak allocation is 1.786 GB and 2.430 GB respectively. These are direct-loss
CUDA checks, not full rollout adapter or distributed qualification.

## Independent distillation comparison

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/distillation_grid.py` calculates probabilities and divergence
values in Python and uses finite differences for logit gradients. Its 54
distributional cases cover both KL directions, JSD, temperatures 0.7/1/2,
top-k 0/1/2, renormalization and tail buckets, and masked labels. Nine sampled
IW-OPD cases independently derive the frozen credit and prefix weights:

    a_t = teacher_logp_t - rollout_logp_t
    w_t = 1 + gamma * (1 - sum_{j<t}|a_j| / max(sum_j|a_j|, epsilon))
    L = -sum_t w_t * a_t * student_logp_t / N_valid

Masked positions contribute neither credit nor loss. Gamma is tested at
0/0.5/1; temperature at 0.7/1/2. All 63 double-precision cases pass. This
verifies the selected IW-OPD formula; it does not imply equivalence to plain
unweighted reverse KL. Current framework arguments choose `iw_opd`, inheriting
gamma 0.5 from the trainer config. Recipe provenance remains a separate audit.

Three precision cases compare against probabilities of the already-quantized
input logits and round the independent derivative to the input gradient dtype.
Before correction, FP16 and BF16 failed the predeclared 1e-6 tolerance:

| Input dtype | Loss error before | Gradient error before |
|---|---:|---:|
| FP16 | 0.000115857 | 0.000244141 |
| BF16 | 0.00163056 | 0.00195313 |
| FP32 | 6.06e-8 | 4.47e-8 |

The sampled IW-OPD path now promotes half logits before temperature/log-softmax.
All 66 cases pass. This preserves FP32/FP64 inputs and does not change gamma,
the objective, or sampling settings. Alternative JSD half-precision paths are
not qualified by this fix. Both new regressions fail against the isolated
original post13 wheel and pass on the candidate.

The broader IW-OPD suite initially failed a nonnegativity assertion on a
four-step same-teacher smoke test at -7e-13. That assertion was invalid: this
frozen-advantage sampled surrogate is signed. It now checks finiteness;
73 tests pass, with one optional Liger skip. No loss clamping was added.

## Fresh real-model groups: three updates of two completions

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_distillation_run.py` runs the cached immutable Qwen
and LFM weights with BF16 base and FP32 rank-4, alpha-8 q/v adapters, learning
rate 1e-4, temperature 0.8, gamma 0.5, deterministic attention, seed 42, and
gradient norm cap 1. The teacher is the same base with adapters disabled.
Adapter B matrices start with recorded standard deviation 0.003 so student and
teacher disagree. Each update collects a fresh group; there is no replay.

This diagnostic bypasses full Trainer, Verifiers, and teacher transport. It
uses a simple arithmetic prompt, not AutomationBench. Prompt tokens are outside
the completion loss; post-EOS padding is masked. It has no multi-turn tool
outputs. Existing AutomationBench experiments are reported separately in
`short-rollout-results.md`.

| Model / budget | Completed / 6 | Truncated / 6 | Maximum loss error | Parameter gradient relative error | Peak GPU allocation |
|---|---:|---:|---:|---:|---:|
| Qwen / 32 | 5 | 1 | 1.98e-8 | 0 on all 3 steps | 2.522 GB |
| LFM Thinking / 32 | 0 | 6 | 5.73e-10 | 0 on all 3 steps | 2.672 GB |
| LFM Thinking / 256 | 5 | 1 | 1.49e-10 | 0 on all 3 steps | 4.000 GB |

All nine updates have finite gradients and nonzero adapter movement near
1e-4. The parameter reference is a vector-Jacobian product through the same
model graph with independently calculated sampled-score credit, separating
loss mathematics from model backward behavior. The LFM 32-token arm has signed
loss on steps 1 and 3, directly confirming why positivity is not an invariant.

Qwen produced both correct answers and incorrect answers such as five or three.
LFM's 32-token outputs remained inside reasoning; at 256 tokens, the completed
responses answered four. This is descriptive behavior, not a reward-improvement
or convergence claim. Different token budgets alter random-number consumption
and subsequent updates; the later trajectories are not a matched causal
comparison. Three steps, one prompt, and a same-base teacher cannot select a
production learning rate or certify task quality.

## Reproduction and remaining gates

Use the isolated research Python at
`/home/hammad/projects/trl-gdpo-capo/.venv/bin/python`, with `PYTHONPATH` set to
`/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft` from the
Posttrain root. Run the policy and distillation grids with `--output`, then the
short distillation script with `--model qwen08` or `--model lfm12`, and
`--max-tokens 32` or `256`. Models use the immutable revisions already recorded
in `initial-results.md` and each evidence JSON. Aggregate JSON files alongside
this report retain raw-file hashes; generated text stays machine-local.

The fixes are pushed as `780bdd3edea61a993151c7dc1f02336b74a399d9` on
`codex/sampo-local-credit`; there is no new wheel
release or production consumer pin. Existing jobs retain their immutable
runtime. Full Trainer/native Verifiers multi-turn masks, teacher transport,
save/resume, SFT/DPO real-model math, GDPO/CAPO complete runs, distributed/fused
paths, Qwen FP16 backward localization, and the earlier later-step BF16
parameter-gradient discrepancy remain open. The broader campaign is active.
