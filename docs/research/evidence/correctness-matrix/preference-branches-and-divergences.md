# Preference branches and divergence arithmetic

2026-09-30. The expanded audit finds a reproducible cancellation bug in
non-default DPO divergence transforms. In matched BF16 model updates,
alpha=.999998 caused 1.9–2.3% parameter-gradient discrepancies. Stable
arithmetic closes those discrepancies without changing the objective or beta.
This is evidence about preference math, not an explanation of SAMPO learning.

## Independent kernel coverage

The retained artifacts (`preference-branches-and-divergences/summary.json`, local archive)
include matched before/after grids, two expanded model runs and four matched
alpha-near-one model runs. The CPU grid covers all fifteen upstream DPO loss
branches: sigmoid, hinge, IPO, sigmoid_norm, SFT, EXO-pair, NCA-pair, robust,
BCO-pair, SPPO-hard, AOT, AOT-unpaired, APO-zero, APO-down and DiscoPOP.
These are upstream branch checks, not fifteen new Posttrain operations.

The 306 cases comprise 180 ordinary branch/beta/length-discount controls and
126 additional allowed divergence controls. Inputs use FP16, BF16 and FP64;
FP64 is a diagnostic reference. Supported half inputs are scored in FP32.
Independent Python probability sums and scalar loss formulas supply expected
values; five-point finite differences with h=1e-4 supply logit derivatives.
This avoids reproducing the Torch autograd graph. The prior central h=1e-6
method produced a cancellation false positive for a large squared loss;
both retained comparisons use the corrected derivative method.

| Grid | Passed | Failed | Interpretation |
| --- | ---: | ---: | --- |
| Before | 237 | 69 | 28 half-input near-alpha-one numerical failures, 28 double-reference exceptions, 13 residual misses |
| After | 293 | 13 | Strict absolute loss-value misses remain; maximum derivative error 3.8e-10 |

Keep the original gates: FP64 absolute tolerance 1e-7; half-input tolerance
2e-6. Remaining loss relative errors are at most 6.14e-7. Four ordinary
SPPO-hard cases and nine divergence cases miss absolute value tolerance;
the largest absolute miss is 3.78e-4 for forward-KL IPO with loss around 616.
These are consistent with FP32 rounding but are still failed gates. No
tolerance was widened to make them pass, and there is no demonstrated
gradient reversal or material parameter error in these residual fixtures.

## Why the repair works

Write a policy/reference log ratio as L and a=alpha-1. The alpha score is
`expm1(a*L)/a`; its derivative with respect to L is `exp(a*L)`.
Computing `exp(a*L)-1` loses significant digits when a*L is small, then
division by a amplifies the value error. A wrong pair score also changes
the outer logistic derivative, so an apparently correct inner derivative
does not protect the final parameter gradient. Forward-KL has the related
score `-expm1(-L)` and benefits from the same stable primitive.

The fork repair preserves its existing alpha-limit cutoff and exponent
clamps. It adds double-input handling solely for references. The ordinary
reverse-KL sigmoid objective and default beta remain unchanged. Focused
f-divergence and half-precision regressions pass 25 tests. Full/fused and
distributed runtime qualification remains separate.
All four native DPO divergence training tests also pass. The published
source-only repair is fork commit
`416b8978053d56bf4bc8674748c652f464999fac`; no wheel was released or selected.

## Actual model updates

Each expanded arm uses BF16 base weights, FP32 q/v LoRA adapters, rank4,
LoRA alpha8, LR1e-4, zero dropout, fixed supplied preference pairs and Adam.
Three SFT steps are followed by three steps per selected preference branch;
each branch resets adapters and optimizer. The ten newly covered branches
are EXO-pair through DiscoPOP above. These are direct loss/model-Jacobian
checks, without native Trainer accumulation or fresh task reward collection.
SFT uses the first chosen answer; preference losses use two chosen/rejected
pairs, so AOT exercises a nontrivial sorting batch.

| Expanded BF16 run | Updates | Strict failures | Parameter-gradient matches | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| Qwen0.8B | 33 | 2, SPPO-hard loss value | 33/33 | 3.82 GiB |
| LFM1.2B | 33 | 2, SPPO-hard loss value | 33/33 | 2.76 GiB |

All updates were finite and moved adapters. SPPO-hard starts with loss50;
the two later steps miss the model harness absolute value gate1e-6 by
1.02–2.85e-6. Parameter-gradient relative errors are zero in both models.
The fixtures do not establish held-out learning or validate full BCO/EXO
training recipes merely because their paired loss branches match.

Matched alpha=.999998 arms reset the same model, pairs and seed before
three SFT and three DPO updates. The first preference step matches because
policy equals reference. Subsequent steps expose cancellation:

| Model | Before preference gradient relative errors | After | Before maximum loss error | After maximum |
| --- | --- | --- | ---: | ---: |
| Qwen0.8B BF16 | 0, 2.097%, 1.930% | 0, 0, 0 | 1.943e-3 | 3.135e-8 |
| LFM1.2B BF16 | 0, 2.274%, 2.104% | 0, 0, 0 | 7.695e-4 | 3.596e-9 |

All six repaired preference steps pass. Changed trajectories after the first
update are expected; each step is compared against the independent math at
its actual current parameters. No FP64 model training was used.

## Reproduce and remaining work

From `/home/hammad/projects/rl`, use the research Python at
`/home/hammad/projects/trl-gdpo-capo/.venv/bin/python`. Set PYTHONPATH to the
isolated renderer and TRL source plus `/tmp/trl-math-peft`,
`/tmp/trl-math-renderers-deps` and `/tmp/posttrain-mathdeps`.
Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/preference_loss_grid.py --extended --output PATH`.
The recorded before DPO source hash is
`c98e6da94120a52d06ada466d96ba40f0cddb2a01abe50dcc2143e486941bab5`;
after is `b7962195dbcfc7fc19f3fc22ae6a7afb102c239cb07b7fb93f639840eeb0374c`.
Baseline source is from fork commit9f0825046ae3509a6be804d74a93fb89d5dc695e.

Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_preference_run.py --model qwen08 --two-pairs
--loss-types exo_pair nca_pair robust bco_pair sppo_hard aot aot_unpaired
apo_zero apo_down discopop --output PATH`, then repeat with `--model lfm12`.
For the matched repair use `--loss-types dpo --divergence alpha_divergence
--alpha 0.999998`. Execute GPU arms serially. Temporary dependency directories
are research setup, not a reproducible published runtime environment.

Open gates include native/fused preference training, broader FP16 model
canaries, boundary/clamp stress, long contexts, full fresh generation and
held-out task behavior. The campaign remains open and runtime pins unchanged.
