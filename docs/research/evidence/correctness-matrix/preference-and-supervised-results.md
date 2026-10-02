# Supervised and preference training: numerical and masking audit

2026-09-30. DPO has two independently reproduced probability-calculation
problems, now repaired in the source candidate. Both real models complete
three SFT and three DPO updates with finite, nonzero adapter movement and
matching independent gradients. Separately, the pinned LFM renderer trains
injected assistant headers in SFT. That masking failure is still open; passing
the loss calculation does not qualify the training targets. Follow-on status:
the renderer source repair is now published; see [LFM mask repair](lfm-renderer-mask-repair.md).

Published source candidate: TRL `18e89c58bee70d25f1231cbe0dc4a865540d6fe5`
on `codex/sampo-local-credit`. No wheel release or production pin update.

## Independent values and gradients

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/preference_loss_grid.py` evaluates the actual unfused DPO
loss with independent Python softmax/log probabilities and central finite
differences. It covers sigmoid DPO, hinge, IPO, sigmoid-normalized DPO, and the
DPO trainer's SFT branch, beta 0.1/0.5, length-discount alpha absent/0.5,
unequal chosen/rejected lengths, excluded prompt/padding tokens, and causal
label shifting. References use quantized input logits and reference scores;
gradient references are rounded to the model input dtype before comparison.

| Precision | Before correction | After half promotion and stable probabilities |
|---|---:|---:|
| FP64 | 20/20 pass | 20/20 pass |
| FP16 | 0/20 pass | 20/20 pass |
| BF16 | 0/20 pass | 20/20 pass |

Tolerance is 1e-7 for double inputs and 2e-6 for half inputs, declared before
execution. For ordinary sigmoid DPO at beta 0.1, BF16 initially has loss error
0.00097196 and maximum gradient error 0.00012207. This is excess probability
rounding, not evidence that a particular run's preference quality deteriorated.

TRL now promotes half logits before scoring the policy, online reference, and
cached reference. Six regressions fail on the isolated original post13 wheel
and pass with the candidate. Float32 and float64 inputs retain their precision.

## Small score errors can produce larger parameter errors

After promotion alone, Qwen's three actual DPO updates still fail the original
1e-4 parameter-gradient relative tolerance: errors are 2.0278%, 1.9251%, and
2.0047%. The loss errors remain below 1.18e-7. Tracking gradients to the raw
model logits locates differences of 7.68e-8 to 8.71e-8; a vector-Jacobian
product using the trainer's exact logit gradient reproduces its parameter
gradient with zero difference. The discrepancy begins in score arithmetic,
not that repeated model-backward control.

The previous selected-logprob helper computes `selected_logit - logsumexp`.
For nearly certain tokens, float32 subtraction/derivative cancellation can
erase a small gradient. Independent double-probability calculations show
Qwen's alternative `log_softmax` gradient is closer: maximum error roughly
1.4e-9 to 1.9e-9, versus 7.5e-8 to 8.7e-8 on the previous calculation.
The source DPO path now uses row-wise `log_softmax` and gather consistently
for policy and both reference paths. This preserves the objective and bounds
temporary vocabulary softmax work to one sequence row.

A minimal negative control uses logits [20, 4, 0] and chooses the first token.
The old helper returns log probability 0 and chosen-logit gradient 0;
double probabilities imply a positive gradient 1.145963e-7. The stable helper
retains the positive gradient within the predeclared 1e-8 tolerance. This is a
numerical correctness correction, not an alternative DPO recipe.

The final Qwen source candidate matches independent parameter gradients
exactly on all three updates. Its preference margin is 0, -0.18885, and
0.43807 before each update. The initial wrong-direction movement is retained
in the evidence: three updates do not certify monotonic improvement, and tiny
numerical perturbations can alter subsequent optimizer trajectories. LFM
already matched parameter gradients after promotion; the final stable path
also matches exactly.

## Actual model and renderer checks

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_preference_run.py` uses the exact pinned
`carbonteq-renderers==0.1.12.post1.dev2`, immutable cached model revisions,
BF16 base, FP32 rank-4/alpha-8 q/v adapters, LR 1e-4, beta 0.1, seed 42,
deterministic attention, gradient norm cap 1, and zero adapter initialization.
The fixed pair prefers “Four.” to “The answer is five.” The base reference
is obtained with adapters disabled. SFT and DPO independently reset adapters
and Adam state; each executes three updates. These are direct losses, not
full Trainer or environment qualification.

| Model | SFT loss step 1 → 3 | DPO loss step 1 → 3 | Final gradient relative error | Peak allocation |
|---|---:|---:|---:|---:|
| Qwen0.8B | 1.11060 → 1.06020 | 0.69315 → 0.67148 | 0 for both algorithms | 2.769 GB |
| LFM1.2B Thinking | 8.11609 → 8.02190 | 0.69315 → 0.66332 | 0 for both algorithms | 2.668 GB |

All twelve final updates have finite gradients and adapter movement near
1e-4. Causal next-token targets and prompt/padding masks are checked through
independent probability reduction and score-gradient coefficients. SFT's
gradient reference agrees with the existing HF causal-loss implementation;
this does not prove that the supplied labels have the right product meaning.

The initial raw Qwen chat-template test stopped before training because full
conversation tokens did not preserve its generation-prompt prefix. The exact
pinned dedicated Qwen renderer passes that requirement. The raw-template
failure is not attributed to Posttrain's renderer.

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/renderer_mask_audit.py` adds six serialized multi-turn cases:
system, user, assistant tool call, tool result, final assistant; train both
assistant turns, only the final turn, or only the first turn. External distinct
content sentinels check inclusion rather than simply trusting renderer metadata.
Qwen passes all three. LFM fails all three because selected SFT target text
contains `<|im_start|>assistant`, which is injected as prompt scaffolding during
inference. Neither model includes system/user/tool-result sentinels or
unselected assistant content. The LFM renderer inherits an attribution-only
fallback without sampled-token metadata; the source ownership is the renderers
fork, not a model-specific Posttrain loss workaround. At this experiment's
checkpoint the failure remained open. The follow-on
[renderer repair](lfm-renderer-mask-repair.md) corrects the source and adds
regressions; a new wheel and production pin remain separate gates.

## Verification and open work

Ten focused DPO precision/collator regressions pass, including near-certainty,
policy gradients, cached/online reference parity, and padding. A broader 21-case
trainer slice across all 15 exposed loss branches, WPO, length-discount, and
f-divergences passes after half promotion and again on the final stable-helper
source (21 passed, 105 deselected). This slice is a compatibility smoke test,
not an independent mathematical oracle for every exposed loss variant.

Run scripts from the Posttrain root with the research Python at
`/home/hammad/projects/trl-gdpo-capo/.venv/bin/python` and candidate TRL on
PYTHONPATH. Real-model and renderer scripts also use isolated renderer and
dependency targets `/tmp/trl-math-renderers`, `/tmp/trl-math-renderers-deps`,
and `/tmp/posttrain-mathdeps`; their installation leaves the project venv and
production pins unchanged. Curated aggregate JSONs alongside this report retain
source/raw hashes. The broader campaign remains active: native multi-turn
rollout extraction, renderer repair, recovery, distributed/fused execution,
GDPO/CAPO complete model runs, and other exposed objective variants remain.
