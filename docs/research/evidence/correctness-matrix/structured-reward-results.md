# GDPO/CAPO: independent credit, actual gradients, and three-update behavior

2026-09-30. Both cached models passed three direct updates for each algorithm,
including independent credit and gradient references, full-batch versus
accumulated gradients, and direct adapter/Adam save-and-replay. This establishes
the tested numerical path; it does not establish task learning, live judge
quality, native Verifiers collection, or full Trainer recovery.

## Independent normalization and correction

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/structured_reward_grid.py` derives sample-standard-deviation
normalization using Decimal precision 100, independently of Posttrain. It tests
interleaved groups, unequal completion lengths, excluded tokens, weighted
components, and overlapping process-error spans (counted as a union).

All 80 ordinary randomized GDPO/CAPO cases passed before correction. Four of
five initial extreme finite-value cases failed because offsets or intermediate
norms exceeded float range, although final normalized credit was finite.
A sixth extreme case exposed silent zero credit when standard deviation plus
epsilon overflowed. The corrected implementation rescales values and epsilon
together only on overflow, preserving the existing nearby-origin centering for
large common offsets with small differences. All 86 cases now pass at 1e-12.

These are numerical robustness failures at extreme values, not evidence that
ordinary live rewards caused poor learning. Products or sums that overflow
before normalization are outside this grid's validated coverage.

## Actual model experiment

The fixture is a fixed pair: correct `Four.` and incorrect `The answer is five.`
for two plus two. Renderer-owned sampled masks exclude system/user context,
headers, padding, and post-stop separators. GDPO uses outcome/style components
with weights 2/1; CAPO uses outcome weight 2 and error-span weight 1, marking the
sampled word `five`. Credit is prescribed rather than provided by a live judge.

Each algorithm resets initialization and AdamW state. Settings: seed 42,
BF16 base, FP32 rank-4/alpha-8 q/v LoRA, dropout 0, LR 1e-4, weight decay 0,
gradient norm cap 1, beta 0, token importance sampling, clip bounds 0.2,
training mode, deterministic math attention. Group size is 2; accumulated
updates use two microbatches of size 1. Model revisions and source hashes are
retained in the adjacent JSONs. TRL source is candidate
`18e89c58bee70d25f1231cbe0dc4a865540d6fe5`; renderer source is candidate
`1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`. These are isolated research sources,
not newly published runtime wheels or production pins.

| Model / algorithm | Updates passed | Max score-gradient error | Parameter-gradient relative error | Accumulation relative error |
|---|---:|---:|---:|---:|
| Qwen0.8B / GDPO | 3/3 | 7.45e-9 | 0 | 0 |
| Qwen0.8B / CAPO | 3/3 | 1.49e-8 | 0 | 0 |
| LFM1.2B Thinking / GDPO | 3/3 | 7.45e-9 | 0 | 0 |
| LFM1.2B Thinking / CAPO | 3/3 | 1.49e-8 | 0 | 0 |

Credit differs from Decimal by at most 2.22e-16. All adapter updates are finite
and nonzero. All four direct recovery cases reproduce final adapter weights,
Adam state, and replayed losses exactly after saving at step 1 and replaying
steps 2–3. This does not test Trainer callbacks, data order, or lineage recovery.

The first harness arm accidentally used evaluation mode, in which TRL omits
training accumulation scaling. Two microbatches then produced twice the
expected gradient. That invalid arm was corrected to training mode; it was
not a production accumulation bug.

## Why loss and clipping alone hide changes

Behavior scores are refreshed before every update on this fixed pair. Thus
pre-update importance ratios equal one. GDPO's centered credit gives loss zero
while its parameter gradient is nonzero: the two answers have different model
Jacobians. CAPO loss is -0.319457 throughout. Neither constant loss implies
unchanged parameters or unchanged probabilities.

The following is `exp(mean_sampled(log p_after - log p_before))`, the geometric
mean token-probability ratio per answer, measured immediately after each update.
It is not the product of all token ratios, a KL measurement, or a clipped-loss
fraction. Even a value above 1.2 does not prove that the next surrogate will
clip: the loss also depends on token ratios, credit sign, and behavior identity.

| Model / algorithm | Step 1 correct / wrong | Step 2 correct / wrong | Step 3 correct / wrong |
|---|---:|---:|---:|
| Qwen / GDPO | 1.033 / 1.031 | 1.117 / 0.996 | 1.185 / 0.983 |
| Qwen / CAPO | 0.972 / 0.990 | 1.214 / 1.035 | 1.137 / 0.985 |
| LFM / GDPO | 0.988 / 0.980 | 1.092 / 0.952 | 1.203 / 0.954 |
| LFM / CAPO | 1.049 / 1.014 | 1.082 / 0.962 | 1.132 / 0.960 |

The first update sometimes lowers both answers' average probability. This is
not automatically a sign error: shared parameters couple tokens, Adam changes
the direction, and a small local gradient prediction need not control a finite
step. Independent parameter gradients match exactly in these cases. Fresh
rollouts and held-out task outcomes remain necessary for learning conclusions.

## Reproduction and remaining gates

### Additional GDPO weighted-aggregate overflow repair

The prior normalization fix did not cover weighting the normalized components.
Four of nine large finite-weight probes fail before final whitening, either
because a product becomes infinite or because `math.fsum` overflows. The
independent normalized result remains finite. Ordinary component normalization
is not the failure point in these cases.

The framework now retains its ordinary float calculation and, only after
weighted aggregation overflows, recomputes that aggregate and final whitening
using Decimal precision1600. Component scores retain their represented float
values and original epsilon. This is CPU reward arithmetic, not high-precision
model training. A common weight scale was rejected after it erased a tiny
third component left by cancellation of large opposing components.

Five regression cases cover product overflow, sum overflow, exact cancellation
with a subnormal epsilon and preservation of a tiny residual component. An
exact pre-repair function substitution fails all five while14 retained cases
pass. The repaired module passes all19 advantage tests,46 related
advantage/evidence/projection tests, the existing86-case independent grid and
all nine new weighted-aggregation probes. Scoped Ruff and Pyright pass; all
nine import-boundary contracts are preserved.

Raw receipts, rejected scale-candidate source, the negative-control plugin and
source hashes remain in the external correctness archive. Only the source
repair, regression tests and written findings are committed. These deliberately
extreme coefficients are numerical boundary tests; there is no evidence that
this overflow explains current-run performance, and no recipe changed.

From the Posttrain root, run the grid with `.venv/bin/python
${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/structured_reward_grid.py --output <path>`. The model runner
is `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_structured_run.py --model qwen08|lfm12 --output
<path>`, using the isolated research Python and dependencies described in the
campaign plan. Models run serially on the free 8GB GPU.

Validation: 52 reward advantage/evidence/projection/recovery/admission tests
pass; scoped Ruff passes; nine import-boundary contracts are preserved.
Evidence includes baseline normalization failures, the corrected grid, both
final model arms, source/input hashes, and direct checkpoint hashes. Checkpoint
payloads remain machine-local. Full Trainer/Verifiers integration, native
multi-turn masks, distributed/fused paths, nonzero KL combinations, rank/alpha
comparisons, and controlled task-learning runs remain open.

## Native veRL GDPO/CAPO updates on recorded task trajectories

1 October 2026. Extend the direct BF16 checks to the native veRL FSDP2 loader,
CPU offload, checkpointed model backward, accumulation and AdamW in both
primary precisions. The qualified matrix covers Qwen3.5-0.8B and
LFM2.5-1.2B-Thinking, each with GDPO and CAPO. This is native engine coverage
on recorded trajectories, not the complete production worker or fresh learning.
The candidate fork remains `d8e472db822f2916ed81a408b8d28192be95e678`;
production dependency pins do not change.

Use the previously audited native AutomationBench trajectories and masks.
Qwen has1119 prompt tokens, response lengths289/420, sampled counts133/165,
and actual task outcomes `[1,1]`. LFM has1018 prompt tokens, response
lengths872/512, sampled counts741/512, and actual outcomes `[1,0]`; the
second LFM response is truncated. Preserve those observed outcomes rather
than inventing a success/failure pair for Qwen.

The added evidence is deliberately controlled: GDPO combines actual outcome
with negative sampled-token count as an effort component, weights2/1. CAPO
uses actual outcome weight2 and process weight1, with four sampled positions
at the end of the second response marked as error spans. Each span is
duplicated to check union semantics. Those effort rewards and spans are audit
inputs, not live judge results or an adopted reward recipe. Posttrain's actual
`compute_gdpo_advantages`/`compute_capo_advantages` outputs agree with the
independent Decimal100 equations within4.45e-16 over all four fixtures;
excluded positions remain zero.

The native actor selects `token_clip`, consistent with Posttrain's structured
RL mapping, rather than SAMPO's sequence-valued stop-gradient ratio. Settings:
token ratios, row-mean/token-mean reduction, bounds0.2/0.2 (the structured
RL defaults), beta0, rank4/alpha8 q/v LoRA with FP32 masters, LR1e-4,
weight decay0.01, norm cap1, seed42, microbatch1 and global batch2.
The old policy and capped sampler-correction weights are frozen from the
initial model on each fixture. FP16 starts at explicitly selected scale1024;
this does not qualify the native default initial scale65536. Torch SDPA is
used; these runners do not explicitly enable global deterministic algorithms,
so candidate repeatability and cross-precision trajectory equivalence are not
established here.

### Applied updates, masking and actual TRL loss agreement

The final eight-arm matrix completes24 applied updates in27 attempts. It
passes54 independent loss/score/mask checks,576 LoRA matrix checks,
2,472 scalar-dot checks and detached AdamW checks (maximum4.10e-9).
Some scalar checks occur on finite portions of skipped backward attempts;
they are not proof that those attempts had finite complete gradients.
Peak Torch allocation is3.992GB for Qwen and1.450GB for LFM.

| Model / algorithm | BF16 applied updates | FP16 applied updates | Last-update BF16 clipped fractions, row1/row2 | Last-update FP16 clipped fractions, row1/row2 |
| --- | ---: | ---: | ---: | ---: |
| Qwen / GDPO | 3 | 3 | 2.256% /0.606% | 6.015% /1.212% |
| Qwen / CAPO | 3 | 3 after backoff | 1.504% /0.606% | 1.504% /1.212% |
| LFM / GDPO | 3 | 3 | 1.080% /0.391% | 0.135% /0% |
| LFM / CAPO | 3 | 3 | 0.540% /0.195% | 0.135% /0.195% |

Clipping is inactive at the first unchanged-policy forward and becomes active
on some sampled tokens during reuse. These are token-clip fractions at0.2
bounds, not directly comparable with SAMPO's tight sequence-ratio clipping.
All independent score-gradient checks give exactly zero excluded-response and
prompt-score gradients. Prompt hidden states can still receive legitimate
conditioning gradients upstream; masking their loss positions does not remove
their causal contribution to an assistant token's probability.

Replay the actual TRL `GRPOTrainer._compute_loss` on exported native scores,
the exact normalized token credits and reconstructed FP32 frozen correction
weights. Compare both loss and derivatives with an independent scalar token
PPO equation, including signs, clipping and row/accumulation denominators.
Across27 attempt states, TRL's maximum reference loss error is1.42e-15,
gradient error3.47e-18, and excluded gradients are exactly zero. The native
veRL microbatch-loss sum differs from this replay by at most6.79e-8,
within the explicit1e-6 gate. This proves agreement of the tested logical
loss at common scores/credits, not native TRL model/scorer/optimizer parity.

### Sparse CAPO whitening and FP16 backoff

Qwen's two actual outcomes are equal. Four controlled error positions among
298 sampled tokens nevertheless produce distinct CAPO credits: ordinary
tokens receive0.11634556 and marked tokens−8.55139886. This is expected
token-population whitening, not an incorrect normalization sign. With
`p=4/298` and `n=298`, raw scores are2 or1, and the sample standard deviation
is `sqrt(n/(n-1)*p*(1-p)) = 0.11527027`. Dividing positive deviation `p`
and negative deviation `-(1-p)` by `std+1e-4` reproduces both credits exactly.
Sparse process penalties can therefore yield much larger standardized credit
than GDPO's approximately±0.707 row credit in these fixtures.

The initial Qwen CAPO FP16 run stops in an experimental linear-hook assertion
when a backward cotangent becomes non-finite. That hook prevents GradScaler
from finishing, so retain it as a failed harness attempt, not a demonstrated
production loss defect. The corrected external instrumentation records
non-finite cotangents and permits the native scaler's existing skip/backoff
path. A first three-attempt control safely skips at scales1024/512/256;
this control passes skip-safety checks but applies no optimizer updates.

A fresh bounded continuation repeats those skips, then applies three updates
at scale128. All three skipped adapter exports and native scores remain
bitwise initial, as does the aborted harness's initial state. Adam state
stays uninitialized on skips, then its counters advance1→2→3 on applied
updates. The initial three-attempt control and aborted hook attempt remain
separate raw evidence outside the27-attempt qualification matrix. The finite
score-level derivatives and later valid scaled updates distinguish this
backward overflow from a demonstrated symbolic policy-loss error. They do
not yet isolate the first overflowing nonlinear operation or qualify every
precision setting. No learning-rate, advantage or production scaler recipe
is changed from this result.

Retain the fixtures/provenance, failed hook log, native/overflow/backoff
receipts, all adapter/gradient exports, actual TRL replay summaries and exact
executed sources under the external `results/native-collection` archive.
Key summaries: `native-structured-matrix-summary.json` (including the
three-attempt zero-update CAPO control),
`native-structured-backoff-matrix-summary.json` (final eight-arm matrix),
and `native-structured-backoff-recovery-audit.json`.
All tools and raw receipts stay outside Git. Live judge integration, actual
structured-evidence worker transport/admission, nonzero KL, repeatability,
native TRL model/optimizer parity, checkpoint continuity and fresh held-out
task learning remain open.
