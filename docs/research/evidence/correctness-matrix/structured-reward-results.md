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
