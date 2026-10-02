# Native Trainer and near-zero KL: a reproduced failure and matched correction

2026-09-30. The real Trainer optimizer loop now passes the tested multi-turn
SAMPO-style credit, accumulation, frozen-behavior clipping, and combined KL
gradient checks on both models. The investigation reproduced a numerical KL
failure and corrected it in published TRL source candidate
`9f0825046ae3509a6be804d74a93fb89d5dc695e`. Production wheels and pins are
unchanged. This is supplied-trace numerical qualification, not fresh Verifiers
collection or evidence of improved task learning.

## Reproduced KL cancellation

Let `d = log p_ref - log p_actor`. The selected legacy sampled estimator is
`k3(d) = exp(d) - 1 - d`; its actor-logprob derivative is `1 - exp(d)`.
Using `expm1(d)-d` improves the value relative to subtracting numbers near one,
but still cancels two nearly equal values near zero. Autograd also subtracts
one from a rounded exponential in the derivative.

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/small_kl_grid.py` uses Decimal80 to derive the value and
derivative from the exact FP32 inputs supplied to the actual loss. Actor and
reference selected log-probabilities are nonpositive. It sets policy credit
to zero and beta to one, isolating KL; an excluded token has a 112-nat reference
gap and missing old score. Both token and sequence importance-sampling paths
are tested, retaining `use_bias_correction_kl=False`.

| Requested positive logprob difference | Before value relative error | Before gradient relative error |
|---|---:|---:|
| 1e-4 | 0.0313% | 0.0118% |
| 1e-5 | 0.0705% | 0.1485% |
| 1e-6 | 9.09% | 4.66% |
| 1e-7 | 45.8% | 20.8% |
| 1e-8 | 100% (zero) | 100% (zero) |

The grid retains actual typed differences and Decimal references, avoiding an
input-rounding confound. Before, 22/28 cases miss the predeclared relative
tolerance 1e-4. After, all 28 pass. These tiny individual KL terms are not a
measurement of the active run's total penalty or proof of its learning failure.

The correction evaluates `d²/2 + d³/6 + d⁴/24 + d⁵/120 + d⁶/720` for
`|d| <= 0.01`, with a differentiable polynomial. Its omitted terms are far
below the tested FP32 tolerance; FP64 boundary controls also pass at 1e-10.
Elsewhere the existing `expm1(d)-d` calculation remains. Because `torch.where`
evaluates both branches, the unused polynomial receives zero on large inputs
to prevent overflow from contaminating backward. Excluded deltas remain masked
before either calculation. This preserves the estimator, beta, clipping,
importance-sampling choice, and bias-correction setting.

## Actual Trainer experiment

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py` instantiates `GRPOTrainer` and calls
`train()`. It supplies one complete frozen group of two multi-turn traces:
an assistant check, an injected calculator response, and a final correct or
incorrect answer. There is no live tool parser or environment collector in
this fixture. The renderer owns serialization and sampled masks. Opposing
first/final-turn credit tests local sequence-ratio gradients; excluded tokens
carry poison credit 99, which must have no effect on the direct loss.

Settings: cached immutable Qwen3.5-0.8B / LFM2.5-1.2B-Thinking weights,
seed 42, rank 4, alpha 8, q/v adapters, dropout 0, LR 1e-4, constant scheduler,
beta 0.01, legacy k3 correction flag false, three iterations of one frozen
group, microbatch 1, accumulation 2, gradient norm cap 1, and deterministic
math attention. Each model runs in BF16 and FP32. Before and after input hashes
match in all four arms; step-zero losses and initialization procedure match.
The baseline source is `18e89c58bee70d25f1231cbe0dc4a865540d6fe5`.

The independent Python-float64 calculation checks the surrogate value and
sampled-logprob derivative. Its derivative is propagated through the same
model graph and compared with actual parameter gradients. A second control
propagates the trainer's exact score derivative: it matches the trainer's
parameter gradient exactly in every case, before and after. Additional hooks
record gradients at the FP32 returned logits and the raw output head, exposing
the precision boundary rather than attributing every difference to loss math.

| Model / base precision | Strict derivative checks before | After | Largest parameter relative error before → after |
|---|---:|---:|---:|
| Qwen / BF16 | 5/6 | 6/6 | 0.0135793 → 0 |
| Qwen / FP32 | 6/6 | 6/6 | 1.095e-6 → 1.142e-6 |
| LFM / BF16 | 4/6 | 6/6 | 0.0060925 → 0 |
| LFM / FP32 | 6/6 | 6/6 | 1.439e-6 → 1.418e-6 |

Before correction, sampled-logprob derivative differences of at most 7.45e-9
can produce larger adapter-gradient differences after lower-precision backward.
The independent near-zero KL failure and matched correction support fixing
the loss calculation rather than loosening tolerance. They do not prove that
all BF16 sensitivity is eliminated or resolve every prior task-trace failure.
The remaining old AutomationBench discrepancy must be rerun separately.

After correction, all 24 microbatch checks pass. Largest loss error is 4.51e-8,
and every excluded sampled-logprob gradient is exactly zero. Qwen masks 30
injected completion-context tokens per trajectory; LFM masks 25. Tool responses
still condition later answers and retain attention; zero direct target gradient
does not mean eliminating their influence through the causal model.

## Frozen replay reaches clipping

| After-correction arm | Step 1 clip region | Step 2 | Step 3 |
|---|---:|---:|---:|
| Qwen BF16 | 0% | 29.17% | 29.17% |
| Qwen FP32 | 0% | 62.50% | 54.17% |
| LFM BF16 | 0% | 58.52% | 46.02% |
| LFM FP32 | 0% | 58.52% | 46.02% |

These are Trainer-reported microbatch-mean clip-region fractions, not
token-weighted global fractions. They use one deliberately reused behavior
group. A recipe with one update per fresh group evaluates ratios at one and
can show zero clipping despite post-update drift. This experiment establishes
working clipping under reuse; it does not recommend three production epochs.
BF16 and FP32 follow different numerical trajectories, so their clip percentages
are not a quality ranking. All twelve optimizer updates remain finite.

FP32 peak tensor allocation fits the available GPU for these short traces:
3.818 GiB Qwen and 4.633 GiB LFM. This does not imply that long task traces fit.

## Validation and continuation

The fork's focused precision/SAMPO slice passes 82 tests, including 28 tiny-KL
cases and 24 FP32/FP64 series-boundary/large-input controls. All 248 existing
independent policy kernel cases still pass. The grid's BNPO accumulation
invariance control remains a reported mismatch because BNPO normalizes each
microbatch locally; it is not hidden or asserted to be globally invariant.

Artifacts include eight matched native before/after arms, both Decimal KL grids,
and the policy grid. Source hashes, model revisions, input hashes, log history,
and memory are retained. Run the two new tools with the isolated research
Python and exact fork/renderer candidates recorded in the campaign plan.

Next gates: repeat the earlier fresh AutomationBench SAMPO discrepancy with this
candidate, qualify native environment projection/collection and full Trainer
save/resume, extend actual optimizer paths to remaining algorithm branches,
and publish/qualify runtime wheels before changing immutable consumer pins.
