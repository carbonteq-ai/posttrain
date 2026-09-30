# Native policy gradients across loss reductions

2026-09-30. The expanded native Trainer checks find precision failures that
the independent loss-only grid does not expose. GRPO and DAPO pass three
accumulated BF16 updates on both requested model families. Other branches
have narrower precision boundaries. Keep numerical correctness separate from
task learning and from support for a complete training recipe.

## What was checked

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py` now selects `--objective` among
SAMPO, GRPO, GSPO, DAPO, Dr. GRPO, BNPO and LUSPO. This is research coverage;
GSPO, Dr. GRPO, BNPO and LUSPO are additional upstream loss branches, not new
Posttrain public operations. The DAPO probe selects its loss kernel, not its
full dynamic-sampling/truncation recipe. OLMo3 remains unqualified here: its
coupled configuration requires sampler probabilities and active refill through
vLLM, which this supplied-trace fixture does not exercise.

Each ordinary arm uses three optimizer attempts over one frozen group of two
supplied multi-turn traces, microbatch 1/accumulation 2, rank 4/alpha 8 q/v LoRA,
FP32 adapters, LR 1e-4, zero dropout, fixed base reference, beta .01, sequence
or token ratios as appropriate, bounds .003/.004, and gradient norm cap 1.
Dr. GRPO divides by configured maximum length 256, not observed output width.
These narrow clipping bounds are diagnostic controls, not copied recommended
settings for every algorithm. The actual reward callback returns [0,1].
These outcome rewards are synthetic fixture values, not semantic task scores.
Only SAMPO uses supplied opposing token credit; other branches compute scalar
group-centered, sample-std-normalized advantages in the native trainer.

The independent Python oracle computes each reduction and its score derivative
from double-precision input values, without copying the trainer's tensor graph.
It covers sequence-shared scalar gradients as well as token-local derivatives.
The harness then sends those coefficients through the actual model Jacobian,
accumulates microbatches, reproduces scaling/unscaling and norm clipping, and
compares against the optimizer's gradient. An exact-score control sends the
trainer's own derivative through the same model and scaler to locate errors.

Predeclared tolerances remain loss 1e-5, score derivative 1e-7 and parameter/
accumulated-gradient relative 1e-4. Excluded tool/padding score gradients must
be exactly zero. Reward normalization is checked independently against
`±0.5/(sqrt(0.5)+1e-4)`; all ordinary arms satisfy that fixture invariant.
Initial reductions are additionally checked from independent sampled-token
counts, rather than trusting the trainer's global denominator. Source hashes,
model revisions, input hashes, scales, finite flags, skips, checks and optimizer
movement are retained in the evidence.

## Ordinary precision matrix

The aggregate evidence (`native-policy-branches/summary.json`, local archive) links 33 complete
JSON artifacts and their hashes: 93 attempted updates, 86 finite nonzero applied
updates, 25 passing arms and 8 failed arms. This count includes repeated controls
and diagnostics, not 33 distinct learning recipes. The 18-arm ordinary matrix
below is the primary comparison; follow-ups preserve the failed originals.

Each cell describes three attempted updates. “Gradient miss” retains finite
updates but fails the strict independent parameter gate.

| Objective | Qwen0.8B BF16 | LFM1.2B BF16 | LFM1.2B FP16, scale1024 |
| --- | --- | --- | --- |
| GRPO | Pass | Pass | Pass |
| DAPO loss | Pass | Pass | Pass |
| GSPO scalar sequence loss | Pass | Pass | Pass |
| Dr. GRPO | Gradient miss | Pass | Gradient miss |
| BNPO | Pass | Pass | Pass |
| LUSPO | Pass | Pass | One skipped update; two applied |

This is 15 passing arms out of18, not qualification of every precision/recipe.
The passing ordinary accumulated-gradient errors are at most1.61e-7. All
excluded-token score gradients are zero. The group is collected once per arm
and old scores remain frozen across reuse. Clipping starts at zero and becomes
nonzero later; the prior absence of clipping is not reproduced under reuse.
No fresh generation, benchmark success trend, renderer extraction transport,
distributed execution or fused optimizer is certified by this fixture.

## Failure mechanisms and controls

**Qwen BF16 Dr. GRPO:** at the second update, one microbatch's independent
parameter gradient differs by1.4106%; the accumulated optimizer gradient differs
by1.1529%. Score derivative error is at most2.33e-10 in that microbatch.
The exact-score model VJP matches the actual gradient exactly. It is a strict
precision miss, not a nonfinite update or evidence that the optimizer applies
the wrong coded derivative. Loss-only FP64 on the same BF16 base passes;
an FP32-base diagnostic also passes. Those controls support amplification of
loss-rounding differences in the low-precision backward path. They do not
justify declaring FP64 loss mandatory or changing supported model precision.

**LFM FP16 Dr. GRPO:** all six unscaled microbatch checks pass, while scaled
accumulated gradients miss by0.10747% and0.04194% on updates2/3. Exact-score
scaled controls match. Loss-only FP64 retains the FP16 base and passes.
The model's half-precision backward is not exactly linear in its incoming
coefficient vector: scaling changes rounding and underflow. Therefore an
unscaled VJP alone is an insufficient optimizer reference.

**LFM FP16 LUSPO:** the first scaled backward is nonfinite, the scaler backs
off1024→512, and the optimizer correctly skips the update with zero parameter
movement. The next two updates are finite. Repeating the same fixture from
initial state at scales512 and128 passes all three attempts. Initial gradient
norm is26.25, versus2.664 for GRPO and.1025 for Dr. GRPO. Length-weighted
reduction increases backward magnitude before the final norm cap can help;
unscaling and clipping correctly happen after backward. A scale qualified for
SAMPO/GRPO does not qualify every reduction. No source correction is made from
this local scale result; scale must be qualified for intended context/coverage.

**Qwen FP16 controls:** GRPO and DAPO at scale1024 skip all three attempts
with no adapter movement. At scale128, GRPO passes three applied updates;
DAPO applies three but retains scaled independent-gradient misses of0.15038%
and0.13418%. Exact-score controls remain below1.32e-7. This prevents treating
“Qwen FP16 at128” as a qualified setting for all objectives.
The corresponding DAPO FP64-loss diagnostic passes all three updates while
keeping the base FP16 and scale128; it supports the same rounding-sensitivity
hypothesis without closing the standard FP32-loss qualification gate.

Repeated Dr. GRPO direction controls preserve exactly the earlier optimizer
records and input hashes. At Qwen's worst accumulated miss, gradient cosine
is 0.99993367 and maximum absolute difference 7.63e-5. LFM's worst cosine
is 0.99999944, with maximum absolute difference 1.53e-5. These nearly aligned
gradients constrain the interpretation: a strict numerical miss is observed,
but a reversed learning direction or explanation of failed reward learning is
not established. Cosines use floating reductions and may differ slightly from
one in otherwise matching controls. A default SAMPO one-update BF16 smoke
control also passes after the runner extension.

## Reduction mathematics and interpretation

For valid token loss terms `ell_it`, group rows B, valid counts n_i and
accumulation windows K, GRPO uses a row mean: `sum_i(sum_t ell_it/n_i)/B`.
Dr. GRPO replaces n_i with fixed maximum length L. DAPO divides the whole
population sum by its total valid-token count, with each microbatch retaining
that global denominator. LUSPO averages row sums without the length division.
Scalar GSPO uses the sequence geometric ratio and its shared derivative;
SAMPO's token credit uses the repaired local derivative instead.

BNPO divides by valid tokens inside each microbatch. At microbatch1 it matches
GRPO's accumulated row means in this fixture; unequal lengths at larger
microbatch widths change the weighting. This is its defined reduction, not a
failed derivative. The 248-case independent kernel grid (`native-policy-branches/policy-grid-native-expansion.json`, local archive) still passes, while
the deliberately tested full-batch/split invariant fails for BNPO by.19087
in maximum coefficient difference. The other four reductions pass that
invariance check. A new configured-length argument in the oracle preserves
the existing grid default while supporting actual Dr. GRPO L=256.

The new failures illustrate the scope gap: tiny score errors can cross rounding
boundaries inside BF16/FP16 model backward even when kernel checks pass. The
strict reference gate should remain visible. It does not by itself establish
that these percent-level gradient perturbations explain poor long-run reward;
direction/magnitude controls and held-out behavior are needed for that claim.

## Reproduction and remaining work

Run serially from `/home/hammad/projects/rl`, using candidate TRL
`9f0825046ae3509a6be804d74a93fb89d5dc695e`, renderer
`1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`, Torch2.13.0+cu130,
Transformers5.16.1 and PEFT0.21.1:

    PYTHONPATH=/home/hammad/projects/renderers-lfm-mask:/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft:/tmp/trl-math-renderers-deps:/tmp/posttrain-mathdeps /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --model qwen08 --dtype bfloat16 --objective dapo --output .posttrain/state/correctness/qwen08-native-dapo-bfloat16.json

Change model/objective/dtype and use distinct filenames. FP16 requires an
explicit `--initial-scale`; diagnostics add `--loss-float64` or use an FP32
base. Temporary dependency paths must be reconstructed elsewhere; this is not
production lockfile qualification. Follow-ups add direction telemetry and a
signed reward check that identifies each full token trace; two one-update
controls verify it. Aggregate evidence also checks signed rewards across all
ordinary traces using their distinct valid-token counts. Numerical settings
and losses remain fixed; artifacts retain exact hashes and repeated controls.

Production pins/defaults are unchanged. Remaining gates include loss-rounding
qualification, longer intended contexts and all-linear adapters, checkpoint/
scaler restoration, native Verifiers collection, OLMo3 sampler/refill semantics,
other preference branches, and task-quality comparisons across seeds.
