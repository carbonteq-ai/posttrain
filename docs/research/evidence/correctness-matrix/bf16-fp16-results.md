# Supported precision qualification: BF16 and FP16

2026-09-30. BF16 and FP16 are the primary qualification targets. FP32 model
runs are diagnostic references. This follow-on extends the real Trainer probe
to FP16 with the actual framework precision helpers, dynamic loss scaling, and
optimizer-update checks. Passing an unscaled loss derivative is insufficient
to certify the scaled optimizer path.

## Current verdict on the supplied multi-turn fixture

| Model / precision | Initial scale | Applied updates / Trainer steps | Largest scaled optimizer-gradient error | Verdict |
|---|---:|---:|---:|---|
| Qwen0.8B BF16 | None | 3/3 | 1.04e-7 | Tested checks pass |
| LFM1.2B BF16 | None | 3/3 | 0 | Tested checks pass |
| LFM1.2B FP16 | 1024 | 3/3 | 1.32e-7 | Tested checks pass |
| Qwen0.8B FP16 | 1024 | 0/3 | Nonfinite; optimizer fenced | Fails update gate |
| Qwen0.8B FP16 | 128 | 3/3 | 0.008168 | Fails strict independent gradient gate |
| Qwen0.8B FP16, FP64 loss diagnostic | 128 | 3/3 | 8.68e-8 | Diagnostic checks pass |

The last row keeps the model and its raw output head in FP16. Only the small
loss-side logprob tensors are promoted to FP64; logits scoring remains FP32.
It is an isolated diagnostic, not a production selection or a proposed
replacement for the supported precision matrix. No existing run setting,
production default, wheel, or immutable pin was changed.

All six unscaled microbatch derivative checks pass in every arm. The stricter
whole-update gate additionally requires three applied, finite, nonzero updates
and optimizer gradients within 1e-4 of the independent, scaler-aware reference.
The failing rows remain in the artifacts rather than being labeled successful
because `global_step` advanced or the loss was finite.

## Loss-scale failure versus mathematical failure

The framework's default FP16 initial loss scale is 1024. On this Qwen fixture,
the scaler backs off 1024 → 512 → 256 → 128 across the three attempts. Every
attempt has nonfinite gradients before the optimizer; Accelerate skips all
three updates. Adapter weights remain exactly unchanged and finite. The
unscaled independent derivative controls remain finite and correct throughout.

Starting Qwen at scale 128 permits all three updates without any scaler skip.
LFM completes all three at 1024 without a backoff. This establishes a
model/fixture-dependent initial-scale issue, not a universal threshold or an
argument that FP16 always fails. A longer run may adapt after initial skips,
but a three-step run can finish with zero learning. Global steps and applied
optimizer updates must therefore be reported separately.

The earlier direct-only Qwen FP16 failure must still be rerun on its exact task
trace under the native precision setup. The native short fixture's finite
unscaled gradients do not certify longer contexts or disprove that earlier
failure. Using autocast/scaling and framework precision helpers is necessary
before attributing a direct harness failure to the production setup.

## Why the gradient reference must include the scaler

Let `c` be the independent derivative of loss with respect to sampled
log-probabilities. In exact arithmetic, `Jᵀc` equals `Jᵀ(Sc)/S`, where `S` is
the loss scale. In finite-precision model backward, scaling changes which tiny
intermediate gradients survive underflow. The unscaled reference can itself
discard information that the actual scaled backward preserves.

The probe now computes both references, sums each over the two microbatches,
and applies the independently calculated gradient norm cap before comparing
with the real optimizer's unscaled, clipped gradient. FP16's scaled optimizer
gradient differs from the unscaled reference by up to 1.41% in the Qwen arm;
that mismatch alone does not establish incorrect scaler behavior.

A separate control uses the trainer's exact sampled-logprob derivative with
the actual scale. It matches the real optimizer gradient exactly for Qwen
FP16 at scale 128, including the step where the independent reference differs
by 0.8168%. Thus accumulation, scaling/unscaling, and clipping implement the
coded derivative in this fixture. The discrepancy appears when a tiny
loss-side rounding difference (sampled derivative errors <=7.45e-9) propagates
through the half-precision backward path.

With FP64 loss-side arithmetic, the Qwen scaled independent comparison passes
at <=8.68e-8, while the difference from the unscaled FP16 reference remains.
This supports precision sensitivity rather than a sign or accumulation error.
It does not prove every FP16 gradient is accurate or establish a production
performance tradeoff. The instrumented runtime contains several diagnostic
backward passes and is not a throughput benchmark.

## Executable scope and controls

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py` uses the real Trainer loop and the
same supplied multi-turn trace group as the preceding native report: two
assistant turns separated by an injected calculator response, poisoned
excluded credit, and three iterations of a frozen behavior group. The model
revisions, seed 42, rank 4 / alpha 8, q/v LoRA, LR 1e-4, beta 0.01, and
deterministic attention remain fixed. Base weights use the selected precision;
trainable adapters remain FP32.

For FP16, the runner invokes the framework's trainable-parameter guard, logits
upcast hook, FP32 logprob trainer wrapper, and configured initial-scale helper.
The unchanged precision-runtime source SHA-256 is
`59bb0ab06ae99865579290f6099f64d8b468bdaff7107ca560cc1dfff750462e`.
TRL source is `9f0825046ae3509a6be804d74a93fb89d5dc695e`; renderer source is
`1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`. Final artifacts retain runner,
trainer, input, and model identities; nonfinite numeric telemetry is stored as
JSON null with finite/skip flags preserved.

The precision test module passes all 21 cases in the isolated research runtime.
The ordinary root environment reports 13 passes and eight optional ML skips;
those skips are not qualification evidence. Scoped Ruff and diff checks pass.

For reproduction, use the campaign's isolated Python/PYTHONPATH and add
`--dtype float16 --initial-scale 1024` or `128`. The diagnostic adds
`--loss-float64`. BF16 uses `--dtype bfloat16` and no scaler. The supported
framework scope currently allows TRL FP16 online RL; other operations still
have their existing qualification restrictions, which this research harness
does not expand.

Next: prioritize BF16/FP16 across remaining algorithms and actual rollout
collection; reproduce the old long-trace Qwen failure; measure loss-scale
adaptation and skipped-update budgets; and compare numerical corrections
against learning outcomes and throughput before changing production recipes.
