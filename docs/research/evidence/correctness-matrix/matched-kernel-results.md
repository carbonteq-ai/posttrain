# Matched SAMPO kernels and sampled KL transition accuracy

Reviewed 1 October 2026. Posttrain selections remain the algorithm authority.
This milestone checks numerical values and gradients against independent
references and against both backend kernels on identical model scores.
Production runtime pins and training recipes are unchanged.

Follow-up: [native veRL Qwen evidence](native-verl-results.md) now exercises
the actual padded FSDP1 loader, accumulation and optimizer in BF16 and
FP16(scale1). The kernel-only scope of the experiments below is unchanged;
full matched native-backend trajectories and other engines remain open.

## The additional arithmetic defect

The first repaired kernels used a sixth-order polynomial through absolute
log-ratio 0.05. A finer sweep found shared value or derivative errors just
outside that boundary. At relative tolerance 1e-6, 268 of the expanded3880
backend checks fail. The earlier narrower2332 grid found the same268 failures.
Maximum value error is1.4712e-6; maximum gradient error is1.1564e-6.
These are small but confirmed numerical defects, not evidence that they
explain poor task learning. TRL and veRL agree on the error: parity alone
does not establish correctness.

For d = reference log probability minus actor log probability, the sampled
k3 term is exp(d) minus1 minus d. Its derivative with respect to d is
exp(d) minus1. `expm1(d)` improves the forward exponential subtraction,
but its ordinary autograd derivative still subtracts one from exp(d).
Outside the short polynomial region that subtraction remains sensitive to
rounding. Promoting scores cannot recover information already rounded away.

Both forks now use a tenth-order Taylor expansion through absolute d0.25,
with the unused polynomial branch restricted to zero before evaluation.
The omitted value and derivative terms are below2e-12 relative in exact
arithmetic over that interval. FP32 evaluation still rounds normally.
All3880 tested represented FP16/BF16/FP32 input cases pass the unchanged
1e-6 relative value and gradient gates against Decimal80 references.
The sweep includes immediate represented neighbors around0.01,0.05 and0.25.
It checks promoted score arithmetic, not FP64 model training.

Published source candidates:

- [TRL5d4f9ad3](https://github.com/carbonteq-ai/trl/commit/5d4f9ad3c5f5d51b1ea50b827d82fdd379231dbf): focused SAMPO tests92 passed.
- [veRL10ad6bab](https://github.com/carbonteq-ai/verl/commit/10ad6babade57546139554a67bc8469118627748): dedicated math/hierarchy slice87 passed.

No beta, KL estimator, legacy derivative convention, clipping bound or
production pin is changed. These published source commits have no new
release-wheel qualification.

## Backend boundary checks

The expanded shared score grid has nine signed-credit policy cases and
fourteen small-KL cases. It selects Posttrain SAMPOSettings with beta0.005,
asymmetric sequence bounds0.003/0.004 and fixed masks. Policy gates require
both cross-backend agreement and an independent token-local derivative.
The exact original veRL runtime source ef1c377 fails17/23 cases. The repaired
dedicated `sampo_token_credit` path passes23/23.

The score grid extracts actual immutable `ppo_loss` and kernel bodies.
Already aligned score arrays use small transport stubs; this does not certify
TensorDict padding or distributed transport. The separate130-test veRL
milestone included18 tests of the actual wrapper with real TensorDict data.
Native GSPO retains its existing cap; the dedicated SAMPO kernel avoids
silently inheriting that different objective. Production Posttrain mapping
to the new registered name remains an adoption gate.

## Real model score and gradient checks

The native TRL Trainer supplies two controlled multi-turn traces with opposing
token credit. System, user and tool context stay masked. Every arm uses
rank4, alpha8, q/v LoRA adapters, LR1e-4, beta0.01, microbatch1,
accumulation2 and two passes over the same frozen two-episode population.
Old scores remain frozen. These settings are correctness controls, not a
selected production recipe or a learning-quality experiment.

For every backward call, the harness sends the same differentiable model
scores through both kernels, compares scalar loss and sampled-score gradients,
then backpropagates each score gradient through the same model graph to LoRA
parameters. The FP16 comparison reproduces the actual scaler multiplier.
TRL applies the optimizer updates; veRL provides a loss kernel. This does
not certify a native veRL model engine, optimizer, sampler or distributed run.

The pre-final-transition six arms all pass: Qwen0.8B and LFM1.2B plus the tiny
Gemma4 architecture fixture, each BF16 and FP16, apply twelve updates and
pass24 microbatch checks. Exact final-source repeats and JSON receipts are
retained below. Tiny Gemma4 is architecture coverage; it is not pretrained
Gemma4 scale or task competence.

Final source repeats also pass all six arms (12 applied updates, 24 checks).
Maximum differences across each arm are:

| Model and precision | Applied updates | Loss difference | Score gradient difference | Scaled LoRA gradient relative difference |
| --- | ---: | ---: | ---: | ---: |
| Qwen 0.8B BF16 | 2 | 7.45e-9 | 7.28e-12 | 0 |
| Qwen 0.8B FP16 scale 128 | 2 | 2.24e-8 | 2.91e-11 | 0 |
| LFM 1.2B BF16 | 2 | 7.45e-9 | 0 | 0 |
| LFM 1.2B FP16 scale 1024 | 2 | 0 | 0 | 0 |
| Tiny Gemma 4 BF16 | 2 | 5.96e-8 | 0 | 0 |
| Tiny Gemma 4 FP16 scale 1024 | 2 | 0 | 0 | 0 |

Peak Torch allocations are approximately 2.30 GiB for Qwen, 2.40 GiB for
LFM and 0.55–0.58 GiB for tiny Gemma. These exclude desktop/driver memory.
The manifest (`matched-kernels/manifest.json`, local archive) hashes all retained JSON and
model logs. It includes the old-source boundary failure, the failing
scale-1024 arm, and final immutable source confirmations.

Qwen FP16 at initial scale1024 skips both attempted updates, reducing the
scale1024 to512 to256. It remains a failed qualification. Initial scale128
applies both updates in this supplied-trace fixture. Its unscaled gradient
reference differs by up to1.48%, while a reference reproducing scaled
backward matches exactly. Report both: multiplication before half-precision
backward changes rounding, and scale-dependent success does not qualify
every Qwen FP16 workload or close the query/key-normalization issue.

## Reproduction and open gates

Run GPU arms serially on the shared8GB device. The native harness accepts
`--verl-repository`, immutable `--verl-revision`, `--iterations2`, model and
dtype flags; see the recorded JSON settings, source hashes and runner hash.
`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/kl_transition_grid.py` accepts both repository paths and
immutable revisions. `--working-source` is a development probe; retained
confirmation must omit it. The score wrapper runner additionally requires
`--policy-loss sampo_token_credit --ppo-wrapper`.

Still required: native veRL engine/model/optimizer agreement; complete
population credit and reduction checks for GDPO/CAPO; broader sampler and
fresh AutomationBench learning experiments; accessible pretrained Gemma;
full intended context/module coverage; and reproducible wheel/runtime
adoption. The campaign remains active.
