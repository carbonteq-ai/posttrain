# Posttrain-normalized TRL and veRL agreement

2026-09-30. Posttrain settings and admitted logical rollout data define the
algorithm contract. TRL and veRL must translate that contract; choosing a
backend must not silently select different defaults or an extra objective cap.
The initial ordinary parity tests passed while independent boundary checks
failed. Source candidates now repair those discrepancies and a finer shared
KL transition defect. [Final matched-kernel evidence](matched-kernel-results.md)
records 3880 passing Decimal-reference boundary checks, 23 passing shared
score cases and six real-model kernel/gradient arms. Native veRL engine
qualification and production adoption remain open. Earlier findings below
are retained as the audit history.

The manifest (`posttrain-cross-backend-parity/manifest.json`, local archive) records the exact
fork references, source hashes and current Posttrain mapping hashes. Some
mapping/test files contain prior unstaged work. This is evidence for the
current working source, not a packaged release qualification or a claim that
those unrelated changes were committed by this audit.

## What Posttrain normalizes

### 2026-10-01 veRL source repair candidate

An isolated candidate at `/home/hammad/projects/verl-posttrain-parity` starts
from runtime source ef1c37715fa75de5973ae5b3c398383cd7e0093d. It adds opt-in
`sampo_token_credit` instead of changing native GSPO's extra ratio cap;
neutralizes excluded scores before the real PPO wrapper's nonlinear math;
and stabilizes k3 values/gradients with a bounded small-delta series.
No Posttrain runtime pin or production recipe is changed.

The candidate source is committed and pushed as
[36c93a03a41f8eafd5af366a2f5d8e4003672bb1](https://github.com/carbonteq-ai/verl/commit/36c93a03a41f8eafd5af366a2f5d8e4003672bb1)
on `codex/posttrain-math-parity`. There is no wheel release or runtime adoption.

130 fork CPU checks pass:14 Decimal KL gradients,12 BF16/FP16 typed score
checks,10 independent signed-credit cases,18 real-wrapper mask cases,
12 hierarchy cases and63 existing core/loss cases. The12 near-zero KL
gradient failures in the old source close. Replacing only the wrapper with
the original pinned source while keeping candidate kernels fixed reproduces
14 failures among18 mask cases; the candidate wrapper passes all18.
The48 existing Posttrain parity tests also pass on the candidate checkout.
These ordinary SAMPO integration tests still select `gspo`; switching the
normalized mapping and exercising the new name end to end remains required.

The half-score tests found an additional boundary effect:0.01 rounds upward
in half precision and escaped the original0.01 series region, producing
three strict value failures. Extending the series through absolute delta0.05
closes them without changing beta or the estimator's derivative convention.
The analogous TRL transition and finer boundary sweep remain follow-up gates.
Independent references use already-represented score inputs; promotion cannot
recover information rounded away before the loss. GPU model/distributed
qualification and immutable publication/pin adoption remain open.

Agreement means the same resolved algorithm settings, shaped rewards, admitted
prompt groups, sampled-token masks and policy/reference/old scores produce
equivalent credit, loss and gradients. Clipping bounds, normalization population,
loss denominator, sampler correction, KL value and derivative, microbatch
accumulation and optimizer schedule must preserve that meaning. Backend names
can differ; backend-native defaults cannot override it.

Settings translation alone cannot establish mathematical correctness. Compare
actual backend computations against each other and independent scalar oracles,
including opposing token credit, unequal sequence lengths, excluded context,
near-zero divergence and extreme ratios. Numerical tolerances must be stated,
and identical shared errors must not be called a pass against the independent
reference. Actual model and distributed optimizer gates remain separate.

## Ordinary parity suite

The first test environment could not import veRL and skipped. After resolving
the retained CPU parity environment and extracting the exact requested fork
source, all48 existing tests pass, without dependency skips:

    packages/train/tests/test_verl_trl_settings_parity.py
    packages/train/tests/test_verl_dapo_parity.py
    packages/train/tests/test_verl_olmo3_parity.py
    packages/train/tests/test_verl_sampo_parity.py
    packages/train/tests/test_verl_active_sampling_parity.py
    packages/train/tests/test_verl_curriculum_parity.py

These compare Posttrain-generated backend settings and actual advantage/loss
code, correction modes and bounds, scaling, masks, active/dynamic sampling,
curriculum and learning-rate schedule controls. SAMPO covers both mean and
mean/std normalization, shaped rewards, hierarchy evidence, loss and gradients
over a split logical batch. They do not constitute full fresh rollout/model
or distributed-worker qualification.

Existing fixtures explicitly select veRL
`ce8e0430018204b03c009b72bfba3b58968696c7`; the test process imported exactly
that source, not the mutable sibling checkout. TRL uses source candidate
`416b8978053d56bf4bc8674748c652f464999fac`. The runtime profile separately
selects veRL `ef1c37715fa75de5973ae5b3c398383cd7e0093d`/post8. Neither the
primary sibling HEAD nor the older test fixture alone is proof of current
runtime behavior. The boundary extension below measures the runtime pin.

## New numerical boundary evidence

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/cross_backend_policy_grid.py` constructs valid Posttrain
SAMPOSettings and reads beta.005 and bounds.003/.004 from that object.
It executes actual immutable veRL GSPO/aggregation/k3 bodies without importing
distributed runtime, and TRL's actual supplied-score loss. Ordinary mapping
tests above verify how the same Posttrain settings select those bodies.
Policy comparisons use an independent scalar, token-local credit derivative;
near-zero KL uses Decimal80 values/derivatives. All scores are FP32; Decimal
is reference arithmetic, not a production model dtype.

Of18 boundary cases,4 pass and14 fail. The ordinary policy value and gradient
agree exactly. TRL's independent policy derivatives match exactly in all four
policy fixtures; its near-zero KL relative errors are at most1.04e-7.

| Boundary | TRL candidate | Runtime-pinned veRL | Consequence |
| --- | --- | --- | --- |
| Ordinary admitted scores | Loss.0030251145 | Same loss and gradient | Pass |
| Sequence log-ratio12 with opposing credit | Loss81376.8984 | Loss11012.7305 | GSPO caps log-ratio10; gradient max difference40688.6992 |
| NaN old scores only at excluded positions | Finite ordinary loss | Nonfinite loss/gradients | Exclusion is not neutralized before ratio arithmetic |
| Reference gap112 only at excluded positions | Finite ordinary loss | Nonfinite loss/gradients | Excluded KL overflow poisons masked aggregation |
| Near-zero admitted KL | Matches Decimal derivative | Cancellation, including zero gradients | Numerical disagreement despite identical estimator name |

The reference gap and NaNs are diagnostic stress fixtures, not claimed values
from a live training batch. They establish that masking after unstable arithmetic
does not implement the same admitted-token contract. The large ratio is also
an explicit boundary control, not evidence of typical policy drift.

Eleven of fourteen near-zero KL cases fail the predeclared derivative-relative
gate1e-5 on veRL. At gaps around1e-8, veRL returns a zero derivative while the
independent derivative is nonzero. Its existing `expm1(d)-d` cancellation is
the same class previously repaired in the TRL candidate. The absolute gradients
are small; these findings do not establish the cause of poor task learning or
justify changing beta. Preserve KL estimator and bias-correction semantics.

## Repairs and acceptance

Keep native GSPO's capped recipe intact. Posttrain SAMPO needs a distinct
uncapped sequence-ratio loss with token-local credit if native GSPO cannot
represent the selected contract. Give it an explicit fork compatibility gate;
do not silently call the capped recipe equivalent at all ratios. Neutralize
excluded scores before ratio and KL exponentiation. Repair near-zero k3
arithmetic without changing the estimator, beta or reference model.

Generic repairs belong in an isolated veRL fork based on the exact runtime
pin, with tests in that fork and its ledger. Posttrain owns resolution, mapping,
compatibility admission, shared fixtures and evidence. Publish fork corrections
before updating immutable runtime pins. Existing dirty worker mappings are
prior work and must not be implicitly staged with this audit.

Acceptance requires all ordinary and independent boundary cases to agree,
explicit defined empty/admission behavior, GDPO/CAPO normalization/reduction
coverage and matched actual-model accumulated/scaled optimizer steps for both
backends. Then qualify pinned packaged runtimes and distributed behavior. The
48 ordinary passes do not close these gates; the broader campaign remains active.

## Reproduce

From `/home/hammad/projects/rl`, use the retained CPU parity Python under
`/tmp/claude-1000/-home-hammad-projects-rl/9dcbdb8a-3c09-497d-bb22-978c505cddb5/scratchpad/verl-vortex/parity-venv/bin/python`.
Set PYTHONPATH to immutable extracted veRL, isolated TRL source, isolated PEFT
and current common/data/train package source directories. Run the six pytest
files above with `-c /dev/null -p no:cacheprovider -q -rs`. The retained
test output (`posttrain-cross-backend-parity/posttrain-cross-backend-existing-tests.log`, local archive)
shows48 passed and28 warnings in32.42s; Ray deprecation warnings are retained.

Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/cross_backend_policy_grid.py --repository
/home/hammad/projects/verl-vortex --revision ef1c37715fa75de5973ae5b3c398383cd7e0093d
--output PATH` in that environment with the same Posttrain/TRL source paths.
The repository is read-only input to git-show; dirty checkout contents cannot
replace the selected immutable function bodies. Scratch environments and paths
are research setup, not a claimed reproducible production image.
