# Native veRL Qwen training and optimizer correctness

1 October 2026. The earlier SAMPO comparisons used veRL kernels on TRL model
outputs. This milestone runs the real veRL loader, FSDP engine, accumulation,
masked loss and AdamW optimizer. Two confirmed integration defects are repaired
in published source [8778c5d6](https://github.com/carbonteq-ai/verl/commit/8778c5d6e2ddd847d5098a24f4dc882f11ac57b4).
Released Posttrain runtime pins are unchanged.

## Repairs

The padded eager path initially required `flash_attn.bert_padding` merely to
convert the batch into native nested tensors. It now reuses veRL's existing
Torch indexing implementation when FlashAttention is absent. Missing
transitive dependencies inside an installed package still raise errors.

The next run failed because veRL replaced Qwen's forward with its legacy
packed implementation, which references instance attributes absent from
Transformers 5.16.1. The ordinary padded path now retains native Transformers
forwards when sequence parallelism and fused kernels are disabled. Specialized
packed forwards and the vision offload corrections remain selected in their
respective paths. This does not qualify newer Transformers for packed Qwen.

Fifteen fork CPU tests pass: four fallback/value/gradient checks, eight
retained padding cases, two padded/packed forward-selection regressions,
and one retained decoder-forward case.
An exact-source negative control substitutes both original functions from
10ad6bab while keeping the current test environment fixed: eight tests fail
and seven pass. The repaired source passes all fifteen.

## Actual native runs

The real checkpoint is Qwen3.5-0.8B at immutable model revision
`2fc06364715b967f1860aea9cf38778875588b17`. The engine loads FP32 master
weights, computes with the selected BF16 or FP16 precision, and keeps LoRA
parameters in FP32. This differs from the earlier half-base TRL fixture;
the results must not be called identical adapter trajectories.

Each population contains two supplied eight-token rows: four context tokens
and four responses. Each response includes an excluded slot with deliberately
large advantage99, plus opposing positive/negative token credits. This is a
numerical probe, not a rendered AutomationBench task or fresh generation.
The config is rank4, alpha8, q/v adapters, dropout0, LR1e-4,
microbatch1, accumulated population2, beta0.01 and SAMPO bounds0.003/0.004.
The same population and behavior/reference scores are frozen for two updates.

| Native arm | Applied updates | Independent score checks | Largest AdamW parameter error |
| --- | ---: | ---: | ---: |
| Qwen BF16 | 2 of2 | 4 of4 | 2.29e-9 |
| Qwen FP16 initial scale1 | 2 of2 | 4 of4 | 2.26e-9 |

The loss and sampled-score derivatives are compared with an independent
scalar SAMPO reference. All context-score gradients are exactly zero;
excluded response slots also match the reference's zero gradient.
Assistant-action likelihoods still depend on context through attention;
those dependency gradients are expected. The maximum
score-gradient error is below1.8e-8. Gates remain loss error1e-5 and
score-gradient error1e-6. A successful optimizer-step hook captures already
unscaled and clipped native gradients and prior Adam state; detached reference
arithmetic predicts the ensuing parameter update. Maximum parameter error
is below2.3e-9 against a1e-6 gate. This checks AdamW arithmetic, not an
independent derivative through every model layer.

Both arms peak at about4.20GiB Torch allocation on the8GB GPU. Desktop,
driver and allocator reservations are additional. A world size of1 makes
FSDP use NO_SHARD: this is the actual native engine, not multi-device sharding
or distributed equivalence qualification.

## FP16 overflow remains a separate issue

Current Transformers5.16.1 at scale1024 applies zero of two attempts;
the scaler backs off1024 to512 to256. The full immutable upstream
Transformers [d6c1e71](https://github.com/huggingface/transformers/commit/d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a), reporting5.19.0.dev0,
also skips both attempts at1024 and128. Both runtimes apply updates at
scale1 on this short fixture. Thus the upstream change is not shown to cause
this fixture's success, and a large loss scale is not qualified by these runs.

The upstream runtime is tested in an isolated overlay with its required
Hub/tokenizer dependencies; no production environment or lockfile changes.
The earlier independent zero/small-vector normalization failure remains valid:
this eight-token run does not deliberately produce those vectors. The isolated
bare normalization helper still fails eight direct half-input cases because
the upstream correction promotes inputs in the caller. Do not confuse a
helper's unpromoted-input experiment with that caller's numerical contract.

Choosing scale1 for this probe demonstrates an executable FP16 engine path;
it does not establish a universal recipe, underflow safety for longer contexts,
or a repair for every skipped update. Loss scaling is data and model dependent.

## Evidence and reproduction

### Native context extension

The next external-runner experiment extends the supplied population beyond
eight tokens. The128-token population has96 context and32 response tokens;
the256-token population has192 context and64 response tokens. Both contain
two rows, alternating positive/negative credit and excluded response positions
with deliberately large credit99. The context-score check covers every
non-response prediction position, including the final unused prediction.
Other model, adapter, accumulation and objective settings remain as above.
These are supplied-token fixtures, not rendered conversations or task learning.

| Native upstream5.19 arm | Applied updates | Independent loss/gradient checks | Peak Torch allocation |
| --- | ---: | ---: | ---: |
| BF16,128 tokens, ordinary fallback | 2 of2 | 4 of4 | 4.34GiB |
| FP16,128 tokens, scale1024, ordinary fallback | 0 of2 | 4 of4 | 4.34GiB |
| FP16,128 tokens, scale1024, FP32 region | 2 of2 | 4 of4 | 4.41GiB |
| FP16,256 tokens, scale1024, FP32 region | 2 of2 | 4 of4 | 5.05GiB |

All16 score checks pass: maximum loss error2.07e-8 and sampled-score
gradient error2.93e-9; context-score gradients are exactly zero. The six
applied updates match the independent native-gradient AdamW calculation
within2.25e-9. The uncorrected FP16 arm retains finite parameters because
the scaler skips both attempts; that is a failed update qualification despite
passing loss checks. Desktop/driver memory is additional to Torch allocation.

The external archive's `results/context-extension/` retains four JSON/log
receipts, the executed runner and a hash manifest. The research runner is
`working/native_verl_context_run.py`, with explicit context/response lengths.
No runner or raw receipt is committed. These tests extend the bounded
precision evidence; full rendered masks, matched initial adapter/Adam states,
fresh trajectories and native LFM/Gemma remain required.

### FP16 gated-delta precision control

The next matched experiment locates and removes the high-scale failure on this
fixture. On the immutable upstream runtime, the first affected layer22 gated
normalization has a finite analytical input gradient with maximum1,095,103.
FP16 can represent at most65,504: the oracle predicts67 overflowing elements,
and native backward produces exactly67 nonfinite elements. The oracle uses
the actual quantized inputs, weights and incoming cotangent. For
`y = w * silu(g) * x / rms(x)`, it evaluates
`dx = v*w*silu(g)/rms - x*mean(v*w*silu(g)*x)/rms^3` independently.
Later layers with already-poisoned cotangents are marked invalid references;
they are not counted as evidence for this derivative.

The research control promotes q/k/v before the Torch delta-rule fallback,
disables autocast within that fallback, and retains its output in FP32 through
gated normalization. The following output projection remains under native
FP16 execution. This preserves a region of the backward graph that needs
more than FP16's range, instead of lowering the loss scale.

| Same fixture, initial scale1024 | Applied updates | Nonfinite output-gradient hooks | Largest independent AdamW error |
| --- | ---: | ---: | ---: |
| Upstream5.19 baseline | 0 of2 | 1648 | No optimizer step |
| Upstream5.19 promotion only | 0 of2 | 1648 | No optimizer step |
| Upstream5.19 disable autocast only | 0 of2 | 1648 | No optimizer step |
| Upstream5.19 FP32-region control | 2 of2 | 0 | 2.31e-9 |
| Current5.16.1 FP32-region control | 2 of2 | 0 | 2.30e-9 |

Each control passes all four independent loss/score/context-mask checks.
The FP32 norm input derivative agrees with the independent oracle within
1.02e-7 relative error across both runtimes; all traced cotangents stay finite.
The upstream control still predicts67 elements beyond FP16 range at the first
norm, but retains all of them as finite FP32 values. This establishes the
precision region as a working correction for this fixture. Subsequent single
component ablations show both changes are necessary here: promotion only keeps
the norm gradient finite but later backward still overflows in the autocast
delta calculation; disabling autocast only leaves67 norm-input nonfinite
elements at its FP16 output boundary. Neither ablation applies an update.
These checks do not qualify fused FLA kernels or every context and scale.

### Independent delta-state and derivative checks

`qwen_delta_rule_grid.py` compares the actual chunked fallback with a separate
token-by-token state equation. For each token, it decays the prior state,
subtracts the value already predicted by the key, writes the beta-weighted
residual as an outer product, and reads the new state with the normalized query.
The reference uses neither chunking nor a triangular solve. A central
directional finite difference also checks the reference's autograd derivative.
Reference double arithmetic is confined to these tiny tensors, not model training.

On each Transformers runtime,24 fixtures cover FP16/BF16-represented inputs,
lengths1/3/5/9 with chunk size4, zero/small/ordinary queries and nonzero initial
states. Checks cover outputs, final state and derivatives with respect to
q/k/v/decay/beta/initial state. The FP32 region passes all24 on each runtime
against a2e-5 relative-error gate. Ambient-autocast controls stay finite but
miss that FP32-accuracy gate in all24 on each runtime. The upstream maximum
relative error falls from0.00943 to3.64e-7 with the explicit FP32 region.
Ordinary half rounding alone is not a formula bug; these controls establish
which precision the supposedly FP32 recurrence actually executes in.

This grid uses FP32 leaves containing exactly represented half fixtures. It
checks the region before half-boundary casts; it does not establish that every
q/k/v gradient remains representable after casting back to a half model.
The separate native scaled-optimizer runs remain necessary evidence.

The explicit `--gdn-fp32-control` switch is confined to the research runner.
No production monkeypatch, Transformers fork pin or released image changed.
Use the same reproduction command below with `--initial-scale 1024
--trace-backward --gdn-fp32-control` for the current-runtime control, and omit
the final switch for its negative control. The upstream comparison additionally
uses the isolated runtime and immutable Transformers source described above.
The manifest retains the executed runner snapshot and all three receipts.
Production adoption remains gated on full rendered traces, longer contexts,
q/k/v derivative checks and both native backend paths.

The manifest (`native-verl/manifest.json`, local archive) hashes JSON/log receipts, source
identities and the research runner. Execute `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_verl_run.py`
with the published fork on PYTHONPATH and the retained CUDA interpreter:

```bash
PYTHONPATH=/home/hammad/projects/verl-posttrain-parity:/tmp/trl-math-peft:packages/common/src:packages/data/src:packages/train/src /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_verl_run.py --dependency-path /tmp/claude-1000/-home-hammad-projects-rl/9dcbdb8a-3c09-497d-bb22-978c505cddb5/scratchpad/verl-vortex/parity-venv/lib/python3.13/site-packages --model-path /home/hammad/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 --dtype float16 --initial-scale 1 --output .posttrain/state/correctness/qwen08-native-verl-final-float16.json
```

The retained CPU dependency environment provides Ray, TensorDict, Hydra and
einops0.8.2; append it after the GPU runtime so it cannot replace CUDA Torch.
Run one GPU arm at a time.

Still open: identical rendered populations and initial adapter/Adam states
through both native backends; native LFM/Gemma and intended contexts/modules;
packed and distributed paths; FP16 overflow/underflow boundary checks; and
release assets with immutable runtime adoption. The broader campaign remains
active. See the [living plan](../../../plan/8gb-algorithm-correctness-campaign.md).
