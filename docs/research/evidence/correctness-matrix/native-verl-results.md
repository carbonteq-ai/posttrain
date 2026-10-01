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

### Rendered multi-turn masks, padding and clipping

Native veRL now also executes the controlled conversation construction used
by the earlier TRL probe: a system/user prompt, assistant continuation, injected
calculator result and final assistant answer. The shared prompt has26 tokens;
responses have42 and45 tokens, with12 and15 sampled assistant tokens.
The shorter response is right-padded. Renderer masks exclude tool output and
serialization tokens; inactive positions carry deliberately large credit99.
This is controlled supplied conversation evidence, not a freshly generated
AutomationBench trajectory or proof of task learning.

| Native rendered arm | Applied updates | Independent loss/gradient checks | Sampled tokens clipped on second update |
| --- | ---: | ---: | ---: |
| BF16, ordinary fallback | 2 of2 | 4 of4 | 10 of27 (37.0%) |
| BF16, FP32 region | 2 of2 | 4 of4 | 17 of27 (63.0%) |
| FP16 scale1024, ordinary fallback | 0 of2 | 4 of4 | 0 of27 |
| FP16 scale1024, FP32 region | 2 of2 | 4 of4 | 17 of27 (63.0%) |

All16 independent loss and sampled-score derivative checks pass. Gradients
at prompt predictions, excluded tool/header positions and response padding
are exactly zero. Jagged rows are checked individually after native padding
conversion. Maximum loss error1.59e-8, score-gradient error4.22e-9 and
applied AdamW parameter error2.26e-9 remain below their existing gates.
Peak Torch allocation is about4.22GiB. Uncorrected FP16 keeps parameters
finite by skipping both updates; it fails the execution qualification.

Behavior/reference scores stay frozen across two attempts. The first attempt
has ratio1 and no clipping. On the second corrected FP16 attempt, the two
sequence ratios are0.991054 and1.029557. Under bounds0.997/1.004, seven
negative-credit tokens in the first row and ten positive-credit tokens in the
second row activate clipping:17/27. Independent derivatives verify those
branches as part of the loss checks. This demonstrates native clipping after
policy movement on a reused population; it does not select production bounds,
establish a preferred clipping rate, or resolve fresh-rollout learning quality.

BF16 with the FP32 region also clips17/27 but has different second-attempt
ratios0.965099/1.013024. Ordinary BF16 has ratios1.003219/1.004107 and clips
10/27. Those actual probability differences remain visible: objective agreement
and finite updates do not imply identical trajectories across precision regions.
No matched initial adapter/Adam-state native TRL comparison is inferred.

The external archive's `results/rendered-native/` retains the fixture, four
final arms, the initial BF16 probe, executed runner/builder copies and hash
manifest. `working/native_verl_rendered_run.py` is local experimental tooling.
Only this written report is committed; production runtime adoption and native
LFM/Gemma, fresh task collection and full backend trajectory parity remain open.

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

## Complete collected Qwen task population and singleton synchronization

The next native FSDP2 probe uses complete observed Qwen AutomationBench traces,
without changing rewards or supplied tokens:1,119 shared prompt tokens,289/420
response tokens and133/165 sampled assistant actions. Both observed task rewards
are1; Posttrain's SAMPO credit computation still produces104 nonzero local
token credits. This direct math probe deliberately bypasses reward-constant
group admission. It does not show that the production worker would train this
population, or measure new task quality.

The first run fails inside native FSDP2 backward before an optimizer result:
`FSDPParam` lacks `_unsharded_param`. Independent minimal GPU controls isolate
an interaction between unused separately sharded branches and synchronization
deferral. Four ordinary-sync cases pass; the same four deferred-sync cases
fail, covering frozen/trainable branches and CPU-offloaded/device parameters.
An unused branch alone is insufficient to cause the failure. The native
no-sync post-backward path is the failing combination in this Torch2.13 runtime.

The veRL candidate keeps ordinary synchronization enabled for a singleton
data-parallel group, where no cross-rank communication can be saved. Multi-rank
deferral remains unchanged. Six new regressions fail before repair and pass
afterward; the entire12-test synchronization suite passes. Four real GPU tests
verify BF16/FP16, offload/device and two-microbatch gradients against a separate
closed-form linear derivative, preserving zero unused-branch gradients. Existing
two-rank Gloo FSDP1/FSDP2 update equivalence also passes. Multi-rank conditional
unused-branch execution remains a separate gate.

With the repair, all three full-task Qwen arms execute two applied updates.
Each uses FSDP2 CPU offload, gradient checkpointing, FP32 LoRA masters,
rank4/alpha8 q/v targets, LR1e-4, microbatch1 accumulated over two rows,
SAMPO bounds0.003/0.004, beta0 and frozen per-token sampler corrections.

| Complete task arm | Applied updates | Maximum loss /score-gradient error | Maximum AdamW error | Active clipping on update2, row1 /row2 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary FP16, scale1024 | 2/2 | 9.93e-10 /7.50e-12 | 2.20e-9 | 66.165% /0% |
| FP16, scale1024, diagnostic FP32 delta region | 2/2 | 8.39e-10 /6.35e-12 | 2.23e-9 | 66.165% /0% |
| Ordinary BF16 | 2/2 | 1.75e-10 /1.36e-11 | 2.24e-9 | 0% /0% |

All12 independent loss/score checks pass; context and excluded response-score
gradients are exactly zero. Peak Torch allocation is3.67GiB; device reservations,
desktop and driver memory are additional. First-update ratios are1 in every arm.
Second-update first-row geometric ratios are1.006149/1.004889/0.997912 for the
three rows in the table respectively. The FP16 upper crossing clips only
positive local-credit positions,88/133, rather than every token in the row.
BF16 gives a different policy movement and no active clipping on this update;
that is not independently a failed mathematical check or a quality result.

Ordinary FP16 now applies updates at scale1024 on this real population, whereas
the earlier short supplied-token populations overflowed at the same scale.
This does not invalidate those failures or qualify arbitrary contexts: it
demonstrates their data dependence and rejects a blanket claim that Qwen FP16
always fails. The explicit FP32 delta-region control also passes here but is
not required to avoid overflow on these particular traces. Its score/update
trajectory differs, so adoption still needs broader derivative and learning
evidence. No production precision policy or dependency pin changes.

The original FSDP2 failure log and step0 adapter, eight minimal sync controls,
the full-task fixture/source traces, three applied-update receipts and exact
runner snapshots remain under the external correctness archive. Runtime repairs
and production regressions belong in the fork; experimental tools and raw
receipts are not committed.

The synchronization repair is published as
[`d8e472db822f2916ed81a408b8d28192be95e678`](https://github.com/carbonteq-ai/verl/commit/d8e472db822f2916ed81a408b8d28192be95e678).
The three Qwen arms have bitwise-identical initial FP32 adapters. A native LFM
FP16 SAMPO regression applies two updates, passes48 aggregated matrix-gradient
and192 scalar-dot checks, matches native manual gradients exactly and AdamW
within2.14e-9 at2.06GiB peak allocation. The new native receipts were produced
before the source commit and record its then-parent plus the corrected engine
hash; the external publication mapping verifies that hash against the published
commit. This distinguishes the failed original source from the corrected runs.

## Native TRL SAMPO comparison on the same Qwen tasks

Six two-update TRL loops execute actual SAMPO loss, conditional Qwen backward
and AdamW on the complete shared task population and exact native initial
adapters. Two use actual TRL scoring; four substitute veRL's exact row-wise
score helper/full head, with or without deliberate next-forward weight alignment.
The bounded external loop is not the full Trainer/worker lifecycle. Beta is0,
local credits and masks are unchanged, and reward-constant admission is bypassed.

| Qwen SAMPO arm | Initial score mean/max absolute difference from veRL | Parameter max difference after update1 /update2 |
| --- | ---: | ---: |
| Actual TRL BF16 | 0.00384128 /0.0811722 | 1.99446e-4 /3.94702e-4 |
| Actual TRL FP16 | 0.000415897 /0.00711083 | 1.92980e-4 /2.29952e-4 |
| BF16, veRL helper/full head | Exact | 2.91e-11 /1.52242e-4 |
| FP16, veRL helper/full head | Exact | 2.91e-11 /7.49298e-5 |
| BF16, helper/full head/common next-forward weights | Exact | 2.91e-11 /1.86e-9 |
| FP16, helper/full head/common next-forward weights | Exact | 2.91e-11 /1.86e-9 |

All twelve applied TRL/control updates pass independent loss/score/mask and
Adam checks: maximum errors1.06e-9/1.26e-11/2.31e-9, with excluded score
gradients exactly zero. Native veRL gradient-capture reruns reproduce all three adapter
snapshots bitwise in both precisions, with96 aggregated LoRA matrix checks and
384 scalar dots. When the helper, full head and next-forward weights match,
both updates' entire unscaled preclip gradient snapshots are bitwise identical
between TRL and veRL, in BF16 and FP16. This extends matched-state model-gradient
agreement to complete Qwen task traces with sparse SAMPO local credit.

An optimizer-only replay again isolates CPU/CUDA rounding: identical captured
initial parameters and gradients reproduce each first optimizer result bitwise.
The maximum FP32 difference is2.91e-11. Here it crosses four BF16 rounding
boundaries in the BF16 arm and22 FP16 boundaries in the FP16 arm. Unlike the
earlier LFM BF16 example, Qwen BF16 is not exempt from this effect. Copying
native weights before the next forward removes the resulting gradient mismatch
while retaining the control's own Adam moments. These are causal controls,
not naturally identical trajectories or task-quality evidence.

The full-head helper controls peak at about4.54GiB Torch allocation. One
allocation-pressure warning recovers and both updates complete; do not treat
an allocator retry as a failed training result. Exact sources remain TRL
`5d4f9ad3c5f5d51b1ea50b827d82fdd379231dbf` and veRL
`d8e472db822f2916ed81a408b8d28192be95e678`. No production scoring policy is changed.

## Default FP16 loss scale and measured sensitivity

The tested veRL scaler uses native initial scale65,536 and growth interval400;
the earlier bounded successful probes explicitly used1,024. Two complete-task
default-scale arms now apply both updates: ordinary FP16 and the diagnostic
FP32 delta region. All eight independent score/loss/mask checks pass, with
maximum AdamW error2.27e-9. Each arm traces1,620 backward outputs and72 actual
gated-normalization input gradients, with no nonfinite values or predicted
FP16 overflows. Native scale stays65,536 on both updates.

| Default-scale Qwen arm | Maximum independent norm-input derivative | Maximum native/reference relative error | Norm-input dtype |
| --- | ---: | ---: | --- |
| Ordinary FP16 | 35,597.37 | 0.0392493% | FP16 |
| Diagnostic FP32 delta region | 36,734.90 | 0.00000923% | FP32 |

The reference uses actual represented inputs, weights and incoming cotangents
in the independently evaluated gated-RMS derivative described above. CPU
reference arithmetic does not change model training precision. Both maxima
are below FP16's65,504 limit; the earlier short-fixture prediction above1e6
remains a real, data-dependent overflow case. This is not arbitrary-context
or automatic-scale-growth qualification.

The first tracing attempt runs out of memory inside diagnostic boolean indexing
of the full LM-head gradient, requesting7.82GiB. It is preserved as a failed
instrumentation attempt. Bounded million-element reductions remove that
allocation. A separate untraced default-scale rerun matches the traced adapter
exports bitwise at steps0,1,2; peak Torch allocation remains3.67GiB. Thus these
measurements do not rely on an unverified assumption that hooks are harmless.

Despite finite updates at both scales, changing1,024 to65,536 changes the first
normalized gradient by0.844489% relative L2. There are295 sign flips among
159,744 coordinates; four coordinates are zero only at1,024 and two only at
65,536. The sign-flipping magnitudes are at most8.65e-7. First-update parameter
L2 difference is7.94874%, and the maximum coordinate difference is1.96619e-4;
after two updates the maximum reaches3.41969e-4. Starting weights are exact.
The high-scale gradient-capture rerun also matches the traced first-update
parameters exactly and passes24 more matrix/96 scalar-dot checks.

The independent first-step Adam equation predicts all parameter differences
within3.37e-11. This establishes a material loss-scale sensitivity within native
half backward, amplified by Adam at small coordinates; it is not an optimizer
formula defect or an instrumentation artifact. Both optimizers pass their own
reference. The origin of the normalized-gradient difference inside half backward
still needs isolation; these checks alone do not identify underflow versus other
rounding or establish which scale improves learning. Lower scale is not shown
to be a harmless substitute merely because it prevents overflow elsewhere.

External receipts include `qwen-native-sampo-parity-summary.json`,
`qwen-defaultscale-norm-summary.json`,
`qwen-defaultscale-gradient-sensitivity.json`, native/TRL arm receipts,
gradient/adapter snapshots and exact executed runners. Raw correctness tools
remain outside Git. Default-scale TRL execution, upstream backward-precision
isolation, broader algorithms/families and fresh held-out learning remain open.

## Captured Qwen output-head underflow and saturated-softmax cancellation

Two one-update native FP16 runs capture six actual credited token positions
at scales1,024 and65,536. Raw logits and unscaled score cotangents are identical
between scales. Both instrumentation runs reproduce their original step0/1
adapters bitwise. A tiny standalone CUDA replay of the native per-row
log-softmax and half-temperature division reproduces all captured raw-logit
gradients bitwise. This locates a real numerical difference before model
backward; it does not yet account for every downstream LoRA sign flip.

The independent distribution derivative is
`g_score * (indicator(target) - probability) / represented_temperature`.
CPU float64 evaluates the reference at the actual represented tempered logits;
selected-coordinate finite differences agree within2.24e-14. Float64 is an
oracle only. The supported training targets remain BF16 and FP16.

| Token position | Target probability | Native gradient relative L2 error, scale1,024 | Scale65,536 | Ideal maximum gradient magnitude |
| --- | ---: | ---: | ---: | ---: |
| Row0, index0 | 0.0573421 | 0.0188715% | 0.0188703% | 9.23e-5 |
| Row0, index29 | 0.9998940 | 0.429667% | 0.0511309% | 1.18e-8 |
| Row0, index86 | 0.9999999976 | 100% | 100% | 2.68e-13 |
| Row1, index3 | 0.9999997357 | 100% | 8.20557% | 2.38e-11 |
| Row1, index6 | 0.8198275 | 0.0364780% | 0.0364670% | 1.61e-5 |
| Row1, index14 | 0.9999999974 | 100% | 100% | 2.35e-13 |

At row1/index3 the low-scale native raw-logit derivative is entirely zero;
the higher scale recovers six vocabulary coordinates. At row0/index29,
lost ideal gradient L1 mass decreases from0.33675% to0.07363%. Vocabulary
zero counts alone would exaggerate the problem: most coordinates carry
negligible probability mass. The fully lost saturated-token derivatives are
tiny in absolute terms; no measured task-quality loss is attributed to them.

Three local controls separate the stages. Promoting logits after half-temperature
division preserves represented forward values but does not consistently improve
gradients: high-scale row1/index3 error becomes30.7132%, versus native8.20557%.
Removing all half gradient casts improves ordinary-token errors to approximately
3e-8–8e-8 relative L2, yet the same FP32 log-softmax backward still has31.8341%
error at row1/index3 and77.1115%/88.1624% at the two most saturated positions.
Thus half underflow is not the only effect. Subtraction of nearly equal values
in the selected-target derivative also loses precision in FP32.

An independent FP32 distribution derivative computes the target's complement
by summing non-target probability mass directly, avoiding `1 - p_target`.
Its maximum relative L2 error against the float64 oracle is1.97e-7 across
all twelve scale/token cases, including saturated tokens. This is a useful
reference and a corrective-backward candidate, not a production implementation:
it bypasses native autograd and final half-gradient casts. Promoting before
temperature also changes represented forward logits and is not an equivalent
forward control. No head precision policy or backend dependency pin changes.

External evidence: `qwen-head-scale1024.json`, `qwen-head-scale65536.json`,
selected head tensors and adapter controls, `qwen-head-gradient-oracle.json`,
and `qwen-head-precision-stages.json`, with exact executed runner snapshots.
Next: evaluate a stable derivative through the real frozen output projection
and model backward, check both primary precisions, and measure how much of the
full normalized-gradient and Adam sensitivity it explains before adoption.

## Stable selected-score backward through native Qwen updates

Three new complete-task native SAMPO arms apply one update each: FP16 at
scales1,024 and65,536, plus BF16. An external custom-autograd control preserves
the native row-wise log-softmax forward exactly and computes the selected-score
derivative using FP32 non-target probability mass. The derivative returns to the
native half dtype; the temperature division and model backward remain native.
Initial weights and all exported initial token scores match the corresponding
original arms bitwise. This deliberately changes only the scoring backward.

All three updates apply without overflow or skipping. Six independent
loss/score/mask checks,72 LoRA linear-gradient checks and288 scalar-dot checks
pass. Independent AdamW maximum error is1.34e-9. Peak Torch allocation is
2.44GB (decimal), within the local8GB card.

| Arm | Gradient difference from native, relative L2 | Sign flips | Update difference, relative L2 |
| --- | ---: | ---: | ---: |
| FP16, scale1,024 | 0.285196% | 78 | 3.08441% |
| FP16, scale65,536 | 0.241620% | 78 | 2.79333% |
| BF16 | 2.04642% | 454 | 10.5256% |

These are differences, not error against a full-model high-precision oracle.
The independent first-step Adam sensitivity equation predicts every changed
parameter coordinate within3.71e-11. BF16's larger changes reinforce that a
mathematically sound local derivative can materially change a half-precision
update; they do not show which update improves task quality.

The stable control's FP16 scale1,024→65,536 gradient gap is0.871751% with296
sign flips, compared with the original0.844489%/295. Its update gap is7.90205%,
compared with original7.94874%. Thus fixing local selected-target cancellation
does not remove the overall scale sensitivity. Do not adopt this control as a
demonstrated remedy or attribute the entire gap to saturated-token cancellation.
Final half-gradient casts, temperature backward, vocabulary projection
accumulation and the remaining model backward still need isolation.

Receipts: `qwen-stable-head-scale1024.json`,
`qwen-stable-head-scale65536.json`, `qwen-stable-head-bf16.json`, corresponding
gradient/adapter snapshots, and `qwen-stable-head-comparison.json`. Exact
executed wrappers and inherited runners are retained in the external manifest.
No framework source, precision defaults or dependency pins change. Next compare
the actual frozen output projection's half versus FP32 gradient accumulation
while preserving represented forward scores and retaining both primary dtypes.
