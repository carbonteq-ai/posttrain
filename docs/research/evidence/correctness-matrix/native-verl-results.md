# Native veRL Qwen training and optimizer correctness

1 October 2026. The earlier SAMPO comparisons used veRL kernels on TRL model
outputs. This milestone runs the real veRL loader, FSDP engine, accumulation,
masked loss and AdamW optimizer. Two confirmed integration defects are repaired
in published source [8778c5d6](https://github.com/carbonteq-ai/verl/commit/8778c5d6e2ddd847d5098a24f4dc882f11ac57b4).
Released Posttrain runtime pins are unchanged.

Current numerical finding: on the complete observed Qwen SAMPO population,
native eager FP16 backward is loss-scale sensitive. Forward-preserving
attention ablations identify intermediate half-gradient casts as a major
contributor. Retaining FP32 intermediates in all six full-attention blocks
reduces the gradient scale gap0.844%→0.268% and update gap7.95%→2.90%.
This is a bounded corrective experiment, not a production kernel replacement
or demonstrated learning-quality improvement. Three-update controls and twelve
fresh episodes now pass their independent checks, but the task's baseline is
already successful. Other-kernel, harder-task and full-model-reference
qualification remain open.

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

## Actual frozen vocabulary-projection derivative references

Three further complete-task native SAMPO updates capture the output projection
in FP16 at scales1,024/65,536 and BF16. For six selected credited positions,
retain the actual raw-logit cotangent, all1,024 hidden-state cotangent
coordinates, and four selected weight columns with all248,320 vocabulary rows.
The corresponding step0/1 adapters and scores reproduce the original native
arms bitwise. All six loss/score/mask checks,72 LoRA matrix checks,288 scalar
dots and independent AdamW(max1.34e-9) pass. Peak Torch allocation is3.95GB
(decimal); instrumentation remains within the8GB card.

The independent projection oracle computes `d_hidden = d_logits @ W` in
CPU float64 over the complete vocabulary. Seventy-two independent scalar
`math.fsum` dots agree with these references within1.70e-21 absolute error.
This reference starts from the actual represented raw-logit gradient, thereby
isolating projection rounding from the preceding score derivative. A separate
whole-head reference uses the stable distribution derivative at represented
tempered logits and actual half-weight values. Neither is an unrounded
full-model gradient reference.

| Selected position | Projection relative L2 error, FP16 scale1,024 | FP16 scale65,536 | BF16 |
| --- | ---: | ---: | ---: |
| Row0, index0 | 0.0343704% | 0.0345619% | 0.0777771% |
| Row0, index29 | 11.8897% | 0.195884% | 0.441773% |
| Row1, index3 | Already-zero input gradient | 189.949% | 0.143477% |
| Row1, index6 | 0.0230884% | 0.0234437% | 0.102990% |

Row1/index3's ideal whole-head hidden-gradient norm over the four selected
coordinates is3.92e-13. Its high-scale projection maximum absolute error is
6.39e-13, so a large relative error here does not establish a material learning
effect. At scale1,024 all its input and hidden gradients are zero; reporting
projection relative error0 would misleadingly suggest accuracy. The other two
most saturated FP16 positions likewise have zero input/hidden gradients at
both scales and whole-head relative error100%, with ideal norms below1.85e-14.
BF16 retains tiny gradients but preceding score cancellation still causes
whole-head relative errors44.86–99.68% at these saturated positions.

For these same six positions, the paired FP16 scale difference is0.000267627%
relative L2 in the raw-logit gradients (1,489,920 coordinates), increasing to
0.00419668% in hidden gradients (6,144 coordinates). Thus the actual projection
amplifies the local scale-dependent discrepancy. It remains much smaller than
the full-population LoRA gradient gap0.844489%; different populations and
gradient spaces prevent assigning the remainder to a specific downstream layer.
Expand capture to all104 credited tokens before attributing the global gap.

External evidence: `qwen-projection-scale1024.json`,
`qwen-projection-scale65536.json`, `qwen-projection-bf16.json`, selected projection
tensors and exact adapter/score controls, `qwen-projection-gradient-oracle.json`,
and exact executed runner/controller snapshots. No production source or
precision/default changes. Whole-population projection sensitivity, corrective
accumulation controls and model-backward isolation remain open.

## Complete credited-population head and projection capture

Expand the same native projection instrumentation to all104 nonzero-credit
tokens in the observed Qwen SAMPO population. Three one-update arms again pass
in FP16(scale1,024/65,536) and BF16: six independent loss/score/mask checks,
72 LoRA matrix checks,288 scalar dots and AdamW(max1.34e-9). Original native
step0/1 parameters and scores remain bitwise identical. All earlier selected
logits, raw/hidden gradients and weight columns also reproduce exactly.
Peak Torch allocation remains3.95GB; raw capture files remain external.

Across all104 positions, FP16 scale sensitivity is0.000769745% relative L2
at the raw-logit gradient (25,825,280 coordinates) and0.00904551% at the hidden
gradient (106,496 coordinates). This establishes amplification through the
projection over the complete credited population, yet remains much smaller
than the final159,744-coordinate LoRA gap0.844489%. Relative norms inhabit
different spaces: this does not quantitatively assign the remainder to one
layer or distinguish model-backward rounding from parameter-gradient
cancellation/aggregation. Those are the next causal boundaries to inspect.

The four-coordinate/full-vocabulary references now cover416 hidden coordinates
per arm. All1,248 scalar `math.fsum` dots agree with independent float64 matrix
references within2.97e-21. Aggregate errors use concatenated vectors rather
than averaging per-token relative errors, which would overweight tiny gradients.

| Arm | Projection-only reference error, relative L2 | Whole-head reference error, relative L2 | Complete raw-gradient zero tokens | Complete1,024-coordinate hidden-gradient zero tokens |
| --- | ---: | ---: | ---: | ---: |
| FP16, scale1,024 | 0.0380959% | 0.0434018% | 21/104 | 39/104 |
| FP16, scale65,536 | 0.0387537% | 0.0429953% | 6/104 | 16/104 |
| BF16 | 0.133143% | 0.310289% | 0/104 | 0/104 |

Higher FP16 scaling retains more tiny derivatives. However, the ideal gradient
L2 norm lost at tokens with all four referenced hidden coordinates zero is only
0.000889621% at scale1,024 and0.00000550768% at65,536. This mass calculation
covers four referenced coordinates, not the complete hidden vector. Zero-token
counts alone would overstate the gradient importance; neither count proves
task-quality damage or improvement.

The actual projection is not always equal to the exact represented-input dot
rounded once to its native half output:66/416 coordinates differ at scale1,024,
115/416 at65,536 and102/416 in BF16. Maximum differences are1.87e-9 for FP16
normalized gradients and7.46e-9 in BF16. This motivates testing accumulation
precision/ordering; it does not identify a formula defect or establish that a
different matrix multiplication improves the full update.

External evidence: `qwen-allcredit-{scale1024,scale65536,bf16}.json`,
all credited-token tensors, `qwen-allcredit-gradient-oracle.json`,
`qwen-allcredit-zero-summary.json`, and exact executed runner/controller sources.
No production source, precision defaults or dependency pins change. Next isolate
projection accumulation controls and downstream backward/cancellation over the
complete population, then return to broader algorithm/family and quality gates.

## Complete decoder-boundary loss-scale trace

Two new native FP16 SAMPO updates capture every decoder-output cotangent over
the complete context/response sequences at scales1,024 and65,536:24 layers
times two trajectories per arm,96 boundary records total. Every unscaled LoRA
gradient matches its original native capture bitwise; initial and post-update
adapter parameters and exported scores also match exactly. Four independent
loss/score/mask checks,48 LoRA matrix checks,192 scalar dots and independent
AdamW(max1.34e-9) pass. The first instrumentation attempt rejects its module
lookup before training because FSDP dynamically subclasses decoder modules;
the corrected wrapper recognizes those subclasses. Failed source/logs remain
external, alongside the final exact-repeat evidence.

The complete boundary trace shows progressive numerical divergence through
backward, rather than only an output-head discrepancy or final optimizer effect.
Layer indices are zero-based. Each boundary denotes the gradient with respect
to that layer's output; going23→22 traverses decoder block23 backward.

| Decoder output boundary | Boundary gradient scale gap, relative L2 | LoRA parameter-gradient gap at that block, relative L2 |
| --- | ---: | ---: |
| 23 | 0.0101537% | 0.181314% |
| 22 | 0.0944382% | No selected LoRA matrices |
| 19 | 0.169466% | 0.740066% |
| 18 | 0.434429% | No selected LoRA matrices |
| 15 | 0.455706% | 0.553177% |
| 11 | 0.602039% | 0.805070% |
| 7 | 0.724136% | 1.04014% |
| 3 | 0.793414% | 0.874294% |
| 0 | 0.820993% | No selected LoRA matrices |

The largest jumps among these boundaries occur traversing blocks23 and19,
which the immutable model configuration identifies as full-attention blocks.
Each block also contains normalizations, residual additions and an MLP; this
does not isolate attention itself or prove a formula defect. The layerwise
LoRA gaps can exceed their activation-boundary gaps because parameter-gradient
projection and cancellation operate in a different space. All original
native linear-gradient references still pass.

Conditioning-prefix hidden gradients are zero at boundary23 but nonzero at
earlier boundaries, where later tokens depend on earlier representations.
This is expected: masking prompt log-probabilities from the objective does
not remove prompt conditioning or its influence on shared model parameters.
The native prompt-score gradient audits remain exactly zero, so these prefix
hidden gradients are not evidence of a masking bug.

External receipts: `qwen-layer-scale1024.json`,
`qwen-layer-scale65536.json`, complete decoder cotangent tensors,
`qwen-layer-gradient-comparison.json`, original-update controls and exact runner
snapshots. No production source/default/pin changes. Next capture the internal
attention, normalization and MLP backward boundaries of blocks23/19 before
selecting a correction, while retaining the broader algorithm/quality gates.

## Internal full-attention block trace and corrective backward control

Two more exact native FP16 updates capture complete internal gradients in
blocks19/23 at scales1,024 and65,536:144 records over both trajectories,
including attention query/key/value inputs and outputs, projection outputs,
normalizations and MLP outputs. All original LoRA gradients, adapter steps0/1
and scores match bitwise; decoder-output captures also match the preceding
layer trace exactly. Four independent loss checks,48 matrix checks,192 scalar
dots and AdamW(max1.34e-9) pass at3.95GB peak allocation.

| Internal cotangent | Block23 scale gap, relative L2 | Block19 scale gap, relative L2 |
| --- | ---: | ---: |
| Decoder output / MLP down-projection output | 0.0101537% | 0.169466% |
| Post-attention normalization output | 0.0285014% | 0.172041% |
| Attention output after output projection | 0.0333351% | 0.175865% |
| Attention output before gating/output projection | 0.0392223% | 0.157954% |
| Attention value input | 0.0401948% | 0.181800% |
| Attention key input | 0.0872843% | 0.417062% |
| Attention query input | 0.292873% | 0.915311% |
| Query normalization output | 0.292959% | 0.915484% |
| Combined query/gate projection output | 0.229385% | 0.957634% |
| Input normalization output | 0.149993% | 0.756499% |

The attention query-gradient route is more sensitive than the incoming
attention-output cotangent. Query normalization output and post-rotary query
gradients have nearly the same gap, narrowing this specific increase to the
attention backward route rather than rotary alone. Norm ratios still do not
prove a formula error or identify a single multiplication/softmax operation.

A research-only custom backward at blocks19/23 preserves native eager
attention forward outputs. Recompute the same represented half logits and
native FP32 softmax distribution; perform internal gradient matrix products,
softmax VJP and grouped-key/value reductions in FP32; cast final query/key/value
gradients back to their native half dtype. The value-gradient branch uses the
actual half probability values, preserving that forward multiplication's
represented inputs. Attention weights are diagnostic outputs with no gradient
contract in this control; it is not a general production attention replacement.

Two controlled native FP16 updates pass all four loss checks,48 matrix checks,
192 scalar dots and AdamW(max1.34e-9), with initial parameters/scores bitwise
native and unchanged3.95GB peak. Scale sensitivity decreases from0.844489%
to0.765690% relative gradient L2 and from7.94874% to7.20128% update L2;
sign flips decrease295→256. This gives causal evidence that internal half
attention backward contributes to the gap, but the two-block control is only
a partial reduction. It does not establish a task-quality improvement.

Against each original arm, the control changes gradients by0.509517% at
scale1,024 and0.249216% at65,536; update differences are5.52681% and2.78135%.
Independent first-Adam sensitivity predicts these parameter differences within
3.96e-11. These are measured changes, not error against a full-model oracle.

The exact candidate class is separately extracted from its executed source
and tested on six small CPU causal/grouped-query cases (FP16/BF16 times three
scales) against independent NumPy distribution/gradient equations at native
represented values. Native forward outputs/probabilities match bitwise;
final-half q/k/v derivatives pass the recorded precision bounds. Four central
finite differences validate the independent ideal float64 equations, not the
discrete half-forward derivative. That oracle remains separate from supported
half training and full-model quality claims.

External evidence: `qwen-block-gradient-comparison.json`, native block captures,
`qwen-attention-vjp-comparison.json`, both control receipts/adapter/gradient
exports, `qwen-attention-vjp-independent-reference.json` and exact executed
sources. No production precision/default/pin changes. Next isolate the remaining
attention blocks and distinguish internal softmax casts, matrix-product
underflow and subsequent model/parameter-gradient cancellation; include native
BF16 and broader algorithm/family/quality qualification before adoption.

## All-attention-block intermediate-cast ablations

Seven further native one-update SAMPO arms cover all six full-attention blocks
(3/7/11/15/19/23). Six FP16 arms pair scales1,024 and65,536 under three backward
policies, followed by a native BF16 arm retaining FP32 intermediates. All retain
the native half-forward scores and distributions. Fourteen independent
loss/score/mask checks,168 LoRA matrix checks,672 scalar dots and independent
AdamW(max1.34e-9) pass; each initial parameter/score export matches native
bitwise. Peak Torch allocation remains3.95GB.

| Backward policy across all six blocks | FP16 gradient scale gap, relative L2 | Update scale gap, relative L2 | Gradient sign flips |
| --- | ---: | ---: | ---: |
| Original native eager backward | 0.844489% | 7.94874% | 295 |
| FP32 products and retained FP32 intermediates | 0.268428% | 2.89572% | 86 |
| FP32 products with native intermediate half casts | 0.843785% | 8.12159% | 311 |
| Native half products and intermediate casts, explicit softmax VJP | 0.851509% | 8.15353% | 317 |

The second policy retains FP32 probability cotangents, softmax-score
cotangents, scaled-score gradients and repeated-key/value gradient sums until
the final query/key/value cast. The third uses FP32 matrix products but
reinstates half casts after probability/scaled-score gradients and repeated
key/value products. The last uses native half matrix products too. Its explicit
softmax VJP can differ from Torch's native kernel, so it is not an assumed
bitwise baseline. These controls isolate the value of retaining intermediate
precision rather than simply replacing matrix multiplication.

Keeping FP32 intermediates produces a substantially smaller scale discrepancy;
FP32 products with half intermediates do not. This supports intermediate
quantization/underflow as a major contributor in this eager path. It does not
assign the effect to one cast or prove equality to a full-model mathematical
oracle. Final half casts, other model backward paths and parameter-gradient
cancellation remain active and require further isolation of the residual gap.

Against native at the same scale, retaining FP32 intermediates changes FP16
gradients0.862168% at1,024 and0.251741% at65,536, with update differences
8.16158% and2.47199%. The BF16 arm changes gradients1.98483% and updates
11.0602%, with493 sign flips. BF16 has no scaler in this arm: its changes
demonstrate sensitivity to intermediate precision, not FP16 range overflow.
Independent first-Adam sensitivity predicts all seven parameter-difference
vectors within3.67e-11. None of these change metrics is a task-quality verdict.

The exact executed ablation class also passes18 small independent CPU cases:
three policies times FP16/BF16 times three scales, with causal masking and
grouped-query reduction. NumPy VJP references explicitly model each intermediate
rounding policy; maximum relative derivative differences are0.0256215% FP16
and0.231374% BF16, including final half-output rounding. This is a bounded
equation/implementation reference, not native full-model derivative truth.
The earlier ideal float64 finite differences remain a separate mathematical
calibration rather than a proposed training dtype.

Evidence: `qwen-attention-ablation-comparison.json`, seven native receipts and
adapter/gradient exports, `qwen-attention-ablation-independent-reference.json`,
and exact executed sources in the external manifest. No production source,
precision defaults or dependency pins change. Next test repeated optimizer
updates and fresh matched rollouts, then compare supported native kernels and
remaining backward boundaries before proposing a qualified correction.

## Three-update trajectories and fresh conditional-model rollouts

Six native trajectories apply three updates each: original and retained-FP32
attention backward in FP16(scales1,024/65,536) and BF16. All18 updates pass
36 independent loss/score/mask checks,432 LoRA matrix checks,1,728 scalar dots
and AdamW(max4.11e-9). Steps0/1 reproduce each earlier native/candidate
parameter and score export bitwise. Peak Torch allocation remains3.95GB.
This deliberately reuses one frozen observed population, its old scores,
credits and sampler correction; it is not an online rollout/refill loop.

| Update | Native FP16 parameter-displacement scale gap | Retained-intermediate FP16 gap |
| --- | ---: | ---: |
| 1 | 7.94829% | 2.89572% |
| 2 | 5.75952% | 2.06189% |
| 3 | 5.14437% | 1.87748% |

These relative L2 gaps consistently use the scale1,024 displacement norm.
Earlier first-step sensitivity used the65,536 norm, explaining its slightly
different7.94874% figure without changed tensors. First-step gradient gaps
under the same low-scale normalization are0.844918% native and0.268428%
candidate. From step2 onward, weights, clipping and optimizer moments differ;
later gradient-vector comparisons are trajectory differences rather than pure
loss-scale errors at a common model state.

At step3, all four FP16 trajectories and the retained-intermediate BF16 arm
have exactly zero current parameter gradients. The first trajectory has88
credited tokens among133 sampled positions; the second has16 among165. Clipped
fractions66.1654% and9.69697% therefore cover all104 credited tokens, although
many sampled positions remain nominally unclipped with zero advantage. Adam
still applies a finite parameter update through existing momentum. Native BF16
retains a nonzero step3 gradient(norm0.00127673) because the second row remains
unclipped. All respective independent loss/gradient/optimizer references pass.
This is a concrete frozen-population/tight-clipping recipe effect, not proof
that Adam, masking or the clipping formula is defective. Repeated optimizer
movement alone must not be called fresh policy learning.

Fresh native Verifiers/AutomationBench collection then compares initial,
native-step3 and candidate-step3 adapters in both precisions, using seeds8400
and9400, a512-token request budget and four turns. The actual conditional Qwen
model preserves native parameter names; copied FP32 LoRA masters match exactly,
and a full-fixture rescore reproduces both native exported rows with zero
maximum error for every loaded adapter. No custom backward is needed during
inference. The initial collection attempt stops before generation for a missing
environment import; restore the exact catalog-pinned environment11f4d712 from
Git outside the repository, verifying all499 source files against Git blobs.
No dirty sibling environment edits are imported. Failed logs remain retained.

| Precision / adapter | Positive task reward | Truncations | Sampled tokens across two episodes |
| --- | ---: | ---: | ---: |
| FP16 initial | 2/2 | 0/2 | 317 |
| FP16 native step3 | 2/2 | 0/2 | 317 |
| FP16 retained-intermediate step3 | 2/2 | 0/2 | 317 |
| BF16 initial | 2/2 | 0/2 | 344 |
| BF16 native step3 | 2/2 | 0/2 | 317 |
| BF16 retained-intermediate step3 | 2/2 | 0/2 | 327 |

All12 episodes have no trace errors and pass native sampled-token/log-probability,
branch, excluded-mask, Posttrain projection and independent hierarchical/token
SAMPO-credit audits. Every two-episode updated prompt/generated-token path
differs from its initial counterpart; equal token counts do not imply identical
generation. Reward is already at2/2 initially, so these tiny samples cannot show
a quality improvement or an unseen-task/generalization result. They establish
valid fresh behavior under the measured adapters, not a production qualification.

External receipts: `qwen-three-update-comparison.json`, six native arm receipts
and step0–3 exports, `qwen-three-update-rollout-summary.json`, six collection
and audit receipts, immutable environment provenance and exact executed runners.
No production defaults/pins change. Next compare supported native attention
kernels and harder tasks with room to improve, isolate residual backward
sensitivity, and continue broader algorithms/families and full worker gates.
