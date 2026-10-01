# Native LFM BF16 and FP16 with FP32 masters

1 October2026. Native veRL now loads LFM2.5-1.2B-Thinking, accumulates two
microbatches and applies AdamW updates on the local8GB GPU. FSDP2's native
CPU-offload policy retains trainable parameters and optimizer moments in FP32;
model computation uses BF16 or FP16. This avoids casting trainable masters
to half precision merely to fit the larger model.

The checkpoint is immutable revision
`95053d21d8e0b7ca99421a2127ae39c64f685ff3`. Native veRL source is published
`8778c5d6e2ddd847d5098a24f4dc882f11ac57b4`, with the isolated Transformers
`d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a` runtime and Torch2.13.0+cu130.
No production runtime pin or framework recipe changed.

## Actual native checks

All arms use two supplied rows, q/v LoRA rank4/alpha8, LR1e-4,
microbatch1, accumulated population2, SAMPO bounds0.003/0.004 and beta0.01.
Alternating token credits include excluded response positions with credit99.
Behavior/reference scores remain frozen across two attempts. The8-token arm
has4 context/4 response tokens; the128-token arm has96 context/32 response
tokens. These are numerical fixtures, not rendered task conversations.

| Native CPU-offload arm | Applied updates | Independent loss/score checks | Peak Torch GPU allocation |
| --- | ---: | ---: | ---: |
| LFM BF16,8 tokens | 2 of2 | 4 of4 | 1.00GiB |
| LFM FP16 scale1024,8 tokens | 2 of2 | 4 of4 | 1.00GiB |
| LFM BF16,128 tokens | 2 of2 | 4 of4 | 1.00GiB |
| LFM FP16 scale1024,128 tokens | 2 of2 | 4 of4 | 1.00GiB |

All eight updates are applied without scaler skips. All16 independent
loss/score-gradient checks pass and context-score gradients are exactly zero.
The maximum loss error is2.88e-8 and score-gradient error1.70e-8. The
independent AdamW calculation, using already-unscaled/clipped native gradients,
matches parameter movement within2.13e-9. Masters reside on CPU in FP32;
the three final-runner arms explicitly record FP32 optimizer-moment dtypes.
CPU parameter/state storage, pinned memory, desktop, driver and CUDA allocator
reservation are additional to the reported Torch GPU allocation.

FSDP1's actor path deliberately disables native CPUOffload because of known
gradient-accumulation limitations. These runs select FSDP2's CPUOffloadPolicy,
not that FSDP1 path. Single-rank shard-local tensor references cover the entire
parameter here; they are not distributed optimizer qualification.

## Evidence and remaining limits

The external archive described in [README](README.md) retains the four JSON/log
receipts, executed runner versions and hash manifest in `results/native-lfm/`.
The first BF16 receipt's textual scope incorrectly says FSDP1; its structured
`engine_strategy=fsdp2` and retained executed source establish FSDP2. That
receipt is retained unchanged and the manifest records the correction.
The working probe is `working/native_verl_offload_run.py`; no runner or raw
receipt is committed.

These checks close the initially unexecuted native LFM engine path. They verify
score derivatives and optimizer arithmetic, not an independent derivative
through every model parameter or a proof of microbatch accumulation equivalence.
The subsequent [partition comparison](native-accumulation-results.md) verifies
accumulation against separately measured rows on the8-token fixture and exposes
precision sensitivity when those contributions cancel.
The collected-task checks below extend the masks, unequal lengths and context
coverage. Broader accumulated LoRA-gradient checks, matched native TRL/veRL
initial states, fresh collection after updates, worker admission/refill,
distributed/fused paths and immutable production adoption remain required.

## Native AutomationBench traces and updates

Fresh LFM episodes now execute through the same pinned Verifiers chat harness,
MCP tools and training-client renderer used in the Qwen collection audit. The
office-closure task starts with1018 prompt tokens. All six episodes pass exact
sampled-token/log-probability and final-branch checks, exclude input nodes from
sampled masks, and project through Posttrain's actual SAMPO method. Execution
success here means the harness finished; task reward and truncation are reported
separately.

| Native collection | Task rewards | Truncated episodes | Executed tool calls | Sampled tokens per episode |
| --- | --- | ---: | ---: | --- |
| BF16,512 tokens per request | 0,0 | 2 of2 | 0 | 512,512 |
| FP16,512 tokens per request | 1,0 | 1 of2 | 1 | 741,512 |
| BF16,1024 tokens per request | 0,0 | 1 of2 | 0 | 1024,589 |

At512, BF16 spends the first completion on reasoning and ends the second
completion partway through a tool-call name. With1024 tokens, the first episode
still truncates. The second closes its tool block but includes
`channel: '#general'` inside a Python-style keyword-argument call. That invalid
syntax is not executed. A larger budget therefore removes one truncation
without producing a successful task. Both BF16 groups have zero SAMPO credit.
The failed completed call also demonstrates why zero executed-tool failures
cannot be read as zero attempted tool-format errors.

The first episode starts at the same seed4200 in each arm. Later seeds are
assigned per model request, so additional turns change the second episode's
seed. These two-episode collections are diagnostic observations, not a matched
multi-seed precision ranking or an estimate of production failure rates.
Both BF16 budget-control episodes retain exactly the same first512 generated
tokens as the512-token arm; that prefix equality was checked directly.

The mixed FP16 group is used unchanged for the native update check. Observed
rewards `[1,0]` produce episode credit `[+0.5,-0.5]`. At their common initial
anchor, discounted sparse returns `[0.95,0]` give turn credit `[+0.475,-0.475]`;
the successful episode's later singleton anchor gives0. Thus its first turn
has token credit+0.975 and its second+0.5, while the truncated episode has
-0.975. Independent equations agree with the actual framework calculation.
All741 and512 sampled tokens retain their real credit and sampling scores;
tool observation and template tokens remain excluded.

Two BF16 and three FP16 updates then run through native veRL FSDP2 CPU offload
with gradient checkpointing, FP32 LoRA masters, q/v rank4/alpha8, LR1e-4,
temperature0.8, beta0 and clipping bounds0.003/0.004. The same fixed population
has872 and512 completion tokens, plus1018 prompt tokens. The old training
scores and per-token sampler corrections are computed once and frozen. The
correction is `min(exp(old_training_logp - sampling_logp),2)` on sampled tokens,
matching the selected SAMPO default; it has no lower cap. Native loss checks
compare against a separate scalar reference using credit times the detached
correction weight. KL is disabled in this slice, so it adds no KL qualification.

| Collected population update arm | Applied updates | Independent loss/score checks | Sampler correction range | Peak Torch GPU allocation |
| --- | ---: | ---: | --- | ---: |
| BF16 | 2 of2 | 4 of4 | 0.753701–1.289352 | 2.27GiB |
| FP16,scale1024 | 3 of3 | 6 of6 | 0.967399–1.032524 | 2.06GiB |

No sampler correction reaches its cap. All ten loss and score-gradient checks
pass; maximum errors are4.44e-8 and1.27e-10. Prompt, excluded response and padding
score gradients are exactly zero. Both arms apply every update with finite
parameters, and independent AdamW errors stay below2.12e-9.

Clipping activates on the real task population. BF16's second-update sequence
ratios are1.003878 and0.998837, so neither row clips. FP16's second-update ratios
are1.001199 and0.996364: all512 negative-credit tokens clip. On the third update,
ratios1.004293 and0.994566 clip all1253 sampled tokens. The policy score gradients
are then exactly zero for both rows. Adam still moves parameters, with maximum
change7.74e-5, because its stored momentum continues to act; the independent
optimizer calculation matches that movement. Clipping a gradient does not erase
optimizer history. Native AdamW also applies weight decay0.01, included in the
independent calculation.

These are actual updates on a population collected through native task tools,
but the population is reused. No new rollout has been collected from the
updated adapters, and worker scheduling/admission/refill is bypassed. Passing
the score and optimizer equations does not establish improved task behavior,
full parameter-Jacobian correctness or matched native TRL/veRL trajectories.

The external `results/native-collection/` directory retains collection receipts,
the projected fixture, update receipts, executed source snapshots and hashes.
Working runners are `native_automationbench_collection.py`,
`analyze_family_native_collection.py`, `build_collected_lfm_fixture.py` and
`native_verl_collected_lfm_run.py`. The first update attempt failed at import
because the collection's ANTLR4.13 dependency conflicted with OmegaConf's4.9
grammar. A separate `/tmp/posttrain-native-verl-update-runtime` selects
ANTLR4.9.3 and borrows the existing research libraries; both successful update
arms use it. The failed log is retained. No production dependency or recipe
was changed, and no runner/raw receipt is committed.

## Exported adapter handoff and temperature arithmetic

**Later qualification correction:** the CPU-offload unscale investigation
below invalidates an unconditional FP16 training-correctness verdict for
these stock-scaler trajectories. The loss/score and conditional AdamW checks
remain valid; they did not establish that the optimizer received correctly
unscaled gradients. The first update/export is unchanged by the scalar-ordering
fix, while later adapter states differ. Retain all earlier behavior results
as observations of their exact exported states.

The collected FP16 update was repeated with exports before training and after
each of its three native optimizer steps. All three updates applied, and the
independent AdamW checks again stayed below2.12e-9. Copying every trainable
LoRA parameter into ordinary HF PEFT reproduces the exported FP32 values
exactly; parameter names and sets are asserted equal.

An initial teacher-forced score comparison was deliberately kept at FP32
temperature division. Across the two collected trajectories and four adapter
states, its sampled mean absolute log-probability discrepancy was0.000699–
0.000893, with maximum0.021009. Matching native veRL's temperature division
instead reduces the largest discrepancy to4.77e-7 and every sampled mean to
below6.7e-8. This isolates the difference to temperature arithmetic on these
fixtures, rather than an incorrect adapter export or different model weights.

The padded unfused veRL path casts temperature to the logits' FP16 dtype and
divides before the score operation. FP16 represents0.8 as0.7998046875 and also
rounds the divided logits. The HF generation provider promotes its scores
before its temperature warper. Thus identical model weights can still produce
slightly different distributions. These are absolute score discrepancies;
they are not measurements of PPO current/old drift or evidence that clipping
is incorrectly activated. Sampler correction remains relevant. A shared
FP32-temperature policy and its backward/optimizer regressions are still an
open normalization gate; no production arithmetic was changed in this probe.

Raw exports, audit receipts and runner snapshots remain in the external
`results/native-collection` directory. The export runner is
`native_verl_export_lfm_run.py`, and the audit is
`audit_native_lfm_adapter_export.py`. The independent comparison uses ordinary
HF PEFT teacher forcing, with the same FP16 autocast and selected token labels.

## Fresh behavior after the verified native updates

Four fresh episodes per arm compare exported adapter steps0 and3 on the same
office-closure task, initial prompt, FP16 autocast provider, temperature0.8
and512-token request budget. Seeds4200,5200,6200,7200 are matched by episode;
each later turn adds its turn index, so extra turns cannot shift the next
episode's seed. These are fresh collections after a fixed-population training
experiment, not a complete collect/update/refill training loop.

| Observation | Before updates | After three updates |
| --- | ---: | ---: |
| Successful tasks | 2/4 | 1/4 |
| Truncated episodes | 2/4 | 3/4 |
| Executed tool calls | 2 | 1 |
| Total sampled tokens | 2,410 | 2,277 |

Seed4200 remains successful with the same433-token first response; seed5200
remains a512-token truncation with identical first-response IDs. At seed6200,
the first416 generated tokens agree. The baseline then emits a valid
`slack_send_channel_message` call and succeeds. The updated adapter instead
emits nonexistent `slaychannels_args` and exhausts512 tokens before closing
the tool block; no tool executes. Seed7200 diverges after278 common tokens,
but both arms truncate and fail. Reduced token total reflects the lost
second turn of a successful task, not an efficiency improvement.

All eight episodes pass exact sampled-ID/log-probability/final-branch checks,
zero masks on nonsampled nodes, actual Posttrain projection and independent
episode/discounted-anchor credit calculations. The sampled outcome is worse;
four paired seeds cannot establish expected reward regression or general
model quality. The evidence does show that applied, mathematically checked
updates and active clipping are insufficient to demonstrate better tool use.
The collection remains a research HF provider, not production vLLM.

External receipts are `lfm-adapter-step0-fresh.json`,
`lfm-adapter-step3-fresh.json`, both `*-analysis.json` files and
`lfm-fresh-adapter-comparison.json`. Runners are
`native_adapter_collection.py` and `compare_native_lfm_fresh_adapters.py`.
Next controls should distinguish intermediate-step behavior, repeated rollout
reproducibility, temperature precision and token-budget effects before
attributing this sample to an algorithm defect or changing the recipe.

### Reproducibility and clipping controls

A separate process repeats adapter-step3/seed6200. Its prompt IDs, all512
generated IDs and every sampling log-probability are exactly equal to the
original failing request. This excludes seed drift or run-to-run variation
for that request under this provider; it does not prove broad determinism.

The export audit now also evaluates sequence ratios from matched step0
baselines under FP32 temperature division and native FP16 division. Across
four states/two trajectories, the largest absolute ratio difference is
0.00015034. All eight directional clipping classifications agree at the
selected upper1.004/lower0.997 bounds. After step1 the successful training
trajectory is below its upper bound and the failed trajectory is below its
lower bound; after steps2 and3 both directions clip. Export states are
measured after optimizer steps, so these classifications describe the next
evaluation with frozen old scores, not the gradient used to create that state.

Thirty-two selected log-softmax positions additionally agree with an
independent Python `math.fsum` softmax denominator within2.17e-7. The FP16
temperature discrepancy is real but does not explain different clipping
classifications on this population. This control does not qualify BF16,
packed/fused implementations or the gradients of a proposed arithmetic change.

Intermediate adapters1 and2 also fail seed6200. Their512 generated IDs are
identical to adapter3's, although selected log-probabilities change (maximum
absolute1→2 change0.0424;2→3 change0.0806). All three diverge from the baseline
after416 common first-response tokens. Native masks, selected scores and final
branch reconstruction pass for each intermediate collection. The behavior
change occurs after update1, whose two training evaluations used current/old
ratio1 with no active clipping. Later reuse/clipping is therefore not required
to produce this particular change; that does not establish the cause or rule
out effects of reuse elsewhere.

A separate adapter3/seed6200 control raises the per-request budget from512
to1,024 tokens. Its first512 token IDs **and** sampling log-probabilities are
exactly equal to the shorter run. The first response now ends at625 tokens,
and the environment returns `error: unknown tool 'slaychannels_args'`.
The model then generates a second1,024-token response and truncates. Reward
remains0; total sampled work is1,649 tokens. Thus the larger budget exposes an
actual invalid tool invocation rather than repairing this case. Environment
execution being `ok` means the harness completed, not that the tool or task
succeeded. Native trace invariants pass for this two-turn budget control too.

Receipts: `lfm-temperature-ratio-audit.json` (including32 independent softmax
positions), `lfm-repeat-audit.json`, `lfm-intermediate-control-summary.json`
and `lfm-budget-control-audit.json`, plus their input collections. The new
intermediate summary runner is `lfm_intermediate_control_summary.py`.
These controls justify prioritizing token-level behavior and the full fresh
collection/update loop over another clipping-only diagnostic. They do not
justify declaring SAMPO wrong or selecting a new recipe from one held-out seed.

## Native linear-gradient audit finds an FP16 offload race

The next audit applies the independent linear derivative `dL/dW = D.T @ X`
to every trainable LoRA A/B matrix, using actual saved layer inputs and output
derivatives under the selected autocast. It sums both native microbatches,
divides by the loss scale and compares against native gradients **before**
global norm clipping. This adds a check upstream of the earlier AdamW oracle,
which deliberately trusted the gradients it received.

The stock FP16 run passes all24 matrices on update1, then fails update2. For
the first A matrix, scaled norm0.477781862 should become0.000466584 at
scale1,024; native unscale instead produces489.248627. This is a factor of
1,048,576 relative to the required gradient. A separate B matrix changes
17.487034→17,906.722656 instead of0.017077. The failure repeats in diagnostic
runs. Reading the CUDA inverse scale as a Python scalar before the host
operation makes the run pass, exposing a synchronization-sensitive failure.

Torch2.13 `ShardedGradScaler` uses a scalar replicator whose copies specify
`non_blocking=True`. The inverse scale originates on CUDA; the CPU foreach
unscale operation can consume the host copy before it is ready. A tensor-only
control inserts a10-million-cycle CUDA delay before unscale. Ordinary CPU
gradients fail3/4 iterations, with exactly the same1,048,576 amplification;
the no-delay controls pass. This rules out needing model, LoRA or DTensor
semantics to produce the race. The DTensor arm of that particular delayed
control passes; do not claim it fails universally. No FP64 model is trained.

The veRL source fix stages inverse-scale and overflow scalars synchronously
on CPU if any gradient has CPU storage. The parent scaler still handles
device-only arithmetic and distributed overflow reduction. Five regressions
pass: CPU finite/overflow, delayed CUDA-to-CPU finite/overflow, and CUDA-only
unscale. The new helper is `verl/utils/sharded_grad_scaler.py`; its engine
selection and regression tests are in the fork, not in experimental tools.
The published source is `269fde84d1769469f6b02b186170f420ec353d9e`;
the corrected native receipts ran the same file contents immediately before
that commit. Production wheels and pins are unchanged.

| Native collected-population audit | BF16 | FP16, scale1,024 |
| --- | ---: | ---: |
| Applied updates | 2/2 | 2/2 |
| Adapter matrix comparisons | 48 | 48 |
| Maximum matched-compute gradient error | 0 | 0 |
| Independent scalar dot checks | 192 | 192 |
| Maximum relative difference from FP32 dot accumulation | 0.002441 | 0.000326 |
| Independent AdamW maximum absolute error | 2.12e-9 | 2.14e-9 |
| Peak Torch allocation, GiB | 2.27 | 2.06 |

The matrix equation oracle does not use autograd to construct gradients;
its matched-compute multiplication reproduces actual half-precision GEMM.
The additional Python `math.fsum` dots independently check selected products
and reductions. FP32 dot differences measure precision sensitivity, not an
algorithm defect. This is not an independent Jacobian of the model's
attention, convolution or normalization blocks. All existing loss, score-mask
and applied-optimizer checks also pass in both corrected runs.

Stock and fixed adapter-step0/1 parameters and native scores are exactly
equal. After update2, maximum parameter difference is8.73e-5 and native
score difference is0.082369 on the recorded trajectories. Thus the verified
first-update held-out failure cannot be attributed to this later observed
unscale race. Corrected later-step fresh behavior is still unmeasured.

External receipts include `lfm-chain-fp16-fixed.json`, `lfm-chain-bf16.json`,
`sharded-unscale-delayed.json`, successful synchronous controls and the failed
chain/unscale logs. Runners are `native_verl_lora_chain_run.py`,
`native_verl_unscale_only_run.py`, `dtensor_cpu_unscale_reproducer.py`.
Distributed overflow, restart/scaler state, production runtime adoption,
other Torch versions and nonlinear-gradient oracles remain open. The broad
campaign is incomplete; this repair closes a specific demonstrated race.

### Scaler recovery and corrected fresh collection

Lifecycle regression commit `dc08945ddecf0693ea42bfb1bf0b11312aa2a0b4`
retains the same scaler source and extends its tests from five to eight.
CPU/Gloo and CUDA/NCCL scaling with CPU parameters each exercise a
finite/overflow/finite/finite sequence. The overflow update leaves parameters
and AdamW step count unchanged. Restoring the parameter, optimizer moments
and scaler state after that skip reproduces uninterrupted parameters and
scaler state after each following update, and final AdamW moments exactly.
The CUDA arm deliberately delays the scalar copy. A separate two-rank Gloo
control injects infinity on rank0 only; both ranks skip, back off1,024→512
and apply the following finite update. All eight tests pass. These controls
use FP32 master tensors, not an independently trained FP32 model. They do
not prove native engine checkpoint round trips or multi-GPU offload ordering.

Corrected adapter-step2 is loaded into the same HF provider for four fresh
matched seeds. Task success is1/4, truncations3/4 and sampled work2,277 tokens,
against baseline step0's2/4,2/4 and2,410. Seed6200 still takes the invalid-tool
branch after416 shared baseline tokens. All four new episodes pass exact
native IDs/scores/final-branch, masks, Posttrain projection and independently
computed credit checks. The fixed adapter handoff passes six row/state score
comparisons within4.77e-7 and24 independent softmax positions within1.46e-7.
Fixing numerical correctness has not yielded a learning improvement in this
small sample. The two-update corrected comparison must not be presented as
a same-step comparison against the old three-update population.

After update2, the corrected positive sequence ratio is1.003932007 and the
negative ratio0.993365688. The earlier faulty trajectory's corresponding
ratios were1.004292754 and0.994566187. At the selected1.004/0.997 bounds,
only the negative trajectory clips in the corrected next evaluation; both
clip in the faulty one. The repair therefore changes whether update3 can
receive a positive policy gradient, even though update1's exported state and
the first held-out failure are unchanged. Fresh data and full collect/update
training remain separate requirements.

Receipts: `lfm-fixed-step2-fresh.json`, `lfm-fixed-step2-analysis.json`,
`lfm-fixed-fresh-comparison.json` and `lfm-fixed-adapter-handoff.json`.

The corrected run is also extended through update3. All72 matrix-gradient
comparisons match exactly,288 scalar dots pass, all six score/loss checks
pass and all three optimizer steps apply with AdamW error below2.14e-9.
During update3, the positive trajectory remains unclipped with nonzero
policy derivative; the negative trajectory clips with zero policy derivative.
This differs from the faulty run, where both derivatives were zero.

Fresh corrected-step3 collection on the same four matched seeds succeeds0/4,
truncates4/4 and samples2,048 tokens. No tool executes. Seed4200 now diverges
from baseline after238 common tokens and starts malformed
`slab_send_channel_message`; seed6200 diverges after268 tokens and reaches a
correct tool name with an overlong argument list before truncating. Other
samples also exhaust the request budget. All four new native/projection/credit
audits pass; every episode, anchor and token advantage is zero in this
observed equal-reward one-turn group. No actual scheduler/refill is run here.
The sample provides no improvement evidence, and cannot estimate general
expected task quality. Lower sampled work again reflects lost successful
second turns. A correct numerical update is insufficient for useful learning.

The clipping ratio also needs careful interpretation. For N sampled actions,
the geometric ratio is `r = exp(sum(new_logp - old_logp) / N)`; the product
of their conditional likelihood ratios is `R = r**N`. Independent Python
`math.fsum` accumulation over recorded native scores gives:

| Exported state | Positive geometric ratio, N=741 | Positive conditional product ratio | Negative geometric ratio, N=512 | Negative conditional product ratio |
| --- | ---: | ---: | ---: | ---: |
| After update1 | 1.001199 | 2.43 | 0.996364 | 0.1549 |
| After update2 | 1.003932 | 18.32 | 0.993366 | 0.0331 |
| After update3 | 1.006815 | 153.35 | 0.991719 | 0.0142 |

These are fixed-token native-training-distribution conditional likelihoods,
not measured KL, sampling-provider ratios or fresh success probabilities.
The reduction deliberately normalizes by length. A narrow geometric clipping
interval therefore does not directly bound the total trajectory likelihood
ratio. Also, clipping truncates a surrogate derivative; it does not project
the AdamW update into a hard probability constraint. Update3 still moves the
negative trajectory after its policy derivative is zero, through optimizer
history. Neither observation alone establishes an incorrect objective.
The large total ratios and adverse fresh sample justify evaluating update
reuse, task grammar and fresh-data admission jointly rather than selecting
a recipe solely from clip fractions.

Additional receipts: `lfm-chain-fp16-fixed3.json`,
`lfm-fixed-step3-fresh.json`, `lfm-fixed-step3-analysis.json`,
`lfm-fixed3-fresh-comparison.json`, `lfm-fixed3-likelihood-ratios.json`.

## Native FP16 checkpoint continuity

The native engine constructed `FSDPCheckpointManager` with optimizer and
scheduler, but did not pass its scaler. Extra state saved RNG and scheduler
only. That omitted the FP16 loss scale, growth tracker and growth/backoff
configuration needed to reproduce the next update. This is separate from
the asynchronous CPU unscale race.

Source `d0d7804795dc1254f7309916fce69898387a2a8a` binds the optional scaler
and adds its state to per-rank extra state. An enabled-scaler full restore
checks for scaler state before loading model/optimizer and fails clearly if
it is absent. Explicit model-only loading preserves the fresh scaler;
BF16 and disabled-scaler legacy extra state remain compatible. Five new
checkpoint tests cover those cases and exact scale/growth-counter restore.
Checkpoint, cleanup and scaler slices pass34 tests together. Production
runtime pins remain unchanged.

The real collected-LFM fixture now executes a native checkpoint experiment:
apply update1, deliberately set scale512 with growth tracker1 and interval2,
save LoRA-only model weights with full optimizer and extra state, and apply
update2 uninterrupted. Restore step1 and replay update2. The intentionally
nondefault scale exposes omission rather than relying on default settings
coincidentally matching. This is a controlled scale change, not an injected
native model overflow. Both restore and replay are bitwise exact for all
trainable parameters, native AdamW state, scaler, scheduler and RNG.

A second, fresh process constructs the same model/engine, loads the saved
checkpoint and computes update2 with the same frozen population/old scores.
It matches the uninterrupted result exactly across the same five state
categories. Independent score/loss/context-mask checks pass and AdamW errors
remain below2.14e-9. Peak Torch allocation is2.06GiB; the native checkpoint
with tokenizer/config is6,712,369 bytes. LoRA-only saving assumes the same
immutable pretrained base; this does not test restoring a changed base model.

The control omitting scaler binding reproduces the prior semantics. It
restores parameters, optimizer, scheduler and RNG exactly but keeps
scale1,024/tracker0 instead of512/tracker1, and fails the restore gate. This
verifies the scaler omission independently of task reward or clipping.

The producer performs two logical updates and a replay of update2, then a
fresh process replays update2 again; instrumentation attempt numbers must
not be interpreted as four distinct training updates. Actual save/load is
used, not a hand-copied adapter or fake checkpoint manager. Full-weight,
distributed native replay, BF16/native other-family checkpoints and complete
fresh collection/refill/update continuity remain open; this closes the
single-rank LFM FP16 native checkpoint gate only.

External receipts: `lfm-native-checkpoint-final.json`,
`lfm-native-fresh-process-replay.json`, `lfm-native-checkpoint-negative.json`
and their logs. `lfm-native-checkpoint-final/` contains the native checkpoint;
`lfm-expected-update2-state.pt` is the uninterrupted reference. The runner is
`native_verl_checkpoint_replay_run.py`. Initial runs and their exact runner
snapshot are retained separately. Final receipts hash the executed engine,
checkpoint manager, fixture and runner sources.

## Fresh task group after native checkpoint restore

The next experiment connects collection to an actual resumed optimizer update.
Load the verified step-one adapter into the native HF/Verifiers collection path
and collect four episodes with the same seed schedule and512-token turn budget.
Observed rewards are `[1,0,0,1]`: two successes, three truncated episodes.
The last successful episode executes the correct tool, then truncates its
following assistant response. Success and truncation are separate measurements.
Sampled lengths are741/512/512/1010; full response lengths including excluded
tool/template positions are872/512/512/1141. Native IDs, log probabilities,
final branches and direct Posttrain projections match exactly. Independent
episode, anchor-return and token-span equations reproduce the SAMPO credit,
including zero credit on excluded positions. No reward is relabeled.

A fresh native veRL process restores the actual step-one checkpoint, checks
exact adapter equality with the collection producer, and updates on this new
four-row group with microbatch1. Optimizer steps advance1→2; the restored
scale512/tracker1 advances to1024/tracker0 under growth interval2. All four
loss/score checks pass: maximum loss error1.09e-8 and score-gradient error
3.54e-11, with zero context and excluded-response score gradients. All24 LoRA
matrix gradients match manual native-arithmetic chain-rule calculations across
four microbatches;192 independent scalar-dot checks pass. The largest AdamW
parameter error is2.12e-9; the finite update is applied rather than skipped.
Peak Torch allocation is2.54GiB. Sampler correction ranges0.962711–1.040920.

Old-policy scores are recomputed by the native training engine before the
update; HF collection scores enter the separate sampler-correction weights.
Each optimizer policy ratio is1 at the first forward on this population,
so active clipping is zero for this update. Preserving previous Adam moments
does not change that initial current/old ratio. This supplies a concrete
fresh-data explanation for zero clipping without asserting that reuse,
minibatch updates or sampler correction must also have zero clipping.

This closes bounded single-rank LFM FP16 fresh-group update continuity. The
experiment still orchestrates collection and updating separately; it does not
qualify production worker admission/refill, rollout synchronization or vLLM.
The24 linear parameter checks do not prove every nonlinear model Jacobian.
External receipts are `lfm-step1-new-group.json`, its audit,
`lfm-step1-fresh-fixture.json`, and `lfm-fresh-group-update.json`; runners are
`build_native_fresh_fixture.py` and `native_verl_fresh_group_run.py`.

Fresh post-update collection repeats the four training-group seeds:

| Measurement | Restored step-one adapter | After fresh-group update |
| --- | ---: | ---: |
| Task successes | 2/4 | 3/4 |
| Truncated episodes | 3/4 | 2/4 |
| Sampled tokens | 2775 | 2908 |
| Executed tools / posted messages | 2 / 2 | 3 / 3 |

The previously failed seed6200 succeeds with645 sampled tokens. The other
three episodes retain the same generated token IDs, including the successful
but truncated final episode. All four post-update native token, mask,
projection and independent episode/turn/token-credit checks pass. The143-token
increase buys an additional completed tool task; it is not a throughput result.
These seeds also produced the training population, so this is recovery on a
matched training sample rather than held-out generalization. Earlier harmful
fixed-population updates remain part of the evidence; this result does not
establish a generally superior recipe or remove the broader correctness gates.
Post-update receipts are `lfm-fresh-group-postupdate.json` and its audit/logs.
