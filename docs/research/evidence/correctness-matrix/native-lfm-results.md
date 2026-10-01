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
