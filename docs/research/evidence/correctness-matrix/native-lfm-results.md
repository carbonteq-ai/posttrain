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
