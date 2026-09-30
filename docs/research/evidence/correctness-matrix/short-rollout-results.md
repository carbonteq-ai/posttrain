# Three-step SAMPO runs: rollout failures, precision, and clipping

The small runs found a reproducible probability-precision problem and reproduced
clipping once real rollouts were reused. They do not establish that the complete
training stack is correct or explain the poor production runs by themselves.

## What actually ran

The RTX3070Ti8GB ran cached Qwen3.5-0.8B and LFM2.5-1.2B-Thinking serially.
Each run used three collection/update rounds with one prompt group of two,
microbatch one, accumulation two, LoRA rank4/alpha8 on q_proj/v_proj, LR0.0001,
gradient norm cap1, temperature0.8, unrestricted top-p/top-k, beta0.01,
and a base-model reference with the adapter disabled. Dropout was zero.

The public AutomationBench adapter source was pinned at
11f4d712806d292c6c6a752af046f4e16c4f037e. Two task builders supplied the exact
user requests, initial simulated Slack worlds, actual Slack message tool, and
authoritative assertion rubric: simple.slack_office_closure and
simple.slack_marketing_campaign. No external Slack service or judge was called.
The harness narrowed tools to one direct action and replaced the system's
50-turn budget with one. It used the cached tokenizer's tool chat template,
preserved sampled token IDs, and stopped after one assistant turn.

This is a direct-loss research harness. It calls framework SAMPO advantage
construction and the actual TRL objective, but bypasses full Trainer scheduling,
native Verifiers transport, vLLM correction, active-group refills, distributed
and fused execution, multi-turn tool-output masking, and checkpoint recovery.
Zero-spread groups were deliberately retained for inspection. Do not call this
a production-equivalent or benchmark leaderboard result.

## Rollout observations

| Arm | Task rewards by round | Truncated completions | Actual update observation |
| --- | --- | --- | --- |
| Qwen, post13, FP32 probability math, 128 tokens | [0,1], [1,0], [1,1] | 0/6 | Mixed groups learned; third round had only KL and optimizer history |
| Qwen, post13 native BF16, 128 tokens | [0,1], [1,0], [1,1] | 0/6 | Same coarse outcomes; different score drift and gradient magnitudes |
| LFM, post13, FP32 probability math, 128 tokens | [0,0], [0,0], [0,0] | 6/6 | Every response stopped during reasoning; no tool call or update |
| LFM, post13, FP32 probability math, 512 tokens | [1,1], [0,0], [1,1] | 1/6 | Every group had zero reward spread; no update |
| LFM, post13, FP32 probability math, 384 tokens | [1,0], [0,0], [1,1] | 3/6 | First round updated; later zero-spread groups still moved via KL/momentum |
| LFM, candidate + deterministic attention, 384 tokens | [0,0], [0,0], [1,0] | 5/6 | Third round made a finite, independently verified nonzero update |

These tiny samples demonstrate mechanisms, not quality estimates. Changing
attention kernels changes numerical sampling paths even at the same seed;
the deterministic LFM row is not a matched quality ablation against the other
rows. Likewise, the token-budget grid is a diagnostic, not a recommended
production budget.

One Qwen failure simply claimed it had posted the notice without making a call.
A successful response emitted a valid native XML tool call containing the
requested message. The initial harness accepted JSON only, falsely assigned
zero reward to those XML calls, and therefore produced no learning. The parser
was repaired in the harness. This was not evidence of a production parser bug;
the invalid initial run is retained separately.

LFM often spent hundreds of tokens explaining an obvious action before calling
the tool. At 128 tokens all six samples truncated in reasoning. At 512, one
sample still truncated inside tool syntax. Another complete campaign call used
"#SpringForward" rather than "Spring Forward" and failed the benchmark's
literal text assertion. This exposes both generation-budget and reward-contract
limitations. A response can also complete a useful tool call before hitting
the token cap, so truncation and task failure must be reported independently.

For a one-turn group with rewards [0,1], mean-centered episode advantages are
[-0.5,+0.5]; identical-anchor turn advantages are also [-0.5,+0.5]. With turn
weight1, SAMPO combines them to token credit [-1,+1]. This is the declared
hierarchical construction, not an accidental extra factor in accumulation.

## Independent mathematical baseline

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_rollout_run.py:independent_math` uses Python float64
arithmetic outside TRL. For each trajectory it holds old-policy scores fixed,
computes the geometric mean ratio, applies signed clipping at0.997/1.004,
adds the selected k3 KL, and derives each sampled-logp gradient analytically.
It includes the two-trajectory accumulation divisor. The resulting gradient
is also used as an independent vector for model backpropagation.

The deterministic Qwen frozen-group run's six loss comparisons differ by at
most 4.51e-8, and logp-gradient comparisons by at most9.32e-10. Repeating an
identical backward vector produces exactly the same parameter gradients with
deterministic math attention. The first update agrees exactly with the
independent parameter gradient. Later updates differ by0.10% and0.33%, failing
the predeclared0.01% threshold despite matching logp gradients. Keep this
failure open: combining policy and KL gradients through a BF16 backbone can
round differently; that is a hypothesis, not yet a completed localization.

With default attention, identical backward vectors differed by approximately
0.6–1.4%, so that configuration cannot support a tight parameter-gradient
comparison. Do not mistake that numerical repeatability failure for proof of
an incorrect SAMPO formula. LFM's deterministic nonzero third update matches
the independent parameter gradient exactly. Fully FP32 short Qwen fixtures
also repeat exactly and produce the correct opposing credit. A full FP32
AutomationBench rollout control exceeded available VRAM; it remains unqualified.

## Clipping was reached with actual generated tokens

One diagnostic reused Qwen's first group for all three optimizer updates,
retaining the original behavior scores and task rewards. This intentionally
differs from the production-style fresh single-use group schedule.

| Update | Failed trajectory's ratio before the next update | Failed trajectory clipped in this loss | Successful trajectory's ratio after update |
| --- | ---: | ---: | ---: |
| 1 | 0.980965 | 0% | 1.000225 |
| 2 | 0.947790 | 100% | 1.000896 |
| 3 | 0.914204 | 100% | 1.001464 |

The ratio column is measured after the named update relative to the frozen
old scores. Loss1 sees ratio1; loss2 and loss3 see the prior update's drift.
The low-side policy contribution is then correctly clipped. The adapter can
still move because KL, other trajectories, and Adam's stored momentum remain.
Clipping does not enforce a hard distance bound on the post-update model.

Fresh-group runs reported zero clipping during loss evaluation, while tokens
moved beyond the band afterward. The diagnostic supports the structural
schedule explanation for zero clipping; it does not prove which production
reuse/minibatch recipe will train best.

## Precision correction and remaining failures

The candidate TRL fork promotes BF16 logits before temperature/log-softmax,
and BF16 external scores before ratio/KL arithmetic. For identical real-model
logits, the scoring error against FP32 probability math fell:

Source correction:
[d1d298bc](https://github.com/carbonteq-ai/trl/commit/d1d298bc3c3528b57c183be4f956ddc921a92c07).
It is committed and pushed, but has no new wheel release or production pin.

| Model | Original BF16 max error, nats | Corrected max error, nats |
| --- | ---: | ---: |
| Qwen0.8B | 0.0172361 | 0.000000417 |
| LFM1.2B | 0.0442148 | 0.000000954 |

Token-ratio objectives also rounded the requested clipping bounds to1.0078125
and0.99609375. Two isolated regressions fail against the installed post13 wheel
and pass against the candidate. The tested sequence-ratio path already promotes
its reduction to FP32; do not attribute this particular clamp-rounding failure
to SAMPO. SAMPO still suffered BF16 score quantization before that reduction.

Twenty-nine precision, SAMPO, reference-adapter and padding regressions pass.
Full and chunked scoring are checked. FP32 probability math increases logits
memory, so long-context/fused/distributed runtime qualification is still needed.
The source correction is separate from the KL estimator flag, which was not
changed. Runtime pins and active runs remain unchanged by this candidate.

Qwen FP16 still produces nonfinite model gradients even with deterministic math
attention. The earliest monitored nonfinite activation gradient is at layer14's
linear-attention in_proj_qkv, then propagates backward to earlier layers. BF16
and short FP32 controls are finite. The optimizer is fenced from nonfinite
gradients. Full Trainer autocast/scaling and the precise failing model operation
remain to be tested before a generic Transformers or framework repair is chosen.

## Next acceptance gates

Follow-up 2026-09-30: a fresh deterministic Qwen BF16 run using TRL
`9f0825046ae3509a6be804d74a93fb89d5dc695e` (stable near-zero KL) passes
all three independent score and parameter-gradient checks; parameter relative
errors are zero. Evidence: fresh stable-KL run (`qwen08-fresh-kl-stable.json`, local archive).
The six simulated episodes yield [0,1], [1,0], [1,1], with no truncations and
no tool execution errors. Each early failed response claims completion without
issuing a tool call. This is semantic failure, not a parser-error count.
The final zero-spread group still moves adapters through KL and Adam momentum;
zero policy advantage does not imply zero parameter movement. This follow-up
resolves the earlier BF16 discrepancy for these fresh traces, while the matched
supplied-trace KL before/after controls establish the numerical fix's mechanism.
Six samples do not establish a learning trend or certify native Verifiers.

Localize Qwen FP16; repeat the resolved BF16 check through native collection,
and check accumulated unequal-length objectives, adapter/reference recovery,
and native multi-turn context/action masks. Broaden independent objectives to
the other public algorithms. Then repeat task experiments through the actual
Verifiers and full Trainer path, including active sampling and refill policy.
No current result certifies that broader scope.

## Reproduction and evidence

From `/home/hammad/projects/rl`, the prepared research Python is
`/home/hammad/projects/trl-gdpo-capo/.venv/bin/python`. Original TRL comes from
`/tmp/trl-post13-wheel-install`; corrected source from
`/home/hammad/projects/trl-sampo-local-credit` at the commit above. PEFT is in
`/tmp/trl-math-peft`, and isolated harness dependencies in
`/tmp/posttrain-mathdeps`. The benchmark source path is
`/home/hammad/.cache/uv/git-v1/checkouts/3a2db303e0343477/11f4d71/environments/automationbench_v1/src`.
These temporary dependencies must be reconstructed for another machine;
their versions are research-environment evidence rather than a production lock.

Use the corrected source, PEFT, isolated dependencies and benchmark source
in `PYTHONPATH`, then run:

    /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_rollout_run.py --model qwen08 --reuse-rollouts --score-fp32 --deterministic --output .posttrain/state/correctness/qwen08-frozen-group-candidate-deterministic.json
    /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_rollout_run.py --model lfm12 --score-fp32 --deterministic --max-tokens 384 --output .posttrain/state/correctness/lfm12-short-candidate-deterministic.json

JSON files in the external experiment archive retain the complete public simulated trajectories,
per-round rewards, truncation, advantages, gradients, updates and drift. Failed
initial parser and OOM runs remain machine-local and are described above. The
runner writes each completed round atomically. Run serially and fence nonfinite
gradients; a shorter fixture is the recovery for the full FP32 OOM, not evidence
that the larger failed control passed.

## Native collection and the reward signal (2026-10-01)

Qwen0.8B now completes fresh AutomationBench episodes through the pinned native
Verifiers runtime, its chat harness, MCP discovery and tool execution, task setup,
finalization and assertion scoring. This uses the `null` harness with a local
subprocess runtime and the task's `limited_zapier` tools. The tools mutate the
benchmark's simulated world; no real Slack messages are sent. The HF inference
provider is an external research adapter, so this is not production vLLM
qualification or a full Trainer learning experiment.

The task is `simple.slack_office_closure`, with a four-turn budget and up to256
tokens per model request. It asks for a facilities-maintenance announcement in
`#general`, work-from-home instructions and Monday reopening. Verifiers is pinned
to `e6a3d9bbfe6959b97878f451fc721793a232cd5f`, AutomationBench to
`11f4d712806d292c6c6a752af046f4e16c4f037e`, and the renderer source to
`1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`. The model snapshot remains
`2fc06364715b967f1860aea9cf38778875588b17`; inference uses BF16 and the isolated
Transformers source `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a`.

| Collection | Episode | Assistant turns | Tool calls | Posts | Reward | Tool failures / truncations |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Scripted transport control | 1 /2 | 3 each | 2 each | 1 each | 1.0 each | 0 /0 |
| Real Qwen, evaluation client | 1 | 4 | 5 | 3 | 1.0 | 0 /0 |
| Real Qwen, evaluation client | 2 | 3 | 2 | 1 | 1.0 | 0 /0 |
| Real Qwen, native training client | 1 | 2 | 1 | 1 | 1.0 | 0 /0 |
| Real Qwen, native training client | 2 | 3 | 2 | 1 | 1.0 | 0 /0 |

All posted messages contain the requested instructions. The evaluation client's
first episode posts the announcement three times and searches for channels twice
in one turn. Its reward still matches the single-post episode. The assertion
checks only for a message in `general` containing `February 27`; it does not
penalize duplicate side effects or redundant searches. That is a measured reward
limitation on this task. These two episodes do not establish its prevalence in
the current training run.

The native training client uses the renderer's token-generation contract and
retains exact sampled spans and sampling log probabilities. The external HF
provider implements `/inference/v1/generate`; Verifiers performs decoding and
tool parsing. The two episodes contain133 and165 sampled tokens, across two and
three assistant turns. Posttrain's actual `_project` method accepts both traces
and preserves those token and turn counts. This checks the direct projection
seam; it bypasses worker scheduling, admission/refill and the optimizer.
On the full-precision repeat, all five calls retain exactly the provider's
sampled IDs and log probabilities. Each final graph branch exactly matches
the final provider prompt plus completion. All system/user/tool input nodes
have zero sampled mask entries. The analysis asserts these invariants and
the independent sparse-return credit equations.

Equal terminal rewards give episode advantages `[0,0]`. With Posttrain's default
discount0.95, sparse terminal rewards give initial returns0.95 and0.9025 for
the two-turn and three-turn episodes. Independent centering therefore predicts
first-turn credits `+0.02375` and `-0.02375`; the framework returns those values.
Later anchors are singletons and carry zero relative credit. With discount1,
all turn and token credits are zero. Thus equal episode rewards can still yield
a length-dependent SAMPO turn signal before admission. The inspected TRL active
sampling path keeps groups using terminal reward standard deviation greater than
zero; it would reject this equal-reward group. We have not executed that native
admission path on these episodes, so its effect here remains a source-backed
prediction.

Two runner mistakes remain visible in the external evidence. The first local
evaluation adapter failed on the second turn because it passed OpenAI's JSON
argument strings to a Qwen template that expects mappings. Conversion of the
renderer copy fixes the transport without changing the original wire request.
The first training-client receipt used Verifiers' default four-decimal JSON
serialization, producing log-probability differences up to4.994e-5 on reload.
Posttrain already opts out through `to_record(float_decimals=None)`. The runner
was corrected to retain full precision and rerun; neither issue proves a
production framework defect.

Runners and raw receipts live under `${POSTTRAIN_CORRECTNESS_ROOT}/working/` and
`${POSTTRAIN_CORRECTNESS_ROOT}/results/native-collection/`. The collection runner
is `native_automationbench_collection.py`; the analysis runner is
`analyze_native_collection.py`. Retain both failed attempts and successful arms.
Further acceptance needs multiple task groups/seeds, native admission/refill and actual matched optimizer
updates. Reward changes belong to the external environment/rubric contract and
need controlled task-quality evidence before adoption.

## Admission replay and temperature audit (2026-10-01)

The observed equal-reward group was replayed through the actual
`GRPOTrainer._prepare_active_sampling_inputs` method and Posttrain's
`_prepare_adaptive_active_sampling_inputs`. Both reject all two rows and report
an exhausted one-round population, even though the projected SAMPO credit is
nonzero on88 and16 first-turn tokens. This executes the admission method bodies
with observed rewards injected at the scoring seam; it does not run fresh
worker selection or refill. The rejection is consistent with the selected
terminal-reward variance rule. Retaining discounted turn signal would be a
recipe/contract decision requiring a controlled comparison.

The same five native requests also provide298 sampled tokens for a temperature
audit. Actual TRL scoring divides the teacher-forced logits by0.8, matching the
native sampler. A direct full-log-softmax baseline agrees within1.91e-6. Fifteen
selected positions checked with an independent Python `math.fsum` softmax have
maximum error1.12e-6. Changing only the scoring temperature to1 worsens the
absolute sampling-score discrepancy on every request. No missing temperature
normalization was found in this path.

The remaining sampling-score discrepancy exists before any optimizer update.
An independent cached forward replay reproduces every recorded BF16 sampling
log probability exactly. Full teacher forcing on the same model and token ids
uses a different execution path and yields the following differences:

| Request | Sampled tokens | Mean absolute log-probability difference | Maximum difference | Geometric mean teacher/sampler ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 88 | 0.008618 | 0.146559 | 0.997934 |
| 2 | 45 | 0.015328 | 0.161058 | 1.004053 |
| 3 | 16 | 0.000468 | 0.007206 | 0.999567 |
| 4 | 129 | 0.002715 | 0.087238 | 1.000556 |
| 5 | 20 | 0.016084 | 0.073517 | 0.997043 |

Across all tokens, the weighted mean absolute difference is0.00713942.
Individual teacher/sampler token ratios range from0.863675 to1.174753. A
diagnostic FP32 model, with TF32 disabled, reduces the cache-versus-teacher-force
weighted mean difference to1.947e-6 and the maximum to4.983e-5 on these same
tokens. Its independent temperature checks also pass. This strongly supports
numerical execution sensitivity; it does not identify a particular layer as
the cause or establish equivalence to the production vLLM kernels. FP32 is a
diagnostic reference, not a proposed replacement for BF16/FP16 training.

These ratios compare the sampler with the training forward. They are separate
from the PPO/SAMPO ratio between current training scores and frozen old
training scores. A fixed model can have sampler/trainer disagreement while its
current/old ratio remains1. Therefore the table does not demonstrate clipping
caused by a learning update. Posttrain's default SAMPO sampler correction is
per-token upper truncation at2, distinct from the sequence-level policy clip.
None of the observed token ratios reaches that correction cap.

The executed external runners are `native_group_admission_replay.py` and
`native_temperature_score_audit.py`. Receipts are `admission-replay.json`,
`temperature-score-audit.json`, `temperature-cached-replay.json` and
`temperature-fp32-cache-control.json` under the same external
`results/native-collection/` directory. Source snapshots and hashes accompany
the receipts. No source fix or production recipe change follows from this
slice; native collection-to-update, LFM fresh collection and matched backend
learning remain pending.
