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
zero. The subsequent admission replay below confirms rejection at the scoring
seam; full fresh worker selection and refill remain unqualified.

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

### Cross-backend admission and truncation revalidation

A later source-seam replay adds veRL's actual `_keeps_group` predicate and
Posttrain's actual `_native_group_reward_std`, using the exact TRL `nanstd`
helper. Functions are AST-extracted without rewriting their bodies; CPU
transport doubles supply the observed rewards and identity gather. This is
not a full worker, fresh collection or refill test, and it does not replace
the earlier executed TRL preparation-method replay.

Seven cases agree across these seams. The observed Qwen rewards `[1, 1]`
reject despite 88 and 16 nonzero sampled credits (absolute credit sums 2.09
and 0.38). Equal failures also reject; mixed rewards retain with sample
standard deviation approximately 0.707107. Fewer than two unmasked finite
rewards reject. These last synthetic cases are controlled boundary checks.

The observed LFM fixture has rewards `[1, 0]`, with only the failed trajectory
truncated. Keeping its rewards unmasked retains the group. Selecting
`mask_truncated_completions=True` instead leaves one finite reward and rejects
the group at these seams. The current SAMPO setting defaults to `False`;
the masked result is a conditional configuration comparison, not a claim
about an active run's settings or full veRL transport of exclusion flags.

The frozen contract in `docs/post-training/05-apis.md` explicitly requires
differing episode rewards for SAMPO active sampling. Rejection therefore
agrees with the contract, even though hierarchical credit can remain nonzero.
Changing admission to retain such credit would change that policy and needs
a recorded baseline amendment plus a fresh controlled quality comparison.
In particular, discounted sparse returns can create a length-dependent
preference between equally successful episodes; nonzero credit alone does
not prove that retaining the group improves task learning. Current objective
probes bypass this admission rule, so passing optimizer checks cannot imply
that these populations would update in production.

External evidence: `sampo-active-admission-source-replay.json` and exact
`replay_sampo_active_admission.py`, retained outside Git with source and fixture
hashes. No product code, admission default or reward semantics changed.

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

## Harder task screening and longer native updates

1 October 2026. Collect twelve fresh baseline episodes using the native
Verifiers training client, renderers and pinned AutomationBench environment.
The tasks are `simple.email_sf_contact_phone_update` and
`simple.weekly_report_sheets_email`. They require retrieving/updating a contact
or writing spreadsheet cells and sending an email. Keep the benchmark prompts,
tools, world assertions and rewards; only select an explicit six-turn budget.
These are screening tasks, not held-out evidence of trained-model improvement.

| Model / collection | Contact rewards | Report rewards | Per-turn token limit |
| --- | --- | --- | ---: |
| Qwen initial screen | 0, 0 | 0.5, 0.5 | 512 |
| Qwen additional seed pair | — | 0, 0 | 512 |
| LFM initial screen | 0, 0 | 0, 0 | 512 |
| LFM follow-up | — | 1, 0 | 1024 |

Each collection has two independent repetitions per selected task. Qwen's
initial base seed is19400; its additional pair uses29400. LFM's follow-up
also uses19400 but selects only the report task, changing its episode-index
seed offset relative to the initial report screen. Therefore the LFM change
is an exploratory follow-up with a larger budget and different seeds, not a
paired causal budget ablation. Every initial LFM episode hits the512-token
limit. Its follow-up success uses963 and354 generated tokens across two turns;
the failure hits1024 on its first turn.

All twelve episode audits check exact sampled IDs and log probabilities,
finite log probabilities, zero masks on nonsampled nodes, final-branch token
reconstruction and Posttrain projection. Compute SAMPO credit separately for
each task's two-rollout population. Independent sparse-return/anchor centering
and token-span assignment agree exactly in these fixtures; excluded positions
have zero credit. The initial audit passed invented group IDs instead of
the authoritative task IDs and correctly failed the framework identity guard.
Preserve that external runner/log, correct its requested IDs, and rerun. This
was an instrumentation error, not a product repair.

An additional oracle reads plain JSON final states without calling the
benchmark scorer: check the contact's exact phone, report-sheet cell values
containing `Feb 23`, and sent-email recipient/subject/body criteria. Its flags
and arithmetic mean reproduce all twelve native partial-credit rewards. This
oracle covers these task instances, not the general benchmark assertion engine.

Qwen sometimes asks for clarification instead of acting. Its initial report
episodes contain tool payload errors (`Invalid cells JSON format`) even though
the native trace is `ok` and has no harness errors. One later row places the
summary in a dictionary key rather than a cell value, so the row assertion
still fails. These distinctions matter: successful transport, successful tool
execution and task completion are different evidence. Equal partial rewards
also produce239/219 nonzero credited tokens yet fail current spread admission.
The screen does not show a transport, reward arithmetic or credit defect.

### Native updates on the observed LFM success/failure group

Use the complete observed follow-up pair `[1, 0]` without reward edits. It has
1573 prompt tokens, completion lengths1675/1024, and1317/1024 sampled positions.
Both initial prompts match exactly. The default unmasked-truncation selection
would admit this population by reward-spread predicate; excluding the truncated
failure would reject it. This remains a collection-to-native-engine fixture,
not execution of live worker admission/refill or reference-model transport.

Run three deterministic native veRL SAMPO updates per precision, retaining
rank4/alpha8 FP32 LoRA masters, learning rate1e-4, sequence-valued/token-local
clipping0.003/0.004, beta0.02, fixed initial reference scores and frozen capped
sampler correction. FP16 scale1024 applies every update. Independent checks
cover12 combined loss/score/mask seams,144 LoRA matrices,576 scalar dot products
and AdamW updates. Maximum loss error is5.83e-8, score derivative error1.13e-10,
and AdamW error2.13e-9. Prompt/excluded derivatives are exactly zero. Peak Torch
allocation is2,439,343,616 bytes (approximately2.44GB); all six updates apply.

By update3, BF16 clips all policy credit on the failed rollout; FP16 clips both
rollouts. The nonzero reference penalty still supplies score derivatives.
Clipping therefore does activate on this longer observed group. This does not
establish fresh-task improvement or justify changing the recipe. Fresh adapter
rollouts, held-out comparisons, real worker/refill, production selections and
broader algorithm/model qualification remain open.

External evidence: `*-headroom-screen-*`, `*-headroom-followup-*`, detailed
token/credit audits, `headroom-world-reward-audit.json`,
`lfm-headroom-native-fixture.json`, the two native precision receipts/adapter
exports and `headroom-native-update-summary.json`. Exact executed sources and
the identity-failure attempt remain outside Git. No product code, default or
dependency pin changed.

## Fresh trained-adapter behavior and native reference contexts

1 October 2026. Compare the longer-task LFM adapters before training and after
three native updates. Use fresh seeds39400/40400 on the same screened weekly
report task, six turns and1024 tokens per turn. Run both BF16 and FP16. The
provider uses SDPA, matching the native training engine, global deterministic
algorithms and the same autocast precision. Within each precision pair, seeds,
initial prompt tokens, tokenizer, tools and budgets match. These are eight
fresh episodes on a previously screened task, not a held-out efficacy study.

All four adapter loads have exact parameter keys and zero copy error. Each
reproduces both recorded native fixture score rows exactly (maximum error0).
Fresh sampled IDs/log probabilities and zero masks on nonsampled nodes pass
their transport checks. Independent final JSON sheet/email checks reproduce
the native rewards. This verifies that the measured native updates reach
inference under the tested loading path.

| Precision | Seed | Starting reward → trained reward | Starting/trained truncation | First-turn common prefix |
| --- | ---: | --- | --- | ---: |
| BF16 | 39400 | 0 → 0 | both | 44 tokens |
| BF16 | 40400 | 0 → 0 | both | 80 tokens |
| FP16 | 39400 | 0 → 0 | both | 675 tokens |
| FP16 | 40400 | 0 → 0 | both | 228 tokens |

Every episode generates1024 tokens on its first turn, opens `<think>` and
never closes `</think>`. None produces a tool response or satisfies either
world-state assertion. The trained model changes the generated sequences,
but these two seeds show no reward or truncation improvement. This is a
negative descriptive result with a reward floor, not a reward ceiling or
proof that updates are absent. It does not isolate the cause of long thinking,
establish an optimal budget, or prove that simply allowing more tokens solves
the tasks. Test matched budget/reasoning controls before changing a recipe.

### Actual native adapter-disable reference checks

Load each trained adapter into the native single-GPU veRL FSDP2 engine without
performing optimizer updates. Its initial active scores exactly reproduce the
trained artifact. Inside `engine.disable_adapter()`, scores exactly reproduce
the recorded step0 base-model scores. The step0 adapter has zero LoRA-B state,
so it is the unchanged base reference in this experiment. After ordinary exit,
active scores restore exactly. An intentional exception inside the disable
context also restores the actor exactly. Maximum recorded response-score
differences between trained actor and base are0.4752505 BF16 and0.5330752 FP16;
these maxima include all recorded response positions, not only sampled credit.
Peak Torch allocation is approximately1.315GB per reference-only arm.

This closes the tested native adapter context/restoration gate. Production
source uses `no_lora_adapter=True` for the in-actor reference and then writes
projected reference scores through TransferQueue. Those live worker/queue
operations are not executed here. A starting adapter already containing
trained weights can also differ from the disabled foundation reference; the
current experiment does not qualify every checkpoint/reference selection.

External evidence: four `lfm-headroom-fresh-*-step*.json` receipts,
`headroom-fresh-adapter-comparison.json`,
`headroom-reasoning-budget-audit.json`, two native reference-context receipts,
exact executed sources and logs. Preserve the initial comparison source and
receipt whose scope mistakenly called reward-floor outcomes "no headroom";
correct the wording without changing its measurements. Reference probes add
zero optimizer updates. All tools/raw evidence remain outside Git. No product
source, budget default, KL coefficient or dependency pin changed.

## Matched 1024 versus 2048 token budgets

1 October 2026. Add eight native fresh episodes at2048 tokens per turn using
the same LFM starting/trained adapters, seeds39400/40400, initial prompt tokens,
six-turn limit, sampling settings, precision, SDPA and deterministic flags as
the preceding1024-token experiment. The derived total output allowance also
doubles from6144 to12288; this comparison does not isolate those two limits.
All four loads again reproduce both recorded native score rows exactly. Peak
Torch allocation is3,460,183,040 bytes (approximately3.46GB).

For every one of the eight budget pairs, the first1024 generated IDs and their
log probabilities match exactly. The longer run continues the shorter run's
sampled prefix, rather than silently changing prompt serialization or initial
sampling behavior. Exact fresh token/logprob/branch/excluded-mask checks and
independent sheet/email world-state reward checks pass on all compared episodes.

| Precision / adapter | Rewards at1024 | Rewards at2048 | Truncated at2048 | Sampled tokens at2048 |
| --- | --- | --- | --- | --- |
| BF16 / starting | 0, 0 | 0, 0 | both | 2048, 2048 |
| BF16 / trained | 0, 0 | 0.5, 0.5 | neither | 1801, 3049 |
| FP16 / starting | 0, 0 | 1, 0 | second | 2901, 2048 |
| FP16 / trained | 0, 0 | 0, 0.5 | first | 2048, 2072 |

The trained BF16 adapter now sends the email in both episodes, but fails the
spreadsheet assertion. FP16 loses the starting adapter's fully rewarded first
episode and gains an email-only second episode. These are native benchmark
assertion outcomes, not general task-quality certification. The longer budget
reveals actions and differences that the1024-token truncation hid, but does not
establish a uniformly better trained policy or make2048 an optimal default.
Two seeds on a screened task cannot justify choosing a precision or declaring
a recipe effective. A correct local gradient/update check likewise does not
guarantee improved fresh reward after three updates of one cached group.

Continue quality comparisons at budgets that permit tool actions, measure
truncation separately, and expand task/seed coverage before adopting a budget
or recipe. Preserve the adverse FP16 pair. No product setting or source change
follows from this result. External evidence: four
`lfm-headroom-budget2048-*-step*.json` receipts, exact provider/controller sources
and `headroom-budget-comparison.json` with per-pair prefix and state checks.

### Native reference routing and score-coordinate projection

Execute the exact AST-extracted veRL `_compute_ref_log_prob` and
`response_from_nested` function bodies with CPU transport doubles. Six cases
cover three unequal prompt/response-length layouts under both in-actor and
separate-reference routing. Unique numerical coordinate labels independently
verify next-token alignment: response token position `j` receives model score
position `j-1`. Excluded response positions retain their coordinates rather
than shortening the response; masking belongs to the subsequent loss.

Both routes read the expected queue fields and store only aligned
`ref_log_prob`, with the selected temperature0.8 and loss/entropy disabled.
In-actor routing sets `no_lora_adapter=True`; separate-reference routing uses
its dedicated worker. One empty-response projection boundary passes, without
claiming that empty trajectories are admitted for training. Its first external
fixture failed because an empty list inferred float dtype while other masks
were integer; specify the mask dtype and rerun, preserving the failed source.

These source-seam checks extend the previous native GPU disable/restore checks.
They do not execute live Ray dispatch, TransferQueue persistence, concurrent
worker ownership or full reference-model transport. No projection or routing
defect is found in these tested valid layouts. Exact sources and
`native-reference-projection-audit.json` remain outside Git. This slice adds
eight fresh episodes and six CPU routing/projection cases, with zero optimizer
updates; the broader correctness campaign remains incomplete.

## Worker flag lifecycle and production SAMPO selection repair

1 October 2026. Execute the exact undecorated `TrainingWorker.infer_batch`
and `train_batch` bodies with real TensorDict/native metadata helpers and an
instrumented engine. Eight cases vary reference disabling, inference loss
calculation and fresh versus explicitly reused input objects. Inference consumes
`no_lora_adapter`; subsequent training uses its bound loss with the adapter
enabled, regardless of inference-only flags. An intentional inference exception
restores its contexts. Fresh training inputs get the configured training
microbatch size; explicit reuse retains the engineering fields already supplied
by inference. These are worker-method/metadata checks, not native model
derivatives, optimizer updates or proof of production input-object reuse.

The source review also confirms a configuration gap: the production normalizer
still selected `gspo`, while the qualified native experiments selected
`sampo_token_credit`. GSPO already avoids the historical sequence-clip
gradient-cancellation problem, but retains an extra sequence-log-ratio cap.
The dedicated loss preserves the required uncapped sequence-ratio value and
token-local credit without redefining GSPO. Earlier probe results alone had
not repaired this production selection.

Repair Posttrain's SAMPO launcher and worker to require that dedicated loss.
Register the published candidate source
`d8e472db822f2916ed81a408b8d28192be95e678` for the new capability without
adding it to historical revision capability sets. Reject a declared clean
legacy post8 source at plan construction and again at override construction.
The explicit existing dirty-candidate opt-in remains available to its owner.
This satisfies the frozen contract's existing requirement to reject unsupported
SAMPO implementations; it does not change the algorithm's product meaning.

The new legacy-runtime regression fails before the repair because no error is
raised. Afterward,120 backend tests pass with8 optional-dependency skips. Two
actual candidate TRL/veRL tests agree on Posttrain-normalized advantages, loss
and sampled-score gradients using the generated corrected Hydra settings.
The first parity collection attempt exposed older tokenizers shadowing the
isolated runtime; moving supplemental dependencies to the end of the path
and adding the environment workspace source resolves setup without installing
packages or weakening assertions. Targeted Ruff/Pyright and all nine import
contracts pass. Earlier native model/optimizer evidence remains the GPU gate;
this slice adds zero optimizer updates.

Runtime pins and images remain post8. Those defaults cannot launch corrected
SAMPO until a compatible runtime is separately published/adopted and qualified.
GRPO/GDPO/CAPO selections are unchanged. Full live worker/TransferQueue,
distributed/packed paths, fresh iterative learning and broader family/algorithm
qualification remain open. External worker-body source/receipt
`native-worker-reference-training-audit.json` stays outside Git; repository
changes contain product selection/gates, regression tests and findings only.

## Second fresh SAMPO group: correct local updates, worse sampled behavior

The FP16 LFM policy regressed on the two controlled weekly-report seeds after
training on its own next rollout group. This result preserves the adverse
behavior rather than treating passing derivative checks as learning success.

Replayed the original three updates with the same deterministic native veRL
FSDP2 engine, LoRA rank4/alpha8, FP32 adapter masters, FP16 scale1024,
LR0.0001, AdamW weight decay0.01, gradient norm cap1, beta0.02 and SAMPO
ratio bounds0.997/1.004. Every step0–3 adapter parameter and recorded native
score reproduced exactly. Then loaded the complete fresh step3 group with
unchanged rewards[0,0.5], retained the live Adam moments/counter3, and applied
two more updates. The old-policy scores were refreshed for this new group;
the reference remained the original base policy through native adapter disabling.
No optimizer, scaler or reference reset occurred at the group transition.

The group has prompt1573, response lengths2048/2409 and sampled-token
counts2048/2072. Exact trace projection and independent hierarchical credits
agree with error0, including zero credit on tool/user response coordinates.
Default unmasked truncation admission retains it; enabling truncation masking
rejects it because only one eligible episode remains. The prior BF16 fresh
group[0.5,0.5] is rejected by the current reward-spread rule and was not forced
through training in this slice.

| Evidence | Result |
| --- | --- |
| Native updates | Five applied, none skipped; counters1→5 |
| Independent loss / score-gradient checks | Ten pass; fresh-group maximum errors1.72e-8 /2.10e-11 |
| LoRA linear derivatives / scalar dot references | 120 matrices /480 dots |
| Independent Adam maximum error | 2.15e-9 |
| Peak CUDA allocation | 3,005,776,384 bytes |
| Step3 fresh rewards, same seeds39400/40400 | [0,0.5]; truncation[true,false] |
| Step5 fresh rewards, same seeds/budget | [0,0]; truncation[true,true] |

Clipping and KL refer to different policies. At old-group update3 both
credited populations were clipped and their policy score-gradient norms were
zero. At new-group update4 both ratios reset to exactly1 and clipping was0,
while sampled-k3 proxies against the unchanged base were already8.39e-5 and
7.55e-5. Update5 ratios were0.9993274/1.0004862 and remained unclipped;
the proxies rose to1.45e-4/1.38e-4. The beta-weighted KL score-gradient norms
were roughly0.08–0.12% of policy score-gradient norms on this new group. These
are sampled surrogates and score derivatives, not full-vocabulary KL or model
gradient ratios; they do not establish an optimal beta.

Step5 inference loads the exact adapter keys/values and reproduces its fresh
fixture native scores with error0. Both fresh episodes consume2048 sampled
tokens in one turn and truncate; the earlier partial-success seed no longer
sends the required email. Four independent plain-JSON final-state checks agree
with the benchmark rewards at steps3/5. The resulting[0,0] population is
rejected and has zero hierarchical credit, so it cannot supply another ordinary
SAMPO update without collecting an eligible replacement group.

This is a second genuinely sampled group with uninterrupted native optimizer
state, not a full native worker/queue/refill or vLLM synchronization test. It
reuses two training seeds on one task and establishes no held-out efficacy.
The adverse behavior does not isolate whether credit structure, Adam history,
update size, reference strength, numerical precision or their interaction is
responsible. Next compare controlled one-update/reuse and optimizer-history
arms from the same step3 state before recommending a recipe change. Published
runtime adoption, Qwen/Gemma companions and broader algorithm gates remain open.

External receipts: `lfm-fresh-iteration-float16.json`,
`fresh-iteration-summary.json`, fresh group and quality group audits, and
`lfm-fresh-iteration-quality-float16-step5.json`. The native receipt retains an
inherited generic scope sentence; its transition record and companion summary
define the expanded two-group scope. All runners, checkpoints and raw receipts
remain under the external experiment directory and are excluded from Git.

## Adam history control: matched gradients, different behavior

Clearing AdamW history at the second-group boundary changes this small result
substantially. Retaining history produces two failed/truncated episodes after
either one or two fresh-group updates; clearing it produces two partial-success
episodes without truncation after either update count. This isolates a concrete
optimizer-state sensitivity, not an incorrect Adam implementation or a general
recommendation to reset state during training.

The new arm replays the original three updates and reproduces every step0–3
adapter parameter/native score exactly. It clears only optimizer moments and
the bias-correction counter immediately before update4. Parameters, model,
scaler1024, LR0.0001, weight decay0.01, loss, fresh training group, sampler
correction, reference, precision and sampling settings are unchanged. The
first fresh loss checks and all captured preclip LoRA gradients are bitwise
identical between arms. The reset counter sequence is1/2/3/1/2; retained is
1/2/3/4/5. Five new native updates include three replay controls and two
fresh-group updates, with no skips. Ten loss checks,120 linear matrices and
480 scalar-dot controls pass, at the same3,005,776,384-byte peak allocation.

Six additional fresh episodes complete the comparison, using the same native
Verifiers training client, seeds39400/40400, initial prompts,2048 per-turn
tokens and six-turn ceiling. Every adapter handoff reproduces native fixture
scores exactly. Exact sampled-token/logprob/excluded-mask checks and ten
independent plain-JSON final-state reward checks across the five arms pass.

| Policy / fresh-group updates | Rewards | Truncation |
| --- | --- | --- |
| Step3 starting policy /0 | [0,0.5] | [true,false] |
| Retained Adam /1 | [0,0] | [true,true] |
| Retained Adam /2 | [0,0] | [true,true] |
| Reset Adam /1 | [0.5,0.5] | [false,false] |
| Reset Adam /2 | [0.5,0.5] | [false,false] |

Independent complete-gradient reconstruction uses the actual saved gradients,
`m = 0.9*m_prev + 0.1*g`, `v = 0.999*v_prev + 0.001*g*g`, bias correction,
epsilon1e-8 and decoupled weight decay. It reproduces all ten parameter steps
across both arms within2.15e-9 maximum absolute error. All gradient norms are
below1, so norm clipping is inactive in this reconstruction. At the first new
update, the retained first-moment history contribution has norm0.0268349,
versus0.00702921 for the current gradient contribution:3.8176 times larger.
Their sum has cosine0.2418 with the current contribution. This is a
reconstruction of the unpreconditioned moment, not the final update direction.

The actual first-update displacement norms are0.0222530 retained and0.0395709
reset, with cosine0.4391. Their difference is161.26% of the retained displacement
norm, and58,434 coordinates move with opposite signs. After two updates,
cosine is0.6019 and the displacement difference is129.60%. Thus the same current
gradient can produce markedly different policy changes through correct optimizer
arithmetic. The earlier fully ratio-clipped update did not erase prior moments.

Reducing this fresh group's reuse from two updates to one does not recover
these retained-history episodes. Clearing history does, but it simultaneously
changes first/second moments and the counter; this experiment cannot attribute
the outcome to first momentum alone. Nor does it show that resetting history
is useful across tasks, seeds, precisions or the production recipe. Next isolate
moment components/decay and update size from the same state, expand task/seed
coverage, and compare a more diverse rollout population before adopting defaults.
Both reset episodes still fail the Sheets assertion and pass only the email
assertion. Their equal rewards also mean this next population is rejected by
current SAMPO spread admission; successful refill remains necessary.

External sources/receipts: `native_verl_fresh_reset_adam_run.py`,
`lfm-fresh-reset-adam-float16.json`, three `lfm-fresh-adam-quality-*` collections
and `fresh-adam-matrix-summary.json`. All tools, checkpoints and raw data stay
outside Git. No product defaults, immutable runtime pins or training semantics
change in this slice; broader correctness and native-worker qualification remain
incomplete.

## Separate moment controls: recovery is not a momentum-only story

Clearing either Adam moment changes these two-seed outcomes, but the numerical
effects are very different. First-moment clearing preserves the starting
partial-success pair. Second-moment clearing produces partial successes on both
seeds while amplifying the parameter update dramatically. This revises the
simple hypothesis that only stale momentum explains the full-reset recovery.

Two new native FP16 arms preserve the exact step3 policy, sampler correction,
fresh[0,0.5] group, base reference, scale1024, LR0.0001 and Adam counter3.
One zeros only exp_avg; the other zeros only exp_avg_sq. The other moment is
verified bitwise unchanged. Both reproduce step0–3 parameters/native scores and
the first fresh loss checks/gradients exactly. Eight applied native updates
include six replay controls and two fresh updates;16 loss checks,192 linear
matrices and768 scalar-dot checks pass, with no skips. Peak remains
3,005,776,384 bytes. A preflight initially omitted the renderer import paths;
restoring the existing PYTHONPATH resolved it without changing dependencies.

| One fresh update from the same step3 policy | Rewards | Truncation | Parameter displacement L2 |
| --- | --- | --- | --- |
| Retain all state | [0,0] | [true,true] | 0.0222530 |
| Clear first moment only, retain counter | [0,0.5] | [true,false] | 0.0109383 |
| Clear second moment only, retain counter | [0.5,0.5] | [false,false] | 6.4064561 |
| Clear both moments and counter | [0.5,0.5] | [false,false] | 0.0395709 |

Four additional fresh episodes use the same two training seeds,2048-token
per-turn allowance and six-turn ceiling. Exact adapter score reproduction,
token/logprob/excluded-mask transport and ten independent episode final-state
checks across the starting/control arms pass. First-moment clearing preserves
the email-only partial success. With second-moment clearing, seed39400 passes
only email, while seed40400 passes only Sheets. Equal reward0.5 therefore does
not mean equal behavior or complete task success.

Independent saved-gradient Adam reconstruction checks all eight new parameter
steps. Maximum error is2.13e-9 in the first-moment arm and2.20e-7 in the
second-moment arm, whose weights move much farther. All raw gradient norms
remain below1. The first-moment displacement has cosine0.5267 with the retained
update; the second-moment displacement has cosine0.05688. Their magnitudes
also differ, so this does not isolate direction from update size.

The largest second-moment coordinate is layer8 q_proj LoRA-B, flat index6257.
Its recorded unscaled gradients at steps1–4 are−0.0011119843,
−0.0011405945,+0.0000036461,0. The retained first moment after step3 is
−0.0001923596. On update4 it decays to−0.0001731237 even though the new gradient
is zero. After clearing the second moment, its new value is0 and the entire
denominator is epsilon1e-8. With counter4 bias correction, independent scalar
Adam predicts parameter5.03440501 versus native5.03440523, error2.20e-7;
the previous value was0.00027725. Thus a zero current gradient does not imply a
zero optimizer update. This does not establish why that gradient was zero or
show that production discards its second moment.

All resulting native scores remain finite. In the second-moment arm, maximum
chosen-token logprob differences from recorded sampler scores reach13.90/13.91
on the two trajectories; mean differences are−0.16563/−0.15659. These are
post-update actor-versus-sampler scores, with known scoring-path mismatch,
not PPO ratios against refreshed old-policy scores or true KL. The first fresh
loss had ratio1 before the update in both arms: clipping that loss is not a
bound on an optimizer-state-induced parameter displacement.

The large-denominator-reset control improves this narrow partial reward, so
calling its behavior a task-quality collapse would be wrong. It remains a
diagnostic intervention, not evidence for a safe/default recipe. Both moment
components influence the observed result; full-reset recovery cannot be
attributed solely to first momentum. Next isolate counter bias correction and
displacement size, then expand tasks, seeds, precisions and real worker/refill
coverage. No production settings, semantics or pins change here.

External receipts: `fresh-moment-controls-summary.json`,
`fresh-second-moment-max-coordinate.json`, two native moment controls and two
matched fresh collections. All runners, checkpoints and raw results remain
outside Git; repository changes contain findings and the living plan only.

## Counter and matched-size controls: smaller updates do not recover this pair

Retaining Adam state and lowering LR to match the first-moment-clear update
magnitude does not recover its partial success. Resetting only the counter also
fails. This narrows the explanation: global displacement size or counter age
alone is insufficient for the observed two-seed recovery.

Both new FP16 arms reproduce the common step0–3 parameters/native scores,
fresh-group loss checks and preclip gradients exactly. Counter clearing retains
both moments bitwise and changes only their bias-correction age from3 to0 before
update4. The size control retains all state and lowers LR only for update4,
from0.0001 to0.00004915403053. That intervention scales the ordinary retained
Adam update to the measured magnitude of the prior first-moment-clear arm.
All other model, objective, reference, correction, precision and sampling
settings remain fixed. These are diagnostics, not proposed production resets.

| One fresh update | Displacement L2 | Cosine with ordinary retained update | Rewards |
| --- | --- | --- | --- |
| Ordinary retained state | 0.0222530 | 1 | [0,0] |
| First-moment clearing, counter retained | 0.0109382674 | 0.5267224 | [0,0.5] |
| Retained state, reduced LR | 0.0109382701 | 0.99999999985 | [0,0] |
| Counter clearing, moments retained | 0.0383423 | 0.9999941 | [0,0] |
| Full state clearing | 0.0395709 | 0.4391237 | [0.5,0.5] |

The reduced-LR displacement matches its target with relative error2.44e-7.
Its direction remains effectively unchanged, whereas clearing first momentum
changes the distribution of parameter movements and preserves the email-only
success at the same global magnitude. Resetting the counter amplifies the
retained direction by roughly1.72, without reproducing full-reset behavior.
The counter and full-reset magnitudes are close but not exactly matched; that
comparison alone does not isolate direction. The matched-small-magnitude
comparison establishes that global norm alone does not describe these outcomes.

Eight new applied native updates comprise six replay controls and two fresh
updates, with16 loss checks,192 linear matrices and768 scalar-dot controls.
Independent temporal Adam reconstruction matches all eight parameter steps
within2.17e-9 maximum error, using the actual per-step LR and deliberate counter
intervention; gradient norms remain below1. Peak is3,005,776,384 bytes. All
exported scores are finite. Four fresh native Verifiers episodes preserve exact
adapter scores, token/logprob transport, excluded masks, initial prompts, seeds
39400/40400,2048-token per-turn budget and six-turn ceiling. Both new policies
score[0,0] and truncate on both seeds. Fourteen independent episode final-state
checks across the seven starting/control arms agree with the benchmark rewards.

No tested Adam, derivative or transport invariant fails here, and changing
counter/size does not solve this small regression. The evidence supports a
dependence on how optimizer history distributes the update across parameters;
it does not certify a general momentum remedy, learning-rate default or active
production-run explanation. These are two reused training seeds on one task,
with a short recipe and FP16 scale1024. Broader precision/task/family behavior
and full native worker/queue/refill qualification remain open.

Next move beyond this single-task optimizer-state suite to the actual native
TrainingWorker model/reference/training methods and return to Qwen/BF16 and
broader algorithm coverage. Counter or norm matching alone is not a reason to
change production settings. Native worker source inspection identifies a local
Torch-distributed construction path; its successful GPU execution is still
unproven and must not be inferred from the earlier instrumented-body checks.

External receipts: `fresh-counter-size-controls-summary.json`, two
`lfm-fresh-counter-size-*-float16.json` native arms and two matched fresh
collections. Runners/checkpoints/raw outputs remain external; only this
findings document and living plan enter Git.

## Actual native TrainingWorker: Qwen/LFM in both primary precisions

The real local worker now reproduces the direct-engine controls exactly. This
replaces the earlier instrumented-engine/body checks for this bounded path;
it does not qualify Ray dispatch, TransferQueue storage or the full training
controller.

Constructed the actual published native TrainingWorker with a real singleton
Torch process group, FSDP2 engine, GPU model and optimizer. Its reset,
set_loss_fn, infer_batch and train_batch methods run through their native
decorators; no AST-extracted body, fake engine or dispatch implementation is
used. Three SAMPO updates per arm retain the existing rank4/alpha8, LR0.0001,
beta0.02, ratio bounds0.997/1.004, native half forward, FP32 adapter masters,
FP16 scale1024 and deterministic SDPA controls. The worker supplies default
engineering fields, including microbatch1, rather than the external runner
pre-filling them.

| Actual worker arm | Applied updates | Peak CUDA allocation | Independent aggregate-loss maximum error |
| --- | --- | --- | --- |
| Qwen3.5-0.8B BF16 | 3 | 3,991,337,984 bytes | 2.01e-10 |
| Qwen3.5-0.8B FP16 | 3 | 3,990,352,896 bytes | 6.29e-10 |
| LFM2.5-1.2B BF16 | 3 | 2,439,047,680 bytes | 1.04e-7 |
| LFM2.5-1.2B FP16 | 3 | 2,438,800,384 bytes | 4.40e-8 |

All step0–3 parameter matrices and scores are bitwise equal to the appropriate
previous direct-engine run. All three captured preclip gradients per arm are
also bitwise equal. Twelve updates apply without skips;24 independent loss
checks,288 linear-matrix checks and1,152 scalar-dot references pass. Maximum
loss/score-gradient errors are5.83e-8/1.13e-10; independent Adam maximum error
is4.04e-9. Worker aggregate loss is checked against the sum of independent
microbatch loss references, including its native metric reduction and CPU return.

Sixty successful native worker inference calls return CPU scores, consume the
inference adapter flag and use resolved microbatch1. Before/after every exported
adapter, reference inference disables LoRA and reproduces the original base
scores exactly; actor inference after it restores trained scores exactly.
Four additional reference calls deliberately throw from a bound inference loss
inside the actual engine. The worker consumes the flag and restores actor scores
exactly afterward. Training inputs deliberately carry compute_loss=False and
no_lora_adapter=True: native train_batch still uses its bound loss and enabled
adapter, and reproduces the direct-engine updates. These are execution checks,
not conclusions based solely on reading its branches.

The first Qwen BF16 attempt failed an incorrect external expectation that
forward-only inference loss would be0. Native FSDP forward_step uses sentinel1
per microbatch when no loss function is supplied; the worker sums it to2 here.
The failed source/log remain external, and a new retry source checks2 explicitly
without changing the fork. Source inspection of classic and v1 TransferQueue
old/reference-logprob methods confirms they select scoring/entropy fields and
do not forward this placeholder as actor training-loss telemetry. That conclusion
is source-backed; the live TransferQueue path was not executed in this slice.

This tests cached real task traces with actual local worker/model/optimizer
execution. Qwen's original equal-episode-reward fixture intentionally bypasses
the host's spread admission; LFM uses the previously observed mixed-reward
weekly-report fixture. No fresh episodes, active-group refills or task-quality
claims are added. Ray transport, real TransferQueue lifecycle, vLLM weight
synchronization, distributed/packed execution, native TRL equivalence and
published runtime adoption remain separate gates, as do broader algorithms and
Gemma qualification. No confirmed product math defect appears in this path.

External receipt `actual-worker-matrix-summary.json` records hashes and
per-arm comparisons; `*-actual-worker-sampo-*-retry1.json` records actual method
execution and exception restoration. Native worker source hash is
fd967075bdb2bb4cc7a49781720a1b330bf27e31edfb078f982e4d290783a8dc at
published d8e472db822f2916ed81a408b8d28192be95e678. All runners, failed logs,
raw receipts and checkpoints remain outside Git. Only findings and the living
plan change in the repository.

## Real TransferQueue transport and Qwen BF16 worker updates

A private local Ray cluster with one SimpleStorage unit now exercises genuine
TransferQueue serialization and the native veRL bridge, without attaching to
production services. CPU controls preserve BF16, FP16 and FP32 scores, boolean
masks and opposing credit row-for-row, including unequal lengths, episode key
order and tags. The bridge materializes inference flags correctly, replaces
returned metadata with worker output metadata, and leaves the input handle's
metadata unchanged. This is transport evidence; the CPU control uses Torch
2.13.0+cpu and Ray2.56.1, not a GPU precision qualification.

Qwen BF16 then executes actual native TrainingWorker inference and training
through that queue with Torch2.13.0+cu130 and the existing published veRL d8
source. Three optimizer updates pass six independent loss/score checks,
72 matrix-gradient checks and288 scalar dot checks. Every parameter and score
at steps0–3, and all three preclip gradients, match the previous local-worker
control bitwise. Maximum loss error is2.20e-10, score-derivative error8.63e-12,
worker aggregate-loss error2.01e-10 and independent Adam error4.02e-9. Peak tensor
allocation is3,991,360,000 bytes. Fifteen successful inference calls and an
intentional reference-loss exception preserve exact actor/base restoration.

Failed attempts are retained. The first CPU attempt exceeded the Unix socket
path limit; subsequent CPU harness corrections handled nested return columns
and the helper's required default argument. The first GPU attempt passed a
local-worker padded fixture through a queue that reconstructs columns as nested
rows. Its response widths then counted padding as sequence length, and the loss
rejected inconsistent offsets before an optimizer step. Native controller code
uses real nested lengths. The retry trims response-aligned fields to those
lengths and removes the local padded max_response_len hint; it passes without
changing product or fork code. This does not establish a production layout bug.

Receipts live externally: live-tq-transport-probe-retry3.json,
qwen-live-tq-worker-bfloat16-retry1.json and live-tq-worker-summary.json.
This slice uses cached native task traces; Qwen's equal-reward group deliberately
bypasses admission. It runs a local GPU worker whose decorated methods use the
real queue, not a Ray-dispatched GPU actor or full training controller. FP16/LFM
model transport, controller admission/refill, fresh rollout/weight synchronization,
held-out behavior and broader algorithm/family coverage remain open. No new
product math defect or production recipe recommendation follows from this slice.

## Completed Qwen/LFM precision matrix through the real queue

Three additional arms extend the preceding Qwen BF16 queue result to Qwen FP16
and LFM BF16/FP16. Across the four arms, twelve updates apply successfully with
no skips. All24 independent loss/score checks,288 matrix-gradient checks and
1,152 scalar dot checks pass. Parameters and scores at every step0–3, and all
three preclip gradients in each arm, match the corresponding local-worker
controls bitwise. Sixty successful inference calls and four intentional
reference-loss exceptions preserve exact actor/base restoration. Both FP16
arms retain diagnostic scale1024 throughout; production default-scale
qualification remains open.

| Model / precision | Peak tensor allocation, GB | Maximum loss error | Maximum score derivative error | Clipped active tokens at update3 |
| --- | ---: | ---: | ---: | ---: |
| Qwen BF16 | 3.991 | 2.20e-10 | 8.63e-12 | 104 / 298 |
| Qwen FP16 | 3.990 | 6.19e-10 | 9.23e-12 | 88 / 298 |
| LFM BF16 | 2.439 | 5.83e-8 | 1.13e-10 | 1,024 / 2,341 |
| LFM FP16 | 2.439 | 5.40e-8 | 7.51e-11 | 2,341 / 2,341 |

GB here means decimal bytes divided by1e9, and excludes GPU allocations outside
Torch's tensor allocator. Maximum independent worker aggregate-loss error is
1.04e-7 and Adam error4.04e-9. Clipping uses the deliberate diagnostic bounds
1−.003 and1+.004 on a reused population. Update1 starts with exact current/old
ratio1 and clips no active token. Update2 clips88 Qwen tokens in either
precision and none for LFM; update3 gives the counts above. These measurements
verify clipping behavior across transported inputs. They do not establish
production clipping frequency or recommend these tight thresholds. A fully
clipped policy term also does not imply an unchanged optimizer: KL gradients
and accumulated Adam moments remain relevant.

The first LFM BF16 attempt exposed another external audit edge case: a genuinely
unpadded response had all positions active, leaving no excluded-gradient values.
The native loss completed, but the oracle's max reduction over the empty set
failed. The new retry reports0 for this empty set and passes without any product
or fork changes. Failed source/log/checkpoint artifacts remain external and
unchanged. The combined receipt is live-tq-matrix-summary.json; final arms are
qwen-live-tq-worker-float16-retry1.json and lfm-live-tq-worker-*-retry2.json.

This establishes a bounded native-worker/transport result for both architectures
and primary precisions. It does not run a Ray-dispatched GPU worker, the full
controller, admission/refill or fresh rollout synchronization. The Qwen fixture
still bypasses equal-reward admission deliberately; LFM's cached group has mixed
episode rewards, but host admission was not executed here. Fresh-task learning,
default FP16 scale, broader algorithms, TRL equivalence and Gemma remain separate
campaign gates. The new evidence does not explain the production run's poor
performance by itself.

## Native controller score storage and entropy correction

Actual veRL v1 controller old/reference scoring method bodies now run against
the real private TransferQueue and Qwen BF16 worker after three updates.
A forwarding facade calls the actual GPU worker; no synthesized scores are
supplied. The controller's stored response scores match independent per-row
causal slices at prompt_length−1 exactly, including tool positions. Masks are
unchanged. Old scores equal the current actor, while reference scores equal
the original base. Their largest response-score difference is1.03025, so the
reference identity check distinguishes trained actor and base. Actor scoring
is restored exactly afterward. Controller metrics contain only actor/entropy,
not the worker's forward-only placeholder loss2.

The first unchunked controller attempt exceeded8GB memory while allocating
full-vocabulary entropy temporaries. The existing engine's chunked entropy
option with chunk_size64 succeeds on the same real traces. The passing control
uses Posttrain's SAMPO seq-mean-token-mean aggregation, checked independently
with a scalar mean of each episode's sampled-token mean. Full controller
construction, Ray GPU dispatch and admission/refill were not executed.

Independent represented-logit probes expose a separate numerical defect:
BF16 logits[10,11] produce entropy0.5625 instead of0.582203, and uniform logits
with a large common offset can produce0 instead oflog(vocabulary size).
The chunked float32 formula also fails for a common offset1e8. These references
use the actual represented logits, so the error is not input quantization.
The old formula subtracts two large values; half arithmetic makes cancellation
visible at ordinary logit scales.

Published veRL sourcec1e477d7d83badb4be6742c9efde483956698f01 computes entropy
from normalized log probabilities, promotes half arithmetic to float32 and
handles zero-probability contributions without nonfinite values/gradients.
Unchunked half entropy now returns float32; gradients retain input dtype.
Twenty-four of36 CPU regressions fail before repair and all36 pass afterward;
six CUDA BF16/FP16/FP32 derivative controls pass too. The related utility slice
passes53 tests. Seventeen external scalar cases have max repaired value error
1.87e-8. Float64 is used only for CPU oracle/compatibility checks, not model
training. Production runtime pins and chunking defaults are unchanged.

The repaired kernel passes the real normalized Qwen controller path, with
entropy aggregation error5.41e-9. The actual metric changes from0.2021292150
to0.2021293342, only1.19e-7. Step0–3 parameters/scores and all three preclip
gradients remain bitwise identical to the pre-repair control; policy updates
in this fixture do not include an entropy bonus. Peak tensor allocation remains
3,991,360,000 bytes. This confirms a kernel defect and a bounded repair,
without attributing the production learning problem to it.

External receipts include native-entropy-translation-probe.json,
native-entropy-translation-after.json, before/after regression logs,
qwen-controller-scoring-seqmean-chunk64-bfloat16.json,
qwen-controller-scoring-entropy-fixed-bfloat16.json and
controller-entropy-repair-summary.json. The unchunked failed source/log and
all intermediate checkpoints remain preserved. Broader entropy/model/precision,
distributed, full controller and fresh-learning gates remain open.

## GRPO, GSPO and DAPO with native LFM outcome credit

The real LFM worker/model/queue path now tests three policy objectives in BF16
and FP16. The cached weekly-report episodes have observed rewards[1,0]. Native
TRL-compatible GRPO advantage generation agrees with independent sample-standard
deviation arithmetic: std0.7071067812, epsilon1e-4 and advantages±0.7070067953.
Each episode receives constant outcome credit on its sampled positions rather
than SAMPO's turn credit. The fixture generator reproduces the derived input
byte-for-byte. Unequal sampled-token counts1,317/1,024 make normalization
differences observable. The truncated row is deliberately retained in this
controlled comparison; host admission and production truncation policy are
not qualified by it.

GRPO uses Posttrain's token_clip loss and seq-mean-token-mean reduction. DAPO
uses token_clip with a global active-token denominator2,341. The GSPO arm
exercises veRL's native gspo branch, including its detached sequence ratio and
local token derivative; it is a research comparison, not a new framework
selection or an assertion of full paper/recipe equivalence. All arms use
matched initialization, rank4/alpha8 q/v LoRA, LR1e-4, beta.02 sampled k3,
two accumulated microbatches, fixed old/reference scores and three updates
on the same population. Clipping bounds.003/.004 and FP16 scale1024 are
diagnostic settings, not adopted production recommendations.

All18 updates apply without skips. Thirty-six independent loss/score cases,
432 linear matrix gradients and1,728 scalar dots pass. Maximum loss error is
6.24e-8, score derivative error7.48e-11, worker aggregate-loss error5.46e-8 and
independent Adam error2.14e-9. Peak tensor allocation is2.440GB. Ninety
successful inference calls plus six intentional reference-loss exceptions
retain exact initial-base identity and actor restoration.

| Objective / precision | Clipped sampled tokens: update1 | Update2 | Update3 |
| --- | ---: | ---: | ---: |
| GRPO BF16 | 0 | 337 | 338 |
| GRPO FP16 | 0 | 336 | 507 |
| GSPO BF16 | 0 | 0 | 0 |
| GSPO FP16 | 0 | 0 | 2,341 |
| DAPO BF16 | 0 | 318 | 337 |
| DAPO FP16 | 0 | 343 | 473 |

Each denominator is2,341 sampled tokens. GRPO/GSPO initial parameters and
scores match exactly, and their first gradients are bitwise equal in either
precision, as expected at ratio1 with constant episode credit. DAPO's first
gradient differs from GRPO by12.38% BF16 and12.33% FP16 in relative L2, with
cosines0.992445/0.992532. This agrees with the independently checked change
from equal episode weighting to token weighting. The different later clipping
patterns are measured outcomes, not evidence that the initial normalization
or clipping arithmetic failed. GSPO's fully clipped FP16 policy term still
allows KL gradients and optimizer history to affect the parameter update.

Twelve additional scalar controls check clipping signs, sampler weighting,
unequal lengths, the GSPO ratio cap and both reductions. Maximum relative
loss error is1.33e-7; the largest absolute derivative error occurs at a very
large exponential ratio and passes the stated relative tolerance. Initial
control attempts omitted required native rollout/microbatch configuration;
those harness failures and sources remain external. No new product defect
is established in this slice.

External receipts: lfm-headroom-outcome-credit-fixture.json,
lfm-native-{grpo,gspo,dapo}-{bfloat16,float16}-outcome-credit.json,
policy-algorithm-scalar-controls-retry2.json and
native-policy-algorithms-summary.json. The per-arm static scope prose retains
older runner wording; explicit algorithm/loss_agg_mode fields and the combined
summary describe these arms. Raw receipts were preserved without rewriting.
No correctness runners or raw data enter Git.

Qwen's cached equal-reward group would give zero outcome advantages, so it
cannot supply a meaningful observed outcome-credit comparison here. Genuine
Qwen mixed-reward groups, native TRL logical equivalence, complete DAPO/GSPO
recipes, full controller admission/refill, fresh task behavior and broader
algorithm/family coverage remain open. These successful arithmetic checks
do not establish that any of the three objectives improves task performance.

## Fresh FP16 task behavior after the matched native updates

Replay the base and step-three LFM adapters on two fresh generation seeds,
71400 and 72400, for simple.weekly_report_sheets_email. All five arms use
matched initial parameters, identical first prompts per seed, temperature0.8,
2,048 output tokens per turn and at most six turns. The SAMPO step-zero
parameters and fixture scores also match the outcome-credit base exactly.
This is ten native Verifiers/MCP episodes, not a complete admission/refill
training loop or an efficacy trial.

| Adapter | Rewards, seeds71400 / 72400 | Full successes | Truncated episodes |
| --- | --- | ---: | ---: |
| Base, step0 | 0 / 1 | 1/2 | 1/2 |
| GRPO, step3 | 1 / 1 | 2/2 | 0/2 |
| Native GSPO, step3 | 0.5 / 0 | 0/2 | 1/2 |
| DAPO, step3 | 0 / 0 | 0/2 | 1/2 |
| SAMPO, step3 | 0 / 0 | 0/2 | 2/2 |

All ten episodes pass exact adapter handoff, fixture-score, sampled-token,
log-probability, excluded-node mask and actual Posttrain projection checks.
Independent final-world checks agree with both native assertions and rewards.
Peak tensor allocation is3,459,928,064 bytes. These checks support the transport
and reward accounting; they do not explain away poor task behavior.

Replay every sampled response through the pinned LFM parser, with an additional
independent AST syntax check that never executes sampled code. DAPO seed71400
emits a complete tool block containing `spreadsheet: 'ss_reports'` inside a
function call. Native parsing reports malformed_structure, executes no tool,
and the episode earns zero without truncating. SAMPO seed71400 reaches an
unclosed_block after lengthy reasoning and truncates; its other seed truncates
before any tool block. GSPO seed71400 executes two parseable calls but earns
only the email assertion; the sheet row does not satisfy the task. GRPO executes
both tools and satisfies both assertions in both seeds.

Every episode has empty trace.errors and no recognized tool-failure response.
Those fields therefore cannot serve as a complete tool-format failure counter:
parser rejection happens before tool execution. This is an observed diagnostic
limitation, not evidence that the parser should execute malformed calls or that
the loss arithmetic is wrong. Retain parser status, truncation, tool execution
and semantic assertions as separate measurements.

Matched generation seeds and initial prompts do not fully isolate later
trajectory differences: pinned AutomationBench creates Gmail and Sheets object
IDs with UUID4 after tool actions. First-response differences precede this
confound. Two seeds on one task cannot justify replacing SAMPO with GRPO or
selecting a production recipe; fresh BF16 checks, more seeds/tasks and a full
iterative admission/refill run remain required.

External receipts: lfm-outcome-fresh-{base,grpo,gspo,dapo,sampo}-float16.json,
outcome-policy-fresh-fp16-summary.json and fresh-tool-grammar-audit.json.
Runners, raw traces and checkpoints remain outside Git.

## Matched fresh BF16 task behavior

The corresponding fresh BF16 matrix completes all ten native episodes and
passes the same adapter, fixture-score, token/logprob/mask/projection and
independent world-reward audits. SAMPO/base initial parameters and fixture
scores match bitwise; peak tensor allocation is3,460,183,040 bytes.

| Adapter | BF16 rewards, seeds71400 / 72400 | Full successes | Truncated episodes |
| --- | --- | ---: | ---: |
| Base, step0 | 1 / 0.5 | 1/2 | 0/2 |
| GRPO, step3 | 1 / 0 | 1/2 | 1/2 |
| Native GSPO, step3 | 0 / 0.5 | 0/2 | 1/2 |
| DAPO, step3 | 0 / 0.5 | 0/2 | 0/2 |
| SAMPO, step3 | 1 / 1 | 2/2 | 0/2 |

Native parser replay identifies an unclosed GSPO block in seed71400 and a
complete malformed DAPO block in that same seed. DAPO again uses the invalid
`spreadsheet: 'ss_reports'` function-argument syntax seen in FP16, despite other
output differences. SAMPO executes both calls and satisfies both assertions in
both BF16 seeds; both FP16 SAMPO episodes instead truncated without a tool
response. GRPO's two FP16 successes also fail to repeat in BF16. These are
precision-sensitive sampled behavior differences on a tiny single-task set,
not isolated causes, a production recipe ranking or proof of improved learning.
The same UUID4 post-action context limitation applies.

External receipts: lfm-outcome-fresh-{base,grpo,gspo,dapo,sampo}-bfloat16.json,
outcome-policy-fresh-bf16-summary.json and fresh-tool-grammar-audit-bf16.json.

## Temperature and score precision: a remaining backend difference

Inspect the candidate veRL FSDP non-fused scoring path and replay the exact
logprobs_from_logits_v2 function body on24 small CPU cases. Inputs are identical
represented BF16/FP16 logits; an independent Python scalar softmax supplies
expected values and derivatives. This separates score-path rounding from model
forward rounding. TRL's current ordinary and chunked GRPO paths promote half
logits before temperature scaling; veRL scales in the original dtype and its
fallback returns half log-probabilities.

For represented logits[-12,-4,0,2], selecting the first token at temperature0.8,
veRL's BF16 score differs by-0.0455992 from the scalar oracle. Its ratio to the
oracle probability is0.955425. For the same case FP16 error is+0.00127576.
Across24 cases, maximum score error is0.0455992 BF16 and0.00370286 FP16;
promotion before scaling reduces the maximum to3.78e-7. Maximum gradient error
is0.00614961 BF16 and0.000453979 FP16. Promoting the score calculation cannot
remove rounding when derivatives are stored back into half model coordinates.

This is a demonstrated numerical difference on controlled logits, not measured
fresh model-token drift or evidence of its contribution to poor task rewards.
The tail-token example has low probability. Both native old/current scorers
can also share the same bias, and sampler correction may account for some
rollout-versus-training differences. Do not equate the illustrative probability
ratio with the actual PPO old/current ratio. Full-model score/gradient controls,
fused-kernel coverage and a deliberate cross-backend correction remain open.
The completed BF16 behavioral run used the unchanged scoring implementation
and checked adapter handoff against the original receipts.

External receipt: temperature-score-precision-audit.json; source hash identifies
the extracted native fallback body. This test uses a scalar reference rather
than an FP64 training runtime.

### Actual model logits under CUDA autocast

Run both model families and both primary precisions on their two cached native
AutomationBench trace rows, using exact saved step-zero q/v LoRA parameters.
Extract the native fallback function body and execute it under the same CUDA
autocast context as FSDP forward_step. Every arm reproduces its saved native
fixture scores with maximum error zero. Chunk64 FP32 controls fit easily on
8GB: peak tensor allocation2.835GB LFM and2.632GB Qwen.

| Model / precision | Maximum sampled-token log-probability difference | Mean absolute difference, row1 / row2 |
| --- | ---: | --- |
| LFM BF16 | 0.153319 | 0.006847 / 0.007397 |
| LFM FP16 | 0.021459 | 0.000819 / 0.000991 |
| Qwen BF16 | 0.080662 | 0.005237 / 0.002369 |
| Qwen FP16 | 0.011010 | 0.000588 / 0.000348 |

Differences compare the native original-dtype temperature division with
promotion before division on the same represented model logits, not two model
forwards. Eight independent full-vocabulary scalar checks per arm verify the
promoted scores within1.36e-6. For LFM BF16 row1, native-versus-promoted token
probability ratios range0.857856–1.131417. These are score-path ratios, not
the optimizer's actual current-versus-old ratios.

Important correction to interpretation of the CPU slice: CUDA autocast makes
the native fallback log_softmax return FP32 in these actual model runs. The
model logits and their temperature division remain BF16/FP16, so the loss
of precision happens before normalization. The earlier CPU test correctly
describes its outside-autocast path, but its half output dtype does not describe
the actual FSDP CUDA path. Moving promotion ahead of temperature division is
the correction to qualify; promoting the final score cannot recover lost bits.

This confirms a scoring difference on real sampled-token inputs in both model
families, while retaining native model-forward rounding. Gradient/update replay,
actual old/current ratio and clipping changes, fused scoring, memory at larger
batch sizes and runtime adoption remain open. No improvement in task rewards
has been attributed to this correction.

External receipts: {lfm,qwen}-model-score-precision-{bfloat16,float16}.json.
All runners and raw evidence remain external; source hashes and exact native
score agreement tie these measurements to the candidate implementation.

### Candidate correction: first native update controls

Implement scale_logits_by_temperature in the maintained veRL candidate fork
and call it from both non-fused FSDP routes. Half logits promote before
division; FP32/FP64 arithmetic and the existing temperature floor are preserved.
Original scaling fails8/15 new scalar score/gradient/overflow/floor regressions;
the candidate passes15/15. Combined CPU utilities pass62 tests, with10
CUDA/distributed cases deliberately deselected. Actual packed/unpacked
prepare_model_outputs methods with the real CPU fallback pass four scalar
score/gradient cases and six existing top-K distillation compatibility cases.
The first packed test adapter omitted inplace_backward; retain that harness
failure and its corrected retry, without calling it a product defect.

Both corrected GRPO native LFM arms apply three finite updates. Twelve independent
loss/score checks,144 matrix checks and576 scalar dots pass. Maximum loss error
is4.59e-8, score derivative7.17e-11, aggregate worker loss5.08e-8 and Adam2.14e-9.
Peak tensor allocation grows from2.440GB to3.714GB but fits8GB.

With exact initial parameters and unchanged cached task inputs, the first
gradient changes by3.070% BF16 /0.402% FP16 relative L2, with cosines0.999530/
0.999992. This comparison includes the score path and its downstream sampler
correction; it is not an isolated model-Jacobian ablation. GRPO clipping counts
out of2,341 active tokens change BF16[0,337,338]→[0,327,332] and
FP16[0,336,507]→[0,354,501]. Promotion does not make first-update clipping
appear: native old/current scores still agree initially. It changes later
counts modestly in these controls. Task-quality improvement is untested.

GSPO native BF16/FP16 arms have also finished successfully; complete combined
analysis awaits the still-running DAPO arms. Candidate source, tests, ledger
and consumer documentation remain unpublished; production pins are unchanged.
Raw receipts and runners remain external. Partial comparison receipt:
temperature-repaired-update-partial-summary.json.

The complete six-arm native matrix subsequently finishes:18/18 updates,
36 independent loss/score-gradient cases,432 linear matrix checks and1,728
scalar dots pass. Max errors are loss4.59e-8, score derivative7.42e-11,
worker aggregate5.08e-8 and Adam2.14e-9. Peak3.714GB. Seventeen focused
CPU/CUDA temperature regressions pass, alongside the62 CPU utility and10
route/distillation cases already recorded. No FP16 update is skipped; scale1024
remains a diagnostic setting rather than default-scale qualification.

| Objective / precision | Baseline clipped tokens, updates1/2/3 | Corrected clipped tokens | First-gradient relative L2 change |
| --- | --- | --- | ---: |
| GRPO BF16 | 0 / 337 / 338 | 0 / 327 / 332 | 3.070% |
| GRPO FP16 | 0 / 336 / 507 | 0 / 354 / 501 | 0.402% |
| GSPO BF16 | 0 / 0 / 0 | 0 / 0 / 0 | 3.070% |
| GSPO FP16 | 0 / 0 / 2,341 | 0 / 0 / 2,341 | 0.402% |
| DAPO BF16 | 0 / 318 / 337 | 0 / 308 / 347 | 3.128% |
| DAPO FP16 | 0 / 343 / 473 | 0 / 345 / 484 | 0.397% |

Every clipping denominator is2,341 sampled tokens. These results support the
correction's numeric/update consistency while showing that it does not explain
zero first-update clipping: old/current scores are both recomputed through the
same corrected scorer. Keep model/head rounding, sampler mismatch and recipe
reuse separate. No corrected fresh-task efficacy comparison has run yet.

Published source candidate: f5333c4f647494e497896eaed14160e2cd7186c4 in
carbonteq-ai/verl, codex/posttrain-math-parity. Production pins/images are
unchanged. Fused/other engines, actual-model SAMPO replay, full admission/refill,
distributed/large-context memory and task-quality attribution remain open.
External complete receipt: temperature-repaired-update-summary.json.

### Qwen SAMPO memory gate after the scoring correction

Replay the unchanged full Qwen native trace population under the corrected
scorer. The initial BF16 arm fails in backward while requesting1.30GiB.
Expandable allocator segments do not solve it: retry1 requests1.42GiB when
only1.23GiB is free, with4.41GiB allocated and126MiB reserved-unused. Preserve
both failures and step-zero adapters. The LFM correction qualification does
not establish that Qwen's corrected update fits this8GB desktop GPU.

Native enable_activation_offload=True also fails: a CUDA mapping allocation
warning precedes AsyncDoubleBufferGroupOffloadHandler.tensor_pop's assertion
that a saved state must no longer be a tuple. This is an observed offload
recovery failure; its independence from memory exhaustion is unproven.
Do not remove that assertion or silently return an unrestored state.

A synchronous reference around prepare_model_outputs using PyTorch save_on_cpu
fails when packing a jagged tensor with symbolic dimensions. Retry4 narrows
the saved-tensor hooks to dense scoring tensors and leaves jagged tensors on
their original device. It retains full traces, model precision, FP32 scoring,
objective, optimizer and native queue; execution and independent derivative/
Adam checks are pending. This is an experimental memory reference, not a
production offload design or a completed Qwen qualification. Production pins
remain unchanged.

External failed receipts: qwen-live-tq-sampo-bfloat16-temperature-repaired
and retry1/retry2/retry3 logs plus step-zero adapters. Retry4 subsequently fails
backward requesting1.42GiB with1.25GiB device-free. Filtering jagged tensors
solves the symbolic-shape hook failure but does not solve peak backward memory.
All five attempts are terminal and preserved. The next correction to qualify
is row-wise FP32 probability scaling directly from half logits, avoiding a full
FP32 logit-gradient buffer while retaining the same trace population and
derivatives. This is a proposed memory diagnosis, not a tested implementation.
All tools/raw evidence remain outside Git.

### Row-wise correction clears the score-only SAMPO memory gate

Candidate temperature_scaled_logprobs unbinds represented half logits into
token rows before FP32 temperature scaling and stable log_softmax. Autograd
returns half row gradients without a full FP32 logit-gradient buffer. Both
non-fused FSDP routes select it for half score-only requests. Entropy,
sum-pi-squared and distillation requests retain the prior full-FP32 path;
their memory requirements are not fixed by this candidate.

All four Qwen/LFM BF16/FP16 SAMPO arms now complete with the original full
trace populations, ordinary allocator and no additional offload. Twelve
updates,24 independent losses/score derivatives,288 matrices and1,152 scalar
dots pass. Max errors are loss5.39e-8, derivative9.96e-11, worker aggregate
4.62e-8 and independent Adam4.12e-9. Sixty successful native inferences and
four intentional reference-loss exceptions preserve base identity and actor
restoration. FP16 scale1024 applies all updates without skips.

Thirty-two independent full-vocabulary scalar score checks agree within1.54e-7.
Initial parameters match the baseline exactly. Qwen BF16 initial scores also
agree with the preserved full-FP32 candidate within2.27e-6. Forty-two focused
CPU/CUDA/route/distillation regressions and95 combined utility/route cases pass,
with four distributed cases deliberately excluded from the combined command.

| Family / precision | Peak tensor allocation | Baseline clipped tokens, updates1/2/3 | Corrected clipped tokens | First-gradient relative L2 change |
| --- | ---: | --- | --- | ---: |
| Qwen BF16 | 3.218GB | 0 / 88 / 104 | 0 / 0 / 16 | 5.246% |
| Qwen FP16 | 3.218GB | 0 / 88 / 88 | 0 / 88 / 88 | 0.724% |
| LFM BF16 | 2.014GB | 0 / 0 / 1,024 | 0 / 0 / 0 | 3.135% |
| LFM FP16 | 2.014GB | 0 / 0 / 2,341 | 0 / 0 / 2,341 | 0.417% |

Clipping denominators are298 Qwen and2,341 LFM sampled tokens. First-update
ratios still equal1. Comparisons include corrected scores and downstream
sampler correction, not an isolated model-Jacobian ablation. The reduction
in BF16 clipping is an observed update difference, not proof of improved
task quality or a reason to force clipping through a different recipe.

Published row-wise source candidate:7cf23e101ba70813a0b23392465b4a6eaf073731
in carbonteq-ai/verl on codex/posttrain-math-parity. This score-only path
bypasses optional flash CE; throughput/fused integration, broader algorithm
replay with this implementation, controller admission/refill, default FP16
scale, distributed/multi-family gates and task-quality impact remain open.
Production pins/images are unchanged. External receipts:
{qwen,lfm}-live-tq-sampo-{bfloat16,float16}-temperature-rowwise.json and
temperature-rowwise-sampo-summary.json; tools/raw artifacts remain external.

## Row-wise GRPO, GSPO and DAPO replay; selected normalization cancellation

The published row-wise score-only candidate now completes six LFM native
model/queue arms, BF16 and FP16 for each algorithm, three applied updates each.
All18 updates,36 independent loss/score checks,432 matrix checks and1,728 scalar
dots pass. Maximum errors: loss4.73e-8, score derivative7.33e-11, worker loss
6.01e-8 and Adam2.14e-9. Peak tensor allocation is2.014GB in every arm.
Initialization and fixture hashes match the pre-temperature controls exactly;
FP16 scale1024 applies every update. Bounds.003/.004 are diagnostic.

| Algorithm / dtype | Baseline clipped sampled tokens, steps1/2/3 | Row-wise corrected | First gradient relative L2 change |
| --- | --- | --- | --- |
| GRPO BF16 | 0 / 337 / 338 | 0 / 338 / 329 | 3.087% |
| GRPO FP16 | 0 / 336 / 507 | 0 / 351 / 497 | .401% |
| GSPO BF16 | 0 / 0 / 0 | 0 / 0 / 0 | 3.087% |
| GSPO FP16 | 0 / 0 / 2341 | 0 / 0 / 2341 | .401% |
| DAPO BF16 | 0 / 318 / 337 | 0 / 323 / 365 | 3.091% |
| DAPO FP16 | 0 / 343 / 473 | 0 / 337 / 487 | .402% |

There are2,341 active sampled tokens per update. The comparison includes the
downstream sampler correction, not an isolated Jacobian change. Stable row-wise
scoring also differs slightly from the earlier full-FP32 implementation; later
BF16 clipping can magnify small gradient differences. These cached updates do
not establish fresh task quality, recipe superiority or full-controller admission.
GSPO remains an explicit research branch rather than a new framework selection.

A separate exact-source scalar audit found a defect in both TRL
selective_log_softmax and veRL logprobs_from_logits_v2. Both FP32 paths computed
selected_logit-logsumexp(logits). In exact arithmetic a common shift cancels:
log p_j=(z_j-max(z))-log(sum(exp(z_i-max(z)))). Its derivative is
1[i=j]-p_i, whose vocabulary sum is zero. The original implementations first
formed the large absolute normalizer, losing its small correction. The
logsumexp backward normalization also inherited rounding of that value.

| Represented equal FP32 logits | Original selected score | Original gradient | Stable scalar reference |
| --- | --- | --- | --- |
| 10000 / 10000 | -.693359375 | .50010610 / -.49989390 | -log2; .5 / -.5 |
| 600000 / 600000 | -.6875 | .49716842 / -.50283158 | -log2; .5 / -.5 |
| 1e8 / 1e8 | 0 | 0 / -1 | -log2; .5 / -.5 |

Twelve of16 exact-source FP32 cases exceeded1e-5 against independent Python
scalar values/derivatives. Dedicated native regressions fail6/8 TRL CPU cases
and5/18 veRL CPU cases before repair; the latter includes FP64 diagnostic
cancellation with tighter tolerance. FP64 is a scalar control, not model training.
Normalize with batch-row log_softmax before gathering in every dtype. After
repair16/16 external scalar cases pass, TRL28 single/top-K/repeated-index CPU/CUDA
tests pass, and veRL68 selected-score/temperature/actual FSDP-route cases pass.
Two intentional FP16 offsets exceeding its finite range skip. Ruff/diff pass.
Half-input behavior, dtype and existing batch-row processing remain intact.

Published source candidates: TRL4020c122e4ba2147829ecc0bbaddf6b566a0c8b5 and
veRL661bbf395e90a060acde0ec0bbed84ef67b5cee3. The repaired FP32/FP64 utilities
retain normalized vocabulary buffers; larger-context backward memory and
throughput need qualification. The independently qualified half score-only
token-row path already uses stable log_softmax. No extreme common offsets have
been demonstrated in actual model traces, so this corner-case defect is not
evidence that it caused the current run's poor performance. Native TRL optimizer,
full controller, runtime assets, production pins and task-quality gates stay open.

External receipts:temperature-rowwise-update-summary.json,
lfm-native-{grpo,gspo,dapo}-{bfloat16,float16}-temperature-rowwise.json,
selected-logprob-offset-baseline.json and selected-logprob-offset-corrected.json.
Runners and raw artifacts remain outside Git.

## Actual-model selected-score qualification

Both repaired source utility bodies were exercised on represented FP32 logits
from actual Qwen3.5-0.8B and LFM2.5-1.2B-Thinking forwards in BF16 and FP16.
Use the existing first recorded AutomationBench response with the complete
shared prompt:1,119 prompt/289 response tokens for Qwen and1,573/1,675 for LFM.
SDPA forward, temperature.8, immutable model/fixture revisions are retained.
Each backend completes backward for the mean selected score at offsets0 and
10000. Offset10000 is artificial stress; oracle inputs preserve its FP32 rounding.

| Model / forward dtype | Response × vocabulary | Peak allocated bytes | Maximum scalar score error | Maximum checked derivative error |
| --- | --- | --- | --- | --- |
| Qwen BF16 | 289 × 248320 | 3151080960 | 1.290e-7 | 2.151e-10 |
| Qwen FP16 | 289 × 248320 | 3151080960 | 1.202e-7 | 5.968e-10 |
| LFM BF16 | 1675 × 65536 | 4544710656 | 3.386e-7 | 1.369e-10 |
| LFM FP16 | 1675 × 65536 | 4544710656 | 1.186e-7 | 9.964e-11 |

All16 scorer backwards pass. Eighty independent Python full-vocabulary scalar
score checks also verify selected, adjacent and maximum-logit gradient coordinates;
maximum per-row vocabulary gradient mass is1.514e-9. The mean-score derivative
includes the explicit1/response_length factor. Model parameters remain on GPU,
but logits are detached before backward: this does not qualify model derivatives,
optimizer updates, retained training activations or full-model backward memory.
The recorded peak is allocated tensor memory, not total device use or throughput.

Observed unshifted scaled logits range Qwen BF16[-28.4375,48.75], Qwen FP16
[-28.3984375,48.59375], LFM BF16[-24.53125,47.5], and LFM FP16
[-24.58984375,47.5390625]. These traces do not show the large offsets that drove
the synthetic cancellation failure. A model's logits can legitimately differ
across precision/forward kernels; no fresh-quality inference follows.

The first audit fails before forward because the external harness assumed a
nested prompt list. Preserve failed-fixture-shape source snapshots and original
Qwen BF16 log; the corrected flat-prompt retry uses distinct filenames. Four
retry arms terminate successfully. External receipts:
{qwen,lfm}-{bfloat16,float16}-selected-score-model-audit-retry1.json/.log.
Tools/raw data are excluded from Git; full model/optimizer backward, throughput,
full controller and native TRL optimizer qualification stay open.

## Native TRL Trainer updates on recorded SAMPO traces

The final four-arm audit runs actual GRPOTrainer.train, model backward,
Accelerate optimizer accumulation and callbacks on full recorded Qwen/LFM
AutomationBench token fixtures, supplied SAMPO credits and stored native rewards.
Each arm applies three optimizer updates from two microbatches. Rank4/alpha8,
LR1e-4, constant schedule, beta.01, diagnostic bounds.003/.004, checkpointing
without reentrancy and supported fused SDPA are explicit. FP16 scale1024 is
controlled; this is not default-scale qualification. No fresh generator runs.

| Model / dtype | Applied updates | Max independent loss error | Max score derivative error | Max Adam update error | Peak tensor bytes |
| --- | --- | --- | --- | --- | --- |
| Qwen BF16 | 3 | 9.726e-10 | 7.276e-12 | 1.838e-9 | 5153868800 |
| Qwen FP16 | 3 | 8.598e-10 | 7.276e-12 | 1.826e-9 | 5153714176 |
| LFM BF16 | 3 | 3.454e-8 | 2.911e-11 | 9.429e-10 | 6091453952 |
| LFM FP16 | 3 | 3.736e-8 | 2.911e-11 | 9.434e-10 | 6091454976 |

All24 loss/score cases pass; excluded tool/padding score gradients are exactly
zero. Scaler-aware accumulated gradients match actual pre-optimizer gradients
exactly, including all LoRA parameters; parameters change finitely at every step.
Independent Adam arithmetic computes updated moments, bias correction, epsilon,
decay and parameter displacement from each step's observed prior optimizer
state and actual gradient. It checks per-step update equations, not independently
maintained moment history or arbitrary checkpoint restoration.

Native logged clipping-region means for steps1/2/3 are Qwen BF16[0,0,.379312],
Qwen FP16[0,.330827,.330827], LFM BF16[0,0,0], and LFM FP16[0,0,1]. These are
logged native aggregation fractions rather than counts or a quality measure.
The reused population allows clipping after the first update; absence of clipping
in LFM BF16 does not by itself establish a broken recipe.

The unscaled backward disagrees with normalized scaled FP16 backward, maximum
relative gradient discrepancies.799984 Qwen and1.253146 LFM. Exact agreement
with the scaler-aware audit does not erase this half-arithmetic sensitivity.
Default65536 overflow/backoff and applied-gradient checks remain a separate gate.

Preserve two LFM BF16 failures. The inherited first harness disables fused SDPA;
math attention requests1.26GiB with113.62MiB free. Enabling fused SDPA gets through
native derivatives, but the full head-gradient diagnostic then requests838MiB
with296.56MiB free. Bound those external norm/count reductions in1,048,576-element
chunks; model/loss/backward stays unchanged and both final LFM arms pass. Initial
Qwen BF16/FP16 arms also pass six updates before the temporal-Adam extension.

This is actual Trainer execution, but supplied cached tokens/rewards bypass
host rollout/admission/refill. Qwen rewards[1,1] would still fail the frozen
host spread gate; LFM's recorded truncation is retained diagnostically and its
metadata is preserved in final arms. Initial adapters are seeded rather than
loaded from matching veRL exports; scoring, sampler correction, beta and native
engine layout are not all matched. Do not claim exact native backend trajectories,
fresh learning or production adoption. Broader native algorithms, default scaler,
full host and runtime gates stay open. No production source/pin changes.

External final receipts:qwen08-trl-recorded-sampo-{bfloat16,float16}-adam-sdpa-retry1.json,
lfm12-trl-recorded-sampo-{bfloat16,float16}-adam-sdpa-bounded-retry2.json,
native-trl-recorded-adam-summary.json plus original terminal logs. Scripts/raw
results remain outside Git.

## Default FP16 scaler and the independent-gradient precision gate

Repeat actual recorded SAMPO Trainer.train at initial scale65536, retaining
the same model precisions, supplied traces, masks, configured seed, optimizer
settings and bounded audit. Both families apply three finite optimizer updates
without scale backoff or skipped steps. Qualification has a split result:

| FP16 model | Qualification | Max loss error | Max scalar score derivative error | Max independent scaled-gradient relative error | Max Adam error | Peak bytes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen | pass | 1.179e-9 | 1.456e-11 | 0 | 1.832e-9 | 5037937152 |
| LFM | fail | 7.026e-8 | 2.911e-11 | .000842657 | 9.429e-10 | 6091454976 |

The gate is relative error below.0001. LFM update errors are[0,.000842657,
.000513361]. All twelve loss/mask checks pass and excluded gradients remain0.
VJPs seeded by the actual native loss gradient reproduce accumulated optimizer
gradients exactly. Seeding them with the independently computed scalar gradient
does not meet the chosen LFM precision gate. Tiny score-derivative differences
can cross FP16 rounding boundaries and accumulate through model backward; this
is numerical sensitivity beyond tolerance, not demonstrated incorrect loss math.
Unscaled-versus-scaled gradient discrepancies remain.805754 Qwen/1.157424 LFM.

The first status interpretation incorrectly treated applied updates/exit0 as
full qualification. Reading qualification_status and each accumulated-gradient
field corrects it. Preserve the LFM fail, the original analyzer assertion and
source; thresholds are not relaxed. A Qwen allocator warning during the run is
not a terminal OOM: all three updates and the final receipt completed.

A diagnostic LFM control promotes only selected scores/loss arithmetic to
FP64 while retaining FP16 model/backward, scaler65536 and the same recorded
input hash. Its three updates pass with all independent scaled-gradient errors0;
loss error2.665e-15, score derivative2.169e-19, Adam9.429e-10, peak6091594752.
This is a small loss-arithmetic reference, not FP64 model training or a recipe
recommendation. It is consistent with FP32 loss-rounding amplification in half
backward. It does not prove equivalent parameter trajectories, current-run
causation, task quality or stability across larger credit ranges. No source/pin
changes follow; broader default-scale precision qualification remains open.

The native logs also reveal an observability distinction. advantages statistics
are calculated over episode reward baselines before supplied SAMPO token credit
replaces advantages. On these Qwen rewards[1,1], logged episode abs mean is0,
but104 of298 active sampled tokens have nonzero credit, with active-token abs
mean.0082885906. LFM episode abs mean.707006812 differs from token-credit abs
mean.9031717215 over2,341 nonzero tokens. Every captured loss credit tensor
agrees with its recorded fixture within1e-7. Episode telemetry is insufficient
to infer absence or magnitude of the token signal; report the two separately.

External receipts:{qwen08,lfm12}-trl-recorded-sampo-float16-default-scale65536.json,
lfm12-trl-recorded-sampo-float16-default-scale65536-loss64-control.json,
native-trl-default-scale-summary.json and native-trl-loss-precision-control-summary.json.
Tools/raw receipts remain outside Git; the full campaign remains active.

## Native TRL GRPO/DAPO on unequal recorded LFM responses

Actual Trainer.train now runs GRPO row-mean and DAPO global-token reductions on
the same recorded LFM AutomationBench group, native rewards[1,0],1,317/1,024
active sampled tokens, BF16 and FP16. Each arm applies three finite updates from
two accumulated microbatches. Model/LoRA, checkpointing, fused SDPA, LR1e-4,
beta.01, bounds.003/.004 and diagnostic truncation retention match the earlier
native audit. FP16 scale1024 is controlled, not the default-scale gate.

The native episode advantages±.707006812 agree with independently normalized
observed rewards using unbiased std and epsilon1e-4 within1.677e-8. This external
oracle replaces the original synthetic[0,1] assumption; no trainer change.

| Algorithm / dtype | Qualification | Max loss error | Scalar-oracle accumulated-gradient relative error | Max Adam error | Peak bytes |
| --- | --- | --- | --- | --- | --- |
| GRPO BF16 | pass | 2.981e-8 | 0 | 9.460e-10 | 6091442688 |
| GRPO FP16 | fail | 2.981e-8 | .000627623 | 9.441e-10 | 6091443712 |
| DAPO BF16 | pass | 2.493e-8 | 0 | 9.432e-10 | 6091442688 |
| DAPO FP16 | fail | 2.577e-8 | .000588603 | 9.441e-10 | 6091443712 |

All12 optimizer updates apply and24 scalar loss/mask/credit cases pass. Maximum
score-derivative error5.821e-11; excluded score gradients remain0. Actual-loss
VJP accumulation reproduces every pre-optimizer gradient exactly, but the
independently rounded scalar-gradient VJP exceeds the.0001 FP16 gate. Preserve
that failed qualification rather than infer success from finite steps.

Native logged clipping-region means, steps1/2/3: GRPO BF16[0,.134744,.147873],
GRPO FP16[0,.151399,.217798], DAPO BF16[0,.140658,.148850], DAPO
FP16[0,.154870,.207001]. These are native aggregate fractions, not independent
token counts or a quality claim. Different denominators are verified directly
by the scalar loss reference on unequal active row lengths.

Two FP16 loss-only precision controls each apply three further updates. Both
pass with all scalar-oracle accumulated-gradient errors0, while retaining the
FP16 model/backward and scale1024. Same recorded input hash/configured seed;
loss arithmetic is a diagnostic higher-precision reference. This extends the
SAMPO rounding-sensitivity observation to GRPO/DAPO. It does not prove a wrong
loss formula or justify FP64 model/loss adoption, a tolerance relaxation, or
attributing poor training quality to this small gradient discrepancy. Complete
native backend trajectory equivalence, conditioning/rounding analysis, default
scale, host admission/refill and fresh quality remain separate.

The first external wrapper fails before model forward because textual insertion
matches its own quoted needle. Preserve failed-policy-wrapper sources and the
original GRPO BF16 log. Correcting the external insertion yields terminal retry
receipts. No production/fork/pin changes. External artifacts:
lfm12-trl-recorded-{grpo,dapo}-{bfloat16,float16}-native-adam-retry1.json,
lfm12-trl-recorded-{grpo,dapo}-float16-native-adam-loss64-control.json,
native-trl-recorded-policies-summary.json and native-trl-policy-loss-controls-summary.json.
Tools and raw results remain outside Git.

## Truncation masking can remove the entire group-relative policy signal

Enable mask_truncated_completions in actual BF16 native Trainer on the recorded
LFM group:1,317 active tokens in the valid response and1,024 in the truncated
response before masking, stored rewards[1,0]. The maintained source excludes
the truncated row from both attention/loss masks and reward normalization.
The first external oracle expected both rewards to remain and failed; preserve
those original receipts and correct only the oracle to respect scorable rows.

| Objective | Final mathematical classification | Productive updates | Masked-row zero-loss cases | Maximum loss error | Adam error |
| --- | --- | --- | --- | --- | --- |
| SAMPO, supplied credit | masked-loss checks pass | 3 | 3 | 4.578e-8 | 9.412e-10 |
| GRPO | expected zero-signal negative control | 0 | 3 | 0 | 0 |
| DAPO | expected zero-signal negative control | 0 | 3 | 0 | 0 |

There are18 final loss/mask checks. All excluded gradients are0. GRPO/DAPO
retain one scorable reward, whose centered advantage is0; the singleton's
unbiased std is undefined and native NaN sanitization yields zero advantages.
Both objectives therefore have zero losses and model gradients. Three optimizer
attempts per objective leave parameters unchanged: these begin with zero
moments and zero weight decay. Their inherited productive-update qualification
remains fail; this is an intended mathematical negative control, not a trainer
bug or release qualification. Existing optimizer momentum could behave differently.
SAMPO retains supplied nonzero token credit on the valid row. It does not
recompute anchor/group credit after masking, so do not infer the production
credit builder or host admission would retain this group. Peaks6.092GB.

The source dynamic/active retention predicate compares group std to a positive
epsilon; NaN fails that comparison. Those sampler modes would exclude the
singleton group. The static native loops used here disable them. Logged
frac_reward_zero_std remains0 for the undefined std, while actual policy signal
is0; that metric alone is insufficient to diagnose uninformative groups.

An independent exact Fraction calculation enumerates all three-state groups
(valid success, valid failure, unscorable/truncated) for27 conditional IID cases.
Let G be completions per prompt, p the probability a completion is scorable and
q its conditional binary success probability. A group has both reward outcomes
with probability:

P(informative)=1-(1-pq)^G-(1-p(1-q))^G+(1-p)^G.

This follows by inclusion-exclusion: subtract groups with no valid success and
no valid failure, then restore their overlap, where every completion is unscorable.
All27 exact enumeration checks match the formula.

| G | Hypothetical p=.5, q=.5 | Hypothetical p=.8, q=.5 |
| --- | --- | --- |
| 2 | 12.5% informative | 32% |
| 4 | 42.96875% | 74.24% |
| 8 | 80.36804% | 96.641024% |

For G2/p.5,75% of groups have at most one scorable completion even before
same-reward filtering. These are toy conditional probabilities, not estimates
from two observed traces. Correlated generations, prompt difficulty, fractional
rewards, host admission, sampling costs and actual truncation policy matter.
Do not adopt a larger group or attribute the current run's poor learning/no
clipping from this table. Measure actual scorable group counts and retention
first. FP16 masked controls, live refill/admission and fresh quality remain open.

External artifacts:lfm12-trl-recorded-sampo-bfloat16-native-truncation-mask.json,
lfm12-trl-recorded-{grpo,dapo}-bfloat16-native-truncation-baseline-retry1.json,
native-trl-masked-baseline-summary.json and masked-group-information.json,
plus original oracle-failure receipts. No production/pin changes; tools/raw
data remain outside Git.

## Native zero-signal groups with retained Adam history

Repeat actual GRPO Trainer.train in BF16 and FP16 with beta0 and one iteration
per supplied population. First override the recorded truncation flags to keep
both responses and obtain one informative[1,0] group. Subsequent populations
restore the recorded truncation flags, leaving one scorable reward and zero
group-relative advantage. Tokens/rewards are supplied fixtures, not freshly
sampled policy trajectories. FP16 scale1024 remains controlled.

| Current step | Current gradient norm | BF16 maximum parameter change | FP16 maximum parameter change |
| --- | --- | --- | --- |
| 1, informative | .132428 BF16 / .131861 FP16 | 9.999989e-5 | 9.999987e-5 |
| 2, singleton masked group | 0 | 6.700571e-5 | 6.700571e-5 |
| 3, singleton masked group | 0 | 5.179560e-5 | 5.179560e-5 |

Both later microbatches at each zero-signal step have exactly zero loss and
score gradients. Actual pre-optimizer gradients are zero; KL and weight decay
are absent. Adam retains the previous first/second moments, so parameters still
move. All twelve scalar loss/mask checks pass; six actual optimizer events agree
with independent per-step Adam equations within2.424e-11. Peak6.093GB. The
inherited successful-update counter reports3 because it tests finite parameter
movement; only the first population supplies nonzero current policy credit.

For one nonzero scalar gradient followed by zeros, with negligible epsilon,
the update magnitude relative to the first is approximately

R_t=[b1^(t-1)(1-b1)/(1-b1^t)] /
sqrt[b2^(t-1)(1-b2)/(1-b2^t)].

With b1=.9,b2=.999, this gives about.670058 at step2 and.517957 at step3,
consistent with the measured displacement. The independent checks retain actual
epsilon, observed prior state and every coordinate. This is normal Adam history,
not a demonstrated optimizer defect. Native/static dead-group execution can
therefore produce movement without fresh policy signal; production host
admission/filtering/refill and any current-run occurrence need direct evidence.

The original raw settings dictionary has inherited hard-coded beta.01 and
num_iterations3. Preserve it; effective control settings are beta0/iterations1.
Reconstruct the exact compiled native source without running the model: its hash
bcbfcc54f80a3155c2c05a92c6d58a5683487d17de95399ddfde58748d92b90d matches both
receipts, and AST literals verify those settings. Companion effective-settings
and summary receipts carry the correction. Future external harness output resolves
literal overrides from its compiled GRPOConfig. This repair changes metadata,
not model/loss/optimizer behavior or production source. Old and corrected harness
snapshots are archived separately; original receipts remain immutable.

External artifacts:lfm12-trl-recorded-grpo-{bfloat16,float16}-native-warm-zero-beta0.json,
native-trl-warm-zero-summary.json and native-trl-warm-zero-effective-settings.json.
All tools/raw data remain outside Git; broader qualification is ongoing.

## Native active sampling: rejection and controlled recovery

Run actual LFM Trainer.train with active_sampling enabled, maximum two rounds,
group size2 and truncated completion masking. DAPO and supplied-credit SAMPO
are supported selections; ordinary GRPO without precomputed credit is rejected
by native configuration. BF16 and FP16 each exercise exhaustion and recovery.

| Control | Arms | Collection rounds per arm | Optimizer updates per arm | Maximum Adam error |
| --- | --- | --- | --- | --- |
| Both candidates masked to one scorable reward | 4 | 2 | 0 | No optimizer state created |
| First candidate masked, second made fully scorable | 4 | 2 | 1 | 2.661e-11 |

Every exhaustion arm raises the expected bounded-refill error before loss calls,
leaves parameters bitwise unchanged and creates no Adam state. Every recovery
arm passes two scalar loss/mask checks and one real update. Maximum scalar loss
error2.493e-8; score derivative and excluded-token errors0; independently scaled
optimizer accumulation error0. FP16 scale remains1024 with no skipped update.
Unscaled-versus-scaled gradient differences4.901% DAPO/4.011% SAMPO are preserved;
the scaled reference matches exactly. Peaks3.698GB rejection/6.090GB recovery.

This executes native active-sampling selection and model scoring over cached
recorded tokens. Recovery deliberately changes only the second candidate's
truncation flags to both scorable, while token/reward bytes and supplied SAMPO
credit remain unchanged. Dataset rows duplicate prompt metadata. Consequently
these controls do not qualify fresh generation, host admission/task uniqueness,
or SAMPO credit recomputation. The native pool must contain two dataset rows for
this control: the original one-row attempt yielded zero batches, then failed
the audit's expected-rejection assertion. Preserve that failure and the earlier
unsupported GRPO attempt; neither establishes a production numerical defect.

External artifacts:lfm-native-active-masked-{dapo,sampo}-{bfloat16,float16}
-rejection-retry1.json/-recovery.json and native-trl-active-sampling-summary.json.
Tools and raw receipts remain external; no production recipe change is adopted.
