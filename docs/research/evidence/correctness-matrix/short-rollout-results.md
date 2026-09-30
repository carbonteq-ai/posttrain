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

JSON files beside this report retain the complete public simulated trajectories,
per-round rewards, truncation, advantages, gradients, updates and drift. Failed
initial parser and OOM runs remain machine-local and are described above. The
runner writes each completed round atomically. Run serially and fence nonfinite
gradients; a shorter fixture is the recovery for the full FP32 OOM, not evidence
that the larger failed control passed.
