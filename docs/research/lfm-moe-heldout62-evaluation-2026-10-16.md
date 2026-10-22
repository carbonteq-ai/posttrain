# LFM2.5 MoE evaluation on AutomationBench heldout62

The new MoE results extend the [Claude report’s heldout62 comparison](https://claude.ai/artifact/WDjeDT4tkeEZzLntrRthGq) with LFM2.5-8B-A1B and LFM2.5-24B-A2B. Qwen3.5 has the highest point scores; 24B is competitive within uncertainty, while 8B is substantially weaker. The table below brings the base models, SFT, DPO and SAMPO checkpoints together.

## All models on heldout62

### One comparison across all ten models

Each model was evaluated on 62 tasks with five attempts per task. Success, partial credit, harms and harm-free rate below use the same **61 fully scored tasks** for every row, excluding `sales.milestone_celebration`, which has an environment scoring bug. **Goal rate uses 55 tasks** with five defined goal-rate observations in all ten models. “Scored / 310” shows coverage of the full run, not the table’s denominator. Values are recalculated on these common cohorts, so they differ slightly from the original Claude table.

| Model or checkpoint | Success ↑ | Partial credit ↑ | Goal rate ↑ | Harms / episode ↓ | Harm-free ↑ | Scored / 310 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LFM2.5-2.6B base | 49.8% | 73.7% | 66.0% | 1.079 | 72.1% | 305 |
| LFM2.5-2.6B SFT v1 | 50.5% | 73.6% | 68.2% | 1.223 | 70.2% | 306 |
| LFM2.5-2.6B SFT v2 | 50.2% | 72.9% | 66.3% | 1.318 | 68.9% | 306 |
| LFM2.5-2.6B DPO v1 | 49.5% | 74.1% | 68.3% | 1.541 | 66.9% | 305 |
| LFM2.5-2.6B DPO v3 | 49.5% | 74.0% | 67.4% | 1.321 | 68.5% | 306 |
| LFM2.5-2.6B SAMPO cont100 | 49.8% | 74.9% | 67.6% | 1.298 | 70.2% | 306 |
| LFM2.5-8B-A1B | 35.4% | 46.7% | 41.2% | 0.646 | 85.9% | 307 |
| LFM2.5-24B-A2B retry | 53.8% | 81.7% | 79.5% | 1.003 | 71.8% | 305 |
| Qwen3.5-4B base | 55.1% | 82.4% | 81.0% | 1.157 | 73.4% | 306 |
| Qwen3.5-9B base | 55.4% | 83.9% | 83.5% | 1.020 | 75.4% | 310 |

#### What this shows

- **24B is a stronger LFM starting point than 8B.** Success rises from 35.4% to 53.8%, with much higher partial credit. It has not established a success advantage over Qwen4B or Qwen9B.

- **The 2.6B training variants cluster near their base model on success.** Their point estimates range from 49.5% to 50.5%; harms rise from 1.079 per episode for base to 1.223–1.541 across the trained variants.

- **Fewer harms alone do not establish a better agent.** The 8B model has the highest harm-free rate, but also the lowest success and partial credit.

Success means `task_completed_correctly == 1`. Partial credit is assertions
passed divided by assertions scored. Values are averaged within task first,
then across tasks; they are not pass@5.

Harm findings are unique valid outcome assessments with lower-is-better
direction and value at least one. A harm-free episode has zero such findings;
this measures the benchmark's declared guards, not general model safety.
The 8B model's lower activity and lower success coexist with fewer harms;
the evaluation does not establish why. Moving from 8B to 24B increased success
by 18.4 percentage points but reduced the harm-free rate by 14.1 points
(paired 95% interval −22.3 to −5.9).

### Goals are measured separately from guard compliance

Goal rate measures the task’s declared obligations, separately from guard compliance. A goal is met when a valid obligation signal reaches one. A rate with no decided obligations is undefined, rather than zero. All rows above use the same 55-task goal cohort; pairwise goal comparisons later in the report retain their separately stated task counts.

## Uncertainty and interpretation

The following intervals resample paired tasks 10,000 times while retaining all
five repetitions within each selected task. They describe task variation in
this selected suite, conditional on score availability, not independent model
training seeds. Intervals are percentile 95% intervals, without a correction
for the many exploratory comparisons in the accompanying JSON.

| Success comparison | Difference in percentage points | Paired 95% interval |
| --- | ---: | ---: |
| 24B minus 8B | +18.4 | +11.5 to +25.9 |
| 24B minus 2.6B | +3.9 | −1.3 to +9.5 |
| 24B minus Qwen4B | −1.3 | −4.9 to +2.3 |
| 24B minus Qwen9B | −1.6 | −7.2 to +3.6 |
| 8B minus 2.6B | −14.4 | −20.7 to −8.5 |

Absolute uncertainty is different: on the same 61 tasks, the absolute success
interval is **42.3–65.6% for 24B** and **25.9–45.9% for 8B**. The tighter paired
intervals above cannot be reused as absolute model confidence intervals.

For 24B versus 2.6B, partial credit improved by **8.0 points** (95% interval
+4.3 to +11.8). On 58 pairwise fully defined goal tasks, goal rate improved
by **13.5 points** (+7.6 to +19.3). Neither harm-count difference (−0.075,
interval −0.580 to +0.574) nor harm-free-rate difference (−0.3 points,
−5.6 to +4.9) establishes a safety improvement. All pairwise metrics and
their actual task cohorts are retained in the numerical results.

## Task categories

Success rates below use the same complete-task sensitivity rule. Domain
subsets are small and descriptive; they are not separate significance claims.

| Category | Fully scored tasks | 8B | 24B retry | 2.6B |
| --- | ---: | ---: | ---: | ---: |
| Simple | 30 | 69.3% | 95.3% | 94.7% |
| Finance | 6 | 10.0% | 20.0% | 10.0% |
| HR | 5 | 0.0% | 16.0% | 0.0% |
| Marketing | 5 | 0.0% | 0.0% | 16.0% |
| Operations | 8 | 0.0% | 22.5% | 0.0% |
| Sales | 2 of 3 | 10.0% | 20.0% | 30.0% |
| Support | 5 | 0.0% | 0.0% | 0.0% |

Across the 31 scored non-simple tasks, success was **2.6% for 8B** and
**13.5% for 24B**. The 24B aggregate therefore should not be interpreted as
broad competence on complex workflows: 30 of 61 scored tasks are simple.

## Coverage and operational qualification

### Missing scores stay missing

| Arm | Scored episodes | Timeout errors | Environment errors |
| --- | ---: | ---: | ---: |
| 8B | 307 of 310 | 0 | 3 |
| 24B original | 298 of 310 | 10 | 2 |
| 24B retry | 305 of 310 | 0 | 5 |
| 2.6B baseline | 305 of 310 | 0 observed | 5 TaskErrors |
| Qwen4B baseline | 306 of 310 | 0 observed | 4 TaskErrors |
| Qwen9B baseline | 310 of 310 | 0 observed | 0 |

The new runs' environment errors reject extra LinkedIn PostContent fields
`article_url`, `article_title` and `article_description` while validating
WorldState during task finalization. No environment pin or scorer was changed
to recover these scores. Missing observations were not zero-filled, silently
rescored, or pooled across retries. The native `max_turns` label on an errored
episode does not establish that turn exhaustion caused its missing score.

On the full 62-task population, assigning every missing success observation
first zero and then one gives **34.84–35.81% for 8B** and **52.90–54.52% for the
24B retry**. These are missing-data identification ranges, not statistical
confidence intervals. Strict complete-score 62-task headlines remain unavailable
for these models. Historical summaries that treated absent values as zero are
not equivalent to the missing-aware comparison here.

### The 24B retry removed observed timeouts

The original 24B arm admitted 64 concurrent episodes to a server with eight
generation slots. Logs repeatedly showed more than 50 waiting requests. Ten
episodes exceeded their unchanged 1800-second timeout; most had generated only
4.3k–7.1k tokens while spending almost the whole timeout in model requests.

The separate retry admitted eight concurrent episodes, retained eight server
slots, and recorded zero timeout errors. All original tasks, token budgets,
sampling, weights, source pins and timeout limits stayed unchanged. This
supports the capacity diagnosis, but is not a controlled throughput comparison
or proof that concurrency was the sole cause of changed outputs. The original
attempt is retained as an operating-point failure. Its complete-task summary
excludes several difficult tasks; its 56-task cohort must not be compared
directly with the retry's 61-task summary.

Native stop labels were: 8B, 307 `agent_completed` and three `max_turns`;
24B retry, 304 `agent_completed`, five `max_turns` and one `max_output_tokens`.
The retry's token-budget ending is retained as an observed outcome. Call finish
labels included one `length` for 8B, two for original 24B and none for the retry.
There were no observed ProviderError nodes. The native schema lacks an
authoritative model transport-attempt counter, so **retry count is unknown**,
not proven zero. Episode identities were unique, and assessment-ID deduplication
found no duplicate finding copies in these retained bundles.

### Native tool-use and channel checks

The 8B native template permits direct answers without a think prefill. Stock
qwen3 parsing initially misclassified a correct final answer as reasoning.
A project-local opt-in parser was qualified on direct and explicit-thinking
outputs, streaming and nonstream, including the real DelegatingParser path.
An intermediate scalar-successful attempt was rejected after artifact review
found think markup in content. Final 8B smoke passed without accepting reasoning
as the final answer. The 24B native template prefills think and retains qwen3.

The first 24B smoke chose `key_readiness` instead of `readiness`. That model
argument error remains a failed attempt. A diagnostic retry explicitly stated
the JSON argument and passed the unchanged argument/result/final-content gate.
This qualified transport and continuation, not benchmark capability.

## Recorded resource consumption

Summing usage from retained native model-call records gives:

| Arm | Recorded model calls | Completion tokens | Included reasoning tokens |
| --- | ---: | ---: | ---: |
| 8B | 1326 | 583,347 | 460,505 |
| 24B original | 1799 | 1,200,385 | 896,159 |
| 24B retry | 1855 | 1,279,172 | 965,881 |

Reasoning is included in completion tokens, not an additional token charge.
These are recorded-call consumption counts; unfinished timed-out requests can
make original-attempt consumption incomplete. They are not wall-time-normalized
throughput or monetary cost estimates. The jobs used the existing workstation;
no paid cloud capacity was enabled. Energy cost and cost per successful task
were not measured.

## Evaluation contract and limits

- Frozen environment: `automationbench-manifest-heldout62-eval-64k20t-t05-v1`,
  source commit `a0c7378b179e2b267382b8125192876502d82f89`.
- All six evaluated arms' retained ordered manifests match the fixture;
  selected-name SHA256 is
  `ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f`.
- New models use BF16, TP1, native-tokenizer renderer `lfm2.5-moe-tools@1`, native
  lfm2 tool parsing, eight server sequences and 65,536 context tokens. No
  quantization, speculative decoding, or shortened task/token budgets was used.
- Each arm: 62 tasks, five attempts, 20 turns, 8192 tokens per reply, 32,768
  episode output tokens. The new LFM arms use temperature 0.5, top-k 50,
  top-p 1.0 and repetition penalty 1.1, matching the dense LFM baseline.
- Qwen baselines retain their reported temperature 1.0, top-k 20, top-p 0.95,
  repetition penalty 1.0 and thinking enabled. Thus Qwen comparisons hold task
  and token budgets fixed, not sampling or native thinking behavior.
- Episode admission was 64 for 8B and original 24B, eight for the 24B retry,
  32 for the dense LFM baseline, and 64 for Qwen. Historical serving/runtime
  settings also differ, including batch invariance. Treat this as a comparison
  of model and serving policies, not a causal model-size or speed experiment.
- vLLM pin: `f09e4479123d348cee87d217c695a22f9b2daacc` (0.29.1.dev4).
  Verifiers pin: `bc70a7deaf64c8f8e0b41e39c00ac2d1ea1d7e0b`.
- [8B pinned weights](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/tree/5dd22602c2e9f6a097b1de4c4efe0658b605015c)
  and [24B pinned weights](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/tree/1769516ac588c4c558089c94f27d96b0dc67b183)
  have approximately 8.47B and 23.97B total parameters; active-parameter counts
  do not describe their full resident weight memory.

## Three priorities for the next decision

1. **Keep 24B as a candidate, without claiming it beats Qwen.** Its partial-credit
   gain over dense LFM is supported here; success gain and safety gains are
   uncertain. The current 8B checkpoint is a substantially weaker baseline on
   this suite. Base-model rankings do not predict OPD learning efficiency.
2. **Repair and version the environment scoring failure before final qualification.**
   Rerun all compared models under the repaired version when a complete 62-task
   headline is needed. Preserve the current frozen results as their own evidence.
3. **Match episode admission to measured serving capacity.** Retain error and
   latency evidence alongside capability. Evaluate intended complex workflows
   separately before selecting a training student or a deployment model.

## Evidence and reproduction

The [numerical audit](lfm-moe-heldout62-results-2026-10-16.json) contains task
scores, missingness, domain summaries, every pairwise interval and source paths.
The reducer is
`scripts/qualification/analyze_lfm_moe_heldout62.py`; pass each downloaded native
bundle with `--arm label=directory` and an `--output` path. Its fixture check
rejects changed task order, task populations, duplicate episodes and conflicting
assessment IDs. Source native episodes remain the replay authority.

Durable native bundles are retained in the `posttrain-lab` project on
[Trackio](https://trackio.carbonteq.com). The evaluation artifact logical name is
`evaluation/models/<model>@bf16/automationbench-manifest-heldout62-64k20t-t05/automationbench-manifest-heldout62-eval-64k20t-t05-v1`.

| Arm | Framework run | Evaluation artifact version and digest |
| --- | --- | --- |
| 8B | `lfm25-8b-a1b-heldout62-20261016-r1` | v0 `6884e023c4c8f5ef53d9d57a04b4d3a61d707bd0e618cf0ffab0604d361bf85b` |
| 24B original | `lfm25-24b-a2b-heldout62-20261016-r1` | v0 `9641f9be6f617953b3af1d75e3b575071fae891af75dc3cd113f1c0f96d2a1dc` |
| 24B retry | `lfm25-24b-a2b-heldout62-20261016-r2` | v1 `33e64d3e5d444f80dfcdfb3b35fb9b7a7669dafc1867e99c75ff82ee193603f9` |

Exact provider/image receipts, failed attempts, parser qualification and launch
commands are retained in [the execution plan](../plan/lfm-moe-heldout62-evaluation.md).
Ruff, Pyright, strict composition, nine import boundaries, targeted smoke tests
and whitespace checks passed. Three broad registry-test failures from nine
older unclassified work packages were pre-existing and remain unrelated debt.
