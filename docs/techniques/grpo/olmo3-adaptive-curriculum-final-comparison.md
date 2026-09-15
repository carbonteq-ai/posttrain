# Adaptive selection reduces OLMo 3 generation cost while preserving task coverage

**Completed 20-step comparison · AutomationBench · 12 September 2026**

Two OLMo 3 GRPO runs used the same LFM2.5-2.6B model, LoRA settings,
AutomationBench population, four attempts per task group, eight retained groups
per optimizer step, and RTX PRO 6000 execution target. The control used fixed
random task sampling. The adaptive arm used the curriculum controller for the
initial batch and every OLMo refill round.

Both runs completed 20 optimizer steps and retained 160 task groups. The
adaptive arm reached that target with 94 fewer candidate groups and finished
about 1.66 hours earlier.

## Completed-run results

| Metric | Random sampling | Adaptive sampling | Adaptive difference |
|---|---:|---:|---:|
| Optimizer steps | 20 | 20 | Same |
| Retained task groups | 160 | 160 | Same training budget |
| Candidate task groups | 382 | **288** | **24.6% fewer** |
| Candidate rollouts | 1,528 | **1,152** | **376 fewer** |
| Generated completion tokens | 5.65M | **4.32M** | **23.6% fewer** |
| Aggregate candidate retention | 41.9% | **55.6%** | **+13.7 points** |
| Refill rounds | 111 | **77** | **30.6% fewer** |
| Mean optimizer-step time | 21.17 min | **16.36 min** | **22.7% lower** |
| Sum of optimizer-step time | 7.06 h | **5.45 h** | **1.60 h lower** |
| Run wall time | 7.21 h | **5.55 h** | **1.66 h lower** |
| Rollout time | 6.23 h | **4.71 h** | **1.52 h lower** |
| Weighted rollout throughput | 436.7 tok/s | **459.5 tok/s** | **5.2% higher** |
| Candidate zero-variance groups | 51.5% | **37.1%** | **−14.5 points** |
| Optimizer-step training reward | **0.345** | 0.321 | −6.8% |
| Mean reward over every candidate rollout | 0.343 | 0.342 | Essentially equal |
| Completion truncation over all candidates | 38.9% | **38.0%** | −0.9 points |
| Failed or unscorable rollouts | 0 | 0 | Same |

The 1.29× optimizer-step throughput comes mostly from avoiding candidates that
are unlikely to survive OLMo's active-sampling filter. Raw decoding throughput
improved only 5.2%; the larger gain came from doing less generation work.

## GRPO signal quality

| Metric | Random sampling | Adaptive sampling | Interpretation |
|---|---:|---:|---|
| Mean training reward | **0.345** | 0.321 | 6.8% lower, but sampled populations differ |
| Reward standard deviation | 0.266 | 0.264 | Overall reward spread was effectively unchanged |
| Mean within-group reward spread | 0.110 | **0.123** | 11.3% more separation within task groups |
| Zero-variation groups | 51.5% | **37.1%** | 14.5 points fewer groups with no relative signal |
| Advantage standard deviation | 0.124 | **0.133** | 7.1% more separation between attempts |
| Mean absolute advantage | 0.087 | **0.094** | 8.7% larger usable update signal |
| Zero-advantage rows | 52.0% | **37.7%** | 14.2 points fewer rows with no update signal |
| PPO clipping region | 0.875% | **0.785%** | Both low; adaptive hit the boundary slightly less |
| Importance ratio mean | 0.9904 | 0.9905 | Effectively identical and close to 1.0 |
| Initial policy-parity delta | **0.0119** | 0.0134 | Both safely below the 0.05 admission limit |
| Completion truncation | 38.9% | **38.0%** | Effectively unchanged |
| Mean completion length | 3,699 tokens | 3,746 tokens | Adaptive candidates averaged 1.3% longer |

The completed comparison is less dramatic than the earlier 14-step interim
run, but it supports the same mechanism. Adaptive selection did not increase
global reward dispersion; it improved the contrast among attempts for the same
task. That is the quantity GRPO can turn into relative advantages. The lower
zero-advantage fraction is therefore more informative for sampling efficiency
than the raw training-reward difference.

## The controller became more efficient as evidence accumulated

| Metric | Adaptive steps 1–10 | Adaptive steps 11–20 | Random steps 1–10 | Random steps 11–20 |
|---|---:|---:|---:|---:|
| Candidate groups | 153 | **135** | 186 | 196 |
| Aggregate retention | 52.3% | **59.3%** | 43.0% | 40.8% |
| Zero-variance groups | 44.2% | **29.9%** | 49.4% | 53.7% |
| Mean step time | 18.04 min | **14.68 min** | 21.22 min | 21.12 min |
| Training reward | 0.275 | **0.368** | 0.345 | 0.345 |

The adaptive second half used 11.8% fewer candidates, retained 7.0 percentage
points more of them, and cut zero-variance groups by 14.4 points. Mean step time
fell 18.6%. Random sampling showed no comparable improvement: it generated more
candidates in the second half and its zero-variance rate increased.

This is the strongest evidence in the run that the controller was responding to
observed task outcomes rather than receiving a one-time favorable sample.

## Task exploration and reuse

| Metric | Random sampling | Adaptive sampling |
|---|---:|---:|
| Unique tasks observed | 152 / 160 | **160 / 160** |
| Population coverage | 95.0% | **100.0%** |
| Candidate groups per unique task | 2.51 | **1.80** |
| Tasks observed exactly once | 32 | **81** |
| Tasks observed more than once | 120 | **79** |
| Largest reuse count for one task | **6 groups** | 7 groups |
| Share received by the ten most sampled tasks | **12.8%** | 18.1% |
| Within-step duplicate fallbacks | Not emitted by fixed sampler | **0** |

The adaptive policy achieved wider discovery with less repetition overall. It
also concentrated more work on a small set of promising tasks: its ten most
sampled tasks received 18.1% of candidate groups. This combination is desirable
for the stated policy—reserve enough work to discover the population, then reuse
tasks that continue to produce useful contrast.

The controller reached all 160 task identities during step 12. After that point,
`new_tasks` correctly became zero and the sampler continued with evidence-based
practice and rechecks. Every adaptive step reported one unique identity per
candidate group, so OLMo refill rounds did not duplicate a task within an
optimizer step.

## Class allocation

Candidate groups were distributed as follows:

| Class | Random sampling | Adaptive sampling |
|---|---:|---:|
| Simple | 97 | 52 |
| Finance | 47 | 60 |
| Marketing | 45 | 46 |
| Sales | 45 | 40 |
| Operations | 47 | 38 |
| Support | 45 | 26 |
| HR | 56 | 26 |

These counts describe generated candidates, not retained training groups or
held-out class performance. They show that the controller moved work away from
the fixed mixture—especially Simple, HR, and Support—and toward Finance. Whether
that shift improves class-level generalization requires a common evaluation
population.

## Optimizer health

| Metric | Random sampling | Adaptive sampling | Interpretation |
|---|---:|---:|---|
| Mean reward standard deviation | 0.266 | 0.264 | Similar retained-group spread |
| Mean entropy | 0.173 | 0.174 | Similar policy uncertainty |
| Mean gradient norm | 0.00431 | 0.00411 | Similar update scale |
| Mean clip fraction | 0.875% | **0.785%** | Both low |
| Mean importance ratio | 0.9904 | 0.9905 | Both close to 1.0 |
| Mean sampling log-probability delta | 0.0373 | 0.0383 | Similar |

Both runs completed without failed or unscorable rollouts. Their entropy,
gradient scale, clipping, and importance ratios were close. The adaptive speedup
therefore did not coincide with an obvious optimizer instability in these
metrics.

The maximum per-row sampling log-probability delta was large in both arms
(10.37 random and 10.45 adaptive), while the mean remained near 0.038. This
suggests rare outliers rather than broad policy/rollout divergence, but the
outlier distribution should remain visible in Observatory.

## What can and cannot be concluded

The completed runs support three conclusions:

1. **Adaptive selection was more compute-efficient.** It delivered the same 160
   retained groups with 24.6% fewer candidate groups and 22.7% less step time.
2. **The gain was consistent with self-correction.** Sampling yield, signal
   density, and step time improved after the controller accumulated evidence.
3. **Efficiency did not come from suppressing exploration.** The adaptive run
   covered all 160 tasks and avoided within-step duplicates.

The runs do not establish that the final adaptive checkpoint is a better model.
Training reward is measured on different sampler-selected populations, and the
adaptive optimizer-step reward was 6.8% lower even though candidate-level mean
reward was almost identical. A fixed, held-out AutomationBench evaluation of
both final checkpoints is required for a model-quality claim.

This is also one run per arm. Repeated seeds are needed to estimate uncertainty
in the 24.6% candidate reduction and the 22.7% time reduction. Completion
truncation remained high at roughly 38% in both arms, so a later experiment
should separate selection gains from effects caused by the completion limit.

## Run provenance

| Arm | Run ID | State |
|---|---|---|
| Random OLMo 3 | `lfm26-olmo3-random20-20260912-r3` | Succeeded, 20/20 |
| Adaptive OLMo 3 | `lfm26-olmo3-adaptive20-discovery-v4-20260912-r1` | Succeeded, 20/20 |

Aggregate candidate retention is total retained groups divided by total
candidate groups. Candidate task groups are obtained by grouping four rollout
traces with the same task identity. First-half and second-half comparisons use
optimizer steps 1–10 and 11–20 respectively.
