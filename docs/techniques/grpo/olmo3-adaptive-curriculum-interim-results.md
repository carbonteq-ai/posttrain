# Adaptive task selection produces denser GRPO signal with fewer rollouts

**Interim matched comparison · 14 optimizer steps · 12 September 2026**

An adaptive curriculum was added to OLMo 3 active sampling for an LFM2.5-2.6B
AutomationBench training run. Both arms use the same model, OLMo 3 update rule,
LoRA training configuration, task population, four attempts per task group, and
RTX PRO 6000 GPU. The only experimental difference is how task identities are
chosen:

- **Control:** sample from the fixed shuffled task mixture.
- **Adaptive:** start with equal category shares, observe reward variation, and
  choose every refill from the controller's latest task and category evidence.

The control completed all 20 planned steps. This interim comparison uses the
first 14 steps from each arm because the adaptive run had completed 14 steps at
the evidence cutoff.

## Results at 14 matched steps

| Metric | Fixed-mixture control | Adaptive curriculum | Difference |
|---|---:|---:|---:|
| Mean training reward | 0.326 | **0.420** | +28.9% |
| Reward standard deviation | 0.276 | **0.437** | +58.1% |
| Mean within-group reward variation | 0.103 | **0.341** | 3.30× |
| Groups with zero reward variation | 50.9% | **15.5%** | −35.4 percentage points |
| Advantage standard deviation | 0.122 | **0.331** | 2.71× |
| Mean absolute advantage | 0.080 | **0.273** | 3.43× |
| Zero-advantage rows | 51.5% | **15.5%** | −36.0 percentage points |
| Entropy | 0.172 | **0.244** | +42.1% |
| Gradient norm | 0.0046 | 0.0117 | 2.55× |
| Policy clipping region | 0.84% | **0.46%** | −45.0% |
| Importance-sampling ratio, mean | 0.990 | 0.987 | Both close to 1.0 |
| Importance-ratio clamped rows | 0.160% | 0.157% | Essentially unchanged |
| Initial policy-parity log-probability delta | 0.0119 | 0.0134 | Both below the 0.05 limit |
| Completion truncation | **41.2%** | 43.3% | +2.0 percentage points |
| Mean completion length | 3,869 tokens | 4,844 tokens | +25.2% |
| Generated rows per optimizer step | 76.3 | **40.6** | −46.8% |
| Useful-row retention rate | 45.2% | **81.0%** | +35.7 percentage points |
| Active-sampling rounds per step | 5.21 | **2.14** | −58.9% |
| Mean optimizer-step time | 21.0 minutes | **13.3 minutes** | −36.5% |
| Cumulative generated tokens | 7.28M | **3.97M** | −45.5% |

Bold values indicate the more favorable direction for the sampling objective.
Gradient norm and entropy are descriptive rather than scores with a universally
better direction.

## What the results show

GRPO learns from differences among fresh attempts at the same task. A group in
which every attempt receives the same reward produces zero relative advantage
and contributes little or no policy-gradient signal.

The adaptive controller reduced zero-variation groups from 50.9% to 15.5%.
Consequently, mean absolute advantage was 3.43 times larger while the run used
46.8% fewer generated rows. The adaptive arm therefore obtained a denser
training signal from substantially less generation work.

This translated into lower collection cost and faster training. By step 14,
the adaptive arm had generated 3.97M tokens, compared with 7.28M for the
control, even though its retained completions were 25.2% longer on average.
Mean step time fell from 21.0 to 13.3 minutes.

The optimizer indicators remain healthy so far. The mean importance-sampling
ratio stayed near 1.0, the clamped fraction remained below 0.2%, and the initial
policy-parity checks passed in both arms. The adaptive run produced larger
advantages and gradient norms without increasing the fraction of updates that
hit the policy-clipping boundary.

## How to interpret the higher reward

The adaptive arm's mean training reward was 0.420, compared with 0.326 for the
control. This is encouraging, but it is not yet evidence that the trained model
is 28.9% better. Training reward is measured on the tasks chosen by each
sampler, and the adaptive controller deliberately changes that population.

The reward result supports a narrower conclusion: the controller found tasks
that produced both useful outcome differences and competitive rewards. A
matched held-out evaluation after training is required to determine whether
the denser signal improves general performance.

## Stability across the adaptive run

The observed benefit did not come from one unusually favorable step:

| Adaptive metric | Steps 1–7 | Steps 8–14 |
|---|---:|---:|
| Mean training reward | 0.426 | 0.414 |
| Mean within-group variation | 0.350 | 0.331 |
| Zero-variation groups | 15.9% | 15.2% |
| Useful-row retention rate | 79.5% | 82.5% |
| Generated rows per step | 41.7 | 39.4 |

Signal density and sampling efficiency remained broadly stable over the two
halves of the available adaptive run.

## Scope and limitations

- This is one qualification run per arm, not a replicated experiment.
- The comparison establishes sampling efficiency and training-signal density;
  final model quality still requires held-out evaluation.
- The adaptive arm was still running at the 14-step evidence cutoff. The final
  20-step comparison may change the aggregate values.
- Completion truncation was 2.0 percentage points higher for the adaptive arm
  and should be considered when interpreting longer sampled tasks.
- Two non-terminal `airtable_record_exists` verifier assertions appeared in the
  adaptive logs. Training continued, but those examples require a separate
  verifier investigation.

## Run provenance

| Arm | Run ID | Provider job | State at cutoff |
|---|---|---|---|
| Fixed-mixture OLMo 3 | `lfm26-olmo3-random20-20260912-r3` | `pt-93868e368643954598c3b43e` | Succeeded, 20/20 steps |
| Adaptive-curriculum OLMo 3 | `lfm26-olmo3-adaptive20-20260912-r4` | `pt-66c1c609dfa568471a62b4b9` | Running, 14/20 steps |

The matched comparison uses the first 14 emitted optimizer-step metric records
from each provider log. Values in the main table are arithmetic means across
those records unless labeled cumulative or initial.
