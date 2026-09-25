# VORTEX next: paired offline simulation

## Verdict

The two-lane *routing contract* behaved as intended, but the proposed variation-magnitude score did **not** reduce waste relative to the existing controller. Do not replace production selection with this prototype. Preserve its full-pool, soft exploration concept for a second candidate, and use the current controller as the efficiency floor.

This is an offline selector test, not a model-training result. The [prototype](./vortex_next.py) and [numeric source](./vortex_next_groups.json) produce the [complete result](./vortex_next_results.json), including every seed, policy, source run, scenario, and step. The result records SHA-256 digests of its source, prototype, and current-controller code.

## Setup and source support

Seven retained AutomationBench runs contributed **2,072** complete four-rollout groups with numeric rewards. Six runs had at least 152 distinct observed tasks and were replayed separately; the older adaptive20 run saw only 11 tasks and was excluded from counterfactual selection. The replay uses 100 steps, eight useful groups per step, up to four rounds, five selectors, two scenarios, and ten paired seeds per source. A useful group here means **any** nonzero within-group reward variance, exactly matching the present admission signal. The stress scenario synthetically turns one task cohort into constant maximum reward after step 40 and lets a different flat-low cohort regain variance after step 60.

Each row below is the unweighted mean of six per-source means; runs are *not* pooled as if they were one policy or environment. Lower candidate groups per retained group is better. Higher completed steps is better.

| Scenario | Selector | Completed / 100 steps | Candidates / retained | Zero-variance candidates | Covered tasks |
| --- | --- | ---: | ---: | ---: | ---: |
| Stationary | Current controller | **90.72** | **1.471** | **31.20%** | **158.3** |
| Stationary | Binary two-lane | 88.73 | 1.498 | 32.03% | 153.0 |
| Stationary | Magnitude two-lane | 87.20 | 1.532 | 33.60% | 156.3 |
| Stationary | Proposed next | 86.77 | 1.536 | 33.61% | 156.3 |
| Stationary | Class-balanced uniform | 75.77 | 1.737 | 41.31% | 158.0 |
| Saturation/recovery | Current controller | **75.72** | **1.731** | **41.69%** | **158.3** |
| Saturation/recovery | Binary two-lane | 67.03 | 1.902 | 46.61% | 156.3 |
| Saturation/recovery | Magnitude two-lane | 65.00 | 1.937 | 47.65% | 157.6 |
| Saturation/recovery | Proposed next | 62.92 | 1.965 | 48.30% | 157.6 |
| Saturation/recovery | Class-balanced uniform | 46.72 | 2.332 | 56.49% | 158.2 |

The proposed next selector needed **4.4% more candidates per retained group** than the current controller in the stationary scenario and **13.5% more** in the synthetic saturation/recovery scenario. It completed 3.95 and 12.80 fewer steps per 100, respectively. This is not an acceptable waste-reduction tradeoff. The binary two-lane ablation was closer to the current controller than the magnitude and stability variants, but still lost, especially under drift.

The exploration lane was used for 20.06% of selections in the stationary replay. It selected a previously unseen task 16.08% of the time versus 12.23% in the normal lane; after step 50, new-task selections across both lanes fell to 1.47%. Thus it provides a *bias*, not a hard novelty quota, and naturally moves to familiar tasks as the inventory becomes known. This verifies the requested routing behavior but not the score's quality.

## Interpretation and next candidate

The current controller estimates both the chance of a mixed-result group from reward means and the empirical probability of a nonzero-variance group, with class-level priors. The prototype instead ranks a bounded *magnitude* of observed variance, smoothed toward an optimistic prior, and discounts inconsistent groups. The replay's admission rule is binary, so these are different objectives. That mismatch is a plausible reason the magnitude versions rank worse; the ablations support it, but do not isolate prior strength, scale, stability penalty, or aging as individual causes. The additional stability penalty did not recover the gap in these data.

**Training-scale correction:** the retained VORTEX v3 training selection in `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` sets `algorithm: olmo3` and `advantage_scaling: none`. In our TRL fork, that means centered group rewards are *not* divided by group standard deviation. Unlike classic group-normalized GRPO, reward-gap magnitude can therefore change the policy-gradient coefficient. The binary admission replay above measures how quickly groups fill a batch, **not** their expected update strength. We must inspect both admitted-group frequency and raw centered-reward variance per rollout cost before claiming lower learning waste. Neither quantity alone certifies gradient quality or held-out improvement.

The next candidate should retain the current controller's calibrated estimate of **usable-group probability** as the primary normal-lane score while removing the hard unseen/familiar pool split. Add uncertainty or staleness only in the soft exploratory lane. Measure raw centered-reward variance and repeated-group behavior beside admission efficiency; do not silently treat either binary yield or raw variance as a complete learning objective. This preserves the elegant full-pool behavior without discarding the strongest observed efficiency signal. Before any production edit, replay that candidate against this same baseline, then qualify with a same-checkpoint live A/B that measures retained-group quality, actual GPU-hours, class/task coverage, and held-out gain. No offline replay can certify those training outcomes.

## Limitations

Historical samplers selected which task outcomes exist. The simulator fills gaps with empirical task groups and class-peer shrinkage; paired seeds share deterministic draws for each task visit, but this cannot recreate counterfactual rollouts or policy-dependent reward changes. The synthetic drift is deliberately assumed, not observed. Different source runs used different environment revisions and training settings, so cross-run averages are descriptive sensitivity checks, not causal treatment effects. The numeric snapshot excludes prompts, transcripts, tool payloads, incomplete/error groups, and the trainer's truncation policy; those require a separate live qualification.

## Reproduce

From `/home/hammad/projects/rl`:

    uv run python docs/research/proposals/simulations/vortex_next.py --input docs/research/proposals/simulations/vortex_next_groups.json --output docs/research/proposals/simulations/vortex_next_results.json --seeds 10
    uv run pytest -q docs/research/proposals/simulations/tests
