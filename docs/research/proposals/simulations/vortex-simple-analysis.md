# VORTEX yield-first soft routing: offline ablation

## Decision

A simpler VORTEX selector is promising **as a shadow candidate**, not yet a production replacement. Keep the current controller's estimate of usable-group probability, but let both the ordinary 80% lane and exploratory 20% lane draw from the full task inventory. The exploratory lane adds uncertainty from recent binary group-yield evidence. Do not add a separate variance-magnitude score or between-group stability penalty to task selection yet.

This differs from the earlier [magnitude-first prototype](./vortex-next-analysis.md), which underperformed the current controller. The new candidate improves offline admission efficiency in all six replayable source runs and both stationary and synthetic-drift scenarios. Stronger exploration restores most task coverage but spends some of that gain.

## Why both yield and reward spread are measured

The retained VORTEX v3 selection sets `algorithm: olmo3` and `advantage_scaling: none` in `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`. The TRL fork's advantage path then uses centered group rewards without dividing by group standard deviation. Thus a nonzero-std group is necessary for a reward-based group-relative update, **and raw reward spread can affect its coefficient**. This differs from original, group-normalized GRPO, where a common scaling of all rewards in one group largely cancels. The run's admission rule remains binary (`group_std > 0`).

The [numeric source](./vortex_next_groups.json) has 2,072 complete groups, 1,296 with nonzero variance. The smallest positive observed std is about 0.00154, far above the fork's `1e-4` numerical denominator epsilon; a tiny-gap cutoff has little leverage in this snapshot. We therefore report two distinct offline proxies: candidate groups per admitted group, and the sum of within-group reward standard deviations per million observed output tokens. With unscaled group advantages, std is the root-mean-square centered reward coefficient; variance is its square. The per-token std ratio is only a heuristic for coefficient magnitude relative to rollout cost, **not** a measured gradient norm or learning gain. Sampling distribution, sequence lengths, clipping, importance weights, truncation, and reward noise also affect training.

## Paired replay results

The [yield-first simulator](./vortex_simple.py) reuses the exact [numeric snapshot](./vortex_next_groups.json) and deterministic task-visit draws from the earlier replay. It runs 100 steps, eight required useful groups, at most four rounds, ten paired seeds, and six source runs separately; the seventh source had only 11 observed tasks and is excluded. The [complete base results](./vortex_simple_results.json) and [exploration sensitivity](./vortex_simple_sensitivity_results.json) retain every seed, source, scenario, and step, with source/code hashes.

Each figure below is the unweighted mean of six per-source means. The last column is computed per replay row as `1,000,000 × std_sum / output_tokens`, then averaged across paired seeds and sources; it is not the std of all rewards pooled across runs. Lower candidates per retained group is better; higher covered-task count and centered-std-per-token are desirable, but neither alone establishes learning.

| Scenario | Selector | Completed / 100 | Candidates / retained | Covered tasks | Sum of group std / million output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Stationary | Current controller | 90.72 | 1.471 | 158.3 | 5.384 |
| Stationary | Binary-yield two-lane | 88.73 | 1.498 | 153.0 | 4.983 |
| Stationary | Yield-first, no exploration bonus | **94.47** | **1.368** | 142.2 | **5.626** |
| Stationary | Yield-first, bonus 0.2 | 93.43 | 1.383 | 146.2 | 5.596 |
| Stationary | Yield-first, bonus 1 | 92.85 | 1.406 | 148.5 | 5.512 |
| Stationary | Yield-first, bonus 4 | 92.32 | 1.430 | 151.7 | 5.441 |
| Stationary | Production yield-first, bonus 4 | 92.72 | 1.428 | 151.1 | 5.436 |
| Synthetic saturation/recovery | Current controller | 75.72 | 1.731 | 158.3 | 4.774 |
| Synthetic saturation/recovery | Binary-yield two-lane | 67.03 | 1.902 | 156.3 | 4.114 |
| Synthetic saturation/recovery | Yield-first, no exploration bonus | **82.82** | **1.590** | 152.0 | **5.056** |
| Synthetic saturation/recovery | Yield-first, bonus 0.2 | 81.88 | 1.604 | 152.3 | 5.004 |
| Synthetic saturation/recovery | Yield-first, bonus 1 | 80.38 | 1.644 | 154.3 | 4.906 |
| Synthetic saturation/recovery | Yield-first, bonus 4 | 78.77 | 1.671 | 155.6 | 4.814 |
| Synthetic saturation/recovery | Production yield-first, bonus 4 | 78.97 | 1.664 | 155.4 | 4.845 |

The low-bonus soft candidate uses **6.0% fewer candidates per admitted group** than the current controller when stationary and **7.3% fewer** under the synthetic drift. This improvement appears in all 12 source/scenario pairs. But its mean coverage drops from 158.3 to 146.2 and 152.3 tasks, respectively. Raising the exploratory bonus to 4 restores mean coverage to 151.7 and 155.6 tasks; candidate efficiency remains better than the current controller, while the std-per-token proxy becomes approximately equal. This is a Pareto tradeoff, not a single winner. The bonus values are sensitivity probes, not tuned production defaults.

The current controller's score itself is not “pure binary frequency”: it combines a reward-mean-based predicted mixed-group probability with observed nonzero-variance yield. The no-bonus and soft-bonus variants keep that estimator. A separate binary-only ablation performs worse, so this replay does **not** support discarding the existing prediction layer.

The production `yield_first` implementation in `packages/train/src/posttrain/train/adaptive_curriculum.py` was replayed separately from the research sampler at the coverage-preserving bonus 4. It used fewer candidates per retained group than the current controller in all 12 source/scenario pairs: 1.428 versus 1.471 stationary and 1.664 versus 1.731 under synthetic drift. It closely reproduces the research bonus-4 tradeoff while preserving 151.1 and 155.4 covered tasks on average. This is a useful implementation check, not an online result; the two samplers have different RNG streams and therefore need not produce identical rows.

## Duplicate-task contract

One task may be sampled at most once within an optimizer step, including all refill rounds. Reuse in later optimizer steps remains necessary for rechecks. The candidate removes selected IDs before every draw and stops if there are too few distinct eligible tasks to fill a requested batch. A [worst-case four-round test](./tests/test_vortex_simple.py) with 40 tasks and all-zero rewards checks 32 distinct picks per step for the current baseline and each new variant; a ten-task exhaustion test confirms the soft candidate reports a shortfall instead of repeating a task. Both pass. The six replayed inventories contain at least 152 tasks, while at most 32 distinct candidates are requested per simulated step. This establishes no duplicates in the tested replay, **not** a global production guarantee: the current production controller has an explicit duplicate fallback when its inventory is exhausted. A production VORTEX change must replace that fallback for this policy with a checked shortfall/pre-submission capacity rule and test the actual job path.

## Limits and next gate

Historical trajectories were chosen by different samplers. The simulator estimates unobserved task visits using each task's retained groups plus class-peer shrinkage; synthetic saturation/recovery is an assumption. Six runs differ in environment and training settings, so the cross-source means are sensitivity summaries, not a controlled treatment effect. The actual VORTEX v3 job requests ten groups and allows ten candidate rounds rather than the simulator's eight groups and four rounds; training sees sequence-length, truncation, and optimizer effects absent here. No offline result certifies policy improvement or GPU-hour savings.

Before a live candidate, choose the exploration bonus against a declared coverage requirement, not post-hoc on this replay; enforce no same-step duplicates at job compilation/runtime; and shadow-score live candidate batches without changing training. A same-checkpoint A/B should then compare retained-group yield, raw within-group reward std, token/GPU cost, class/task coverage, actual gradient statistics, and held-out improvement. Do not ship solely because the offline admission metric improved.

## Reproduce

From `/home/hammad/projects/rl`:

    uv run python docs/research/proposals/simulations/vortex_simple.py --input docs/research/proposals/simulations/vortex_next_groups.json --output docs/research/proposals/simulations/vortex_simple_results.json --seeds 10
    uv run python docs/research/proposals/simulations/vortex_simple_sensitivity.py --input docs/research/proposals/simulations/vortex_next_groups.json --output docs/research/proposals/simulations/vortex_simple_sensitivity_results.json --seeds 10
    uv run pytest -q docs/research/proposals/simulations/tests
