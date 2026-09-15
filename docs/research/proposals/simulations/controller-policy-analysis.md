# Controller policy simulation

This experiment asks whether the adaptive controller spends candidate generation more efficiently than shuffled sampling without replacement. It uses the real `AdaptiveCurriculumController` against a synthetic learner so the selection logic can be tested quickly and reproducibly. It does not simulate gradients or establish an LLM quality gain.

## Experiment

Each arm runs 20 model steps over 400 tasks in four classes. A step needs ten retained groups. Every task produces four binary outcomes, and the OLMo-style retention rule keeps groups with nonzero within-group variance. A refill requests only the missing groups, up to four rounds. Both policies avoid duplicate task identities inside a step.

The random arm walks a seeded shuffled inventory without replacement. The adaptive arm uses 20% cumulative task discovery, 20% cumulative class coverage, distinct-task class evidence, recent task evidence, and fresh selection after every refill. Forty paired seeds use identical hidden task populations and deterministic outcome streams.

The synthetic profiles isolate different behaviors:

| Profile | Hidden learner behavior |
| --- | --- |
| Normal | Classes and tasks start at different solve probabilities and improve at a moderate rate |
| Fast saturation | Fresh tasks usually produce contrast, then become easy quickly after useful practice |
| Rare geometry | Only two geometry tasks are useful; the other 98 are almost always solved |
| Regression | Arithmetic success drops after step 10 |
| Persistent noise | Probability tasks remain at 50% success and never learn |

## Results

Lower candidate groups per completed step means less generation for each successful optimizer update. Higher retained fraction means a larger share of generated groups reached the update. Unique tasks measures breadth, not efficiency.

| Profile | Policy | Candidate groups | Candidates / completed step | Retained fraction | Completed steps | Unique tasks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Normal | Random | 261.4 | 13.52 | 76.3% | 19.38 | 261.4 |
| Normal | Adaptive | **256.8** | **13.16** | **77.8%** | **19.55** | 185.0 |
| Fast saturation | Random | **230.3** | **11.56** | **86.9%** | **19.93** | 230.3 |
| Fast saturation | Adaptive | 244.5 | 12.38 | 81.7% | 19.78 | 173.7 |
| Rare geometry | Random | 343.3 | 25.93 | 56.2% | 13.63 | 343.3 |
| Rare geometry | Adaptive | **289.2** | **15.97** | **68.6%** | **18.23** | 206.6 |
| Regression | Random | 253.8 | 13.06 | 78.7% | 19.48 | 253.8 |
| Regression | Adaptive | **249.9** | **12.73** | **80.0%** | **19.65** | 179.9 |
| Persistent noise | Random | 252.8 | 12.83 | 79.0% | **19.73** | 252.8 |
| Persistent noise | Adaptive | **247.8** | **12.62** | **80.7%** | 19.65 | 176.6 |

Paired differences below are adaptive minus random. Intervals are two-sided 95% Student-t intervals over the 40 paired seeds. A candidate delta below zero favors adaptive sampling.

| Profile | Candidate-group delta | Candidates / completed-step delta | Retained-fraction delta | Completed-step delta |
| --- | ---: | ---: | ---: | ---: |
| Normal | -4.63 `[-7.65, -1.60]` | -0.36 `[-0.69, -0.03]` | +1.45 pp `[+0.56, +2.33]` | +0.18 `[-0.20, +0.55]` |
| Fast saturation | +14.25 `[+11.66, +16.84]` | +0.82 `[+0.63, +1.00]` | -5.12 pp `[-6.04, -4.20]` | -0.15 `[-0.32, +0.02]` |
| Rare geometry | -54.13 `[-57.78, -50.47]` | -9.96 `[-11.51, -8.42]` | +12.40 pp `[+11.58, +13.22]` | +4.60 `[+3.81, +5.39]` |
| Regression | -3.95 `[-6.67, -1.23]` | -0.33 `[-0.66, +0.00]` | +1.32 pp `[+0.38, +2.25]` | +0.18 `[-0.16, +0.51]` |
| Persistent noise | -5.10 `[-8.16, -2.04]` | -0.21 `[-0.48, +0.07]` | +1.61 pp `[+0.57, +2.65]` | -0.08 `[-0.34, +0.19]` |

## Interpretation

The adaptive controller improves candidate efficiency in four profiles, but the size differs sharply. The rare-geometry profile is the clearest success: it uses 15.8% fewer candidate groups, completes 4.6 more model steps, and directs only 9.6% of candidates to geometry after learning that most of that class is unproductive. Normal and regression profiles show small candidate savings. Their paired intervals exclude zero for raw candidate count, while completed-step differences remain uncertain.

Fast saturation exposes the main tradeoff. Fresh tasks are unusually valuable and practiced tasks lose value almost immediately, so shuffled sampling without replacement is close to the ideal policy. Adaptive sampling still preserves some familiar-task reuse and needs 6.2% more candidates. The task estimator now predicts mixed outcomes from recent success rates and observed variance, which reduces this loss, but cannot remove it while retaining an exploitation path.

Persistent noise shows why variance is an admission signal rather than proof of learning. Adaptive sampling gives the non-learning probability class 28.1% of candidates, above its 25% base share, because those tasks reliably produce mixed groups. Its retained fraction improves, but completed updates do not. A real qualification needs fixed-population progress evidence to identify this failure mode.

Adaptive sampling uses 25% to 40% fewer unique tasks because it deliberately reuses tasks with predicted contrast. That is efficient only when those predictions remain valid. The 20% discovery setting is therefore a floor: additional discovery competes with familiar practice using their predicted yields, and class coverage continues independently. The controller never repeated a task within a step in these runs.

## Data quality and limits

The raw dataset contains 400 rows: five profiles, two policies, and 40 seeds. The generator verifies unique `(profile, policy, seed)` keys, bounded candidate and completion counts, class shares summing to one, and zero duplicate fallbacks for this inventory. Re-running the script deterministically reproduces the files.

The random baseline is stronger than sampling with replacement because it traverses a shuffled inventory. Candidate groups have equal cost in this experiment. Outcomes within a group are independent Bernoulli draws, class transfer is synthetic, and the learner update is a probability shift rather than GRPO. These results qualify controller behavior and accounting; they do not qualify model learning, wall-clock savings, or task quality on AutomationBench.

Reproduce from the repository root:

```bash
uv run python docs/research/proposals/simulations/controller_policy_experiment.py
```

Artifacts:

- `controller_policy_runs.csv`: row-level data for analysis.
- `controller_policy_runs.jsonl`: the same rows in machine-readable event form.
- `controller_policy_results.json`: aggregate metrics and paired intervals.
