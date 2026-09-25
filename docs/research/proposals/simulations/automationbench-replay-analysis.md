# AutomationBench curriculum replay: seven-run empirical approximation

This is a research simulation, not a production selector change or a measured RL improvement. The [compact source snapshot](./automationbench_groups.json), [replay code](./automationbench_replay.py), [main results](./automationbench_replay_results.json), and [no-truncation sensitivity](./automationbench_replay_no_truncation_results.json) make every number below reproducible. The export began on 2026-09-23 at 12:10:27 UTC. The latest VORTEX run was still changing near collection time; rerun the export before using it as a release gate.

## Objective: learning preserved, waste removed

The objective is **more learning per GPU-hour**, achieved by removing rollout work that cannot contribute to a sound optimizer update. We should not optimize novelty share, reuse share, reward variance, or short trajectories as ends in themselves. A useful group here means only that its four rewards vary and can produce nonconstant group-relative advantages; it is a *waste-screening proxy*, not a measure of learning value. A zero-variance group consumes rollout compute without this learning signal, while a high-variance group can still be noisy, biased, or redundant.

The data support testing a *same-unit next-group usefulness predictor* as an admission-efficiency mechanism, not hard-coding a 20% discovery quota. On both VORTEX runs, previously seen tasks yielded useful reward variation more often than new tasks in the observed selections. The proposed predictor starts all tasks from an equal Beta prior; class evidence updates the prediction for unseen tasks, task evidence updates seen tasks with class shrinkage, and posterior sampling occasionally rechecks uncertain tasks. These are probabilities of a **useful four-rollout group**, so new and familiar candidates can be compared directly. None of these observations proves that more reuse improves training reward or generalizes to unobserved tasks.

## Source and quality profile

Seven retained LFM/AutomationBench training runs contributed 8,128 rollout traces. All required metadata fields were present. They formed 2,035 observed groups; 2,028 contained exactly four rollouts and were eligible for fitting. The seven incomplete groups were all in the newest VORTEX v3 snapshot and were excluded, not scored as failed groups. No exported record contains prompts, tool outputs, transcripts, or initial task state. The source snapshot records task ID, class, optimizer step, group ID, useful/non-useful label, tokens, and truncation/error counts only.

| Source run (short name) | Complete groups | Distinct observed tasks | Useful groups | New-group share | New useful | Previously seen useful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random20 (Sep 12) | 382 | 152 | 41.9% | 39.8% | 40.8% | 42.6% |
| adaptive20 (Sep 12) | 199 | 11 | 80.4% | 8.5% | 64.7% | 81.9% |
| adaptive discovery v4 (Sep 12) | 288 | 160 | 55.6% | 55.6% | 42.5% | 71.9% |
| adaptive oversample (Sep 15) | 284 | 160 | 70.4% | 56.3% | 61.9% | 81.5% |
| adaptive GRPO 50 (Sep 20) | 400 | 160 | 71.5% | 40.0% | 57.5% | 80.8% |
| VORTEX experiment v2 (Sep 23) | 309 | 160 | 64.7% | 51.8% | 53.1% | 77.2% |
| VORTEX experiment v3 (Sep 23) | 166 | 127 | 68.1% | 76.5% | 62.2% | 87.2% |

“Previously seen” means seen in an *earlier optimizer step*, not merely earlier in the same step. The older adaptive20 run observed only 11 distinct tasks, so it is useful for calibration but **not** a credible 100-step counterfactual task-inventory replay. Its high observed yield is heavily selected; it must not be compared with random20 as if it were a randomized treatment effect.

Truncation remains material: 277 of 1,236 VORTEX v2 rollouts and 215 of 680 v3 rollouts were marked truncated in this snapshot. Excluding every group containing a truncated rollout leaves 176 v2 and 74 v3 complete groups. This narrows support but does not erase the observed seen-task usefulness gap: v2 is 55.1% new versus 71.8% seen; v3 is 56.1% new versus 94.1% seen. These are still policy-selected populations, not causal estimates.

## Predictive check before simulation

The chronological calibration scores every group using only earlier optimizer steps, then updates the class and task histories. On VORTEX v2, the hierarchical predictor's Brier score is **0.163**, versus **0.231** for a global-prior predictor; on v3 it is **0.183** versus **0.222**. Lower is better. A stricter cross-run check fits on v2 and predicts v3 without seeing any v3 outcomes: **0.136** Brier versus **0.219** for the v2 global rate, across 166 v3 groups. Reversing the direction gives **0.144** versus **0.229**, with source task identity available for 264 of 309 v2 groups. The transfer check supports a useful signal, but shared tasks and related models mean it is not independent future-run validation.

## 100-step waste-screening replay, not a training outcome

For each source run separately, the simulator draws empirical outcomes with replacement for observed tasks, shrunk toward same-class peers when task support is sparse. It compares uniform selection, posterior-mean ranking, and posterior sampling over 100 synthetic steps, eight useful groups per step, at most four rounds, 20 paired random seeds, and no same-step duplicate task. The model does **not** learn from the simulated steps. The following numbers are mean candidate groups per completed step, conditional on the source run's observed task inventory:

| Source | Uniform | Posterior mean | Posterior sampling |
| --- | ---: | ---: | ---: |
| random20 | 43.94 | 11.05 | 12.06 |
| adaptive discovery v4 | 27.58 | 10.57 | 10.83 |
| adaptive oversample | 14.65 | 8.48 | 8.53 |
| adaptive GRPO 50 | 15.66 | 8.74 | 8.66 |
| VORTEX v2 | 19.09 | 8.96 | 9.23 |
| VORTEX v3 | 13.62 | 9.27 | 9.26 |

The 11-task adaptive20 inventory is intentionally omitted: it cannot supply a full four-round set of distinct tasks. On the no-truncation VORTEX sensitivity, the same uniform/posterior-sampling comparison is **20.16/8.20** for v2 and **16.15/8.57** for v3. This is robust within the replay model, not proof of real compute savings. Output-token and input-token totals, paired-seed details, and all cross-run comparisons are in the JSON results.

Cost ranking exposed a failure mode. The naive `predicted useful probability / predicted total tokens` policy chased cheap, low-yield groups: it completed only **5.3/100** v2 and **26.3/100** v3 simulated steps, versus **99.9** and **99.8** for probability-only posterior sampling. Useful groups in those two source runs averaged roughly 281K/307K total tokens, while non-useful groups averaged 159K/201K; low token count can signal a failed or prematurely ended trajectory. A *guarded* cost policy first requires enough predicted group usefulness to make step completion plausible, then ranks feasible candidates by useful probability per predicted token. It completed **98.0/100** v2 and **97.6/100** v3 steps, with about **2.05M** and **2.34M** total tokens per completed step versus **2.95M** and **2.92M** for probability-only posterior sampling. This is a possible token-work tradeoff, not a GPU-time win. The guard assumes independent groups for a 99% binomial completion target; correlated tasks and posterior error explain why realized completion fell short of 99%. The no-truncation sensitivity likewise completed 98.0/97.8 guarded steps versus 99.9/99.9 probability-only steps. It needs a calibrated risk model before production use.

The useful next design is **not** to maximize this predictor. Keep the learning objective and update semantics fixed, then use predictions to avoid waste in the candidate/refill path: preferentially avoid groups unlikely to yield valid advantages, stop generating once the required usable groups exist, recheck stale task estimates, and keep class coverage so the learner does not collapse onto an easy subset. Track wasted rollout GPU-seconds separately for zero-variance, truncated/invalid, excess-after-target, and scheduling/straggler causes. Do not call a merely shorter trajectory a saving if it also eliminates useful learning evidence.

## What this does *not* establish

The traces record outcomes only for tasks the historical selector chose. There is no observed reward for a truly unselected task, and the replay cannot estimate that counterfactual. Reusing each source run's task inventory suppresses discovery of the wider AutomationBench inventory; the simulated new-task shares are therefore not policy targets. Outcomes are treated as stationary across 100 steps, so model improvement, deterioration, recheck after policy drift, and feedback from training are absent. A fixed task/peer mixture protects against treating one source observation as a perfect task oracle, but its strength of four groups is a modeling assumption. The four-rollout usefulness label uses reward variance, not downstream gradient quality or wall-clock GPU cost. Token totals are only a work proxy, and the risk guard assumes independent groups. The explicit age-based production recheck mechanism is not reproduced; posterior sampling provides uncertainty-driven revisits only.

## Next qualification gate

Do not switch the production selector from this replay alone. First run a *shadow* predictor alongside the current selector, logging the prior prediction for every candidate, the actual next-group usefulness, the reason for new/reuse/recheck, and token and GPU-time costs, without changing selection. Calibrate on one run and evaluate on a later, unchanged held-out run. Then compare a bounded live 100-step A/B from the same checkpoint with the same optimizer, rollout, and active-sampling semantics. The primary outcome is **held-out improvement per GPU-hour** at comparable training stability; the mechanism readout is GPU-seconds spent on non-admitted/invalid/excess groups per completed optimizer step. Keep class/task coverage, gradient/advantage distributions, KL, truncation, and reward-component validity as guardrails. If compute drops but held-out learning drops more, this is not an optimization. Keep the current path as rollback until that gate passes.

## Reproduce

From `/home/hammad/projects/rl`:

    uv run python docs/research/proposals/simulations/automationbench_replay.py export --output docs/research/proposals/simulations/automationbench_groups.json
    uv run python docs/research/proposals/simulations/automationbench_replay.py simulate --input docs/research/proposals/simulations/automationbench_groups.json --output docs/research/proposals/simulations/automationbench_replay_results.json --seeds 20
    uv run python docs/research/proposals/simulations/automationbench_replay.py simulate --input docs/research/proposals/simulations/automationbench_groups.json --output docs/research/proposals/simulations/automationbench_replay_no_truncation_results.json --seeds 20 --exclude-truncated
    uv run pytest -q docs/research/proposals/simulations/tests/test_automationbench_replay.py

The export reads the remote Trackio source but does not mutate it. The simulator is fully offline and deterministic for a fixed snapshot and seed count.
