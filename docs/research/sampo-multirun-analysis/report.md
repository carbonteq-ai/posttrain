# Why SAMPO gains do not reliably hold

## Why the improvements did not hold reliably

The model got better at parts of the workflows, while complete solutions remained unreliable. In the higher-LR FP16 run, average advertising reward rose during training, yet only one of 72 attempts solved the whole task. The next 30 attempts all missed something. A successful rollout showed that the model could produce the right procedure; it did not show that the model had learned to repeat it reliably. The evidence is too weak to call this forgetting of a mastered skill.

The trainer had a specific credit-assignment bug. SAMPO calculated credit for individual turns, but the historical loss spread that credit across the whole trajectory. An episode that did relatively well could reinforce both its useful actions and its mistakes. Credit for a correct decision could be diluted or misplaced, and simpler partial solutions still earned reward. Active sampling supplied comparisons and FP16 optimizer updates ran successfully. Both used the same flawed loss. The bug is confirmed, but we need a matched run with the corrected loss to measure how much of the task failure it caused.

The replay found another problem in the current credit calculation. Seven of 15 actions that lowered the task score still receive positive training credit. Each of these 15 turns has no comparable sibling at the same observed state. SAMPO therefore assigns it zero relative turn credit, leaving the episode's credit to determine the sign. Increasing turn weight cannot fix a zero signal. A separate cost on the harmful action's sampled tokens fixes this sign conflict in the replay. We still need a training comparison to find out whether it improves task success.

The proposed system keeps positive credit for useful actions and assigns verified harmful actions their own negative credit. It also separates training costs from the feedback used to choose tasks. Once a complete solution actually enters an optimizer update, bounded fresh rechecks can test whether the model learns to repeat it.

In this report, reward is the scorer's assessment of what happened. Advantage, or training credit, is the signal that tells the optimizer which sampled behavior to encourage or discourage. An anchor groups turns with the same preceding observations. A singleton anchor has only one turn, so it supplies no relative comparison.

The analysis covers 74,006 stored training episodes from eight runs: seven substantial runs and one interrupted attempt. Two additional launches stored no episodes. It includes 93 selected complete training transcripts, an additional full-sibling diagnostic fixture of 114 native episodes, and 14 base/checkpoint evaluation datasets with 1,120 held-out transcripts. The diagnostic selections can overlap and are not added as independent observations. The reward sweep uses 44,262 complete nontruncated training candidates; 168 allocation simulations use the current VORTEX code over recorded outcome banks. Training data drives the diagnosis; held-out results check whether the changes generalize.

The rollouts often retrieve relevant data and then act incorrectly on an exception or procedure branch. VORTEX supplies candidates with different rewards, although many states have no comparable sibling trajectory. The loss then spreads the available turn credit across the response. Meanwhile, the scorer can reward several correct writes even when the episode also makes forbidden writes. Tight budgets and policy drift cause additional problems in particular runs. We observed these behaviors or confirmed their mechanics in code; the runs do not isolate each one's contribution to final success.

All substantial runs use seed 42 and LoRA rank four, and the continuations share ancestors. They test several configurations without providing independent seed replications. Every run uses TRL post10 or post12, before the credit correction, so these results cannot tell us how corrected SAMPO would learn the same tasks.

## Our explanation: partial procedures get reinforced before complete ones become reliable

My leading hypothesis is that the training setup reinforces commonly sampled partial procedures while failing to reliably reinforce the few decisions that complete them. Several mechanisms can combine to produce this outcome. This is a proposed explanation for the run results, with a confirmed loss defect inside it; the whole explanation has not been causally established.

A full solution can receive the highest reward while incomplete solutions still dominate the learning opportunities. In the higher-LR advertising run, the sole complete solution scores task partial credit 1.0, compared with a later incorrect episode's partial credit of 0.6. Its native reward after mistake penalties is 0.92. There is no evidence here that the scorer prefers that failure to the full solution. The problem is that a flawed episode can beat its other flawed siblings, receive positive episode-relative credit, and contain both useful and forbidden actions.

The complete advertising solution also had a strong centered native episode-reward proxy: 0.92 minus its group's mean of 0.176667, or about +0.743. Its rarity therefore cannot be explained by saying every useful advantage was tiny. The unresolved question is whether that credit reached the decisions responsible for success, survived the admission cut, and was reinforced across later samples.

We checked this directly using complete non-Simple candidate groups with no truncated sibling. For each episode, we subtracted its group's mean native reward. Among episodes that violated at least one explicit negative guard, the following still scored above their group mean:

| Training arm | Guard-violating episodes | Above group mean native reward | Share |
| --- | --- | --- | --- |
| Continue FP16 100, 16 turns · 100 | 5,735 | 2,699 | 47.1% |
| FP16 base, LR 1e-4 · 60 | 3,217 | 1,630 | 50.7% |
| FP16 base, LR 6e-5 · 100 | 5,420 | 2,792 | 51.5% |
| Continue original 120 · 43 | 2,245 | 1,134 | 50.5% |
| Original SAMPO · 150 | 3,858 | 1,810 | 46.9% |
| H100, no truncation penalty · 50 | 3,276 | 1,612 | 49.2% |
| Continue 160, fixed tools · 39 | 2,052 | 1,015 | 49.5% |

Across all seven substantial runs, 46.9% to 51.5% of these guard-violating episodes have a positive centered native episode-reward proxy. The denominator is guard-violating episodes within the eligible nontruncated groups, not all episodes or guard assertions. This calculation is before batch admission and excludes groups affected by truncation shaping. It does not reconstruct supplied combined turn advantages or applied gradients. Positive credit for a flawed episode is normal in relative policy optimization; these numbers alone do not prove an incorrect reward ranking. They establish that violating a guard does not prevent positive episode credit, leaving local credit responsible for distinguishing the bad action from the rest of the episode.

That distinction is where the historical loss failed. It differentiated one sequence ratio, so turn credits affected the same trajectory-wide average. A retrieval action and a wrong write could get the same policy-credit direction. Useful partial procedures could therefore be reinforced together with their errors. This mechanism is confirmed in source and the CUDA canary; its contribution to the quality results is still unmeasured.

The available local signal also has gaps. About 39% to 51% of recorded anchors are singletons, so those anchors supply no local relative comparison. In the substantial FP16 runs, prepared local-turn credit accounts for about 27% to 31% of total absolute episode-plus-turn credit. Even with the corrected loss, a negative local term can be outweighed by positive episode credit. Retrieval usually earns reward only through later correct actions, and a message containing several parallel writes receives one aggregate turn reward. These limits can survive a derivative repair. We have not measured how often the actual combined credit on a forbidden action is positive.

Finally, yield-first sampling seeks groups likely to have nonconstant reward. That makes a batch trainable, but it does not guarantee repeated practice of a newly sampled complete procedure. The advertising task has one complete solution in 72 attempts and none in its next 30 attempts. Its five other siblings already failed at the successful step, so this behavior was unreliable before any later update. All 120 non-Simple training tasks appear in the substantial runs; a universal lack of task coverage is not supported. The sampling hypothesis concerns the kind and repetition of useful decisions within those tasks. We also did not reconstruct whether the successful candidate survived the final surplus cut.

Together, these mechanisms predict a recognizable pattern: tool execution and average partial reward improve, complete conditional success stays rare, and occasional successful rollouts are followed by failures. The observations fit that pattern. They do not yet show that later updates destroyed a previously reliable skill.

There are competing explanations worth keeping separate:

| Explanation | Evidence and limit | Test that could weaken it |
| --- | --- | --- |
| Misapplied or conflicting credit is the main bottleneck | Historical gradient mixing is confirmed; positive episode-credit proxies on guard failures are common. Actual combined credit at each bad action remains unknown. | Corrected local gradients and appropriately negative bad-action credit still leave complete success unchanged under a matched setup. |
| Retrieval and branching are too weak in the sampled policy | The complete advertising branch is rare; I-9 guide retrieval and updated conversion-policy retrieval are often missing. Other tasks fail after retrieving data. | Supplying the exact required policy and rows produces little improvement; conversely a large improvement would strengthen the retrieval explanation. |
| Updates erase useful behavior through drift or interference between tasks | The policy moves, and the original run's later KL and entropy rise. The FP16 examples alone do not establish forgetting. | Repeated fixed-checkpoint tests show no reproducible decline, or actual post-update movement is small while failures persist. |
| The model or rank-four adapter cannot represent the required logic reliably | No matched capacity ablation exists. One full advertising success shows some support, without proving adequate general capacity. | A larger adapter or stronger model gives no reliable gain after credit, reward and observations are held fixed. |

The first explanation is my priority because it contains a demonstrated implementation defect and directly addresses how a sampled success should affect later behavior. Retrieval is a second, task-dependent bottleneck. Drift is a plausible secondary mechanism, and capacity remains open. None should be declared the sole cause from these runs.

## What changed during training

For each run, early and late reward averages cover the same non-Simple tasks seen in the first 20 and final 20 observed rollout steps. Each task contributes its mean episode reward, so a task sampled more often does not automatically get more weight. The number of matched tasks varies. Sampling and truncation still depend on the evolving training process, and the two windows overlap by one step in the 39-step continuation.

| Training arm / observed steps | Stored episodes | Matched hard tasks | Early reward | Late reward | Truncated |
| --- | --- | --- | --- | --- | --- |
| Original SAMPO · 150 | 10,787 | 110 | 0.322 | 0.361 | 15.0% |
| Continue original 120 · 43 | 6,654 | 113 | 0.362 | 0.362 | 10.5% |
| Continue 160, fixed tools · 39 | 5,940 | 115 | 0.366 | 0.367 | 9.8% |
| FP16 base, LR 6e-5 · 100 | 16,986 | 116 | 0.346 | 0.363 | 17.3% |
| FP16 base, LR 1e-4 · 60 | 10,200 | 114 | 0.366 | 0.373 | 17.5% |
| H100, no truncation penalty · 50 | 8,583 | 116 | 0.377 | 0.393 | 9.9% |
| Continue FP16 100, 16 turns · 100 | 14,184 | 118 | 0.381 | 0.395 | 12.0% |

Failed calls, mistakes and truncation changed as follows:

| Training arm | Failed-call episodes, early → late | Mistake episodes, early → late | Truncation, early → late |
| --- | --- | --- | --- |
| Original SAMPO · 150 | 22.1% → 19.8% | unavailable → unavailable | 24.5% → 8.8% |
| Continue original 120 · 43 | 16.7% → 15.0% | 43.6% → 37.4% | 12.4% → 9.0% |
| Continue 160, fixed tools · 39 | 15.5% → 20.7% | 47.9% → 48.2% | 8.0% → 11.3% |
| FP16 base, LR 6e-5 · 100 | 14.9% → 14.3% | 49.2% → 47.8% | 18.7% → 16.0% |
| FP16 base, LR 1e-4 · 60 | 14.5% → 13.1% | 50.3% → 44.5% | 16.1% → 18.9% |
| H100, no truncation penalty · 50 | 13.2% → 11.2% | 48.5% → 44.7% | 8.5% → 11.1% |
| Continue FP16 100, 16 turns · 100 | 13.8% → 12.1% | 44.7% → 48.1% | 11.9% → 12.2% |

These rates count episodes in each window's candidate population. They use a different weighting from the matched-task reward averages. `tool_mistakes` includes mistakes beyond failed calls; the original run did not report it, so its value is unavailable. Changes in the sampled population limit what we can attribute to training. The model can make fewer tool errors or finish more episodes while remaining weak at conditional decisions.

During the original SAMPO run's first 100 updates, candidate non-Simple reward rose from 0.325 at steps 1 to 20 to 0.372 at steps 81 to 100. These episode averages follow a changing task population, whereas the earlier comparison matches tasks. Optimizer and checkpoint evidence also confirms that the policy changed. Higher scores and policy movement do not establish reliable procedural competence. Later in the original run, KL and entropy rose sharply; its continuations still struggled with many conditional tasks.

The training transcripts show where those partial scores came from:

- At step 18 of the fixed-tools continuation, the wire-transfer rollout reads Gmail and the sheet and earns 5/6. It still marks the $15,000 transfer `Sent` when it should remain pending approval. At step 27, the same task changes a prohibited row and misses a required row, earning 4/6. Retrieving the data did not lead to correct decisions.
- At step 6 of that continuation, the advertising rollout earns 4/5 while pausing a protected launch. At step 37, it pauses protected campaigns and misses a required campaign, scoring 0.5. The tools execute the requested changes, but the model chooses the wrong campaigns.
- In the original run, sampled wire-transfer reward rises from about 0.625 in the first 20 steps to 0.722 in the last 20. Low-CTR advertising falls from 0.304 to 0.231 over those windows. Improvement differs by task.
- The higher-LR FP16 run fully solves the advertising task at step 26. Task partial credit is 1.0; four tool mistakes lower native reward to 0.92. At step 59, a rollout pauses the protected launch and misses Mid-Market Awareness, earning partial credit of 0.6. These examples show a correct solution and a later failure. Their frequency and the intervening task groups matter before calling this a decline.
- In the same FP16 run, the step-5 wire-transfer rollout earns 5/6 while incorrectly marking the international transfer `Sent`. At step 51, it sends the notification emails and earns 0.5, but writes `Approved` where the task requires `Sent` / `Pending Approval`. The workflow ends in the wrong state despite successful tool execution.

Counting every stored candidate for these two tasks gives the following success rates:

| FP16 LR 1e-4 training task | Steps | Attempts / task groups | Mean partial credit | Complete successes |
| --- | --- | --- | --- | --- |
| finance.wire_transfer_approval | 1–20 | 24 / 4 | 0.500 | 0/24 |
| finance.wire_transfer_approval | 21–40 | 18 / 3 | 0.556 | 0/18 |
| finance.wire_transfer_approval | 41–60 | 12 / 2 | 0.472 | 0/12 |
| marketing.google_ads_pause_low_ctr | 1–20 | 36 / 6 | 0.281 | 0/36 |
| marketing.google_ads_pause_low_ctr | 21–40 | 12 / 2 | 0.317 | 1/12 |
| marketing.google_ads_pause_low_ctr | 41–60 | 24 / 4 | 0.394 | 0/24 |

The advertising success was one of six attempts at step 26; the group's mean was 0.25. It was the only complete success among 72 attempts. The next revisits, at steps 30, 41, 52, 55 and 59, produce 30 attempts with no complete success. Yet average partial credit rises from 0.281 to 0.317 to 0.394 across the three windows. Partial performance improves while the complete procedure remains unreliable. Wire-transfer reward moves from 0.500 to 0.556 to 0.472, with no complete successes. There are only a few task groups per window, and sibling episodes share a task and update. Their outcomes are correlated; they are not repeated independent tests of a fixed checkpoint. We did not reconstruct whether each particular successful candidate entered the final optimizer batch.

The successful advertising trace reads the worksheet rows and Gmail before pausing the four correct campaigns. The step-59 failure reads sheet metadata and campaign data but skips the worksheet rows. The later rollout therefore loses a necessary retrieval step as well as making the wrong decision; the two attempts did not have identical observations. Overall gains can also hide regressions on individual tasks. To distinguish forgetting from sampling noise, selection and interference between tasks, we need repeated rollouts from fixed checkpoints on the same task and temperature.

The selected transcripts are the first, last and highest-reward episodes for four tasks in each run. They help explain the failures but cannot estimate their frequency. Full task and step aggregates are in `evidence/analysis.json`; the 20-step task counts are in `evidence/task-stability.json`.

## What active sampling and VORTEX did

Active sampling filled the update batches. The work-package snapshots specify a yield-first curriculum with refill, and the curriculum and sampling logs show that it ran. Choosing candidates for the batch leaves the policy loss and task reward unchanged, including their problems.

| Training arm | Generated groups/update | Target groups | Used / generated | Singleton anchors | Informative turn advantages |
| --- | --- | --- | --- | --- | --- |
| Original SAMPO · 150 | 18.09 | 16 | 89.1% | unavailable | unavailable |
| Continue original 120 · 43 | 25.79 | 24 | 93.3% | 50.9% | 45.5% |
| Continue 160, fixed tools · 39 | 26.00 | 24 | 92.6% | 42.4% | 52.5% |
| FP16 base, LR 6e-5 · 100 | 28.31 | 24 | 84.9% | 40.2% | 53.5% |
| FP16 base, LR 1e-4 · 60 | 28.33 | 24 | 84.9% | 39.5% | 53.3% |
| H100, no truncation penalty · 50 | 28.62 | 24 | 84.2% | 40.7% | 53.9% |
| Continue FP16 100, 16 turns · 100 | 35.46 | 30 | 84.8% | 45.1% | 48.4% |

Generated groups equal logged episode rows divided by configured attempts per group. Used/generated is the average of target episode rows divided by generated rows for each completed update. In the FP16 runs, oversampling produced more eligible groups than the target batch needed. The historical metric `candidate_groups_retained` counts eligible episode rows before the final cut. For example, the FP16 base run generated 28.31 groups per update, marked about 26.68 eligible, and optimized a target of 24. The roughly 94% eligibility rate overstates the fraction that received an optimizer update.

Several limits remained:

1. Yield-first favors tasks with varying reward and refills groups whose reward is constant. This supplies more usable comparisons without supplying the missing procedure or action. Tasks that always fail may still contribute no useful update, and easy successes may be discarded.
2. Different native rewards can reflect mistake penalties while task progress stays unchanged. That happened in 65 non-Simple groups in the first older continuation and 37 in the fixed-tools continuation. These groups were a minority of the varying groups, so penalties alone cannot explain the failure pattern.
3. About 39% to 51% of anchors were singletons in runs that logged the hierarchy. With no sibling trajectory at the same anchor, the local group-relative advantage is zero. Other turns often had informative advantages before the policy loss applied them.
4. VORTEX's sampler/trainer importance correction showed mean log-probability gaps around 0.001 in the FP16 runs, with zero or negligible clamping. Those runs had smaller mismatch than older runs, though other settings also changed. The correction was not suppressing FP16 learning across the board, and the precision change left the SAMPO gradient bug in place.

The FP16 logs also rule out skipped updates from overflow as the explanation. The substantial runs report zero skipped optimizer steps over 100, 60, 50 and 100 updates, with loss scale 1024 and nonzero gradient norms. A separate replay of four tasks and 530 tokens measured policy movement between checkpoints 20 and 40 at both learning rates. That small, off-policy probe does not estimate representative KL or convergence. A higher learning rate can move the policy further while leaving its exception handling weak.

## Why Simple tasks leave little to learn

At temperature 0.5, the held-out base model already scored 96.2% on Simple tasks and 37.0% on non-Simple tasks. Simple tasks in this manifest have one or two assertions. Retries and redundant discovery sometimes push their executed traces beyond three tool calls.

| Training arm | Complete candidate groups | Simple constant reward | Hard constant reward | Hard tasks never above 0.10 |
| --- | --- | --- | --- | --- |
| Original SAMPO · 150 | 2685 | 75.9% | 11.7% | 11/120 |
| Continue original 120 · 43 | 1109 | 68.2% | 3.2% | 12/120 |
| Continue 160, fixed tools · 39 | 990 | 78.8% | 3.9% | 15/120 |
| FP16 base, LR 6e-5 · 100 | 2831 | 68.4% | 3.2% | 6/120 |
| FP16 base, LR 1e-4 · 60 | 1700 | 67.5% | 4.0% | 11/120 |
| H100, no truncation penalty · 50 | 1428 | 79.4% | 4.0% | 10/120 |
| Continue FP16 100, 16 turns · 100 | 3546 | 75.4% | 5.4% | 7/120 |

Most Simple candidate groups have constant native reward, usually because the model already succeeds. Filtering them avoids spending updates on comparisons with no reward difference. The claim that all six rollouts score one is too broad: several runs use four attempts, and some Simple groups fail, vary, truncate or incur mistake penalties.

Many hard groups have varying reward, and some sampled assertion outcomes improve. In the original run, 11/120 non-Simple tasks never achieve a complete-group mean task partial credit above 0.10. That reproduces the quoted count for that run and reward definition; other runs have different counts. A success that never appeared in the stored samples could still have nonzero probability.

## The loss spread turn credit across the response

The analyzed runs use TRL post10 or post12. Both versions compute one differentiable probability ratio for the whole trajectory:

$$
r_i = \exp\left(\frac{1}{T_i}\sum_{t=1}^{T_i}\left[\log p_{\theta}(y_{it}\mid x_{it})-\log p_{\mathrm{old}}(y_{it}\mid x_{it})\right]\right)
$$

Before clipping takes effect, the episode's policy loss reduces to:

$$
\mathcal{L}_i = -r_i\frac{1}{T_i}\sum_{t=1}^{T_i}w_{it}A_{it}
$$

Here the token credit combines turn and episode credit, and the weight is a detached sampler correction. The sums include scored policy tokens only. Every scored token receives the derivative of the same weighted average credit. Positive credit for retrieving useful data can offset negative credit for a forbidden write. Even without exact cancellation, the gradient loses which turn earned which credit. KL adds separate gradients but cannot put the policy credit back on the intended turns.

The CUDA LoRA test demonstrates the defect. With equal weights and opposing credit, the historical policy gradients cancel to zero. The corrected GSPO-token derivative gives `[-0.5, 0, +0.5]` across two scored tokens and an unscored tool token, and updates the adapter. TRL post13 automatically uses this correction for token-aligned advantages. None of the analyzed runs used it. The test confirms that the old loss can erase opposing local credit; it does not measure how often that happened in training or how much the repair improves AutomationBench success.

These SAMPO runs had more than one scalar episode reward. They recorded live turn rewards from partial-credit changes and tool-mistake penalties, then combined episode advantages with anchor-relative return advantages. The advantage calculator received substantial nonzero turn signal. The loss failed to preserve it at the intended tokens. Singleton anchors and coarse assertion rewards further limited the available credit.

Each fresh population receives one optimizer update. The numerical current/old ratio is therefore one during the training forward pass, so a near-zero clip fraction is expected. Gradients can still be nonzero, and that ratio does not constrain how far the next update moves the policy. This schedule can train vanilla GRPO, but it differs from the authors' use of separate turn rows and a derivative that preserves local credit.

Source and reproducible canary: `docs/plan/sampo-post-update-measurement.md` and `docs/research/evidence/sampo-opposing-credit-cuda-canary.json`.

## How small advantages get diluted

Small advantages spread across long responses are a plausible part of the failure. The stack centers rewards without normalizing SAMPO advantages to unit standard deviation. Recent runs had relatively small prepared episode and turn differences:

| Training arm | Prepared absolute episode advantage | Prepared absolute turn advantage | Mean episode completion tokens | Target episodes/update |
| --- | --- | --- | --- | --- |
| Continue original 120 · 43 | 0.116 | 0.045 | 4,726 | 144 |
| Continue 160, fixed tools · 39 | 0.124 | 0.054 | 4,707 | 144 |
| FP16 base, LR 6e-5 · 100 | 0.126 | 0.055 | 5,161 | 144 |
| FP16 base, LR 1e-4 · 60 | 0.122 | 0.052 | 5,280 | 144 |
| H100, no truncation penalty · 50 | 0.109 | 0.051 | 5,867 | 144 |
| Continue FP16 100, 16 turns · 100 | 0.115 | 0.046 | 5,378 | 120 |

These values average per-batch hierarchy diagnostics over candidate populations. Episode and turn magnitudes use different observation units. On a turn's policy tokens, turn credit is added to episode credit. Completion-token counts describe candidates and do not give exact lengths of the retained policy loss masks. The analysis did not reconstruct every historical token advantage or training denominator.

The historical configuration uses GRPO's average over episodes, with each episode first averaged over its scored policy tokens. With one device, full-batch accumulation and B used episodes, the unclipped policy loss is:

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}r_i\frac{1}{T_i}\sum_{t=1}^{T_i}w_{it}A_{it}
$$

The historical log-probability derivative at each scored token is:

$$
\frac{\partial\mathcal{L}}{\partial\log p_{\theta}(y_{it}\mid x_{it})} = -\frac{r_i}{B T_i}\left(\frac{1}{T_i}\sum_{u=1}^{T_i}w_{iu}A_{iu}\right)
$$

Each episode has equal weight; this differs from averaging all tokens together across the batch. T denotes the number of scored policy tokens in that episode.

The reward scale, sequence gradient and batch average affect learning in different ways:

1. Centering leaves a 0.05 advantage at that scale instead of rescaling it to one. The policy signal can be weak relative to other gradients. Its magnitude alone does not determine the parameter update: Adam, gradient alignment, loss scaling and KL also matter.
2. The historical sequence gradient dilutes a decisive turn and spreads its credit to other tokens. Consider an illustrative episode with 4,000 scored tokens, where only 20 tokens carry +0.05, all other credit is zero, and correction weights are one. Average trajectory credit is 0.00025. At `B=144`, every token gets a log-probability derivative of about `4.34e-10`. The corrected derivative would give about `8.68e-8` to each of the intended 20 tokens and zero policy credit elsewhere, a 200-fold increase on those tokens. The total scalar credit mass need not increase; its location and the resulting parameter-gradient direction change. These values illustrate the mechanism and are not measurements of a historical episode. Actual episode credit and nonzero sibling turn credit can substantially change the average.
3. More episodes do not automatically reduce the expected average gradient. When all `B` episodes supply comparable, aligned evidence, their summed contributions offset the `1/B` factor in expectation. If only a few episodes contain a useful decision, each contributes a small share, and conflicting gradients can cancel. Batch size also changes variance and the number of optimizer updates available under a fixed sample budget. Reducing `B` may increase noise or reduce prompt coverage.

Episode credit can outweigh a turn penalty. SAMPO adds episode credit to weighted local turn credit; in the FP16 hierarchy, local turn credit is about 27% to 31% of the total absolute prepared episode-plus-turn credit. A bad turn can still receive a positive advantage when its episode does relatively well. For example, +0.12 episode credit plus −0.05 turn credit gives +0.07 at unit turn weight. This is an illustration, not a paired historical observation. The corrected loss preserves the supplied sign, so it cannot turn an already positive label into a negative one. We need to measure supplied credit on forbidden actions to separate insufficient turn weight from the loss's mixing of credit.

The historical `train/rl/advantage_abs_mean` value near 0.7 measures the wrong advantages for this question. TRL logged it before replacing its scalar normalized advantages with precomputed SAMPO values. It does not measure the combined SAMPO token advantages. The helper that logs supplied policy-token credit was added later; the separate hierarchy magnitudes above are the valid historical measurements.

The policy loss mask excludes tool outputs and padding. The documented active-refill `num_items_in_batch` bug affects DAPO-style global token denominators. These runs use GRPO's episode/token averages, so that bug is not a proven additional cause here. After correcting the loss, measure per-turn gradient contributions and compare batch sizes at equal sampled-episode budgets.

## The I-9 and conversion failures at checkpoint 50

Both failures come from checkpoint 50 of the H100 run without a truncation penalty. These are held-out diagnostic cases; the earlier wire-transfer and advertising examples came from training. Exact rollout and checkpoint references are in the source appendix.

In the I-9 rollout, the model searches Drive, reads the tracker and sends seven similar manager emails. It misses the Gmail compliance guide, the legal branch and the required sheet updates, scoring 0/8 assertions. Both preservation guards pass. All three I-9 attempts at temperature 0.1 from this checkpoint score zero and fail to retrieve the guide. Across all 14 primary datasets, the guide text returns in 13/56 attempts, and the best reward is 0.778. Missing the guide is common; the model does search Gmail in other attempts. Finding the procedure still does not guarantee following it.

In the conversion rollout, the model repairs an argument error and creates a dummy conversion before finding any deal data. It then reads deal emails, the procedure and exclusion rules, says it must check exclusions, and sends eight conversions. Four are ineligible: three violate sheet rules and one violates the later $10,000 minimum. It never retrieves the leadership update that supersedes the old procedure. Four required conversions pass; four guards and the exact-count assertion fail, producing 4/9 = 0.444. Six other negative guards remain satisfied. The model acts too early, ignores some rules it read, and misses the updated policy.

Of the 56 conversion attempts, 49/56 return the SOP content, 13/56 return exclusion rules, and only 2/56 return the email with the updated minimum value. These counts check literal content in tool responses, not what the assistant says it searched or read. The highest conversion reward is 0.571; no attempt in this audit fully solves the task.

H100 checkpoint 50 violates 11.1% of guard checks at temperature 0.1 and 13.3% at 0.5. Failures concentrate in particular rules and tasks, contradicting the claim that almost every "don't do X" task violates all its guards. Preserving guards alone also says little about completion: doing nothing can preserve every guard while missing every positive goal.

## Why the reward permits partial success with forbidden actions

The scorer includes negative assertions. An assertion that is already true in the initial world earns no free positive credit for remaining true. If the agent breaks it, the scorer counts it as a failure. Explicit task-level exclusion settings also affect the count.

When no positive goals are complete, either preserving or breaking a guard can leave partial credit at zero. Once `c` of `P` positive goals pass, breaking `g` initially satisfied guards changes the fraction approximately from `c/P` to `c/(P+g)`, depending on other assertions and forced-count settings. Completing additional positive writes can outweigh the cost of new forbidden writes. In the conversion rollout, valid uploads still earn credit alongside invalid uploads and a failed count assertion.

Live turn scoring exposes some changes earlier, but retrieving an SOP usually has no assertion of its own. It earns reward through a later correct action. Respecting an exception may yield a delayed benefit, or only avoid increasing the denominator. Tool-mistake penalties arrive sooner. Reducing those mistakes can improve execution without improving the workflow decision.

When one assistant message contains several parallel tool calls, valid and forbidden writes can receive a single net turn reward. Even the corrected derivative cannot recover each call's deserved credit from that aggregate. It preserves the credit supplied by the scorer without adding more detailed reward evidence.

This scoring behavior accounts for high partial scores in training episodes that still mishandle a protected action. Report complete success and guard violations alongside partial credit, and measure success conditional on retrieving the required policy. A larger penalty alone has not been shown to fix retrieval or the gradient defect.

## What the held-out checkpoints show

### Temperature 0.1

| Checkpoint | All tasks | Hard tasks | Delta vs base [95% task bootstrap] | Guard violations |
| --- | --- | --- | --- | --- |
| base | 0.617 | 0.390 | baseline | 14.8% |
| Original SAMPO 120 | 0.624 | 0.400 | +0.006 [-0.055, +0.065] | 18.5% |
| Continuation 120 + 40 | 0.555 | 0.286 | -0.063 [-0.138, +0.007] | 8.6% |
| FP16 base 100, LR 6e-5 | 0.598 | 0.330 | -0.019 [-0.088, +0.048] | 15.4% |
| FP16 base 60, LR 1e-4 | 0.627 | 0.378 | +0.009 [-0.038, +0.064] | 15.4% |
| H100 no-penalty 50 | 0.636 | 0.394 | +0.019 [-0.040, +0.079] | 11.1% |
| FP16 100 + 100, 16 turns | 0.595 | 0.367 | -0.022 [-0.103, +0.052] | 19.1% |

### Temperature 0.5

| Checkpoint | All tasks | Hard tasks | Delta vs base [95% task bootstrap] | Guard violations |
| --- | --- | --- | --- | --- |
| base | 0.607 | 0.370 | baseline | 10.0% |
| Original SAMPO 120 | 0.619 | 0.366 | +0.012 [-0.030, +0.056] | 18.5% |
| Continuation 120 + 40 | 0.592 | 0.320 | -0.016 [-0.076, +0.042] | 17.4% |
| FP16 base 100, LR 6e-5 | 0.645 | 0.409 | +0.038 [-0.004, +0.083] | 14.1% |
| FP16 base 60, LR 1e-4 | 0.614 | 0.366 | +0.007 [-0.034, +0.048] | 15.2% |
| H100 no-penalty 50 | 0.636 | 0.402 | +0.029 [-0.008, +0.067] | 13.3% |
| FP16 100 + 100, 16 turns | 0.634 | 0.424 | +0.027 [-0.032, +0.083] | 20.4% |

The primary evaluations use the same 20 tasks: eight Simple and twelve non-Simple. Each task has three attempts at temperature 0.1 and five at 0.5. Earlier policies and the base use an older environment version; recent policies use a newer version. The AutomationBench adapter and scorer are unchanged between them. The update moves Verifiers' boxed-math scoring off the main thread, so evaluation semantics match on this point. Training conditions still differ. The exact task-selection digest and environment revisions are retained with the evidence.

Deltas compare task means. The 95% intervals resample the 20 task deltas 5,000 times with seed 42. They measure task-sampling uncertainty in a small test set, not variation across independently trained seeds. Checkpoint selection was not preregistered.

Some FP16 runs earn higher non-Simple reward at temperature 0.5 and lower reward at 0.1. Gains are small, differ by task and have broad uncertainty. The higher LR, H100, longer-budget and continuation runs change several factors at once, so they cannot isolate rank or the loss. Earlier evaluations with different task/scorer contracts remain supplementary and are excluded from these primary comparisons.

## Checking the original claims

| Original claim | Evidence-backed verdict |
| --- | --- |
| Simple tasks already solved, little RL signal | Supported in tendency; not every group is perfect, and group sizes vary. |
| Hidden-step and conditional failures on I-9/conversions | Confirmed in the quoted traces and in analogous training tasks; failure after retrieval is also common. |
| 37% of hard assertions never pass in five temperature-0.5 samples | Not reproduced as a universal rate. The base has 39/130 positive assertion identities never passing (30.0%); H100 step 50 has 37/130 (28.5%). Denominators including initially satisfied guards give different rates. |
| 11/120 training tasks never above 0.10 | Reproduced for the original run's observed complete groups; counts differ across runs. |
| 49% of assertions vary | Close to one five-attempt held-out base population (47.8% across all non-Simple assertion identities). It is not an invariant training statistic. |
| One number over about 3,800 tokens cannot reach the decisive tokens | Scalar-episode description is wrong for these SAMPO runs; turn rewards existed. The historical derivative's loss of locality is confirmed. Token lengths vary by population and budget. |
| Only failed-tool episodes moved, 0.42 to 0.38 | Cannot generalize those two rates across these runs. Task-specific reward, truncation, entropy, KL and mistakes also move; `tool_mistakes` and failed calls are distinct measures. |
| Rank four is enough; LR ablations prove it | Not established. Every substantial arm uses rank four, and no matched rank/full-finetuning ablation exists. |

Even a behavior with an independent 20% success rate has a 32.8% chance of failing five times in a row. Five failures therefore cannot establish an exploration ceiling. This calculation is illustrative; it does not assume independence among these rollouts. Policy-gradient sampling limits direct evidence for an action, while shared parameters can still change unsampled actions through generalization.

[Yue et al., arXiv 2504.13837](https://arxiv.org/abs/2504.13837) studies RLVR exploration in specific reasoning domains. Its results do not establish a hard bound for these multi-turn workflows. [Thinking Machines, LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) finds low-rank RL competitive in its experiments and proposes an information-budget heuristic. These structured-reward runs have no measured one-bit-per-episode information rate, and the paper does not establish that rank four is sufficient here. The [ARLArena implementation](https://arxiv.org/html/2602.21534v3) helps check how SAMPO should apply local credit; its published recipe does not establish quality for this stack.

## What remains wrong or untested in the current setup

The workspace now selects TRL 1.12.0.post13, which includes the loss correction. The analyzed runs used post10 or post12. We therefore know the old runs had a defective loss, while the current dependency selection has the repair. We still need to verify the deployed job's version and measure learning with it.

The current SAMPO credit builder still adds episode credit to weighted anchor-relative turn credit. Its profile defaults are mean centering, step weight 1.0, discount gamma 0.95, and KL reference `base`. Mean centering leaves small reward gaps small; adding the two terms can give a bad turn positive combined credit. Defaults can be overridden, so these source values are not a complete resolved configuration for a future run.

Other open recipe issues are the sparsity of comparable sibling states, delayed credit for retrieving a procedure, and aggregate credit for messages with several tool calls. Yield-first admission does not by itself track mastery or preserve a discovered solution. These are limits that a token-local derivative cannot remove. The historical singleton and magnitude measurements describe the analyzed runs; the current anchor implementation has changed and needs fresh measurement.

The adapter also explicitly selects `use_bias_correction_kl=False` for compatibility, and the historical schedule used one optimizer update per fresh population. The near-zero loss-time clipping statistic does not bound the ensuing parameter movement. The current source has optional policy-update selections, with no selection in the basic SAMPO profile default. Their presence is not evidence that a new schedule is qualified or was used by these runs. Post-update movement and retention need to be measured for the chosen schedule.

I would keep FP16 and rank four for the first controlled comparison. The runs made finite, nonzero updates without skipped optimizer steps, and no matched capacity experiment has implicated rank four. I would also keep the known adequate rollout budget fixed during that comparison. Budget limits caused truncation in some runs, but changing them together with the loss would obscure which change helped. Higher LR, a larger batch or a larger adapter are not my first proposed fixes.

The current-source audit, including file hashes, is in `evidence/current-recipe.json`. It records local source and dependency selection, not deployment or quality qualification.

## Recorded-data simulations and the candidate system

## What the offline simulations changed

I would keep the corrected loss and current VORTEX exploration settings, then give harmful actions their own negative credit on the sampled tokens that produced them. A larger episode penalty still disappears from relative credit when every sibling makes the same violation. The task-selection replay also shows a tradeoff: choosing more groups with usable reward differences reduces coverage. The masked-cost proposal fixes the tested credit signs, but its effect on learning remains unmeasured.

### How to read the strategies

The strategies change different parts of training. Reward shaping changes how a whole episode is ranked against its siblings. Turn weighting changes the credit assigned after a shared observation. A local mask changes which generated tokens receive a harm penalty. VORTEX changes which tasks get sampled. Practice changes when a task is revisited. Length weighting changes how much influence a long episode has. Fixing one part does not automatically fix the others.

The visuals below use two actual training trajectories. Each outlined row is an assistant turn; its small blocks summarize the recorded sampled-token mask, with the exact token count alongside. Credit values come from the current-recipe replay, not retained historical trainer advantages. Green means positive credit and red means negative credit. Proposed cost masks are labeled separately. We do not infer precise call-token boundaries from text.

[[trajectory:vendor]]

In this vendor-payment trace, a forbidden email lowers task credit but still receives positive combined credit. The corrected derivative preserves that supplied sign; fixing delivery alone would not penalize this action. This makes it a useful case for comparing the mask strategies below.

The reward search tested 44 variants on 44,262 complete, nontruncated non-Simple training episodes from seven runs. Three older runs selected the candidate; the four FP16 runs checked it afterward. Minimizing positive credit for violations first favored a cost of 2, but that roughly tripled mean absolute episode credit. I then selected the smallest change that reduced this credit-conflict rate by at least 90% in each older run, within the stated limits on complete-solution credit, safe partial-progress credit and usable-group yield. The chosen cost is 1.125; a completion bonus was available but unnecessary for that target. Both selection rules used the refined grid. The earlier exploratory 20-variant output was replaced.

### Reward shaping helps when a safe sibling exists

The strategy is to make a guard violation count explicitly against the episode reward, while optionally adding a bonus for complete, guard-safe solutions. After changing the scores, we recompute sibling-relative episode credit on the same recorded groups. This asks whether a violating episode would still be reinforced relative to a safe alternative; it does not ask whether the model would discover a different action after training. A common positive rescaling limits the change in credit magnitude.

For example, two equally productive episodes can receive different credit when only one breaks a guard. If both break it, the same penalty is subtracted from both, and centering cancels that penalty. This motivates testing local harm credit separately.

For this replay, an episode loses 1.125 if it breaks any explicitly classified guard. We then subtract the sibling group's mean score, as episode credit does. The comparison counts violating episodes that still receive positive credit, considering only groups with at least one safe sibling. This tests the episode label before admission; it does not recover the historical gradient on the harmful action.

| Training arm | Mixed-group unsafe credit: current / candidate | All unsafe episodes still positive | Candidate variance yield |
| --- | --- | --- | --- |
| Continue FP16 100, 16 turns · 100 | 47.8% / 0.9% | 33.8% | 94.4% |
| FP16 base, LR 1e-4 · 60 | 51.8% / 4.1% | 33.9% | 95.7% |
| FP16 base, LR 6e-5 · 100 | 54.0% / 4.6% | 34.7% | 96.0% |
| Continue original 120 · 43 | 50.8% / 2.5% | 30.1% | 96.2% |
| Original SAMPO · 150 | 54.0% / 1.1% | 29.7% | 88.4% |
| H100, no truncation penalty · 50 | 50.9% / 2.6% | 33.0% | 95.2% |
| Continue 160, fixed tools · 39 | 49.3% / 4.0% | 31.6% | 95.5% |

All four FP16 validation arms improve the mixed-group metric from 47.8–54.0% to 0.9–4.6%. The fraction of clean complete episodes with positive centered credit is unchanged in all seven arms, and clean partial episodes gain positive credit more often. Candidate variance yield rises slightly. These observations concern the same recorded actions; they are not new success rates.

Across all violating episodes, 29.7–34.7% still receive positive relative credit. In an all-violating group, subtracting the same cost from every sibling leaves their centered advantages exactly unchanged. Some constant failure groups also remain filtered. Increasing the cost cannot fix those labels or create a missing correct procedure. This is why the candidate needs a separately observed violation transition and local credit, followed by real qualification.

The penalty also increases credit magnitude. A common scale of 0.479, fitted on the older runs, restores their mean absolute episode credit. Positive rescaling preserves signs; it does not guarantee equal parameter gradients or KL. Native rewards here range from -0.1 to 1.0. A fixed offset and scale can bound that captured range, but a deployed scorer needs declared bounds. Even an accepted safety-adjusted score changes what VORTEX treats as task success. Goal outcome, violations and usable-credit evidence need separate definitions.

### Increasing turn weight cannot fix a missing comparison

This strategy increases the weight of the anchor-relative turn term compared with episode credit. An anchor groups turns with the same preceding observations; SAMPO compares their discounted returns. We test turn weights 0, 1, 2 and 4 under mean centering and standard-deviation normalization. A larger weight could make an observed bad branch more negative when a comparable better branch exists. Standard-deviation normalization could amplify small differences, but also changes their relative scale.

If no sibling reaches that observation state, the local comparison contains only one turn and yields zero relative credit. Neither increasing its weight nor normalizing it creates the absent alternative. Episode credit then determines the combined sign.

The independent reviewer retrieved every sibling in 21 selected groups, covering 114 episodes and 775 assistant turns. These groups include safe and unsafe siblings, complete solutions, and actions that lower the task score. Using retained sampled-token counts and the current anchor function, the replay runs the actual SAMPO credit calculation. The original trainer advantages and loss weights were not retained. These results describe what the current recipe assigns to the recorded actions, not the historical parameter updates.

For 477 of 775 turns (61.5%), no sibling has the same preceding observations. These singleton anchors receive zero relative turn credit under every tested weight and normalization. Multiplying zero leaves the missing comparison unresolved.

| Normalization / turn weight | Local/shared derivative sign disagreements | Negative immediate turn reward with positive combined credit |
| --- | --- | --- |
| mean / 0 | 0/694 | 14/42 |
| mean / 1 | 9/694 | 12/42 |
| mean / 2 | 23/694 | 12/42 |
| mean / 4 | 25/694 | 12/42 |
| mean_std / 0 | 0/694 | 14/42 |
| mean_std / 1 | 31/694 | 12/42 |
| mean_std / 2 | 34/694 | 12/42 |
| mean_std / 4 | 45/694 | 12/42 |

The derivative comparison assumes a probability ratio of 1, importance weight of 1 and the GRPO mean loss, with KL excluded. With mean centering and turn weight 1, the corrected and historical losses give opposite derivative signs on 9 of 694 turns with nonzero credit. Standard-deviation normalization and larger weights change more signs, but we do not know whether those changes help the task. Keep mean centering and weight 1 in the first controlled loss comparison. The repair applies local credit correctly; this selected sample cannot establish that it alone explains the quality gap.

In this diagnostic selection, 7/15 turns with negative assertion progress still receive positive combined credit. The expanded cases put all 15 negative-progress turns at singleton anchors, in episodes ending with an explicit guard failure. Increasing the global turn weight cannot change their zero relative turn term. Negative progress is a scalar task-score regression; the final guard failure does not by itself establish which individual assertion changed at that turn. Native turn reward also includes tool-mistake costs, so its sign is a separate diagnostic. The current scorer defines assertion progress as current minus previous task credit and does support negative values. The population scan rules out the mistaken explanation that it simply clips all negative progress.

At step 2 of the FP16 base run, the vendor-payment rollout sends an email to a prohibited recipient on assistant turn 3. The task score falls by 0.03333. With no comparable sibling at that state, relative turn credit is zero, and the episode's positive credit leaves the action with +0.14048 at every tested turn weight. The final assertion confirms the recipient was prohibited. The scalar turn assessment records the score drop, although it does not identify the timing of each individual assertion change. The exact rollout reference is in the source appendix.

### Give the harmful action its own credit

We compare two strategies using an observed nonnegative turn cost. Subtraction keeps the existing combined credit and subtracts a weighted cost from it. Masked replacement keeps ordinary credit outside the cost span, but replaces credit inside that span with a negative cost. The first can remain positive if episode credit is large; the second makes the cost span negative whenever its cost and weight are positive. Both keep the original loss denominator. We test cost weights 0, 0.25, 0.5, 1, 2, 4 and 8.

[[trajectory:mask]]

In this replay the evidence identifies an entire assistant turn, so its sampled policy tokens form the mask. A finer mask around the harmful call's name and arguments would require reliable per-call attribution and original token spans. Tool observations, prompts and padding remain excluded. The diagram shows the tested whole-turn version.

The credit reviewer tested another 14 variants on the same 114 episodes. Each applies the following cost to the sampled tokens of that assistant turn:

$$
c_{it} = \max\left(0,-r^{\mathrm{native}}_{i,\mathrm{turn}(t)}\right)
$$

Here the native reward belongs to the assistant turn containing token t. The label combines falling task score and tool penalties. It does not isolate each harmful call, and a positive net turn can still contain a harmful action.

| Local-cost method | Weight | Negatively rewarded turns still positive | Positive-progress turns losing positive credit |
| --- | --- | --- | --- |
| subtract | 0 | 12/42 | 0/198 |
| subtract | 0.25 | 12/42 | 0/198 |
| subtract | 0.5 | 11/42 | 0/198 |
| subtract | 1 | 3/42 | 0/198 |
| subtract | 2 | 2/42 | 0/198 |
| subtract | 4 | 2/42 | 0/198 |
| subtract | 8 | 0/42 | 0/198 |
| disjoint_override | 0 | 12/42 | 0/198 |
| disjoint_override | 0.25 | 0/42 | 0/198 |
| disjoint_override | 0.5 | 0/42 | 0/198 |
| disjoint_override | 1 | 0/42 | 0/198 |
| disjoint_override | 2 | 0/42 | 0/198 |
| disjoint_override | 4 | 0/42 | 0/198 |
| disjoint_override | 8 | 0/42 | 0/198 |

Simply subtracting a weighted cost from existing combined credit leaves conflicts until weight 8. An alternative assigns ordinary task credit outside the cost span and negative cost credit inside it:

$$
A^*_{it} = (1-V_{it})A_{it} - V_{it}\kappa c_{it}
$$

The binary mask V marks the cost span, c is the nonnegative observed cost, and κ is its positive weight. This removes positive cost-span credit by construction, while preserving the original episode/token denominator and full sampled KL support. No sampled positive-progress turn loses positive credit in this fixture. All 105 positive-credit complete-episode turns are unchanged because none has a net-negative cost label. These checks describe the selected data; mixed useful and harmful calls could lose useful credit under a whole-turn mask.

I prefer testing the explicit disjoint-support contract over relying on a large global penalty. Its sign guarantee is an intended surrogate objective, not an unbiased estimator of the old task reward or proof of learning. It requires provenance-backed labels, a new admission rule for constant-outcome groups with valid process credit, and backend qualification. Start with whole-turn support where that is all the scorer can justify. Use call-name and argument masks only when original token serialization and the action ledger establish the span. Keep future outcome evidence out of the model's earlier conditioning context. The detailed design is in `reward-and-mask-design.md` and the competing proposal in `credit-strategy-notes.md`.

### VORTEX trades usable groups against task coverage

This strategy changes the next task selected, leaving recorded rewards and actions fixed. We compare uniform task selection with the current VORTEX defaults, removing its uncertainty bonus, and increasing its exploration share. The defaults use 20% exploration and uncertainty weight 4; the other VORTEX settings use weight 0 or 50% exploration respectively. Uniform selection ignores the learned priority. Removing uncertainty tests whether favoring already productive groups sacrifices task coverage. More exploration tests whether broader sampling finds more useful groups or complete solutions.

I executed the current controller's select and observe methods against frozen banks of actual native groups. There are 168 allocation simulations: seven arms, early and late banks, four settings and three allocation seeds. Each uses 30 rounds of 24 distinct candidate groups. This is equal candidate-group budget, without refills or model updates. Only tasks observed in both halves are included: 100–137 per arm. Missing outcomes were not invented. Historical task sampling and truncation exclusions limit the banks; simulator seeds do not constitute training replications.

The current exploitation score combines a beta-model mixed-group prediction from reward history with empirical useful-group yield. Its uncertainty lane uses aged binary-yield evidence. Calling the whole controller a pure binary-yield policy would omit part of its code.

| FP16 base-100 later-bank allocator | Useful candidate groups | Tasks covered out of 114 | Complete non-Simple episodes |
| --- | --- | --- | --- |
| uniform | 78.8% | 113.7 | 4.6% |
| current_defaults | 93.4% | 108.0 | 3.7% |
| no_uncertainty_bonus | 96.2% | 101.3 | 3.6% |
| more_exploration | 90.5% | 112.0 | 3.6% |

Removing the uncertainty bonus raises useful-group share in every later bank and reduces coverage in every arm. In the FP16 base-100 bank, it moves yield from 93.4% to 96.2% while coverage drops from 108.0 to 101.3 of 114 eligible tasks. Raising exploration to 50% improves coverage but costs yield; it does not consistently improve complete non-Simple episode share. Uniform selection's higher overall completion rate is partly driven by Simple tasks, so the table explicitly separates non-Simple completion. None of these static outcomes measures retention of a newly learned skill. Keep the 20% exploration setting and uncertainty weight 4 until a live comparison measures that objective.

The current controller drops feedback groups containing a finite score outside [0,1]. The independent retention agent found that feeding the cost-1.125 reward directly would reject 71.6–81.8% of these complete hard groups. Even native rewards would reject 0–17.4% under the current source; those figures do not establish what the historical deployed controllers received. The allocator replay records invalid observation groups and still counts their sampled outcomes. A signed optimization cost needs a separate curriculum feedback contract rather than silently flowing into its beta success model.

A separate composition sensitivity identifies rare-success tasks using only the first half of each run, then inspects their recorded later outcomes. A 25% practice mixture would change observed useful-group yield by less than one percentage point in every arm; it does not predict learning or prove 25% is best. Practice should reserve candidate selections across later populations, preserve within-population task uniqueness and collect fresh RL trajectories. Constant-success groups may still be filtered. Learning from a stored solution needs a separately named demonstration method.

[[trajectory:ads]]

The practice strategy is to revisit a task after a rare complete solution instead of relying entirely on the allocator to select it again. The 25% mixture is a static composition check using recorded outcomes. The proposed queue is a separate, untested training intervention: only verified optimizer-used successes enter it, later visits collect fresh rollouts, and admission still decides whether those rollouts produce an update. A successful sampled episode therefore does not by itself guarantee reinforcement.

The retention agent proposes an event-triggered recheck queue instead of a blanket reserve: enqueue a task only after a complete branch is verified as optimizer-used, then schedule bounded fresh rechecks in later populations. Expire the queue after a declared number of visits or consistent independent success. This is a competing strategy to qualify. The step-26 advertising success had five failing siblings at the same checkpoint; reliable mastery followed by forgetting is unproven. Frozen-checkpoint execution and the successful branch's actor-score trajectory are needed to distinguish noise from regression.

### Length compensation changes which episodes dominate

This strategy increases an episode's loss weight with its scored-token length. We compare no compensation, square-root compensation and linear compensation, using a fixed reference length for scale. With equal credit, square-root weighting narrows the per-token gap between short and long episodes; linear weighting removes it. The price is greater total influence for long episodes. These calculations use recorded completion lengths as a diagnostic proxy, not reconstructed historical loss-mask lengths or model parameter gradients.

For equal advantage and a shared episode count B, at probability ratio and importance weight one, the corrected unclipped GRPO derivative per scored token has magnitude:

$$
\left|\frac{\partial\mathcal{L}}{\partial\log p_{\theta}(y_{it}\mid x_{it})}\right| = \frac{|A_{it}|}{B T_i}
$$

In these runs, the 90th-percentile completion is 3.09–3.96 times as long as the 10th-percentile completion. A square-root length weight reduces that ratio to about 1.76–1.99; linear compensation removes the per-token difference but increases total influence of long trajectories. This is an objective change. It is not evidence that gradients are divided by the whole run's episodes, or that raising LR repairs attribution. Retain the current denominator for the first controlled pilot and measure action-level influence before choosing a new weighting rule.

### Candidate system and the tests it still needs

The offline candidate uses the corrected token-local derivative, mean centering and turn weight 1, with FP16 and rank 4 held fixed. Keep VORTEX's exploration lane, and measure candidate selection, useful-group admission and final optimizer use separately. Test the selected guard cost in a versioned scorer fixture with declared reward bounds and matched credit scale; retain positive-goal progress. The tested terminal cost is an episode-label intervention. A new transition-local violation term must be attributed only as finely as native evidence allows, and must be checked after episode and turn terms combine. It has not been validated by the terminal sweep. Multi-call messages require per-call evidence or a separately costed diagnostic with one mutating call per turn.

Use a fixed conditional-task panel to determine whether complete procedures become more frequent and survive later updates. If sampled complete branches remain rare, compare fresh scheduled practice and verified procedure demonstrations as separate methods. Post-update actor-score probes distinguish drift from wrong credit. A live experiment must verify the deployed trainer pin, scorer, masks, resolved settings and actual allocations.

The stopping point for this offline search is a candidate with fewer demonstrably wrong episode labels, preserved sampled complete-solution credit and explicit unresolved local-credit and learning gates. Recorded rollouts cannot supply a causal learning result. The 30-update paired training pilot specified in `report.md` remains necessary before calling this a good learning system.

Reproduce with `simulate_recorded_groups.py`, `replay_recorded_curriculum.py`, `replay_turn_credit.py`, `simulate_local_cost.py`, then `build_simulation_report.py` in this directory. Raw inputs and native full-group captures are hashed; output tables are in `evidence/recorded-group-simulation.json`, `evidence/recorded-curriculum-replay.json`, `evidence/turn-credit-replay.json` and `evidence/local-cost-simulation.json`. No live training, scorer or configuration was changed.


## What I propose

The simulations narrow the candidate to corrected token-local credit, explicit local-cost support, separately defined curriculum feedback, and bounded fresh rechecks of verified optimizer-used complete solutions. FP16, rank four and the current VORTEX exploration settings stay fixed initially. The loss repair, local-cost objective and practice queue need separate comparisons so their effects remain identifiable. The local-cost objective extends SAMPO and requires a named recipe and qualification; no new training or live reward change has been launched.

### First, verify the credit delivered to real actions

Use the retained complete advertising rollout and guard-violating training rollouts, together with their complete sibling groups. Recompute an explicit, versioned credit fixture from the native rewards and turn assessments under declared shaping and anchor rules; historical supplied token advantages were not retained. Preserve original token IDs and masks where available. If serialization must be reconstructed, use the pinned template and record that limitation. Freeze the same context, token advantages, weights and actor scores for both loss replays. Confirm that positive retrieval credit and negative write credit produce their intended local derivatives, with tool outputs excluded. Record episode credit, turn credit, combined credit and the policy gradient separately. Treat a synthetic cancellation canary as a prerequisite, not a substitute for this real-trajectory check.

For new rollouts, identify turns where a previously satisfied guard becomes violated. Measure whether their actual supplied combined credit is positive, zero or negative, and whether it survives admission and receives the intended gradient. Also record admission of rare complete solutions. These measurements distinguish a weak or wrong label from failure to apply a correct label. Observe small post-update actor-score probes on the same contexts; a loss-time ratio of one is insufficient.

### Then run a comparison that changes only credit delivery

Start both arms from the same model and task inventory. A concrete pilot is 30 optimizer updates with 24 prompt groups and six attempts per group, giving a target of 144 used episodes per update and 4,320 per arm. Hold LR at 6e-5, LoRA rank at four, FP16, beta at 0.01, reward, VORTEX controller parameters and seed, normalization, masks and rollout budgets fixed. These are proposed pilot settings based on the substantial FP16 base recipe, not claimed optimal values. The primary budget is completed updates and used episodes. Set equal candidate-generation and probe caps, then report actual generated episodes, probe tokens and GPU time. An adaptive controller can produce different task allocations and candidate counts as the policies change; those differences are part of the recipe's response and must be recorded. Matching used episodes, generated episodes and update count simultaneously is not generally possible. An equal-generated-budget comparison is a separate analysis and must report its differing used episodes and updates.

Compare an explicitly selected historical derivative with the corrected derivative in isolated, versioned research configurations on the same current runtime. Using the whole post12 wheel as the control would also change KL numerics and other fixes. Do not revert the workspace's corrected pin or alter a live run. Both arms need the same scorer and measurement hooks; historical score plots alone are not a controlled baseline. The first pilot should use the existing known budgets, and deployment must verify the immutable image and actual-job dependency versions.

At fixed checkpoints, repeat a fixed panel of conditional training tasks and held-out tasks, with Simple tasks as retention checks. Keep task world, sampling temperature and attempt counts identical within each comparison. Track complete success, positive-goal progress, guard violations, required-policy retrieval, and success after retrieval. Repeated failures at the same checkpoint help estimate sampling variability; a reproducible drop between checkpoints supports forgetting. Multiple seeds and more tasks are required before claiming general improvement.

If the corrected arm makes complete solutions reliably more frequent at equal budgets, the delivery defect gains causal support. If local credit and gradients are correct but complete success remains flat, investigate the labels and missing procedures before raising LR or rank.

### If bad actions still receive positive credit, change the reward and credit construction

Test a separate scorer variant with an explicit cost for each newly violated guard at the violating turn, plus a terminal bonus for completing the workflow without guard violations. Retain positive-goal progress so doing nothing is not the only safe outcome. The current scorer's guard-denominator effect can be delayed or absent at zero progress; the proposed cost makes that transition visible immediately. Define the intended ordering of complete, partial and guard-violating outcomes before tuning weights. Implement task meaning in the environment scorer, not as task-specific logic inside the reusable trainer.

Check the supplied combined advantages in real traces and fixtures. A stronger guard penalty is insufficient if positive episode credit still gives the violating action the wrong sign. If that happens, test a separately weighted outcome and violation-credit construction that preserves the bad-action signal instead of broadcasting the episode term over it. This would be a new, explicit recipe requiring qualification. Do not invent token-level attribution when the scorer only identifies a whole turn. For mixed parallel writes, compare a diagnostic arm with one mutating call per turn, or obtain reliable per-call evidence before assigning finer credit. Measure its execution cost.

Test this scorer/credit variant against corrected SAMPO with the old scorer. Keeping the derivative fixed isolates whether labels were the remaining bottleneck. If guard signs improve but policy retrieval and complete success do not, the intervention solved only one part of the problem.

### If complete branches remain rare, add practice and procedure support

Keep VORTEX for usable-group yield and test an event-triggered recheck queue after a complete branch is verified as optimizer-used. A bounded experimental setting is at most two of 24 initial candidate selections per later population, with explicit expiry and queue spillover. The replay does not establish the best quota, and the earlier 25% reserve remains a composition sensitivity rather than the chosen system. Preserve task uniqueness within a collection/refill population; practice repeats across later populations. Recollect fresh rollouts for RL instead of silently inserting stale successful trajectories as if they were on-policy. Report both selected and optimizer-retained practice groups. An all-success or all-failure group can still be discarded for constant reward, so a revisit does not guarantee reinforcement. Guaranteeing an optimizer-used practice quota would change admission or the algorithm and needs explicit qualification. Demonstration-based practice is a separate method.

For a few fixed tasks, compare six and twelve sibling attempts from the same checkpoint and at matched generated-episode budgets. Measure repeated-anchor coverage, local credit and complete-solution frequency. More siblings may improve comparison density but reduce task breadth, so report both. Do not merge different world states merely to increase anchor counts.

Use an oracle-information diagnostic on the held-out I-9 and conversion tasks: provide the exact required policy and relevant rows in a separate diagnostic condition. A large improvement would implicate retrieval; continued failure would implicate applying the rule or the available model capacity. If procedure access is the bottleneck, test verified procedure demonstrations or teacher-assisted discovery in a separately named training method. A demonstration that changes the method should not be presented as ordinary SAMPO exploration.

### If behavior actually regresses, control drift separately

When fixed-panel success falls reproducibly, correlate it with actual post-update actor movement and the tasks trained between the checkpoints. Compare a smaller LR or a controlled update schedule while holding credit and reward fixed. A stronger KL coefficient is another isolated experiment; base-model KL alone does not directly protect a skill learned later. The compatibility-selected KL gradient also needs its own qualification before changing it. Success on one rollout followed by failure on another does not justify these drift interventions by itself.

Compare a larger adapter or stronger model after those checks, using matched observations and a tuned, controlled training recipe. I would change batch size only if measured useful decision credit or interference supports it. The number of episodes alone does not establish dilution.

The resulting decision is conditional: fix delivery if the labels are right; fix labels if forbidden actions are rewarded; supply procedures if the policy cannot find them; constrain updates if previously reliable behavior demonstrably regresses. That sequence gives each proposed change a measurable reason to exist.

## Source appendix, data gaps and reproduction

The source is Trackio server `https://trackio.carbonteq.com`, project `posttrain-lab`, accessed read-only. Retained native Verifiers episodes are the authority for replay. Capture dates (UTC): 2026-10-01, 2026-10-02; the runs span September 27 to October 1. Capture checks expected page sizes, unique trace IDs and provider trace counts at the start and end. These checks verify extraction of all stored traces. They cannot establish that the provider stored every generated episode.

Stored trace counts differ from generated-row logs in these runs: Original SAMPO · 150: 10,787 stored versus 10,856 logged generated; Continue 160, fixed tools · 39: 5,940 stored versus 6,084 logged generated; H100, no truncation penalty · 50: 8,583 stored versus 8,586 logged generated. The original run has 21 incomplete groups containing 47 stored episodes; these are excluded from complete-group statistics. H100 manual r5 is missing three episodes at step 38, with 204 generated and 201 stored. That leaves three incomplete groups containing 15 stored episodes. The original run's 69 missing episodes occur at steps 41, 83 and 87. The fixed-tools continuation is missing 144 episodes at step 11. The gaps may be systematic, so we do not assume they are random. An interrupted run can also generate a rollout step without completing its optimizer step. Sampling logs and native traces are kept at their separate units of observation. See coverage and group_issues in `analysis.json`.

Metric definitions and limits:

- "Hard" means outside the native `simple` domain. It does not measure difficulty. Non-Simple tasks vary greatly in assertion count and workflow complexity.
- Task reward is native `rewards.partial_credit.score`. Native episode reward also includes mistake penalties, and the trainer's truncation shaping further changes batch admission. Constant-native-reward rates describe candidate groups; they do not reconstruct the population retained by the optimizer.
- Candidate groups use native prompt-group IDs. Each group must have the configured attempt count and a common task/optimizer step. Incomplete or mixed groups are excluded from group statistics without repair.
- Assertion identity combines task, predicate and canonical parameters. A guard's predicate contains `_not_` or ends with `_unchanged`. Preserved guards count in the guard-check denominator even when the scorer excludes them. Positive assertion coverage uses the remaining predicates.
- Training "never passed" counts every stored sample in the run. Held-out rates use three or five attempts. Their different opportunities for success prevent a direct comparison of exploration capacity.
- Trace counts are read-only projections. The selected training transcripts explain behaviors without estimating their frequency. Complete held-out transcript populations support frequency estimates for those 20 tasks. The policy-content check uses literal markers and can miss equivalent wording.
- Episodes from the same task and related checkpoints are correlated. A single seed, changing curriculum and environment revisions, budget changes and checkpoint selection limit causal conclusions. There is no corrected-loss quality run, rank ablation or representative replay of token gradients.

### Training lineage

| Training run / provider ID | Observed rollout step | TRL commit | Environment commit | Starting model |
| --- | --- | --- | --- | --- |
| `train.sampo-lfm26-sampo-turns-150-lr5e5-kl5e3-20260927-r2` / `c49136f7fe9e42bbbe79431b2c060fdb` | 150 | `4950b99d457faacbec856cbd5305732e7b3cf7b0` | `3a486b0ab173ece56f480f135c4cadfbf0135b57` | base |
| `train.sampo-lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1` / `aab30e6e079b4974bb82512a703e8592` | 43 | `4950b99d457faacbec856cbd5305732e7b3cf7b0` | `0afb73d7bd36403cbfb02fa2179f974817ce2344` | `lfm26-sampo-turns-150-lr5e5-kl5e3-20260927-r2` checkpoint 120 / `a3e838ab75288ba152d9681a8eddaf0687e3645c6f445c1d21442b77865a9f4d` |
| `train.sampo-lfm26-sampo-cont40-fixed-tools-20260928-r1` / `fbcdd7918f8b46f7b29322da3b61d4c9` | 39 | `4950b99d457faacbec856cbd5305732e7b3cf7b0` | `61448b5d941b791280ab3be5eb7d72cb35200f8e` | `lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1` checkpoint 40 / `2fe7344b937927b6c30965027fb09239e93327b73b003da23e44b809f90e79b9` |
| `train.sampo-lfm26-sampo-fp16-base-r4-lr6e5-kl1e2-t08-20260929-r1` / `476f5dd9a2fb4097aeb48055e9c62bcd` | 100 | `c4d0db051a7839fe1b1d587185fac33ba88c784f` | `0bad6187f40e23666150234ade99efd35ac567a0` | base |
| `train.sampo-lfm26-sampo-fp16-base-r4-lr1e4-kl1e2-t08-60-20260930-r1` / `444abb71c7194b83a2703a5c8f504402` | 60 | `c4d0db051a7839fe1b1d587185fac33ba88c784f` | `0bad6187f40e23666150234ade99efd35ac567a0` | base |
| `train.sampo-lfm26-sampo-fp16-base-lr6e5-notrunc-6k14t-h100sxm-50-20260930-manual-r5` / `ccd2fa436bbe42018a316076cc97aed4` | 50 | `c4d0db051a7839fe1b1d587185fac33ba88c784f` | `0bad6187f40e23666150234ade99efd35ac567a0` | base |
| `train.sampo-lfm26-sampo-cont100-g30x4-16t-lr6e5-kl1e2-20260930-r1` / `2fbe8ca613d34e31a1d7348db85cd194` | 100 | `c4d0db051a7839fe1b1d587185fac33ba88c784f` | `0bad6187f40e23666150234ade99efd35ac567a0` | `lfm26-sampo-fp16-base-r4-lr6e5-kl1e2-t08-20260929-r1` checkpoint 100 / `43949bbe1748dc67990a420cd60f3493f83e269543222863a99b81f03c8d213c` |
| `train.sampo-lfm26-sampo-fp16-base-lr6e5-notrunc-6k14t-h100sxm-50-20260930-manual-r4` / `38feee2f624f4c6a9679d9ee0e3dca4a` | 4 | `c4d0db051a7839fe1b1d587185fac33ba88c784f` | `0bad6187f40e23666150234ade99efd35ac567a0` | base |

### Rollout references

The explanations identify examples by task, run and step. The native IDs below let another reader retrieve the exact episodes without interrupting the narrative.

| Example | Native episode ID |
| --- | --- |
| Fixed-tools continuation: wire transfer, step 18 | `8d3a003b15334b37a263471edf196730` |
| Fixed-tools continuation: wire transfer, step 27 | `e99f47fd17cd496a80442973b43e2071` |
| Fixed-tools continuation: advertising, step 6 | `b2e32f28d2d74335bd721e8c799db1bf` |
| Fixed-tools continuation: advertising, step 37 | `d142dac174724f1cb654c3fd586a8b80` |
| Higher-LR FP16: complete advertising solution, step 26 | `1ee1fe7e66154fa69d61458765377f12` |
| Higher-LR FP16: advertising failure, step 59 | `d7914ef23b0a4eeaba46ffe05ef488e7` |
| Higher-LR FP16: wire transfer, step 5 | `466aa76e4e7b4f5aadfa7390cc605f35` |
| Higher-LR FP16: wire transfer, step 51 | `f6365a0fc7da4db282b0e2f2ebebc59b` |
| H100 checkpoint 50: held-out I-9 failure | `fd6ba6dba1af4b4ea2c17f6c8b9bc368` |
| H100 checkpoint 50: held-out conversion failure | `63e2268e478848b695db16bfeb63f546` |
| FP16 base: vendor-payment credit conflict, step 2 | `b668657dab574811be86d2df0714464b` |

The H100 checkpoint-50 adapter digest is `711fe9f0749fbd4b73f75bc5fdb36b4ab0441d0fabb54ababb3fa757eed0f343`. The primary evaluation task-selection digest is `4526f8a12081555b9061fa5c45924fa2d2b635dee832c27e6bb72cb9234941d7`. Exact trainer revisions are retained in the training-lineage table and source data.

Exact held-out checkpoint source-run IDs, local checkpoint steps, adapter digests and versions are in `report-snapshot.json` and `analysis.json`. Continuation 120 + 40 has 160 ancestral updates; FP16 100 + 100 has 200. Local steps and ancestral update counts are distinct.

From `/home/hammad/projects/rl`, reproduce without writing to Trackio:

```bash
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt .venv/bin/python docs/research/sampo-multirun-analysis/export_evidence.py
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt .venv/bin/python docs/research/sampo-multirun-analysis/export_transcripts.py
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt .venv/bin/python docs/research/sampo-multirun-analysis/export_training_samples.py
.venv/bin/python docs/research/sampo-multirun-analysis/analyze.py
.venv/bin/python docs/research/sampo-multirun-analysis/audit_cases.py
.venv/bin/python docs/research/sampo-multirun-analysis/task_stability.py
.venv/bin/python docs/research/sampo-multirun-analysis/credit_conflict.py
.venv/bin/python docs/research/sampo-multirun-analysis/audit_current_recipe.py
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt .venv/bin/python docs/research/sampo-multirun-analysis/replay_turn_credit.py
.venv/bin/python docs/research/sampo-multirun-analysis/simulate_recorded_groups.py
.venv/bin/python docs/research/sampo-multirun-analysis/replay_recorded_curriculum.py
.venv/bin/python docs/research/sampo-multirun-analysis/simulate_local_cost.py
.venv/bin/python docs/research/sampo-multirun-analysis/build_simulation_report.py
.venv/bin/python docs/research/sampo-multirun-analysis/build_report.py
```

Capture requires read access and the installed Trackio client. Raw derived inputs are ignored under `.posttrain/state/analysis/sampo-multirun`. Their SHA-256 hashes, coverage records, aggregate results and methods are retained here. Capture reuses complete files whose counts still match the provider. If a provider count changes, it re-extracts the run, after which the report must be regenerated. The analysis changed no experiment, reward or live configuration.

The independent buddy checked pinned trainer behavior, log units, training transcripts and denominators. Its notes are in `review-notes.md`, and the report includes its corrections.
