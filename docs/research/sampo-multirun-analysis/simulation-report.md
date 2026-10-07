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
