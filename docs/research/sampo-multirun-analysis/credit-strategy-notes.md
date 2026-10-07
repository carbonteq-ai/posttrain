# Independent credit strategy: keep comparison credit and explicit costs separate

Research proposal, 2026-10-02. This is a new objective to qualify, not a description of existing SAMPO or a demonstrated learning improvement. No training was launched.

## What the existing replay establishes

The current builder in `packages/train/src/posttrain/train/sampo_advantages.py` computes episode-relative credit plus discounted return credit centered among matching observation anchors. An initial diagnostic reconstructed 14 groups, of which 298 of 476 turns had singleton anchors. The final expanded fixture below replaces that initial sample for pooled results. Every singleton's relative turn credit is zero. Changing its weight or standard deviation normalization cannot create a comparison.

In that initial 14-group diagnostic, the reconstructed mean-centered, weight-one recipe assigned positive credit to 5 of 25 turns with a negative immediate native turn reward. All five are singleton anchors in FP16 base100 onboarding episode `977c581b028b49848948770f16274d57`. Their immediate score is -0.02, episode advantage is +0.0188888889, and turn advantage is zero. The five turns remain positive at every tested turn weight. These scores represent native tool-mistake penalties; this does **not** establish the sign of credit on an assertion violation. That initial selection contained no negative assertion-progress components; the final expanded selection explicitly adds them.

The terminal guard-cost sweep separately shows the limits of episode centering. If all siblings have the same violation indicator, subtracting a uniform guard cost changes no centered episode advantage. If all rewards are equal, episode-variance admission still rejects the group. Neither fact is fixed by increasing the scalar penalty.

This combination suggests a structural gap: comparison credit ranks sampled procedures, but a known invalid action may require an absolute local teaching signal. Missing retrieval is a third problem: the scorer does not currently give a reliable immediate reward merely for obtaining the required procedure. A local action penalty cannot supply that positive behavior.

The buddy subsequently expanded the replay to 21 groups, 114 episodes and 775 turns, including seven groups selected for negative assertion progress. `simulate_local_cost.py` now replays this expanded sample. There are 42 net-negative native rewards, of which 12 have positive combined credit; all 12 are singleton anchors. Subtracting the uncentered cost `kappa * max(0, -turn_reward)` leaves 12, 12, 11, 3, 2, 2 and 0 positive conflicts at kappa 0, .25, .5, 1, 2, 4 and 8. Disjoint override assigns negative cost credit at every nonzero kappa and leaves no positive conflicts by construction. Neither method removes positive credit from any of the 198 positive-progress turns or 105 turns belonging to complete episodes in this sample. These are prepared-credit properties, not sampled outcome improvements or an unbiased population estimate. Net-negative rewards mix assertion progress and tool penalties; positive net turns can still contain costs, so this is incomplete support rather than a causal guard mask.

The expanded fixture is hashed in `evidence/local-cost-simulation.json`. Current mean-weight-one advantages and native sampled-mask counts define its support. It does not invent JSON argument boundaries or remove original input/conditioning context.

## Candidate contract

Keep the corrected token-local derivative, current rank, dtype, KL recipe, mean centering, turn weight and denominator in the first comparison. Add one separately named, versioned objective for local evidenced costs.

For episode i, let T_i count its original sampled policy tokens, B count the declared complete optimizer population, and A_it be the current episode-plus-turn advantage on token t. Let V_it identify tokens inside a valid, provenance-backed violation span. Let c_it > 0 be the declared cost on that span. Invalid, missing, inapplicable and abstained assessments do not become zero cost or valid spans.

Use disjoint policy contributions:

    ordinary contribution: (1 - V_it) * A_it
    local-cost contribution: -V_it * kappa * c_it
    A*_it = (1 - V_it) * A_it - V_it * kappa * c_it
    L_policy = -(1/B) sum_i (1/T_i) sum_t rho_it * A*_it

The policy ratio must retain the corrected token-local derivative, with backend qualification still required; importance correction and clipping remain separately specified. Neither V nor c nor the prepared advantages is differentiated. Tool output, input, padding and nonsampled positions have no policy contribution. The KL support stays at the original sampled policy support, independently of the disjoint policy supports.

This prevents episode-relative success elsewhere from assigning positive policy credit to a verified violation span. The cost is deliberately not centered within the sibling group or normalized by an empirical standard deviation. Zero is the declared immediate-cost baseline. Singleton anchors and all-unsafe siblings therefore cannot erase it. Keep T_i as the original policy length even when some tokens move between contribution sets; do not silently renormalize each set to its own token count or sum two full-strength losses. A positive common credit scale may control overall magnitude, but actual gradient norms and post-update KL must still be measured.

The objective removes positive task credit on the whole selected span. That is a real tradeoff and an intentional surrogate, not an unbiased policy-gradient estimator for the original terminal task reward. Its value is a testable contract: an evidenced invalid action should not be reinforced by unrelated progress. Do not describe the sign guarantee itself as learning evidence.

An initially passing guard should contribute no positive achievement merely for staying true. A true-to-false transition can supply a cost, while repeated observations of the same already-broken guard must not charge it again. A restored state and subsequent new violation need an explicit repeated-transition policy; restoration does not erase an irreversible external side effect. Positive goal progress and safe completion are separate components. Retain scorer values and provenance independently from advantages and masks.

When tool serialization provenance maps a harmful call to original sampled JSON tokens, the call name and causally decisive arguments can define a qualified semantic span. A byte substring matched in a final transcript cannot establish an original token span. The policy support may be sparse while conditioning support retains the full actual prefix; KL can retain full sampled support. If a proposed mask keeps only arguments, measure whether omitting call-name credit changes tool selection. No selected span should extend backward into reasoning merely because the final outcome was bad.

Removing most policy support changes total update influence even with the same denominator. Do not compensate by dividing by the sparse token count. Report selected sampled tokens, effective mean credit, action counts and gradient norms; sweep a declared positive cost weight against measured post-update KL instead. A stronger weight is an intervention to test, not evidence that all existing advantage values were too small. If unsupported residual harm remains, leave it unattributed or use an explicitly declared coarse whole-turn residual cost rather than distributing it to an invented causal mask.

For the first fixture, native negative tool-mistake evidence can select the whole assistant turn, because that evidence is already available. Guard violations require a different scorer projection: record the actual before/after assertion state and causal tool call when the environment can establish it. A final failed guard cannot identify a particular turn. If one message makes both useful and invalid calls and native evidence cannot distinguish their token spans, select the complete turn and report collateral loss of useful credit. A one-mutating-call-per-turn diagnostic is a separately costed collection condition, not evidence that existing multi-call messages had finer attribution.

## Admission is part of the contract

The current SAMPO baseline admits groups through episode reward variance. A local-cost extension cannot silently bypass that rule. It needs a separately named admission rule: retain a complete group when episode comparison credit is informative **or** at least one valid nonzero local-cost contribution exists. Constant-success groups without local costs remain uninformative for this objective.

Count all generated candidates, retained complete groups, ordinary contributions and local-cost contributions separately. Preserve within-population task uniqueness. Retain complete native siblings for preparation even when a contribution selector excludes part of their policy support. Because the new admission may retain constant-failure groups previously rejected, compare at both equal optimizer population budget and reported actual candidate cost; do not promise the same historical rollout allocation after changing admission.

This is not expressible as an innocent scalar-score change to current SAMPO. The accepted semantic-span and resolved-update contracts in canonical `05-apis.md` allow versioned qualified definitions, but existing SAMPO semantics remain protected and backend qualification is still required. A production implementation requires a named objective/admission selection and the appropriate plan.

## Alternatives and why they are not the first choice

| Alternative | Useful property | Unresolved problem |
| --- | --- | --- |
| Terminal guard cost | Improves relative ranking when safe siblings exist | Shared violations cancel; no causal action location |
| Larger turn weight | Strengthens informative anchor comparisons | Singleton term remains zero; large weights can change good-turn signs |
| Mean/std normalization | Amplifies small differences | Can amplify tiny-score noise; no comparison still yields zero |
| Add an uncentered cost to every turn's existing credit | Small implementation delta | Positive episode credit can still override the cost unless its weight becomes very large |
| Clamp evidenced spans to min(A, -kappa*c) | Guarantees negative sign | Keeps arbitrarily large existing negative magnitudes; clipping semantics are less transparent |
| Learned prefix value baseline | Can supply singleton-state comparisons and delayed-return credit | Needs independently trained/calibrated value evidence; existing logs do not establish generalization |
| Merge observation anchors more aggressively | Creates more comparisons | Can compare different actual states and corrupt conditional decisions |
| Replay complete procedures as RL | Reuses rare good behavior | Old trajectories are not fresh on-policy data; requires an explicit off-policy method |

## Retrieval and consolidation need their own experiment

Use a frozen panel with native conditions: procedure retrieved, relevant rows read, decision rule followed, positive goals achieved and guards preserved. A policy-document keyword or successful search call is not proof the correct procedure was obtained or used.

If a complete procedure is sampled rarely, reserve fresh candidate selections for previously observed rare-success tasks across later optimizer populations. Measure whether correct retrieval and full success become more frequent and remain so; candidate replay alone cannot answer that. Constant-success filtering may prevent this from becoming rehearsal. If verified demonstrations are needed, name and qualify a separate supervised or teacher-scored objective; do not label it SAMPO's on-policy improvement.

For delayed retrieval, a bounded same-prefix branching diagnostic can compare retrieval versus mutation from the same actual state. It requires fresh environment rollouts and measured cost. Alternatively, exact required-policy information supplied at the start is a capability diagnostic: it tests reasoning after discovery, while intentionally removing the discovery requirement. Neither intervention is evidence that a static replay learned retrieval.

## Minimal falsifiable sequence

1. Replay the finalized 21 full groups, 114 episodes and 775 turns using disjoint local net-negative native reward support. Verify all 42 negative native-turn scores select their native sampled-turn support and have negative prepared local credit, including the 12 singleton conflicts. Report the unchanged token denominator, affected positive-progress turns and complete-procedure spans. This is a sign/mask fixture only, and the cost support is not a causal guard oracle or a JSON-argument mask.
2. Create a native guard-transition fixture that includes mixed siblings, all-unsafe siblings, a singleton violation, multiple calls in one turn, a restored state and missing/failed assessments. Recover only evidence the environment actually retained. Stop if causal provenance cannot be established.
3. Run a corrected-loss control versus corrected-loss plus the local-cost extension on the same fixed starting checkpoint and declared fresh rollout budget. Keep ordinary sampling and all other settings fixed initially. Record changed admission and realized allocations. In a later comparison, add fresh rare-success practice independently.
4. Assess full conditional success and guard preservation on the frozen panel at several checkpoints, plus positive-goal retention, excessive avoidance, tool-mistake rate, FP16 applied/skipped updates and measured post-update score/KL changes. Do not select a winner solely by a guaranteed prepared-credit sign metric.

Falsifiers: if local-cost signs are correct but guard violations persist, the cost scale, learned representation or interference may dominate. If guard preservation rises while positive goals collapse, the surrogate is too broad or encourages avoidance. If exact required information does not improve conditional success, retrieval is not the primary obstacle. If fixed-panel full success falls reproducibly after rising while local credit is sound, investigate update interference and consolidation. If corrected-loss-only matches the extension, the extra cost objective is unnecessary. These possibilities should remain open.

Sources inspected: current SAMPO advantage builder; current sibling TRL loss source; canonical post-training README and relevant `05-apis.md`/`06-observation-and-lineage.md`; `replay_turn_credit.py`; `build_simulation_report.py`; and `evidence/turn-credit-replay.json`. The turn sample is purposive, current-anchor reconstructed evidence rather than historical applied gradients or a population prevalence estimate.
