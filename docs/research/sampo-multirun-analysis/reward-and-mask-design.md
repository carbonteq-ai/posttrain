# AutomationBench action rewards and policy masks

Research design, 2026-10-02. No scorer or trainer change has been deployed.

The proposed system scores a change in the benchmark world, retains the action that caused it, and directs policy credit to that action's original sampled tokens. A successful API response is evidence of execution. The benchmark's rules and assertions determine whether execution helped or harmed the requested workflow.

For the conversion task, an accepted upload for an excluded account is harmful even though the tool returned success. For the advertising task, pausing an eligible low-CTR campaign advances a goal; pausing the protected launch campaign breaks a constraint. Reading the correct policy can enable both decisions, but its value needs a separately verified dependency rather than a generic bonus for calling Gmail.

## Scorer evidence

The current pinned turn scorer computes a scalar change in partial credit and subtracts classified tool-mistake penalties. It supports negative progress; those values occur in the recorded population. A scalar turn delta still hides which goal advanced, which guard broke, and whether several actions offset each other.

Retain a before/after vector of assertion truth values and an execution ledger with task, world snapshot, branch, assistant node, call ID, ordered result, affected resource IDs and scorer revision. A goal transition from false to true is positive evidence. A protected assertion transitioning from true to false is harmful evidence, including guards excluded from the positive completion denominator. Count an initial guard violation when the task contract defines it; do not penalize a pre-existing state as if the sampled model caused it. Keep terminal completion as a separate outcome component.

Reward each net goal gain once. A later reversal removes its earned progress. Repeated toggling or rereading must not accumulate credit. Record unverifiable transitions as unavailable instead of assigning zero or inventing an attribution. A score from final trajectory evidence has full-trajectory observation scope; a decision-time state score has current-step scope. The later label may train an earlier action, but future state must not enter that action's policy context.

Check this ledger under the actual discount and credit computation. With gamma below one, a gain followed by an equal loss can leave positive earlier return even when the undiscounted sum is zero. A once-only earned-credit ledger or discount-consistent potential shaping needs explicit terminal boundary conditions and toggle fixtures. For potential shaping, the added term has the form gamma*Phi(next state) minus Phi(current state); its terminal condition must preserve the intended episode outcome. A scalar net-zero assertion delta is insufficient verification.

For policy retrieval, verify that the returned message or document is the task's authoritative procedure. Award a small, once-only prerequisite component only if the subsequent action uses a rule whose evidence is available in that result. This diagnostic requires retained procedure identity and task-specific rule evidence. A task-wide Gmail bonus would reward irrelevant searches. A verified retrieval label cannot stand in for following the policy.

Name that component a verified-used prerequisite: it depends on later execution and can miss useful discovery followed by failed execution. Retain verified acquisition of the authoritative procedure as a separate diagnostic. A separate discovery-reward arm could pay a capped, once-only acquisition bonus without requiring later success, but would be a declared proxy reward whose effects on irrelevant reading, total calls and rule compliance need measurement.

## Attribution and masks

An isolated mutating call with an attributable world transition can receive a call-span assessment. Prefer the sampled function-name and argument tokens that chose the target, value or exception branch. The exact span depends on the retained serializer/token map; a text search over reconstructed JSON is insufficient. Include syntax tokens where excluding them would detach the action from its valid call representation.

If one assistant message makes several calls, retain their order and state snapshots between executions. Attribute a transition to a call only when the ledger supports that association. With only one before/after snapshot for the whole message, use the sampled assistant-turn span and label it as coarse. A diagnostic with one mutating call per turn can improve attribution, but changes execution cost and available behavior, so it is a separate arm.

Intersect every policy mask with original sampled-policy coordinates. Tool output, user prompts, padding and fabricated token positions never receive policy credit. Store the native trace/node/call IDs, half-open spans, projection revision and evidence reference. Overlapping assessments need a declared composition rule; do not count one token twice accidentally. Missing alignment should abstain or reject the fixture.

## Credit and admission

Outcome credit and local harm credit need separate definitions. SAMPO currently adds centered episode credit to centered anchor-return credit. A singleton anchor has no relative turn signal. A guard penalty shared by all siblings disappears after episode centering. Neither problem is repaired by raising the corresponding weight.

A candidate violation objective applies a signed, action-addressed cost on the eligible span with a declared baseline, separately from relative outcome/progress credit. It must not be centered inside an all-violating sibling group if doing so erases the constraint signal. This is a new objective, not an innocuous SAMPO parameter change. Immediate action costs and discounted future violation returns also have different meanings: the former trains the observed harmful action; the latter can assign responsibility to earlier choices. Choose and name the intended scope.

Inspect the final combined advantage, not just the penalty component. Positive outcome credit can still outweigh a negative local term. A sign-preserving safety constraint or a dedicated loss lane can prevent that overwrite, but changes the objective and requires its own comparison. Native tool-mistake penalties are useful fixtures for this mechanism; they are not labels for semantic guard violations.

An all-constant-outcome group may contain identifiable harmful actions. The current outcome-variance filter can discard it. A process-credit admission lane should admit the group only when valid, aligned, nonzero process evidence survives preparation. It needs a bounded budget, declared global weights and separate counts. Mark whether each complete solution and harm span was selected, eligible, cut as surplus, and actually optimizer-used.

Keep the curriculum's bounded outcome/progress evidence separate from signed optimization costs. Passing negative costs into the current controller can invalidate an entire observation group. An affine transformation satisfies a numeric range only within declared bounds and changes the interpretation of its beta reward model. The sampler should observe clearly named goal outcome and usable-credit indicators if it is redesigned.

## Denominators and KL

Masking tokens and increasing their weight are separate changes. With the original episode denominator T, selecting M action tokens reduces the total supported policy mass. Dividing by M instead amplifies each selected token by T/M. A tiny call span could then dominate an update. Specify the population, support and denominator for each policy component; report supported token count, effective mass and component gradient measurements.

First compare full-turn and call-span support at a matched declared policy-credit budget, with a cap on action/span weights. Do not retune LR to conceal a denominator change. Preserve complete sibling populations for any relative credit computation even when only selected spans enter the loss. A mask must not silently alter the centering population.

KL support is a separate choice. Retaining KL on all sampled tokens can constrain unselected reasoning and language while policy credit concentrates on action spans. Restricting KL to those spans changes regularization and belongs in a separate ablation. Loss-time clipping and a ratio of one do not establish bounded movement after the update; measure actor scores on the same native contexts afterward.

## Qualification

Build native fixtures for an eligible action, a successful prohibited action, a goal reversal, an initially protected assertion, repeated reward toggles, unrelated retrieval, verified prerequisite retrieval, mixed parallel calls, a singleton anchor and a constant-outcome group with valid harm evidence. Verify scorer truth, span alignment, combined credit, admission, loss derivative and post-update movement as separate stages.

Use the corrected token-local derivative throughout the reward/mask comparison. Hold the model, precision, adapter, reference policy, rollout budget and task panel fixed. Compare scalar turn credit, state-transition turn masks, and evidence-supported call masks with matched declared contribution budgets. Track complete workflow success, guard violations, positive-goal progress, retrieval followed by rule compliance, coverage, generated/used episodes and tokens, and retention at repeated fixed checkpoints.

The existing research replays can establish bad labels, lost local signals and allocation tradeoffs. They cannot establish that a new scorer discovers missing procedures or that a masked objective improves learning. Those claims require the controlled training comparison after native fixture qualification.
