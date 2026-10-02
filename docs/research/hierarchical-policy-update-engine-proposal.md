# Proposal: a general policy update engine for Posttrain

2026-10-02 · Revised after independent critique and mathematical review.
Design proposal for team review. This proposes capabilities; it does
not approve implementation, change a training recipe or qualify a backend.

## The decision

Build one engine that preserves trajectory structure while algorithms choose
where credit comes from, which regions contribute to each loss, and how those
contributions form an optimizer update. GPU packing must preserve those choices.
GSPO and SAMPO are important examples, but the boundary must also cover thinking
tokens, tool context, dense process rewards and other policy objectives.

A developer should be able to express “learn from these reasoning steps, retain
the conversation as context, combine process and outcome credit, and update over
several episodes” without writing a separate training loop.

The center of the design is a resolved specification of one optimizer update
over preserved trajectory evidence. Shared machinery prepares and executes that
update; an explicit, versioned algorithm owns its mathematical meaning. This is
a general engine for supported algorithms, not a universal objective compiler.
Adding an algorithm means declaring and qualifying its contracts, not assuming
every combination of settings is valid.

## A concrete example

Episode A has three assistant turns; Episode B has two. Each turn may contain
reasoning steps, a tool call and a final answer. A separate process reward model
can score individual steps. The policy algorithm converts that evidence into
credit and chooses which spans receive direct loss.

    Task-group memberships → episodes → assistant turns → tokens
                                          ↕
                             Annotated reasoning/answer/tool spans
                                          ↑
                                 External process scores
                                          ↓
                              Credit estimator + objective terms
                                          ↓
    Optimizer minibatch: A1, A2, A3, B1, B2
    GPU packs:          [A1, A2] → [A3, B1] → [B2]
                                          ↓
                       Accumulated gradient → ONE optimizer update

Changing packs must not reset context, recalculate advantages or give every pack
equal weight. Equal episode weighting must not give A extra influence merely
because it has more turns.

## Preserve structure; annotate regions

Preserve native episode → turn → token containment and stable action coordinates.
Prompt groups, anchor-state groups and comparison populations are named
memberships over those records; they may overlap rather than becoming storage
parents. Each credit estimator declares which populations must be complete.
Group-relative methods require complete groups, but that requirement need not
be imposed on every future estimator.

Reasoning steps, answers and tool calls are annotations containing one or more
original token intervals, with stable identities and parent links. They may
overlap. A reasoning step need not equal a turn or an environment action. Native
Verifiers traces remain replay authority.

Original sampled-token provenance remains authoritative. System/user messages
and tool results may remain context without becoming sampled policy actions.
Boundary extraction must be versioned, with an explicit policy for malformed
boundaries. Re-tokenizing scorer text must not silently change policy alignment.

Preserve the actual conditioning view seen by each action, including its template,
positions and branch lineage. A rolling window, summary or truncated tool result
must not become a reconstructed full transcript during replay. Also distinguish
the sampling distribution, frozen old-policy scores, current-policy scores and
KL reference; these roles need not share the same model or scoring runtime.

## What a resolved update contains

| Choice | Responsibility |
| --- | --- |
| Trajectory evidence | Original action identities, actual conditioning views, native trace references and frozen provenance. |
| Credit preparation | Required populations, reward evidence and the estimator producing returns, baselines and advantages. |
| Objective terms | Selected contributions, required statistics, exact values and derivatives, and reductions. |
| Optimizer schedule | Contributions per update, ordering and reuse of the frozen population. |
| Backend execution | Dependency evaluation, replay, accumulation, numerical policy and the optimizer transition. |

These are responsibilities, not a requirement for five public classes. Algorithm
implementations own the supported combinations; users select qualified controls.

Scheduling may use episode, turn or token budgets. Oversized records and final
partial minibatches need explicit rules; budgets must not silently truncate turns.
Group admission before credit estimation differs from loss masking afterward.

Repacking or repartitioning the same scheduled contributions must preserve that
update within justified numerical tolerances. Changing update boundaries, order,
reuse or selector refresh timing changes the training trajectory. Splitting an
episode across two Adam updates is not equivalent to repacking one update.

## Independent masks and statistics

One global training mask cannot represent the intended capabilities.
Context, evidence transformations and objective masks have distinct contracts;
they are not interchangeable switches.

| Contract | Example |
| --- | --- |
| Context visibility | Keep retrieved documents available to later reasoning. |
| Policy-token eligibility | Exclude system/user/tool-result tokens from sampled actions. |
| Policy-loss selection | Apply direct policy loss to reasoning spans. |
| KL and auxiliary-loss selection | Regularize all eligible actions or explicitly choose another region. |
| Ratio support | Calculate a token, turn or episode ratio over declared tokens. |
| Reduction and denominator | Equal episode weight, equal turn weight or global active-token normalization. |
| Reward shaping | Discount reasoning length without implicitly removing those tokens from loss. |

Statistical selectors also declare their population, required statistics, tie
handling and refresh timing. Computing a fresh entropy quantile in each GPU pack
can change the objective when packing changes. Random selection requires explicit
inclusion probabilities and correction weights when estimating the original loss.

Selecting thinking tokens for direct loss does not freeze answer behavior:
parameters are shared, and context or episode-ratio derivatives may couple other
tokens to selected contributions.

Overlapping annotations do not determine their arithmetic. A selector declares
its set operations, such as union, intersection or exclusion. Evidence projection
declares how scores map onto action coordinates. Objective composition declares
whether credit is combined before clipping or separate terms are clipped and
added; those operations need not agree. Preserve algorithm-specific overlap rules.

For example, an equal-episode reduction could be:

\[
L=\frac{1}{|E|}\sum_{e\in E}
  \frac{\sum_{t\in S_e}w_t\ell_t}{d_e}.
\]

Resolve the episode population E, selected tokens S_e, weights w_t and denominator
d_e explicitly. Dividing by selected-token count differs from dividing by original
eligible-token count. Each policy, KL or auxiliary term owns its reduction and
empty-selection policy: reject, omit with declared renormalization, or contribute
zero under the original denominator. GPU packs cannot decide those policies.
Distributed accumulation must reproduce the declared global weighting, including
any gradient averaging applied by the backend; averaging unequal pack means fails.

## Process rewards and granular credit

A separate scorer can supply evidence addressed to an episode, turn, reasoning
step or token span. Preserve its model/version, score meaning, alignment,
coverage and validity. Missing evidence is not zero reward.
Record what the scorer observed—prefix, current step or full trajectory—and its
snapshot and refresh timing. Hindsight evidence must not silently become a
prefix-only value or state-only baseline.

The estimator must distinguish step quality from incremental reward, a value
estimate or an advantage. A high score for a prefix does not establish that the
last step caused its improvement. The algorithm defines how evidence becomes
rewards, returns and advantages, including credit for later success.
Return construction also declares the discount unit and terminal/truncation
boundaries, including any bootstrap. A difference between adjacent quality
scores is not automatically an incremental reward.

For three scored reasoning steps followed by a verified outcome, a declared
estimator could combine process and outcome streams, calculate step advantages,
then project them onto each step's original eligible tokens. Component weights,
normalization and baseline scope must be explicit. Copying a scalar to every
token without a declared reduction can make longer steps dominate.

Training consumes a neutral evidence contract; composition binds external
inference. Reusable training code must not import a serving backend. Online
reward-model training has a separate lifecycle. Accepting process scores alone
does not constitute implementation of a full algorithm such as PRIME.

## Algorithms retain their identities

Ratio scope, credit scope and weighting remain independent. Local turn credit
can use an episode ratio; a turn ratio is another objective. PPO-style surrogate
clipping and CISPO's clipped detached importance weights have different gradients.

Every objective declares both ratio value support and gradient support. A shared
episode ratio with a fully coupled derivative can aggregate opposing local
advantages; the same ratio value with locally routed derivatives can preserve
their separate signals. Loss values and clipping decisions alone cannot establish
equivalence. [GSPO-token, equations 13–17](https://arxiv.org/html/2507.18071v2#S4.SS3)
explicitly uses stop-gradient operations for this distinction.

For fixed advantages +1 and −1 with inactive clipping, a shared differentiable
ratio s gives J=(s−s)/2 and zero gradient. A GSPO-token-style construction has the
same forward value but gradient s(∇log p_1−∇log p_2)/2. This is a contract example,
not evidence of an error in the current implementation. Shared sequence credit
intentionally has different semantics. Specify the ratio formula as well: GSPO's
exponentiated mean log-ratio differs from a full sequence likelihood-product ratio.

Each versioned algorithm definition declares required model outputs, credit
estimator, supported selectors, ratio and derivative convention, normalization
and schedule constraints. Unsupported combinations must fail explicitly.
A common engine does not make every combination mathematically valid.

Canonical Posttrain SAMPO currently uses a trajectory-wide ratio. Author-style
turn objectives require a separately named selection. Migration must preserve
existing selection meanings.

## Dependencies determine execution

Selecting some turns or reasoning spans can still require scores and gradients
from other parts of the episode. Obtain the complete dependency set with weights
fixed until the update completes. Behavior scores remain frozen across reuse;
current scores refresh after successful updates.

Bounded scoring and faithful recomputation are candidate execution strategies,
with a complete-graph reference to check gradients. They are not a guaranteed
execution path for every supported objective. A sampled-logprob-only interface
is too narrow: entropy, distribution
KL and teacher objectives can need additional statistics and derivatives.
Objectives declare those requirements; adapters reject unsupported ones.

Recomputation must preserve context, positions, stochastic state and parameter
version. Detaching a context cache is not a valid default shortcut. Some schedules
repeat prefix work or score unselected spans. Masking does not guarantee
proportional compute savings. Record dependency costs, successful optimizer
updates and FP16 skipped updates. BF16 and FP16 are primary qualification targets.

For an objective factoring through model statistics q_i, first-order execution
can evaluate each statistic's objective derivative at fixed parameters, then
replay model computation to accumulate the corresponding vector–Jacobian products.
Direct parameter-dependent terms require explicit handling. The objective's
declared stop-gradient rules remain part of that calculation.

Limiting retained graphs does not bound one long context or a full-vocabulary
statistic cache. Adapters declare supported statistics, replay behavior and
capacity. Oversized dependencies require a qualified checkpointing, offload or
sharding path, or rejection before the update. Stateful or stochastic model
execution requires faithful replay; deterministic MoE routing is not inherently
unsupported. A saved seed alone does not establish replay equivalence.

## Research anchors

The accompanying [research review](selective-policy-objectives-research.md)
records primary sources, ablations and unresolved author-code reconciliation.
These precedents motivate separate capabilities, not one universal recipe.

| Precedent | Capability it motivates |
| --- | --- |
| GRPO/DAPO and GSPO/SAMPO | Distinct token/sequence objectives, reductions and hierarchical credit. |
| [GSPO-token](https://arxiv.org/html/2507.18071v2#S4.SS3) | Sequence-wide ratio values with token-local derivative routing. |
| [Search-R1](https://arxiv.org/html/2503.09516v1) | Retrieved context excluded from direct policy loss. |
| [OM-GRPO](https://arxiv.org/html/2608.03119v1) | Answer-derived rewards with reasoning-region policy/KL loss. |
| [Beyond the 80/20 Rule](https://arxiv.org/html/2506.01939v1) | Statistical token selection with explicit threshold scope. |
| [Discounted RL](https://arxiv.org/html/2510.23486v3) | Region-specific reward shaping distinct from loss masks. |
| [PRIME](https://arxiv.org/html/2502.01456v1) | Dense process evidence combined with sparse outcome credit. |
| [CISPO](https://arxiv.org/html/2506.13585v1) | Importance-weight clipping and stop-gradient semantics. |
| [NAT](https://arxiv.org/html/2603.06619v1) | Token sampling with inclusion-probability correction. |

Recommended profiles should follow research and relevant ablations. Small-card
checks establish mathematical fidelity and backend agreement; they cannot rank
large-scale training recipes.

## Scope and acceptance

The proposal covers annotations, objective contracts, credit inputs, scheduling
and execution. It does not commit to implementing every cited algorithm in the
first release. Preserve existing objectives, then qualify additions individually.

Initial scope is first-order updates of an autoregressive text policy over
admitted trajectory evidence. External step scores are credit inputs. Online
reward-model or critic training lifecycles, arbitrary differentiable simulators,
higher-order objectives and asynchronous replay correction are outside the initial
scope. Critic-based PPO, paired preferences and teacher-distribution objectives
are useful design checks; citing them does not claim implementation support.

Posttrain owns resolved meaning. TRL and veRL must produce equivalent logical
updates within justified numerical tolerances. Evidence must show mask coverage,
credit provenance, denominators, dependency work, successful updates and reuse.
Packing changes should preserve the objective. An intentional microbatch-dependent
selector must be named explicitly rather than presented as packing invariant.

Qualification compares matched fixed evidence and resolved updates: unscaled
gradients first, then parameter and optimizer-state transitions. Include mixed-sign
local credit, unequal lengths, overlapping and empty spans, independent policy/KL
reductions, non-unit ratios and active clipping, multiple pack layouts and unequal
distributed partitions. Compare whole-graph and replay execution where supported.
FP16 overflow must not partially update parameters; consumption/retry and schedule
advancement are explicit. Numerical policy names precision for sensitive statistics
and accumulation in addition to the model's BF16/FP16 dtype.

This proposal does not amend the frozen product baseline. Adoption of new public
span/credit/objective contracts requires a recorded decision and narrow canonical
amendments before implementation. Reconcile the earlier implementation plan after
agreement on the proposal.

## Feedback requested

1. Does the span/evidence model cover thinking steps, tool use and future process
   reward models? Which concrete use case would it fail to express?
2. Are masks, ratio scope, credit and denominators explicit enough to preserve
   algorithm meaning across packing and backends?
3. Which objective definitions should we qualify first? Which combinations should
   initially be rejected?
4. Which objectives justify scoring/recomputation, and what capacity or backend
   limitations should constrain the initial execution paths?

Agreement should precede implementation and default-recipe selection.
