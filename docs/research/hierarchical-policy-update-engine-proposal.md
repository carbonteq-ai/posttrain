# Proposal: a generic policy update engine for Posttrain

2026-10-01 · Design proposal for team review. This proposes capabilities; it does
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

## A concrete example

Episode A has three assistant turns; Episode B has two. Each turn may contain
reasoning steps, a tool call and a final answer. A separate process reward model
can score individual steps. The policy algorithm converts that evidence into
credit and chooses which spans receive direct loss.

    Complete task groups → episodes → assistant turns → tokens
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

The structural hierarchy is group → episode → turn → token. Reasoning steps,
answers and tool calls are annotations over original token spans, with stable
identities and parent links. Annotations can overlap rather than forming one
strictly nested partition. A reasoning step need not equal an assistant turn or
an environment action.

Original sampled-token provenance remains authoritative. System/user messages
and tool results may remain context without becoming sampled policy actions.
Boundary extraction must be versioned, with an explicit policy for malformed
boundaries. Re-tokenizing scorer text must not silently change policy alignment.

## Four independent choices

| Choice | Responsibility |
| --- | --- |
| Logical population | Complete task groups and behavior-policy snapshot used for credit estimation. |
| Objective semantics | Credit, ratio support, masks, clipping derivatives and weighting. |
| Optimizer schedule | Contributions per update, ordering and reuse of the frozen population. |
| GPU execution | Context-preserving packs and accumulated gradients within a scheduled update. |

Scheduling may use episode, turn or token budgets. Oversized records and final
partial minibatches need explicit rules; budgets must not silently truncate turns.
Group admission before credit estimation differs from loss masking afterward.

## Independent masks and statistics

One global training mask cannot represent the intended capabilities.

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

## Process rewards and granular credit

A separate scorer can supply evidence addressed to an episode, turn, reasoning
step or token span. Preserve its model/version, score meaning, alignment,
coverage and validity. Missing evidence is not zero reward.

The estimator must distinguish step quality from incremental reward, a value
estimate or an advantage. A high score for a prefix does not establish that the
last step caused its improvement. The algorithm defines how evidence becomes
rewards, returns and advantages, including credit for later success.

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

I recommend bounded scoring and faithful recomputation, with a complete-graph
reference to check gradients. This is a proposed design, not a performance
result. A sampled-logprob-only interface is too narrow: entropy, distribution
KL and teacher objectives can need additional statistics and derivatives.
Objectives declare those requirements; adapters reject unsupported ones.

Recomputation must preserve context, positions, stochastic state and parameter
version. Detaching a context cache is not a valid default shortcut. Some schedules
repeat prefix work or score unselected spans. Masking does not guarantee
proportional compute savings. Record dependency costs, successful optimizer
updates and FP16 skipped updates. BF16 and FP16 are primary qualification targets.

## Research anchors

The accompanying [research review](selective-policy-objectives-research.md)
records primary sources, ablations and unresolved author-code reconciliation.
These precedents motivate separate capabilities, not one universal recipe.

| Precedent | Capability it motivates |
| --- | --- |
| GRPO/DAPO and GSPO/SAMPO | Distinct token/sequence objectives, reductions and hierarchical credit. |
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

Posttrain owns resolved meaning. TRL and veRL must produce equivalent logical
updates within justified numerical tolerances. Evidence must show mask coverage,
credit provenance, denominators, dependency work, successful updates and reuse.
Packing changes should preserve the objective. An intentional microbatch-dependent
selector must be named explicitly rather than presented as packing invariant.

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
4. Is bounded scoring/recomputation an acceptable cost for full dependencies,
   or is there a better execution design preserving the same semantics?

Agreement should precede implementation and default-recipe selection.
