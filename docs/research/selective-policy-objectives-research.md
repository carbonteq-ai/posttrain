# Research pass: selective and hierarchical policy objectives

2026-10-01. Primary-paper review to inform the generic Posttrain engine proposal.
This is design evidence, not qualification or adoption of these algorithms.
Paper versions below are explicit; released author implementations have not all
been reconciled. No training experiments were run for this review.

## Main finding

A reusable engine needs more than episode/turn batching. Published methods
independently vary the tokens contributing to policy loss, the tokens used for
KL, credit assignment, statistical selection, reward shaping and importance
weight derivatives. Those distinctions should survive GPU packing.

The strongest direct precedents for targeting thinking tokens are answer-span
masking and reasoning-span reward shaping. Entropy-based selection is another
mechanism; high entropy is not a semantic label identifying reasoning text.

## Primary-source comparison

### Beyond the 80/20 Rule: high-entropy token policy gradients

[Paper v1, sections5.1–5.4](https://arxiv.org/html/2506.01939v1).
The DAPO-based objective selects high-entropy tokens through an indicator on
the advantage term. It compares top10/20/50%, all tokens and bottom80%.
Top20% preserves Qwen3-8B performance and improves larger tested models;
bottom80% performs poorly. Reported settings include token clip.2/.28, training
batch512/minibatch32, explicitly16 gradient steps, LR1e-6 and no KL/entropy
loss. The paper describes thresholding within a batch or microbatch.

Engine requirement: declare the threshold population, eligible mask, entropy
source, tie handling and denominator. Computing a fresh quantile in each GPU
pack can change the objective when packing changes. Author-faithful microbatch
selection and packing-invariant logical-scope selection must be distinguished.
Predictive entropy requires distribution statistics, not sampled-logprob
surprisal alone. Token masking does not establish proportional FLOP savings.

### OM-GRPO: outcome rewards with answer-span masking

[Don't Peek at the Answer v1, sections3–4 and Appendix D.2](https://arxiv.org/html/2608.03119v1).
This recent preprint addresses label-free, consensus-reward RL. It retains
answer-derived rewards while masking the final-answer span from direct policy
loss. Appendix D.2 explicitly masks both policy-gradient and KL terms. Component
ablations remove the outcome mask and report deterioration/collapse in the
tested setting. This is a direct precedent for optimizing reasoning regions
without directly reinforcing answer tokens; it is not evidence that all
ground-truth-supervised RL should mask answers.

Engine requirement: independently specified reward source and optimization
span, explicit policy/KL mask binding, and versioned answer-boundary extraction.
Masked direct loss does not freeze shared model parameters or guarantee an
unchanged answer distribution. Malformed/missing boundaries need an explicit
policy, not guessed token positions.

### Discounted RL: reasoning-only reward shaping

[Learning to Reason Efficiently with Discounted RL v3, section4](https://arxiv.org/html/2510.23486v3).
The reasoning indicator counts K reasoning tokens; the return is
gamma^K times correctness reward plus undiscounted formatting reward. Explicit
reasoning tags delimit the region. This mask counts reward-bearing duration;
it is not automatically a policy-loss exclusion. The paper compares selective
reasoning discounting with whole-response discounting and reports that the
latter can harm formatting/accuracy. It also uses a periodically refreshed KL
reference and matched token budgets.

Engine requirement: span-derived statistics and independently transformed
reward components before credit estimation. Discounting reasoning length is
different from SAMPO's discount over environment turns. Preserve reward units,
normalization and component lineage; one generic gamma switch is insufficient.

### Search-R1: preserve retrieval context, exclude retrieved loss

[Paper v1, section3.1 and Table4](https://arxiv.org/html/2503.09516v1).
Search-R1 interleaves model reasoning/search calls with retrieved text and
masks retrieved tokens from optimization. A with/without-mask comparison
supports the masking choice. Retrieved text remains available as context for
subsequent model actions.

Engine requirement: provenance-aware sampled-token eligibility distinct from
attention/context visibility. Reasoning, search queries and answers can be
policy-generated, whereas retrieved documents are environment-generated.
An excluded direct-loss token may still participate in differentiable context
processing. Do not equate loss exclusion with detached activations.

### PRIME: dense process credit plus sparse outcome credit

[Paper v1, sections3.1–3.2 and5.4](https://arxiv.org/html/2502.01456v1).
PRIME derives dense token rewards from an implicit process reward model, updates
that model online using outcome supervision, and combines dense and sparse
returns. It evaluates advantage estimators including GAE and Monte Carlo
variants, selecting leave-one-out in its instantiation.

Engine requirement: token-aligned reward/credit inputs and independently
computed reward streams before aggregation. The reward-model update has its
own state and lifecycle; a configurable token mask alone cannot implement
PRIME. A generic policy engine can consume qualified dense credit without
claiming it owns or already supports the full reward-model training algorithm.

### CISPO: clipped detached weights, retained token gradients

[MiniMax-M1 v1, section3.1](https://arxiv.org/html/2506.13585v1).
CISPO clips importance weights and detaches them, then weights token logprob
gradients. That differs from PPO/GRPO surrogate clipping, which can eliminate
particular policy-gradient contributions. The report discusses16 off-policy
update rounds per generation batch and compares optimization behavior.

Engine requirement: represent weight clipping, loss masking and stop-gradient
semantics separately. A universal clipped-surrogate reducer cannot faithfully
represent every policy optimizer. These results do not supersede ARLArena's
task-specific agentic stability findings.

### NAT: random selection with inclusion-probability correction

[Not All Tokens are Needed v1, sections3–4](https://arxiv.org/html/2603.06619v1).
This preprint uses uniform random token sampling and random prefix cutting.
Its Horvitz–Thompson estimator multiplies selected contributions by1/p_t and
retains the full-sequence denominator, requiring positive inclusion probability
for every token. Prefix selection permits shortening model execution for its
token-local objective. Deterministic permanent prefix exclusion differs from
this estimator.

Engine requirement: record sampling probability, selection seed, correction
weights and original denominator. The unbiasedness claim is conditional on
the specified estimator and mask sampling; it does not transfer automatically
to a nonlinear episode-ratio objective or adaptive parameter-dependent masks.
Context or ratio dependencies can prevent skipping supposedly unselected work.

## Existing implementation evidence

In the inspected campaign TRL checkout at
255aa649636410f0c06b6ecfb53e941b40357ccf,
`trl/trainer/grpo_trainer.py` already contains `top_entropy_quantile` and
`get_high_entropy_mask`. It gathers eligible entropies across processes for
the current invocation, applies the entropy mask to policy loss, then adds KL
after that operation. Thus this path's entropy selection does not automatically
mask KL. Its invocation scope must be reconciled with a logical optimizer
minibatch spanning execution packs. This is evidence about that exact checkout,
not proof that the consumer-pinned runtime has equivalent behavior or that a
new generic schedule can reuse it unchanged.

## Implications for the proposal

Use separate token roles, selectors, credit estimators, policy objectives and
reductions. A selector may target semantic reasoning/answer/tool-call regions,
token statistics or random samples; it declares its scope and whether it
changes the objective or estimates an unchanged objective. Keep context masks,
policy masks, KL masks, ratio masks and normalization masks distinct. A removed
loss contribution does not necessarily remove its score/gradient dependencies.

Selectors can also exclude turns or episodes, but group admission before credit
estimation differs from loss masking after credit estimation. Record the order
explicitly. Statistical masks must be computed at a declared logical scope,
then held consistently during scoring/replay at a fixed model version. Separate
population-frozen masks from masks refreshed each optimizer step; both are
possible algorithm choices with different meanings.

The prior proposal's token-logprob-only derivative interface is too narrow for
all potential objectives. Entropy selection needs distribution statistics;
entropy losses, distribution KL or distillation can require differentiable
distribution outputs. Keep model-statistic requirements and derivative support
extensible. This is a design deduction from the reviewed methods, not a claim
that arbitrary objectives can be composed safely.

Recommended design anchors are GRPO/DAPO for token objectives and reductions,
GSPO/SAMPO for hierarchical sequence control, Search-R1 for context provenance,
OM-GRPO for semantic loss masks, the80/20 method for statistical selection,
and CISPO for alternative derivative conventions. PRIME and NAT expose further
requirements without requiring their entire algorithms in the first release.

## Evidence limits and next decision

This pass establishes prior work for the requested capabilities. Direct
masking/selection ablations exist, but task/model/reward settings differ. No
reviewed paper supplies the complete generic engine proposed for Posttrain.
Released author-code reconciliation remains open for precise boundaries,
denominators, mask refresh timing and native schedule meanings. The engine
proposal should now be revised around these reusable contracts, while algorithm
support is qualified individually. No default mask or research recipe is selected
merely because it fits the abstraction.
