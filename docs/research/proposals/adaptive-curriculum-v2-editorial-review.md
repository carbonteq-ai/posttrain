# Why v2 loses the reader, and what the rewrite changes

Editorial review · 14 September 2026

The main problem is the order in which v2 asks the reader to understand things. It contains strong examples and careful distinctions, but its main reading path follows the needs of a technical specification. A reader trying to understand the proposal must hold those details in memory while waiting for the central mechanism to come together.

This assessment compares the [v2 specification](configurable-curriculum-and-data-preparation-v2.md) and its generated report with Alex Zhang and Omar Khattab's [Language model harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/). It is an editorial judgment, not a user study or an independent validation of either research claim.

## What the blog does well

The blog states its thesis early, introduces an experimental result before the detailed machinery, and returns to the same explanation across its examples. Its major headings express claims, so scanning them reconstructs an argument. The progression from motivation to mechanism to experiments gives each technical section a purpose the reader already understands. Figures and their surrounding paragraphs explain what to notice and why it matters. [Source article](https://alexzhang13.github.io/blog/2026/harness/)

These are strengths of sequencing rather than proof that the language is uniformly simple. The equivalence-relation discussion is dense, and some broad architectural claims are more confident than their surrounding qualifications. Its research results also provide a narrative advantage our proposal does not yet have. We should borrow its explanatory order without borrowing its certainty. [Source article](https://alexzhang13.github.io/blog/2026/harness/)

## Where the original report breaks the chain

### 1. The opening creates a question, then postpones its answer

Section 1's arithmetic example works: a rising aggregate score can hide stalled skills. Fixed class balancing then appears as a plausible first response. That naturally leads to the reader's next question: how will adaptive selection improve on balancing?

Instead of completing that answer, Section 2 introduces hierarchy and compound class identifiers. Section 3 derives the GRPO objective. Sections 4 and 5 introduce variance decomposition, observation windows, reference estimates, progress intervals, and boundary-aware binomial intervals. The selection rule arrives in Section 7.

Using simple whitespace counts on the Markdown source, roughly 3,500 words precede Section 7 and 5,600 precede the worked examples in Section 9. These are approximate source counts, including notation, tables, and markup. The original does contain examples earlier; the problem is that the end-to-end selection example is delayed.

**Editorial consequence:** readers understand isolated pieces before they understand why those pieces are needed. The rewrite puts one ten-problem request immediately after the minimum explanation of grouped reward contrast.

### 2. Too many adjacent concepts sound like the selection objective

V2 discusses success, variance, informativeness, progress, forgetting, uncertainty, coverage, and freshness. Each matters, but they do different jobs. In particular, the base policy predicts mixed groups; progress measures whether the resulting practice helps; progress-based scheduling is a separate alternative.

The original eventually says this explicitly in Section 5.3. Before that clarification, a reader can reasonably assume that all of these measurements feed one combined adaptive score.

**Editorial consequence:** introduce the allocation signal once, name its limitation immediately, and introduce the other measurements when their purpose arises. The rewrite says that contrast is a proxy, then explains comparable evaluation as the way to test that proxy.

### 3. Headings often identify storage locations for information

“Evidence for curriculum decisions,” “Profile settings,” and “Curriculum manifest” identify subjects. They do little to carry a causal argument. A reader returning after an interruption must reconstruct how each subject relates to the proposal.

The revised headings carry the explanation: “For GRPO, different answers create a learning signal,” “A class makes a first guess; the problem can prove it wrong,” and “The sampler must earn its complexity through better learning.” Each heading states the point its section will establish.

This does not make topic headings bad. They remain appropriate in the technical reference, where readers are looking up a formula or a contract rather than following an argument.

### 4. The strongest example is buried

Section 9.2's A17 example shows the proposal in miniature. Arithmetic looks mostly solved, but discovery still selects an exception. Its mixed outcomes correct its own forecast. Later practice responds to those results.

That example makes class prediction, exploration, task-specific evidence, and cross-step repetition understandable together. In the original, readers encounter it after the formulas defining those mechanisms. The rewrite brings it into the core explanation and uses the two-class forecast slider directly before it.

### 5. Precision arrives before the reader has a reason to want it

For example, “The class floor is conditional” is compact and mathematically appropriate once the eligible-pool sampling law has been introduced. A first-time reader needs the operational meaning first: a class can receive discovery only while it has unseen problems available.

Similarly, “A task leaves the unseen pool when its selection is committed” is necessary accounting language. In the main explanation, the simpler concept comes first: unseen means never previously selected during this run. The technical reference retains commit timing, execution failure, retries, and recovery semantics.

The problem is not merely long words. It is asking the reader to interpret a compressed abstraction before showing the decision it governs.

### 6. Essential safeguards and optional extensions compete for attention

Within-step uniqueness, discovery accounting, and class coverage are central to the proposed selector. Teacher generation, efficiency rewards, algorithm transfer, and weighted updates are additional experiments. Their similar section prominence makes the report feel like several proposals joined together.

The main reading path now establishes the sampler, its correction mechanisms, and its evaluation before discussing extensions. It keeps teacher enrichment optional, as the current v2 specifies. Earlier proposal history is not used to override that decision.

### 7. The interactive panel asks readers to learn another vocabulary

The original player introduction explains lane shares, tile colours, reward bars, missing evidence, histories, controls, replay, and retention. This is useful documentation but a substantial interruption to the argument.

The rewritten report first states what to inspect: one ordinary task, a useful exception in a mostly solved class, and a noise case where contrast fails to produce learning. The full player is expandable. It remains embedded and usable, while a reader can finish the argument without operating it.

## What changed

The report now has an approximately 2,600-word narrative, eight explanatory sections, and a technical-reference entry. It follows this progression:

1. Fixed mixtures can give poorly matched practice.
2. GRPO needs differences among attempts at the same problem.
3. The selector combines familiar practice, discovery, and reassessment.
4. Class forecasts help with unseen problems, and task outcomes correct them.
5. Fresh evidence and persistent accounting prevent stale or misleading decisions.
6. Matched-cost, held-out comparisons decide whether this helps learning.
7. The simulation exposes decisions and failure cases under stated assumptions.
8. Teacher enrichment and changes to the learning rule remain later experiments.

The complete formal specification remains embedded in an expandable reference. Its source Markdown is unchanged. This is a rewrite of the report's reading path, not a simplification of the mathematical contract. The rendering script reads the new narrative and the original specification so neither needs to be duplicated by hand.

The key editorial principle is to give each detail a reason to appear. Readers should reach a formula, caveat, or configuration choice already knowing which question it answers.
