# Architecture documents


> **STALE — pending reconciliation (2026-07-21).**
> Canonical design: [docs/post-training/](../post-training/README.md).
> Do not treat this document as the current product contract. Gap list: [RECONCILIATION.md](./RECONCILIATION.md).
> The explicitly accepted LAN release-runner document is a delivery architecture
> and is not covered by this stale product-architecture warning.

The architecture is a target MVP revamp derived from the post-training baseline:
[01 · Workflow](../post-training/01-workflow.md),
[02 · Primitives](../post-training/02-primitives.md),
[03 · Work and Evidence](../post-training/03-work-and-evidence.md),
[04 · Framework](../post-training/04-framework.md),
[05 · APIs](../post-training/05-apis.md), and
[06 · Observation](../post-training/06-observation-and-lineage.md).
Architecture reconciliation with this baseline is intentionally pending;
existing code remains prototype evidence rather than an interface constraint.

| Document | Question answered |
| --- | --- |
| [Platform architecture](../architecture.md) | What is the overall system and MVP boundary? |
| [Layers and ownership](./layers-and-ownership.md) | What belongs in each layer, and what is shared versus job-specific? |
| [Profiles and model variants](./profiles-and-model-variants.md) | How do model profiles compose train/eval/serve defaults, and when does an adapter become a reusable profile? |
| [Evaluation and environments](./evaluation-and-environments.md) | How do general evaluation, domain evaluation, Verifiers, and serving benchmarks relate? |
| [Proposed evaluation signal interpretation](./proposed-evaluation-signal-interpretation.md) | How should environment-native rewards, metrics, pass rate, task slices, aggregation, and comparison be modeled without reinterpreting traces? |
| [Tool-using environment execution](./tool-using-environment-execution.md) | How do portable tool requirements, packed Verifiers/MCP execution, model protocols, and backend parsers compose without environment-specific framework code? |
| [LAN release runner](./lan-release-runner.md) | How do GitHub approval, a private-network runner, development qualification, stable promotion, and tag-last finalization form one release transaction? |
| [Training and inference](./training-and-inference.md) | What do the reusable train/eval/serve packages expose, and which framework details remain internal? |
| [Observability](./observability.md) | What is recorded for runs, metrics, events, samples, and native artifacts? |
| [Trackio architecture](./trackio.md) | What does Trackio observe, and which authoring and execution concerns explicitly remain outside it? |
| [Lineage and metadata](./lineage-and-metadata.md) | How are models, datasets, environments, runs, observations, and decisions linked? |

Current durable decisions: [ADR 0004](../decisions/0004-lifecycle-driven-mvp-platform.md),
[ADR 0005](../decisions/0005-trackio-verifiers-traces.md), and
[ADR 0006](../decisions/0006-trackio-observation-model.md). Model conversation
ownership is defined by [ADR 0008](../decisions/0008-model-conversation-contracts.md).
Canonical supervised and preference data ownership is defined by
[ADR 0011](../decisions/0011-canonical-posttraining-data.md).

## Revision history

- 2026-08-05: Added the accepted LAN release-runner delivery architecture.
- 2026-07-21: Froze post-training baseline; marked architecture docs stale;
  added [RECONCILIATION.md](./RECONCILIATION.md) gap list (keep/rewrite/delete).
- 2026-07-21: Pointed architecture at the numbered post-training baseline
  (workflow → primitives → work/evidence → framework → APIs → observation);
  reconciliation still pending.
- 2026-07-21: Updated product-document links and recorded that architecture
  reconciliation with the new baseline is pending.
- 2026-07-20: Linked the canonical post-training data package and adapter decision.
- 2026-07-20: Linked the shared model conversation and backend-parser decision.
- 2026-07-20: Made train/eval/serve package APIs the cross-project reuse
  boundary and their framework adapters internal.
- 2026-07-20: Made the document set code-first and Trackio purely observational.
- 2026-07-20: Added the dedicated Trackio architecture and observation-model decision.
- 2026-07-19: Added the profiles and model variants architecture concern.
- 2026-07-19: Simplified the target around four engine packages, base configs, published Verifiers environments, and fine-tuning jobs.
- 2026-07-19: Created the target MVP architecture document set.
