# Architecture documents

The architecture is a target MVP revamp derived from the [post-training job lifecycle](../functional/finetuning-lifecycle.md). Existing code is prototype evidence and is not treated as an interface constraint.

| Document | Question answered |
| --- | --- |
| [Platform architecture](../architecture.md) | What is the overall system and MVP boundary? |
| [Layers and ownership](./layers-and-ownership.md) | What belongs in each layer, and what is shared versus job-specific? |
| [Profiles and model variants](./profiles-and-model-variants.md) | How do model profiles compose train/eval/serve defaults, and when does an adapter become a reusable profile? |
| [Evaluation and environments](./evaluation-and-environments.md) | How do general evaluation, domain evaluation, Verifiers, and serving benchmarks relate? |
| [Training and inference](./training-and-inference.md) | What do the reusable train/eval/serve packages expose, and which framework details remain internal? |
| [Observability](./observability.md) | What is recorded for runs, metrics, events, samples, and native artifacts? |
| [Trackio architecture](./trackio.md) | What does Trackio observe, and which authoring and execution concerns explicitly remain outside it? |
| [Lineage and metadata](./lineage-and-metadata.md) | How are models, datasets, environments, runs, observations, and decisions linked? |

Current durable decisions: [ADR 0004](../decisions/0004-lifecycle-driven-mvp-platform.md),
[ADR 0005](../decisions/0005-trackio-verifiers-traces.md), and
[ADR 0006](../decisions/0006-trackio-observation-model.md).

## Revision history

- 2026-07-20: Made train/eval/serve package APIs the cross-project reuse
  boundary and their framework adapters internal.
- 2026-07-20: Made the document set code-first and Trackio purely observational.
- 2026-07-20: Added the dedicated Trackio architecture and observation-model decision.
- 2026-07-19: Added the profiles and model variants architecture concern.
- 2026-07-19: Simplified the target around four engine packages, base configs, published Verifiers environments, and fine-tuning jobs.
- 2026-07-19: Created the target MVP architecture document set.
