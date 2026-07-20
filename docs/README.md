# Docs map

Start here: **[Project overview](./overview.md)** · **[Post-training job lifecycle](./functional/finetuning-lifecycle.md)** · **[Product capabilities](./functional/overview.md)** · **[Architecture](./architecture.md)** · **[Thread handoff](./HANDOFF.md)**.

Knowledge base for this lab. Runtime is a **uv workspace** with
`packages/{common,train,eval,serve,reports}`.

| Area | Purpose |
| --- | --- |
| [overview.md](./overview.md) | Project goals, architecture, workflows |
| [functional/overview.md](./functional/overview.md) | Product domains, user needs, ownership boundaries, and initial capability scope |
| [functional/finetuning-lifecycle.md](./functional/finetuning-lifecycle.md) | Post-training job stages, required platform capabilities, and reusable work at each stage |
| [architecture.md](./architecture.md) | Target MVP platform architecture and links to concern-specific documents |
| [architecture/](./architecture/README.md) | Layers, evaluation, training/inference, observability, and lineage |
| [HANDOFF.md](./HANDOFF.md) | Context for a new agent/thread |
| [techniques/](./techniques/) | Post-training methods (SFT, GRPO, …) + recipes + heuristics |
| [tooling/](./tooling/) | Notes **about tools** (TRL, vLLM, Verifiers, mise/uv, hardware) |
| [datasets/](./datasets/) | HF datasets **and** RL task/env cards (prompt, reward, contract) |
| [research/](./research/) | Papers, surveys, cross-cutting summaries |
| [decisions/](./decisions/) | Architecture / stack ADRs |

## Flow

```text
research/ + datasets/  →  techniques/*/recipes + heuristics
                              ↓
             code-based jobs + reusable package operations
                              ↓
                Trackio runs, artifacts, and lineage
                              ↓
              promote durable knowledge into profiles / docs
```

## Conventions

- **Setup env** (CUDA, mise, uv workspace) → `tooling/mise-uv/`
- **Source data documentation** → `datasets/<name>/` cards
- **Executable task environments** → versioned Verifiers tasksets; use the same taskset for eval and online RL
- **General/domain programs** → typed eval definitions referencing published Verifiers packages; **serving definitions** → the `serve` package
- **Actual executions and evidence** → Trackio; **intent, behavior, and decisions** → versioned Python jobs and owning packages
