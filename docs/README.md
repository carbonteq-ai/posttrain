# Docs map

Start here: **[post-training baseline — FROZEN](./post-training/README.md)**
([01](./post-training/01-workflow.md) → [06](./post-training/06-observation-and-lineage.md)) ·
**[Architecture](./architecture.md)** (stale — see [reconciliation](./architecture/RECONCILIATION.md)) ·
**[Thread handoff](./HANDOFF.md)** (next: implementation plan).

Knowledge base for this lab. Runtime is a **uv workspace** with
`packages/{common,data,train,eval,serve,reports}`.

| Area | Purpose |
| --- | --- |
| [overview.md](./overview.md) | Project goal, implementation orientation, and setup |
| [consumer-setup.md](./consumer-setup.md) | Install from the internal index, trust the CA, run local or dstack jobs, hand off trained models between packages |
| [tooling/mise-uv/setup-environment.md](./tooling/mise-uv/setup-environment.md) | Developer checkout setup, GPU profiles, verification, and backend isolation |
| [post-training/01-workflow.md](./post-training/01-workflow.md) | Mental model: framework-shared assets, screen → train → qualify |
| [post-training/02-primitives.md](./post-training/02-primitives.md) | Exact model, data, environment, inference, training, evaluation, workload, and execution selections |
| [post-training/03-work-and-evidence.md](./post-training/03-work-and-evidence.md) | Project, work package, stages, job kinds, framework-shared reuse, views |
| [post-training/04-framework.md](./post-training/04-framework.md) | Developer experience: packages, boundaries, catalog, and composition playthrough |
| [post-training/05-apis.md](./post-training/05-apis.md) | Target public APIs: primitive selections, jobs, config-first composition |
| [post-training/06-observation-and-lineage.md](./post-training/06-observation-and-lineage.md) | Metrics, traces, artifacts, lineage, and observer wiring |
| [post-training/dataset-management.md](./post-training/dataset-management.md) | Proposed dataset authoring, Python builders, reproducible materialization, and package conventions |
| [contributing.md](./contributing.md) | Working on the framework: setup, the validation ladder, boundaries, and what surprises people |
| [publishing.md](./publishing.md) | Cutting a release: coordinated versions, the index, and when runtime images must be rebuilt |
| [release-and-consumption.md](./release-and-consumption.md) | Package publication order, project installation, remote operation, and release gates |
| [architecture.md](./architecture.md) | Target MVP architecture (**stale** pending reconcile) |
| [architecture/RECONCILIATION.md](./architecture/RECONCILIATION.md) | Keep / rewrite / delete gap list vs post-training baseline |
| [architecture/](./architecture/README.md) | Layers, evaluation, training/inference, observability, and lineage |
| [HANDOFF.md](./HANDOFF.md) | Context for a new agent/thread |
| [techniques/](./techniques/) | Post-training methods (SFT, GRPO, …) + recipes + heuristics |
| [tooling/](./tooling/) | Notes **about tools** (TRL, vLLM, Verifiers, mise/uv, hardware) |
| [datasets/](./datasets/) | HF datasets **and** RL task/env cards (prompt, reward, contract) |
| [research/](./research/) | Papers, surveys, cross-cutting summaries |
| [decisions/](./decisions/) | Architecture / stack ADRs |
| [design/](./design/README.md) | Revision-aware product-design references, explorations, and accepted contracts |
| [dx-improvements/](./dx-improvements/README.md) | Release-scoped developer experience critiques and proposed improvements |
| [plan/observatory-product-implementation.md](./plan/observatory-product-implementation.md) | Detailed living implementation plan for the Observatory read product |

## Canonical design sequence

Desired post-training behavior is established before implementation structure.
This prevents prototype code or a convenient API from silently becoming the
contract.

1. **Workflow** — done ([01](./post-training/01-workflow.md))
2. **Primitives** — done ([02](./post-training/02-primitives.md))
3. **Work and evidence** — done ([03](./post-training/03-work-and-evidence.md))
4. **Framework** — done ([04](./post-training/04-framework.md))
5. **APIs** — done ([05](./post-training/05-apis.md))
6. **Observation** — done ([06](./post-training/06-observation-and-lineage.md))
7. **~~Design freeze~~** — **FROZEN 2026-07-21** ([post-training README](./post-training/README.md))
8. **Implementation plans** — current release-scoped work in
   [plan/](./plan/); historical package-boundary decisions in
   [baseline-implementation.md](./plan/baseline-implementation.md)
9. **Architecture reconciliation** — parallel/later ([RECONCILIATION.md](./architecture/RECONCILIATION.md)); do not block plan slices that already match 05/06
10. **Code and validation** — implement plan slices; code is not the contract

Existing code is useful evidence about constraints, but it is not the authority
for the intended workflow during this revamp. If implementation reveals that an
assumption is invalid, **unfreeze** and update the post-training documents
first, then the plan and code. Do not document an accidental implementation
detail as a requirement merely because it already exists.

The numbered post-training documents are the frozen baseline. Architecture
docs remain stale until reconciled.

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
