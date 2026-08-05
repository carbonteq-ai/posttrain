# Docs map

Documentation for the Posttrain framework. The runtime is a Python 3.13 `uv`
workspace with reusable catalog, project, data, train, eval, serve, tracking,
work, execution, environment, jobs, and runtime-image packages plus the CLI,
Lab, Observatory, runtime worker, and release applications.

New here? Read the [top-level README](../README.md) first — it explains what
the framework is and why it exists. Unfamiliar terms are defined in the
[glossary](./glossary.md).

## Pick your door

| You are… | Start with |
| --- | --- |
| **Using Posttrain to train models** (project developer) | [install.md](./install.md) → [getting-started.md](./getting-started.md) |
| **Working on the framework itself** (contributor) | [contributing.md](./contributing.md) → [tooling/mise-uv/setup-environment.md](./tooling/mise-uv/setup-environment.md) |
| **Cutting or auditing a release** (maintainer) | [release-engineering.md](./release-engineering.md) → [publishing.md](./publishing.md) |

To learn the concepts, read the frozen baseline in order:
[01 · Workflow](./post-training/01-workflow.md) →
[02 · Primitives](./post-training/02-primitives.md) →
[03 · Work and evidence](./post-training/03-work-and-evidence.md) →
[04 · Framework](./post-training/04-framework.md) →
[05 · APIs](./post-training/05-apis.md) →
[06 · Observation and lineage](./post-training/06-observation-and-lineage.md).

## Guides

| Document | Purpose |
| --- | --- |
| [install.md](./install.md) | Installing a release: internal index or GitHub wheelhouse, constraints, remote servers. The single source of truth for install commands |
| [getting-started.md](./getting-started.md) | First-day project-developer walkthrough: trust, machine config, first project, local and dstack jobs, model handoff |
| [glossary.md](./glossary.md) | Every term of art, defined once, with links to owning contracts |
| [contributing.md](./contributing.md) | Working on the framework: setup, the validation ladder, boundaries, and what surprises people |
| [remote-gpu-qualification.md](./remote-gpu-qualification.md) | Qualifying a remote GPU machine against a release |
| [tooling/mise-uv/setup-environment.md](./tooling/mise-uv/setup-environment.md) | Developer checkout setup, GPU profiles, verification, and backend isolation |

## Concepts and contracts

| Document | Purpose |
| --- | --- |
| [post-training/README.md](./post-training/README.md) | The frozen product baseline: governance, amendments, and the document set |
| [post-training/01-workflow.md](./post-training/01-workflow.md) | Mental model: framework-shared assets, screen → train → qualify |
| [post-training/02-primitives.md](./post-training/02-primitives.md) | Exact model, data, environment, inference, training, evaluation, workload, and execution selections |
| [post-training/03-work-and-evidence.md](./post-training/03-work-and-evidence.md) | Project, work package, stages, job kinds, framework-shared reuse, views |
| [post-training/04-framework.md](./post-training/04-framework.md) | Package ownership, boundaries, catalog, and composition playthrough |
| [post-training/05-apis.md](./post-training/05-apis.md) | Target public APIs: primitive selections, jobs, config-first composition |
| [post-training/06-observation-and-lineage.md](./post-training/06-observation-and-lineage.md) | Metrics, traces, artifacts, lineage, and observer wiring |
| [post-training/dataset-management.md](./post-training/dataset-management.md) | Dataset authoring, Python builders, reproducible materialization, and package conventions |
| [developer-experience.md](./developer-experience.md) | The DX authority: project-author journey, vocabulary, configuration ownership, golden path |
| [overview.md](./overview.md) | Project goal and implementation orientation |
| [releases/v0.3.md](./releases/v0.3.md) | Developer features, qualification coverage, reliability improvements, and support boundaries across v0.3.0–v0.3.2 |

## Release and operations

| Document | Purpose |
| --- | --- |
| [release-engineering.md](./release-engineering.md) | Candidate/final workflows, artifact graph, registry contract, and remaining gates |
| [publishing.md](./publishing.md) | Cutting a release: coordinated versions, the index, and when runtime images must be rebuilt |
| [architecture/lan-release-runner.md](./architecture/lan-release-runner.md) | Release control plane, isolated LAN runner, one-build promotion flow, and recovery contract |
| [operations/](./operations/) | Service runbooks (dstack, Trackio, object storage, hardware) |

## Reference material

| Area | Purpose |
| --- | --- |
| [techniques/](./techniques/) | Post-training methods (SFT, GRPO, …) + recipes + heuristics |
| [tooling/](./tooling/) | Notes **about tools** (TRL, vLLM, Verifiers, mise/uv, hardware) |
| [datasets/](./datasets/) | HF datasets **and** RL task/env cards (prompt, reward, contract) |
| [research/](./research/) | Papers, surveys, cross-cutting summaries |
| [decisions/](./decisions/) | Architecture / stack ADRs |
| [design/](./design/README.md) | Revision-aware product-design references, explorations, and accepted contracts |

## Internal working material

Everything below is process scratch for maintainers and agents — plans,
critiques, and handoffs. It is **not** developer documentation and may be
stale, superseded, or mid-flight; nothing here overrides the baseline or the
guides above.

| Area | What it is |
| --- | --- |
| [plan/](./plan/) | Release-scoped implementation plans and execution records |
| [dx-improvements/](./dx-improvements/README.md) | Release-scoped developer experience critiques |
| [HANDOFF.md](./HANDOFF.md) | Context handoff for a new agent/thread |
| [developer-experience-audit.md](./developer-experience-audit.md) | Superseded DX audit (historical diagnosis only) |
| [feedback/](./feedback/) | Ad-hoc feedback notes |
| [templates/](./templates/) | ADR / plan / PR authoring templates |
| [architecture.md](./architecture.md) | Target MVP architecture (**stale** pending [reconciliation](./architecture/RECONCILIATION.md)) |

### Canonical design sequence (governance)

Desired post-training behavior is established before implementation structure,
so prototype code or a convenient API cannot silently become the contract:

1. **Workflow → Primitives → Work and evidence → Framework → APIs →
   Observation** — done; **FROZEN 2026-07-21**
   ([post-training README](./post-training/README.md)).
2. **Implementation plans** — current release-scoped work in [plan/](./plan/);
   historical package-boundary decisions in
   [baseline-implementation.md](./plan/baseline-implementation.md).
3. **Architecture reconciliation** — parallel/later
   ([RECONCILIATION.md](./architecture/RECONCILIATION.md)); do not block plan
   slices that already match 05/06.
4. **Code and validation** — implement plan slices; code is not the contract.

Existing code is useful evidence about constraints, but it is not the
authority for the intended workflow. If implementation reveals that an
assumption is invalid, **unfreeze** and update the post-training documents
first, then the plan and code. Do not document an accidental implementation
detail as a requirement merely because it already exists.

## Conventions

- **Setup env** (CUDA, mise, uv workspace) → `tooling/mise-uv/`
- **Source data documentation** → `datasets/<name>/` cards
- **Executable task environments** → versioned Verifiers tasksets; use the same taskset for eval and online RL
- **General/domain programs** → typed eval definitions referencing published Verifiers packages; **serving definitions** → the `serve` package
- **Actual executions and evidence** → Trackio; **intent, behavior, and decisions** → versioned Python jobs and owning packages
