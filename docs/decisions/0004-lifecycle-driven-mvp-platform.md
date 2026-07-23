# ADR 0004 — Code-first lifecycle platform boundaries


> **STALE — pending reconciliation (2026-07-21).**
> Canonical design: [docs/post-training/](../post-training/README.md). Gap list: [RECONCILIATION.md](../architecture/RECONCILIATION.md).

## Status

Accepted. Supersedes ADRs 0001, 0002, and 0003 as platform architecture guidance.

## Context

The prototype demonstrated TRL training, vLLM serving, Verifiers evaluation,
Trackio logging, and a uv workspace. Its generic YAML profile resolver and
filesystem run records also mixed reusable definitions, job intent, execution,
and observability.

Product discovery requires different teams to improve training, evaluation,
serving, environments, and reports independently. Job authors need normal
Python composition, reusable typed definitions, and the ability to replace an
execution implementation without changing lineage or observability concepts.
The reusable unit across projects must be the complete train, eval, or serve
package API—not a low-level runner object from one framework.
Trackio should record evidence without becoming a workflow engine or definition
registry.

This is an MVP revamp. Existing code and files are not compatibility contracts.

## Decision

- Organize reusable code as `common`, `train`, `eval`, `serve`, and `reports`
  uv packages with isolated dependency lifecycles. Treat `train`, `eval`, and
  `serve` as reusable project-facing products.
- Make jobs importable Python modules containing stable metadata and explicit
  decorated actions. Decorators perform discovery only; actions remain ordinary
  Python and invoke typed package operations.
- Use typed Python values/factories for reusable definitions. Permit TOML/YAML
  only as optional data input to an implementation-owned type, not as a generic
  workflow or inheritance system.
- Keep cross-engine model profiles lightweight. Keep train, eval, and serve
  definitions beside the package that validates and executes them.
- Give `train`, `eval`, and `serve` stable public operation, typed input/result,
  profile, and instrumentation contracts. Keep framework runner/adapter
  protocols internal to their owning package.
- Allow those public operations to run without a lab Job or Trackio. Accept an
  optional host execution/observation context; this lab supplies the
  Trackio-backed implementation.
- Use TRL as the first internal training adapter without making its runner the
  cross-project abstraction. A future trainer is added behind the public
  `train` operations with its own typed config.
- Use Verifiers directly for task data, environments, rewards, metrics, and
  traces. Publish general and domain environments as independent packages.
- Use vLLM and SGLang as independent serve implementations. Keep TurboQuant,
  MTP, kernels, and model-family enablement in `packages/serve` and its typed
  profiles.
- Use Trackio purely as durable observability for runs, resolved snapshots,
  metrics, traces, artifacts, and observed lineage. Do not use it to define
  jobs, profiles, dispatch, scheduling, branching, thresholds, or promotion.
- Use `packages/reports` as the read-only computation and frontend-view boundary
  over Trackio. Do not create another results store.
- Make every execution observed by this lab belong to one code-defined job.
  Direct consumers outside the lab are not required to adopt this hierarchy.
  Allow one job to contain many actions, invocations, run kinds, repetitions,
  checkpoints, and artifact branches.
- Create a new job for a distinct objective, owner, or lifecycle—not for every
  model branch. Derive exact model lineage from immutable artifact edges.
- Create derived model profiles only for descendants deliberately promoted as
  reusable entry points.

## Consequences

- Job code is reviewable, composable, testable, and versioned with Git.
- Other repositories, CLIs, notebooks, and services can reuse `train`, `eval`,
  and `serve` without adopting this lab's Job hierarchy or Trackio deployment.
- Shared profiles ship with the implementation that supports them, so teams can
  publish improvements independently and jobs can import them after upgrading.
- Switching TRL or an inference backend does not change the owning package's
  public operation, job identity, Trackio concepts, or model artifact lineage.
- No universal configuration schema can hide incompatible framework semantics;
  callers select a concrete typed config through the stable package operation.
- Trackio has a clear one-way write boundary and remains queryable across all
  teams without controlling them.
- One job can naturally represent a complete objective with several attempts
  and branches; reports choose explicit comparable populations.
- Heavy dependencies remain isolated from job authoring and from one another.
- The generic profile resolver, YAML job tree, and durable filesystem run
  registry become migration targets rather than interfaces to preserve.

## Alternatives Considered

### Generic YAML jobs and profile inheritance

Rejected because unrelated framework schemas require different validation and
behavior. Normal Python composition is clearer for loops, branching, helper
reuse, and implementation-specific types.

### Expose runners as the reusable API

Rejected because it would make every project assemble framework lifecycle,
checkpoint, tracing, error, and compatibility behavior itself. The owning
package provides that coherent product surface; runner/adapter objects remain
internal seams.

### A universal framework configuration

Rejected because TRL, Torchtune, Verifiers, vLLM, and SGLang expose different
lifecycles and invariants. The platform standardizes identity, execution
context, outputs, and observation—not every option.

### Trackio as orchestration and registry

Rejected because it would make stored observations responsible for source
intent, dispatch, branch policy, and promotion. Trackio is strongest as the
durable evidence layer.

### One job per model branch

Rejected because artifact branching and objective boundaries are different.
Sibling technique branches can answer one job's question, while independent
serving work may deserve another job even without new weights.

### Put every reusable definition in `common`

Rejected because backend-specific profiles must evolve with their adapter,
dependencies, compatibility checks, and tests. `common` contains shared types;
owners publish concrete definitions.

### Build custom model/environment/result registries

Rejected for the MVP. Python packages and the Environment/Model Hubs distribute
definitions and artifacts; Trackio observes runs and lineage; reports provide
discovery views.

## Implementation Notes

- Target architecture: [Post-training platform architecture](../architecture.md).
- Ownership: [Layers and ownership](../architecture/layers-and-ownership.md).
- Profiles: [Profiles and model variants](../architecture/profiles-and-model-variants.md).
- Execution: [Training and inference](../architecture/training-and-inference.md).
- Evaluation: [Evaluation and environments](../architecture/evaluation-and-environments.md).
- Observability: [Trackio architecture](../architecture/trackio.md) and [ADR 0006](./0006-trackio-observation-model.md).
- Define and test direct, Trackio-free public operations before integrating them
  with the lab's Job/action observation context.
- Internal adapter discovery remains package-local and should stay explicit
  until independently distributed implementations require another mechanism.

## Revision History

- 2026-07-20: Made `train`, `eval`, and `serve` the reusable cross-project
  products, moved runner/adapter protocols inside those packages, and made the
  lab Job and Trackio observation context optional host integrations.
- 2026-07-20: Replaced the YAML-centric architecture with code-based jobs,
  typed configs, package-owned reusable profiles, flexible artifact branching,
  and a pure Trackio observability layer; the later revision above clarified
  that packages rather than runners are the public reuse boundary.
- 2026-07-20: Removed the prototype implementation as a compatibility surface and made Trackio the sole durable run evidence store.
- 2026-07-19: Established the lifecycle-driven package, profile, and environment boundaries.
