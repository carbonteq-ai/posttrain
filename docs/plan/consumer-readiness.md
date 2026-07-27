# Make the framework usable by a library consumer

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`.

## Purpose / Big Picture

Everything qualified so far was qualified from inside the framework's own
checkout. That is not the position a library consumer is in. A consumer
installs wheels, sets an environment variable, and expects to run a job.

Tested directly, that consumer gets four steps in and stops. Installing,
initialising a project, resolving the catalog, and matching runtime images all
work from the wheel alone. Planning a job does not:

        error: framework source checkout could not be discovered;
        configure registry.framework_source_root

The actual-job image installs framework code from twelve real source
directories, each needing its own `pyproject.toml`. A wheel install has none of
them. So the framework can tell a consumer exactly which images it needs, and
then cannot build the job that would use them.

After this work a developer with no clone can install the framework, configure
a registry, and run a benchmark, then an SFT job, then a GRPO job over the SFT
output — and read the artifact lineage connecting them. The steps they took
become the setup documentation, because they were executed rather than
imagined.

## Progress

- [ ] Milestone 1 — Produce a pre-release on GitHub Releases: build every
  wheel, attach them to a tagged release, and record the exact install command
  including the Git-sourced dependency that cannot resolve implicitly. This is
  what unblocks packing, because the framework then resolves as an ordinary
  project dependency rather than needing a checkout to copy.
- [ ] Milestone 2 — Confirm a consumer can pack and run with no framework
  checkout: the project's resolved lock names the published `posttrain`, the
  actual-job image installs it as a dependency, and `framework_source_root` is
  needed only by framework developers overriding with a working tree.
- [ ] Milestone 3 — Walk the consumer path end to end on real hardware:
  install, configure, `doctor`, then a serving benchmark, an SFT job, and a
  GRPO job whose model seat is the SFT output.
- [ ] Milestone 4 — Verify artifact tracking across that chain: that the SFT
  run's retained adapter is addressable, that GRPO can consume it by reference,
  and that lineage is legible in Observatory.
- [ ] Milestone 5 — Write the setup documentation from the executed steps, and
  record every place the consumer had to know something the framework did not
  tell them.

## Surprises & Discoveries

- Observation: a consumer install fails before it starts, on a transitive
  direct Git dependency.
  Evidence: `uv pip install --find-links <dist> posttrain` refuses with
  "`posttrain-tracking-trackio` ... depends on carbonteq-trackio @
  git+https://github.com/carbonteq-ai/trackio.git@9b0c4af...; add it to your
  dependencies or constraints file". Direct URLs are not resolved implicitly,
  so the documented install command must name that dependency explicitly, or
  the fork must be published somewhere an index can serve.

- Observation: the shipped supply chain works from the consumer's seat, which
  is the part that was expected to be hard.
  Evidence: in a virtual environment containing only wheels, with no framework
  checkout on any parent path, `posttrain doctor` reported Python 3.13, the
  project, `catalog: framework-v1, 47 resolved selections`, `registry:
  registry.lan/carbonteq/posttrain-job (from POSTTRAIN_REGISTRY=...)`, and
  `runtime_images: 5 variants match release 0.1.0`. Catalog data, the published
  image manifest, and registry resolution all came out of package data.

- Observation: packing is the wall, and it is a design assumption rather than a
  missing file.
  Evidence: `_FRAMEWORK_INSTALL_ROOTS` in
  `apps/cli/src/posttrain_cli/execution_planning.py` lists twelve source
  directories (`apps/runtime`, `packages/catalog`, `packages/common`, ...), and
  `_FRAMEWORK_SOURCE_INCLUDES` adds the workspace `pyproject.toml`. The
  actual-job image copies those trees and installs them with
  `--no-build-isolation`. None exist in a wheel install, so
  `_default_framework_source_root()` walks to the filesystem root and raises.

- Observation: a CI job named `external-consumer` already existed and did not
  catch the blocker, because it stops exactly where the blocker starts.
  Evidence: `.github/workflows/quality.yml` runs `pytest -q tests/consumer`,
  and `tests/consumer/test_wheel_project.py` builds a wheelhouse, installs it
  outside the workspace, and asserts
  `test_installed_wheels_discover_external_project_and_compose_catalog`. That
  covers discovery and catalog composition, which work. Nothing in it plans or
  packs a job. `tests/consumer` is also absent from `testpaths`, so a local
  `uv run pytest` never runs it either.
  Consequence: extending that suite to plan and pack is the cheapest way to
  keep this class of gap from reappearing, and it belongs with Milestone 2.

## Decision Log

- Decision: the framework reaches the job image as an ordinary project
  dependency, not through any framework-specific path.
  Rationale: `posttrain init` already scaffolds
  `dependencies = ["posttrain[observatory,trackio,trl]==0.1.0", ...]`, and the
  project's resolved lock already becomes what the actual-job image installs.
  Once those wheels are installable from an index, the framework lands in that
  lock like any other package and needs no staging, no extra image level, and
  no rule change. Source packing exists only because the wheels are not
  published, so the framework falls back to copying itself.
  Consequences: the blocker is not the mechanism but the pre-release. Milestone
  2 therefore comes first and Milestone 1 largely dissolves into it. Framework
  developers keep source packing as an explicit override, which is unchanged
  from today rather than a new mode.

- Decision: REVERSED — an earlier entry decided the job-kind image would carry
  the framework distributions. That is retracted.
  Rationale: it was chosen as the zero-configuration option before noticing the
  project template had already solved the same problem with a plain versioned
  dependency. Compared against that, baking into kind buys no consumer benefit
  while making every framework edit a rebuild of six shared images and
  reversing `validate.py`'s dependency-only rule. A fourth image level between
  kind and actual-job was also considered and rejected for the same reason;
  it is what `containers/posttrain-job-runtime/` was, and it was superseded
  once already.

- Decision: treat this as a blocker for calling the framework consumable,
  rather than as a documentation gap to be papered over with "clone the repo
  first".
  Rationale: the entire runtime image supply chain exists so a consumer does
  not need the framework's checkout. Requiring the checkout for packing
  reintroduces exactly the coupling that work removed, one layer down.

## Outcomes & Retrospective

To be completed as milestones land.

## Context and Orientation

The actual-job image is built from a staged context containing framework
source, project source, resolved configuration, dependency locks, environment
wheels, and materialized datasets. Framework source is snapshotted from
`registry.framework_source_root`, defaulting to a discovered checkout.

The staged framework source is installed inside the image with
`uv pip install --no-build-isolation --no-deps --requirement
locks/code.requirements.txt`, where that lock names local paths under
`sources/framework`. The image also reconstructs a `[tool.uv.workspace]` so
those packages build as workspace members.

The dependency closure is already resolved separately and hash-locked in
`locks/runtime.control.requirements.txt`, so framework source is installed for
its code, not to resolve its dependencies.

## Plan of Work

Milestone 1 is the load-bearing change. The actual-job image should obtain
framework code the same way the consumer did: as built distributions. The
options are to ship framework wheels into the staged context, to install them
from an index, or to keep source packing as an opt-in for framework developers
while making wheels the default. Whichever is chosen, `framework_source_digest`
must keep identifying exactly what went into the image, because job package
identity depends on it.

Milestone 2 produces something installable. `uv build --all-packages` already
emits twenty-two wheels. What is missing is a decision about where they live
for a pre-release and how the Git-sourced tracking dependency is satisfied.

Milestone 3 runs the real thing on the GPU: a serving benchmark, an SFT job,
and then a GRPO job that takes the SFT adapter as its model seat. That last
step is the one that exercises artifact tracking rather than merely producing
artifacts.

Milestone 4 checks that the chain is legible: that the SFT output is
addressable as a catalog reference, that GRPO resolves it, and that Observatory
shows the lineage.

Milestone 5 writes the documentation from what was actually done.

## Validation and Acceptance

A developer with no framework checkout, given only the documented install
command and a registry, can run all three jobs and read the lineage between
them. Every step in the documentation was executed, and any step the consumer
had to infer is recorded as a defect rather than smoothed over in prose.

## Idempotence and Recovery

The consumer environment is a disposable virtual environment and project
directory; recovery is deleting and repeating. No framework state is mutated by
this exercise apart from images and runs published to the configured registry.

## Interfaces and Dependencies

Milestone 1 touches `apps/cli/src/posttrain_cli/execution_planning.py`,
`packages/execution-pack`, and the actual-job Dockerfile. It changes how
framework code enters the image, so it changes `framework_source_digest` inputs
and therefore every job package key once.
