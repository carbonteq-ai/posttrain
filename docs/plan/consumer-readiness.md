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

- [x] Milestone 1 — Produce a pre-release: superseded by an internal package
  index. `pypi.lan` (devpi) serves the framework and mirrors PyPI, so the
  framework resolves as an ordinary dependency without a GitHub wheelhouse.
  Published through 0.1.12; the maintained forks are constrained explicitly
  because uv does not resolve a transitive direct URL implicitly.
  ORIGINAL: Produce a pre-release on GitHub Releases: build every
  wheel, attach them to a tagged release, and record the exact install command
  including the Git-sourced dependency that cannot resolve implicitly. This is
  what unblocks packing, because the framework then resolves as an ordinary
  project dependency rather than needing a checkout to copy.
- [x] Milestone 2 — Confirmed. From `/home/hammad/devsim`, with no framework
  checkout on any parent path, a wheel-installed developer planned, packed,
  published, and ran a job. Framework code reaches the image as staged wheels
  rather than as a copied source tree.
  ORIGINAL: Confirm a consumer can pack and run with no framework
  checkout: the project's resolved lock names the published `posttrain`, the
  actual-job image installs it as a dependency, and `framework_source_root` is
  needed only by framework developers overriding with a working tree.
- [~] Milestone 3 — Walked end to end on the RTX 4090 for two jobs: SFT
  (`54346a58`, succeeded, 3 retained artifacts) and GRPO (`bdcb9ed4`,
  succeeded, 4 retained artifacts). Both reconciled `consistent` with no
  missing roles. The serving benchmark and the SFT-seeded GRPO chain are not
  yet done.
  ORIGINAL: Walk the consumer path end to end on real hardware:
  install, configure, `doctor`, then a serving benchmark, an SFT job, and a
  GRPO job whose model seat is the SFT output.
- [~] Milestone 4 — Artifact tracking verified for each job independently:
  both runs' adapters, recovery checkpoints, and summaries are addressable on
  `trackio.lan` and join to provider state. Consuming the SFT adapter as the
  GRPO model seat is not yet done, so cross-job lineage is unverified.
  ORIGINAL: Verify artifact tracking across that chain: that the SFT
  run's retained adapter is addressable, that GRPO can consume it by reference,
  and that lineage is legible in Observatory.
- [x] Milestone 5 — `docs/consumer-setup.md` is written from the steps that
  were executed, and the places a consumer had to know something the framework
  did not tell them are recorded under Surprises & Discoveries.

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

- Observation: the framework cannot resolve as an ordinary dependency without a
  package index, so a GitHub Releases pre-release does not reach that design on
  its own.
  Evidence: environment dependencies are resolved by
  `ImmutableEnvironmentDependencyCompiler` through `uv pip compile`, which
  queries a package index. A wheelhouse tarball attached to a GitHub release is
  not an index, so `posttrain==0.1.0` declared by a scaffolded project has
  nothing to resolve against at pack time. The earlier reversal to "the
  framework is just a dependency" is therefore correct in shape but blocked in
  practice until the wheels sit on PyPI or a private index.
  Consequence: with GitHub Releases as the channel, the framework must be
  staged into the job context from a wheelhouse the consumer already has, which
  is the same mechanism selected environments already use. Awkward compared to
  an index, and workable without one.

- Observation: the interpreter bump left the control dependency lock claiming
  Python 3.12 in three places, one of them a contract assertion.
  Evidence: `execution_pack/service.py` wrote `python_version="3.12"` into the
  control `RuntimeDependencyLock` and into its resolution digest, and
  `execution/job_package.py` asserted `control.python_version != "3.12"` for
  veRL capsules. The images run 3.13.12 and `verl-py313/profile.toml` already
  declares `control_python = "3.13.12"`, so the manifest described an
  interpreter that no longer existed. The resolution digest feeds job package
  identity, so this was recorded identity rather than a stray comment.

- Observation: the actual-job image never resolves the framework's
  dependencies, so the fix is far smaller than "make posttrain resolve as a
  dependency".
  Evidence: a retained `locks/runtime.control.requirements.txt` holds 103
  requirements and contains `datasets` but not `typer`, `pydantic`, `trl`,
  `torch`, or `verifiers`. That lock is compiled from the selected
  environments' dependencies alone, constrained by the workspace lock with
  kind-provided packages excluded. The framework's own dependencies and the
  entire ML stack are already installed in the job-kind image.
  Consequence: the actual-job image needs framework *code* and nothing else.
  It does not need resolution, an index at build time, or a change to the
  runtime lock. Installing the framework distributions with `--no-deps` is
  sufficient, which is the same shape as the environment wheels already staged
  beside them.

- Observation: eleven defects stood between a wheel-installed developer and a
  finished job, and every one of them was invisible from a checkout.
  Evidence, in the order they were hit:
  1. `posttrain-runtime` provides the image's ENTRYPOINT but no consumer
     installs it, so the image was built without its own entry point.
  2. Framework packages declared each other by bare name, so upgrading
     `posttrain` left ten siblings behind at an older version.
  3. `uv pip download` does not exist, so obtaining framework wheels failed.
  4. pip reads `PIP_INDEX_URL` while the documented setup sets `UV_INDEX_URL`,
     so the private index was ignored.
  5. Build definitions ship as package data, and buildx refuses to read
     outside the project without an explicit filesystem entitlement.
  6. `sources/framework` was digested unconditionally in three separate
     places: the image definition, the pack service, and the job runtime.
  7. `release/github-constraints.txt` omitted `trl` and `verifiers`, so the
     documented install command could not resolve.
  8. Writing `execution.toml` for the local provider's hostname silently
     discarded `POSTTRAIN_REGISTRY`.
  9. A failed submission reported only that its outcome was unresolved and to
     retry; retrying reported the same, naming nothing to fix.
  10. A run that died before opening a tracking run held its machine's
      admission placement forever, and cancel, cleanup, and reconcile each
      refused for a different reason.
  11. The Trackio endpoint was recorded only from the execution environment
      file, so a run configured through the shell wrote its evidence to the
      remote server and was then looked for in a local one.
  Consequence: none of these are visible to the framework's own test suite,
  because it runs from a checkout where the source tree exists, the packages
  are all one version, and the definitions are inside the project.

- Observation: two settings are traps whose failures surface far from their
  cause, and both remain traps after this work.
  Evidence: `providers.local.trust_bundle` becomes the container's
  `SSL_CERT_FILE`, replacing the image's certificate authorities instead of
  adding to them. Supplying only the internal CA made every public TLS
  connection fail, which surfaced as `Can't load the configuration of
  'Qwen/Qwen3.5-2B'` rather than as a trust problem. It must be the complete
  set of authorities the job should trust. Separately, `posttrain job run`
  requires `providers.local.canonical_hostname`, which no template writes and
  no documentation mentions.

- Observation: the framework's verification refused to publish an image whose
  staged contents did not match its manifest, and was right to.
  Evidence: `posttrain job pack` intermittently failed with `package manifest
  key differs from PACKAGE_KEY`. The staged context was proven correct three
  ways: the directory name, the passed variable, and the key derived from the
  context's own `package.json` all agreed, and deriving it again inside the
  kind image with the same framework version reproduced it exactly. The same
  pack then succeeded unchanged on retry, and pruning the buildx cache also
  cleared it, so the image was reading a stale layer.
  Consequence: this is a BuildKit cache defect, not a framework one, and the
  framework's check is what caught it. It is a real consumer cost even so,
  because the message describes an identity mismatch rather than a stale
  build, and the remedy is a retry that nothing suggests.

- Observation: the internal PyPI mirror silently served a stale project list.
  Evidence: `aiolimiter`, `chardet`, and `zipp` all returned 404 from
  `pypi.lan` while `pypi.org` served them. devpi logged `serving stale
  projects for 'pypi', getting data timed out after 5 seconds` before each,
  because refreshing the mirror downloads PyPI's entire project list and its
  default `--request-timeout` is five seconds. Raising it to sixty fixed all
  three.
  Consequence: the failure reads as a missing package, with nothing pointing
  at the mirror. Any index this framework is consumed from needs the same
  treatment.

## Decision Log

- Decision: when no framework checkout is configured, stage the framework
  distributions as wheels into the job context and install them with
  `--no-deps --require-hashes`, exactly as selected environments are handled.
  Rationale: the consumer already installed those distributions from
  `pypi.lan`, so the wheels are obtainable at pack time on the machine that is
  packing. Staging them keeps the image build free of index access and network
  policy, and makes identity a digest over real bytes rather than a version
  string that an index could later re-resolve.
  Consequences: `framework_source_digest` becomes a digest over the staged
  framework wheel set rather than over a source tree. `plan_job_pack` already
  accepts that value as an opaque string, so the plan contract is unchanged.
  `code.requirements.txt` names the staged wheels instead of
  `./sources/framework/...`, and `sources/framework` is omitted entirely.
  Framework developers keep source packing through `framework_source_root`,
  which stays the override it already is.


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
