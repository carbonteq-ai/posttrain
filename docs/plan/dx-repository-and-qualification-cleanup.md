# Consolidate framework qualification and clean the repository root

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

Source finding: `docs/dx-improvements/v0.2.5/README.md` finding 22. The finding
was verified against tag `v0.2.5`; the implementation target is the current
0.3.x source tree. This plan is self-contained.

## Purpose / Big Picture

The Posttrain repository is both a framework workspace and the framework's own
qualification project. The qualification project is currently split across a
dummy root Python project named `lab`, tracked `.posttrain/` configuration at
the repository root, Python composition in `apps/lab`, temporary launchers in
`scripts/qualification`, and a second Lab-dependent project presented as an
example. Running a project-aware command anywhere in the checkout can therefore
discover the framework-maintainer qualification project, and machine-local
packing has already accumulated multi-gigabyte OCI layouts below the root
`.posttrain/state` directory.

After this plan, the repository root is only the framework workspace and its
owner-level build, documentation, release, and test surfaces. `apps/lab` is one
self-contained Posttrain project: it owns its Python package, tracked
`.posttrain` manifest, qualification overlays and work packages, gate registry,
tests, and maintainer qualification command. The generic `posttrain` CLI still
executes every job; Lab only chooses and evaluates framework release gates.
Temporary launch paths are either promoted into that Lab surface, moved to
their real owner, or deleted after parity. Local state is classified before it
is relocated or pruned, so provider handles and recovery receipts are never
mistaken for cache.

A maintainer can see the result by running:

    uv run posttrain --project-root apps/lab project show
    uv run --package posttrain-lab posttrain-lab qualification list --project-root apps/lab
    uv run --package posttrain-release posttrain-release repository-check --report-only

The first command must identify `foundation-models` below `apps/lab`; the
second must classify every retained qualification work package; the third must
confirm that no root Posttrain project, tracked ignored file, unowned root
surface, or broken maintained documentation link remains.

## Progress

- [x] (2026-08-01) Audited the v0.2.5 tag and current 0.3.x tree; recorded the
      ownership split, root-state footprint, unreferenced work packages,
      temporary harness contract, and documentation drift.
- [x] (2026-08-01) Authored this self-contained cleanup plan and linked it to
      the release-scoped DX critique.
- [x] (2026-08-01) Milestone 1: added an executable, report-only
      repository-ownership contract and a complete Lab qualification-gate
      registry without moving files. The registry now separates 11 active
      release/extended gates from 14 explicit candidate experiments. Focused
      release and Lab tests pass; the built Lab wheel contains the registry
      resource.
- [x] (2026-08-01) Recorded a complete, tested ownership/parity inventory for
      root qualification scripts and the Lab-dependent remote-GPU fixture.
      The inventory makes the pending Milestone 4 moves and removals
      evidence-gated rather than filename-driven.
- [x] (2026-08-01) Removed the unowned, non-runnable `catalog/example`
      fragment. Its only semantic test now builds a temporary project overlay;
      `posttrain init` remains the supported user-facing project example.
- [x] (2026-08-01) Added and tested Lab's self-contained source-snapshot
      declaration (`.` with only its README, `pyproject.toml`, and `src`).
      The full two-job pack proof remains pending the Milestone 3 control-tree
      move and staged release-wheel path.
- [x] (2026-08-01) Added a parameterized temporary nested-Lab proof for the
      data-preparation and managed-evaluation release jobs. It proves Lab
      source and project-control closure are separate and that an explicit
      framework wheelhouse suppresses checkout-source capture. Actual OCI
      materialization remains pending a release-staged environment whose
      installed package identities match the staged wheels.
- [x] (2026-08-01) Made release staging project the authored version into the
      copied workspace lock without resolving dependencies or changing the
      source lock. A full staged tree now passes `uv sync --locked --offline`;
      its lock diff contains exactly the 24 first-party version replacements.
- [ ] Milestone 2: prove that `apps/lab` can be a self-contained nested
      Posttrain project and pack two representative jobs from its own root.
- [x] (2026-08-01) Milestone 3: moved all 37 tracked qualification-control
      files into `apps/lab/.posttrain`, made the root a virtual uv workspace,
      and updated discovery, release checks, tests, and maintainer
      documentation. `posttrain project show` now fails at the repository root
      and succeeds explicitly from `apps/lab`; 203 focused tests, ruff,
      pyright, lock validation, and the release check pass.
- [ ] Milestone 4: consolidate qualification launchers, fixtures, generators,
      and operational documentation under their real owners; delete only paths
      with demonstrated parity.
- [ ] Milestone 5: relocate durable Lab state and prune rebuildable root cache
      through classified, idempotent commands.
- [ ] Milestone 6: pass the full quality/release gates plus real packaged data
      and evaluation jobs from the new Lab project before removing migration
      compatibility.

## Surprises & Discoveries

- Observation: root `.posttrain` is not a copied framework catalog; it is a
  legitimate qualification-project overlay in the wrong physical owner.
  Evidence: the current packaged base contains 46 selections, the root overlay
  contains 43 selections, and their family/id intersection is empty.
- Observation: the split existed in the launched v0.2.5 release rather than
  being introduced by current 0.3.x work.
  Evidence: tag `v0.2.5` contains root project `name = "lab"`, 37 tracked
  `.posttrain` files, `apps/lab`, 15 files under `scripts/qualification`, the
  Lab-dependent `examples/gpu-qualification`, and the tracked `.agents` plan.
- Observation: the repository already declares the qualification scripts
  temporary.
  Evidence: `docs/plan/multi-environment-algorithm-qualification.md` says
  `scripts/qualification/` is a parity harness, not the supported launch
  interface, and requires its removal after the normal CLI has equivalent
  execution and evidence.
- Observation: the source-control footprint is small while the local disk and
  recovery-risk footprint is large.
  Evidence: tracked root `.posttrain` source is under 200 KiB, while the
  current ignored `.posttrain/state` is about 9.4 GiB. Almost all bytes are OCI
  publication layouts, but `state/executions` also contains submission intent,
  provider locators, journals, and reconciliation records and cannot be
  deleted as cache.
- Observation: merely moving `.posttrain` breaks current pack assumptions.
  Evidence: `packages/project/src/posttrain/project/pack_config.py` resolves
  `project_packages` and `source_includes` strictly below the discovered
  project root, while the root project currently selects `apps/lab`. A nested
  Lab project must pack itself as `.` and must not rely on unresolved `../..`
  workspace paths inside its staged source.
- Observation: the AutomationBench environment's immutable source cannot
  yield the declared `automationbench-v1` distribution.
  Evidence: the base catalog requests that distribution from the pinned
  AutomationBench repository root, whose immutable `pyproject.toml` names the
  package `automation-bench` and has no `automationbench_v1` adapter. The
  immutable wheel builder correctly rejects the name mismatch. This does not
  affect the selected `data.prepare` or `eval.domain` nested-pack proofs.
- Observation: the local AutomationBench adapter had a second incompatible
  contract: it declared Python `<3.13` and its lock resolved only Python 3.12,
  while framework online-RL capsules are Python 3.13.
  Evidence: the adapter metadata and lock now validate a 3.12/3.13 range, and
  its test suite and built wheel pass under Python 3.13. The catalog source
  must not change until this adapter revision is published, because immutable
  source selection cannot name an unpushed commit.
- Observation: the local pack command still cannot prove release wheels from
  this editable source checkout.
  Evidence: an explicit 0.3.0 wheelhouse reaches distribution identity
  validation before dataset/environment materialization, but the checkout
  environment has editable packages at `0.0.0`. A release-staged environment
  with installed 0.3.0 distributions is therefore a prerequisite for the
  final OCI proof, not a reason to fall back to copying framework source.
- Observation: rendering staged package metadata alone invalidated the copied
  workspace lock because uv records editable workspace package versions.
  Evidence: before the fix, `uv sync --locked` rejected a freshly staged 0.3.0
  tree whose lock still named first-party packages as 0.0.0. The deterministic
  stage projection changes those 24 versions only; third-party and artifact
  lock bytes remain unchanged and the source lock is byte-identical.
- Observation: documentation drift is part of the ownership problem.
  Evidence: a local-link audit found 12 broken Markdown targets, including
  three maintained documents pointing at the nonexistent
  `.agents/plan/baseline-implementation.md`; `.gitignore` ignores `.agents/`
  while one stale `.agents` plan remains tracked.
- Observation: the root may be a uv workspace without being a Python package.
  Evidence: after removing its `[project]` table, the pinned toolchain accepts
  the workspace lock and sync, while release validation needs to inspect only
  publishable member pyprojects rather than every pyproject in the checkout.

## Decision Log

- Decision: `apps/lab`, not the repository root and not a new top-level
  `qualification/` directory, becomes the qualification project root.
  Rationale: Lab already owns scenario policy, concrete composition, backend
  integration tests, and hardware release gates. A new peer directory would
  add another boundary without an owner, while keeping root `.posttrain` would
  preserve implicit project discovery throughout the framework checkout.
  Date/Author: 2026-08-01 / plan author.
- Decision: the repository root becomes a virtual uv workspace with no
  `[project]` identity; if the installed uv version exposes a blocking defect,
  a temporary `posttrain-workspace` non-package project is allowed only as a
  documented fallback and must not own Posttrain project configuration.
  Rationale: `lab` is already the name of a real workspace member. The root
  owns dependency groups and tool configuration, not an installable or
  executable project.
  Date/Author: 2026-08-01 / plan author.
- Decision: the generic `posttrain` command remains the only execution path;
  `posttrain-lab qualification` selects a gate and evaluates its retained
  evidence without reimplementing pack, submit, provider, or tracking logic.
  Rationale: qualification policy belongs to Lab, but framework execution must
  be exercised through the same public path that consumers use.
  Date/Author: 2026-08-01 / plan author.
- Decision: classify qualification gates in one Lab-owned manifest rather than
  adding release lifecycle fields to generic `WorkPackage` contracts.
  Rationale: release, extended, experimental, and retired status is
  maintainer policy. It must not leak into every consumer's work-package
  schema, and one registry can prove that no YAML file is an unowned orphan.
  Date/Author: 2026-08-01 / plan author.
- Decision: keep repository ownership and Lab gate validation as composable
  commands, rather than making `posttrain-release` import the Lab application.
  Rationale: the root auditor owns framework-checkout inventory; the registry
  and its work-package semantics belong to Lab. A release-app dependency on a
  reference application would reverse that ownership direction and make a
  release-only installation incomplete. CI invokes both commands as the
  framework qualification contract.
  Date/Author: 2026-08-01 / plan author.
- Decision: source moves precede subtraction, and every old launcher is removed
  only after request, provider-state, evidence, and acceptance parity is
  tested.
  Rationale: the temporary scripts contain useful failure characterization;
  deleting them based only on filename references could erase the only check
  for a real release gate.
  Date/Author: 2026-08-01 / plan author.
- Decision: an active Lab gate must be release or extended; experimental work
  is a candidate record with a family, hypothesis, owner, and actionable
  promotion/replacement/retirement/deletion condition.
  Rationale: static acceleration combinations are evidence-seeking candidates,
  not independent framework release requirements. Keeping their work packages
  classified preserves reproducibility without turning historical experiments
  into required Lab work.
  Date/Author: 2026-08-01 / plan author.
- Decision: remove the static Lab `catalog/example` fragment and test project
  overlay composition using a temporary fixture instead.
  Rationale: the fragment was not a runnable user example and had one test
  consumer. The public `posttrain init` project is the maintained example;
  fixture-only catalog data must not occupy a tracked project overlay.
  Date/Author: 2026-08-01 / plan author.
- Decision: an explicit framework wheelhouse is authoritative for job packing
  and suppresses checkout-source auto-discovery.
  Rationale: a wheelhouse is the caller's request to qualify published
  distributions. Capturing importable source from the maintainer checkout
  would make nested-project qualification differ from consumer execution and
  invalidate the source/wheel boundary.
  Date/Author: 2026-08-01 / plan author.
- Decision: release staging may project editable first-party versions in the
  copied `uv.lock`, but it must never invoke dependency resolution.
  Rationale: the release manifest is the one authored version and the source
  lock is the one authored dependency graph. Updating only workspace package
  versions makes the staged metadata internally consistent without introducing
  a second dependency lock or release-time network behavior.
  Date/Author: 2026-08-01 / plan author.
- Decision: this plan consumes, rather than duplicates, the state split and
  `posttrain state migrate` from
  `docs/plan/dx-configuration-authority.md` milestone 2.
  Rationale: that plan already defines `executions/` as retained control and
  `cache/` as disposable materialization. Repository cleanup must not invent a
  second state taxonomy.
  Date/Author: 2026-08-01 / plan author.
- Decision: do not create a `docs/dx-improvements/v0.3.0` review before 0.3.0
  is launched.
  Rationale: DX review directories describe immutable released behavior. This
  split was verified in v0.2.5, so it is finding 22 in that review; a later
  v0.3.0 review must assess what actually shipped.
  Date/Author: 2026-08-01 / plan author.

## Outcomes & Retrospective

Planning outcome: the cleanup is an ownership migration, not a bulk deletion.
The tracked Lab catalog and work packages remain durable source; the root
project identity, parallel launchers, false example, stale agent plan, and
documentation-only root categories are the removable noise. State relocation
is gated on explicit cache/control classification. Implementation outcomes
must be appended here after every milestone, including the exact two packaged
jobs used to qualify the new project root and any path retained because parity
was not achieved.

Milestone 1 outcome (2026-08-01): `posttrain-release repository-check` now
provides a non-blocking inventory of unreviewed root paths, tracked ignored
files, and broken maintained local Markdown links. Lab owns a packaged typed
registry and `posttrain-lab qualification list`; it classifies all 25 current
root work-package YAMLs exactly once. The two checks deliberately remain
separate so a framework-owner release tool does not depend on the reference
qualification application. The audit currently reports legacy migration work;
it is not enabled as a CI failure yet.

Consolidation outcome (2026-08-01): Lab reports 11 active gates and 14
candidate experiments, rather than treating all retained YAMLs as equal
requirements. `apps/lab/qualification-surfaces.toml` gives every current root
qualification script and the remote-GPU fixture an owner, replacement,
public-path parity checklist, and deletion condition. The AutomationBench
adapter now supports Python 3.12/3.13 locally, but its catalog selection stays
blocked until a published Posttrain revision can be pinned to the adapter
subdirectory and an actual job pack succeeds.

Nested-pack outcome (2026-08-01): both retained release jobs now have a
temporary Lab-rooted planning proof. It verifies that `.posttrain` control
data is bundled independently of the Lab source snapshot and that an explicit
framework wheelhouse never falls back to framework checkout source. Final OCI
materialization is intentionally deferred until the same 0.3.0 distributions
are installed in a release-staged qualification environment.

Release-stage outcome (2026-08-01): the complete staged 0.3.0 workspace now
installs with `uv sync --locked --offline`. The copied lock is a deterministic
version projection of the authored source lock, not a re-resolution. This
removes the staged-install blocker for the final local OCI qualification.

Milestone 3 outcome (2026-08-01): all tracked qualification-project source is
now owned by `apps/lab/.posttrain`; root `.posttrain` contains only ignored
machine state. The root has no `lab` package identity or Posttrain source-pack
configuration. The generic CLI deliberately no longer discovers a project
from the framework root, but resolves `foundation-models` with explicit
`--project-root apps/lab`. The release checker now treats the root as a virtual
workspace and continues to validate the 24 publishable package members.

## Context and Orientation

The repository is a uv monorepo. Reusable distributions live under
`packages/*`; executable products and owner tools live under `apps/*`;
independently publishable Verifiers environments live under `environments/*`.
The root `pyproject.toml` currently combines workspace configuration with a
non-package project called `lab`. Its `[tool.posttrain.pack]` selects
`apps/lab`, while `.posttrain/project.toml` at the repository root selects
`posttrain_lab.entry:configure`. This makes the Python application and the
project configuration two halves of one project at different roots.

Posttrain project discovery is implemented in
`packages/catalog/src/posttrain/catalog/project.py`. It searches upward for
`.posttrain/project.toml` unless the CLI receives `--project-root`. A *project
root* is therefore not cosmetic: it owns the tracked control manifest,
catalog overlays, work-package directory, project brief, runtime-value file,
and ignored state path.

Project source selection for immutable job packing is implemented in
`packages/project/src/posttrain/project/pack_config.py`. The selected install
roots and source includes must stay below the project root. The current root
project uses `project_packages = ["apps/lab"]`. After the move,
`apps/lab/pyproject.toml` must instead select itself as `.` and include only its
owned source and qualification resources. Framework packages continue to be
staged through the separate framework-distribution boundary; they must not be
copied into Lab source.

The framework base catalog is package data below
`packages/catalog/src/posttrain/catalog/base/`. It contains shared models,
datasets, environments, inference bindings, targets, workloads, and training
defaults released for all projects. The current root `.posttrain/catalog`
contains only `foundation-models` qualification additions. Moving those files
does not move or fork the base catalog.

`apps/lab/src/posttrain_lab` owns code-defined scenarios and qualification
wrappers. `apps/lab/tests` currently reaches back to root `.posttrain` through
`Path(__file__).resolve().parents[3]`; those tests are an executable sign of
the split. `scripts/qualification` contains provider and evidence probes that
the existing qualification ExecPlan explicitly calls temporary. The public
execution flow is `posttrain job plan|pack|run`; Lab may call public Python
services from `posttrain.project`, but it may not shell out to another CLI or
import private `posttrain_cli` modules.

Machine-local state is not one kind of data. Durable control records include
submit/cancel intent, provider handles, journals, and reconciliation receipts.
They are needed to cancel, recover, or explain a run even when Trackio has
metrics. Build contexts, source snapshots, environment wheels, downloaded
datasets, and OCI publication layouts are caches that can be rebuilt. The
state migration in `docs/plan/dx-configuration-authority.md` must establish
that separation before this plan removes old root state.

The desired repository shape is:

    apps/lab/
      pyproject.toml
      README.md
      src/posttrain_lab/
        qualification/
          gates.toml
      tests/
      .posttrain/
        project.toml
        project.yaml
        catalog/
        datasets/
        work_packages/
        state/                 # ignored, never packaged

    packages/                  # reusable framework distributions
    environments/              # independently released Verifiers packages
    tests/                     # cross-package and external-consumer tests
    docs/                      # product, architecture, operations, plans
    release/                   # release manifest and immutable release inputs
    pyproject.toml             # virtual uv workspace and quality tools only
    uv.lock

`tools/quantization` remains top-level because it is a deliberately isolated
tool environment referenced by catalog recipes and runtime-image locks.
`release` remains top-level because it contains authored release inputs rather
than application code. Executable infrastructure remains in the sibling
`ai-infra` repository; framework-facing operational prose moves from `ops/`
to `docs/operations/`.

## Plan of Work

Milestone 1 makes the intended ownership executable before moving anything.
Add `posttrain-release repository-check` in `apps/release`, backed by a small
pure service and focused tests. The check reads a reviewed root allowlist,
reports tracked files that are also ignored, and validates maintained relative
Markdown links. Lab's own `posttrain-lab qualification list` validates the
gate registry; CI combines the two commands without coupling the release tool
to the Lab application. Do not scan external URLs. Initially report known
violations without enabling the CI gate; enable it only in milestone 4 when the
tree is clean, so this addition does not force an unsafe all-at-once deletion.

Add `apps/lab/src/posttrain_lab/qualification/gates.toml` and typed loading code
beside it. Each entry names a stable gate id, the project-relative work-package
path, job id, tier (`release`, `extended`, or `experimental`), lifecycle state
(`active`, `candidate`, or `retired`), expected job kind, and the acceptance adapter that
interprets retained evidence. Every YAML file under the Lab work-package
directory must be referenced exactly once or listed in an explicit temporary
exclusion carrying an owner and removal condition. Duplicate paths, missing
files, unknown jobs, retired release gates, and unclassified YAML fail tests.
Expose a read-only `posttrain-lab qualification list --project-root PATH`
command. This milestone classifies files; it does not submit jobs.

Milestone 2 is an additive packaging proof. Copy the tracked project config
into a temporary fixture rooted like `apps/lab` and configure
`project_packages = ["."]` with normalized includes for `pyproject.toml`,
`README.md`, `src`, and any Lab-owned qualification resource outside `src`.
Use the public planner and local OCI publication path to prove one
dataset-backed job and one environment/evaluation-backed job. The exact target
work packages should be the smallest current release gates that exercise both
closures; at the time this plan was authored,
`sft_data_prepare_qualification.yaml` and
`qwen2b_eval_qualification.yaml` are the candidates. If their bindings change,
record the replacements and why in the Decision Log.

Inspect `apps/lab/pyproject.toml` during this proof. Its
`automationbench-v1 = { path = "../../environments/automationbench_v1" }`
source is development configuration, but the Lab wheel does not import that
package directly. Remove the base/extra dependency if selected environment
packaging already owns it; otherwise teach the packer's selected dependency
closure to stage the workspace environment by identity. Do not permit `../..`
paths in the immutable project source snapshot and do not include the whole
framework checkout merely to make the proof pass. Acceptance is an installed
actual-job image whose project source root is Lab, whose framework packages
come from the framework wheel boundary, and whose selected environment comes
from its declared environment source.

Milestone 3 performs the tracked ownership migration. Move root `.posttrain`
source into `apps/lab/.posttrain`, update all tests and documentation to use
that location, and replace upward-discovery assumptions with explicit
`--project-root apps/lab` in workspace maintainer commands. Do not add a
special case to generic discovery that recognizes the Posttrain monorepo.
Lab's `pyproject.toml` receives the qualified pack config from milestone 2.
Root `pyproject.toml` loses `[tool.posttrain.pack]`, the `lab` project identity,
and arbitrary runtime dependencies; it retains workspace membership,
dependency groups, index, test, lint, type, coverage, and import-boundary
configuration. Update `apps/release/src/posttrain_release/versioning.py` so
release checks treat a root virtual workspace as tooling rather than requiring
`project.version = "0.0.0"`. Keep all publishable member projects and staged
metadata checks unchanged.

Validate uv's virtual-workspace behavior with the repository's pinned uv
version before committing the root conversion: `uv lock --check`, `uv sync
--all-packages --group dev --locked --python 3.13`, and `uv run pytest` must
still discover every member. Correct `mise.toml` to the repository's actual
Python 3.13 contract in this milestone. From the repository root, `posttrain
project show` without `--project-root` must now fail with the standard
project-not-found error; the same command with `--project-root apps/lab` must
show `foundation-models`. This intentional failure proves implicit framework
checkout discovery is gone.

Milestone 4 consolidates the remaining qualification surfaces by ownership.
Move reusable scenario descriptions, acceptance calculations, and provider-
neutral evidence validation from `scripts/qualification` into
`posttrain_lab.qualification`. The Lab command may construct a public
`JobIntent`, call the public pack/run service, and evaluate a `RunView`; it may
not import private CLI modules, rebuild an `ExecutionRequest` by hand, or copy
framework source files into an ad hoc bundle. For each temporary script, add a
parity test first, route it as follows, then delete the original:

- Algorithm scenarios and run validation become Lab gate registry and
  acceptance adapters.
- Local/dstack runtime, queue, cancellation, model-backward, artifact-consumer,
  and serving probes become classified Lab gates when they test framework
  behavior; any direct-launch implementation disappears after public-path
  parity.
- Deployed Observatory/Ansible topology checks move to `ai-infra`; retain only
  provider-neutral Observatory client assertions in this repository.
- `examples/gpu-qualification` moves to an external-consumer or Lab
  qualification fixture. It must not be called an ordinary example while its
  manifest imports `posttrain_lab`.
- `scripts/materialize_serving_corpus.py` moves beside the serving package
  resource it deterministically generates, with its existing `--check` test.
- `scripts/run_qwen08b_serving_benchmark.py` is removed after its selected work
  package proves equivalent behavior; if it exposes a distinct supported
  capability, promote that capability before deleting the script.
- Markdown-only `ops/` content moves to `docs/operations/`; executable
  deployment remains in `ai-infra`.
- Preserve any still-current decision from `.agents/plan/posttraining-
  platform-refactor.md` in canonical docs or an existing `docs/plan` file,
  repair links, then delete the tracked ignored `.agents` source.

Enable `posttrain-release repository-check` in `.github/workflows/quality.yml`
at the end of this milestone. The check must not ban legitimate new root
owners; changing its allowlist requires an explicit ownership description and
test. It should prevent accidental root dumping, not encode every historical
filename forever.

Milestone 5 relocates state only after
`docs/plan/dx-configuration-authority.md` milestone 2 has separated durable
`executions/` from disposable `cache/`. Extend its idempotent `posttrain state
migrate` with a project-relocation mode that copies and verifies durable
control records from the old root project to `apps/lab`, refuses to proceed
while the controller or provider reports an unresolved active run, and never
deletes the source automatically. Machine/site configuration goes to its named
profile, not the new project tree. Cache relocation is unnecessary: rebuild it
under the new Lab state root on demand.

Add a dry-run cache-prune operation that reports paths, byte counts, reasons,
and whether each entry is protected, active, temporary, or rebuildable. It may
remove only recognized cache/publication layouts after an explicit apply flag;
unknown files and all control receipts fail closed. OCI ingest temporary files
are removable only when no publisher owns them and their bounded stale-age
policy has elapsed. Run dry-run first against the current root state and retain
its compact report in this plan. After durable-record copy and read-back
verification, remove the old state only with explicit maintainer approval;
record the paths and recoverability.

Milestone 6 is the migration and release acceptance gate. Build the exact
framework wheels from the staged release source, pack the same dataset-backed
and evaluation-backed Lab jobs used in milestone 2, and execute them through
the public path on an appropriate target. The goal is not model quality; it is
proof that project source, selected dataset/environment closure, runtime image,
tracking, provider lifecycle, and retained evidence all work after the move.
Inspect both jobs through Trackio/Observatory and reconcile their provider
state before declaring success. Then run the full repository quality ladder,
external-wheel consumer test, release consistency check, documentation link
check, and repository ownership check. Only after these pass may root
compatibility paths and old state be removed.

## Concrete Steps

Work from the repository root `/home/hammad/projects/rl-0.3.0`. At every
milestone start with a clean status and preserve unrelated work:

    git status --short
    uv lock --check

Milestone 1 focused validation:

    uv run pytest apps/release/tests apps/lab/tests -q
    uv run --package posttrain-lab posttrain-lab qualification list --project-root .
    uv run --package posttrain-release posttrain-release repository-check --report-only

Before milestone 3 the Lab command still points at the root project; after the
move the same listing command uses `--project-root apps/lab`. Expected output
contains one row per registered gate and ends with zero unclassified work
packages.

Milestone 2 and 6 package proofs use locally staged 0.3.x framework wheels so
they do not accidentally download an older installed framework:

    uv run posttrain-release check
    qualification_stage=$(mktemp -d /tmp/posttrain-release-qualification.XXXXXX)
    rmdir "$qualification_stage"
    uv run posttrain-release stage "$qualification_stage"
    uv build --directory "$qualification_stage" --all-packages --wheel --out-dir /tmp/posttrain-wheels
    uv run posttrain --project-root apps/lab job plan \
      sft_data_prepare_qualification.yaml --job prepare
    uv run posttrain --project-root apps/lab job pack \
      sft_data_prepare_qualification.yaml --job prepare \
      --local --framework-wheelhouse /tmp/posttrain-wheels

Use the actual job ids declared in the work packages; update this transcript if
`prepare` is not the current id. Repeat plan and pack for
`qwen2b_eval_qualification.yaml`. A successful local pack prints a content
digest and local OCI layout. Inspect the retained package manifest to prove its
`project_manifest` points into the staged Lab project and it contains only the
selected work package and transitive catalog closure.

Milestone 3 discovery proof:

    uv run posttrain project show
    # expected: project-not-found; the repository root is not a Posttrain project

    uv run posttrain --project-root apps/lab project show
    # expected: project_id foundation-models and root .../apps/lab

State migration and pruning must always begin with dry runs. Exact command
names may follow the configuration plan's implementation, but behavior is:

    uv run posttrain --project-root apps/lab state migrate \
      --from-project-root /home/hammad/projects/rl-0.3.0 --dry-run
    uv run posttrain --project-root apps/lab cache prune \
      --state-root /home/hammad/projects/rl-0.3.0/.posttrain/state --dry-run

The migration reports durable records separately and refuses active runs. The
cache command reports rebuildable bytes but performs no deletion without an
explicit apply flag. Do not place a broad `rm -rf .posttrain` command in docs,
tests, or this plan.

The final validation ladder is:

    uv lock --check
    uv sync --all-packages --group dev --locked --python 3.13
    uv run --package posttrain-release posttrain-release check
    uv run --package posttrain-release posttrain-release repository-check
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Record exact pass/skip counts in `Outcomes & Retrospective`. Also run the
installed external-consumer suite and the two real packaged jobs; unit tests
alone do not qualify this ownership migration.

## Validation and Acceptance

The cleanup is accepted only when all of the following behavior is observed:

- From the framework root, implicit project discovery fails; with
  `--project-root apps/lab`, the CLI opens `foundation-models`.
- `apps/lab` contains the only tracked qualification `.posttrain` manifest and
  packs itself without escaping its root or copying the entire framework tree.
- The packaged base catalog remains in `posttrain-catalog`; Lab overlays have
  no accidental promotion into the global catalog and retain selection
  provenance.
- Every retained Lab work package is classified exactly once as an active,
  candidate, or retired release, extended, or experimental gate. The Lab list command and
  CI fail on an orphan YAML file.
- One dataset-backed job and one environment/evaluation-backed job plan, pack,
  execute, reconcile, and expose retained evidence through the public
  `posttrain` path using staged release wheels.
- `posttrain-lab qualification` contains no copied provider implementation,
  private CLI imports, or ad hoc framework source bundler.
- Every deleted `scripts/qualification` entry has a named replacement and a
  parity test or a recorded proof that it was obsolete.
- Root `pyproject.toml` has no `lab` project or Posttrain pack config; uv lock,
  sync, all workspace packages, and release staging still work.
- The state migration retains every execution/control record byte-for-byte or
  through a versioned semantic migration; caches are reported separately and
  no destructive pruning occurs without an explicit apply operation.
- No tracked file is ignored, maintained relative documentation links resolve,
  and the root ownership check passes.
- The final top-level tree contains only reviewed framework owners; `ops`,
  `examples`, root `.posttrain`, root qualification scripts, and tracked
  `.agents` do not survive merely as historical buckets.

## Idempotence and Recovery

Milestones 1 and 2 are additive and safe to repeat. The qualification registry
loader and repository check are pure reads. Local OCI pack output is
content-addressed; a repeated pack either reuses the same digest or exposes a
real input change.

Use version-control moves for tracked `.posttrain` source so history survives.
Do not move the ignored state directory in the same filesystem operation.
During milestone 3, the last known-good root project remains recoverable from
Git until the new Lab project passes both package proofs. If the nested pack
fails, revert only the tracked move and keep the failure evidence in
`Surprises & Discoveries`.

State relocation is copy, verify, switch, and later prune—not move. A repeated
migration must report that destination records already match. A conflicting
record or active run stops the operation without overwriting either side. The
old root state remains the recovery copy until the controller, provider, and
tracking views agree and the maintainer explicitly authorizes removal.

Cache pruning defaults to dry-run, accepts only exact classified paths below a
validated state root, and never follows symlinks. If interrupted, a later
inspection must distinguish completed content-addressed entries from bounded
temporary ingest files and safely resume.

## Artifacts and Notes

Initial audit evidence from the current checkout:

    tracked root .posttrain files: 37
    tracked root .posttrain source: less than 200 KiB
    current ignored .posttrain/state: approximately 9.4 GiB
    current pack/publications: approximately 9.4 GiB
    current execution control records: approximately 44 KiB
    base catalog selections: 46
    qualification overlay selections: 43
    duplicate base/overlay ids: 0
    work-package YAML files with no named source/doc/test reference: 19 of 25
    broken local Markdown links found by initial audit: 12

These counts are orientation, not deletion criteria. Update them when the
implementation begins and attach the qualification-registry classification,
state dry-run report, two package manifests, and final root tree here.

## Interfaces and Dependencies

In `apps/lab/src/posttrain_lab/qualification`, define immutable types equivalent
to:

    @dataclass(frozen=True, slots=True)
    class QualificationGate:
        id: str
        work_package: str
        job_id: str
        tier: Literal["release", "extended", "experimental"]
        state: Literal["active", "retired"]
        job_kind: str
        acceptance: str

    def load_qualification_gates() -> tuple[QualificationGate, ...]: ...
    def validate_qualification_project(
        layout: posttrain.catalog.ProjectLayout,
        gates: tuple[QualificationGate, ...],
    ) -> QualificationInventory: ...

The inventory exposes classified, retired, excluded, and unclassified work
packages. The command returns non-zero for unclassified or invalid entries and
supports JSON through a stable Lab-owned schema for CI.

In `apps/release`, add a framework-owner repository check with a pure service
and thin Typer command:

    @dataclass(frozen=True, slots=True)
    class RepositoryCheck:
        root_entries: tuple[str, ...]
        tracked_ignored: tuple[str, ...]
        broken_doc_links: tuple[str, ...]
        qualification_errors: tuple[str, ...]

    def check_repository(root: Path, *, report_only: bool = False) -> RepositoryCheck: ...

This command depends on Lab only as a subprocess or declarative file validator;
`posttrain-release` must not import `posttrain_lab` as a runtime dependency.
Prefer a schema-neutral TOML validation helper in `apps/release` or invoke the
installed Lab command in CI so application boundaries remain explicit.

Project discovery and packing continue through public types from
`posttrain.catalog` and `posttrain.project`. Do not add monorepo-specific paths
to those reusable packages. The release checker must accept a root
`pyproject.toml` without `[project]` while continuing to render every
publishable member's `0.0.0` source template to the version in
`release/manifest.toml`.

State classification and migration use the interfaces delivered by
`docs/plan/dx-configuration-authority.md`; provider/control safety uses the
durable locators and run view from
`docs/plan/dx-run-lifecycle-and-control.md`. The two package proofs depend on
the materialization/publication work in
`docs/plan/dx-packing-environments-datasets.md`. Record those exact milestone
revisions when implementation starts rather than copying their logic here.

Revision note (2026-08-01): Initial plan created from a v0.2.5 tag audit and
the current 0.3.x repository inspection. It chooses `apps/lab` as the complete
qualification-project root, makes root virtual-workspace conversion and state
classification explicit, and requires two packaged real-job gates before any
compatibility deletion.

Revision note (2026-08-01): Milestone 3 is complete. The living sections now
record the virtual-workspace behavior, the explicit Lab discovery proof, and
the 203-test focused validation. The gate lifecycle contract is corrected to
match the implemented `candidate` state, and the staged-package command now
requires an absent temporary destination so it remains safely repeatable.
