# Pack what the project actually is: environments, datasets, and closure

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

Source findings: `docs/dx-improvements/v0.2.5/README.md` findings 7, 8, 13,
14, 15, 16, 17, and the production incident recorded in
`docs/feedback/verifiers-environment-data-packaging.md`. This plan is
self-contained.

## Purpose / Big Picture

A posttrain job runs inside an immutable, content-addressed OCI image (the
"actual-job image") containing project code, resolved configuration,
materialized datasets, and installed Verifiers environment packages. The
guaranteed shape of a real project is a repository holding harness code, one
or more Verifiers environment packages, and dataset preparation — yet that
shape iterates slowest and fails latest today. An environment package living
in the project repo must be committed and *pushed* before every pack; a
`data_path` inside an environment activation can reference a file that is
never packed and only fails on the GPU (this happened in production); the
image smoke test stops at Python imports; dataset preparation scripts are
invisible to the framework; and editing an unrelated catalog entry
invalidates an otherwise unchanged job image.

After this plan, a developer edits an in-repo environment and re-packs
without any git push; a missing activation data file fails `job plan` with an
actionable message instead of killing a submitted run; the image is qualified
by actually loading the selected taskset; a two-environment project with a
dataset builder scaffolds and packs with zero hand-edited pack configuration;
and a purely local Docker run no longer requires pushing to a remote
registry.

## Progress

- [x] (2026-08-01) Plan authored from the v0.2.5 release-scoped critique.
- [x] (2026-08-01) Cross-plan architecture review completed; generated locks,
      phase boundaries, qualification, discovery, and builder behavior revised.
- [x] (2026-08-01) Follow-up review bound the complete discovered family set
      and versioned layer provenance into package identity.
- [ ] Milestone 1: named activation resources with generated locks.
- [ ] Milestone 2: image qualification exercises `Taskset.load()` and staged
      datasets.
- [ ] Milestone 3: `project-path` environment source kind.
- [ ] Milestone 4: intent/materialization/publication/launch split and
      local-image publication.
- [ ] Milestone 5: selected transitive configuration closure.
- [ ] Milestone 6: standard project layout, selective convention discovery,
      `posttrain env new`.
- [ ] Milestone 7: declared dataset builders (`source.kind: built`) with
      automatic cache rebuild and frozen replay.

## Surprises & Discoveries

- Observation: asking a developer to copy a SHA into catalog YAML would move
  the existing generated-lock problem to a new surface.
  Evidence: the project snapshotter already computes content digests, so the
  resource lock can derive the value before materialization.
- Observation: “fail planning when builder inputs changed” conflicts with a
  plan that also says materialization executes the builder automatically.
  Evidence: separating `JobIntent` from `MaterializedJobPackSpec` permits a
  side-effect-free plan, automatic cache rebuild, and explicit frozen replay.

## Decision Log

- Decision: milestone 1 goes first even though milestone 3 is the loudest
  developer complaint.
  Rationale: milestone 1 closes the only finding that has already failed in
  production (Ambient Agent DAPO packaging), and its resource contract is a
  prerequisite for validating path-sourced environments in milestone 3.
  Date/Author: 2026-08-01 / plan author.
- Decision: `project-path` sources derive identity from the staged source-tree
  digest, not from git state.
  Rationale: the package key is already content-addressed over project source;
  reusing that mechanism preserves immutability without demanding a push, and
  a dirty worktree packs honestly as its actual bytes.
  Date/Author: 2026-08-01 / plan author.
- Decision: developers declare resource identity and source, while planning
  computes size and digest into the materialized lock.
  Rationale: a SHA for a project file is derived data. Requiring it in catalog
  YAML recreates the manual-lock drift this framework is meant to remove.
  Date/Author: 2026-08-01 / architecture review.
- Decision: qualification runs the actual runtime entry point offline, without
  secrets, with a timeout and temporary writable directory.
  Rationale: arbitrary environment loading during a Docker build can otherwise
  depend on network, credentials, or persistent host state and produce a smoke
  that cannot be reproduced.
  Date/Author: 2026-08-01 / architecture review.
- Decision: semantic planning never executes a project dataset builder.
  Rationale: planning should be inspectable and side-effect free. Materialize
  rebuilds cache misses; `--frozen` is the explicit no-rebuild replay mode.
  Date/Author: 2026-08-01 / architecture review.

## Outcomes & Retrospective

- Planning review outcome: semantic intent is side-effect free; materialization
  owns generated bytes and locks; publication and launch are later phases.
  Implementation outcomes remain pending.

## Context and Orientation

Vocabulary. A *Verifiers environment* is an installable Python package that
owns tasks and verification for rollouts; the catalog binds one via an
`EnvironmentBinding` whose `source` says where the package comes from and
whose `activation` is a serializable Verifiers configuration (for example
`taskset: {id, split}` and optionally a `data_path`). *Packing* (`posttrain
job pack`) stages sources, wheels, datasets, and configuration into a
content-addressed build context and publishes the actual-job image.

Key files:

- `packages/eval/src/posttrain/eval/requests.py` — `EnvironmentSource`
  (requires an HTTPS `repository` plus a full 40-hex-character commit
  `revision`; no local-path variant) and `EnvironmentBinding`. Schema in
  `packages/eval/src/posttrain/eval/catalog_schema.py`.
- `packages/execution-pack/src/posttrain/execution_pack/planning.py` —
  derives per-binding git-fetch and wheel-build requests; `_activation_lock`
  copies the activation config verbatim, checking only JSON serializability.
  Nothing inspects or stages `data_path` (finding 13).
- `packages/execution-pack/src/posttrain/execution_pack/service.py`
  (~1,236 lines) — staging, validation, digesting, manifest, retention.
  `_project_config_bundle` (in `apps/cli/src/posttrain_cli/
  execution_planning.py`) globs *every* file under *every* catalog overlay,
  while work packages are already narrowed to the selected one (finding 8).
- `packages/execution-pack/src/posttrain/execution_pack/environment_wheels.py`
  — builds wheels with `uv build --wheel` from a git checkout only.
- `packages/execution-buildkit/src/posttrain/execution_buildkit/job_image.py`
  — always builds with `type=image,push=true` and verifies via
  `docker buildx imagetools inspect`; there is no load-only path, so local
  Docker requires a remote registry (finding 7).
- Actual-job image smoke stage (`posttrain-job` Dockerfile) runs
  `posttrain-runtime --help` and parses `package.json` only — no activation,
  no `Taskset.load()` (finding 15).
- `apps/cli/src/posttrain_cli/pack_config.py` — `[tool.posttrain.pack]` with
  exactly `project_packages` (default `["."]`) and `source_includes`.
- `apps/cli/src/posttrain_cli/scaffolding/init_project.py` — scaffolds one
  `src/<package>/`; no environments or datasets convention (finding 16).
- `packages/data/src/posttrain/data/catalog.py` — dataset source kinds are
  static: `fixture | huggingface | jsonl | nemo | parquet` (finding 17).

## Plan of Work

Milestone 1 extends the environment activation contract with named resource
declarations:

    activation:
      kind: verifiers-config
      config:
        taskset:
          id: episode-qa-v1
          split: train
          data_path: { $resource: task_data }
      resources:
        task_data:
          source: { kind: project-path, path: data/kg_extract/train_env.jsonl }

Planning verifies each declared project resource exists, computes its size and
digest, and records those derived values in the materialized pack lock. Packing
stages resources below a deterministic `environment-resources/` directory so
they join the package identity; before the native Verifiers configuration is
constructed at runtime, the reserved one-key `{$resource: NAME}` value is
resolved to the staged path. Detached planning *rejects* an undeclared
project-relative file path inside an activation for portable execution, with an
error naming the exact repair (declare it as a resource, or package the data
inside the environment). An optional expected digest is allowed only for bytes
fetched from an external source; users do not maintain hashes for project-tree
files. Explicit host paths remain valid for in-process experimentation only.

Milestone 2 adds `posttrain-runtime qualify --offline` and runs it inside the
actual-job image with networking disabled, no project runtime secrets, a
temporary writable directory, and a per-binding timeout. It constructs every
selected activation, calls `Taskset.load()`, and opens/parses every staged data
package. Portable environments must keep taskset construction offline; a
binding that declares deferred qualification is visible in the plan and is
rejected by production policy unless explicitly waived. Failures are packaging
errors before publication, never admitted runs.

Milestone 3 adds the `project-path` source kind to `EnvironmentSource` and
its schema: `source: {kind: project-path, path: environments/episode_qa}`.
The path must be inside the project root and contain a `pyproject.toml`. The
packer snapshots that tree with the same bounded snapshotter used for project
source, derives identity from the tree digest, and builds the wheel from the
staged tree exactly as it builds from a git checkout. Full-commit git sources
remain the form for sharing environments across projects; document the
distinction in the schema error messages.

Milestone 4 separates four values: `JobIntent` (resolved job meaning without
executing project code), `MaterializedJobPackSpec` (source, builder outputs,
environment wheels, resources, and their locks), `JobPublicationPlan` (local
image store or OCI destination), and `JobLaunchPlan` (provider, policy, mounts,
run id). `job plan` needs no registry. Add a content-addressed local-image
publisher for local Docker (BuildKit `--load` or a local OCI layout); dstack
requires an OCI destination only at publication/submission preflight.

Milestone 5 replaces the whole-overlay glob in `_project_config_bundle` with
the transitive closure of catalog entries reachable from the selected job and
binds that configuration digest into `JobIntent`. The digest covers the full
`FamilyRegistryLock` from the public-authoring plan plus, for every selected
layer, layer id/revision, manifest digest, resolved include/exclude rules,
selected file paths/digests, and entry provenance. Pack fails with "project
configuration changed after planning" when the selected closure or discovered
family set drifts. Editing an unrelated catalog entry does not change the key;
installing a different family-provider extra does, because it changes the
resolution environment recorded by the lock.

Milestone 6 scaffolds and documents the standard layout —
`src/<package>/` for harness code, `environments/<name>/` for env packages,
`datasets/` for builders — and derives pack configuration from it: every
`environments/*` directory containing a `pyproject.toml` becomes an install
candidate, but only candidates reachable through the selected job are staged;
explicit `[tool.posttrain.pack]` entries remain as overrides. Add `posttrain
env new NAME` to scaffold a minimal Verifiers package in place, and update
`posttrain init --template grpo` to generate a project whose environment
lives in `environments/` and is bound via `project-path`.

Milestone 7 adds a declared dataset builder:

    dataset:
      datasets/kg-extract-sft@3:
        source:
          kind: built
          builder:
            kind: python-file
            path: datasets/build_kg_extract.py
            callable: build
          inputs: [datasets/build_kg_extract.py, data/raw/episodes.jsonl]

`job plan` records the builder specification and input digests but does not run
it. Materialization executes the builder in an isolated project environment,
uses the input digest as the cache key, and records output identity in the
materialized lock. Normal `job pack` rebuilds a cache miss automatically;
`job pack --frozen` fails if the recorded output is absent or stale. Static
kinds remain unchanged.

## Concrete Steps

Work from the repository root.

    uv run pytest packages/execution-pack/tests packages/eval/tests -q
    uv run pytest apps/cli/tests -q

Milestone-1 tests, mirroring the feedback document's acceptance: a
package-owned split loads with the project checkout absent; a declared project
JSONL has its digest generated and is staged; an optional expected digest on an
external source fails before publication when wrong; an undeclared relative
`data_path` fails detached planning with the repair message. Milestone-3 test:
a fixture project containing
`environments/toy_env` packs twice — identical tree, identical package key;
edit one file, different key; no git operations occur (assert the git packer
was never invoked).

End-to-end check once milestones 1–3 and 6 land, in a scratch directory:

    posttrain init demo --template grpo
    cd demo
    posttrain env new episode_qa
    posttrain job plan .posttrain/work_packages/grpo.yaml
    posttrain job pack .posttrain/work_packages/grpo.yaml

Expected: plan and pack succeed with the environment sourced from
`environments/episode_qa` and qualification output includes a
`taskset-load: ok` line per binding.

## Validation and Acceptance

- An environment package inside the project repository packs from the working
  tree by content digest, with no commit or push.
- A declared activation resource is staged and `Taskset.load()` succeeds
  during image qualification; an undeclared project-relative path fails
  planning with an actionable error.
- A project with two in-repo environments and one dataset builder scaffolds,
  packs, and submits without hand-editing `[tool.posttrain.pack]`.
- A dataset builder whose inputs changed is rebuilt automatically during normal
  materialization; the same condition fails under `job pack --frozen`.
- Local Docker execution completes with no remote project-image registry
  configured.
- Editing a catalog entry not reachable from the selected job does not change
  the package key; changing the installed catalog-family set does change it;
  `job diff` separately names family-registry, layer-manifest, and selected
  entry differences.

## Idempotence and Recovery

All packing changes preserve the verify-after-write discipline already in
`service.py`: staged trees are digested and reconciled against the materialized
lock, and any drift is a hard error naming what moved. New source kinds and the
resource contract are additive; existing git-sourced bindings and static
datasets continue to work unchanged. The closure change (milestone 5) alters
package keys for previously packed jobs — release-note it explicitly; old
images remain valid and addressable by their recorded keys. Builder outputs are
written to a temporary cache entry, verified, then atomically promoted so an
interrupted build never appears complete.

## Artifacts and Notes

Keep the milestone-2 qualification transcript (taskset-load lines) and a
before/after `job diff` from milestone 5 here as indented evidence.

## Interfaces and Dependencies

In `packages/environment/src/posttrain/environment/contracts.py`:

    @dataclass(frozen=True)
    class ProjectPathEnvironmentSource:
        kind: Literal["project-path"]
        path: str                      # project-relative, must contain pyproject.toml

    @dataclass(frozen=True)
    class ActivationResource:
        source: PackResourceSource     # project path, package resource, or immutable external artifact
        expected_sha256: str | None    # external sources only

In `packages/execution-pack`, extend `EnvironmentActivationLock` with
`resources: Mapping[str, StagedResourceLock]` (name, staged relative path,
sha256, size), and give the runtime a resolver that maps declared names to
staged absolute paths before constructing Verifiers configuration. Dataset
builders depend only on `posttrain.data` contracts. For `kind: python-file`,
the staged file is loaded by path in an isolated process; its named callable
receives resolved input paths and an output directory, returns the produced
file path, and cannot rely on the submitting process's import state.
Materialization owns digesting and caching.

The environment and resource contracts live in the dedicated
`posttrain.environment` package introduced by
`docs/plan/dx-public-api-and-authoring.md` milestone 2. They must not add
Verifiers-specific imports to `posttrain.common`.

`MaterializedJobPackSpec` also carries the `FamilyRegistryLock` and selected
layer locks defined by `docs/plan/dx-public-api-and-authoring.md`. Actual-job
runtime consumes the frozen catalog result and does not rerun entry-point or
overlay discovery on the provider.

## Revision Notes

- 2026-08-01: Architecture review made resource digests generated lock data,
  defined offline runtime qualification, split intent/materialization from
  publication and launch, made convention discovery selective, and changed
  dataset builders from plan-time staleness failures to automatic materialize
  rebuilds with an explicit frozen replay mode.
- 2026-08-01: Follow-up review made the complete discovered family registry and
  versioned layer provenance explicit package-identity inputs while retaining a
  selected transitive catalog-entry closure.
