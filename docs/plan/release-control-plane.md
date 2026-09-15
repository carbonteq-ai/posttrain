# Replace reactive release scripts with a typed control plane

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current as work proceeds. Maintain this document according to `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this work, a maintainer can ask the repository what release transition is legal, see why another transition is blocked, and execute the legal transition through the same tested implementation locally and in GitHub Actions. Generated OCI state will cross the candidate-to-final boundary as one immutable, content-addressed materialization rather than as files rediscovered and copied by shell. Failures will report their owning boundary, whether any remote mutation occurred, and the safe recovery action. Go Task will provide the concise command catalog; the Python release package will remain the correctness engine; protected GitHub environments will remain the authority for publication.

This work implements the existing product meaning in `docs/post-training/03-work-and-evidence.md`, `docs/post-training/04-framework.md`, `docs/release-engineering.md`, and `docs/architecture/lan-release-runner.md`. It does not amend the frozen post-training baseline. It makes the already-documented separation between authored source and generated release evidence executable.

## Progress

- [x] (2026-09-15 09:09Z) Audited current release workflows, release package, runner contract, ai-infra playbook, publishing documentation, git repair history, and recent GitHub Actions outcomes.
- [x] (2026-09-15 09:09Z) Isolated work on `codex/release-system-audit` at `/tmp/rl-release-system-audit`; preserved the dirty primary Posttrain and ai-infra worktrees.
- [x] (2026-09-15 09:09Z) Reconciled the contradictory generated-record instructions in `docs/publishing.md` and recorded the incident-backed target in `docs/release-engineering.md`.
- [x] (2026-09-15 09:09Z) Added the v1 runtime materialization receipt, verifier, staged-tree projector, focused tests, candidate production, and final consumption.
- [x] (2026-09-15 09:09Z) Pinned Go Task 3.50.0 and added a root command catalog for setup, quality, release audit, read-only status, and explicit protected dispatch.
- [x] (2026-09-15 09:35Z) Validated the materialization slice: 65 release tests and the complete locked Task ladder pass; the full suite reports 1711 passed and 33 skipped, Pyright reports zero errors, all eight import contracts are kept, workflow YAML parses, CLI help exposes the three materialization commands, and diff checks pass.
- [ ] Add a read-only release planner whose JSON result names observed state, blockers, proposed transition, side effects, and an idempotency key.
- [ ] Add structured gate-result and failure-class schemas and replay fixtures for the historical incident families.
- [ ] Move candidate admission, GitHub evidence resolution, GPU lifecycle, index publication, final promotion and tag recovery from YAML into tested transition commands.
- [ ] Add a versioned ai-infra runner-qualification receipt in an isolated ai-infra worktree; do not mix it with the current dirty Doris log-rotation work.
- [ ] Measure the new workflow against the historical corpus and remove superseded shell only after equivalent protected runs succeed.

## Surprises & Discoveries

- Observation: Documentation encoded two incompatible generated-state models.
  Evidence: `docs/publishing.md` required committing `published.toml` and runtime locks, while `docs/architecture/lan-release-runner.md` said generated OCI state need not be committed between workflows. Final publication required both models simultaneously.

- Observation: The candidate workflow accepted free-form dispatches that could never pass its own policy.
  Evidence: 25 of the 63 failed runs in the latest 100 candidate runs were dispatched from `main`; preflight later rejected anything outside `codex/*`.

- Observation: The most expensive boundary also has the least structured recovery.
  Evidence: 23 candidate failures ended in the packed GPU qualification shell block, which combines capacity, packing, submission, waiting, reconciliation and cleanup under `set +e` and exit-code mutation.

- Observation: The materialization repair was known before v0.4.1 but remained unenforced.
  Evidence: v0.3.2 introduced candidate-manifest handoff repairs; v0.4.1 final run `34946588849` failed the same invariant and needed PR 110 plus final run `34948932897`.

- Observation: Go Task 3.50.0 is available through mise's `task` registry entry, while the current user-level `task` command is an npm wrapper.
  Evidence: `mise registry` maps `task` to `aqua:go-task/task`; `mise.toml` now pins the actual Go Task distribution so repository operators do not depend on the wrapper.

- Observation: The repository tool selector contradicted the executable package contract.
  Evidence: `mise.toml` selected Python 3.12 while `apps/release/pyproject.toml` requires Python 3.13 and both protected workflows sync with Python 3.13. It now selects Python 3.13 and the same uv 0.12.3 used by required Quality.

- Observation: The documented local validation environment omitted a dependency that the tests import unconditionally.
  Evidence: a clean `uv sync --all-packages --group dev` made Pyright reason differently without Torch and made pytest collection fail because `trl` was absent, while GitHub Quality installs the `trl` extra. `task quality` now depends on an explicit locked setup with that extra.

- Observation: The repository audit itself had drifted from the repository surface.
  Evidence: it reported the new reviewed `Taskfile.yml` as unowned and found a proposal link to a Markdown file that does not exist in any reachable tree at the recorded revision. Task is now explicitly owned and the prose no longer advertises a broken local artifact.

## Decision Log

- Decision: Generated runtime manifests and locks are candidate outputs, not authored Git source.
  Rationale: Their identity depends on protected registry readback. Committing them after qualification changes the source identity, forces redundant CI, and recreates the candidate/final mismatch. A receipt can bind source and generated output without conflating them.
  Date/Author: 2026-09-15 / Codex

- Decision: Go Task is a thin operator interface, not the release engine.
  Rationale: Task is good at discoverable commands, prerequisites and constrained inputs. Python already owns release domain contracts and tests; moving correctness to Task YAML would reproduce the current workflow-YAML problem.
  Date/Author: 2026-09-15 / Codex

- Decision: Keep GitHub protected environments as the mutation authority.
  Rationale: Local planning should be fast and useful, but it must not bypass approvals, repository-scoped tokens, LAN credentials or the dedicated runner trust boundary.
  Date/Author: 2026-09-15 / Codex

- Decision: Migrate by boundary and replay historical incidents before deleting the old path.
  Rationale: A wholesale rewrite would create another unqualified release system. Each replacement must demonstrate the same success case and a stronger failure case before shell is removed.
  Date/Author: 2026-09-15 / Codex

## Outcomes & Retrospective

The audit has produced a measured failure taxonomy and exposed the core design error: release identity is distributed among source, generated runtime state, remote artifacts and mutable YAML variables, without a single executable transition contract. The first implementation slice removes the recurring manifest-copy protocol and gives operators a pinned Task entrypoint. Full state-machine migration and a protected live candidate remain outstanding, so this branch must not yet be described as a complete release-control-plane replacement.

The first slice is locally qualified. The VORTEX v0.4.2 release will be its first protected candidate-to-final exercise; that run must be added here before the materialization boundary is called live-qualified.

## Context and Orientation

`apps/release/src/posttrain_release` owns framework-maintainer release logic. `apps/release/src/posttrain_release/cli.py` exposes it as `posttrain-release`. `.github/workflows/release-candidate.yml` publishes immutable release candidates and runs one packed GPU canary on the LAN release runner. `.github/workflows/release.yml` rebuilds final-version Python metadata from an accepted candidate, publishes to stable, and creates the GitHub tag and release. `scripts/release/build-python-distributions` creates an isolated source stage and builds the coordinated wheelhouse.

A materialization is the complete generated runtime input to a Python release build: `published.toml` plus every lock under the job-kind runtime lock directory. A receipt is JSON that names the materialization's source identity and hashes every file. Content-addressed means changed bytes have a changed SHA-256 digest. Projection means copying verified generated bytes into an isolated staged source tree, never into authored Git history.

The release system spans a second repository, `/home/hammad/projects/ai-infra`, which configures the self-hosted runner, rootless BuildKit, private CA trust, registry and index access. That worktree currently contains unrelated Doris log-rotation edits. Any runner-contract implementation must use a separate ai-infra branch and must name the Posttrain and ai-infra commit order explicitly.

## Plan of Work

Milestone 1 replaces the candidate-to-final file-copy protocol. Add `materialization.py` with create, verify and apply operations. The receipt must name a fixed schema, target and candidate versions, candidate source commit and tree, readiness digest, and the path, size and digest of each file. Only `published.toml` and descendants of `runtime-locks/` are legal. Verification must finish before destination mutation. Modify the build script to accept a materialization directory and project it into the isolated stage. Candidate writes and uploads the materialization; final verifies its expected source and version, then passes it to the build without requiring generated files committed to `main`.

Milestone 2 introduces a pure planner. Add `release_state.py` with enums for phases, gate names, failure classes and retry classes. Add `planner.py` whose adapters supply GitHub, index, registry, runner and git observations. The pure core accepts observations and returns a `ReleasePlan`; it performs no network calls. Add `posttrain-release plan --format json` and human text rendering. The plan must reject `main` as a candidate source before dispatch, identify an already-published exact artifact as reusable, distinguish unavailable evidence from a negative result, and hash its normalized request into an idempotency key.

Milestone 3 gives every boundary a gate result. A gate result records schema, release ID, transition, exact inputs, status, failure class, retry class, whether mutation started, evidence references, and next legal actions. Candidate and final workflows always upload this result, including failure. Add fixtures modeled on source rejection, capacity unavailable, BuildKit dependency failure, private-CA failure, development-index byte conflict, GPU terminal failure, cleanup queued, materialization mismatch, stable success followed by tag failure, and resume with an incompatible artifact.

Milestone 4 extracts orchestration from YAML. Implement transition commands that invoke existing lower-level functions through injectable adapters. Replace each workflow shell block only after unit and replay coverage exists. Keep checkout, locked tool install, protected environment selection and artifact upload in Actions. Move polling, remote-state classification, file-set validation, source-diff policy, publication idempotency and recovery decisions into Python. A final YAML file should read as a short sequence of install, execute and upload steps.

Milestone 5 versions the LAN qualification boundary. In a clean ai-infra worktree, add a read-only qualification command or script that emits a receipt for runner identity, private CA, BuildKit socket, cache disk, registry, stable/development index and dstack connectivity. Posttrain consumes that receipt before mutation. Commit and publish ai-infra first, then update the Posttrain consumer and document both SHAs. Never read secrets into logs.

Milestone 6 qualifies and simplifies. Run the full local ladder, dispatch a non-publishing planner exercise, then run a real candidate on the protected runner. Compare elapsed time, failed-step clarity and redundant work against the incident corpus. After a successful candidate and final release use the new paths, remove replaced YAML shell and compatibility copying. Update this plan and `docs/release-engineering.md` with exact run IDs and measured results.

## Concrete Steps

Work in `/tmp/rl-release-system-audit` on `codex/release-system-audit`. For the current materialization slice, run:

    uv run pytest apps/release/tests/test_materialization.py -q
    uv run pytest apps/release/tests -q
    uv run ruff check apps/release scripts/release
    uv run ruff format --check apps/release
    uv run pyright apps/release
    uv run lint-imports
    task --list
    task release:audit
    git diff --check

For a synthetic candidate materialization after generated runtime inputs exist, run:

    uv run posttrain-release materialization-create .release/readiness.json --candidate-version X.Y.ZrcN --destination .release/materialization
    uv run posttrain-release materialization-check .release/materialization --target-version X.Y.Z --source-sha <candidate-sha> --source-tree <candidate-tree>

The first command must fail if the destination exists, the readiness schema is wrong, the RC does not belong to the target version, or generated runtime files are absent. The second must fail if any byte, path, size, source identity or undeclared file differs.

After planner implementation, `task release:plan` must print a zero-side-effect report. Explicit candidate dispatch remains:

    task release:candidate:dispatch SOURCE_REF=codex/<branch> PROFILE=rtx-pro-96gb GPU=true

Final dispatch remains explicit and protected:

    task release:final:dispatch MERGED_SHA=<40-hex-main-sha> CANDIDATE_RUN_ID=<successful-run-id>

## Validation and Acceptance

The materialization tests must prove that stale generated files in source are replaced in the isolated stage, tampered bytes leave the destination unchanged, extra files fail, unsafe paths fail, identity mismatches fail, and candidate/final distribution receipts contain the same runtime-manifest digest. Workflow syntax must parse and the candidate artifact declaration must include `materialization/**`; final must consume that directory and must not compare the committed manifest with `cmp`.

The planner is accepted when an operator can run it on `main` and see “candidate dispatch blocked: source branch must match codex/*” without creating a GitHub run. Given a green eligible source fixture, it must name exactly one next transition and a stable idempotency key. An unavailable registry or GitHub API must render `unavailable`, never `missing`, `zero` or `safe to republish`.

The state machine is accepted when every historical incident fixture maps to one owning gate and one safe recovery, and property tests prove that no path reaches stable publication or tag creation without source readiness, accepted candidate materialization, qualified runtime evidence and exact index readback. Repeating a transition with the same key must either reuse byte-identical state or stop before mutation.

The whole plan is accepted only after one protected candidate and one final publication complete using the typed paths, their receipts are retained, and the workflow YAML no longer owns domain loops or artifact reconstruction.

## Idempotence and Recovery

Receipt creation refuses an existing destination so a retry cannot blend two candidates. Delete only a disposable, explicitly named `.release/materialization` after preserving its workflow artifact, then recreate it from the same immutable inputs. Verification is read-only. Projection verifies everything before replacing staged files and is safe only for a disposable stage; never point it at the repository root.

Remote transitions must check exact bytes before upload. If exact bytes already exist, record reuse. If a version exists with different bytes, classify a conflict and allocate a new RC; never delete or overwrite it implicitly. If stable bytes exist but tagging failed, resume the same release instance and candidate evidence rather than rebuilding. If external state is unavailable, stop without inferring absence.

## Artifacts and Notes

The initial audit snapshot was:

    Candidate runs, latest 100: 32 success, 63 failure, 5 cancelled
    Final runs since 2026-08-05: 28 success, 23 failure, 1 cancelled
    Candidate failure concentration: 23 packed GPU, 27 preflight/job-level
    Final failure concentration: 10 candidate materialization restore

These figures are dated observations. The exact GitHub run logs and the committed replay fixtures are the durable evidence; refresh counts before comparing later reliability.

## Interfaces and Dependencies

`posttrain_release.materialization.create_materialization(repository_root: Path, readiness_receipt: Path, destination: Path, *, candidate_version: str) -> dict[str, object]` creates the v1 receipt and copied payload. `verify_materialization(materialization_root: Path, *, target_version: str | None, source_sha: str | None, source_tree: str | None) -> dict[str, object]` is read-only. `apply_materialization(...)` verifies first and projects only into a staged Posttrain source tree.

The future `ReleasePlan` and `GateResult` must be serializable without provider-specific objects. GitHub, devpi, OCI, dstack and git clients remain private adapters in `apps/release`; the core state machine accepts normalized observations. No reusable train, eval, serve, common or runtime-images package may import the release application. Go Task invokes public CLI commands only and is pinned in `mise.toml`.

Plan revision note, 2026-09-15: created from the system-wide incident audit and recorded the first materialization and Task milestones so the next contributor can continue without chat history.
