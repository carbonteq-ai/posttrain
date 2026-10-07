# Release Posttrain 0.4.15 with current environments and stable forks

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Since 0.4.14 (2 October 2026) the working branch `wip/automationbench-reward-redesign-2026-10-04` gained SAMPO turn-credit options (group-relative goal credit, verified-sign credit, environment-state anchor fallback), per-pack backward in the resolved policy-update engine (peak trainer memory on the 2.6B recipe fell from 84 GiB to 13.8 GiB), Observatory collection-axis charts, purge fixes, Trackio dev35 and Verifiers `58df1306`. The AutomationBench simulator also became deterministic per call and its Zapier tools were corrected, but that work lives only on branches of the AutomationBench fork and of `carbonteq-ai/verifiers-environments`.

After this release a user installing Posttrain 0.4.15 from `carbonteq/stable` gets all of that, and every external component the release selects is itself a published, immutable release: the environment package at a commit on `verifiers-environments` `main`, the AutomationBench fork at a GitHub release `carbonteq-v1.0.5.post2`, Verifiers at a tagged fork release, and Trackio dev35 promoted to stable. Success is visible as the GitHub release `v0.4.15` on `carbonteq-ai/posttrain`, `posttrain==0.4.15` installing from `carbonteq/stable`, and `uv run posttrain-release fork-ledger` naming only released fork tags.

## Progress

- [x] (2026-10-07 10:40Z) Working branch validated: full ladder green after fixing two unformatted files, research-note whitespace and two type errors (`a8793bf2`); 2,830 tests pass; `posttrain-release check` and `readiness` pass; committed `published.toml` matches the dependency locks.
- [x] (2026-10-07 11:00Z) Inventory of forks and environments (see Context). GitHub CLI re-authenticated as `carbon-teq`.
- [x] (2026-10-07 13:30Z) Milestone 1: AutomationBench `carbonteq-v1.0.5.post2` (merge `106d7924`, PRs #3/#4), then `carbonteq-v1.0.5.post3` (merge `9bfbdd70`, PRs #5/#6) carrying the Buffer `due_at` fix that existed only in the environment's vendored copy. Wheel `682f9852…`, sdist `036dc954…`.
- [x] (2026-10-07 14:00Z) Milestone 2: Verifiers `carbonteq-v0.3.2.dev102` at `74dd3fbf` (`58df1306` plus repository formatting, required by the fork's push hooks); release branch `codex/carbonteq-verifiers-latest` fast-forwarded; ledger records the release.
- [x] (2026-10-07 17:20Z) Milestone 3: environments PR #3 (vendored simulator equals post3, Verifiers dev102 in all six packages; automationbench_v1 4218 tests pass) and PR #4 (package versions); released as `carbonteq-2026.10.07` at `17e0cd02` with per-package tags.
- [x] (2026-10-07 17:40Z) Milestone 4: Posttrain on `codex/release-0.4.15`: Verifiers `74dd3fbf` in every pin, lock and job-kind profile (uv 0.12.3, vLLM line kept); veRL lock digests; catalog lock digest and lab catalogs; AutomationBench post3 in `release/forks.toml` and the ledger code; environments `17e0cd02` in constraints, base catalog, eval programs, CLI starter, CI and tests; three kind images republished locally (`published.toml`); both consumer pages; CHANGELOG; version 0.4.15. Ladder green (2,830 tests, pyright 0), `check` and `readiness` pass.
- [ ] Milestone 5: Release PR, CI, retained-asset publication of the fork to `carbonteq/dev`, Prepare candidate, qualification.
- [ ] Milestone 6: Merge, promote forks to stable, Publish release, tag `v0.4.15`, after-release checks.

## Surprises & Discoveries

- Observation: the environment's vendored simulator had a Buffer `due_at` fix that the fork never had, so post2 could not be vendored byte for byte; post3 moved the fix into the fork.
  Evidence: `diff -r` of the post2 tag against `environments/automationbench_v1/src/automationbench` showed only `tools/zapier/buffer/posts.py` (and the environment-only `tools/api/schemas/index.txt`).
- Observation: `58df1306` failed the Verifiers fork's own pre-push format and lint hooks; the release commit `74dd3fbf` adds formatting only. The hooks also need uv 0.12.3 on `PATH` and `UV_SYSTEM_CERTS=1`.
- Observation: GitHub rejected all object writes for the organization for about ten minutes (git push and the blob API returned HTTP 500 while ref creation worked); retrying succeeded.

- Observation: the release constraints already conflict. `release/github-constraints.txt` selects Verifiers `58df1306` but environment packages at `11f4d712`, whose `automationbench_v1` and siblings require Verifiers `e6a3d9bb`.
  Evidence: `git show 11f4d712:environments/automationbench_v1/pyproject.toml` lists `verifiers @ …@e6a3d9bb…`. Moving the environment pin is therefore required, not optional.
- Observation: AutomationBench is vendored inside the environment package (`environments/automationbench_v1/src/automationbench`) at every revision, so the fork release and the environment commit must carry the same simulator source.

## Decision Log

- Decision: version 0.4.15, not 0.5.0.
  Rationale: chosen by the release owner; the changes are additive settings and fixes in the v0.4 series.
  Date/Author: 2026-10-07, user.
- Decision: include the latest environment and fork work and require every selected fork to be a published release.
  Rationale: requested by the release owner; also forced by the Verifiers/environment conflict above.
  Date/Author: 2026-10-07, user and Claude.
- Decision: held-out evaluation suites in the lab overlay stay on environment `24e1fc9`.
  Rationale: simulator fixes change scores; the lab keeps one revision for comparable evaluations. The framework base catalog moves to the new environment commit.
  Date/Author: 2026-10-07, Claude.

## Outcomes & Retrospective

Not started.

## Context and Orientation

A "maintained fork" is a CarbonTeq copy of an upstream project with a documented delta (`docs/tooling/forks.md`). Each fork keeps `CARBONTEQ_FORK.md` (its ledger) and is released manually from its own checkout as an immutable GitHub release with retained SHA-256 hashes. Posttrain then copies those exact bytes to the internal index `carbonteq/dev` with the retained-asset publisher workflow, qualifies them in a release candidate, and promotes them byte for byte to `carbonteq/stable`. `release/forks.toml` lists the required forks and `apps/release/src/posttrain_release/fork_ledger.py` checks the exact versions and hashes; `uv run posttrain-release fork-ledger` prints the closure.

Inventory on 7 October (local checkouts under `/home/hammad/projects`):

The AutomationBench fork (`automationbench`, origin `carbonteq-ai/AutomationBench`) is released as `carbonteq-v1.0.5.post1` at `908db2ab`. Branch `codex/deterministic-sim-runtime` (`afb92ec`) contains the tool-fidelity commits plus the deterministic simulation runtime (11 commits); `main` has one newer docs commit (`0f7b4bb`, the post1 release record). It publishes `carbonteq-automation-bench` (wheel and sdist).

The Verifiers fork (`verifiers`, origin `carbonteq-ai/verifiers`) was last tagged `carbonteq-v0.3.2.dev93`; Posttrain pins commit `58df1306` (branch `codex/native-assessment-runtime-cost`, 8 commits beyond the tag) by Git URL. The shared checkout has 37 uncommitted files from other work; use a separate worktree.

`verifiers-environments` (origin `carbonteq-ai/verifiers-environments`) has no tags; consumers pin commits. Posttrain's release pins `11f4d712`. The environment work is on `wip/automationbench-simple-manifests-2026-10-06` at `7141454` (57 commits ahead of `main`, 2 behind), checked out at `/home/hammad/projects/worktrees/venv-simple-manifests`; its 4,218 tests pass.

Trackio's newest release is `carbonteq-v0.31.5.post14.dev35` (already pinned). TRL `post14`, renderers `0.1.12.post1.dev3`, vLLM `0.29.1.dev4` and veRL `0.9.0.post9` have no newer releasable work and stay.

Places in Posttrain that pin the environment commit or the AutomationBench release: `release/github-constraints.txt`, `release/forks.toml`, `apps/release/src/posttrain_release/fork_ledger.py` (`_automationbench_entry`), `packages/eval/src/posttrain/eval/programs/automationbench.py` and `general_smoke.py`, `packages/catalog/src/posttrain/catalog/base/environments.yaml`, `apps/cli/src/posttrain_cli/scaffolding/init_project.py`, `.github/workflows/quality.yml`, tests in `apps/lab/tests/test_catalog.py`, `apps/cli/tests/test_cli.py`, `apps/release/tests/test_release.py`, `packages/eval/tests/test_api.py`, and the consumer pages `docs/tooling/automationbench/README.md` and `docs/tooling/verifiers/README.md`.

## Plan of Work

Milestone 1, AutomationBench `1.0.5.post2`. In `/home/hammad/projects/automationbench`, branch `codex/release-1.0.5.post2` from `origin/codex/deterministic-sim-runtime`, merge `origin/main`, set the version to `1.0.5.post2`, and extend `CARBONTEQ_FORK.md` with the deterministic-runtime delta (`automationbench/sim_runtime.py`, the codemod over schema/tools/utils, its tests) and the post2 status. Run the ledger's validation commands. Open a PR to `main` with `gh pr create`, merge it, tag the merge commit `carbonteq-v1.0.5.post2`, build with `uv build` from the tag, create the GitHub release with both distributions, and record the SHA-256 values in a follow-up ledger commit on `main`, as post1 did.

Milestone 2, Verifiers. In a fresh worktree at `58df1306`, follow the fork ledger's release procedure (read `CARBONTEQ_FORK.md` there first): tag `carbonteq-v0.3.2.dev101` or the version the ledger prescribes, build, create the GitHub release, record the hashes. Posttrain keeps the Git pin `58df1306`.

Milestone 3, environments. Open a PR from `wip/automationbench-simple-manifests-2026-10-06` to `main` after merging `origin/main` into it; confirm the vendored `src/automationbench` equals the AutomationBench post2 package source (document any intended difference); run the environment test suite; merge. The merge commit becomes the new environment pin.

Milestone 4, Posttrain. On `codex/release-0.4.15` (from the working branch), move every pin listed in Context to the new environment commit and AutomationBench post2 (versions, hashes, revision), update both consumer pages, run `uv run posttrain-release prepare 0.4.15`, write the CHANGELOG `0.4.15` entry and update `docs/releases/v0.4.md`, then run the full ladder, `posttrain-release check` and `readiness`.

Milestone 5 and 6 follow `docs/publishing.md` "The sequence": push the branch, open the release PR, wait for CI, publish images locally only if `images plan` reports work, dispatch the retained-asset publisher for AutomationBench post2, dispatch Prepare candidate from the release branch, repair on the same branch if needed, merge after the candidate passes, promote the required forks (Trackio dev35, AutomationBench post2) to stable, dispatch Publish release for the merged commit, and confirm the tag and a clean stable install.

## Concrete Steps

Working directories and commands are given per milestone above; every command runs from the named checkout root. Validation for Posttrain, from `/home/hammad/projects/worktrees/rl-perf`:

    UV_SYSTEM_CERTS=1 uv sync --frozen --all-packages --group dev --extra trl --extra verifiers --python 3.13 --inexact
    uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
    uv run --no-sync pyright
    uv run --no-sync lint-imports
    uv run --no-sync pytest -q
    uv run --no-sync posttrain-release check
    uv run --no-sync posttrain-release readiness --destination .release/readiness.json
    git diff --check origin/main..HEAD

Do not use `uv sync --locked` without extras in a development checkout: it removes torch and the Verifiers extra.

## Validation and Acceptance

Accepted when `gh release view v0.4.15 --repo carbonteq-ai/posttrain` shows the release, `pip install posttrain==0.4.15` from `carbonteq/stable` succeeds in a clean environment, and `posttrain-release fork-ledger` on the tagged commit lists AutomationBench `1.0.5.post2` and Trackio dev35 with stable hashes.

## Idempotence and Recovery

Index uploads and tags are irreversible. Failed candidates consume an RC number; fix on the same release branch and dispatch again. A fork tag is created only after its PR is merged; if a GitHub release upload fails, retry the upload for the same tag without rebuilding. Never move or delete a published tag.

## Interfaces and Dependencies

No public Posttrain interface changes beyond what the working branch already contains. Dependency transitions: environments `11f4d712` → new `main` commit; AutomationBench `1.0.5.post1` → `1.0.5.post2`; Verifiers stays `58df1306` (now tagged); Trackio dev35 promoted to stable.
