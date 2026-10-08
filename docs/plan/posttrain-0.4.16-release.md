# Release Posttrain 0.4.16: public install, faster and correct evaluations

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this release a developer outside CarbonTeq can download the `v0.4.16` wheelhouse from the public GitHub Release and install Posttrain with public PyPI alone; v0.4.15 failed with "carbonteq-trackio was not found in the package registry". AutomationBench evaluations also finish in 9-13 minutes instead of about 23, lose no attempts to failed model calls, and score manifest goals that correct episodes actually meet. Success is visible as the GitHub release `v0.4.16`, `posttrain==0.4.16` installing from `carbonteq/stable`, and the candidate's new step "Prove a public install from the wheelhouse and PyPI alone" passing.

The frozen product baseline (`docs/post-training/01`-`06`) is unchanged: the changes are packaging, evaluation execution, fork pins and environment manifests.

## Progress

- [x] (2026-10-08 08:55Z) Verifiers `carbonteq-v0.3.2.dev109` released at `bc70a7de` on `codex/carbonteq-verifiers-latest` (merge of transport retries `9d95aeb3`/`3cd9045a`/`b358d355` and compact archives `c4ba45e1`); wheel `0b2bffc5…`, sdist `ecd52fb8…`; fork ledger records it (`2e4355d3`).
- [x] (2026-10-08 09:40Z) verifiers-environments PR #5 (eleven manifest check fixes, boundary-check pin) and PR #6 (all packages on dev109, automationbench-v1 0.5.1, gsm8k-v1 0.3.2, others 0.1.2) merged; released as `carbonteq-2026.10.08` at `d430dd87` with per-package tags. automationbench_v1 4219 tests pass locally; GitHub-hosted package jobs cannot resolve `pypi.lan` (also failing on `main`).
- [x] (2026-10-08 09:50Z) Posttrain on `codex/release-0.4.16`: Verifiers `bc70a7de` and environments `d430dd87` in every pin, lock, profile, catalog, constraint, CI and test; lock digests regenerated (uv 0.12.3, vLLM keeps `0.29.1.dev4`); `lock-dependencies` and `prepare 0.4.16` agree; trace-evidence streaming fix cherry-picked; CHANGELOG, v0.4 notes and both consumer pages updated. 673 targeted tests pass; catalog valid.
- [ ] Republish the online-RL TRL, online-RL veRL and eval kind images locally (`images plan` reported exactly these three) and commit `published.toml`; `posttrain-release check` and `readiness` pass.
- [ ] Full ladder, push, release PR, Quality, Prepare candidate (GPU canary), merge, Publish release, tag `v0.4.16`, public-install readback.

## Surprises & Discoveries

- Observation: the v0.4.15 public wheelhouse could not be installed by anyone outside the internal network.
  Evidence: a clean venv with `UV_NO_CONFIG=1` and `--index-url https://pypi.org/simple` failed on `carbonteq-trackio==0.31.5.post14.dev35`; `uv.lock` lists four internal-index packages (carbonteq-trackio, carbonteq-renderers, trl, causal-conv1d), all available on their forks' public GitHub Releases. Bundling the pure-Python three made the same install succeed. The candidate's consumer check installed from the internal dev index, so it never noticed.
- Observation: a 100-episode AutomationBench eval trace was 1.18 GB, 92% assessment batches repeating one 25 KB source and 22 KB view reference per batch across four lifecycle states.
- Observation: uv 0.12.3 again rewrote vLLM to `0.29.1.dev4+precompiled` for the same commit; the version line was restored as in 0.4.15.

## Decision Log

- Decision: bundle only pure-Python fork wheels in the wheelhouse; document the 250 MB CUDA `causal-conv1d` wheel as a separate public download.
  Rationale: bundling it grows every wheelhouse from about 5 MB to 255 MB for a dependency only GPU training extras need.
  Date/Author: 2026-10-08, Claude.
- Decision: the release is 0.4.16 and pins Verifiers dev109 rather than dev108.
  Rationale: hatch-vcs counts the merge commit that joined the transport-retry and compact-archive line to the release branch.
  Date/Author: 2026-10-08, Claude.
- Decision: training-run catalog entries keep their recorded environment revisions; only the framework adoption set (as in 0.4.15) and the 64K luna-v2 eval move to `d430dd87`.
  Rationale: experiment catalogs record what each run used; the next training run selects the new release explicitly.
  Date/Author: 2026-10-08, Claude.

## Outcomes & Retrospective

To be completed after publication.

## Context and Orientation

Release mechanics are in `docs/publishing.md` ("The sequence") and `docs/release-engineering.md`. `release/manifest.toml` holds the authored version; `release/github-constraints.txt` pins Git dependencies; `release/forks.toml` lists maintained forks the candidate verifies. `scripts/release/build-python-distributions` builds the wheelhouse and now calls `scripts/release/bundle-public-fork-wheels`. `.github/workflows/release-candidate.yml` and `release.yml` are the protected workflows on the LAN runner.

## Plan of Work

Republish the three kind images locally, commit `published.toml`, run the ladder, open the release PR from `codex/release-0.4.16`, wait for Quality, dispatch Prepare candidate with the `rtx-pro-96gb` profile, fix on the same branch if it fails, merge after it passes, dispatch Publish release for the merged commit, then verify the tag, a clean stable install, and a public PyPI-only install of the published wheelhouse.

## Concrete Steps

From `/home/hammad/projects/worktrees/rl-perf`:

    .venv/bin/posttrain-release images plan --registry registry.lan/carbonteq --receipt-root .posttrain/state/release-receipts --trust-bundle /usr/local/share/ca-certificates/carbonteq-local-ai-caddy.crt
    .venv/bin/posttrain-release images publish --registry registry.lan/carbonteq --receipt-root .posttrain/state/release-receipts --trust-bundle /usr/local/share/ca-certificates/carbonteq-local-ai-caddy.crt --framework-version 0.4.16
    .venv/bin/posttrain-release check --repository-root .
    uv run --no-sync posttrain-release readiness --destination .release/readiness.json

## Validation and Acceptance

`posttrain-release check` reports staged metadata, dependency locks and published images OK. After publication, `curl -LO https://github.com/carbonteq-ai/posttrain/releases/download/v0.4.16/posttrain-wheelhouse-0.4.16.tar.gz`, extraction, and `uv pip install --index-url https://pypi.org/simple --constraint wheelhouse/github-constraints.txt --find-links wheelhouse posttrain posttrain-observatory` in a clean venv with `UV_NO_CONFIG=1` succeeds and `posttrain --version` prints `0.4.16`.

## Idempotence and Recovery

Image publication is idempotent for unchanged inputs. Index uploads and tags are irreversible; a failed candidate consumes its RC number and the next dispatch allocates another. Never move a published tag.

## Artifacts and Notes

    environments carbonteq-2026.10.08  d430dd87c8d6a37c0520acc3624d74f2de31659e
    verifiers carbonteq-v0.3.2.dev109   bc70a7deaf64c8f8e0b41e39c00ac2d1ea1d7e0b

## Interfaces and Dependencies

`scripts/release/bundle-public-fork-wheels UV_LOCK WHEELHOUSE_DIR` copies every locked pure-Python wheel of an internal-index package from `https://github.com/<fork>/releases/download/carbonteq-v<version>/<file>` after checking its `uv.lock` SHA-256.
