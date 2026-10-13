# Keep the platform lean: remove unused material, keep a week of what is costly to redo

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Disk on the workstation, the workers, the registries and the Doris host fills with material nobody needs: extra checkouts, job images of finished runs, package and build caches, downloaded models, session scratch, old backups, logs and registry images. After this work, anything unused that can be rebuilt or fetched again is removed as soon as it is unused, and only what is costly to recreate or may be resumed is kept for seven days. The operator runs one preview command per area and sees exactly what would go; timers apply the safe classes daily. Run evidence (runs, metrics, traces, checkpoints) never expires on a timer: runs that are not worth keeping are listed weekly and purged only after the owner approves the exact run ids.

`docs/operations/cleanup-playbook.md` is the operator's page: what accumulates where, how to measure, preview and apply, what is protected, and the remaining gaps.

This does not change the frozen product baseline: purge, retention classes and tombstones stay as `docs/post-training/03-work-and-evidence.md` defines them.

## Progress

- [x] (2026-09-27) Surveyed existing tooling in this repository, `ai-infra` and the Trackio fork; measured the workstation (927 GB disk, 77% used: uv cache 116 GB, BuildKit 85 GB, Docker images 42 GB, Hugging Face 50 GB, ten extra checkouts 27 GB).
- [x] (2026-09-27) Wrote `docs/operations/cleanup-playbook.md`.
- [x] (2026-09-27) Milestone 1 code: `scripts/operations/cleanup_workstation.py` with tests (`scripts/operations/tests`) and user systemd units (`scripts/operations/systemd/`). First preview: about 32 GB removable right away (eight checkouts, two job images) plus `uv cache prune`.
- [x] (2026-09-27) First `--apply` on the workstation: 7 checkouts and 2 job images removed, BuildKit and Posttrain caches pruned; free disk 219 GB → 238 GB. `rl-rc-verify` stayed because it holds the live VORTEX KL run's record (`state migrate` refused). `uv cache prune` could not get the uv cache lock (see Surprises); fixed.
- [ ] Milestone 1 rollout: install the timer from the main checkout after merge.
- [ ] Milestone 2 (`ai-infra`): BuildKit layers kept 7 days; Doris backup expiry tool; execution-log rotation.
- [ ] Milestone 3: weekly run-candidate report (after the Trackio dev30 server and the 0.4.11 semantic layer are deployed).
- [ ] Milestone 4 (`ai-infra`): registry image-root retention tool.
- [ ] Milestone 5 (Trackio fork): GC of artifact versions and blobs orphaned by run deletion; dead-letter fragments expire after 7 days.
- [ ] Milestone 6: small gaps (stale admission ledger command; dstack release component sets; RunPod volume sweeper).

## Surprises & Discoveries

- Observation: six extra checkouts hold their own run records (`apps/lab/.posttrain/state/executions`, the submission receipts of runs launched from them). Deleting such a checkout would strand those runs.
  Evidence: `ls <checkout>/apps/lab/.posttrain/state` on 2026-09-27; `posttrain state migrate --from-project-root` copies them and refuses while any run is unresolved.
- Observation: the workstation's Docker daemon also serves other projects (analytics Postgres, spec-lab graph database, Supabase, curbside); every stopped container on it belongs to another project.
  Evidence: `docker ps -a --filter status=exited`, `docker system df -v`.
- Observation: `uv cache prune` needs the uv cache to itself, and every `uv run` process (including the cleanup tool started with `uv run`, and long-running workers) holds it; the first apply waited five minutes and failed.
  Evidence: "Timeout (300s) when waiting for lock on ~/.cache/uv/.lock" on 2026-09-27. The timer now starts the environment's Python directly, the tool calls the environment's `posttrain` executable, the uv step runs last with a 60-second lock timeout and reports "uv busy" to retry on the next run.
- Observation: `ai-infra/scripts/qualify_trackio_doris_backup.py` drops its Doris repository at start and deletes its bucket unless `--retain-backup` is given, so reusing a retained backup's name destroys it.

## Decision Log

- Decision: unused, rebuildable material goes right away; a seven-day retention applies only to the BuildKit build cache, Hugging Face models and datasets, Claude session scratch, and superseded Doris backups.
  Rationale: the owner's direction ("only something can be 7 days old, others can go right away"); the warm build cache makes a release's image build take minutes, models are gigabytes to fetch again, and sessions can be resumed.
  Date/Author: 2026-09-27, owner direction; Claude chose the classes.
- Decision: run evidence never expires on a timer; runs are listed weekly and purged through the existing plan/review/apply flow with the owner approving the exact ids.
  Rationale: metrics, traces and checkpoints of real experiments are results; the purge flow already protects lineage and pinned runs.
  Date/Author: 2026-09-27, Claude; owner's standing rule that purges are confirmed.
- Decision: a checkout counts as in use when a process is working inside it (its working directory), not when files changed recently.
  Rationale: removal is safe once its commit is merged or pushed and its run records moved; recency kept finished checkouts for days.
  Date/Author: 2026-09-27, Claude.
- Decision: the workstation tool is an operator script (`scripts/operations/`), not a `posttrain` command.
  Rationale: it manages git checkouts, the Docker daemon, uv, Hugging Face and Claude scratch; framework packages and the CLI stay about jobs and evidence.
  Date/Author: 2026-09-27, Claude.
- Decision: in `ai-infra`, work that others started and have not committed (worker Docker GC, Doris log retention) stays theirs; this plan builds on it once it lands and does not commit it.
  Rationale: `AGENTS.md`: preserve others' uncommitted work.
  Date/Author: 2026-09-27, Claude.

## Outcomes & Retrospective

(To be written at milestones.)

## Context and Orientation

The main checkout is `/home/hammad/projects/rl` (branch `main`); extra checkouts are git worktrees of it. `apps/lab` is the project root the `posttrain` CLI uses. Posttrain's project cache lives under `apps/lab/.posttrain/state/cache` and is pruned by `posttrain cache prune` (`apps/cli/src/posttrain_cli/state_layout.py`, `prune_cache`). `posttrain-builder` is the local BuildKit builder used by `posttrain-release images publish`. `ai-infra` (`/home/hammad/projects/ai-infra`) deploys the shared services; its `.state/` directory holds secrets, receipts and plans and is not in git. The Trackio fork (`/home/hammad/projects/trackio-run-notes`, branch `codex/run-notes`) owns Trackio storage.

## Plan of Work

Milestone 1 (this repository): `scripts/operations/cleanup_workstation.py`, one planner and one applier per class (`worktrees`, `posttrain-images`, `buildkit`, `uv`, `huggingface`, `posttrain-cache`, `scratch`), preview by default, `--apply` to remove, `--max-age-days` (default 7) for the retained classes; `scripts/operations/systemd/posttrain-cleanup-workstation.{service,timer}` run it daily at 04:30.

Milestone 2 (`ai-infra`, in a clean worktree of its `main`): `config/buildkit/buildkitd.toml` keeps build layers 168 hours instead of 720; `scripts/expire-doris-backups plan|apply` keeps the newest retained snapshot of each database and removes older ones seven days after the newer one was taken (drops the Doris repository, deletes the bucket, writes a receipt); `scripts/execution_log.py` rotates `events.jsonl` keeping seven days.

Milestone 3: a `runs` report listing terminal runs older than seven days that failed, were cancelled, or made fewer than ten updates (a semantic-layer query), with `posttrain run purge` preview commands; the owner approves ids.

Milestones 4-6 as listed in Progress.

## Concrete Steps

From the main checkout:

    .venv/bin/python scripts/operations/cleanup_workstation.py
    uv run --no-sync pytest -q scripts/operations/tests

Expected preview lines look like:

    [worktrees] remove /home/hammad/projects/rl-release-045 5.4 GB: clean, merged, no process working in it
    [worktrees] keep   /home/hammad/projects/rl-perf-guard: commit is neither merged nor on a remote branch
    [posttrain-images] remove posttrain-local:057a…-6e5f… 8.8 GB: used by no container (built 2 days ago)

## Validation and Acceptance

`scripts/operations/tests` (4 tests) show that a clean merged checkout is removed while a dirty, unpushed or busy one stays, that a checkout whose run records cannot move stays, and that scratch older than the retention period goes while newer scratch stays. On the workstation, a preview lists only Posttrain-owned material and an `--apply` frees what it listed; `docker ps -a` and other projects' volumes are unchanged.

## Idempotence and Recovery

Every class can run repeatedly; a failed item is reported and the others continue. Removed checkouts are recoverable from git (merged or pushed); images rebuild or pull again; caches refill on use. Run records are copied before a checkout is removed, and removal stops if any record is missing afterwards.

## Artifacts and Notes

First preview (2026-09-27): eight checkouts (5.5 GB, 5.4 GB and six under 0.8 GB), two `posttrain-local` job images (8.8 GB, 8.6 GB), BuildKit at 79.8 GB (under the 80 GB cap), uv cache 115.5 GiB.

## Interfaces and Dependencies

Python 3.12 standard library; `huggingface_hub.scan_cache_dir` when installed (the planner skips the class otherwise); the `git`, `docker`, `docker buildx` and `uv` commands; `posttrain state migrate` and `posttrain cache prune`.
