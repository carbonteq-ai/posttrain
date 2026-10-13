# Cleanup playbook

Use this when disk runs low, after a release, or after a batch of experiments.
It lists everything that accumulates across the workstation, the training
workers, Trackio/Doris, the registries and the builders, with the command that
measures it, the command that previews a cleanup, the command that applies it,
and what must never be removed.

Every command here was checked against the tools in this repository, the
CarbonTeq Trackio fork and `ai-infra` on 2026-09-27. Where no tool exists the
section says so under **Gap**; do not improvise a broad command there.

## Rules

1. **Measure, preview, review, apply.** Run the inventory first, then each
   tool's preview (most tools preview by default and need `--apply`). Show the
   preview to the owner and get an explicit yes for anything irreversible:
   runs, artifacts, images, registry manifests, backups, branches, worktrees.
2. **Protected, always:**
   - a run that is not terminal, and the checkout, image and workspace it was
     launched from;
   - checkpoints, model artifacts and datasets that another run consumes
     (lineage), and anything with `pinned` retention;
   - image digests used by live dstack runs, the deployed Trackio and
     Observatory, the current stable release and its rollback;
   - retained Doris backups (listed below) and backup repositories;
   - the Trackio dead-letter directory (`inbox-dead-letter/`), which is evidence
     of writes that failed to import.
3. **Shared hosts hold other projects' data.** The workstation also runs other
   projects' databases and containers (for example analytics Postgres, the
   spec-lab graph database, Supabase). Never run `docker system prune
   --volumes`, `docker volume prune`, or an unfiltered `docker image prune -a`
   there. Remove Posttrain-owned things by name.
4. **Caches that make work fast are capped, not wiped.** The `posttrain-builder`
   BuildKit cache is what makes a local image build take minutes instead of
   20 minutes; prune it by age or size, never with `-af` except as the
   documented last resort.
5. **One host, one thing at a time, while nothing is running on it.** Check
   `posttrain workers` and `posttrain run list` first; registry and builder
   maintenance also require zero active RunPod Pods and an idle publication
   controller (the `ai-infra` tools enforce this).
6. **Record what was removed** in the relevant plan or `ai-infra` execution log
   (sizes before and after, receipts written by the tools).

## Timing

Anything unused that can be rebuilt or fetched again goes as soon as it is
unused. Only what is costly to recreate or may be resumed is kept for seven
days: the BuildKit build cache, Hugging Face models and datasets, and Claude
session scratch. Retained Doris backups follow their own rule below. Run
evidence never expires on a timer: runs are purged only after review.

## When to run it

- Disk alerts: alert at 70% used, stop scheduling at 80%
  (`docs/operations/dstack-trackio/unraid-o0-inventory.md`).
- After a release: retire the previous release's temporary worktrees, local
  job-kind images and development-index candidates.
- After an experiment series: review cleanup candidates among runs.

## Inventory (read-only)

Workstation (`pop-os`), from the repository root:

```bash
df -h / /home
du -sh ~/.cache/uv ~/.cache/huggingface ~/.local/share/uv /tmp/claude-* 2>/dev/null
docker system df
docker buildx du --builder posttrain-builder | tail -3
uv run posttrain --project-root apps/lab cache status
git worktree list
```

Snapshot on 2026-09-27 (927 GB disk, 77% used): uv cache 116 GB, BuildKit
`posttrain-builder` state 85 GB, Docker images 42 GB (20 GB unused), Hugging
Face cache 50 GB, ten extra `rl` checkouts 27 GB, Claude scratch 9 GB,
Posttrain project state 0.3 GB.

## Workstation

`scripts/operations/cleanup_workstation.py` does everything in this section
except merged remote branches. It previews by default:

```bash
.venv/bin/python scripts/operations/cleanup_workstation.py            # preview
.venv/bin/python scripts/operations/cleanup_workstation.py --apply    # remove
.venv/bin/python scripts/operations/cleanup_workstation.py --only worktrees --only posttrain-images
```

A daily user timer runs it with `--apply` at 04:30. Install it once from the
main checkout:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/operations/systemd/posttrain-cleanup-workstation.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now posttrain-cleanup-workstation.timer
journalctl --user -u posttrain-cleanup-workstation.service   # what each run removed
```

The sections below say what each class removes and keeps.

### Extra checkouts (git worktrees)

**What.** Release, run-launch and feature worktrees (`~/projects/rl-*`,
`~/.codex/worktrees/*/rl`, `~/projects/worktrees`), each 0.5-6 GB (a `.venv` or
`node_modules` makes them large).

**Removed right away** when it has no uncommitted or untracked work, its commit
is merged into `origin/main` or pushed to a remote branch, and no process is
working inside it. Run records a checkout holds
(`apps/lab/.posttrain/state/executions`) are first copied into the main
checkout with `posttrain state migrate`, which refuses while any run is
unresolved, so a live run's launch checkout stays. The same check by hand:

```bash
git fetch origin
for wt in $(git worktree list --porcelain | awk '/^worktree /{print $2}'); do
  [ "$wt" = "$(git rev-parse --show-toplevel)" ] && continue
  head=$(git -C "$wt" rev-parse HEAD)
  printf '%s\t%s\tchanges=%s\t%s\tremote=%s\t%s\n' "$(du -sh "$wt" | cut -f1)" "$wt" \
    "$(git -C "$wt" status --porcelain | wc -l)" \
    "$(git merge-base --is-ancestor "$head" origin/main && echo merged || echo unmerged)" \
    "$(git branch -r --contains "$head" | wc -l)" "$(git -C "$wt" log -1 --format=%cr)"
done
```

**Protect.** A checkout with changes, an unpushed branch, a checkout a process
is working in, and one whose run records cannot move yet. Other people's checkouts
(`../trackio`, `../verifiers`, `../verifiers-environments`) are never removed.

**Apply** (per worktree, after review): `git worktree remove <path>` (refuses
if dirty; never add `--force` without review), then `git worktree prune`.

### Merged remote branches

`git branch -r --merged origin/main` lists branches already in `main` (19 of 39
on 2026-09-27). Deleting one is `git push origin --delete <branch>` and needs
review: open pull requests and release branches stay.

### Posttrain project cache

**What.** Pack contexts, local OCI layouts, staging directories under
`apps/lab/.posttrain/state/cache/`.

```bash
uv run posttrain --project-root apps/lab cache status      # protected vs reclaimable
uv run posttrain --project-root apps/lab cache explain <path|key>
uv run posttrain --project-root apps/lab cache prune       # preview
uv run posttrain --project-root apps/lab cache prune --apply
```

The tool keeps anything with an active lease, publication receipts, durable
execution records and unpublished layouts. Checkpoints, adapters, weights,
reports and traces are never cache. **Gap:** the 40 GiB budget and 30-day age
policy are reported by `status` but not enforced by `prune`.

### uv cache

`uv cache prune` removes entries no longer referenced (safe; keeps what current
environments use). `uv cache clean` empties it and forces every environment to
re-download; use only when the disk is critical and nothing is building.

### Hugging Face cache

```bash
uv run hf cache ls                 # repositories and sizes
uv run hf cache prune              # detached revisions and incomplete downloads (safe)
uv run hf cache rm <repo-id>       # one model, after review
```

Keep models the catalog or a running job uses (the LFM2.5 family, current
Qwen and Gemma selections). Large one-off downloads (for example a 19 GB
model tried once) are the usual candidates.

### Docker images and containers (Posttrain-owned only)

```bash
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
  | grep -E '^(posttrain-|registry.lan/carbonteq/posttrain-)'
docker ps -a --filter status=exited --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
docker image prune            # dangling layers only (safe)
docker image rm <repo:tag>    # named Posttrain images after review
```

Removed right away: `posttrain-local:*` and local `registry.lan/carbonteq/posttrain-*`
images no container uses (about 9 GB each; they rebuild or pull again), and
`posttrain-observatory:*` images except the two newest. Containers are never
removed by the tool: every stopped container on this machine belongs to another
project.

### BuildKit cache (`posttrain-builder`)

```bash
docker buildx du --builder posttrain-builder
docker buildx prune --builder posttrain-builder --filter until=720h     # older than 30 days
docker buildx prune --builder posttrain-builder --max-used-space 80GB   # cap the size
```

Prune only when no build is running. `docker buildx prune -af` empties it and
the next release rebuilds everything cold (last resort, `docs/publishing.md`).

### Claude and Codex scratch

`/tmp/claude-<uid>/<project>/<session>/scratchpad` and `~/.codex/worktrees`
hold session files and worktrees. Remove a finished session's scratchpad after
copying anything worth keeping (backup receipts are kept in `ai-infra/.state`,
not here).

## Training workers

### Run workspaces

Workers keep `/var/lib/posttrain/runs/<run-id>` until the daily
`posttrain-worker-gc` timer removes terminal ones (succeeded after 7 days,
failed after 3). Preview on a worker with `sudo posttrain-worker-gc`; it
removes only directories carrying a valid terminal marker.

Releasing one finished run's provider execution, workspace and run-scoped
image (evidence, checkpoints and receipts stay):

```bash
uv run posttrain --project-root apps/lab run cleanup <run-id>
```

`run reconcile` does this automatically once a run is settled.

### Worker Docker images and build cache

**In progress (not yet merged in `ai-infra`):** `posttrain-worker-docker-gc`
(dry run by default; `--apply`; images unused for 7 days, build cache after
24 hours; below 200 GB free it drops the age filters and keeps an 80 GB
reserve). Until it lands, on a worker: `docker image prune` for dangling
layers and `docker image rm` by name for finished runs' images, never while a
run is placed there (`posttrain workers`).

### RunPod volumes

dstack deletes a run's network volume during run cleanup. **Gap:** no sweeper
finds leaked volumes; check the RunPod console for volumes without a live run
before a billing period ends.

## Runs and evidence (Trackio, Doris, artifact storage)

### Finished runs that are not needed

1. Find candidates with the semantic layer, for example training runs that
   never reached ten updates, smoke runs and failed bring-up attempts:

   ```sql
   select r.id, r.job_kind, r.status, r.work_package, count(u.update_seconds) as updates
   from runs r left join updates u on u.run_id = r.id
   group by r.id, r.job_kind, r.status, r.work_package
   having count(u.update_seconds) < 10 order by r.id
   ```

   (`uv run posttrain --project-root apps/lab query --sql "..."`). Show the list to the owner; nothing
   is purged without explicit confirmation of the exact run ids.
2. Plan, review, apply (the plan is immutable and digest-bound):

   ```bash
   uv run posttrain --project-root apps/lab run purge <full-run-id> --reason disposable-smoke
   uv run posttrain --project-root apps/lab purge show <purge-id>
   uv run posttrain --project-root apps/lab purge apply <purge-id> --expect-digest sha256:<digest>
   ```

   `--cascade` includes runs that consume this one; `--orphan` handles a
   tracking run with no local submission receipt (stale after 24 hours by
   default). The planner blocks runs that are not terminal or reconciled,
   evidence another run consumes, pinned runs and shared images; a tombstone
   remains. Registry blob reclamation is a separate infrastructure step below.

Never delete runs with Trackio's unguarded `delete_run` / `delete_project`
calls: they skip the plan and leave artifact versions and blobs behind.

### Incomplete artifact uploads

On the Trackio host (local mode only):
`trackio cleanup-uploads --project <p> --older-than-hours 24` previews,
`--apply` removes. Completed sessions are never touched.

### Gaps

- Artifact versions and blobs orphaned by an old `delete_run` have no GC.
- `inbox-dead-letter/` has no replay or cleanup command (kept by design).
- `ai-infra/.state/execution-log/events.jsonl` is append-only with no rotation.

## Registries

All registry maintenance runs from `ai-infra` and refuses while a RunPod Pod is
active or the publication controller is busy. Plans expire after one hour.

```bash
# obsolete repositories
./scripts/cleanup-registry-repositories plan --repository <name>
./scripts/cleanup-registry-repositories apply --confirm-exact-repositories
# blob garbage collection, LAN and/or R2
./scripts/garbage-collect-cloud-registry plan --target lan
./scripts/garbage-collect-cloud-registry apply --confirm-maintenance
```

Retention roots: the stable release and its rollback, images of active dstack
runs, retained release manifests and accepted evidence, and every verified
publication root. No age-only retention and no `--delete-untagged`.
**Gap:** retiring old framework, service and occupancy image roots has plans
and receipts under `ai-infra/.state/maintenance/*-image-retirement/` but no
committed script; `posttrain-release` has no registry-retention command.
Development-index candidates are retired with `posttrain-release
candidate-retirement-check` / `candidate-retirement-complete` around the index
deletion.

## Doris, backups and service logs

### Retained Doris backups

Each schema migration keeps a restore-verified snapshot (S3 bucket on the
storage host plus a Doris repository):

| Database | Repository | Bucket | Taken before |
| --- | --- | --- | --- |
| `trackio_candidate` | `trackio_group_facts_candidate_repo_20260923` | `trackio-group-facts-candidate-20260923` | v3 |
| `trackio` | `trackio_group_facts_production_repo_20260923` | `trackio-group-facts-production-20260923` | v3 |
| `trackio_candidate` | `trackio_candidate_pre_v4_repo_20260927` | `trackio-candidate-pre-v4-20260927` | v4 |
| `trackio` | `trackio_production_pre_v4_repo_20260927` | `trackio-production-pre-v4-20260927` | v4 |

Receipts are under `ai-infra/.state/artifacts/`. **Proposed rule** (not yet
agreed): keep the newest pre-migration snapshot of each database; an older one
may go seven days after the next migration's server has been running cleanly.

**Hazard.** `ai-infra/scripts/qualify_trackio_doris_backup.py` drops its
repository at start and, without `--retain-backup`, deletes its bucket at the
end. Always pass new, unique `--repository`, `--bucket` and `--snapshot` names
and `--retain-backup` when the snapshot must be kept; never reuse a name from
the table above.

### Doris restore clones and test projects

The backup qualifier drops its restore database itself. Test projects in the
test database (`trackio_candidate`), such as `semantic-sql-*`,
`trackio-*-qualification` and `trackio-replacement-bench`, may be removed with
the purge flow above when that database is next tidied.

### Logs and metrics

- Doris FE/BE logs: **in progress** in `ai-infra` (a tmpfiles rule keeping one
  day under `/srv/data/doris/{fe-log,be-log}`).
- Container logs: json-file, 20 MB × 3 per container.
- VictoriaMetrics: 14-day retention.
- RunPod inventory Parquet: 180 days.
- Backups: `./scripts/backup` keeps 7 daily, 4 weekly, 6 monthly restic
  snapshots.

## Open gaps (owners to decide)

1. Enforce the Posttrain cache budget and age policy in `cache prune`.
2. A command to remove a stale admission ledger (`posttrain workers` only warns).
3. A committed image-root retirement tool and a `posttrain-release` retention plan.
4. Scripted, bounded pruning of the release and remote BuildKit caches.
5. Retention for old dstack release component sets.
6. Rotation for the `ai-infra` execution log.
7. Trackio GC for orphaned artifact versions and blobs; dead-letter replay.
8. An agreed expiry for retained Doris backups (proposal above).
9. A RunPod volume sweeper.
10. Land the worker Docker GC in `ai-infra`.
