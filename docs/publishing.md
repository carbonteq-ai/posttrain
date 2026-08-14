# Publishing a release

This is the maintainer's runbook. Installing a release is
[`install.md`](./install.md); the end-to-end release process is
[`release-engineering.md`](./release-engineering.md); the services a release
is published to are operated from the `ai-infra` repository.

A release is two artifacts that must agree: the **distributions** on the
internal index, and the **runtime images** in the registry. Publishing one
without the other produces a framework that refuses to run, by design — the
manifest records which constraint lock each image was built against, and every
command checks it.

There is a second agreement that is easy to forget: the **GitHub tree**. Wheels
and images without a merged, CI-green commit have no reviewable source. GitHub
records and approves the release, but the build runs on an isolated LAN runner
because the supported package index and OCI registry are private services. See
the [LAN release runner architecture](./architecture/lan-release-runner.md).

The protected `release-candidate` and `release` workflows use the LAN runner
and required environment approval. Do not use an older tag-triggered workflow:
a GitHub-hosted runner cannot resolve `pypi.lan`, and tagging first would
reverse the required release order.

## What forces a full release

Most changes only need new distributions. Rebuilding images is forced by
anything that changes the job-kind constraint lock:

    packages/runtime-images/src/posttrain/runtime_images/containers/
      posttrain-job-kinds/locks/workspace.lock.txt

Its hash is the `org.carbonteq.posttrain.lock-digest` label on the published
base and job-kind images. Changing a third-party pin — a fork commit, a
dependency bound — changes that file, and every published image is then stale.

You will not be allowed to forget. The manifest loader refuses:

    base: published image records lock digest ac0f21a8…, but the shipped
    containers/posttrain-job-kinds/locks/workspace.lock.txt hashes to
    c93d274e…. The image must be republished, or the manifest regenerated

That error fails dozens of tests at once and blocks `doctor`, `job pack`, and
`job run`. It is the guard working, not a bug.

Base should rarely rebuild. Kinds `FROM` the registry base; when the committed
base still matches the lock digest it is reused automatically. When a rebuild
is unavoidable, BuildKit is seeded with `cache-from` the previous registry
digest so a wiped local cache still pulls layers. Prefer
`--base-image …@sha256:…` when the digest is already on `registry.lan`.

## Runtime-image performance contract

Runtime-image work is deliberately split so that a source/configuration change
does not rebuild the ML dependency stack. This is the current operating model,
its evidence, and the rules that preserve its benefit.

| Concern | Implemented optimization | Evidence or operating rule |
| --- | --- | --- |
| Image shape | An actual job has three stages: `packaged-context` → immutable runtime kind → `smoke`. The job layer carries the bounded package and first-party wheels; heavyweight third-party dependencies stay in the kind. | A source-only repack must not be used to change an already baked TRL, veRL, or Trackio version. Change the kind lock and rebuild the affected parent instead. |
| Parent reuse | The base is reused by digest when its lock still matches; kinds use that registry base and BuildKit imports cache from the previous digest. | Pass a verified `--base-image …@sha256:…` when the LAN registry already has the base. Do not rebuild merely to relabel an identical parent. |
| Repeat job packs | Actual-job publication reuses a verified receipt for the same publication key. A per-key publication lock makes concurrent callers wait for the producer, then reuse its verified result. | Do not work around a simultaneous pack by creating a second mutable context or deleting the first caller's context. |
| Build parallelism | Changed kind variants build concurrently with rootless BuildKit; unchanged variants are retained from the committed manifest only when their lock digest still matches. | Use `--no-parallel` only to diagnose a BuildKit problem, not as a normal release setting. |
| Push cost | Default pushes use zstd level 1, omit provenance/SBOM attestations, and avoid recompressing already-compressed layers. | Enable attestations, higher compression, or forced recompression only when a release policy explicitly needs them. |
| Python work | The runner has a persistent `UV_CACHE_DIR`; the job build isolates cached dependency wheels from the small first-party package build. | Caches accelerate work but are never evidence. Receipts, source locks, and registry digests remain authoritative. |
| Cold worker start | Workers pull the immutable kind by digest. In the observed TRL candidate, the bounded job context was about 2.9 MiB and cached first-party wheel assembly about one second; the cold path was multi-gigabyte parent-layer transfer. | Measure and prewarm the selected worker's kind digest when startup matters. Do not try to fix a cold pull by folding model dependencies into every job image. |

### Release operator checklist

1. Determine whether the change affects a runtime profile or the generated
   workspace lock. If it does, expect the affected kind images to change.
2. Reuse a digest-pinned base where the lock permits it, and retain the image
   receipt. A locally warm cache is not a qualification result.
3. Keep actual-job packs small and let the verified receipt short-circuit an
   identical retry. The second caller should wait on the publication lock.
4. Distinguish build time from worker pull time in any incident report. The
   first points to BuildKit/cache inputs; the second points to registry and
   worker cache state.
5. Before reclaiming registry capacity, make a retention plan from generated
   manifests and job evidence. Never broadly delete repository blobs or actual
   job images to accelerate a candidate.

### Current boundary and next decision

The shared `workspace.lock.txt` is intentionally a global runtime input. A
third-party package change therefore invalidates every kind that consumes it;
the latest candidate showed that this can make a one-package change take about
eight minutes even though the actual-job context is small. This is correct for
the current provenance model: every kind declares the same complete internal
dependency closure.

If that cost becomes material, the next optimization is **kind-scoped
constraint locks and digest inputs**, not a shortcut around lock verification.
It changes image provenance and rebuild boundaries, so it requires a separate
design/decision record, migration plan, and equivalence tests before
implementation. It is not part of ordinary release troubleshooting.

## Versioning

Every first-party distribution carries **one coordinated version**. They are
released together and pin each other exactly, so a mixed installation cannot
resolve.

That pinning is load-bearing. Declared by bare name, `posttrain` could be
upgraded while its siblings stayed behind: an environment reporting `0.1.3`
with ten packages still at `0.1.1` is individually satisfiable, matches no
release, and gets packed into a job image as though coherent.

The candidate workflow builds the authored final version once and publishes it
only to `carbonteq/dev`. If the authored target is `0.3.17`, the candidate
distributions already contain `0.3.17`; a successful final workflow promotes
those exact bytes to `carbonteq/stable` without rebuilding. The candidate run
and its receipt are the RC identity. A PEP 440 `0.3.17rcN` distribution would
have different package metadata and therefore could not be renamed or promoted
as final `0.3.17`.

Development files are normally immutable. The only same-version retry is the
audited whole-version retirement of a failed, never-accepted candidate: its
retained receipt must match every development file, stable must contain none of
the version, and the workflow must retain a deletion receipt. Any partial,
unexplained, accepted or stable version requires a new framework version.

Three version traps, all important:

- A package that drifted out of lockstep will collide. `posttrain-catalog` was
  already published at `0.2.0` when every other package arrived at `0.2.0`, and
  its shipped catalog had changed in between — so that version would have named
  two different sets of bytes. The whole release moved to `0.2.1` rather than
  forking one package off again, which is what caused the drift originally.
- The index is **non-volatile**. A published version can never be replaced
  after acceptance or stable publication. The narrow failed-candidate
  retirement exception above requires a complete receipt and deletion audit.
  Check before uploading, not after.
- A PEP 440 RC cannot be renamed into a final release. Candidate qualification
  therefore uses the final-version bytes that promotion will expose.

## Trust and release protocol

Do this **before** publishing a candidate or cutting the final tag. Candidate
artifacts are immutable evidence, but only final publication to stable is an
accepted release.

1. **Local ladder green on the commit you intend to ship.** Unpushed commits
   are invisible to CI. If the branch is red locally, fixing it after merge is
   already too late for anyone who pulled the merge commit.
2. **Push the internal release branch** (`git push -u origin HEAD` when the
   upstream is new). Candidate workflows never execute fork revisions or
   `pull_request` events.
3. **Wait for CI green on that push.** The last green check on an older SHA
   does not cover commits that were never pushed. `MERGEABLE` on a pull
   request only means GitHub sees no conflicts — it is not “ready.”
4. **Dispatch Prepare candidate** through the protected release environment.
   It builds the authored final version once, publishes only to
   `carbonteq/dev`, qualifies changed OCI images and real jobs, and generates
   receipts. It does not create the final tag or write Python artifacts to
   stable.
5. **Repair failures on the same release branch.** Push the fix, wait for CI,
   then dispatch a new candidate run. Reuse the authored version only through
   verified whole-version retirement; otherwise advance it.
6. **If images changed, commit generated `published.toml` and image receipt
   references on the release branch.** Hand edits are forbidden. Wait for CI
   again so the exact generated manifest that will merge is verified.
7. **Merge only after a candidate passes.** Prefer the repository's normal
   merge method and required checks. Do not merge from a dirty or divergent
   local tree.
8. **Dispatch Publish release** only for the exact merged default-branch
   commit and the accepted candidate run. It restores and verifies the retained
   final-version distributions, promotes them unchanged to stable, verifies
   readback, and creates the final tag last. It does not rebuild or requalify.

GitHub Actions is the control plane: it records candidate and final commits,
approvals, logs, receipts, prereleases, the final tag, and release assets. A
repository-scoped self-hosted runner with the `lan-release` label is the
execution plane. It requires no public IP or inbound firewall rule; it polls
GitHub over outbound HTTPS and reaches `pypi.lan` and `registry.lan` over the
private network. It runs rootless BuildKit rather than a host Docker socket,
must be isolated from `ai-control`, and never runs automatic PR workflows.

## What runs locally, and what uses the protected runner

Do the ordinary source and release-preparation work locally: implement, test,
stage, build, inspect and hash assets, and run deterministic readiness checks.
CI independently verifies the pushed source. For Posttrain, the protected
runner performs the credentialed transaction: publishing retained artifacts to
the internal index, registry-backed runtime-image qualification, the accepted
remote canary, stable promotion, tagging and GitHub Release creation. It is not
a substitute for local release preparation.

This boundary is especially important for maintained forks.  Forks have no
release runner.  Their assets are built and released manually in the fork;
Posttrain's manually dispatched retained-asset publisher may only retrieve
those immutable bytes by tag, verify their hashes, publish them internally,
and prove a clean install.  See [maintained fork documentation](./tooling/forks.md).

### Cold preflight before dispatch

Run the deterministic checks locally before reserving the protected runner or
GPU:

```bash
uv sync --all-packages --locked --python 3.13
uv run posttrain-release lock-runtime-dependencies --check
uv run pytest -q \
  apps/release/tests/test_release.py \
  packages/runtime-images/tests/test_runtime_images.py
uv run pytest -q tests/consumer/test_wheel_project.py
uv run --no-sync posttrain-release readiness \
  --destination .release/readiness.json
uv run --no-sync posttrain-release readiness-check \
  .release/readiness.json
```

The runtime-lock check includes the workspace-derived kind locks; the transform
lock remains governed by `tools/quantization/uv.lock` and must also agree with
the selected maintained-fork versions. Release tests enforce parity between
the lock, package metadata and public-CI mirror URLs.

The candidate must repeat the consumer installation from `carbonteq/dev` with
`uv pip install --no-cache`. It then compares the installed job Dockerfile and
Bake definition with the retained `posttrain-runtime-images` wheel before
packing. This is intentionally redundant with local wheel testing: local tests
prove the source, while the cold index-only install proves the published bytes
and defeats stale runner cache state.

Do not dispatch until maintained-fork assets have immutable release hashes and
their required server revisions are deployed. Private-CA validation, live
service compatibility, registry readback, named hardware capacity and the real
GPU canary remain protected-runner checks because a workstation cannot prove
those external states.

If a candidate fails, classify the owning boundary before retrying. A private
CA failure belongs to runner trust configuration; an installed/retained wheel
mismatch belongs to cache and publication identity; an artifact-upload conflict
after cleanup belongs to the deployed service's metadata/blob recovery; and an
unavailable qualification profile belongs to live capacity selection. A retry
without a new proof for the failed boundary only spends runner and GPU time
again.

When a PR is already open, keep landing work on that branch until the head is
green; do not open a second PR for the same release line without a reason.

Useful checks:

```bash
git status -sb
git log --oneline origin/main..HEAD
gh pr view <n> --json state,mergeable,statusCheckRollup,url
gh pr checks <n>
```

## The sequence

1. **Set the authored version once** with
   `uv run posttrain-release prepare X.Y.Z`. Source `pyproject.toml` files stay
   at the release-neutral `0.0.0` template and keep bare workspace dependencies.
2. **Regenerate dependency locks only when dependencies changed** with
   `uv lock` followed by `uv run posttrain-release lock-dependencies`. Training
   selections reference the one generated catalog lock record rather than
   copying its digest.
   Internal packages used by runtime images have a second, deliberate phase:
   keep the last published OCI lock intact while the pull request is reviewed,
   then let the protected candidate run
   `posttrain-release lock-runtime-dependencies`. The command projects the exact
   wheel URLs and hashes already recorded in `uv.lock` without re-resolving the
   public closure. The candidate retains that generated lock beside
   `published.toml`; commit both before merge and return to strict validation.
   A retained-fork candidate uses an equivalent disposable candidate lock: its
   repository source still names the stable consumer index, while the protected
   runner resolves that named source through `carbonteq/dev` and retains the
   resulting `uv.lock`, runtime lock, and image receipt together. That lock is
   candidate evidence, never a replacement for the committed stable lock.
   After byte-identical fork promotion, run the same protected builder with
   `dependency_channel=stable`; retain and commit its generated runtime lock
   and manifest so strict default-branch validation precedes the framework
   release.
3. **Stage and inspect the target release metadata** with
   `uv run posttrain-release stage /tmp/posttrain-X.Y.Z`. Build all packages
   from that isolated tree using `uv build --all-packages --no-sources`; the
   resulting wheel metadata must contain version `X.Y.Z` and exact first-party
   `==X.Y.Z` pins.
4. **Run the full ladder.** It must be green before anything is published,
   because publishing is irreversible.
5. **Write the CHANGELOG entry**, then commit.
6. **Open or update the release PR and dispatch Prepare candidate.** The
   workflow stages the authored `X.Y.Z`, builds its distributions once, and
   publishes the receipt-listed files to `carbonteq/dev`.
7. **If the constraint lock or image inputs changed, let the candidate workflow
   publish and qualify the images.** It publishes to the registry projects
   actual jobs pull from (`registry.lan/carbonteq`). Posttrain does not use GHCR
   as a release registry. The protected workflow verifies a sanitized dstack
   capacity receipt and places the bounded canary on the known idle
   `carbonteq-ai-workstation.lan` RTX PRO worker; scheduler-reported idleness
   alone is not sufficient when unrelated host GPU processes may exist.

   ```bash
   uv run posttrain-release images publish \
     --registry registry.lan/carbonteq \
     --framework-version <version> \
     --receipt-root .posttrain/state/release-receipts
   ```

   The automated path builds kind variants concurrently with rootless BuildKit,
   pushes to the LAN registry, reads each digest back rather than predicting
   it, and regenerates `published.toml` only after the image receipt is
   accepted. The current protected workflow then runs one bounded packed
   transformation canary through dstack; a complete changed-kind matrix is a
   separate follow-up gate. Commit the regenerated manifest.

   **Base should rarely rebuild.** When the committed base still carries the
   current lock digest it is reused automatically. Force reuse of an already
   published base (including one copied onto the LAN):

   ```bash
   uv run posttrain-release images publish \
     --registry registry.lan/carbonteq \
     --framework-version <version> \
     --receipt-root .posttrain/state/release-receipts \
     --base-image registry.lan/carbonteq/posttrain-base@sha256:<digest> \
     --variant supervised \
     --variant online-rl-trl-py312 \
     --variant eval \
     --variant serve
   ```

   Unlisted kinds are reused from the committed manifest when their lock digest
   is still current, or from a matching build receipt. `--revision` pins
   `CREATED` / `SOURCE_REVISION` so receipt keys stay stable across retries.

   Push defaults skip provenance/SBOM and use zstd level 1 without
   force-recompression — attestation manifests dominated multi-GB push time.
   Opt in with `--attestations`, `--compression-level 3`, and
   `--force-compression` when a release policy requires them. Use
   `--no-parallel` only when diagnosing a Bake failure.

8. **Qualify the candidate.** Install only from `carbonteq/dev`, run the
   independent-consumer test, pack a real job, execute the changed-kind dstack
   matrix, retain Trackio evidence, and read it through Observatory. If any gate
   fails, fix the branch and return to step 6 with a new candidate run.
9. **Commit required generated image records, rerun CI, and merge the passing
   release PR.** The accepted candidate already contains the final Python
   version and binds the source, OCI inputs and retained wheelhouse.
10. **Dispatch Publish release for the merged commit and successful candidate
    run.** The runner validates source ancestry or tree equality, restores the
    retained candidate wheelhouse, rechecks its hashes and verifies the same
    files remain in `carbonteq/dev`. It does not rebuild or repeat the GPU
    canary.
11. **Promote those retained files from `carbonteq/dev` to
    `carbonteq/stable`.** Promotion is server-side: do not rebuild or perform a
    second upload. Read the stable files back and verify their hashes against
    the candidate and promotion receipts.
12. **Tag last.** After stable readback, create `v<version>` on the exact merged
    commit and create the GitHub Release with the already-retained bundle and
    receipt. If this final step fails, retry it without rebuilding or
    republishing.

### Retained maintained-fork candidates

Fork maintainers build, test, tag, and create GitHub Release assets locally.
They do not run release automation or receive private-index credentials in the
fork. Posttrain's retained-asset candidate publisher may only download those
already released assets by tag, verify caller-supplied hashes, and upload them
to `carbonteq/dev`. It must prove the files are stored by `dev`, not merely
inherited from `stable`, before a candidate job can consume them.

After the candidate's consumer, image, and workload gates pass, Posttrain's
separate promotion workflow re-verifies the `dev` files and uses a server-side
transfer to `carbonteq/stable`. It reads the stable files back against the same
hashes. No promotion rebuilds, re-uploads, or runs fork source.

## After a release

- **Existing job images do not gain runtime changes.** Anything living in the
  job runtime — the certificate merge, package verification — reaches a job only
  when its image is repacked. Tell consumers to re-run `posttrain job pack`.
- **The Observatory ships separately** from `ai-infra`
  (`scripts/package-observatory`, `scripts/deploy-observatory`). It is built
  from a framework commit, so a framework release does not update it. Confirm
  the Trackio server exposes the APIs the Observatory client calls (for example
  `/get_run_lifecycles`) before deploying a new Observatory image.

## Things that will bite you

- **Publish after merging, not before.** A version on the index whose tree is
  uncommitted or unmerged has no reviewable source. Publish-after-commit was
  gotten wrong once; publish-before-merge is the same class of mistake.
- **Build distributions from the staged tree, not the source checkout.** The
  source workspace intentionally declares `0.0.0` and bare first-party
  dependencies. Only `posttrain-release stage` renders the authored release
  version and exact sibling pins that are safe to upload.
- **`MERGEABLE` is not ready.** No conflicts ≠ CI green ≠ local ladder green.
- **Unpushed commits are unverified.** CI evaluates the last push, not your
  working tree.
- **`uv sync` alone does not install workspace members.** Use
  `uv sync --all-packages`, or the tree looks broken in ways that have nothing
  to do with your change.
- **A pack can fail with `package manifest key differs from PACKAGE_KEY`.** That
  is a stale BuildKit cache, not a real mismatch — the staged context is
  verifiable as correct. Retry; if it persists, `docker buildx prune -af`
  (and expect a cold rebuild — prefer registry `cache-from` / `--base-image`
  over pruning when you can).
- **Do not rewrite plan or decision records** to match a new version. They
  describe what was true when written.
- **Do not upload directly to `stable`.** Qualification happens from `dev`, and
  stable receives the same accepted files through promotion. A direct stable
  upload bypasses the evidence gate and cannot be undone.
- **Do not rebuild between channels.** The development index, stable index, and
  GitHub Release must agree with one receipt. A locally rebuilt wheel is a new
  artifact even when its version and source commit appear equal.
- **Do not casually overwrite a failed candidate.** Reuse of the authored
  version is allowed only through audited whole-version retirement before any
  acceptance or stable publication. Otherwise advance the version.
- **Do not build a PEP 440 RC and call it final.** Candidate package metadata
  and exact first-party pins already name `X.Y.Z`; final publication promotes
  those retained bytes unchanged.
- **Do not assume an older LAN tag is the new base.** Confirm the digest
  (`imagetools inspect registry.lan/carbonteq/posttrain-base@sha256:…`) before
  passing `--base-image`.
