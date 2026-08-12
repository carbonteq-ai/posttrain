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

The protected workflow implementation and the LAN runner are now present on
the release branch, but the GitHub `release-candidate` and `release`
environments still need their required-reviewer rules and the first merged
dispatch. Do not use any older tag-triggered workflow to publish `v0.3.1`: a
GitHub-hosted runner cannot resolve `pypi.lan`, and tagging first would reverse
the required release order.

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

## Versioning

Every first-party distribution carries **one coordinated version**. They are
released together and pin each other exactly, so a mixed installation cannot
resolve.

That pinning is load-bearing. Declared by bare name, `posttrain` could be
upgraded while its siblings stayed behind: an environment reporting `0.1.3`
with ten packages still at `0.1.1` is individually satisfiable, matches no
release, and gets packed into a job image as though coherent.

Candidate versions use PEP 440 prerelease identifiers. If the authored target
is `0.3.1`, the protected workflow allocates `0.3.1rc1`, `0.3.1rc2`, and so on.
Candidate files are published only to `carbonteq/dev` and never overwritten.
They let maintainers repair the release branch without consuming `0.3.1` or
creating a misleading final tag.

Three version traps, all important:

- A package that drifted out of lockstep will collide. `posttrain-catalog` was
  already published at `0.2.0` when every other package arrived at `0.2.0`, and
  its shipped catalog had changed in between — so that version would have named
  two different sets of bytes. The whole release moved to `0.2.1` rather than
  forking one package off again, which is what caused the drift originally.
- The index is **non-volatile**. A published version can never be replaced.
  Check before uploading, not after.
- An RC cannot be renamed into a final release. `0.3.1rc2` is embedded in wheel
  metadata and internal dependency pins, so final `0.3.1` is a separate
  build-once artifact set from the accepted merged commit.

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
   It derives the next RC, publishes only to `carbonteq/dev`, qualifies changed
   OCI images and real jobs, and generates receipts. It does not create the
   final tag or write Python artifacts to stable.
5. **Repair failures on the same release branch.** Push the fix, wait for CI,
   then dispatch the next RC. Do not replace a prior candidate or advance the
   final target version merely because an RC failed.
6. **If images changed, commit generated `published.toml` and image receipt
   references on the release branch.** Hand edits are forbidden. Wait for CI
   again so the exact generated manifest that will merge is verified.
7. **Merge only after a candidate passes.** Prefer the repository's normal
   merge method and required checks. Do not merge from a dirty or divergent
   local tree.
8. **Dispatch Publish release** only for the exact merged default-branch
   commit. It builds and qualifies final distributions, promotes to stable,
   verifies readback, and creates the final tag last.

GitHub Actions is the control plane: it records candidate and final commits,
approvals, logs, receipts, prereleases, the final tag, and release assets. A
repository-scoped self-hosted runner with the `lan-release` label is the
execution plane. It requires no public IP or inbound firewall rule; it polls
GitHub over outbound HTTPS and reaches `pypi.lan` and `registry.lan` over the
private network. It runs rootless BuildKit rather than a host Docker socket,
must be isolated from `ai-control`, and never runs automatic PR workflows.

## What runs locally, and what uses the protected runner

Do the ordinary release work locally: implement, test, stage, build, inspect
and hash release assets, create the GitHub release, and run deterministic
readiness checks.  CI independently verifies the pushed source.  The protected
Posttrain runner is deliberately narrow: it performs actions that need private
LAN reachability or credentials, namely publishing already-retained artifacts
to the internal index, registry-backed runtime-image qualification, and the
accepted remote canary.  It is not a substitute for local release preparation.

This boundary is especially important for maintained forks.  Forks have no
release runner.  Their assets are built and released manually in the fork;
Posttrain's manually dispatched retained-asset publisher may only retrieve
those immutable bytes by tag, verify their hashes, publish them internally,
and prove a clean install.  See [maintained fork documentation](./tooling/forks.md).

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
3. **Stage and inspect the target release metadata** with
   `uv run posttrain-release stage /tmp/posttrain-X.Y.Z`. Build all packages
   from that isolated tree using `uv build --all-packages --no-sources`; the
   resulting wheel metadata must contain version `X.Y.Z` and exact first-party
   `==X.Y.Z` pins.
4. **Run the full ladder.** It must be green before anything is published,
   because publishing is irreversible.
5. **Write the CHANGELOG entry**, then commit.
6. **Open or update the release PR and dispatch Prepare candidate.** The
   workflow derives the next unused `X.Y.ZrcN`, stages that version without
   changing the authored target, builds its distributions once, and publishes
   the receipt-listed files to `carbonteq/dev`.
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
   fails, fix the branch and return to step 6 with the next RC.
9. **Commit generated image records, rerun CI, and merge the passing release
   PR.** The accepted candidate is evidence for the source and OCI inputs; it
   is not renamed into the final Python version.
10. **Dispatch Publish release for the merged commit.** The runner stages final
    `X.Y.Z`, builds every final wheel and source distribution once, writes a
    hash-addressed receipt, uploads those exact files to `carbonteq/dev`, and
    runs an index-only install plus final dstack canary.
11. **Promote the qualified final files from `carbonteq/dev` to
    `carbonteq/stable`.** Promotion is server-side: do not rebuild or perform a
    second upload. Read the stable files back and verify their hashes against
    the release receipt.
12. **Tag last.** After stable readback, create `v<version>` on the exact merged
    commit and create the GitHub Release with the already-retained bundle and
    receipt. If this final step fails, retry it without rebuilding or
    republishing.

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
- **Do not overwrite a failed RC.** Fixes receive the next candidate number.
  This keeps workflow artifacts, index files, OCI receipts, and qualification
  evidence unambiguous.
- **Do not promote an RC as the final version.** Final package metadata and
  exact first-party pins must name `X.Y.Z`, so the merged commit produces one
  separately qualified final artifact set.
- **Do not assume an older LAN tag is the new base.** Confirm the digest
  (`imagetools inspect registry.lan/carbonteq/posttrain-base@sha256:…`) before
  passing `--base-image`.
