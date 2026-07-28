# Publishing a release

This is the maintainer's runbook. Consuming a release is
[`consumer-setup.md`](./consumer-setup.md); the services a release is published
to are operated from the `ai-infra` repository.

A release is two artifacts that must agree: the **distributions** on the
internal index, and the **runtime images** in the registry. Publishing one
without the other produces a framework that refuses to run, by design — the
manifest records which constraint lock each image was built against, and every
command checks it.

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

That error fails 47 tests at once and blocks `doctor`, `job pack`, and
`job run`. It is the guard working, not a bug.

Rebuilding is cheaper than it sounds: the base layers are shared and already in
the registry, so only the kind-specific layer re-uploads — roughly 0.9 GB per
variant rather than the 6 GB each image measures.

## Versioning

Every first-party distribution carries **one coordinated version**. They are
released together and pin each other exactly, so a mixed installation cannot
resolve.

That pinning is load-bearing. Declared by bare name, `posttrain` could be
upgraded while its siblings stayed behind: an environment reporting `0.1.3`
with ten packages still at `0.1.1` is individually satisfiable, matches no
release, and gets packed into a job image as though coherent.

Two version traps, both hit in practice:

- A package that drifted out of lockstep will collide. `posttrain-catalog` was
  already published at `0.2.0` when every other package arrived at `0.2.0`, and
  its shipped catalog had changed in between — so that version would have named
  two different sets of bytes. The whole release moved to `0.2.1` rather than
  forking one package off again, which is what caused the drift originally.
- The index is **non-volatile**. A published version can never be replaced.
  Check before uploading, not after.

## The sequence

1. **Bump every workspace member** — root, `apps/*`, `packages/*` — to the same
   version.
2. **Re-pin siblings** so each intra-workspace dependency names the new version
   exactly. A test derives this from the workspace, so a missed pin fails the
   ladder rather than shipping.
3. **`uv lock`**, then realign the catalog. `packages/catalog/src/posttrain/
   catalog/base/training.yaml` records `dependency_lock_sha256`, the hash of
   `uv.lock`. It drifts on every bump and fails one lab test until corrected.
4. **Run the full ladder.** It must be green before anything is published,
   because publishing is irreversible.
5. **Write the CHANGELOG entry**, then commit.
6. **Build and publish the distributions.**

   ```bash
   uv build --all-packages
   ```

   ```bash
   uv publish --system-certs --publish-url https://pypi.lan/carbonteq/stable/ dist/*
   ```

   Development builds go to `carbonteq/dev` instead. Publish to `stable` only
   for a tagged release: packing a job image downloads these distributions from
   the index, so a fix cannot be tested until it is published, and that pressure
   is what put thirteen development versions on `stable` once already.
7. **Tag the release** and push the tag. Without it the wheels on the index have
   no traceable source.
8. **If the constraint lock changed, republish the images:**

   ```bash
   uv run posttrain-release images publish \
     --registry ghcr.io/carbonteq-ai \
     --framework-version <version> \
     --receipt-root .posttrain/state/release-receipts
   ```

   This builds every variant, pushes, reads each digest back from the registry
   rather than predicting it, and regenerates `published.toml`. Commit the
   regenerated manifest.
9. **Mirror into any registry projects pull from**, by digest:

   ```bash
   posttrain runtime images mirror --from ghcr.io/carbonteq-ai --registry registry.lan/carbonteq
   ```

10. **Re-run the ladder.** The 47 failures from step 8's guard should be gone.

## After a release

- **Existing job images do not gain runtime changes.** Anything living in the
  job runtime — the certificate merge, package verification — reaches a job only
  when its image is repacked. Tell consumers to re-run `posttrain job pack`.
- **The Observatory ships separately** from `ai-infra`
  (`scripts/package-observatory`, `scripts/deploy-observatory`). It is built
  from a framework commit, so a framework release does not update it.

## Things that will bite you

- **Publish after committing, not before.** A version on the index whose tree is
  uncommitted has no traceable source. This ordering was gotten wrong once and
  needed a commit written afterwards to explain the artifacts.
- **`uv sync` alone does not install workspace members.** Use
  `uv sync --all-packages`, or the tree looks broken in ways that have nothing
  to do with your change.
- **A pack can fail with `package manifest key differs from PACKAGE_KEY`.** That
  is a stale BuildKit cache, not a real mismatch — the staged context is
  verifiable as correct. Retry; if it persists, `docker buildx prune -af`.
- **Do not rewrite plan or decision records** to match a new version. They
  describe what was true when written.
