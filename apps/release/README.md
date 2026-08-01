# posttrain-release

Framework-owner tooling. Builds the universal base and job-kind images, pushes
them to the framework's public release registry, reads the resulting digests
back from that registry, and rewrites the `published.toml` that ships inside
`posttrain-runtime-images`.

This is deliberately **not** part of the `posttrain` distribution. A consumer
installs `posttrain`, which does not depend on this package, so publishing a
release is unreachable from a consumer environment. Consumers pull, mirror, or —
only where neither is possible — rebuild locally; none of those operations can
alter what a release claims to be.

`release/manifest.toml` is the only authored release version. Workspace
projects remain release-neutral (`0.0.0` with bare first-party dependencies),
while `posttrain-release stage DESTINATION` renders standards-readable versions
and exact sibling pins into an isolated release tree. Wheels and sdists are
built from that tree, so a version bump does not rewrite every source
`pyproject.toml`.

`posttrain-release lock-dependencies` generates the catalog's single named
dependency-lock record from the pinned TRL source revision and `uv.lock`.
Image digests remain captured outputs: the published image manifest is
generated from registry readback and is never edited by hand.
