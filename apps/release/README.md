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

The manifest is generated, never edited. Hand-maintaining image digests and
lock hashes is what previously allowed a published job-kind image and the
framework that used it to disagree without anything noticing.
