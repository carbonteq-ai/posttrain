# posttrain-runtime-images

Ships the framework's container definitions as package data so an installed
wheel carries the inputs that produced its runtime images, not just a reference
to them.

Three image levels live under `src/posttrain/runtime_images/containers/`:

| Directory | Level | Published |
| --- | --- | --- |
| `posttrain-base/` | universal base | once per framework release |
| `posttrain-job-kinds/` | per-variant job-kind images | once per framework release |
| `posttrain-job/` | actual-job image | per job, by `posttrain job pack` |

The nested `containers/` path segment is load-bearing. Every Dockerfile refers
to its inputs as `containers/posttrain-job-kinds/locks/...`, and `docker buildx
bake` resolves a target's `dockerfile` relative to its `context`. Pointing the
build context at the directory holding `containers/` therefore keeps every path
in the shipped Dockerfiles and bake files valid without editing them, whether
the build runs from a source checkout or from an installed wheel.

`validate.py` under `posttrain-job-kinds/` remains a framework-repo check: it
compares runtime profiles against the workspace `uv.lock`, which does not exist
in an installed wheel. The consumer-side equivalent is the drift check in
`posttrain doctor`, which compares a published image's recorded lock digest
against the lock shipped here.

This package owns definitions and the published manifest only. It does not
build images, push them, or hold registry credentials.
