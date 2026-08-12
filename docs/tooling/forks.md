# Maintained fork documentation

This workspace may depend on a maintained fork when an upstream release does
not yet provide behavior required by a reusable backend. A fork is part of the
runtime supply chain, so its delta must be understandable without chat history
or access to the consuming application.

## Two linked records

Every maintained fork has two documentation homes:

1. The consuming workspace keeps `docs/tooling/<tool>/README.md`. It explains
   why the framework selects the fork, the supported integration boundary,
   the exact immutable pin, operational configuration, qualification evidence,
   and the remaining release gates.
2. The fork keeps `CARBONTEQ_FORK.md` at its repository root. It explains the
   upstream base, the complete maintained delta, affected source and test
   surfaces, compatibility constraints, validation commands, and how to rebase
   or retire the delta.

Neither record replaces the other. Consumer-only notes leave fork maintainers
without context; fork-only notes do not explain how the framework selects and
operates the dependency.

## Required fork ledger

`CARBONTEQ_FORK.md` must contain:

- fork status: candidate, unpublished, published, or retired;
- upstream repository and exact base commit;
- expected `origin` and `upstream` remotes;
- one entry for each maintained behavioral change, naming concrete source and
  regression-test files;
- why upstream behavior is insufficient and which behavior must survive an
  upstream merge;
- compatible dependency/runtime constraints;
- focused validation commands and any hardware-only release gates;
- upstream-rebase procedure and conflict-sensitive areas;
- immutable fork commit for every published release;
- deferred or intentionally unsupported behavior.

An experimental dirty checkout may record a candidate delta, but the consumer
must not describe it as a reproducible fork release. Before a production pin,
the delta must be committed in the fork, pushed to its maintained remote, and
selected by full commit hash. A base `HEAD` plus uncommitted patches is valid
research evidence only.

## Change workflow

### Publication boundary

Maintained forks are released manually from the fork checkout.  The maintainer
builds and validates the distributions locally, creates an immutable GitHub
release whose assets and SHA-256 values are retained, then records the exact
tag and commit in both fork records.  A fork does **not** receive a release
runner, private-index credentials, or a fork-controlled publication workflow.

When Posttrain needs the released fork on the internal index, a maintainer
manually dispatches the repository-owned retained-asset publisher from
Posttrain.  That narrow workflow downloads the already-released assets by
immutable tag, verifies the supplied hashes, publishes those same bytes, and
performs a clean install/readback.  It never builds fork source and never
executes fork workflow code.

For a fork-backed implementation, make changes in this order:

1. Resolve and record the upstream base commit and current remotes.
2. Check upstream issues and pull requests for duplicate work, following the
   fork's own contribution instructions.
3. Implement the generic library change and regression tests in the fork.
4. Update the fork's `CARBONTEQ_FORK.md` in the same change.
5. Run the fork's focused tests and hardware gate where applicable.
6. Commit and push the fork change.
7. Update the consuming workspace's immutable dependency pin and lockfile.
8. Update `docs/tooling/<tool>/README.md` with the selected fork commit,
   operational settings, and qualification evidence.
9. Run the consuming workspace's integration and boundary tests.

Do not mix uncommitted changes from multiple repositories into one implied
commit. A handoff or pull request must name each repository, commit order,
validation command, and pin transition.

## Revision rule

Update both records whenever a patch is added, removed, superseded upstream,
rebased, or found to require a new runtime constraint. Revise the existing
documents rather than creating dated status notes. Qualification artifacts may
remain immutable; the tooling page should link the currently relevant run.
