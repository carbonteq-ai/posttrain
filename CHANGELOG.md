# Changelog

All notable changes to Posttrain are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a coordinated
version across first-party distributions.

## Unreleased

## 0.2.1 - 2026-07-28

The framework was qualified from a library consumer's seat for the first
time: installed from an index, with no checkout on the machine. Eleven
defects stood between that developer and a finished job, none of which the
test suite could see, because it runs from a checkout where the source tree
exists, every package is one version, and the build definitions are inside
the project.

### Added

- An internal package index as the supported distribution channel, with the
  maintained forks constrained explicitly because uv does not resolve a
  transitive direct URL implicitly.
- `posttrain job diff`, explaining why two packed job packages differ.
- A `trust` readiness check reporting which certificate authority reaches
  jobs, and warning when it is absent from the machine's own store.
- `docs/consumer-setup.md`, written from steps that were executed rather
  than imagined.
- Primary-CLI work-package execution through an explicit project host.
- Apache-2.0 licensing, private security reporting, compatibility policy, and
  upgrade guidance.
- A reproducible remote GPU release-gate workflow.

### Changed

- Framework packages pin each other exactly. Declared by bare name, they
  allowed `posttrain` to be upgraded while every sibling stayed behind, which
  was individually satisfiable, matched no release, and was packed into job
  images as though coherent.
- A job image obtains framework code as built distributions when there is no
  checkout, rather than requiring twelve source directories to copy.
- Additional certificate authorities are merged with those the job image
  already trusts, and resolved from `/etc/posttrain/trust/internal-ca.pem`
  when nothing is configured. Execution providers no longer set
  `SSL_CERT_FILE`, which replaced the trust store rather than extending it.

### Fixed

- A run that died before opening a tracking run held its machine's admission
  placement permanently, and cancel, cleanup, and reconcile each refused it
  for a different reason.
- Tracking evidence was written to the configured server but looked for in a
  local one, so a succeeded run reconciled as pending with no artifacts.
- Writing `execution.toml` for the local provider's hostname discarded
  `POSTTRAIN_REGISTRY`.
- A failed provider submission reported only that its outcome was unresolved
  and to retry, naming nothing to act on.
- `release/github-constraints.txt` omitted `trl` and `verifiers`, so the
  documented install could not resolve.

### Note

Versions 0.1.1 through 0.1.13 were development builds published while
qualifying this work, because packing downloads framework wheels from the
index and a fix could not be tested until it was published. They are not
supported releases. Development builds now stage on a separate index.

## 0.1.0-rc.1 - 2026-07-23

### Added

- Portable `.posttrain` project discovery, initialization, catalog overlays,
  work packages, and ignored machine-local state.
- The `posttrain` CLI for diagnostics, project inspection, catalog inspection,
  and composition validation.
- Versioned framework catalog resources and reusable data, train, evaluation,
  serving, tracking, and work-composition packages.
- Trackio and W&B tracking adapters with provider-neutral evidence contracts.
- Observatory Python, HTTP, MCP, report, and frontend surfaces.
- Installed-wheel consumer acceptance with real local Trackio persistence and
  Observatory readback.
- GitHub Release wheelhouses with immutable fork constraints and SHA-256
  checksums.

[Unreleased]: https://github.com/carbonteq-ai/posttrain/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/carbonteq-ai/posttrain/compare/v0.1.0-rc.1...v0.2.1
[0.1.0-rc.1]: https://github.com/carbonteq-ai/posttrain/releases/tag/v0.1.0-rc.1
