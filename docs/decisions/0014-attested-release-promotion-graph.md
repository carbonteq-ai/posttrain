# ADR 0014 — Releases are attested promotion graphs

## Status

Proposed.

Date: 2026-08-09
Deciders: Posttrain framework and AI infrastructure maintainers
Related Plan: `docs/plan/sft-dapo-256-experiment-and-framework-release.md`
Supersedes: None
Superseded By: None

## Context

Posttrain releases cross several independently authoritative systems. Git owns
reviewed source; the internal Python index owns distributions; the OCI registry
owns runtime-image digests; Trackio owns observation and artifact persistence;
dstack owns remote execution state; and Observatory reads the resulting
evidence. The current candidate workflow performs many of those transitions in
one job and relies on generated files in the checkout to connect them.

The `0.3.3` candidates exposed the resulting ambiguity:

- wheelhouse construction initially preceded OCI publication, so the wheel
  shipped an old image manifest;
- moving OCI publication earlier still failed because `git archive HEAD`
  correctly excluded the generated manifest from staged source;
- the real job then reached Trackio but exposed a client/server artifact
  compatibility defect that had not been qualified as a dependency release;
- an attempted Trackio internal-publication workflow could not run because the
  LAN runner is intentionally scoped to the Posttrain repository.

These are ownership and attestation failures, not reasons to weaken the
canaries. A successful component build does not prove package publication,
deployment, live configuration, or compatibility. A generated file in a dirty
checkout is not an immutable release input.

## Decision

Model a platform release as a directed graph of independently attested
promotions.

1. Every maintained dependency, beginning with CarbonTeq Trackio, owns its
   source build and release receipt. The receipt binds repository, source
   commit, version, filenames, SHA-256 hashes, internal-index readback, service
   image digest when applicable, deployed version/configuration, and canary
   evidence.
2. Dependency publication and deployment complete before the framework
   candidate starts. Posttrain verifies dependency receipts; it does not
   publish another repository's package or deploy its service as an incidental
   candidate step.
3. LAN release execution is granted per repository through a protected
   environment and a runner group restricted to approved release workflows.
   Credentials stay on the runner and are never passed through Posttrain or
   workflow inputs.
4. Framework release staging has two immutable input classes: committed source
   from `git archive`, and declared generated release inputs. OCI image receipts
   and `published.toml` are written outside the source tree, hashed in a release
   materialization receipt, and passed explicitly to the staging command.
   Arbitrary working-tree mutations remain excluded.
5. The framework candidate consumes one materialization receipt that binds the
   source SHA, target/candidate version, dependency receipts, lock digests,
   Python artifacts, OCI image digests, and qualification evidence.
6. Candidate qualification is layered: clean index-only install, runtime-image
   readback, a provider-neutral artifact round trip through the deployed
   Trackio service, a bounded packed dstack job, and Observatory readback from
   that same run. Each layer reports its own failure.
7. Final publication promotes exact retained bytes. Stable-index readback and
   deployment/read-product checks happen before the final tag and GitHub
   Release. A failed candidate allocates a new RC; an accepted stable version is
   never overwritten.
8. Framework runtime-image dependency profiles and Python package metadata
   derive from one resolved dependency lock. CI rejects duplicated Trackio or
   other maintained-fork versions that disagree.

## Consequences

- A framework release cannot begin while Trackio is merely committed or built;
  it needs published, deployed, and live-qualified dependency evidence.
- Trackio needs its own protected internal-release execution path. Granting the
  Posttrain runner to arbitrary Trackio PR code is not acceptable; only a
  protected tag workflow may receive index credentials.
- Candidate workflows become shorter and their failures are attributable to a
  named edge in the graph.
- The release tooling gains a materialization receipt and explicit staging
  input instead of copying generated files into a dirty checkout.
- Observatory becomes a real release gate, not only a package included in the
  wheelhouse. Its version/digest and run readback are retained with the release.
- More receipts are retained, but they replace undocumented cross-system state
  and make retries deterministic.
- Existing RCs and unreferenced OCI digests remain immutable failed evidence;
  they are not repaired in place.

## Alternatives Considered

### Keep extending the single Posttrain candidate workflow

Rejected because it makes Posttrain the accidental publisher and deployer for
other repositories, broadens credential exposure, and cannot prove which
dependency deployment a canary exercised.

### Copy the generated OCI manifest into the source checkout before staging

Rejected as the durable contract. A narrow overlay fixes one symptom but still
uses mutable workspace state as an implicit build input.

### Allow the candidate to build missing runtime images or upload missing blobs

Rejected because qualification would silently mutate the system it is
supposed to inspect and could make a failed release appear healthy.

### Treat Trackio and Observatory as optional external health checks

Rejected for this release. Artifact publication and evidence readback are part
of the supported framework workflow, so their compatibility is release
behavior.

## Implementation Notes

- Add a provider-neutral release materialization model under
  `apps/release/src/posttrain_release` and make
  `scripts/release/build-python-distributions` accept the materialization path.
- Make `posttrain-release stage` project only the declared generated files into
  the staged tree and validate their hashes, framework version, source
  revision, and lock digests.
- Add a maintained-dependency receipt file under `release/` and validate it in
  `posttrain-release check`.
- Enforce one Trackio version across `packages/tracking-trackio/pyproject.toml`,
  `uv.lock`, and runtime-image profiles/locks.
- Give `carbonteq-ai/trackio` a protected LAN publication/deployment workflow
  and retain its index and service receipts with the GitHub Release.
- Add a Trackio service canary covering direct multipart upload, artifact
  manifest commit, download, server restart, stale completed-session recovery,
  and purge/retention behavior against a dedicated prefix.
- Extend the Posttrain candidate with a small artifact round trip and
  Observatory readback before the packed transform canary is accepted.
- Remove the temporary cross-repository Trackio publisher from the Posttrain
  release branch once the owning Trackio workflow is operational.

## Revision History

- 2026-08-09: Initial proposed decision after the `0.3.3` candidate exposed
  implicit generated inputs, missing dependency promotion evidence, and a live
  Trackio client/server artifact incompatibility.
