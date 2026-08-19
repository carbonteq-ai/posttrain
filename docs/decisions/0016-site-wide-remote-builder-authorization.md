# ADR 0016 — Remote-builder credentials authenticate site infrastructure

## Status

Accepted.

Date: 2026-08-19
Deciders: Posttrain framework and AI infrastructure maintainers
Related Plan: `docs/plan/site-wide-remote-builder-authorization.md`
Supersedes: the project-scoped authorization decision recorded in
`docs/plan/portable-runtime-image-supply-chain.md`
Superseded By: None

## Context

The optional Posttrain developer job builder moves actual-job OCI image
construction from a developer machine to infrastructure-managed rootless
BuildKit. Its first implementation maps each bearer-token digest to a
`PrincipalGrant` containing an explicit set of allowed `project_id` values.
That model confuses infrastructure authentication with project identity. A
developer must ask an operator to expand a token whenever a new Posttrain
project is created even though the service, registry, build definition, trust
boundary, and infrastructure principal have not changed.

A Posttrain `project_id` is still security-relevant untrusted input. It scopes
the repository selected by the service, durable build records, receipts, and
audit facts. It must be validated and must never let a client choose an
arbitrary registry repository. Those requirements do not imply that a secret
belongs to the project or that the server needs a credential-to-project
allowlist.

The bearer secret also cannot belong to the framework distribution. Framework
packages are portable across sites and users; embedding a site secret would
make it broadly shared, difficult to rotate, and impossible to revoke without
republishing unrelated framework bytes. The correct owner is the deployed
infrastructure site, with a protected reference in machine configuration.

## Decision

1. A remote-builder bearer token authenticates one site infrastructure
   principal. It carries no Posttrain project allowlist.
2. The client references that credential once from the operator-owned machine
   configuration under `[services.job_builder]`. Raw token bytes remain in a
   mode-`0600` credential file outside every project and framework package.
3. Every publication request continues to carry the immutable manifest's
   `project_id`. The server validates it as an untrusted namespace label and
   uses it for repository derivation, request isolation, receipts, cache and
   audit evidence.
4. The server derives the only permitted OCI repository from its configured
   site repository prefix and validated `project_id`. Authentication identity
   must not alter image identity. A client-supplied repository must match
   exactly before any blob is admitted.
5. Protected server configuration maps token digests to infrastructure
   principals only. Multiple digests may temporarily map to the same principal
   solely to permit non-disruptive credential rotation; this does not create
   project credentials.
6. The old `token_grants` configuration containing `project_ids` is rejected
   rather than silently broadened. Deployment must replace it atomically with
   `infrastructure_grants` when the service code is upgraded.
7. Distinct infrastructure credentials are permitted only when independent
   operational attribution or revocation is required, such as separating a CI
   principal from developer machines. Project creation alone never requires a
   new credential.

## Consequences

Creating or renaming a Posttrain project no longer requires an infrastructure
authorization change. A credential compromise has site-principal scope, which
matches the intended trust boundary and makes protecting and rotating that
credential an infrastructure responsibility. Project namespaces remain
isolated in durable service state and registry paths, but they are not access
control lists.

The server configuration schema changes incompatibly. This is deliberate: an
old project-scoped grant cannot safely become site-wide through an implicit
parser fallback. The live service therefore needs one coordinated code and
protected-config deployment with a retained rollback copy. Client machine
configuration does not change except for replacing misleading credential
aliases such as `job-builder-ambient` with an infrastructure-level name.

## Alternatives Considered

### Keep one token and enumerate every project ID

Rejected because it preserves the wrong ownership boundary. A new project
would still require infrastructure mutation, and an omitted ID would surface
as an authorization incident even though the caller is already trusted to use
the site builder.

### Embed one global credential in the framework release

Rejected because portable framework bytes must not contain a site secret.
Rotation and revocation would become coupled to framework publication and all
installations would share one credential.

### Accept any client repository after site authentication

Rejected because project source and small packed datasets can be private. The
service must retain exclusive control of registry destination and push
credentials even when authorization is site-wide.

### Remove `project_id` from the remote-builder protocol

Rejected because project identity is part of the immutable job manifest and is
needed for repository derivation, durable isolation, receipts, and audit. It
is a namespace, not a credential scope.

## Implementation Notes

- Replace `PrincipalGrant(project_ids=...)` with an infrastructure grant that
  contains only the validated principal name.
- Replace protected server JSON `token_grants` with
  `infrastructure_grants`; reject configurations containing the old field.
- Keep request and receipt storage scoped by infrastructure principal,
  `project_id`, and publication key.
- Derive client and server repositories as
  `<site-prefix>/<project_id>/posttrain-job`; keep the infrastructure principal
  only in durable service state and audit evidence.
- Add conformance coverage proving that one token can publish two different
  project namespaces while each arbitrary repository is rejected.
- Complete the `ai-infra` Ansible role so the protected configuration,
  rootless BuildKit, API unit, Caddy route, deployment provenance, rollback,
  and live qualification are reproducible from committed source.

## Revision History

- 2026-08-19: Initial accepted decision. Reason: move credential ownership to
  the infrastructure trust boundary while retaining project namespace
  isolation and server-owned repository policy.
