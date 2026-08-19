# Make remote-builder authorization site-wide and reproducible

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

After this change, one infrastructure-owned remote-builder credential can
build immutable actual-job images for every valid Posttrain project namespace
at the site. Creating a project does not require issuing a credential or
editing a server allowlist. Project identity remains visible in registry paths,
durable request state, receipts, and audit evidence, and the server still
rejects arbitrary repositories before accepting any source blob.

The same change makes the deployed service reproducible. The Posttrain
repository owns the client, admission, queue, and builder behavior. The sibling
`/home/hammad/projects/ai-infra` repository owns the builder VM, rootless
BuildKit, protected server configuration, systemd units, Caddy ingress,
deployment, rollback, and live qualification. No raw credential is committed.

## Progress

- [x] (2026-08-18 21:22Z) Read the canonical remote-builder contract, current
  implementation, tests, live machine configuration, and live deployment.
- [x] (2026-08-18 21:22Z) Record ADR 0016 and amend the frozen canonical
  baseline before implementation.
- [x] (2026-08-19 09:10Z) Replace project-scoped grants with
  infrastructure-scoped grants, derive repository identity from the site
  prefix plus `project_id`, and add multi-project conformance tests.
- [x] (2026-08-19 09:10Z) Complete the `ai-infra` job-builder role, protected
  configuration model, immutable-source deploy script, and read-only
  qualification playbook without committing secrets.
- [x] (2026-08-19 09:18Z) Run focused Posttrain and ai-infra validation and
  record the observed evidence. The broader Posttrain suite has one unrelated
  catalog lock-hash failure, described below.
- [x] (2026-08-19 10:04Z) Commit and push Posttrain on
  `codex/site-wide-builder-auth`, then commit and push `ai-infra` against the
  immutable Posttrain source. The deployed Posttrain source revision is
  `c65e346df1b909790a76b0781641266e14ceedcc`.
- [x] (2026-08-19 10:38Z) Deploy and qualify the live service from committed
  revisions. The protected receipt records ai-infra revision
  `80cca2aaac6e9a3ec1e033ed34e89d73f5591320`, the Posttrain revision above,
  archive SHA-256, and `qualification: succeeded`.
- [ ] (2026-08-19 06:13Z) Publish the accepted implementation as Posttrain
  v0.3.20, update ai-infra to the immutable release revision, redeploy, and
  qualify the released source rather than the pre-release commit.
- [x] (2026-08-19 06:27Z) Prepare and qualify the retained v0.3.20 candidate in
  protected workflow run `32223062319`. Exact-source CI, the no-cache
  index-only consumer install, and the real packed RTX PRO job all passed; the
  development-index wheelhouse and evidence are retained for byte-identical
  promotion.

## Surprises & Discoveries

- Observation: the credential that appeared project-local is actually under
  `/home/hammad/.config/posttrain/credentials`, outside every project, but its
  `job-builder-ambient` alias misleadingly suggests project ownership.
  Evidence: `/home/hammad/.config/posttrain/config.toml` references that file
  from the machine-level `[services.job_builder]` table.
- Observation: the live Caddy route and builder are deployed even though their
  complete infrastructure source is not committed.
  Evidence: live `ai-control` routes `/job-builder/*` to
  `192.168.110.117:8080`; `ai-builder` runs both systemd units, while local
  `ai-infra` has a modified Caddy template and untracked partial role.
- Observation: the live API is installed editable from
  `/opt/posttrain-job-builder`, and its checkout has a modified runtime image
  manifest. This is not an immutable deployment boundary.
  Evidence: the `posttrain_job_builder` distribution `direct_url.json` reports
  an editable source and `git status` on the host reports the manifest change.
- Observation: the current local client credential file is protected but does
  not contain a valid `POSTTRAIN_JOB_BUILDER_TOKEN` assignment, so an
  authenticated capabilities request returns HTTP 401.
- Observation: the authenticated principal cannot participate in OCI
  repository identity. Doing so would make a routine token rotation or
  principal rename change image names even when the project and publication
  are unchanged.
  Evidence: repository derivation now uses only the configured site prefix and
  validated `project_id`; the principal remains in durable queue and audit
  paths.
- Observation: the full Posttrain suite currently has one failure unrelated to
  this change: the Qwen PEFT catalog records dependency-lock SHA-256
  `dd326c...`, while the current `uv.lock` hashes to `583db4...`.
  Evidence: `uv run pytest -q` completed with 1 failed, 1261 passed, and 23
  skipped; the failure is
  `apps/lab/tests/test_catalog.py::test_peft_bindings_settings_and_quantization_load_from_filesystem_catalog`.
- Observation: repository-wide Pyright also has one unrelated pre-existing
  adapter-test error: `SDKRun` is passed where Trackio's `Run` type is expected
  in `packages/tracking-trackio/tests/test_adapter.py:264`. Type checking the
  files changed by this plan reports zero errors.
- Observation: the machine CA bundle was parsed into `MachineConfig` but the
  remote-builder HTTPX client did not receive it, so a valid site credential
  still failed against private HTTPS. The client now creates its TLS context
  from the configured absolute CA bundle; the real framework client receives
  HTTP 200 from authenticated capabilities.
- Observation: adding the BuildKit and runtime-image workspace dependencies to
  the composed job-builder service changed `uv.lock`, so required Quality
  correctly rejected the stale generated catalog dependency digest. Running
  `posttrain-release lock-dependencies` regenerated the digest from the exact
  release lock before the v0.3.20 candidate gates.
- Observation: the v0.3.20 version bump did not change any OCI input or runtime
  lock. The candidate reused the registry-verified image digests and generated
  only the required `published.toml` framework-version update from 0.3.19 to
  0.3.20.

## Decision Log

- Decision: authenticate one infrastructure principal and remove
  credential-to-project allowlists.
  Rationale: `project_id` is an immutable namespace and audit dimension, not a
  credential owner. Infrastructure access should not change when a project is
  created.
  Date/Author: 2026-08-19 / user and Codex.
- Decision: retain project namespace validation and server-derived repository
  policy.
  Rationale: site-wide authorization must not allow a client to select an
  arbitrary registry destination or collide with another namespace's durable
  records.
  Date/Author: 2026-08-19 / Codex.
- Decision: derive OCI repositories as
  `<site repository prefix>/<project_id>/posttrain-job`, independently of the
  authenticated infrastructure principal.
  Rationale: project identity is stable product identity; the principal is an
  operational authentication and audit identity that can rotate without
  renaming published images.
  Date/Author: 2026-08-19 / Codex.
- Decision: reject the old `token_grants` schema instead of silently treating
  an old project-scoped token as site-wide.
  Rationale: implicit broadening is a security migration hazard. Code and
  protected configuration must move together with a retained rollback copy.
  Date/Author: 2026-08-19 / Codex.
- Decision: do not deploy, commit, or overwrite the current credential during
  this implementation turn without explicit authorization.
  Rationale: the working trees contain unrelated user changes, and live
  deployment must be tied to immutable source revisions.
  Date/Author: 2026-08-19 / Codex.
- Decision: after the user explicitly authorized the immutable rollout, use
  clean worktrees from current `origin/main` rather than switching either dirty
  primary checkout.
  Rationale: this kept unrelated release work intact while ensuring both
  deployed repositories were clean, committed, and pushed.
  Date/Author: 2026-08-19 / user and Codex.

## Outcomes & Retrospective

The site-wide authorization architecture is implemented, committed, pushed,
deployed, and qualified. The live builder selects immutable Posttrain revision
`c65e346df1b909790a76b0781641266e14ceedcc`; its rootless BuildKit and API units
are active. Public liveness returns 200, readiness returns 204, authenticated
capabilities returns 200 through the framework client, and a structurally valid
request naming an arbitrary repository returns 403 before blob admission.

The workstation now references protected machine credential
`job-builder-infrastructure`; the invalid five-byte `job-builder-ambient` file
was retained under a dated retired name for recovery. Focused acceptance passes
58 tests, affected-file Pyright reports zero errors, and Ruff, import contracts,
Ansible syntax, shell syntax, and whitespace checks pass. The previously
recorded unrelated repository-wide catalog-hash and Trackio test typing issues
remain outside this change.

## Context and Orientation

An actual-job image is the immutable OCI image containing one resolved
Posttrain job. The developer CLI packs its declared source and configuration,
then either invokes local BuildKit or uploads the declared content-addressed
blobs to the optional remote builder. `apps/job-builder` is the remote API and
durable queue. `packages/execution-job-builder` is its HTTP client.
`packages/execution-buildkit` is the only component that invokes BuildKit.

Today `apps/job-builder/src/posttrain_job_builder/http.py` defines
`PrincipalGrant`, which includes `project_ids`. `service.py` loads protected
JSON `token_grants`, and every request asks `ProjectRepositoryPolicy` whether
the authenticated grant contains the request project. The store deliberately
uses `principal/project/publication-key` paths; those paths remain because they
are useful isolation and audit boundaries even after project IDs stop being an
authorization list.

The live site uses `ai-control.lan` as TLS ingress and forwards to the dedicated
`ai-builder` VM at `192.168.110.117`. The sibling infrastructure repository has
only the host reservation committed. Its Caddy route is modified locally and
its `ansible/roles/job_builder` directory is untracked and incomplete. Raw
server configuration exists only as a protected mode-`0600` file on the host.

## Plan of Work

First change the product contract because the existing frozen amendment says
only that the remote builder is machine-configured, while the implementation
plan recorded project-scoped bearer grants. ADR 0016 and the canonical API now
state that the credential authenticates site infrastructure and `project_id`
is only a validated namespace.

Next replace `PrincipalGrant` with `InfrastructureGrant` in
`apps/job-builder/src/posttrain_job_builder/http.py`. The new value contains
only `principal`. `BearerTokenAuthorizer` continues to compare SHA-256 token
digests in constant time. `ProjectRepositoryPolicy.repository_for` validates
the request `project_id` and derives the repository from the site prefix and
project namespace without consulting an allowlist. The authenticated principal
continues to scope durable service state and audit evidence but does not change
OCI image identity.

Change `apps/job-builder/src/posttrain_job_builder/service.py` so protected
JSON requires `infrastructure_grants` and rejects `token_grants`. Update exports
and tests. Add an HTTP conformance test that uses one token to plan two project
namespaces and proves each arbitrary or cross-project repository fails before
blob upload. Existing store and worker methods keep principal and project
arguments because those are durable isolation keys, not authorization policy.

Then complete `/home/hammad/projects/ai-infra/ansible/roles/job_builder` with
defaults, tasks, handlers, templates for both systemd units and protected
server JSON, and a playbook selecting only the builder host. Token bytes or
digests are supplied from ignored protected state; templates never contain a
raw token. Replace the hard-coded Caddy upstream with the inventory builder
address. Add a read-only qualification playbook or script proving both health
endpoints, authenticated capabilities, an unauthorized repository rejection,
and service provenance without launching a full job.

Posttrain must be committed and pushed before `ai-infra` records or deploys its
immutable source revision. `ai-infra` must then be committed and pushed before
deployment. Deployment must retain the old protected config and systemd units
for rollback, apply code and config together, restart once, and qualify the
public TLS route. No commit or deployment is performed unless explicitly
authorized.

## Concrete Steps

From `/home/hammad/projects/rl`, run focused tests first:

    uv run pytest apps/job-builder/tests packages/execution-job-builder/tests
    uv run ruff check apps/job-builder packages/execution-job-builder
    uv run pyright
    uv run lint-imports
    git diff --check

From `/home/hammad/projects/ai-infra`, validate syntax and idempotence without
applying:

    uv run ansible-playbook --syntax-check -i ansible/inventory/generated.yml ansible/playbooks/job-builder.yml
    uv run ansible-playbook --check --diff -i ansible/inventory/generated.yml ansible/playbooks/job-builder.yml
    git diff --check

The check-mode command may report systemd restart changes but must not expose a
token, token digest, rendered protected JSON, or private key.

After explicit commit and deployment authorization, the release order is:

    cd /home/hammad/projects/rl
    # commit and push the reviewed Posttrain change
    cd /home/hammad/projects/ai-infra
    # pin that immutable Posttrain revision, commit and push infrastructure
    # produce a saved deployment plan, apply it, and run qualification

## Validation and Acceptance

Unit acceptance requires one token to plan publications for two distinct
project IDs without changing server configuration. Both derived repositories
must be project-isolated. A plan naming any other repository must receive HTTP
403 before the store creates an upload record. A missing or invalid token must
remain HTTP 401. A protected server configuration using `token_grants` or a
grant containing `project_ids` must fail at startup with a clear contract
error.

Infrastructure acceptance requires a clean committed `ai-infra` tree, an
immutable Posttrain source revision, active rootless BuildKit and API units,
HTTP 204 readiness through `https://ai-control.lan/job-builder`, authenticated
capabilities, and a retained qualification receipt naming both Git commits and
the active service configuration digest. Project creation must require no
server grant edit.

## Idempotence and Recovery

Unit and syntax validation are read-only and repeatable. The content store and
published OCI images are not migrated because principal/project/publication
paths remain unchanged. The protected configuration filename should be
versioned or backed up before replacement. If the upgraded service fails
readiness, restore the previous service source and protected config together,
reload systemd, restart the two exact units, and verify the old readiness
endpoint. Never translate `token_grants` to site-wide authorization implicitly.

## Artifacts and Notes

Current live route:

    https://ai-control.lan/job-builder/* -> 192.168.110.117:8080

Current incomplete infrastructure source:

    M ansible/roles/control/templates/Caddyfile.j2
    ?? ansible/roles/job_builder/

These files predate this plan and must be completed without discarding their
useful rootless BuildKit settings.

## Interfaces and Dependencies

`InfrastructureGrant` has one field, `principal: str`, validated as a safe path
segment. `BearerTokenAuthorizer.authenticate` returns that grant.
`ProjectRepositoryPolicy.repository_for(project_id)` returns the exact
repository and validates `project_id`, but performs no allowlist lookup.
`ServiceConfig.infrastructure_grants` maps protected SHA-256 token digests to
those grants. The public HTTP request schema and client headers remain
unchanged, so no client package protocol migration is required.

Revision note, 2026-08-19: created this plan after live inspection showed that
the prototype project-scoped authorization and manually deployed
infrastructure did not match the intended site-wide trust boundary.
