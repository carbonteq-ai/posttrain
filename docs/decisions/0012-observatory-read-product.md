# ADR 0012 — Observatory is the single post-training read product

## Status

Accepted.

## Context

The workspace introduced `packages/reports` as a Trackio-specific query and
materialization library, then planned `apps/observatory` as a separate API,
frontend, and MCP consumer of job-aware views. Those boundaries overlap: both
surfaces answer the same run, work-package, stage, lineage, comparison, and
health questions for the same ML expert. Keeping both would create competing
Python entrypoints, duplicated view ownership, and ambiguity about where report
math, telemetry definitions, alert rules, and compatibility queries belong.

Tracking providers still require a smaller reusable boundary. Hosts need to
write observations, and read products need provider-neutral access to raw run,
metric, trace, event, and artifact evidence. That transport boundary should not
own product-specific navigation or presentation.

## Decision

- Make `apps/observatory` the dedicated, read-only post-training observability
  product.
- Retire the `posttrain-reports` distribution and remove `packages/reports`
  after its useful query, work-package, and materialization behavior has been
  migrated into Observatory.
- Keep `packages/tracking` as infrastructure: run lifecycle contracts,
  provider-neutral raw evidence readers, normalized evidence models, and shared
  backend conformance contracts.
- Move job telemetry definitions, `RunViewService`, work-package/stage/lineage
  analysis, serving Pareto views, alert evaluation, comparisons, and report
  exports under Observatory ownership.
- Expose one Observatory application service to every read surface. Its Python
  API, HTTP routes, MCP tools, frontend, CLI, and materialized report exports
  must call the same service and use the same strict models.
- Keep Trackio and W&B SDK/API access inside their adapter packages. Observatory
  selects a `RunDataSource` in its composition root and never reads provider SQL
  or APIs directly.
- Do not maintain `posttrain.reports` as a peer compatibility product. This is
  an internal, unreleased workspace, so callers and tests migrate with the
  rename; any temporary import shim must have an explicit removal test and may
  not contain logic.
- Keep the source and server implementation in `apps/observatory`, but
  distribute the running product primarily as one versioned OCI container
  image. The image contains the Python service, compiled frontend, HTTP API,
  Streamable HTTP MCP endpoint, report exporters, and both tracking readers.
- Publish immutable semantic-version tags and image digests. Production
  deployment pins a version or digest and never depends on a mutable `latest`
  tag.
- Provide Docker Compose for local/self-hosted evaluation and an OCI-distributed
  Helm chart when Kubernetes deployment is introduced. Both deploy the same
  image and runtime contract.
- Treat the full server wheel as an internal build/test artifact, not the
  supported installation path for other teams. Publish a separate thin
  `posttrain-observatory-client` wheel when remote Python/CLI integration is
  implemented; it contains typed HTTP clients and an optional stdio MCP bridge,
  but no provider SDK or analysis logic.
- Prefer one centrally hosted Observatory. Also support self-hosting the same
  image when data or infrastructure boundaries require it. Provider credentials
  remain server-side in deployment secrets and never enter frontend, client, or
  MCP configuration.

## Consequences

- ML experts have one product and vocabulary for run monitoring, analysis,
  comparison, lineage, alerts, and exports.
- Python, HTTP, MCP, frontend, and report files cannot drift into separate
  interpretations because they share one application service.
- `packages/tracking` becomes easier to reuse in another host or product because
  it no longer owns Observatory-specific presentation policy.
- Observatory is larger than a thin frontend application. It requires explicit
  internal modules for domain views, application queries, provider composition,
  transports, and frontend assets.
- The migration must update imports, scripts, tests, dependency declarations,
  documentation, and import-linter contracts together. Deleting
  `packages/reports` before behavior is ported would lose existing
  work-package-query coverage.
- Observatory remains read-only. Accept/revise/reject decisions and run
  execution remain outside the product even when Observatory displays their
  evidence.
- A single image keeps the service, schemas, and frontend on one compatible
  release. The thin client can evolve independently only within the published
  HTTP and MCP compatibility contract.
- Including both Trackio and W&B readers produces a larger image than backend-
  specific variants, but avoids a matrix of subtly different products. Split
  images are deferred until measured operational need justifies them.

## Alternatives Considered

### Keep reports as a library and Observatory as a thin UI

Rejected because the library and application would still compete for ownership
of view models, report calculations, and Python-facing analysis APIs.

### Keep job-aware views in tracking and make all other surfaces thin

Rejected because job telemetry, health interpretation, comparison policy, and
report exports are product semantics rather than provider-neutral tracking
transport. Raw evidence models remain in tracking.

### Extend Trackio's bundled product instead

Rejected because Observatory must render the same post-training semantics for
Trackio and W&B and must expose a backend-neutral Python and MCP surface.

### Create separate reporting and monitoring services

Rejected for the current scope. Both operate on the same known job kinds and
evidence. A future deployment split can preserve one logical Observatory
service contract if scaling requirements justify it.

## Implementation Notes

- Establish `apps/observatory/src/posttrain_observatory` with domain,
  application, provider-composition, API, MCP, CLI, and report-export modules.
- Port `packages/reports/src/posttrain/reports/query.py` and
  `work_packages.py` behavior behind `RunDataSource`; do not port their Trackio
  SQL.
- Move the current job telemetry registry and `RunViewService` from
  `packages/tracking` into Observatory after adapter conformance remains green.
- Replace `trackio-query` with an Observatory CLI query/export command.
- Move package-specific tests from `packages/reports/tests` to
  `apps/observatory/tests`, then add API, MCP, frontend, and cross-backend
  product tests there.
- Remove `packages/reports`, its workspace test path, Trackio dependency, and
  `posttrain.reports` import-linter entry when migration tests pass.
- Amend the living plan at
  `docs/plan/dual-backend-tracking-observatory.md` as implementation proceeds.
- Build the frontend and Python runtime in separate Docker stages and copy only
  their runtime outputs into the final image.
- Expose `/health`, versioned `/api`, and `/mcp` endpoints from the same service.
  Streamable HTTP is the supported remote MCP transport; stdio remains a local
  client bridge.
- Publish the OpenAPI document, MCP tool schema, image digest, and source
  revision with each release. Generate the future client wheel from the
  released OpenAPI contract.

## Revision History

- 2026-07-22: Accepted Observatory as the single read/analysis product and the
  retirement of the parallel reports package.
- 2026-07-22: Accepted the OCI-first distribution model, central and self-hosted
  deployment modes, server-side secrets, one image with both readers, and a
  separate future client wheel for Python/CLI integration.
