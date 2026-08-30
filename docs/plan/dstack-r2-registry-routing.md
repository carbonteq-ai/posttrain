# Productionize dstack cloud execution with R2, RunPod, durable recovery, and Trackio

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. Maintain this document in accordance with `docs/templates/PLAN.md`.

For autonomous implementation, use `docs/plan/dstack-r2-cloud-execution-agent-runbook.md`. This plan remains the architecture and acceptance authority; the runbook supplies the distraction-resistant milestone loop and goal prompt.

## Purpose / Big Picture

After this change, a Posttrain job may be submitted once to dstack and may run either on an existing LAN GPU worker or on cloud capacity provisioned by dstack. The job keeps one immutable, digest-pinned OCI image identity. A LAN worker downloads that image from the existing LAN registry, while a cloud worker downloads the equivalent image from a public registry whose blobs are stored in Cloudflare R2. Multi-gigabyte cloud pulls do not traverse the CarbonTeq LAN.

The location choice is deliberately transparent to Posttrain and dstack. Both receive the same canonical image reference, such as `registry.<public-domain>/<project>/posttrain-job@sha256:<digest>`. Managed LAN provisioning pins that hostname to `ai-control` and installs the deployment-local certificate authority; fresh external cloud machines receive neither override, so public DNS and the ordinary public trust store take them to Cloudflare and the R2-backed registry. Both registry instances retain byte-identical manifests and blobs, so the digest names the same image in both places. The LAN registry emits manifest-push events to an infrastructure-owned durable mirror worker. Job Builder only builds and pushes the canonical image; it does not call, configure, or understand the mirror.

The observable acceptance scenario is one job submitted with LAN and cloud capacity eligible. When LAN capacity is available, worker logs and registry access evidence show a pull from the LAN endpoint and no R2 blob read. When LAN capacity is unavailable, dstack provisions a cloud machine; the machine requests the same canonical digest, the public registry returns the manifest, large blob requests redirect to signed R2 URLs, the job starts successfully, and dstack deprovisions the machine according to its configured idle policy.

This is a production design, not a disposable prototype. Deployment is additive: the existing `registry.lan` filesystem registry remains authoritative until the R2-backed replica and the real cloud qualification both pass. No existing registry data is moved or deleted during rollout.

The post-canary scope also closes the gaps that the successful stateless A100 test deliberately did not prove: automatic cross-driver CUDA selection, mirror-aware cloud admission, public Trackio writes and direct checkpoint transport to CarbonTeq's self-hosted S3 service, provider-authoritative interruption reporting, durable hooks, run-scoped spot storage, recovery, inventory, and terminal cleanup. These functional paths are built and qualified first. Security hardening is performed last, but remains a non-bypassable prerequisite for any production cloud enablement.

## Progress

- [x] (2026-08-29 11:47Z) Read the repository plan contract and the canonical Posttrain ownership, execution, and image-publication documents.
- [x] (2026-08-29 11:47Z) Inspected the current Posttrain, dstack, and ai-infra source and recorded the exact repository revisions and dirty-worktree constraints.
- [x] (2026-08-29 11:47Z) Chose canonical-hostname plus split-horizon DNS routing over a dstack-specific registry-rewrite feature; retained only a narrow generic dstack credential-injection fix for an explicitly named private registry.
- [x] (2026-08-29 11:47Z) Verified from current upstream documentation that R2 exposes the S3 operations needed by the Distribution S3 driver and that Distribution can redirect blob reads to presigned object URLs.
- [x] (2026-08-29 12:06Z) Proved the split-horizon premise with two disposable Distribution 3 registries: LAN and cloud clients resolved `registry.posttrain.test` to different backend IPs and catalogs while both returned manifest digest `sha256:e94e7d9ead2a5efeb21a9913051e39730fe37c43e26108c088a548346d7cd3b4`.
- [x] (2026-08-29 12:06Z) Proved the same behavior through two isolated Docker 29.1.3 daemons. Each daemon pulled the same canonical digest through its different DNS answer and retained the identical canonical `RepoDigest`.
- [x] (2026-08-30) Completed Milestone 0 edit isolation without creating a worktree. Recorded starting revisions: framework `841e78aba299972da109b40d1c740404cc4dc42a` on `codex/main-realign-20260824`; dstack `89c050f97c3b700a92a041ca154884ce00d7310e` on `codex/registry-default-auth`; Trackio `ec7d0635f5cbf215a7566e4c3a9d54952504c4c7` on `codex/trace-summary-projection`; ai-infra `b1938da26fa1a8cdb77f851e4e800062d5c3f1a9` on `codex/r2-cloud-registry`. Milestone A overlaps only the framework's existing runtime/image optimization work in `apps/runtime` and `packages/runtime-images`; those edits are preserved and extended in place. dstack's diagnostic-redaction edits, ai-infra's registry-mirror/Caddy edits, Trackio's clean checkout, and every unrelated framework cache/release/purge edit remain outside Milestone A.
- [x] (2026-08-29 13:10Z) Completed Milestone 1 against the supplied private R2 bucket. The pinned Distribution 3 image passed multipart push, stable repeated digest, separate signed `HEAD` and `GET` redirects, two-range digest reconstruction, clean Docker pull by digest, manifest deletion, and stopped-service garbage collection. Cleanup removed the unique R2 prefix and all disposable Docker resources; the safe receipt is `/home/hammad/projects/ai-infra/.state/artifacts/cloud-registry/compatibility.json`.
- [x] (2026-08-29 14:29Z) Reassigned replica policy, durable work state, exact-digest copying, release-root retention, and verification receipts to an ai-infra registry mirror. The framework continues to own image content and immutable identity but does not manage registry topology or R2.
- [x] (2026-08-29 15:03Z) Deployed the pinned R2-backed Distribution registry, unprivileged SQLite registry mirror, authenticated Cloudflare Tunnel connector, and public read-only `registry.carbonteq.com` route without recreating Trackio, dstack, Doris, or the LAN registry.
- [x] (2026-08-29 15:03Z) Passed a live exact-digest publication canary for `carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`; the durable controller receipt reached `verified` after one attempt and the public manifest returned the same digest.
- [x] (2026-08-29 15:03Z) Proved the external security boundary: anonymous discovery returned 401, authenticated discovery and manifest reads returned 200, and an authenticated manifest deletion returned 405. The controller rejects caller-supplied source or destination fields.
- [x] (2026-08-29 15:03Z) Added and tested an ai-infra runtime-image release seeder. Its plan resolves the committed framework manifest to seven immutable retention roots; apply waits for controller verification and can write a mode-0600 receipt.
- [x] (2026-08-29 15:03Z) Added UniFi Host A records for the canonical hostname and the two private management hostnames, all targeting `ai-control`. The canonical A answer is live on the workstation, control VM, builder, and release runner. Managed-worker `/etc/hosts` pinning and Docker CA trust are implemented but await an idle-window Docker restart.
- [x] (2026-08-29 20:00Z) Removed the job-builder readiness client after architecture review. Job Builder now has no mirror URL, token, target-registry knowledge, or replica-status behavior; its tracked source is back to the pre-mirror diff while all 12 tests, focused Pyright, and repository import contracts pass. ai-infra configures its fixed repository prefix to the canonical hostname and owns event-driven mirroring.
- [x] (2026-08-29 20:00Z) Implemented the provisioning boundary explicitly: managed LAN workers pin the canonical hostname to `192.168.110.53`, install the private CA for that host, and qualify a canonical digest pull; external cloud machines retain public DNS and public trust. Separate live Caddy listeners now send LAN canonical traffic to the filesystem registry and Tunnel traffic to the authenticated read-only R2 registry.
- [x] (2026-08-29 20:00Z) Added and applied a protected LAN Distribution configuration that sends manifest-push notifications directly to the durable mirror queue. The Cloudflare managed Tunnel origin moved from `http://caddy:80` to the dedicated `http://caddy:8081` listener before the LAN route switched, so the public registry stayed available throughout.
- [x] (2026-08-29 20:00Z) Proved the corrected live path with a temporary `qualification/event-mirror-20260829-2009` alias: pushing only to the LAN registry produced a SQLite `verified` receipt after one attempt and the R2 registry returned the same `sha256:67624f...15f15` digest. Removed the two qualification manifests and their exact temporary receipt afterward. A repeated focused apply reported `changed=0`.
- [x] (2026-08-29 15:16Z) Implemented the generic dstack exact-host credential rule on a normal `codex/registry-default-auth` branch based on published CarbonTeq commit `275b81bc725967c8925b5b12d96500dc60a45370`. Exact match, port mismatch, malicious prefix/suffix mismatch, explicit-auth precedence, incomplete credentials, and legacy unqualified behavior pass in 38 focused tests plus Ruff. The branch remains uncommitted and unselected pending review, push, and one immutable release.
- [x] (2026-08-30) Selected and configured RunPod as the first external dstack backend. The protected API key authenticates, `community_cloud: false` keeps discovery on Secure Cloud, and scoped input-bound Ansible plan/apply evidence is retained. The existing SSH fleet and its long-running LAN task were preserved.
- [x] (2026-08-30) Designed the provider-lifecycle extension in ADR 0017 and the infrastructure lifecycle architecture: dstack-owned run-scoped network volumes, attempt-aware spot recovery, durable typed hooks, bounded cleanup, and a provenance-bearing inventory read model. This is a design milestone; no dstack migration or provider resource has been created yet.
- [x] (2026-08-30) Verified the newly supplied `runpod_api_key` exists in ai-infra's ignored mode-0600 secret file and authenticates through the maintained dstack fork without displaying it. Selected an opt-in backend `run_storage` policy so ai-infra can require one per-logical-run volume for RunPod spot attempts while ordinary on-demand and unconfigured backend behavior remains unchanged.
- [x] (2026-08-30) Refined checkpoint recovery into two tiers: Trackio publishes verified durable artifacts for all workloads, while the RunPod backend requires a run-scoped network volume only for interruptible/spot attempts. Same-volume retry is the fast path; restoring the latest Trackio-verified checkpoint into a new volume is the explicit cross-data-center fallback. Verified the live Trackio service still uses its `local` artifact backend and has no worker-reachable S3 presign endpoint, so direct cloud multipart publication remains an implementation and qualification gate.
- [x] (2026-08-30) Published dstack fork commit `d2586c3871525e461bcbc442deaa511af2a87758`. It applies exact-host registry credentials and replaces static RunPod GPU spot discovery with bounded live capacity queries while retaining the offline catalog for on-demand, CPU, and cluster planning. All seven RunPod backend tests and the 39 combined RunPod/registry-default tests pass.
- [x] (2026-08-30) Qualified a public-image RunPod spot lifecycle on an isolated candidate control plane without touching the production database or LAN task. A zero-node RunPod-only fleet selected an RTX PRO 6000 Blackwell Server Edition in `EUR-IS-1` at $2.09/hour; CUDA reported 97,887 MiB visible VRAM and driver 595.91.07. dstack reported `submitted -> provisioning -> running -> terminating -> done`, fleet deletion completed, and the RunPod API reported zero active Pods afterward.
- [x] (2026-08-30) Published the cache-preserving veRL replacement to the canonical LAN repository at `sha256:fea65ed6037f52f44f5f901fb1b95c5888fe6d47c8799e0dbcdc6f1318add28d` after FlashInfer JIT, CUDA/NCCL, Qwen3.5 vLLM generation, exact-parent, cold-pull, warm-pull, and actual-job qualification gates passed on the current 8 GiB Pop!_OS GPU. The updated release manifest reduces the seven-root deduplicated compressed union to 9,217,281,255 bytes, so release-root R2 seeding is within the 10 GB-month free allowance.
- [x] (2026-08-30) Qualified the real R2 registry path with the veRL runtime on RunPod A100 spot capacity. The universal base remained `sha256:55987a...`; the veRL-only image `sha256:9bf4ff...` adds NVIDIA's pinned 104,195,781-byte CUDA 13 compatibility payload after all 24 previously qualified layers, leaves it inactive by default, passes local CUDA on the Pop!_OS RTX 3070 Ti, and passes PyTorch CUDA 13 execution on an A100 80 GB PCIe spot Pod in `CA-MTL-3` at $1.39/hour when the RunPod task activates the payload. The mirror receipt reached `verified`, the task completed with exit code 0 for $0.0711, and the RunPod API confirmed the Pod was deleted.
- [x] (2026-08-30) Removed raw environment values from the dstack runner's `Starting exec` diagnostic trace. The unpublished fork successor logs sorted variable names only, and the complete Go executor package passes under Go 1.25. Until that successor is released, diagnostic mode remains forbidden for credential-bearing cloud jobs.
- [ ] Promote dstack commit `d2586c3871525e461bcbc442deaa511af2a87758` to the production control plane. Candidate server and component digest checks passed, but the release gate correctly rolled back because `pop-os.lan` was already unreachable and the RTX PRO 6000 LAN worker is busy with run `6dffd449-12e5-4bda-b449-defdcb30a0a4`. Do not cancel that run or weaken the two-worker qualification gate.
- [x] (2026-08-30) Completed Milestone 2: the external R2-backed registry, private push ingress, public read-only pull ingress, canonical hostname, and split public/LAN routing are deployed and qualified. Managed-worker restart/trust proof remains a rollout gate in Milestone 5 rather than registry construction work.
- [x] (2026-08-30) Completed Milestone 3: LAN registry notifications feed the durable exact-digest mirror, canonical repository filtering and restart-safe receipts are deployed, release roots are seeded, and the infrastructure-owned reconcile/readiness API is available. Cloud admission still needs to consume that readiness result before it creates billable compute.
- [ ] Complete Production Milestones A–C (automatic guarded CUDA activation; cloud provisioning wait on exact verified registry receipt; public Trackio writes plus direct presigned artifacts to the existing self-hosted RustFS S3 service).
- [ ] Complete Production Milestones D–G (provider-authoritative attempts; transactional lifecycle hooks; spot-only per-run network volumes with fencing and cleanup receipts; same-volume and cross-data-center checkpoint recovery; truthful inventory).
- [ ] Complete Production Milestone H functional resilience matrix while production fallback remains disabled.
- [ ] Complete Production Milestone I security hardening last: publish diagnostic redaction, validate least-privilege credential boundaries, and pass negative/security tests while preserving the working credentials.
- [ ] Complete Production Milestone J: deploy one immutable server/runner/shim release only when the LAN promotion gate is idle, then enable bounded project-scoped fallback with rollback retained.
- [x] (2026-08-30) Corrected Milestone C after storage-boundary review: Cloudflare R2 remains exclusive to OCI registry blobs; Trackio direct multipart artifacts target the existing self-hosted RustFS S3 service through its private server endpoint and a separately qualified worker-reachable presign endpoint.
- [x] (2026-08-30) Corrected the security scope at the user's direction. Existing credentials remain in protected ai-infra state; the final security milestone validates handling and privilege without replacing them.
- [x] (2026-08-30) Added the goal-runnable agent runbook with one-milestone focus, scoped context loading, focused-test-first validation, bounded external mutations, receipt-driven progress, and explicit recovery and stop conditions.
- [x] (2026-08-30) Refined the runbook's focus rule: completed decisions are not reopened speculatively, while adjacent evidence-backed improvements to quality, maintainability, operability, tests, and developer experience may ship with the active release unit.
- [x] (2026-08-30) Implemented the code and image half of Production Milestone A. `posttrain-runtime execute` now performs a typed CUDA driver preflight before importing backend code, selects native or the image-declared compatibility payload, and re-execs at most once. The veRL image writes the declaration beside the pinned payload in one final additive layer without a global `LD_LIBRARY_PATH`. The focused release, BuildKit, runtime-image, and runtime suites pass `171 passed, 1 skipped`; scoped Ruff, Pyright, and diff checks are green. Publication, descriptor comparison, exact-digest local qualification, and the no-override RunPod receipt remain open.

## Surprises & Discoveries

- Observation: the automatic selector belongs to the actual-job entrypoint, while the veRL kind image owns only the compatibility declaration and payload.
  Evidence: `posttrain-runtime` is installed when the actual-job image is built, not in the reusable kind layer. The Milestone A cross-host gate must therefore run one immutable actual-job digest on both the local RTX 3070 Ti and RunPod; qualifying the kind image directly would bypass the new selector. The local host reports CUDA driver API version 13000, equal to the image runtime requirement, so its expected selection is native.

- Observation: a release-commit-derived `SOURCE_DATE_EPOCH` is reproducible only within one commit and is therefore incompatible with cache-preserving additive image releases.
  Evidence: the first Milestone A publication attempt used epoch `1788055395` from the new framework commit, while the qualified 24-layer veRL lineage uses `1787491908`. BuildKit began rewriting every rebuilt layer to the new epoch, so the attempt was interrupted before manifest push. The epoch is now an explicit immutable cache-lineage property, while commit-varying label arguments are declared only after filesystem-producing instructions. Focused validation passes `111 passed, 1 skipped`; the canonical published digest remained unchanged.

- Observation: stable timestamps alone do not preserve an existing descriptor when BuildKit misses the prior cache and a nominally equivalent system-package step has mutable side effects.
  Evidence: rejected candidate `sha256:ad9bd883...` used the correct old epoch but changed layer 10 by one compressed byte because the rebuilt apt layer's `alternatives.log` differed; copying the modified profile also changed layer 22. The canonical manifest was restored to `sha256:9bf4ff...`. Cache-lineage inputs now live in an uncopied definition file, while separate late `RELEASE_*` arguments keep OCI labels truthful without perturbing the legacy filesystem cache key. This design preserves immutable OCI references and uv locks; the lineage values are cache seeds only, never runtime or release identity.

- Observation: ai-infra and the Trackio fork already contain the self-hosted S3 path required for cloud checkpoints; a new object-storage provider is unnecessary.
  Evidence: `../ai-infra/ansible/roles/storage/files/compose.yml` runs pinned RustFS with retained `/srv/data/rustfs` storage; `../ai-infra/ansible/roles/control/templates/control.env.j2` already renders Trackio's private S3 endpoint, distinct presign endpoint, bucket, prefix, and RustFS-backed credentials; and `../trackio/trackio/artifact_storage.py` plus `direct_uploads.py` implement verified presigned multipart transport. Production still defaults to `local`, and no worker-reachable RustFS presign route is configured, so migration, ingress, capacity, backup, and live RunPod qualification remain real gates.

- Observation: dstack already avoids a registry pull for a cached non-`latest` image.
  Evidence: `../dstack/runner/internal/shim/docker.go` function `pullImage` lists the exact image reference and returns before `ImagePull` when it exists. Posttrain always submits digest-pinned references, so warm-worker reuse requires no new runner behavior.

- Observation: dstack currently carries one `image_name` and one optional `registry_auth` in `JobSpec`, but this does not require a dstack fork change for location selection.
  Evidence: `../dstack/src/dstack/_internal/core/models/runs.py` defines the single image and credential pair; `packages/execution-dstack/src/posttrain_execution_dstack/adapter.py` submits one `request.image.value`. A single canonical pull hostname with topology-dependent DNS preserves that contract.

- Observation: dstack's server-owned default registry credentials are not currently applied when an image already contains an explicit registry hostname.
  Evidence: `../dstack/src/dstack/_internal/server/services/docker.py` function `apply_server_docker_defaults` returns immediately when `parse_image_name(image_name).registry` is non-null. Posttrain correctly submits a fully qualified digest, so a private canonical hostname needs a narrow generic fix that applies server-owned credentials when the explicit host equals the configured default registry. This is credential injection, not image routing.

- Observation: the existing LAN-only registry cannot itself become the canonical cloud pull hostname because it is named `registry.lan` and uses a private Caddy CA.
  Evidence: `../ai-infra/ansible/roles/control/templates/Caddyfile.j2` serves `https://registry.lan` with `tls internal`, and BuildKit plus workers install that private trust path. Cloud provider control planes and fresh spot machines must instead use a publicly valid hostname and TLS chain.

- Observation: split-horizon routing works through Docker Engine and does not change OCI image identity.
  Evidence: two disposable Distribution 3 registries exposed different marker catalogs through the same `registry.posttrain.test` hostname. LAN resolved it to `172.22.0.2`, cloud resolved it to `172.23.0.2`, and both returned `Docker-Content-Digest: sha256:e94e7d9ead2a5efeb21a9913051e39730fe37c43e26108c088a548346d7cd3b4`. Two independent Docker 29.1.3 daemons then pulled `registry.posttrain.test:5000/shared/canary` by that digest successfully. The disposable containers, networks, and image aliases were removed after the test.

- Observation: the LAN and public endpoints do not need to serve the same TLS certificate or trust chain.
  Evidence: TLS validation happens after DNS selects an endpoint and validates the requested hostname, not the registry's storage identity. Managed LAN workers already receive the Caddy private root and can trust an internal certificate for the canonical hostname; external cloud workers instead reach Cloudflare's publicly trusted edge certificate. This avoids adding a Cloudflare DNS provider module to the currently standard Caddy image.

- Observation: the current workstation resolves `.lan` through `192.168.200.1`, while its default DNS server is `192.168.30.1`; BuildKit's managed configuration separately requires `192.168.30.1`.
  Evidence: `resolvectl status` and `../ai-infra/scripts/buildkit_builder.py` show those paths. The production DNS gate must query the actual dstack server, builder, and every managed LAN worker, not infer their answer from this workstation.

- Observation: every actual-job image is unique even when its parent layers are shared, so mirroring only release base and job-kind images is insufficient.
  Evidence: the canonical API contract derives a new package/publication identity from exact code, configuration, dependencies, data, and kind-image identity. The remote job builder must make the actual-job manifest available in both registries before automatic cloud fallback is safe.

- Observation: the current published runtime graph occupies 11.805 GB of unique compressed layer blobs, while replacing the published veRL image with the verified candidate projects the union to approximately 9.04 GB before actual-job suffixes.
  Evidence: live registry manifest inspection on 2026-08-29 found 48 unique blobs and 11,805,216,367 bytes. R2 Standard currently includes 10 GB-month, one million Class A operations, ten million Class B operations, and free Internet egress; actual billing must be checked again at rollout time.

- Observation: the final measured release-root graph is 9,217,281,255 compressed bytes, not the earlier 9.04 GB projection.
  Evidence: live manifests for the updated `published.toml` contain 50 unique layer digests. The individual manifest sums are 2,800,687,414 bytes for base; 4,180,475,091 eval; 4,240,774,463 TRL; 4,651,350,481 veRL; 4,092,584,762 serve; 3,078,312,077 supervised; and 3,017,292,809 transform. Shared ancestry must be counted once for R2 capacity.

- Observation: RunPod's A100 hardware was not the cause of the first veRL canary failure; the selected Secure Cloud host exposed an R570-era CUDA 12.8 driver to a CUDA 13 image.
  Evidence: the unchanged image initialized successfully on the local newer driver but reported CUDA error 803 on the first RunPod host. Adding the pinned NVIDIA CUDA 13 forward-compatibility libraries only to the veRL child and activating them for the RunPod task made PyTorch 2.11.0+cu130 execute on the A100. Globally activating those libraries is invalid because it produced error 803 on the newer local driver, so the image runtime must compare the native driver capability before imports and activate compatibility only when required.

- Observation: the cloud compatibility fix preserves every existing cacheable veRL blob.
  Evidence: `sha256:9bf4ff...` has the exact ten-layer base prefix and exact 24-layer `sha256:fea65ed...` prefix, followed by one 104,195,781-byte compressed layer `sha256:529afeaa...`. The image totals 4,755,546,262 compressed layer bytes; publication and the local pull reused all prior layers without recompression.

- Observation: recursive OCI accounting adds only 139,875 bytes to the seven-root layer union, but notification scope can dominate storage.
  Evidence: configs, indexes, attestations, and manifests bring the canonical release graph to 9,217,421,130 unique bytes. Live S3 inventory nevertheless measured 16,944,150,733 bytes because automatic notifications had admitted temporary `carbonteq-qualification/*` candidates. Pending qualification copies were removed without touching LAN artifacts; automatic mirroring is now restricted to canonical `carbonteq/` repositories, and already-uploaded candidate blobs require offline target-registry garbage collection.

- Observation: R2 presigned URLs are method-specific.
  Evidence: reusing a redirect signed for `HEAD` as a ranged `GET` returned HTTP 403 in the first disposable test. The corrected harness obtains and validates separate redirects for each method, sanitizes request failures so signed URLs cannot appear in retained output, and cleaned the failed run's unique object prefix immediately.

- Observation: the exact S3 endpoint supplied by Cloudflare should be accepted directly rather than reconstructed from an account ID.
  Evidence: the supplied endpoint and bucket-scoped credential passed direct bucket, object, multipart, range, and presign checks plus the complete Distribution 3 compatibility gate. This also accommodates jurisdiction-specific endpoint forms.

- Observation: the overlapping ai-infra work is no longer an implementation blocker.
  Evidence: remote job-builder infrastructure is merged through PR #6, the Observatory/Doris/job-builder reconciliation is merged through PR #11, and R2 work is isolated on `codex/r2-cloud-registry` from current main. Framework and dstack changes must still preserve their current unrelated work.

- Observation: Distribution notifications are suitable as the normal ownership boundary but not as the only recovery record.
  Evidence: current Distribution documentation states that notification endpoint queues are in memory and recommends monitoring pending events. The receiver commits each accepted repository/digest into SQLite before acknowledging it, while an infrastructure-only reconcile command can repair a missed event. This keeps the normal build path event-driven without pretending the sender queue itself is durable.

- Observation: `regctl image copy` already implements the desired OCI transfer behavior.
  Evidence: regclient v0.11.5 checks the target and transfers only missing blobs, preserves the image digest, supports recursive repair, and handles multi-platform images. The controller can pin that implementation rather than reimplementing registry upload state machines.

- Observation: a single-file Docker bind mount is unsafe for atomically replaced Caddy configuration.
  Evidence: Ansible rendered the new public route on the host, but the existing Caddy container remained attached to the old file inode and its successful reload still adapted the old configuration. The deployment now mounts a dedicated configuration directory and recreates only Caddy when that mount contract changes; subsequent atomic replacements are visible through directory lookup.

- Observation: container process liveness was insufficient tunnel evidence.
  Evidence: the pinned cloudflared image runs as UID/GID 65532 and initially could not read a root-owned 0600 token. The deployment now installs the token as 65532:65532 mode 0400 and waits on cloudflared's `/ready` endpoint. Cloudflare then reported the connector Healthy and Connected.

- Observation: real R2 uploads have nontrivial per-request latency even for a modest image.
  Evidence: the 477 MB live canary needed approximately 132 seconds to copy and verify, including a 309 MB layer upload that took approximately 103 seconds. Publication readiness therefore remains an explicit wait backed by durable state rather than an assumption that a registry push returns immediately.

- Observation: using an already-present image graph under a new repository does not make R2 mirroring instantaneous.
  Evidence: the live event-driven qualification remained in `copying` long enough to observe repeatedly before reaching `verified` in one attempt. Repository-scoped blob links and R2 request latency still matter even when global blob content is shared, so real cloud admission must tolerate mirror lag.

- Observation: provisioning is the correct place to distinguish managed LAN workers from fresh cloud workers.
  Evidence: `../ai-infra/ansible/roles/worker/tasks/main.yml` already owns `/etc/hosts`, the Docker host-scoped CA directory, and the Docker restart for enrolled machines. Omitting those private-network changes on dstack-provisioned external instances naturally selects public DNS and Cloudflare TLS without changing the submitted OCI reference.

- Observation: UniFi's exact Host A override does not make the canonical name wholly authoritative.
  Evidence: the workstation, control VM, builder, and release runner receive the desired `192.168.110.53` A answer first but also receive Cloudflare's public AAAA answers. Since `ai-control` has no stable global IPv6 address, managed LAN workers must not rely on address ordering. Their provisioning now pins the canonical hostname to the LAN IPv4 address in `/etc/hosts` and installs the Caddy root under that Docker registry hostname; cloud machines retain public DNS and public trust.

- Observation: the dstack checkout's newly resolved 2026 dev environment does not make the broad submitted/running pipeline baseline entirely green.
  Evidence: the two broad pipeline files produced 106 passes, 110 PostgreSQL skips, and eight SQLite failures, all in pre-existing multinode placeholder, cluster-lock, or placement-group state expectations. Those tests do not exercise the changed `apply_server_docker_defaults` path. The directly affected Docker-default and job-service suite passes all 38 tests, including every exact-host security case. Do not attribute the eight failures to this delta or claim a fully green dstack suite; re-run the fork's pinned release environment before publication.

- Observation: RunPod cannot use a post-provisioning certificate or DNS hook for the primary job-image pull.
  Evidence: `../dstack/src/dstack/_internal/core/backends/runpod/compute.py` passes `job.job_spec.image_name` and a RunPod registry-auth object to `create_pod`; RunPod creates and pulls the container before dstack can reach its SSH service. The first image must therefore already be reachable through public DNS, public TLS, and valid OCI pull credentials. Post-provisioning trust installation remains useful only for services contacted after the container starts.

- Observation: dstack's RunPod backend is not currently configured in ai-infra.
  Evidence: `../ai-infra/ansible/roles/control/templates/dstack-config.yml.j2` contains the `main` project, encryption, and permissions but no `backends` entry, and the protected secret inventory contains no RunPod key. Current dstack requires a RunPod API key and validates it when configuring the backend; `community_cloud` defaults to false.

- Observation: the ai-infra `worker` role is specific to retained SSH-fleet hosts and cannot be applied to a RunPod pod.
  Evidence: the role installs Docker, NVIDIA container-toolkit, systemd units, a `dstack` Unix account, sudo and SSH policy, host cache directories, private CA files, `/etc/hosts` routing, and cleanup timers. RunPod instead creates the job container directly from the requested image. Its host and pre-pull lifecycle are not exposed to Ansible.

- Observation: the maintained dstack release currently gives remote containers a LAN-only runner download URL.
  Evidence: `../ai-infra/ansible/roles/dstack_release/templates/compose.yml.j2` sets `DSTACK_RUNNER_DOWNLOAD_URL` to `https://dstack.lan/dstack-components/{version}/binaries/dstack-runner-linux-{arch}`. The RunPod backend's container bootstrap command executes `curl` for that URL inside the newly created pod. A RunPod qualification therefore requires a publicly trusted, read-only route for the immutable runner binary even though the dstack API and shim artifact can remain private.

- Observation: a cloud job does not need direct access to most ai-infra services.
  Evidence: Job Builder and devpi are build-time services; PostgreSQL is private to dstack; Doris and RustFS are private dependencies of Trackio; Observatory and the mirror controller are control/inspection surfaces. A running Posttrain container needs the public OCI pull before startup and the Trackio write endpoint during execution. dstack separately needs its runner artifact to be downloadable from inside the pod.

- Observation: dstack currently infers RunPod spot loss from runner/SSH disappearance even though the RunPod pod query already retrieves provider lifecycle state.
  Evidence: the running-job pipeline maps a disconnected spot instance to `INTERRUPTED_BY_NO_CAPACITY`, while the RunPod `get_pod` query requests `desiredStatus` but the adapter uses the result only to update runtime port data. Provider observation must become the primary interruption evidence; transport loss remains a fallback.

- Observation: dstack has provider volume create/delete support and timed idle cleanup, but not a transactional run-scoped storage request.
  Evidence: a run can reference an existing named volume, `VolumeModel` records last-job use and `auto_cleanup_enabled`, and the idle-volume worker deletes eligible volumes later. There is no persisted owner relation that creates a volume after placement, retains it across attempts, and deletes it through a logical-run cleanup barrier.

- Observation: existing dstack events are suitable for operator messages but not for durable external automation.
  Evidence: events contain human-readable messages and targets without a stable kind, versioned payload, transactional delivery rows, idempotency key, or retry/dead-letter state. Runtime Ansible or shell integration therefore needs a typed outbox rather than callbacks attached to log messages.

- Observation: the current RunPod adapter presents catalog offers as available without live capacity evidence.
  Evidence: `runpod/compute.py` constructs offers from the catalog and assigns `InstanceAvailability.AVAILABLE` without a provider availability observation. The fleet surface must label these catalog candidates and show source/freshness rather than conflate them with retained machines or active pods.

- Observation: the published gpuhunt RunPod catalog is neither spot-complete nor an availability authority.
  Evidence: pinned gpuhunt 0.1.27 and current 0.1.30 both returned zero RunPod spot rows. A bounded live RunPod query returned current RTX PRO 6000 and A100 80 GB Secure Cloud stock in approximately six seconds, and dstack's candidate `offer` path returned nine RTX PRO 6000 rows at $2.09/hour. RunPod Pod creation remains the final capacity check because stock can race after discovery.

- Observation: dstack 0.20 requires an explicit fleet even when a cloud backend is configured.
  Evidence: a task constrained to RunPod failed before provider creation with `The project has no fleets`. A backend fleet with `nodes: 0..1` remained a zero-cost template, scaled to one Pod for the matching task, and was deleted after the run. Backend credentials alone are therefore not a provisioning policy.

- Observation: no current RunPod backend setting can default or require storage for runs.
  Evidence: `RunpodBackendConfig` currently contains only `regions` and `community_cloud` in addition to credentials. A run's `configuration.volumes` defaults to an empty list, and `RunpodCompute.run_job` passes a network-volume ID only when that list already resolves to one explicit volume. The apply-policy plugin can rewrite a submitted spec but runs before final backend/data-center selection and cannot atomically create, own, retry, or delete a provider resource.

## Decision Log

- Decision: Registry and DNS topology, not dstack, select the pull location.
  Rationale: The image content and digest are identical at both locations. A canonical hostname with split-horizon DNS solves reachability without adding a Posttrain-specific mirror model to dstack, changing dstack job identity, or missing container-native cloud backends that may pull an image before the shim starts. dstack remains responsible for placement and lifecycle only.
  Date/Author: 2026-08-29 / Codex

- Decision: Extend dstack only so server-owned default credentials apply to an explicit image whose registry host exactly matches the configured default registry.
  Rationale: The canonical registry must remain private, and Posttrain must continue submitting a fully qualified digest. The current early return prevents dstack's existing protected default credentials from reaching provider-native pulls and the shim. Exact host matching is generic, does not expose R2, DNS, LAN, or Posttrain concepts to dstack, and preserves explicit per-run credentials as the higher-precedence value.
  Date/Author: 2026-08-29 / Codex

- Decision: Qualify RunPod before publishing or deploying the dstack registry-auth fork.
  Rationale: Provider configuration, offer discovery, spot selection, pod creation, and pod deletion are independent prerequisites that can be proved with a public image. This separates provider failures from registry-auth or mirror-readiness failures. Start with RunPod Secure Cloud (`community_cloud: false`); Community Cloud is a later explicit expansion after the secure path is qualified.
  Date/Author: 2026-08-30 / Codex

- Decision: Treat public-image availability as a pre-provisioning condition for RunPod.
  Rationale: RunPod pulls the image as part of pod creation, so a certificate installed during provisioning is too late. The real cloud gate must ensure the exact manifest is verified in the public R2-backed registry before RunPod becomes eligible, without moving R2 knowledge into Job Builder or Posttrain.
  Date/Author: 2026-08-30 / Codex

- Decision: Keep ai-infra service ownership centralized and expose only narrow cloud runtime edges.
  Rationale: RunPod does not replace ai-infra. The control plane, builder, registry mirror, Trackio storage, Doris, RustFS, devpi, Observatory, and LAN-worker enrollment remain managed where they are. Add separate public, authenticated or read-only edges only for the canonical OCI registry, the immutable dstack runner binary, and Trackio writes. Do not expose databases, object-storage credentials, the publication controller, devpi, Job Builder, or Observatory to RunPod.
  Date/Author: 2026-08-30 / Codex

- Decision: dstack core owns typed lifecycle truth and a transactional hook outbox; ai-infra owns the allow-listed automation behind those hooks.
  Rationale: a generic signed-webhook or fixed-command executor lets an operator trigger Ansible and custom commands without importing site repositories, playbook names, Trackio, or secrets into dstack. At-least-once delivery with stable IDs also survives server restart and spot loss, unlike in-process callbacks.
  Date/Author: 2026-08-30 / Codex

- Decision: combine a dstack-owned run-scoped volume with Trackio-owned durable checkpoint publication.
  Rationale: dstack alone can coordinate provider placement, attachment, retries, cancellation, terminal state, exclusive writer fencing, and idempotent deletion. Trackio already owns verified artifact transport and lineage. A hook-created volume could be leaked or destroyed between external execution and scheduler persistence, while Trackio alone would make every spot retry wait for a multi-gigabyte restore. RunPod retries retain the same network volume and data center when possible; a new volume elsewhere restores only the latest Trackio-verified recovery artifact.
  Date/Author: 2026-08-30 / Codex

- Decision: install CUDA forward-compatibility libraries in the veRL child image and select them automatically in the image runtime before any CUDA consumer is imported, never from provider policy, the universal base, or a global image environment variable.
  Rationale: the incompatibility is specific to a CUDA 13 runtime placed on an older supported data-center driver. Keeping the payload in one additive veRL layer preserves the shared base and all established cache descriptors. An image-owned native driver-version probe gives the same immutable image correct behavior on RunPod R570 and newer LAN drivers without hardware-specific task definitions; provider configuration does not need to understand CUDA library paths.
  Date/Author: 2026-08-30 / Codex and user

- Decision: define two production gates: stateless cloud execution and interruptible training.
  Rationale: the successful R2/A100 canary proves provisioning, pulling, CUDA, terminal reporting, and compute deletion, but it does not prove retained Trackio evidence, checkpoint durability, volume reuse, eviction classification, or cleanup after interruption. Stateless cloud execution may be enabled only after immutable dstack deployment, registry-readiness admission, guarded CUDA activation, Trackio write ingress, and LAN/cloud rollback gates. Spot training additionally requires all Milestone 4B storage, checkpoint, interruption, fencing, and cleanup evidence; it must not inherit a weaker "canary passed" label.
  Date/Author: 2026-08-30 / Codex

- Decision: implement functional readiness before security hardening, then make security the final non-bypassable release gate.
  Rationale: this is the user's requested execution order. Functional development remains limited to isolated qualification projects, strict provider bounds, and production fallback disabled. Deferring hardening does not authorize a production rollout, diagnostic logging of secrets, or broader external exposure; redaction publication, credential-boundary validation, and negative tests must all pass immediately before promotion.
  Date/Author: 2026-08-30 / Codex and user

- Decision: preserve the existing working credentials throughout this plan.
  Rationale: security qualification validates protected storage, least privilege, non-disclosure, and revocation capability without replacing or revoking the active RunPod, R2, RustFS, registry, Trackio, hook, or Tunnel credentials. Credential replacement is not a milestone or release action here.
  Date/Author: 2026-08-30 / Codex and user

- Decision: Cloudflare R2 is used only by the external OCI registry; Trackio artifacts use CarbonTeq's existing self-hosted RustFS S3 service.
  Rationale: the maintained Trackio fork already supports direct presigned multipart uploads to any S3-compatible endpoint, and ai-infra already operates a pinned RustFS service with persistent storage. Trackio can use RustFS privately for server operations while signing a separate worker-reachable S3 hostname for direct uploads and downloads. Checkpoint bodies bypass the Trackio process but, unlike cloud object storage, traverse CarbonTeq's public ingress and land on CarbonTeq-owned storage. This is the intended ownership and bandwidth tradeoff. Trackio remains the artifact and digest authority; RustFS stores bytes, and R2 remains unrelated to Trackio.
  Date/Author: 2026-08-30 / Codex and user

- Decision: activate mandatory spot storage through an optional provider-backend `run_storage` policy.
  Rationale: ai-infra needs to require storage for interruptible RunPod jobs without changing ordinary on-demand jobs, existing dstack behavior globally, or teaching Posttrain about RunPod. When the setting is absent, planning and provisioning remain byte-for-byte compatible and no implicit volume exists. When `mode: per_run`, `required_for: [spot]`, and `required: true` are set, the selected backend creates one owned volume after placement chooses a data center and fails spot provisioning rather than falling back to ephemeral storage. The policy is a generic capability model that future providers may implement; RunPod is only its first adapter.
  Date/Author: 2026-08-30 / Codex

- Decision: model spot eviction as an attempt interruption and report it from provider-observed server state.
  Rationale: an evicted pod may never run shutdown code. The dstack server must record `attempt.interrupted`, preserve the logical run while retry policy permits, and let a generic receiver reconcile Trackio. Periodic checkpointing supplies resume state; no design promises an impossible last-second checkpoint.
  Date/Author: 2026-08-30 / Codex

- Decision: expose inventory as separate backend, retained-capacity, provider-candidate, allocation, storage, and automation sections.
  Rationale: owned LAN machines, a provider catalog, live pods, billed volumes, and failed hooks have different authorities and meanings. Each provider-derived row carries source and observation freshness; unknown or stale is never rendered as available or zero.
  Date/Author: 2026-08-30 / Codex

- Decision: Use a real public DNS name as the canonical pull identity and retain separate management names for writes.
  Rationale: Cloud workers cannot resolve or trust `registry.lan`. The public route for the canonical name must have a publicly trusted certificate. Inside the LAN, authoritative DNS resolves it to `ai-control` and managed provisioning installs the private CA for that same name; public DNS resolves it to the external pull ingress. Private names such as `registry.lan` or `registry-cloud-push.lan` remain operational endpoints but never appear in the submitted cloud-capable image identity.
  Date/Author: 2026-08-29 / Codex

- Decision: Use location-appropriate TLS for the same canonical hostname: the existing private Caddy CA on managed LAN systems and Cloudflare edge TLS externally.
  Rationale: OCI identity depends on manifest bytes, repository path, and digest, not certificate bytes. Requiring a public ACME certificate on the private LAN route would add DNS-01 automation and a Cloudflare DNS-edit token without improving cloud compatibility. The worker role installs the existing Caddy root during LAN worker provisioning under the canonical registry host as well as `registry.lan`; unmanaged LAN clients are outside the supported pull path until they receive that root. Cloud execution must not depend on provisioning-time CA installation because provider-native container backends can pull the job image before a startup script runs. External probes and cloud workers therefore validate through the ordinary public trust store.
  Date/Author: 2026-08-29 / Codex

- Decision: Keep the LAN filesystem registry and deploy a second Distribution 3 instance backed by one private R2 Standard bucket.
  Rationale: This avoids routing LAN pulls through the Internet and avoids a destructive storage migration. R2 Standard has no retrieval charge and suits repeated cold image pulls; Infrequent Access would add retrieval charges and a minimum storage duration.
  Date/Author: 2026-08-29 / Codex

- Decision: The R2-backed registry handles OCI manifests and authorization, then redirects large blob downloads to presigned R2 URLs.
  Rationale: Raw R2 is not an OCI Distribution API. Distribution 3 already supplies the OCI API and its S3 driver supports S3-compatible `regionendpoint`, path-style requests, Signature V4, multipart upload, and blob redirect. Cloud workers therefore never receive an R2 key, and large transfers bypass the LAN registry process.
  Date/Author: 2026-08-29 / Codex

- Decision: Use the explicit `cloud_registry_r2_endpoint` issued with the R2 credential instead of deriving an endpoint from a separately configured account ID.
  Rationale: the exact endpoint is already the S3 API authority and passed the real compatibility gate. Treating it as authoritative avoids incorrect assumptions about jurisdiction-specific endpoint forms while keeping the same strict HTTPS and R2-host validation.
  Date/Author: 2026-08-29 / Codex

- Decision: Registry garbage collection is an offline, infrastructure-owned operation.
  Rationale: the compatibility proof deletes the manifest, stops the disposable registry, runs the pinned Distribution binary's collector, and verifies the unreferenced layer is absent from R2. Ordinary publication, project cleanup, and a running production registry never invoke it.
  Date/Author: 2026-08-29 / Codex

- Decision: Cloud-capable publications are complete only after both registry copies verify the same manifest digest.
  Rationale: dstack may select cloud only after submission. If external replication were asynchronous, a valid placement could fail before user code because the selected image was absent. Synchronous verification makes submission the safety barrier. A later optimization may make cloud replication lazy only if dstack gains an explicit image-readiness offer gate; that is outside this plan.
  Date/Author: 2026-08-29 / Codex

- Decision: Preserve inherited OCI blobs byte-for-byte and never recompress while copying.
  Rationale: The destination must expose the exact same manifest and ordered layer digests. Recompression would change blob and manifest digests, defeat caches, and violate the single-image-identity premise.
  Date/Author: 2026-08-29 / Codex

- Decision: Registry infrastructure owns replication and retention; the cloud scheduler only waits for readiness after selecting the cloud backend.
  Rationale: The frozen baseline already assigns registry operation, credentials, replication, caches, and retention to infrastructure. One registry mirror can serve framework base releases and per-job images, keep destination credentials out of framework code, and evolve registry topology without changing the public Job Builder protocol or Posttrain's dstack adapter.
  Date/Author: 2026-08-29 / Codex

- Decision: Make Distribution manifest-push events the normal mirror trigger and retain explicit reconciliation only for infrastructure recovery, release seeding, and readiness inspection.
  Rationale: Registry replication belongs to registry infrastructure, not Job Builder. The receiver writes the immutable repository/digest into SQLite before returning success and retries from that durable queue. An operator can reconcile retention roots or a missed event without exposing a mirror API or token to framework services.
  Date/Author: 2026-08-30 / Codex

- Decision: Select LAN versus external registry access during machine provisioning, without rewriting the image.
  Rationale: Managed LAN machines need the private hostname override and Caddy CA; external cloud machines must use public DNS and a publicly trusted certificate, including provider-native image pulls that may occur before startup scripts. Both receive the same canonical digest, so dstack remains responsible only for placement, credentials, and machine lifecycle.
  Date/Author: 2026-08-30 / Codex

- Decision: Pin regclient `regctl` v0.11.5 by multi-platform manifest digest for copying and verification.
  Rationale: It performs registry-to-registry copies without a local image store, skips blobs already present at the target, preserves manifest identity, and supports recursive repair. Pinning `ghcr.io/regclient/regctl:v0.11.5@sha256:dbe356c6cf9f8f85e302b9e47fed481ef3f1b04807350e99b02ab2cadee0a993` makes the controller reproducible.
  Date/Author: 2026-08-29 / Codex

- Decision: Do not amend the frozen Posttrain product baseline.
  Rationale: The baseline already assigns immutable image contents and publication contracts to Posttrain, registry and credentials to infrastructure, and placement to dstack. This work adds a site-owned replica and transport route without changing job, run, artifact, or provider meaning.
  Date/Author: 2026-08-29 / Codex

## Outcomes & Retrospective

Planning and disposable tests established that no dstack image-routing feature is required. The same canonical digest was pulled successfully by separate Docker daemons whose DNS answers selected different registries. Milestone 1 proved the pinned Distribution 3 image against the real private R2 bucket, including multipart storage, method-specific signed redirects, ranged resume, clean pull, deletion, and offline GC. The production edge now implements the corrected path: the LAN canonical endpoint returns the filesystem-registry manifest, the dedicated Tunnel listener returns the byte-identical R2 manifest, public anonymous/authenticated/delete checks return 401/200/405, and a real LAN manifest event reached a verified R2 receipt without any Job Builder call. The optimized seven-root graph is seeded and occupies 9,321,477,036 unique compressed layer bytes after the veRL-only CUDA compatibility addition. The incorrect framework-facing Job Builder readiness seam has been removed. Remaining registry rollout gates are managed-worker Docker trust, bounded cloud admission during mirror lag, and an actual-job rather than kind-image receipt.

The provider-only gate and the registry-backed workload gate both pass on the isolated candidate control plane. The latter pulled canonical veRL digest `sha256:9bf4ff...` through public R2, activated the image-local CUDA 13 compatibility payload for an R570-era host driver, executed PyTorch CUDA on a RunPod A100 80 GB PCIe spot Pod, completed for $0.0711, and left the provider Pod absent. This proves stateless execution mechanics but not production-safe training. The current production fleet still has one running job and one idle worker, so the immutable dstack promotion gate remains closed without disturbing work. The remaining functional sequence is guarded CUDA activation, actual-job registry admission, public Trackio plus self-hosted RustFS transport, provider-authoritative attempts, hooks, run-scoped storage, recovery, inventory, and the resilience matrix. Diagnostic redaction publication, credential-boundary validation, and the complete security audit follow as the final engineering milestone before any production rollout; active credentials remain unchanged.

## Context and Orientation

An OCI image is a content-addressed container image. Its manifest lists a configuration blob and ordered filesystem-layer blobs by SHA-256 digest. A registry is the HTTP service that implements the OCI Distribution API; object storage such as R2 holds bytes but is not by itself a Docker-compatible registry. A mirror is another registry location holding the same manifest and blob bytes. The hostname is not part of an OCI manifest digest, so two registries can serve the same digest as long as they retain identical bytes under the same repository path.

Split-horizon DNS means one hostname has different answers depending on the requesting network. For this plan, `registry.<public-domain>` is the canonical pull hostname. LAN DNS returns the private `ai-control` address. Public DNS returns the Cloudflare ingress for the R2-backed registry. The repository path and digest are unchanged. Use a real operator-owned domain in implementation; the placeholder must never ship literally.

The current Posttrain lifecycle is:

    Posttrain resolves one job and materializes a sealed context.
    The optional remote job builder builds one immutable actual-job image.
    The builder pushes registry.carbonteq.com/<project>/posttrain-job@sha256:<digest> through the LAN route.
    The LAN registry event durably enqueues the same repository and digest for R2 mirroring.
    packages/execution-dstack submits that digest to dstack.
    dstack selects a worker, pulls the image, starts it, observes it, and cleans it up.

`packages/execution-pack` owns `JobPackageManifest`, `ImagePublicationSpec`, and the application-facing publisher protocols. `packages/execution-buildkit` owns the BuildKit implementation and registry inspection. `apps/job-builder` is the isolated service that accepts only a sealed context and server-owned settings. `packages/execution-dstack` translates a framework launch request into dstack configuration but must not learn about R2.

`../ai-infra/services/registry-publication-controller/` is currently named for its recovery API but functions as the site-specific registry mirror worker. A LAN Distribution manifest-push event names only a validated repository and immutable `sha256` manifest digest. The receiver records the work in SQLite before responding, derives fixed source and destination registries from protected configuration, invokes pinned `regctl` to copy missing OCI content, verifies both manifest digests, and stores a durable receipt. Its explicit reconcile/status API is infrastructure-only: release seeding, readiness inspection, and recovery from a missed event use it; Job Builder does not.

In `../dstack`, `src/dstack/_internal/core/models/runs.py` owns `JobSpec.image_name` and `registry_auth`. `runner/internal/shim/docker.go` owns image cache lookup and pull. `src/dstack/_internal/server/services/backends/provisioning.py` has a placement-aware image hook, but this plan does not extend it because DNS can solve the location problem for VM, SSH, Kubernetes, and provider-native container pulls consistently. `src/dstack/_internal/server/services/docker.py` owns server default registry and credential resolution; its current early return for explicit registry hosts is the one dstack seam this plan changes.

In `../ai-infra`, `ansible/roles/control/files/compose.yml` runs the current pinned `registry:3.0.0` with filesystem storage at `/srv/ai-control/registry`. `ansible/roles/control/templates/Caddyfile.j2` exposes it only as `registry.lan` using the deployment-local CA. The job-builder role and playbooks are committed and merged; R2 work is isolated on a dedicated branch from current `main`. `scripts/verify` invokes the LAN registry qualifier and the new real-R2 compatibility harness.

Cloudflare R2 supplies an explicit S3-compatible HTTPS endpoint; its S3 region is `auto`. The deployment accepts that endpoint directly rather than deriving it from an account ID, so jurisdiction-specific forms remain valid. The dedicated bucket must be private. The registry service receives one bucket-scoped Object Read and Write S3 credential in protected infrastructure state. Cloud workers receive only OCI pull credentials. Distribution generates time-limited, method-specific signed R2 URLs for authorized blob reads. R2 presigned URLs use the R2 S3 API hostname rather than a custom domain; this is expected and the redirect host must be allow-listed by qualification without retaining its signed query string.

The external registry needs two ingress surfaces over one R2-backed registry service. A private push hostname is reachable from the builder and permits OCI writes. The canonical public hostname permits authenticated `GET`, `HEAD`, and the minimum Docker-discovery requests only; Caddy must reject manifest upload, blob upload, and delete methods before they reach Distribution. Because blob GET responses redirect to R2, the public ingress handles small manifests and authorization rather than multi-gigabyte response bodies.

The normal implementation mirrors every immutable manifest pushed to the LAN registry. A build completing means the source image is valid; it does not falsely claim that the R2 copy is already ready. Before enabling cloud fallback, qualification must prove a bounded infrastructure-owned readiness behavior for the slow-copy race, either by delaying cloud eligibility until the durable receipt is verified or by proving the selected dstack backend retries an initially unavailable manifest without leaking a failed framework run. Do not move that wait back into Job Builder and do not silently claim cloud readiness when only the LAN copy exists.

## Plan of Work

### Remaining production-readiness sequence — authoritative after the RunPod canary

The milestones below are the authoritative order for the work that remains after the successful R2-backed A100 canary. Earlier milestones remain as implementation history and detailed component context. This sequence does not amend the frozen Posttrain product baseline: Posttrain continues to own the logical run, immutable actual-job package, evidence contract, artifacts, and reconciliation; dstack owns placement and provider lifecycle; ai-infra owns services, credentials, registry topology, and deployment. Security hardening is deliberately the last engineering milestone at the user's direction. Nothing may be called production or enabled for ordinary jobs until that last milestone passes.

Treat each row below as a separately reviewable release unit. Do not combine dirty work across repositories or advance a consumer pin before its producer commit is published.

| Order | Release unit | Owning repository | May deploy before security? | Promotion dependency |
| --- | --- | --- | --- | --- |
| A | Automatic CUDA compatibility selection and rebuilt veRL image | `rl` | Qualification only | Exact local and RunPod digest gates |
| B | Generic image-readiness precondition | `dstack`, then `ai-infra` | Candidate control plane only | Published dstack fork commit and actual-job mirror proof |
| C | Trackio public writes and direct self-hosted RustFS artifact backend | `trackio`, then `rl` pin if changed, then `ai-infra` | Isolated qualification only | Verified migration, worker-reachable multipart, backup, and rollback read path |
| D | Provider-authoritative observations and attempt persistence | `dstack`, then `ai-infra` | Candidate control plane only | Migration and forced-disappearance proof |
| E | Transactional lifecycle outbox and allow-listed hook receiver | `dstack`, then `ai-infra` | Candidate control plane only | Restart/deduplication/dead-letter proof |
| F | Run-scoped spot storage and cleanup reconciliation | `dstack`, then `ai-infra` | Bounded disposable runs only | Same-volume retry and provider-absence receipt |
| G | Cross-region checkpoint recovery and inventory | `dstack`, `trackio` if generic fixes are needed, `rl`, then `ai-infra` | Bounded disposable runs only | Immutable artifact restore and truthful inventory proof |
| H | Full functional resilience matrix | Evidence only across deployed candidates | Qualification projects only | Every functional receipt complete; no leaked resources |
| I | Redaction release, credential-boundary validation, ingress hardening, and negative tests | `dstack`, `trackio` if needed, `ai-infra` | This is the final release gate | All security receipts complete; existing credentials unchanged |
| J | Immutable production deployment and bounded project rollout | `ai-infra` with published producer revisions | Yes, after I only | Idle LAN gate, canaries, rollback readiness |

Within a release unit, the order is fork code and tests, fork ledger, fork commit and push, consumer pin/configuration, candidate deployment, live qualification, then retained receipt. A failed live gate rolls back only that unit's candidate resources and does not rewrite already verified image or artifact identities.

#### Production Milestone A — Make CUDA compatibility automatic and image-owned

Remove the manual `LD_LIBRARY_PATH` setting from RunPod task definitions without touching the universal base image. The veRL child already contains the pinned CUDA 13 compatibility payload at `/usr/local/cuda-13.0/compat`; retain that one additive layer and every existing parent descriptor. Add a small declarative compatibility record to the veRL image containing the compatibility path, CUDA runtime API version, and payload source digest. Do not add another image lineage, recompress inherited blobs, or set a global compatibility path.

Implement early runtime selection in `apps/runtime/src/posttrain_runtime/cuda_compat.py` and invoke it from `apps/runtime/src/posttrain_runtime/cli.py` before importing `posttrain_runtime.execute` or any backend that may import PyTorch. In `auto` mode, load the host-provided `libcuda.so.1`, call the driver API version query, and do nothing when the native driver already satisfies the image's CUDA runtime. When the driver is older and the image declares a compatibility payload, re-exec `posttrain-runtime` exactly once with the compatibility directory prepended to `LD_LIBRARY_PATH`; use a private guard variable to prevent a loop. `off` exists for diagnosis and must retain native selection; `force` exists only for qualification. A missing declaration, unsupported GPU/driver combination, failed compatibility initialization, or re-exec loop fails native runtime preflight with a typed safe error before user code starts.

Add pure tests under `apps/runtime/tests/` for no declaration, native-newer, native-older, off, force, re-exec guard, malformed declaration, and safe errors. Extend `packages/runtime-images/tests/test_verl_release_gate.py` so the compatibility payload remains veRL-only, follows the established 24-layer graph, and is inactive globally. Rebuild with `force-compression=false`; prove the base and prior veRL descriptors remain byte-identical. Qualify the exact new digest on the local RTX 3070 Ti without an environment override and on a RunPod A100/R570 host without an environment override. Acceptance is the same digest printing successful CUDA execution on both hosts and the RunPod Pod becoming absent after completion.

If automatic activation fails on either host, stop here. Keep the last locally qualified digest canonical, disable cloud eligibility for CUDA 13 veRL jobs, and retain the disposable RunPod evidence. Do not change the base, downgrade PyTorch, create a hardware-specific image, or globally force compatibility as a workaround.

#### Production Milestone B — Gate cloud provisioning on registry verification

Prevent the mirror-delay race from creating a billed Pod that cannot pull its image. Add a generic placement-time provisioning precondition to the dstack fork, configured per backend and absent by default. Its first implementation is an authenticated HTTP readiness probe whose request is derived from the already-resolved image repository and immutable manifest digest. The precondition runs only after dstack selects RunPod and before provider resource creation; LAN SSH placement never calls it. A tag-only image is ineligible for this guarded cloud backend.

Persist the guard snapshot and state with the attempt: `waiting`, `ready`, `timed_out`, or `failed`. A pending mirror keeps the logical job waiting without creating a Pod or changing framework run identity. A verified receipt permits the existing RunPod provisioning path. Timeout produces a typed pre-start no-capacity/readiness result that is safe for bounded retry and cannot be confused with user-code failure. Restarting the server resumes the same wait from persisted state. The HTTP adapter is generic; only ai-infra configuration names the internal registry publication controller and its fixed verified-state contract. Job Builder, `packages/execution-dstack`, and Posttrain requests receive no R2 or mirror fields.

Implement the core model and server pipeline in `../dstack/src/dstack/_internal/core/` and `../dstack/src/dstack/_internal/server/background/pipeline_tasks/`; render its RunPod-only configuration in `../ai-infra/ansible/roles/control/templates/dstack-config.yml.j2`. Extend the registry controller status response only if a stable machine-readable state is missing; do not expose copy commands, endpoints, or credentials. Tests must prove absent configuration is byte-for-byte compatible, LAN placement bypasses the guard, pending state creates no provider call, verified state creates exactly one provider call, timeout is pre-start, malformed/non-digest images fail closed, duplicate processing is idempotent, and restart resumes waiting.

The real gate publishes a tiny actual-job image through the existing remote builder, delays its mirror worker deliberately, submits one RunPod-constrained run, and proves the account has no Pod until the receipt becomes `verified`. It then releases the mirror, observes exactly one Pod, executes the immutable digest, and confirms deletion. This is the point at which actual-job rather than job-kind mirroring is proven.

#### Production Milestone C — Give cloud jobs durable Trackio transport

Expose only Trackio's authenticated write API through a canonical public-TLS hostname. LAN DNS may route that hostname privately while external DNS routes through Cloudflare Tunnel, but both paths must reach the same Trackio application contract and use the framework run identity. Doris, Observatory, PostgreSQL, the publication controller, and object-store credentials remain private. Keep this independent from the OCI R2 registry.

Switch Trackio artifact bytes from the current local backend to its existing S3-compatible direct-upload implementation, backed by ai-infra's pinned self-hosted RustFS service. Create a dedicated `trackio-artifacts` bucket and production prefix with a Trackio-only S3 identity. Trackio uses the private RustFS address for server operations and `TRACKIO_ARTIFACT_S3_PRESIGN_ENDPOINT` for a worker-reachable public-TLS S3 hostname. Trackio keeps the S3 identity server-side, creates short-lived multipart URLs, and exposes only those URLs to the worker. The worker uploads parts directly to RustFS, while Trackio commits the artifact version only after object size and SHA-256 verification. Metrics, events, and small trace batches continue through the Trackio API.

Complete configuration in `../ai-infra/ansible/roles/control/templates/control.env.j2`, the control Compose file, the storage role, the public-TLS S3 route, and protected secret state. The public route is an object-data plane only: it forwards the S3 operations required by valid presigned multipart uploads and downloads and does not expose the RustFS console or management API. Choose multipart part size within the qualified ingress request limit. Reuse `../trackio/trackio/direct_uploads.py`, `artifact_storage.py`, and the maintained fork's existing conformance tests; generic storage fixes belong in the Trackio fork, while Posttrain-specific run/artifact meaning remains here. Add an additive migration command that copies and verifies retained local artifacts before switching the production backend; keep the local store readable for rollback until the compatibility window closes. Add capacity alerting and a tested backup/restore path because RustFS data remains CarbonTeq-owned persistent state rather than cloud-provider durability.

Acceptance uses one RunPod job that opens the expected Trackio run, streams metrics and checkpoint events, uploads an artifact larger than one multipart part directly to the public RustFS data endpoint, and finishes. Network evidence must show the artifact body bypassing the Trackio process and arriving in the self-hosted bucket; it will traverse CarbonTeq's ingress by design. Trackio must report the immutable artifact digest and Observatory must read the completed run. Killing the uploader midway must leave the prior verified artifact authoritative, expose the interrupted upload as incomplete rather than committed, and allow an idempotent retry. A restore test must recover the object and metadata from the documented RustFS backup before this milestone is complete.

#### Production Milestone D — Make provider state authoritative and attempts explicit

Extend dstack's compute contract with a typed provider lifecycle observation containing provider resource identity, presence, desired state, interruptible flag, observed timestamp, safe reason, and source freshness. Implement RunPod using its Pod query and distinguish normal completion, explicit cancellation, provider disappearance for a spot Pod, API outage, and transport-only runner loss. Provider evidence is primary; SSH/runner disconnection is a fallback and never fabricates provider termination.

Persist logical run versus attempt identity so a spot loss terminates one attempt without terminating the logical run while retry policy permits. Every attempt records its image digest, backend, data center, Pod id, start/finish observations, terminal classification, and evidence reconciliation state. Duplicate observations and server restart must converge on one attempt transition. Provider API unavailability retains `unknown/stale` rather than converting it to interrupted or zero capacity.

The principal changes belong in the dstack fork's core backend model, RunPod adapter, SQLAlchemy models/migrations, running-job pipeline, schemas, and tests. Framework execution continues to expose one canonical run and must record provider attempt identities as execution metadata rather than creating a second logical Posttrain run. Acceptance force-deletes a disposable spot Pod from RunPod, observes `attempt.interrupted` from provider absence, and proves no success/failure is inferred merely from a dropped runner connection.

#### Production Milestone E — Add a transactional lifecycle outbox and safe infrastructure hooks

Create versioned lifecycle events and delivery rows in dstack's database in the same transaction as their state transition. Each event has a stable id, schema version, kind, project/run/attempt/resource identities, safe payload, created time, and sequence. Each configured delivery has retry count, next attempt, acknowledgement, terminal failure, and dead-letter state. At-least-once delivery plus receiver-side event-id deduplication is the contract.

Implement two generic executors: a signed HTTP webhook and a server-admin fixed-command executor. The command executor accepts only a configured argument vector, passes the event document as data, never invokes a shell, and cannot receive unlisted environment values. In ai-infra, add a separately authenticated receiver whose allow-list maps stable hook ids to repository-owned Ansible playbooks or fixed commands. This supplies the clean platform hook requested for provisioning and cleanup without teaching dstack CarbonTeq hostnames, playbook names, Trackio, or Posttrain semantics.

Cover commit/dispatch races, restart, duplicate delivery, bounded exponential retry, timeout, acknowledgement, dead letter, disabled hooks, receiver deduplication, and safe payloads. A live proof restarts the dstack server after committing an event but before delivery and observes exactly one receiver action after recovery. Hooks report lifecycle; they do not own RunPod Pods or volumes and cannot override authoritative provider state.

#### Production Milestone F — Add spot-only run-scoped network storage

Add optional `run_storage` to provider backend configuration exactly as described later in Milestone 4B. Absence preserves current behavior. For `mode: per_run`, `required_for: [spot]`, and `required: true`, evaluate policy only after RunPod offer/data-center selection. Persist the resolved policy and logical-run owner before creating a volume, create exactly one RunPod network volume in that data center, and pass its provider id to every attempt Pod at `/workspace`. Never silently fall back to container disk or ephemeral volume disk.

Create a durable run-volume ownership model separate from named shared volumes. It records logical run, provider, data center, volume id, size, mount path, policy revision, writer attempt, lifecycle state, finalization deadline, deletion attempts, provider absence, and cleanup receipt. A retry loads that owner rather than evaluating defaults again. RunPod's one-network-volume limit means an explicit named volume conflicts with mandatory per-run policy and must fail before provisioning.

Enforce an exclusive writer lease. A new attempt may attach only after authoritative provider evidence confirms the prior Pod absent or a provider-specific fence succeeds. Prefer same-volume, same-data-center retry. Terminal success or cancellation enters a cleanup barrier: compute must be absent, required finalizers must acknowledge or hit their bounded deadline, volume deletion must be requested idempotently, provider absence must be confirmed, and only then is the cleanup receipt complete. A reconciler resumes partial deletion and reports orphans without deleting an unowned resource.

Unit and integration tests must prove no-policy compatibility, no implicit volume for on-demand or LAN runs, exactly one volume for a logical spot run, attach-before-command, failure-closed create/attach, retry reuse, named-volume conflict, fencing, terminal cleanup, cancellation cleanup, server restart during deletion, and orphan reporting. The real canary writes a marker beneath `/workspace`, loses its first Pod, attaches a second attempt to the same volume, reads the marker, completes, and leaves both Pod and volume absent with a retained receipt.

#### Production Milestone G — Add checkpoint recovery and truthful inventory

Keep local save cadence and durable publication cadence distinct. A trainer writes a checkpoint atomically to `/workspace`; only complete checkpoints are candidates for Trackio background publication. Trackio's verified artifact pointer is the provider-neutral recovery authority. Same-data-center retry reads the volume first. When bounded local capacity is exhausted, create a new volume elsewhere, restore the latest exact Trackio checkpoint into it, record that it may be older than the volume state, and start a new attempt. Never select an incomplete multipart upload or a mutable `latest` pointer without resolving it to an immutable digest.

Add `GET /api/project/{project_name}/inventory` and `dstack inventory`. Return separate backend configuration, retained LAN capacity, provider candidates, active allocations, run-scoped storage, and automation deliveries. Every provider-derived record has source, observed time, and freshness; stale or unknown is never rendered as available. Surface logical run, attempt, Pod, volume, cleanup, and hook relationships without embedding Trackio payloads or secrets.

Acceptance performs a cross-data-center restore into a fresh volume, resumes from the last Trackio-verified checkpoint, and then cleans both generations of temporary compute/storage according to ownership. Inventory must show the transition while active and no active allocation or volume afterward; the cleanup receipt, attempt history, and durable artifact remain queryable.

#### Production Milestone H — Run the complete functional resilience matrix

Before security hardening, run the functional matrix only on isolated qualification projects and bounded resources. It includes: LAN canonical pull with zero R2 blob traffic; cloud exact-digest wait during mirror lag; warm and cold R2 pulls; successful Trackio metrics and direct RustFS checkpoint upload; normal completion; explicit cancellation; forced spot disappearance; same-volume retry; cross-data-center restore; dstack restart during lifecycle-event dispatch; restart during volume deletion; RunPod API outage; Trackio unavailability; RustFS unavailability; public S3 ingress interruption; and incomplete multipart cleanup. Each case records cost, time to first user process, pull bytes, provider transitions, attempt identity, artifact digest, finalizer result, and confirmed provider absence.

Then run automatic placement with both LAN and cloud eligible. Occupy LAN capacity using a bounded qualification workload rather than stopping a worker, submit one unchanged Posttrain run, and prove dstack selects RunPod without rebuilding or rewriting the digest. Release LAN capacity and prove the next run returns to LAN according to the declared offer policy. Keep production fallback disabled after the matrix.

Any leaked Pod, volume, ambiguous writer, missing artifact commitment, unbounded retry, or provider state reported as success blocks advancement. Clean only qualification-owned resources by exact provider id. Do not use registry-wide garbage collection, cancel unrelated work, or weaken the two-worker LAN release gate.

#### Production Milestone I — Perform security hardening last

Security is the final engineering milestone, as requested, and a hard release gate. Complete and publish the dstack runner change that logs environment names but never values; build the server, runner, and shim from one immutable fork commit. Do not use diagnostic mode on credential-bearing runs before this release is deployed. Preserve the current working credentials in mode-0600 ai-infra state: this milestone inspects their scopes and handling but does not regenerate, replace, or revoke them.

Audit and minimize every credential boundary: R2 registry write key, OCI pull credential, registry publication token, Trackio write token, RustFS Trackio identity, hook-signing key, Cloudflare Tunnel token, and dstack/RunPod key. Workers receive only the OCI pull credential, Trackio write token, and short-lived presigned artifact URLs. They never receive R2, RustFS access/secret keys, Doris, PostgreSQL, registry-copy, or Cloudflare account credentials. Enforce public method/path allow-lists for the registry, Trackio, and the presigned S3 data endpoint; bounded URL expiry; request/body limits; safe CORS; rate limits; and separate read/write identities. Keep the RustFS console and administrative API private.

Run secret scanning over source, rendered configs, retained receipts, normal logs, diagnostic logs, dstack API responses, lifecycle events, hook deliveries, Trackio records, and qualification artifacts. Negative tests must reject anonymous registry reads, every registry mutation method, invalid Trackio tokens, expired/replayed hook signatures, arbitrary hook commands, path traversal, foreign repository readiness queries, stale presigned URLs, and cross-project artifact access. Prove that signed query strings and authorization headers never enter logs or durable state.

If security qualification fails, revoke or disable only the affected external edge and keep LAN operation unchanged. Functional qualification evidence may remain, but cloud fallback stays disabled. No exception or "temporary" credential bypass is allowed at release.

#### Production Milestone J — Promote and roll out with rollback retained

Wait until both retained LAN workers are healthy and idle before deploying the immutable dstack release or restarting their Docker/runner services. Apply ai-infra from reviewed immutable revisions, verify the server/runner/shim component digests, install canonical registry trust on LAN workers, and repeat the LAN and cloud canaries. Preserve `registry.lan`, existing repository references, the upstream dstack deployment, local Trackio artifact data, and cloud-backend disable automation through the compatibility window.

Enable cloud fallback first for an allow-listed qualification project with a strict maximum price, maximum duration, Secure Cloud only, spot-only run storage, and zero-node fleet bounds. Observe provisioning latency, R2 registry operations, RustFS capacity/latency, Trackio health, public-ingress traffic, interruption rate, cleanup lag, orphan count, and actual RunPod cost. Expand project eligibility only after a defined observation window has no leaked resources, missing evidence, or unclassified provider terminal states.

Rollback disables RunPod eligibility and new hook dispatch, lets in-flight finalizers finish within their bounds, reconciles every owned Pod and volume, and returns new execution to LAN. It does not delete verified OCI manifests, Trackio evidence, RustFS artifacts, or cleanup receipts. Production acceptance is one unchanged digest choosing LAN when capacity exists, RunPod when LAN is unavailable, surviving a forced spot interruption, retaining verified evidence, and ending with no provider resource while the next LAN job remains unaffected.

### Milestone 0 — Preserve current work and establish immutable starting revisions

Before implementation, finish or isolate the existing remote-builder work. In `/home/hammad/projects/rl`, do not modify the currently dirty `apps/job-builder/tests`, runtime-image reduction, cache, release, or purge changes until their owner has either committed them or provided a clean worktree. In `../ai-infra`, the uncommitted job-builder role and playbooks must likewise be committed as their own logical change before R2 registry work builds on them. Do not commit or push merely to satisfy this plan without explicit user authorization.

Record `git status --short --branch`, `git rev-parse HEAD`, and relevant remote URLs for all three repositories in this plan. The expected implementation order is the framework publication seam; the generic dstack credential fix, tests, fork ledger, commit, and push; the ai-infra deployment pinned to immutable framework and dstack revisions; and finally live qualification. Follow `docs/tooling/forks.md` for the dstack change, and do not describe it as reproducible until the fork commit is published and the framework consumer page names that revision.

Acceptance is a clean or deliberately isolated edit surface for every file named by later milestones, with unrelated dirty changes still present and unchanged in their owning worktree.

### Milestone 1 — Prove Distribution 3 and R2 compatibility in isolation

Create a private R2 Standard bucket dedicated to the cloud registry. Use a non-production prefix or disposable bucket for the proof. Create one bucket-scoped read/write S3 credential; install it directly into mode-protected ai-infra secret state and never print it in logs or place it in Git.

Add a disposable ai-infra qualification command that starts the same pinned Distribution 3 image used by production with S3 storage configured for region `auto`, the supplied R2 `regionendpoint`, HTTPS, Signature V4, path-style requests, and redirects enabled. Use a unique `rootdirectory` so retry and cleanup cannot touch a future production registry prefix. Configuration names must be verified against the pinned Distribution 3 binary rather than copied blindly from prose.

Push a synthetic image containing at least one layer larger than the multipart threshold. Pull it from a clean Docker data root. Capture safe HTTP evidence showing that manifest/config requests reach Distribution and blob `HEAD` and `GET` requests independently return method-specific redirects whose host is the R2 S3 endpoint; never retain either query string. Use the `GET`-signed URL for ranged blob download and interrupted-and-resumed reconstruction. Test manifest deletion and offline garbage collection after stopping Distribution. Re-push the same bytes and prove that the manifest digest is unchanged and the second transfer reuses existing blobs.

If any required R2 API is incompatible with the pinned registry image, stop this milestone and record the exact operation and error. Do not change the production registry or weaken TLS verification. The next action is to evaluate a newer pinned Distribution patch or a proven OCI registry frontend over R2, preserving the same acceptance contract.

### Milestone 2 — Deploy the external registry and canonical pull identity

In `../ai-infra`, add an Ansible-owned R2 registry service beside, not in place of, the filesystem registry. Keep its container image digest-pinned. Render its S3 configuration from protected variables for endpoint, bucket, access key, secret key, and root prefix. Add readiness checks that validate both `/v2/` and a real private-bucket object round trip without printing credentials or signed URLs.

Add a private write ingress for the remote builder and a public read-only ingress for workers. The write ingress may use the LAN CA because only managed infrastructure calls it. The canonical hostname uses Cloudflare's publicly trusted edge certificate externally. Prefer a Cloudflare Tunnel or an equivalently outbound-only ingress because the registry returns small metadata and R2 redirects for large blobs; no inbound firewall opening is needed. The tunnel origin may be private HTTP on the container network because the tunnel itself is authenticated and encrypted, but it must not be reachable from an untrusted interface. Caddy must authenticate pulls and deny mutating OCI methods on the public route. Store the pull credential in dstack server protected state, not in project config or image metadata.

Create split-horizon DNS for the canonical hostname. LAN resolution must point to `ai-control` and public resolution to the external pull ingress. Configure the LAN Caddy route for this hostname with `tls internal`, then extend the existing worker, builder, and dstack-server provisioning roles so the same private root is installed for the canonical registry host before those systems may accept jobs. Restart Docker after changing host-scoped registry trust and prove a clean pull. Do not require that private root on cloud workers: provider-native backends may pull before provisioning hooks execute, so they must validate Cloudflare's edge certificate through the default public trust store. Keep `registry.lan` as an operational compatibility name during migration.

Add a deterministic qualification command under `../ai-infra/scripts/` and invoke it from `scripts/verify`. It must resolve the canonical hostname from the dstack server, builder, every eligible LAN worker, and an external probe; prove that internal and external clients reach different registry backends; compare an exact digest from each; and perform a clean Docker pull through each route. The LAN probes must verify against the installed private root and the external probe against the default public trust store. Its logs may contain safe hostnames, digests, byte counts, and status codes, but never authorization headers, R2 keys, signed query strings, or registry passwords.

### Milestone 3 — Make registry mirroring event-driven and recoverable

Add `../ai-infra/services/registry-publication-controller/` as a small unprivileged service with a SQLite work database on a retained volume. `POST /v1/publications` accepts only `{repository, digest}` after bearer authentication, validates a lower-case OCI repository path and a `sha256:<64 hex>` digest, inserts the unique request durably, and returns its current state. `GET /v1/publications/<repository>/<digest>` returns `pending`, `copying`, `verified`, or a safe retryable failure without exposing credentials or raw command output. Repeating either the request or a delivered notification for the same key is idempotent.

The worker invokes pinned `regctl image copy --force-recursive` from the fixed LAN source registry to the fixed private cloud write registry. Neither registry hostname nor credentials appear in the request. After copying, it independently reads the source and destination manifest descriptors and records `verified` only when both equal the requested digest. Store attempt counts, timestamps, safe error codes, and a receipt; keep full credentials in a mode-protected regctl configuration mounted read-only. On restart, change stale `copying` rows back to `pending` and retry them.

Configure the existing LAN Distribution instance to send manifest-push notifications to the mirror worker with a protected static authorization header. The webhook validates the media type, action, repository, and digest and commits the same unique mirror key to SQLite before acknowledging it. This is the normal trigger. Because Distribution's sender queue is in memory, retain the explicit reconcile command as an infrastructure recovery operation and use it for release-root seeding; it is not a framework service dependency.

Add an ai-infra release seeder that reads the committed framework `published.toml`, submits every universal-base and job-kind repository/digest to the controller, waits for verified receipts, and writes a compact retained inventory. Those manifests are retention roots: ordinary job cleanup cannot delete them, R2 object lifecycle rules must not expire their underlying objects independently, and offline registry garbage collection must consume the retained root inventory before deleting anything.

Configure Job Builder's server-owned repository prefix to the canonical hostname. Its managed DNS/hosts and private CA route that push to the LAN filesystem registry. Keep `JobPackageManifest`, `ImagePublicationSpec`, package key, publication key, and the external v1 request/response unchanged. Job Builder must contain no mirror client, mirror token, target hostname rewrite, R2 configuration, or replica status.

Unit tests must cover validation, authentication, idempotent enqueue, duplicate notification delivery, successful copy, already-present cache hit, recursive repair, unreachable destination, digest mismatch, process restart recovery, fixed destination policy, safe errors, and redaction. A two-registry disposable integration must prove the manifest bytes and digest are identical before the live R2 controller is used.

### Milestone 4 — Configure and qualify RunPod, then apply canonical registry credentials

RunPod is the selected first cloud backend. Create a dedicated RunPod API key with the permissions needed to inspect GPU offers and create, inspect, and delete pods. Store it only as `runpod_api_key` in ai-infra's protected mode-0600 secret state; do not place it in Git, the framework environment, or chat. Extend `../ai-infra/ansible/roles/control/templates/dstack-config.yml.j2` with a `runpod` backend under project `main`, using `creds.type: api_key`, the protected key, and `community_cloud: false`. Keep regions unrestricted initially so qualification exposes actual availability; restrict them later only from observed cost or locality requirements. Applying the server config requires a controlled dstack server restart.

Qualify the provider before changing the deployed dstack build. First validate the API key and enumerate RunPod offers without creating a pod. Then submit a minimal public-image task constrained to `backends: [runpod]` and `spot_policy: spot`, record the chosen Secure Cloud region, GPU, bid, and price, prove CUDA execution, and prove the pod is deleted after the run. Use a tiny public CUDA image for this provider-only canary so registry mirroring and private auth cannot obscure provisioning failures. Do not disable or mutate LAN workers to force this test; constrain only the canary.

Before that canary, add a public-TLS, read-only route for the immutable dstack runner artifact and change only `DSTACK_RUNNER_DOWNLOAD_URL` to that canonical external hostname. Keep the dstack server API and shim artifact on `dstack.lan`; RunPod needs the runner URL because its bootstrap command downloads the binary from inside the pod, whereas it does not need the LAN control UI. The artifact path must remain versioned and immutable and qualification must compare its SHA-256 with the retained dstack release declaration.

Only after that provider-only canary passes, publish and deploy the dstack exact-host credential fork described below. Then repeat the RunPod canary with a digest-pinned image from `registry.carbonteq.com`, after its mirror receipt is `verified`, and prove RunPod's control plane authenticates to the OCI frontend and follows R2 blob redirects. Because this pull happens before the container exists, do not attempt to install the LAN CA or split-horizon override on RunPod.

Keep the existing `remote` backend and `local-gpu-workers` fleet eligible before cloud capacity so normal work consumes zero-cost LAN capacity first, subject to dstack's actual offer and priority semantics. Automatic fallback is not enabled until the later mixed-placement qualification proves the intended priority and bounded behavior when the public mirror is not ready.

Posttrain continues submitting one digest-pinned canonical image. Do not add R2 fields, mirror lists, or backend conditionals to `packages/execution-dstack`. In the maintained dstack fork, change `apply_server_docker_defaults` so an explicit image registry that exactly equals `DSTACK_SERVER_DEFAULT_DOCKER_REGISTRY` receives `DSTACK_SERVER_DEFAULT_DOCKER_REGISTRY_USERNAME` and `DSTACK_SERVER_DEFAULT_DOCKER_REGISTRY_PASSWORD` only when the run did not provide explicit `registry_auth`. An explicit different registry receives no default credential, and explicit run auth keeps precedence. Do not rewrite the image name. Add focused tests in `src/tests/_internal/server/services/test_docker.py` for exact match, port-sensitive mismatch, malicious suffix/prefix mismatch, explicit-auth precedence, missing half of the credential pair, and unchanged unqualified-image behavior.

Configure the read-only canonical registry credential on the dedicated external Caddy listener and install it only in protected dstack server state so provider-native container backends and VM shim pulls can authenticate. The canonical LAN listener remains reachable only through managed private routing and proxies the existing LAN registry directly, preserving the current builder and worker behavior. Confirm that secrets are redacted in stored run configuration, API responses, and logs according to dstack's existing encryption and interpolation behavior. Update the fork root `CARBONTEQ_FORK.md` and the framework consumer page `docs/tooling/dstack/README.md`, run the fork tests, commit and push the fork, then deploy its immutable commit through ai-infra before relying on the behavior.

Run dstack's existing unit tests around job configuration, server Docker defaults, provider provisioning, and runner image pulling, including the new exact-host credential tests. If a provider performs image pulls through its own control plane, verify it can follow the registry's R2 redirect and authenticate to the OCI frontend; it must not need the R2 key. If a backend cannot use the canonical registry auth path, record it as unsupported rather than adding provider-specific credentials to Posttrain.

### Milestone 4B — Add durable cloud lifecycle, storage, hooks, and inventory

Implement ADR 0017 in the maintained dstack fork as a sequence of generic
capabilities. This milestone follows the public-image RunPod provisioning
canary so provider credentials and basic pod creation are already known-good;
it precedes any real spot training qualification.

First add provider lifecycle observation. Define a typed observation on the
compute backend and implement RunPod from the pod API's presence,
`desiredStatus`, interruptible flag, and safe timestamps. Reconcile active
provider instances on a bounded interval. Prefer provider evidence when
classifying a missing spot pod, retain SSH/runner loss as a fallback, and add
regression tests for success, explicit provider termination, disappeared spot
pod, provider API outage, and duplicate observation.

Next add versioned lifecycle events and hook deliveries as a transactional
outbox. Migrate existing operator messages without treating their prose as an
API. Implement a signed webhook executor and a protected server-admin command
executor that accepts only configured argument vectors, passes redacted event
JSON as data, and never invokes a shell. Add dispatcher restart, retry,
idempotency, timeout, fail-open/fail-closed, dead-letter, redaction, and
authorization tests. In ai-infra, add a separately authenticated receiver whose
allow-list maps hook IDs to repository-owned Ansible playbooks or fixed commands.
The receiver must deduplicate event IDs and expose safe delivery status.

Then add a generic optional `run_storage` policy to provider backend
configuration. Its initial rendered ai-infra form is:

    projects:
      - name: main
        backends:
          - type: runpod
            creds:
              type: api_key
              api_key: {{ runpod_api_key }}
            community_cloud: false
            run_storage:
              mode: per_run
              required: true
              required_for: [spot]
              size_gb: 100
              mount_path: /workspace
              retain_across_attempts: true
              cleanup: after_run
              finalization_timeout: 15m

The policy is absent by default and therefore changes no existing provider or
run. It is evaluated only after placement selects the backend, offer type, and
RunPod data center; evaluating it at initial submission would incorrectly create
cloud storage for a run ultimately placed on the LAN. `required_for: [spot]`
leaves ordinary on-demand attempts unchanged. For a matching spot attempt,
`required: true` means provider volume creation or attachment failure fails that
provisioning attempt safely and never launches the workload on ephemeral
storage. The selected data center, policy snapshot, and generated volume owner
are persisted before pod creation.

Add `RunScopedVolumeSpec` with resolved size, mount path, retention policy, and
the backend-policy revision that produced it. Create and durably own one network
volume for the logical run, attach it while creating every attempt pod, and pin
retries to that data center. A retry reuses the stored owner relation rather
than re-evaluating defaults and creating a second volume. RunPod supports only
one network volume, so a run that supplies an explicit named volume while the
mandatory `per_run` policy is active must be rejected with a clear conflict
instead of silently overriding either ownership model. Persist cleanup states
and deadlines. Logical terminal or cancellation must confirm compute absence,
settle required finalizers or their deadline, delete idempotently, confirm
provider absence, and record a cleanup receipt. Add a reconciler for interrupted
deletion and orphan detection. Named shared-volume behavior remains unchanged
on backends without the mandatory policy.

In the framework and Trackio integration, keep local checkpoint save cadence
separate from durable publication cadence. A completed checkpoint is first
written atomically to the mounted workspace, then selected checkpoints are
published through `Run.log_artifact(..., background=True)`. Configure Trackio's
S3-compatible artifact backend and a worker-reachable presign endpoint so the
RunPod worker uploads bytes directly rather than proxying multi-gigabyte blobs
through Trackio. Trackio must verify size and digest before advancing the
durable recovery pointer. This artifact bucket and credential boundary are the
existing self-hosted RustFS service and are separate from the OCI registry's
Cloudflare R2 bucket and credentials.

Give one attempt at a time an exclusive writer lease for the run-scoped volume.
A replacement attempt cannot start until provider evidence confirms the old pod
absent or a provider-specific fence is effective. Prefer retrying with the same
volume in its data center. When bounded same-data-center capacity policy is
exhausted, an explicit portability path creates a new volume elsewhere,
restores the latest Trackio-verified recovery artifact, records the older
recovery point, and starts a new attempt. It never copies a partial checkpoint
or treats an incomplete Trackio upload as recoverable.

Finally implement `GET /api/project/{project_name}/inventory` and
`dstack inventory`. Return separate backend, retained capacity, provider
candidate, active allocation, storage, and automation sections. Every
provider-derived record includes source, observation time, and freshness. Stop
representing an unverified RunPod catalog shape as live available capacity.
Add API/CLI serialization and stale/unknown-state tests before adding any UI.

Acceptance requires two real RunPod disruption tests. The first writes a
checkpoint marker, forces a bounded spot interruption, observes a durable
`attempt.interrupted` delivery, retries with a new attempt on the same volume
and data center, resumes from the marker, and completes. It must also prove a
selected checkpoint reaches Trackio's S3-compatible store by direct multipart
upload and is committed only after digest verification. The second creates a
new volume in another data center, restores that committed checkpoint, and
resumes without the original volume. A cancellation case proves bounded
finalization and cleanup. In every case, prove the pod is absent, every
run-scoped volume is absent, the cleanup receipt remains, and inventory contains
no active resource. Restart the dstack server between one terminal transition
and hook delivery to prove outbox and cleanup recovery. Do not claim spot safety
from a graceful `SIGTERM` test alone.

Backward-compatibility tests must prove that omitting `run_storage` produces the
same RunPod job request as the current fork and creates no implicit volume; an
on-demand RunPod attempt and a LAN-selected run create no implicit RunPod volume
when the policy applies only to spot; and unconfigured providers remain
unchanged. Policy tests must prove exactly one volume is created per logical
spot run, every attempt receives its provider ID, a
creation or attachment error cannot start the command, an explicit named-volume
conflict is rejected, retries do not allocate a second volume, and terminal,
cancelled, dispatcher-restarted, and partially deleted cases converge to one
confirmed cleanup receipt.

### Milestone 5 — Roll out additively and qualify both paths

Mirror the currently published base and job-kind graph into the external registry and record source/destination manifest equality and unique transferred bytes. Publish one tiny actual-job canary through the remote builder and verify its dual-publication receipts. Do not enable cloud fallback for production jobs yet.

First qualify the LAN path. Submit the canary constrained to `local-gpu-workers`. Observe that canonical DNS resolves to the LAN registry, dstack either reuses the cached image or pulls it from the LAN endpoint, the job completes, and R2 read counters do not move beyond bounded control noise. Then qualify the cloud path with LAN capacity deliberately excluded through a reversible dstack placement constraint, not by stopping or corrupting LAN workers. Observe cloud provisioning, public-registry authentication, R2 blob redirects, successful CUDA execution, retained Trackio logs/evidence, terminal reconciliation, and provider deprovisioning.

Before expecting retained Trackio evidence from the cloud canary, provide one canonical public-TLS Trackio API hostname that routes through a narrow Cloudflare ingress to the existing Trackio service and enforces the existing write token. This is transport exposure only: Trackio remains backed by private Doris and artifact storage, and neither backend is exposed to RunPod or moved into R2 as part of the OCI registry work. LAN and cloud jobs should receive the same canonical Trackio URL; split routing may keep LAN traffic local provided both paths enforce the same application contract.

Finally run the real automatic-fallback scenario: make both LAN and cloud eligible, occupy or otherwise make LAN GPU capacity unavailable using a bounded qualification workload, submit a second canary, and prove dstack selects cloud without changing the image digest or resubmitting a different Posttrain run. Release the bounded LAN workload and prove the next eligible canary returns to LAN placement.

Record cold and warm pull duration, bytes served by registry ingress, bytes served by R2, BuildKit-to-R2 upload bytes, R2 Class A/B operation deltas, dstack provisioning time, time to first user process, and deprovision time. Compare the cold cloud measurement against the OCI descriptor sum so missing or proxied traffic is visible.

Only after all gates pass should configuration enable cloud fallback for real jobs. Retain `registry.lan`, the old repository references, and rollback automation through a defined compatibility window. Update `../ai-infra/README.md`, the registry/buildkit/worker runbooks, the dstack deployment consumer page, and the framework plan outcome. Do not call the system complete while the R2 proof is synthetic or the cloud job has not actually run.

## Concrete Steps

At every stopping point, re-read repository status and update `Progress`, `Surprises & Discoveries`, and `Decision Log` before continuing.

From `/home/hammad/projects/rl`, establish current framework state without changing it:

    git status --short --branch
    git rev-parse HEAD
    uv run pytest packages/execution-buildkit/tests apps/job-builder/tests packages/execution-dstack/tests -q
    uv run lint-imports

From `/home/hammad/projects/dstack`, establish the maintained fork baseline:

    git status --short --branch
    git rev-parse HEAD
    uv run pytest src/tests/_internal/server/services/test_docker.py \
      src/tests/_internal/server/background/pipeline_tasks/test_running_jobs.py -q
    go test ./runner/internal/shim/...

Adapt the exact Go package selector if the module root requires running from `../dstack/runner`; record the working command in this plan rather than silently omitting the test.

From `/home/hammad/projects/ai-infra`, validate authored infrastructure before applying it:

    git status --short --branch
    git rev-parse HEAD
    ./scripts/preflight
    ./scripts/plan
    uv run ansible-playbook -i ansible/inventory/generated.yml \
      ansible/playbooks/qualify-cloud-registry.yml --check

After a reviewed plan, apply only the additive R2 registry resources and run the focused qualification:

    ./scripts/configure-cloud-registry plan
    ./scripts/configure-cloud-registry apply
    ./scripts/qualify-cloud-registry

The script names above are required outputs of this plan; implementation must make their plan/apply semantics and exact targets explicit. They must not recreate the existing LAN registry or run registry-wide garbage collection.

Once focused tests pass in each repository, run the normal framework validation ladder from `/home/hammad/projects/rl`:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Run ai-infra's full verification only after the focused cloud-registry and cloud-worker gates pass:

    cd /home/hammad/projects/ai-infra
    ./scripts/verify

Execute the remaining production milestones one at a time. From `/home/hammad/projects/rl`, Milestone A uses:

    uv run pytest apps/runtime/tests packages/runtime-images/tests -q
    uv run ruff check apps/runtime packages/runtime-images
    uv run pyright apps/runtime
    uv run posttrain-release images publish \
      --registry registry.lan/carbonteq \
      --default-prefix registry.lan/carbonteq \
      --framework-version 0.3.21 \
      --receipt-root .posttrain/state/release-receipts \
      --repository-root . \
      --variant online-rl-verl-py313 \
      --base-image registry.lan/carbonteq/posttrain-base@sha256:55987a566f5fa7b3fd8af219b2a2dc51315219e2ca82f597f6c940bc7332d595 \
      --no-force-compression

Before accepting that publication, inspect both manifests and expect the exact base prefix and prior veRL prefix to match; a changed inherited descriptor is a failure even if the container runs. Record the new digest, unique layer bytes, local CUDA result, RunPod CUDA result, Pod id, cost, and confirmed deletion.

From `/home/hammad/projects/dstack`, Milestones B and D through G use focused tests first. Create the named test modules when their production code is added:

    uv run pytest \
      src/tests/_internal/server/background/pipeline_tasks/test_image_readiness.py \
      src/tests/_internal/server/background/pipeline_tasks/test_provider_observations.py \
      src/tests/_internal/server/background/pipeline_tasks/test_lifecycle_outbox.py \
      src/tests/_internal/server/background/pipeline_tasks/test_run_storage.py \
      src/tests/_internal/server/services/test_inventory.py -q
    uv run pytest src/tests/_internal/core/backends/runpod -q
    uv run ruff check src/dstack src/tests
    uv run pyright src/dstack
    git diff --check

Run Go validation from `/home/hammad/projects/dstack/runner`. If Go 1.25 is not installed on the host, use the pinned container form so formatting and tests are not silently skipped:

    docker run --rm --user "$(id -u):$(id -g)" \
      -e GOCACHE=/tmp/go-cache -e GOPATH=/tmp/go \
      -v /home/hammad/projects/dstack:/src -w /src/runner golang:1.25 \
      sh -c '/usr/local/go/bin/gofmt -w internal/runner && /usr/local/go/bin/go test ./internal/runner/...'

From `/home/hammad/projects/trackio`, Milestone C uses:

    uv run pytest \
      tests/unit/test_artifact_storage.py \
      tests/unit/test_artifact_migration.py \
      tests/unit/test_artifact_upload_cleanup_cli.py \
      tests/unit/test_artifact_server.py -q
    uv run ruff check trackio tests
    uv run pyright trackio
    git diff --check

Use ai-infra wrappers for every stateful qualification. Milestone implementation must add these plan/apply/qualification commands rather than relying on chat-only shell history:

    ./scripts/qualify-runpod-runtime plan|apply|verify|cleanup
    ./scripts/qualify-cloud-admission plan|apply|verify|cleanup
    ./scripts/qualify-trackio-cloud-artifacts plan|apply|verify|cleanup
    ./scripts/qualify-runpod-spot-recovery plan|apply|verify|cleanup
    ./scripts/qualify-cloud-security plan|apply|verify
    ./scripts/promote-cloud-execution plan|apply|rollback

Every `plan` is read-only and names exact provider resources plus a maximum estimated cost. Every `apply` writes a mode-protected intent and receipt before creating external state. Every `cleanup` targets only ids retained by that intent, confirms absence, and is safe to retry. The security and promotion commands remain unavailable until Milestones A through H have complete receipts.

## Validation and Acceptance

The implementation is accepted only when all of the following behavior is demonstrated.

The R2 registry compatibility gate uses the pinned production registry image and a private R2 Standard bucket. Its synthetic multipart-sized layer obtains a manifest through Distribution and layer bytes through signed R2 redirects. Interrupted pulls resume, digest verification passes, repeated pushes reuse blobs, and safe deletion/offline garbage collection works. The later mirror and cloud gates exercise the real multi-gigabyte image graph. No credential or signed query string appears in retained evidence.

The mirror gate proves every base, job-kind, and canary actual-job source manifest has an exact byte-identical destination manifest. The destination uses the same config and ordered layer digest sequence. Pushing a manifest alone does not claim external readiness; the durable mirror receipt does. Retrying repairs only missing work, and the framework-facing Job Builder remains independent of that status.

The DNS gate proves the canonical hostname resolves to the LAN registry from the dstack server, builder, and managed LAN workers and to the public R2-backed registry from an external probe. Both endpoints return the exact same manifest digest and support a clean Docker pull by that digest. LAN TLS verifies through the existing managed Caddy root under the canonical registry host; external TLS verifies through the default system trust store. `registry.lan` continues to work during the compatibility window.

The LAN execution gate proves a dstack run constrained to `local-gpu-workers` completes with the canonical image digest and does not transfer image-layer bytes from R2. Existing warm-cache behavior remains intact.

The cloud execution gate proves a fresh dstack-provisioned GPU machine pulls the same canonical digest, follows R2 redirects without possessing R2 credentials, passes a CUDA smoke workload, sends required logs/evidence, reaches a terminal state, and is deprovisioned. Cloud provider terminal state alone is not framework completion; retained-evidence reconciliation must also pass.

The storage-policy gate first runs the same RunPod canary with `run_storage`
omitted and proves the provider request contains no implicit volume, preserving
the current behavior. It then enables ai-infra's spot-only `per_run` policy,
proves an on-demand job still creates no implicit volume, and proves one logical
spot run creates exactly one network volume in the selected data center before
its first pod, exposes `/workspace` inside every attempt, and never starts user
code if storage creation or attachment fails. A forced spot
interruption must create a new attempt attached to the same provider volume.
Success and cancellation must both finish with provider-confirmed pod and volume
absence plus a retained cleanup receipt. A run placed on `local-gpu-workers`
during the same configuration must create no RunPod volume.

The checkpoint-durability gate configures Trackio with a dedicated bucket on
the existing self-hosted RustFS service and a worker-reachable presign endpoint,
distinct from the OCI R2 registry. It first runs ai-infra's existing RustFS S3
round-trip qualification, then publishes a multipart-sized checkpoint from a
RunPod worker, verifies that the bytes bypass the Trackio process and arrive in
RustFS, confirms size and SHA-256 before the artifact becomes the durable
recovery pointer, and restores it into a fresh volume. Killing the uploader
mid-transfer must leave the preceding verified checkpoint authoritative and
allow safe retry without a false committed artifact. The gate finishes by
proving the RustFS backup and restore path retains the same object digest.

The automatic-fallback gate proves one unchanged Posttrain run can be placed on cloud when LAN capacity is unavailable, without a second image build, a second image identity, manual registry selection, or a Posttrain provider retry after user code starts. When LAN capacity returns, dstack may place subsequent work on LAN according to configured offer policy.

The security gate proves the public registry denies anonymous pulls and all mutation methods; cloud workers receive no R2 access key; project clients cannot select replica destinations; protected configuration is mode restricted; and logs redact registry authorization, R2 credentials, and presigned query strings.

## Idempotence and Recovery

Bucket and registry provisioning must be declarative and repeatable. Re-running Ansible must report no changes after convergence. Never recreate the R2 bucket to update registry configuration. Never delete the existing filesystem registry during this plan.

Image replication is content addressed. On retry, inspect the destination manifest and blobs, transfer only missing bytes, and rewrite a replica receipt atomically only after verification. An incomplete upload may be aborted; a verified manifest must not be deleted as generic retry cleanup.

If the R2 registry fails qualification, remove only the disposable qualification prefix or bucket after recording evidence. Leave production DNS and dstack configuration unchanged. If public ingress fails after rollout, disable the cloud backend or cloud-fallback target and continue LAN execution through the retained compatibility path. Do not repoint canonical DNS to an unverified endpoint.

If a cloud run is accepted but its response is lost, recover through the existing deterministic dstack run identity and Posttrain reconciliation. Do not submit a second execution attempt merely to repair registry evidence.

If Trackio loses a response after issuing multipart URLs, recover by upload ID and immutable artifact intent. Abort expired or abandoned RustFS multipart uploads through a bounded reconciler, but never advance the durable checkpoint pointer until Trackio has verified the completed object's size and SHA-256. Retrying an upload must preserve the preceding verified checkpoint.

If dstack restarts after committing a lifecycle transition, its transactional outbox resumes every unacknowledged delivery with the same event ID. Receivers deduplicate that ID. A dead-lettered optional hook is visible in inventory; a required finalizer keeps terminal cleanup pending only until its recorded deadline, after which policy determines the explicit failed-finalization state.

If a Pod disappears or a volume operation response is lost, reconcile the exact provider IDs from durable attempt and run-storage ownership rows before issuing another mutation. Never create a replacement volume while ownership of the prior volume is unknown, never attach two writers, and never delete a volume that lacks the logical-run ownership record. Provider absence, not a successful API request alone, completes cleanup.

Failure in the final security milestone leaves all production cloud eligibility disabled. Revoke the affected credential or external edge, retain non-secret functional evidence, and rerun the entire security gate after correction; do not repeat expensive functional canaries unless the correction changes their runtime path.

Rollback means disabling cloud eligibility, restoring the prior project registry prefix where necessary, and continuing to use `registry.lan`. R2 objects remain intact until a separately reviewed retention or purge operation proves they have no surviving image owners. Registry garbage collection remains infrastructure-owned and is never invoked by ordinary project cache cleanup.

## Artifacts and Notes

Retain safe, compact evidence under ignored ai-infra state. A suggested structure is:

    .state/artifacts/cloud-registry/
      compatibility.json
      mirror.json
      lan-pull.json
      cloud-pull.json
      automatic-fallback.json

    .state/artifacts/cloud-production/
      cuda-runtime.json
      actual-job-readiness.json
      trackio-direct-upload.json
      provider-interruption.json
      same-volume-recovery.json
      cross-region-recovery.json
      lifecycle-outbox.json
      functional-matrix.json
      security.json
      promotion.json

Each receipt should include timestamp, source and destination hostnames, manifest digest, ordered descriptor digest summary, compressed byte counts, safe HTTP status/redirect host, dstack logical-run and attempt IDs, provider Pod and volume IDs, selected backend/data center, worker hostname when safe, artifact digest, state transitions, pull/provision/recovery/cleanup durations, bounded cost, and the immutable component revisions used. It must exclude credentials, authorization headers, signed URLs and query strings, project source contents, checkpoint bodies, and user data.

The disposable DNS proof produced the following safe evidence before cleaning up all test containers and networks:

    canonical digest: sha256:e94e7d9ead2a5efeb21a9913051e39730fe37c43e26108c088a548346d7cd3b4
    LAN answer/catalog: 172.22.0.2, ["lan-only/marker", "shared/canary"]
    cloud answer/catalog: 172.23.0.2, ["cloud-only/marker", "shared/canary"]
    Docker result: both isolated daemons downloaded the canonical digest and recorded the same RepoDigest

Official behavior used by this plan should be refreshed before rollout:

- Cloudflare R2 pricing: `https://developers.cloudflare.com/r2/pricing/`.
- Cloudflare R2 S3 compatibility and region `auto`: `https://developers.cloudflare.com/r2/api/s3/api/`.
- Cloudflare R2 token scopes: `https://developers.cloudflare.com/r2/api/tokens/`.
- CNCF Distribution S3 driver, `regionendpoint`, `forcepathstyle`, and redirect behavior: `https://distribution.github.io/distribution/storage-drivers/s3/`.

## Interfaces and Dependencies

The remaining production work adds the following contracts. Names are intentionally provider-neutral in dstack and backend-neutral in Posttrain; RunPod, R2, RustFS, Trackio, and CarbonTeq automation appear only in their owning adapters or deployment configuration.

The veRL child image carries a read-only declaration at `/opt/posttrain/runtime/cuda-compat.json`:

    {
      "schema_version": 1,
      "runtime_api_version": 13000,
      "compat_path": "/usr/local/cuda-13.0/compat",
      "payload_digest": "sha256:<64 hex>"
    }

`posttrain-runtime` reads this declaration before importing execution backends. Its internal selector returns `native`, `compat`, or a typed preflight failure. The only operator input is `POSTTRAIN_CUDA_COMPAT_MODE=auto|off|force`, with `auto` as the image default; the re-exec guard is private runtime state and is never part of a job or provider contract.

In dstack, an optional backend-level `ImageReadinessPolicy` contains `kind`, `endpoint`, `credential_ref`, `timeout`, `poll_interval`, and `required_digest`. Its resolved attempt snapshot contains image repository, manifest digest, policy revision, state, first/last observation times, safe error code, and verified receipt identity. The first adapter uses authenticated HTTP, but the core pipeline understands only the stable `waiting`, `ready`, `timed_out`, and `failed` result. Absence of the policy preserves current behavior. Readiness is evaluated after placement and before any provider resource mutation.

Add a typed `ProviderLifecycleObservation` to dstack's compute boundary with provider resource ID, `present`, desired/observed state, `interruptible`, safe reason code, observed timestamp, and freshness. Persist `RunAttempt` separately from the logical run. Provider adapters may report `unknown` or stale; they may not translate an API outage or runner disconnect into provider absence.

Lifecycle automation uses two durable tables: a versioned event row committed with the owning state transition, and one delivery row per configured target. The executor interfaces are `SignedWebhookDelivery` and `FixedArgvDelivery`; neither accepts an arbitrary shell fragment. Event payloads contain stable framework-neutral IDs and safe state facts. They do not carry secrets, image credentials, checkpoint bodies, Trackio payloads, or site-specific playbook names.

Run-scoped storage uses the `RunStoragePolicy` and durable ownership record described below. The logical run, not an attempt or hook, owns the volume. A writer lease binds at most one live attempt. Cleanup becomes a reconciled state machine ending only after provider-confirmed absence and a durable receipt.

Trackio remains the only artifact-publication API. Its deployment selects the existing self-hosted RustFS service through Trackio's provider-neutral S3 backend, with a private server endpoint, separate public presign endpoint, dedicated bucket/prefix, Trackio-only key ID and secret, multipart thresholds, URL lifetime, capacity policy, and backup retention. Workers receive a Trackio write token and short-lived part URLs only. Posttrain continues to publish immutable `ArtifactRef` values after Trackio verification; neither dstack nor the OCI mirror participates in artifact identity or upload. Cloudflare R2 credentials and buckets are never configured in Trackio.

Inventory is a read model, not a new owner. `GET /api/project/{project_name}/inventory` and `dstack inventory` compose backend configuration, retained capacity, provider candidates, active allocations/attempts, run storage/cleanup, and lifecycle deliveries from their authoritative records. Every external observation includes source, timestamp, and freshness, and project authorization is applied before composition.

The external R2 registry uses the existing pinned Distribution 3 image unless Milestone 1 proves it incompatible. Its server-owned storage settings include R2 endpoint, region `auto`, bucket, root prefix, access key, secret key, Signature V4, path style, TLS verification, multipart sizing, and redirect enablement. Exact environment-variable spellings must be verified against that pinned release.

In `../ai-infra/services/registry-publication-controller/`, retain the current filesystem path for compatibility but identify the deployed component and documentation as the **registry mirror**. Its stable recovery/readiness HTTP surface is:

    POST /v1/publications
    Authorization: Bearer <server-owned token>
    {"repository": "carbonteq/posttrain-base", "digest": "sha256:<64 hex>"}

    GET /v1/publications/<percent-encoded-repository>/<digest>
    Authorization: Bearer <server-owned token>

The response includes repository, digest, state, attempts, safe error code, and verified timestamp. It excludes registry passwords, R2 values, authorization headers, signed URLs, and raw subprocess output. The process accepts fixed source/target registry authorities and regctl configuration only through protected deployment settings. Use Python's standard library for the HTTP/SQLite controller and the pinned regctl binary for OCI operations so the service has no runtime package-index dependency.

In `apps/job-builder`, keep the public job-builder v1 request and `JobPublicationImage` response unchanged. The builder receives only a server-owned canonical repository prefix. It has no controller URL, mirror token, R2 credential, destination selection, or replica-waiting interface.

In `../ai-infra`, keep all supplied secret values in the existing ignored, mode-`0600` file `.state/secrets/vars.yml` and render them only into mode-`0600` service files through Ansible tasks marked `no_log: true`. The real R2 proof requires `cloud_registry_r2_access_key_id` and `cloud_registry_r2_secret_access_key`, created by Cloudflare as an S3-compatible token scoped to Object Read and Write for only the dedicated private bucket. The full public route additionally requires `cloud_registry_tunnel_token` or an equivalently protected tunnel credential. Generate `cloud_registry_pull_username` and a random `cloud_registry_pull_password` locally; these are OCI credentials, not Cloudflare credentials. Keep the bucket name, explicit R2 endpoint, canonical hostname, private push hostname, and root prefix as non-secret deployment variables unless local policy chooses to protect them. Do not add secret defaults to tracked inventory, pass secrets on command lines, or print them during qualification.

No Cloudflare DNS-edit token is required by the selected design. Public DNS can be bound to the managed Tunnel, while the internal route uses the existing Caddy private CA. If DNS records themselves are later automated through the Cloudflare API, introduce a separate zone-scoped token with only Zone Read and DNS Edit; never reuse the bucket token or tunnel credential.

Add Ansible validation that rejects placeholder hostnames, a public bucket, missing external TLS, disabled redirect, or a canonical hostname still ending in `.lan`. Validate `.state/secrets/vars.yml` ownership and mode before reading it, and fail safely if either R2 credential is absent rather than falling back to another S3 key.

`packages/execution-dstack` remains unchanged and continues submitting one
`RuntimeImageRef`; mandatory storage is a server/backend deployment policy, not
a Posttrain or project field. The maintained dstack fork first changes
`apply_server_docker_defaults` and its focused tests so exact-host server
defaults authenticate an already-qualified canonical image. Its later lifecycle
change adds the generic policy and ownership interfaces below. Do not add image
mirror or R2 vocabulary. Update `../dstack/CARBONTEQ_FORK.md` and
`docs/tooling/dstack/README.md` in each logical fork change, publish the fork
commit first, and deploy that immutable commit through ai-infra before describing
it as reproducible.

In `../dstack/src/dstack/_internal/core/models/volumes.py`, define a
provider-neutral `RunStoragePolicy` with `mode`, `required`, `size_gb`,
`required_for`, `mount_path`, `retain_across_attempts`, `cleanup`, and
`finalization_timeout`. `mode` initially accepts only `per_run`; the field is
optional on a provider backend and absence is semantically identical to the
current release. In the RunPod backend model, expose
`run_storage: Optional[RunStoragePolicy] = None`. The durable resolved record
must include the logical run ID, backend type, selected region, provider volume
ID, policy snapshot, state, cleanup deadline, and cleanup receipt ID.

Apply the policy in the provisioning pipeline after an offer/backend/region is
selected and before `RunpodCompute.run_job()` creates a pod. Introduce a generic
compute capability that resolves, creates, observes, and deletes run-scoped
storage; an unsupported provider with an absent policy behaves unchanged, while
an unsupported provider configured with a required policy fails configuration
validation. Retries load the existing logical-run ownership record instead of
calling create again. The RunPod adapter continues passing one
`network_volume_id` to its existing pod API path.

In `../ai-infra/ansible/roles/control/templates/dstack-config.yml.j2`, render the
authenticated RunPod backend and mandatory spot policy from protected
`runpod_api_key` plus non-secret policy defaults. Keep the credential only in
`.state/secrets/vars.yml`, mark rendering/application tasks `no_log: true`, and
validate that `required: true` cannot be combined with an off/empty mode or an
invalid mount path. The initial site policy applies to spot attempts, uses 100
GB mounted at `/workspace`, retains it across attempts, deletes it after
logical-run terminal state, and has a 15-minute finalization deadline.

The first real cloud provider is an external dependency and must be selected and configured before Milestone 4 can complete. Store its credentials only in dstack's protected server configuration. The provider must support the selected GPU, container image pulls from a private OCI registry, and automatic instance termination. Provider choice does not alter the R2 or image-identity contracts in this plan.

## Revision Note

2026-08-29: Created the plan after inspecting the current framework, maintained dstack fork, ai-infra deployment, live registry image sizes, and current R2/Distribution documentation. The plan chooses registry replication plus split-horizon DNS and explicitly avoids a dstack-specific registry-routing feature because one canonical digest can remain valid across both locations. A same-day revision added the narrow exact-host default-credential fix after source inspection proved dstack currently skips server defaults for fully qualified images.

2026-08-29: Re-evaluated the DNS premise with two disposable registries and then two isolated Docker daemons. Both tests proved that different DNS answers can select different registry storage while preserving one canonical digest. Revised TLS handling so managed LAN systems use the existing Caddy private root and external systems use Cloudflare edge TLS, removing an unnecessary DNS-edit credential and custom Caddy build. Documented the exact R2, Tunnel, and OCI credential boundaries and expanded the production DNS gate to cover the actual dstack server, builder, LAN workers, external resolver, and Docker pull path.

2026-08-29: Completed Milestone 1 with a reusable ai-infra qualification harness and real private R2 credentials. Recorded the method-bound redirect discovery, made the supplied endpoint authoritative, added stopped-service garbage collection and exact object-absence verification, and corrected stale job-builder branch state. Production services and DNS remain unchanged pending the canonical hostname and Tunnel credential required by Milestone 2.

2026-08-29: Moved replication and retention responsibility from a framework `JobImagePublisher` decorator into an infrastructure-owned registry publication controller. Distribution notifications remain an advisory trigger because their queues are in memory; an authenticated explicit reconcile API, SQLite work queue, pinned regctl copier, and verified receipt now form the cloud-readiness barrier. Trackio and all non-OCI artifacts remain outside this rollout.

2026-08-30: Corrected the integration after review: Job Builder must not call or understand a publication controller. The LAN registry now owns the normal manifest-push trigger, the SQLite worker owns copying and verification, and the explicit API remains only for infrastructure recovery and release seeding. Managed LAN provisioning installs the canonical host override and private CA; external cloud provisioning deliberately omits both and therefore selects the public R2 route. The plan now records the slow-mirror/cloud-admission race as a remaining infrastructure qualification gate instead of hiding it inside Job Builder.

2026-08-30: Expanded the RunPod milestone after lifecycle review. The maintained dstack fork will own provider-observed spot interruption, run-scoped network volumes, a durable typed hook outbox, bounded cleanup, and truthful inventory. ai-infra may execute allow-listed Ansible playbooks or fixed commands behind the generic hook contract, but hooks do not own provider resources or terminal truth. ADR 0017 and the infrastructure lifecycle architecture record the durable decision and acceptance tests.

2026-08-30: Made mandatory spot storage an opt-in provider-backend policy after
confirming current dstack settings cannot supply it. The policy is absent by
default, is resolved only after backend/data-center placement, creates one
volume per logical run, forbids ephemeral fallback when required, and leaves LAN
and unconfigured providers unchanged. Added the exact ai-infra configuration,
ownership interface, conflict behavior, compatibility tests, and real storage
qualification gate.

2026-08-30: Refined storage after checkpoint-path review. Trackio publication
is the durable provider-neutral tier for every workload; RunPod network volumes
are mandatory only for spot attempts and provide fast same-data-center retry.
Added exclusive-writer fencing, verified cross-data-center restore, a dedicated
Trackio S3/presign qualification gate, and an explicit boundary from the OCI R2
registry. The live Trackio deployment remains on its local artifact backend, so
this design is documented but not yet operationally qualified.

2026-08-30: Reordered the remaining work after the real R2-backed A100 canary.
Separated stateless cloud execution from spot-training production readiness,
recorded diagnostic redaction and credential-boundary validation as release blockers,
made guarded CUDA compatibility activation explicit, and placed Trackio ingress
and registry-readiness admission before billed automatic fallback. Live
production inspection still shows one busy LAN worker, so source publication may
proceed but control-plane promotion must wait for the unchanged two-worker gate.

2026-08-30: Expanded the remaining work into Production Milestones A through J
and moved security hardening to the final engineering milestone at the user's
direction. Functional work and bounded qualification may proceed first on
isolated projects, but production fallback remains disabled until diagnostic
redaction publication, credential-boundary audit, and negative tests
all pass.

2026-08-30: Corrected the Trackio storage decision after user review. Cloudflare
R2 remains exclusive to the OCI registry. Trackio uses the already deployed,
self-hosted RustFS S3 service through its existing direct multipart contract,
with a private server endpoint and a separately qualified worker-reachable
presign endpoint. This keeps artifact storage CarbonTeq-owned and makes the
bandwidth consequence explicit: checkpoint bodies bypass Trackio but traverse
CarbonTeq's public ingress.

2026-08-30: Narrowed the final security milestone at the user's direction. The
working credentials remain unchanged. The gate retains diagnostic redaction,
protected-state and least-privilege validation, public-edge restrictions,
secret scanning, and negative tests.
