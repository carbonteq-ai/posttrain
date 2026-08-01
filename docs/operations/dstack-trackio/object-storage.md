# Trackio artifact registry and backing storage

Status: simplest first path selected; implementation alignment and large-model
qualification remain pending.

## Decision

Use Trackio as the internal research artifact registry. Training and evaluation
code logs final model directories, retained checkpoints, native traces, and
other outputs as versioned Trackio artifacts. Later jobs consume a pinned
`TrackioArtifactRef`. Project developers register that version as a catalog
`ModelVariant` (`artifact.kind: trackio`) and bind it on the next work package
— see
[consumer-setup §9](../../consumer-setup.md#9-pass-one-jobs-model-into-the-next).

Trackio owns:

- logical artifact names and types;
- immutable versions such as `v0` and `v1`;
- moving aliases such as `latest`, `candidate`, and `qualified`;
- content manifests and digests;
- producer and consumer run lineage; and
- the project-facing artifact browser and download API.

Trackio's configured bucket or dataset owns the artifact bytes. That storage is
an implementation detail of the Trackio server, not a second framework
registry and not a job binding.

The first execution path is:

```text
immutable runtime image       -> private OCI registry
Verifiers environment + data  -> CarbonTeq code/package at an exact commit
base model                    -> immutable HubModelRef, cached on each worker
result model + checkpoints    -> versioned Trackio artifacts
metrics, traces, and lineage  -> the same Trackio run
artifact bytes                -> Trackio-managed private HF Storage Bucket
```

Do not provision a separate `ai-storage` VM or expose object-store credentials
to jobs for the first integration.

## Dataset boundary

Package the selected task data with the Verifiers environment or project data
adapter when it is bounded enough to ship and cache as code. The source commit,
locked package resolution, and OCI image digest then identify the exact bytes
used by a remote job.

This is particularly natural for a Verifiers environment: environment code,
rubric/verifier code, task fixtures, and the bounded task population are one
published CarbonTeq environment revision. A job capsule records that revision;
it does not copy the dataset again.

Do not silently apply this rule to an unbounded corpus. If a future dataset
makes source checkout, image construction, worker transfer, or cache pressure
materially expensive, promote it to a versioned Trackio dataset artifact or an
external immutable dataset reference without changing job semantics.

## Base-model boundary

Do not put foundation-model weights in Git or the OCI image. A base model stays
an immutable `HubModelRef` with a full commit SHA. The runner materializes it
before GPU admission and may reuse a bounded worker-local Hugging Face cache.

If the model is private, the runner injects a scoped model-read token. This
token is independent of Trackio's artifact-backing-store credential and must
not appear in the capsule, image, receipt, or Trackio metadata.

## Result-model boundary

At the end of training, log a complete loadable model directory:

```python
logged = trackio.log_artifact(
    output_dir,
    name="ambient-agent-2b-sft",
    type="model",
    aliases=["candidate"],
)
```

The artifact must contain the files needed to load the exact result, including
weights or adapter weights, model and tokenizer configuration, tokenizer
assets, renderer/chat-template facts, and safe generation defaults. Run
configuration, base-model revision, dataset/environment revision, backend
revision, and training selection belong in artifact metadata and run lineage.

The framework persists the returned identity as:

```text
TrackioArtifactRef(
    project="<trackio-project>",
    name="ambient-agent-2b-sft",
    version="vN",
    alias="candidate",
)
```

Downstream jobs consume the immutable `vN`. They may resolve an alias during
planning, but the resolved version is snapshotted before execution. `latest`
never means qualified.

After evaluation passes the recorded gate, move `qualified` to that exact
version. Publishing the model to a Hugging Face model repository remains an
optional distribution/export step, not part of every experiment.

## Generalized-capability checkpoints

Reusable intermediate capability gains use the same registry:

- log only retention-policy checkpoints, not every trainer save;
- give each logical capability branch a stable artifact name;
- retain the exact producing run, parent artifact, training selection, and
  evaluation evidence;
- evaluate and pin a version before another work package consumes it; and
- use aliases only as planning conveniences, never as the immutable run input.

This preserves useful checkpoints without keeping unbounded recovery data.
Trainer recovery checkpoints that are not selected as reusable evidence expire
under the run's cleanup policy.

## Runtime bindings

GPU jobs receive only Trackio connectivity for artifact publication:

- `POSTTRAIN_TRACKIO_SERVER_URL`;
- `POSTTRAIN_TRACKIO_PROJECT`; and
- `TRACKIO_WRITE_TOKEN`, injected as a dstack secret or a local mode-`0600`
  environment file.

The Trackio server alone receives backing-store configuration. The current
deployment uses its durable server volume. A later deployment may select a
private Hugging Face Storage Bucket or another Trackio-owned backend without
changing job bindings.

Workers never receive `TRACKIO_BUCKET_ID`, bucket credentials, S3 endpoints, or
storage-administrator credentials. They upload through Trackio's artifact API.
The job capsule contains required output roles and immutable input references,
not endpoints, credentials, datasets, model bytes, or an output-manifest
location.

## Existing implementation seam

The CarbonTeq Trackio fork already provides the required registry behavior:

- `trackio.log_artifact` logs files or directories;
- `trackio.use_artifact` resolves `latest`, `vN`, or a custom alias and records
  the consuming run;
- artifact content is SHA-256 addressed and de-duplicated;
- versions, aliases, manifests, producer/consumer links, and downloads are
  stored by `SQLiteStorage`;
- remote uploads flow through the Trackio server; and
- the dashboard has an Artifacts view with manifests and lineage.

The framework defines both `TrackioArtifactRef` and provider-neutral
`StoredArtifactRef`. Work-package composition intentionally normalizes the
former into `StoredArtifactRef(provider="trackio")` before it crosses the
tracking contract. This is not a mismatch: it keeps train, eval, serve, and work
contracts independent of Trackio while the Trackio adapter retains exact
version and lineage behavior.

The current implementation candidate adds two missing pieces:

- `WorkPackageJobResult.published_artifacts` exposes provider-committed exact
  versions instead of returning only temporary local output paths; and
- CarbonTeq Trackio `0.31.5.post2` adds resumable artifact-blob transport
  behind the existing `log_artifact` API.

The resumable transport:

1. check whether the SHA-256 blob already exists;
2. initiate an upload for the digest and expected size;
3. stream bounded numbered chunks to server-side temporary storage;
4. resume from acknowledged chunks after interruption;
5. finalize only after size and SHA-256 verification;
6. atomically expose the content-addressed blob; and
7. commit the artifact version only after every manifest blob is durable.

This remains a Trackio API. Workers do not receive direct bucket credentials.
The candidate has passed a separate-process 512 MiB round trip with 8 MiB
chunks and approximately 15 MiB server RSS growth. Publication, deployment,
and remote qualification remain pending.

## Finalization and cleanup

A run is not artifact-complete merely because training exited successfully:

1. Trackio commits the named artifact version and manifest.
2. The client receives the resolved version and manifest digest.
3. The framework finalizer validates the returned exact version and manifest
   digest while the local workspace still exists.
4. Required files and digests match.
5. The run records the produced-artifact edge.
6. Only then may cleanup remove the worker-local result directory.

Interrupted uploads remain retryable. Incomplete Trackio upload sessions are
reported in dry-run mode and can be expired without touching completed blobs.
Cleanup may remove unselected recovery checkpoints after their bounded retry
window, but it must not remove the only recoverable copy of a required result.

## First integration gate

Before a real training run:

1. package a synthetic Verifiers environment with a small bundled dataset and
   prove both runners execute the identical revision;
2. materialize one pinned base model through the worker cache;
3. upload a synthetic multi-file model directory from each runner;
4. verify Trackio versioning, content de-duplication, aliases, manifest
   digests, and producer lineage;
5. materialize the pinned Trackio version on the other runner and perform a
   model-load smoke test;
6. record that version as an input and verify consumer lineage;
7. interrupt and retry a large upload without creating a false successful
   version;
8. restart Trackio and prove the exact artifact remains downloadable;
9. verify a worker cannot access the backing bucket directly; and
10. exercise cleanup for retained result artifacts and expired recovery
    checkpoints.

The first limited GPU run should qualify actual adapter and merged/full-model
sizes after resumable streaming is implemented. Small-file success is not
sufficient evidence that multi-gigabyte model upload, timeout, restart, and
download behavior are safe.

## What Trackio does not replace

Trackio does not replace:

- the OCI registry used to distribute runtime images;
- Git/package publication for Verifiers environment code;
- the foundation-model source and worker cache;
- a public or partner-facing model hub when polished distribution is required;
  or
- bounded backups of Trackio control data and its backing bucket.

A future local object store can replace the Trackio server's bucket backend
without exposing storage-provider types to framework jobs.

## Primary source

- [CarbonTeq Trackio artifact documentation](https://github.com/carbonteq-ai/trackio/blob/main/docs/source/artifacts.md)
