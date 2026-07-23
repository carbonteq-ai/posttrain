# Add reproducible SFT validation and observability

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

After this work, a developer can train on any canonical supervised dataset even
when its publisher provides only one split. The framework can deterministically
reserve validation records and an untouched future-use population, prove that
the three populations do not overlap, and reproduce the same assignments from
the recorded source revision and partition plan. A later milestone will feed
the validation population into `train.sft` and project learning, stability,
rendered-data quality, token efficiency, and system evidence in Observatory.
Generated-response task evaluation is explicitly outside this plan.

The frozen post-training baseline does not need an amendment. It already states
that a dataset selection records an exact split and subset, and that filtering
and transformation provenance belongs to the dataset selection.

## Progress

- [x] (2026-07-22 08:02Z) Confirmed that the canonical dataset selection already owns exact train, validation, and held-out subsets; no baseline amendment is required.
- [x] (2026-07-22 08:07Z) Added a deterministic, provider-neutral supervised dataset partition contract under `packages/data`.
- [x] (2026-07-22 08:07Z) Proved stable assignments, disjoint populations, group leakage prevention, and failure behavior with 20 passing package-specific tests; Ruff and Pyright pass for `packages/data`.
- [x] (2026-07-22 08:12Z) Added the pinned Smol-SmolTalk source and deterministic train, validation, and reserve manifest materialization.
- [x] (2026-07-22 08:18Z) Extended `SFTRequest` and the TRL adapter with bounded loss-only validation inside the `train.sft` run.
- [x] (2026-07-22 08:22Z) Emitted rendered-data, validation, stability, token-efficiency, and timing observations without adding generated task evaluation.
- [x] (2026-07-22 08:31Z) Extended the SFT Observatory definition and focused frontend view while preserving missing evidence honestly.
- [x] (2026-07-22 08:47Z) Ran focused package validation and a successful two-step QLoRA run, and proved live container reads over real Trackio data without an Observatory restart.
- [x] (2026-07-22 08:50Z) Completed Codex in-app Browser verification over the real SFT run and saved the final Observatory screenshot.
- [x] (2026-07-22 09:08Z) Added schema-owned metric definitions and accessible information popovers across curated job and system metric surfaces; verified the real SFT projection in the Codex in-app Browser.

## Surprises & Discoveries

- Observation: The existing `GSM8KSupervisedSource` selects a contiguous range
  from the published train split and cannot produce a validation or reserve
  manifest.
  Evidence: `apps/lab/src/posttrain_lab/data/gsm8k.py` stores only `offset` and
  `count` and always loads `split="train"`.

- Observation: The canonical `SupervisedDataset` already requires stable unique
  example IDs, so deterministic partitioning can be provider-neutral and does
  not need Hugging Face row indices.
  Evidence: `packages/data/src/posttrain/data/models.py` rejects empty datasets
  and duplicate example IDs.

- Observation: Copying a Trackio SQLite database and its WAL at Observatory
  startup makes the product a stale export. Directly mounting the directory
  read-only is also insufficient for WAL mode because SQLite may need to create
  or update the shared-memory coordination sidecar.
  Evidence: The first rebuilt container returned an unavailable source with
  `sqlite3.OperationalError: unable to open database file` while configuring a
  read-only connection; the startup-copy implementation required a restart to
  observe later commits.

## Decision Log

- Decision: Published train/test splits are optional inputs, not a framework
  requirement. When a source lacks a suitable validation population, partition
  its canonical supervised snapshot into `train`, `validation`, and `reserve`.
  Rationale: This makes validation reproducible across providers and prevents a
  dataset adapter from silently choosing records.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: The first partition mechanism uses deterministic hash allocation by
  stable example or group identity, with optional metadata-based grouping and
  stratification. It records explicit example IDs and a manifest digest.
  Rationale: Hash allocation is independent of input ordering, works for local
  and remote sources, and prevents related examples from leaking across splits
  when a group key is supplied.
  Date/Author: 2026-07-22 / Codex.

- Decision: `reserve` means an untouched population whose eventual use is not
  assumed. It is not automatically named `test`.
  Rationale: Repeated validation is model-selection evidence; a future task
  evaluation or final loss check may need a population that was never used for
  tuning.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Teacher-forced validation loss is part of `train.sft`; generated
  behavioral evaluation jobs remain out of scope.
  Rationale: Validation loss is a bounded forward-only check of held-out SFT
  targets and does not require Verifiers or model generation.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Containerized Observatory reads Trackio through Trackio's live HTTP
  read API. A private Trackio service owns the SQLite/WAL volume; the
  Observatory service has no evidence mount and no storage credentials.
  Rationale: This preserves live reads and a strict ownership boundary, avoids
  snapshots, and matches the provider-neutral reader shape already used for
  W&B. The Trackio fork now provides `trackio.Api(server_url=...)` with the same
  logical read model as the local API.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Human-readable metric meaning belongs to the versioned job
  telemetry definition, not to frontend-only copy. Every declared job metric
  must provide a definition, reading guidance, optional unit, and any important
  comparability caveat; system metric cards carry the same metadata.
  Rationale: UI, HTTP/Python, and MCP consumers should explain identical metric
  semantics, while generic unregistered metrics remain explicitly uninterpreted.
  Date/Author: 2026-07-22 / Codex and user.

## Outcomes & Retrospective

The data partition, trainer validation, SFT telemetry, and live deployment
milestones are complete. The successful QLoRA run
`acbffebc-1f81-481a-87eb-1da36364131c` records training and validation loss,
rendered-data quality, stability, and efficiency evidence. A canonical live
probe run `2626ef5e-353e-4227-89d5-1cc9a4cb0cc7` appeared in Observatory after
both services had started; container IDs and start times remained unchanged.
Codex in-app Browser verification confirmed the curated SFT projection, its
three chart modes, rendered-data utilization, validation lineage, and produced
artifacts. No application console errors were observed. The screenshot is
stored at
`docs/design/observatory/implementation/sft-validated-live-20260722.png`.
The focused SFT surface now exposes keyboard-accessible information buttons for
summary, selected-step, rendered-data, and system metrics. The popover separates
the definition, how to read the signal, caveats, unit, and canonical metric name.

## Context and Orientation

`packages/data/src/posttrain/data/models.py` owns canonical trainer-neutral data
records. A `SupervisedDataset` is an immutable in-memory snapshot with stable
example IDs. `packages/data/src/posttrain/data/adapters/huggingface.py`
normalizes common external row formats into that contract; provider-specific
split logic must not leak into training.

`packages/train/src/posttrain/train/requests.py` currently gives `SFTRequest`
one supervised `data` source. `packages/train/src/posttrain/train/backends/trl/sft.py`
renders only that source and passes only `train_dataset` to TRL. The callback in
`packages/train/src/posttrain/train/backends/trl/common.py` forwards numeric TRL
logs but does not normalize validation names, time optimizer steps, or record
gradient clipping indicators.

`apps/observatory/src/posttrain_observatory/telemetry.py` is the single source of
truth for job-aware summaries and charts. The React application consumes those
projections; it must not duplicate an independent SFT metric list beyond labels
and formatting.

A partition manifest is the immutable record of which stable example IDs were
assigned to each population. A group key is an example metadata field such as
`conversation_id`; all examples with the same value must remain together. A
stratification key is metadata such as `source`; hashing is performed within
that label so each sufficiently large source receives comparable proportions.

## Plan of Work

Create `packages/data/src/posttrain/data/partitioning.py`. Define a frozen
`SupervisedPartitionPlan` with stable identity, revision, seed, validation
fraction, reserve fraction, and optional group and stratification metadata keys.
Define `SupervisedPartitionManifest` with explicit train, validation, and
reserve example IDs plus a deterministic SHA-256 digest. Define
`PartitionedSupervisedDataset` and `partition_supervised_dataset`. Put groups
in deterministic hash order and examples within each group in stable ID order
so the derived training order does not depend on provider row order. Derive
dataset metadata that cites the source
snapshot and manifest, reject missing grouping metadata, reject ambiguous
groups spanning strata, and reject non-zero allocations that produce an empty
population.

Export those interfaces from `posttrain.data`. Add
`packages/data/tests/test_partitioning.py` for order-independent assignment and
derived ordering, exact disjoint coverage, group cohesion,
stratification behavior, manifest stability, and clear invalid-plan errors.

Then add a pinned Hugging Face source in `apps/lab` for the selected permissive
conversational SFT dataset. Normalize external `messages` rows through
`supervised_from_huggingface`, retain `source` metadata, apply the partition
plan, and materialize JSON manifests as durable data evidence. Do not use the
publisher's test split for repeated tuning; keep it as separately recorded
future-use data.

Extend `SFTRequest` with an optional validation source and add an explicit
validation schedule to SFT settings. Render both populations with the same
tokenizer, renderer, maximum length, and supervision mask. Pass
`eval_dataset` to TRL with loss-only step scheduling. Normalize TRL `eval_*`
logs to `train/validation/*` so they cannot be mistaken for separate
`eval.domain` jobs.

Add rendered profile evidence before training: counts, sequence-length
quantiles, truncated examples and tokens, supervised tokens, supervision-token
ratio, and padding or packing utilization. Add step time, processed and
supervised token throughput, token entropy, gradient norm, and a per-step
gradient-clipped indicator. Keep per-rank and profiler-only diagnostics
conditional.

Finally revise the SFT telemetry definition and focused React view around
learning, stability, data utilization, and efficiency. The existing System
metrics and lineage sections remain separate. Missing validation or device
evidence must appear as unavailable rather than zero.

## Concrete Steps

Run all commands from `/home/hammad/projects/rl`.

During partition development:

    uv run pytest packages/data/tests/test_partitioning.py
    uv run ruff check packages/data
    uv run pyright packages/data

During SFT integration:

    uv run pytest packages/train/tests apps/lab/tests/test_gsm8k_jobs.py
    uv run pytest apps/observatory/tests
    npm --prefix apps/observatory/frontend test
    npm --prefix apps/observatory/frontend run build

Before completion:

    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The real GPU and network run command must be added after the selected dataset
source and SFT settings have stable CLI/catalog names. It must use an immutable
dataset revision, a unique tracking run ID, and a bounded validation schedule.

## Validation and Acceptance

Partition acceptance requires that the same source revision, plan revision, and
seed always yield the same manifest and derived ordering even when input row
order changes. Every
source example appears exactly once across train, validation, and reserve. No
group appears in more than one population. A non-zero requested population may
not silently become empty. Derived dataset metadata cites the source dataset,
partition name, plan, and manifest digest.

SFT acceptance requires a real tracked run whose train and validation losses
share logical steps, whose validation is explicitly labeled as a bounded probe
or full pass, and whose data profile proves the populations and supervision
mask. Observatory must show real provider data and preserve missing evidence.
No generated-response evaluation is required for this plan.

## Idempotence and Recovery

Partitioning is pure and performs no writes, so rerunning it is safe. Manifest
materialization must write inside the run workspace and may be retried only in a
fresh run directory, consistent with existing training behavior. External
dataset credentials, if needed, are read from the environment and never stored.
The working tree contains extensive user changes; edits must stay within the
named package, app, and plan files and must not revert unrelated work.

## Artifacts and Notes

Record the chosen dataset repository, immutable revision, license, source split,
manifest digests, exact population counts, real tracking run IDs, and final
Observatory screenshots here. Do not store access tokens or downloaded dataset
contents in Git.

Partition milestone evidence from 2026-07-22:

    uv run pytest packages/data/tests -q
    20 passed in 0.10s

    uv run ruff check packages/data
    All checks passed!

    uv run pyright packages/data
    0 errors, 0 warnings, 0 informations

Live Observatory evidence from 2026-07-22:

    uv run pytest packages/tracking-trackio/tests apps/observatory/tests -q
    24 passed

    Observatory mounts: none
    Trackio storage mount: /home/hammad/.cache/huggingface/trackio -> /evidence/trackio rw=true
    probe run: 2626ef5e-353e-4227-89d5-1cc9a4cb0cc7, succeeded
    observatory container identity/start time unchanged: yes
    trackio container identity/start time unchanged: yes

Final focused validation:

    uv run pytest packages/data/tests packages/train/tests apps/lab/tests packages/tracking-trackio/tests apps/observatory/tests -q
    122 passed

    uv run ruff check packages/data packages/train apps/lab packages/tracking-trackio apps/observatory
    All checks passed!

    uv run pyright packages/data packages/train apps/lab packages/tracking-trackio apps/observatory
    0 errors, 0 warnings, 0 informations

    uv run lint-imports
    8 contracts kept, 0 broken

    npm --prefix apps/observatory/frontend test
    2 passed

    npm --prefix apps/observatory/frontend run build
    built successfully; Vite reported one non-blocking 503 kB chunk warning

Metric-help increment:

    uv run pytest apps/observatory/tests -q
    16 passed

    npm --prefix apps/observatory/frontend test -- --run
    3 passed

    npm --prefix apps/observatory/frontend run build
    built successfully; Vite reported the existing non-blocking 503 kB chunk warning

## Interfaces and Dependencies

`packages/data/src/posttrain/data/partitioning.py` must expose interfaces
equivalent to:

    @dataclass(frozen=True, slots=True)
    class SupervisedPartitionPlan:
        id: str
        revision: str
        seed: int
        validation_fraction: float
        reserve_fraction: float
        group_by: str | None = None
        stratify_by: str | None = None

    @dataclass(frozen=True, slots=True)
    class SupervisedPartitionManifest:
        source_id: str
        source_revision: str
        plan_id: str
        plan_revision: str
        train_ids: tuple[str, ...]
        validation_ids: tuple[str, ...]
        reserve_ids: tuple[str, ...]

        @property
        def digest(self) -> str: ...

    def partition_supervised_dataset(
        dataset: SupervisedDataset,
        plan: SupervisedPartitionPlan,
    ) -> PartitionedSupervisedDataset: ...

No new dependency is required for partitioning; use the Python standard
library. Hugging Face loading continues through the existing optional
`datasets` dependency. Training remains behind the existing TRL adapter, and
Observatory remains provider-neutral.

Revision note (2026-07-22 08:02Z): Created this plan after validation loss was
accepted as an in-run SFT diagnostic and the user required a mechanism to
reserve records even when an upstream dataset does not publish validation or
test splits.

Revision note (2026-07-22 08:07Z): Completed the provider-neutral partition
contract and recorded focused validation evidence. The remaining milestones
connect a pinned source, SFT validation, telemetry, and Observatory.
