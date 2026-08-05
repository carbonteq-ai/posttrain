# Qualify the Gemma 4 dense matrix and release it in 0.3.2

This is a living execution plan. It follows `docs/templates/PLAN.md`; keep
`Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes &
Retrospective` current after every milestone. It supersedes the narrower
12B-only scope in `docs/plan/gemma4-unified-12b-support.md`, while retaining
that document as historical implementation evidence.

## Purpose / Big Picture

After this work, a posttrain user can resolve and qualify four immutable Gemma
4 dense instruction checkpoints through the same model-family contract:
E2B, E4B, 12B Unified, and 31B. The two smaller and two larger checkpoints
share a family renderer and catalog shape, but each has its own exact model and
paired MTP assistant provenance. The user-visible proof is a successful,
tracked, text-only vLLM generation for each size and a successful TRL GRPO
qualification with MTP enabled for 12B. The release notes for 0.3.2 will link
to those run IDs, image digests, and cleanup receipt rather than claiming
support from static YAML alone.

This plan does not claim multimodal training, production throughput, or a
general 32K/256K benchmark. “Smoke” means one bounded request that proves the
model loads and returns non-empty text. “MTP” means speculative decoding using
a separately pinned assistant checkpoint; it is a serving/rollout execution
optimization, not an additional training objective. “Qualification” means a
real dstack run on the RTX PRO target whose runtime, model identity, metrics,
and terminal state are retained in Trackio.

## Progress

- [x] (2026-08-05) Audited the existing Gemma 12B implementation, the
  `carbonteq-ai/vllm` fork, release constraints, and the dirty worktree.
- [x] (2026-08-05) Added the reusable Gemma dense matrix and exact model and
  paired-assistant revisions for E2B, E4B, 12B Unified, and 31B.
- [x] (2026-08-05) Corrected 12B modality facts to text/image/audio; executable
  profiles remain deliberately text-only.
- [x] (2026-08-05) Added family-level TRL assistant materialization and MTP
  validation without adding a job-level Gemma branch.
- [x] (2026-08-05) Re-qualified the 12B TRL MTP path with `max_length=1024`:
  run `gemma4-trl-mtp-qualification-4` completed two traces, one optimizer
  step, reward mean 1, completion truncation 0, and MTP acceptance 0.937888.
- [x] (2026-08-05) Ran the E2B serving smoke
  `gemma4-e2b-serve-smoke-1`: healthy/model-available=1, two output tokens,
  35 ms TTFT, and 44 ms request latency.
- [x] (2026-08-05) Ran the E4B serving smoke
  `gemma4-e4b-serve-smoke-1`: healthy/model-available=1, two output tokens,
  45 ms TTFT, and 56 ms request latency.
- [x] (2026-08-05) Ran the 31B serving smoke
  `gemma4-31b-serve-smoke-1`: healthy/model-available=1, two output tokens,
  75 ms TTFT, and 115 ms request latency on the 96 GiB target.
- [x] (2026-08-05) Finished the final repository validation after the cleanup
  planner edits: 1046 tests passed and 19 skipped; Ruff, Pyright, import
  contracts, and `git diff --check` are green.
- [ ] Reconcile the old 12B plan and add the complete run/image evidence here.
- [x] (2026-08-05) Previewed the five temporary/diagnostic runs with exact
  dependency closure and applied only the diagnostic 12B run's remaining local
  state. The failed run's dstack workspace, actual-job OCI manifest, and
  Trackio record were already completed by the earlier journaled apply; the
  recovery preview contained one local execution target and no blockers. The
  accepted 12B MTP run and all three serving smoke runs remain retained for
  release evidence. Shared images, caches, and unrelated projects were not
  selected.
- [x] (2026-08-05) Added the 0.3.2 Gemma qualification gate to
  `docs/release-and-consumption.md`; publish remains blocked until the
  candidate workflow, cleanup receipt, and release notes are complete.
- [x] (2026-08-05) Focused validation passes: 330 tests passed with 11 skips,
  Ruff passed, all 8 import contracts passed, targeted Pyright reported zero
  diagnostics, and `git diff --check` passed. The full repository suite then
  passed with 1044 tests and 19 skips; full Ruff, Pyright, import contracts,
  and whitespace checks are green.
- [x] (2026-08-05) Generated no-change purge previews for the two 12B MTP
  attempts and the E2B/E4B/31B smoke runs. Each preview had one terminal
  provider record, one digest-pinned actual-job manifest, one Trackio run, and
  no blockers. Applying the diagnostic `...-3` preview first hit a transient
  dstack no-capacity cleanup failure; after the provider adapter began waiting
  for an exact healthy worker, the retry completed provider, OCI, and Trackio
  cleanup. A second immutable preview then removed only the allowed local
  execution directory. The accepted evidence runs were not applied.
- [x] (2026-08-05) Catalog validation reports 70 framework and 80 project
  entries; all four new work packages report complete static composition
  validation.
- [x] (2026-08-05) Advanced the authored release authority in
  `release/manifest.toml` from 0.3.1 to 0.3.2. The release consistency check
  reports 24 publishable packages, 107 internal pins, and a valid published
  image manifest; the release test module passes 28 tests.
- [ ] Update the release workflow inputs and 0.3.2 release notes; publish only
  after the pre-release qualification gate passes.

## Surprises & Discoveries

- Observation: The vLLM fork already contains Gemma 4 tower, Unified, and MTP
  model implementations and registry entries.
  Evidence: `../vllm` at `7817d845727af570352622dc8d58f2d43c76d89d` contains
  `gemma4.py`, `gemma4_unified.py`, `gemma4_mtp.py`, and parser tests. No fork
  patch is justified unless a smoke produces a reproducible backend defect.

- Observation: E2B, E4B, and 31B all load on the same single RTX PRO 6000 96 GiB
  target with the existing vLLM smoke job.
  Evidence: the three Trackio runs above each recorded health, model discovery,
  and a non-empty request metric; 31B was slower to cold-load but did not fail.

- Observation: 12B MTP initially appeared unsuccessful because the training
  `max_length` was 256 while the qualified completion budget was 512.
  Evidence: run 3 completed but every completion was truncated and reward was
  absent. Raising `max_length` to 1024 produced run 4 with two complete traces,
  reward 1, and zero truncation. This is a configuration boundary, not an MTP
  or CUDA defect.

- Observation: A prior longer MTP retry failed in Verifiers setup before any
  vLLM or CUDA work.
  Evidence: `c8cf775c-9ebe-415a-9be0-68c5c3bed66e` ended with
  `HarnessError: harness setup timed out` and zero trainable branches. It must
  remain a retry caveat, not be presented as a model failure.

- Observation: The runtime image installs Python dependencies while building
  the job image and the smoke container performs no package installation.
  Evidence: the E4B and 31B build logs show hashed wheel/code installation in
  BuildKit layers; runtime qualification only validates the prepared manifest.

- Observation: The current CLI exposes exact-run purge, but not a separate
  `posttrain job purge` command.
  Evidence: `posttrain run purge --help` produces an immutable cross-plane
  preview, while `posttrain job purge --help` returns “No such command”. The
  cleanup milestone therefore uses exact run IDs and records this as a DX gap;
  it must not emulate job cleanup with a broad project purge.

- Observation: Applying the exact purge for the failed 12B retry stopped at the
  provider plane because the dstack cleanup task returned `status=failed`.
  Evidence: purge plan `purge-5d637e103ade950b` journaled only a started and
  failed provider action. The failure was `FAILED_TO_START_DUE_TO_NO_CAPACITY`
  while the RTX PRO worker was still occupied; it was not a model or memory
  failure. The dstack bridge now waits for the exact healthy worker and uses a
  deterministic retry name for a terminal failed cleanup task.

- Observation: The partial apply exposed a cross-plane path ownership bug.
  Evidence: the submission's dstack workspace is a remote `/var/lib/...`
  path, but the local purge executor correctly rejects paths outside local
  state roots. The catalog now always targets the local execution receipt
  directory for dstack and only includes a workspace for the local provider.
  A new preview derives already-completed planes from the prior immutable
  journal, so recovery is resumable without mutating or weakening the old
  plan.

## Decision Log

- Decision: Support E2B, E4B, 12B Unified, and 31B as four entries in one
  `gemma4` family, rather than adding a new architecture axis or separate job
  implementations. Rationale: the immutable checkpoint owns concrete
  Transformers/vLLM architecture; the shared renderer and adapters are the
  reusable framework boundary. Date/Author: 2026-08-05 / Codex.

- Decision: Keep model capability facts accurate, but qualify text-only smoke
  profiles with bounded context and output settings. Rationale: upstream
  modalities and native context are not evidence that this release exercises
  multimodal inputs or a full-context benchmark. Date/Author: 2026-08-05 / Codex.

- Decision: Treat the 12B run with `max_length=1024` as the accepted TRL MTP
  proof and retain the earlier truncated and setup-timeout runs as diagnostics.
  Rationale: the accepted run has complete traces and reward; deleting failed
  evidence would obscure the configuration lesson. Date/Author: 2026-08-05 / Codex.

- Decision: Do not change the vLLM sibling fork unless a model-specific smoke
  fails with a reproducible backend error. Rationale: all four dense model
  paths currently pass the real serving contract. Date/Author: 2026-08-05 / Codex.

## Outcomes & Retrospective

Current outcome is an implementation and qualification candidate, not a
published release. The matrix, TRL MTP path, model-specific serving smokes,
and diagnostic cleanup are complete. The remaining work is release hygiene:
run the final validation ladder, update the 0.3.2 release workflow inputs and
notes, and pass the documented pre-release workflow before any publication or
tag claim.

## Context and Orientation

The canonical product contracts are in `docs/post-training/README.md` and
`docs/post-training/01-workflow.md` through `06-observation-and-lineage.md`.
`packages/common` owns framework-neutral model identities and renderer
contracts. `packages/catalog` owns reusable base model selections. `packages/train`
owns TRL loading, LoRA, and the paired-assistant MTP adapter. `packages/serve`
owns provider-neutral inference bindings and its private vLLM adapter.
`apps/lab` composes the reference catalog overlay, qualification gates, and
declarative work packages. Trackio is the read-only evidence source for the
Observatory view; dstack is the remote execution provider.

The exact model revisions used by this plan are:

    google/gemma-4-E2B-it        3e22461f65e89153144f8adb70e3b8c2cc9845a7
    google/gemma-4-E4B-it        ee0ef6023621cff504d758262d4e04895a5af4a2
    google/gemma-4-12B-it        707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    google/gemma-4-31B-it        842da3794eaa0b77d5f08bae87a17459d91ff475

Their paired assistant revisions are pinned in
`packages/common/src/posttrain/common/variants/gemma4.py` and the catalog;
never replace them with a branch, tag, or `latest` reference.

## Plan of Work

First, finish the model matrix implementation already started in
`packages/common/src/posttrain/common/variants/gemma4.py`, its registry exports,
`packages/catalog/src/posttrain/catalog/base/models.yaml`, and the associated
common/catalog tests. Verify that each selection retains exact upstream model
type, architecture provenance, parameter count, modalities, context facts, and
assistant provenance. Keep the execution bindings in
`apps/lab/.posttrain/catalog/gemma4-unified-qualification.yaml` text-only and
model-specific; do not put target or job policy in the base catalog.

Second, keep the TRL MTP implementation in
`packages/train/src/posttrain/train/backends/trl/common.py` family-aware and
reusable. It must validate `method=mtp`, a positive token count, a full
assistant revision, and model/assistant provenance before constructing the
trainer. It may materialize an exact model snapshot into the existing HF cache,
but it must never install or upgrade Python packages at runtime. Add or update
unit tests in `packages/train/tests/test_trl_common.py` for valid and invalid
assistant mappings and for unchanged non-Gemma behavior.

Third, retain the three serving work packages under
`apps/lab/.posttrain/work_packages/` and the corresponding candidate gates in
`apps/lab/src/posttrain_lab/qualification/gates.toml`. If a smoke fails, inspect
the Trackio run, vLLM server log artifact, provider state, and GPU evidence
before changing utilization, context, or MTP settings. Fix the owning layer,
rerun only the affected model, and record the reason in this plan.

Fourth, reconcile `docs/plan/gemma4-unified-12b-support.md` so it no longer
states that E2B/E4B/31B are unsupported, claims video for 12B, or treats the
old 128-token run as the final MTP proof. Point readers to this matrix plan and
preserve historical run IDs as evidence. Then update release documentation
(`docs/release-and-consumption.md`, any applicable `docs/plan/` release plan,
and `.github/workflows/`) to state that 0.3.2 requires pre-release
qualification, immutable image digests, and explicit cleanup before publish.

Finally, build a cleanup preview from exact run IDs and image digests. Use the
existing `posttrain run purge` command (there is no `posttrain job purge`
surface yet) only after checking its help and generated dependency closure.
Preserve Trackio run
evidence and accepted release artifacts; remove only temporary provider runs,
local workspaces, and temporary job image references that the preview names.
Record the resulting receipt and verify no nonterminal run, shared base image,
machine cache, or unrelated project is selected.

## Concrete Steps

Run from `/home/hammad/projects/rl`:

    uv run pytest packages/common/tests/test_model_variants.py packages/common/tests/test_contracts.py packages/catalog/tests apps/lab/tests/test_catalog.py apps/lab/tests/test_qualification_gates.py packages/train/tests/test_trl_common.py -q
    uv run ruff check packages/common packages/catalog packages/train apps/lab
    uv run lint-imports
    git diff --check

Use the local Observatory to inspect each retained run. Locate it with:

    curl -skG --data-urlencode run_id=gemma4-e4b-serve-smoke-1 http://127.0.0.1:7861/api/v1/runs/locate

Then query the returned `run_key` at `/api/v1/runs/<run_key>/view?mode=job`
and `/metric-series`. Required serving evidence is health=1,
model_available=1, and a positive output-token metric. Required 12B MTP
evidence is two complete traces, reward present, zero truncation, and positive
speculative draft/accepted counters.

Before cleanup, run the command help and a dry-run/preview using only the exact
temporary IDs. Do not use a project-wide purge. Save the preview and final
receipt under the existing ignored local state or release evidence location
only if the release process explicitly requires a committed summary; never
commit tokens, signed URLs, raw logs containing secrets, or `.posttrain/state/`.

## Validation and Acceptance

Acceptance requires the focused tests above and the normal repository ladder:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

If full Pyright again scans ignored nested environments or hangs, run targeted
Pyright for every changed production module, record the workspace limitation in
this plan, and leave the full gate for CI rather than claiming it passed.

The four serving runs must be terminal `succeeded` on
`targets/carbonteq-rtx-pro-6000-96gb`; each must expose the expected endpoint
model and positive output tokens in Trackio. The 12B TRL run must retain the
exact assistant revision, complete both rollouts without truncation, emit
reward, and record MTP acceptance and KV-cache metrics. A release reviewer
must be able to navigate from 0.3.2 notes to the run IDs and immutable job
image digests.

## Idempotence and Recovery

All work packages use deterministic run IDs; retrying a failed run requires a
new suffixed ID after inspecting the old evidence. Reusing a successful ID is
not a new qualification. BuildKit and HF caches are reusable and must not be
purged as part of smoke cleanup. If a provider run remains nonterminal, stop
and reconcile it before retrying. If a cleanup preview contains an unexpected
dependency, abort and narrow the selector; never broaden it to “make the noise
go away.”

## Artifacts and Notes

Retain these evidence identifiers in the release review:

    gemma4-trl-mtp-qualification-4
    gemma4-e2b-serve-smoke-1
    gemma4-e4b-serve-smoke-1
    gemma4-31b-serve-smoke-1

The accepted smoke job images recorded in run source metadata are:

    E2B  registry.lan/carbonteq/posttrain-job@sha256:563f8e8e2d6b34b2421061ed9653df67004b559012dc3955a04dbed86571d33b
    E4B  registry.lan/carbonteq/posttrain-job@sha256:1cfb817a6037da83e0ad1f0edf047b9120a4dd159d823a0e39841ae1b58357ff
    31B  registry.lan/carbonteq/posttrain-job@sha256:420c5c4590ae3e5afcaeb2c2bb9661f36efdbdc5c13caa13e27e76c998931802

The job-kind image was the existing
`registry.lan/carbonteq/posttrain-kind-serve@sha256:3b49e756fc8eed3fe39b09ddb7f7aa6c3429be2ccea3683b914dfc0ebf371613`.
Re-read these values from the run views during release preparation; never
replace them with mutable tags.

## Interfaces and Dependencies

The implementation must preserve these interfaces:

    posttrain.common.variants.GEMMA_4_E2B_IT
    posttrain.common.variants.GEMMA_4_E4B_IT
    posttrain.common.variants.GEMMA_4_12B_IT
    posttrain.common.variants.GEMMA_4_31B_IT
    posttrain.train.backends.trl.common.vllm_rollout_options(...)
    posttrain_tracking_trackio.TrackioDataSource.metric_series(...)

The runtime dependency versions remain the catalog/lockfile selections already
qualified, including vLLM 0.25.1 and TRL 1.8.0. The external Verifiers
environment remains pinned by its immutable commit. Any dependency or sibling
fork change requires its own tests, immutable pin update, and an explicit
decision-log entry before a release claim.

### Change note

2026-08-05: Expanded the original 12B plan into a Gemma 4 dense-matrix and
0.3.2 release plan after E2B, E4B, and 31B serving smokes passed and the
non-truncated 12B TRL MTP run was accepted. This change records the corrected
12B modality facts, the max-length truncation fix, the exact evidence IDs, and
the required cleanup/release gates.

2026-08-05: Added journal-aware purge recovery after a real dstack cleanup
race exposed that remote worker paths must never be treated as local state.
The provider bridge now waits for an exact healthy worker, and a new immutable
recovery preview can apply only unfinished planes. The diagnostic run's local
state was removed and accepted evidence was retained; release publication
remains pending the candidate workflow and release notes.
