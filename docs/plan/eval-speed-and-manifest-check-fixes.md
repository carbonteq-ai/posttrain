# Faster AutomationBench evaluations and correct manifest check verdicts

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

A 100-episode AutomationBench evaluation (20 held-out luna-v2 tasks, 5 attempts each, 64K context) took about 23 minutes on the dstack workstation although the model spent only about 3 minutes generating. Its goal and harm components were also wrong for several tasks, because some manifest checks could never be met by a correct episode. After this work an evaluation of a LoRA adapter or the base model finishes in about 13 minutes (about 8 minutes once the remaining trace bloat is removed), and goal rates credit work the benchmark itself accepts. Someone comparing training checkpoints can trust the components and rerun a full comparison within an hour.

This work does not change the frozen product baseline in `docs/post-training/01` to `06`. It changes how one evaluation job caches compiler output and stores its native result, and it corrects task manifests in the external `carbonteq-ai/verifiers-environments` repository.

## Progress

- [x] (2026-10-08 06:10Z) Stopped the running evaluation chain and cancelled `luna2-heldout-64k20t-t05-cont100-final-20261008`.
- [x] (2026-10-08 06:40Z) Measured the evaluation timeline: 2.5 min local packing, 2.3 min vLLM start (111 s engine init of which 83 s CUDA graph capture), 3 min rollouts, about 15 min after rollouts (about 10 min uploading a 1.18 GB `traces.jsonl` at about 2 MB/s, about 3 min trace synchronization, about 80 s hashing).
- [x] (2026-10-08 07:00Z) Audited eval manifests (seven check bugs in five tasks) and fixed them in verifiers-environments `3f0a9e4` on branch `codex/manifest-check-fixes-20261008`; eval pin `66e79e07` (branch `codex/eval-3f0a9e4-verifiers-b358d355`) adds Verifiers `b358d355`.
- [x] (2026-10-08 07:05Z) Framework commit `3566d5ac`: compiler caches under the compile-cache mount, gzip `traces.jsonl.gz`, trace synchronization in a worker thread. Commit `d994661d` points the 64K luna-v2 eval environment at `66e79e07`.
- [x] (2026-10-08 07:41Z) Verified on the workstation: first run filled the caches (13.5 min); the next run started warm (engine init 13.5 s, graph capture 1 s, total 12.9 min); packing took 6 s on a pack-cache hit.
- [x] (2026-10-08 08:00Z) Audited 76 training episodes; fixed four more check bugs in verifiers-environments `346406c` (training-only tasks).
- [ ] Rerun all comparison evals on `66e79e07`: old SAMPO cont100 final (done), base (done), v2 final, v2 steps 100, 200, 300 (running, `runs/luna2-64k-evals-e2.sh` in the session scratchpad).
- [ ] Verifiers fork: persist only `complete` assessment batches and reference shared sources/views by digest, so a trace holds each block once (expected 1.2 GB raw to tens of MB; removes most of the remaining 5 minutes after rollouts and the Trackio dead letters over 10 MB).
- [ ] Trackio fork: upload artifact parts in parallel over one connection.
- [ ] Merge the verifiers-environments fixes to main and include them, with the training environment pin, in the next release.

## Surprises & Discoveries

- Observation: the 1.2 GB eval artifact is 92% assessment batches; the conversations are about 3 MB.
  Evidence: one `support.zendesk_hubspot_org_sync` episode held 4,848 batches (1,212 assessment runs x 4 lifecycle states) each repeating the same 25 KB source identity and 22 KB view reference; 232 MB for one episode. gzip -6 compresses it 21x, zstd -3 66x.
- Observation: each assessment appears twice in an episode because the `partial` and `complete` lifecycle batches carry identical assessments with the same `assessment_id`. Counting must keep `complete` batches or dedupe by `assessment_id`; an earlier ad hoc scorer doubled harm counts.
- Observation: the compile-cache mount existed but no tool used it; `/var/lib/posttrain/cache/compile` had been empty since July. The CUDA driver's compute cache (`CUDA_CACHE_PATH`) is what removes the 80 s graph capture; vLLM and Triton caches alone saved only about 30 s.
- Observation: LoRA adapters share compiled kernels. Triton keys LoRA kernels by shape (rank, hidden size, dtype, adapter slots), not weights; a different rank or base model adds cache entries rather than reusing wrong ones.
- Observation: the base model already loads from the workstation's Hugging Face cache (0.26 s) and only the 67 MB adapter is downloaded per job; nothing large is packed locally. The 1.4 GB local pack cache is old environment checkouts.

## Decision Log

- Decision: compress `traces.jsonl` with gzip level 6 (deterministic, `mtime=0`) instead of zstd or a tar of the folder.
  Rationale: zstd is not in the job image and would need a base-image rebuild; gzip is in the standard library and turns a 10-minute upload into about 30-50 s. The other two files are a few KB, so a tar adds nothing, and `gzip.open` lets readers stream episodes. The real size fix belongs in Verifiers.
  Date/Author: 2026-10-08, Claude.
- Decision: route compiler caches through environment variables (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `CUDA_CACHE_PATH`, `CUDA_CACHE_MAXSIZE`) set by `ExecutionRequest.compile_cache_environment` whenever a compile-cache mount exists, not `XDG_CACHE_HOME`.
  Rationale: `XDG_CACHE_HOME` would also move the Hugging Face cache away from its own mount. No separate warm-up job is needed; the first evaluation after a driver, vLLM or model-shape change fills the cache.
  Date/Author: 2026-10-08, Claude.
- Decision: bump the existing `automationbench-manifest-luna2-eval-64k20t-t05-v1` environment revision in place and rerun every compared model.
  Rationale: old artifacts record their environment revision, and all compared rows are regenerated on `66e79e07`, so a duplicate environment, plan, work package and gate would add boilerplate without improving lineage.
  Date/Author: 2026-10-08, Claude.
- Decision: keep the upstream keyword assertions (for example "received" in `simple.email_zendesk_ack_reply`) unchanged.
  Rationale: they are the upstream benchmark's success criterion and keep comparability with upstream and earlier evaluations; manifest goals deliberately do not judge reply wording, and they show the real failure (no Legal group on the ticket).
  Date/Author: 2026-10-08, Claude.

## Outcomes & Retrospective

Evaluation wall time fell from about 23 to about 13 minutes with warm caches; dropped attempts fell to 0 with the Verifiers transport retry. Eleven manifest check bugs were fixed (seven eval-task, four training-task). The remaining cost is trace bloat after rollouts (about 5 minutes), addressed by the pending Verifiers change. The checks for false negatives should be part of manifest qualification: replay recorded episodes that pass upstream assertions and flag goals that are never met.

## Context and Orientation

An evaluation job is submitted with `posttrain job run <work package> --job evaluate --provider dstack [--model-from-run <training run> [--model-checkpoint-step N]] --run-id <id>` from `apps/lab` of a clean worktree. The CLI packs framework sources into a job image layer (`packages/execution-buildkit`), submits through dstack to the GPU workstation, and the container starts a vLLM server and runs the Verifiers environment `automationbench-v1`. `packages/eval/src/posttrain/eval/api.py` (`evaluate`) runs the backend, publishes the native output folder as a `verifiers-evaluation` artifact and records metrics. `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py` (`_run`) streams `traces.jsonl` episodes to the tracking backend while rollouts run. `apps/cli/src/posttrain_cli/execution_planning.py` (`_mounts`) mounts the host model cache and compile cache; `packages/execution/src/posttrain/execution/contracts.py` (`ExecutionRequest.launch_environment`) builds the container environment.

A manifest is a JSON contract per task under `environments/automationbench_v1/src/automationbench_v1/contracts/tasks/` in verifiers-environments, with a byte-identical draft under `manifest-drafts/tasks/<task>/draft.json` and a digest in `tests/installed_candidate_manifests.json`. Obligations ("goals", direction higher) and guards ("harms", direction lower; `.compliance` is the inverse) are evaluated per episode into assessment batches stored in the trace.

## Plan of Work

Remaining work, in order: finish the evaluation reruns and publish the comparison; change Verifiers (`verifiers/v1/assessment_archive.py`, `verifiers/v1/assessments.py`, trace serialization) to persist one terminal batch per assessment run and reference shared sources and views by digest, keeping `restore_history` able to read old traces; pin it in verifiers-environments and this repository; add parallel part upload to the Trackio fork's `remote_client.py`; release.

## Concrete Steps

From `/home/hammad/projects/worktrees/rl-perf`:

    .venv/bin/python -m pytest -q packages/eval/tests packages/execution/tests apps/lab/tests apps/cli/tests
    .venv/bin/ruff check packages/eval packages/execution
    .venv/bin/lint-imports

From `/home/hammad/projects/worktrees/venv-check-fixes/environments/automationbench_v1`:

    UV_SYSTEM_CERTS=1 ~/.local/share/mise/installs/uv/0.12.3/uv-*/uv run --frozen pytest -q

Expected: framework packages pass; the environment suite reports 4219 passed, 7 skipped.

## Validation and Acceptance

An evaluation submitted after a previous one on the same workstation shows in its `serving-log` artifact `init engine ... took` under 20 s and `Graph capturing finished in 1 secs`; its `verifiers-evaluation` artifact contains `traces.jsonl.gz` and no `traces.jsonl`; and the run finishes within about 13 minutes of submission. Replaying the audited episodes with the fixed manifests credits the attempts that did the work (for example all four hiring-manager summaries in `hr.airtable_recruitment_analytics` attempt 2) and still fails wrong recipients and values.

## Idempotence and Recovery

The compile cache can be deleted at any time (`/var/lib/posttrain/cache/compile` on the workstation); the next job refills it. Reverting `3566d5ac` restores raw traces. Evaluation reruns use new run ids (`...-e2`) and never overwrite earlier evidence.

## Artifacts and Notes

    base (warm)   init engine (profile, create kv cache, warmup model) took 13.45 s (compilation: 5.29 s)
    cont100 (cold) init engine (profile, create kv cache, warmup model) took 111.12 s (compilation: 11.8 s)
    traces.jsonl 1,184,352,248 bytes -> traces.jsonl.gz 108,198,254 bytes (oldcont100-final-e2)

## Interfaces and Dependencies

`posttrain.execution.ExecutionRequest.compile_cache_environment() -> dict[str, str]` returns the cache variables for the first `compile-cache` mount, or an empty mapping. `posttrain.eval.api._compress_traces(output_dir: Path) -> None` replaces `traces.jsonl` with `traces.jsonl.gz`. Readers of `verifiers-evaluation` artifacts must accept either file name.

Revision note (2026-10-08): created after the work was largely done, to record measurements, decisions and remaining steps for a reader without the session history.
