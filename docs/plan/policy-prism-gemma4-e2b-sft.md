# Run and qualify the Policy Prism Gemma 4 E2B SFT experiment

This ExecPlan is a living execution document maintained according to
`docs/templates/PLAN.md`. It is the authority for adding Gemma 4 E2B support,
running the complete Policy Prism SFT experiment, publishing the exact adapter,
and finalizing both sealed domain evaluations. Update `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` whenever execution
advances.

## Purpose / Big Picture

After this work, CarbonTeq can reproduce one rank-32 LoRA SFT of the exact
`google/gemma-4-E2B-it` revision on the existing immutable Policy Prism dataset,
inspect the run through Trackio and PostTrain Observatory, load the exact adapter
from a private CarbonTeq Hugging Face repository, and compare its complete sealed
scope and rule-recovery evidence with other Policy Prism runs in the standard
five-file format.

Policy Prism owns dataset meaning, project bindings, sealed evaluation
environments, and permanent evaluation evidence. PostTrain owns the foundation
model profile, model-specific rendering, LoRA execution, artifact lineage,
managed vLLM serving, tracking, and remote orchestration. The dataset remains a
model-neutral sequence of chat messages and is not rebuilt or republished.

## Progress

- [x] (2026-08-05) Verified clean pushed branches: PostTrain
  `feat/policy-prism-gemma4-e2b-sft` at
  `a02dc43b04fd99ec4dc9e6d0d67a8328777daa2d`; Policy Prism
  `feat/gemma4-e2b-policy-prism-sft` at
  `6caab4dd667855733b126d9433c35baf988e7442`.
- [x] (2026-08-05) Verified Hugging Face, OpenRouter, dstack, Trackio, Docker,
  registry configuration, and both Git remotes without printing credentials.
- [x] (2026-08-05) Resolved E2B revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, exact parameter count
  `5,123,178,051`, architecture, context, and tokenizer/template assets.
- [x] (2026-08-05) Proved the pinned E2B and E4B repositories have identical
  tokenizer, tokenizer-config, and chat-template blobs.
- [x] (2026-08-05) Added the E2B variant, catalog entry, shared small-Gemma
  renderer alias, and model/catalog/loader/LoRA/rendering regressions. Focused
  validation passed: 29 tests, 7 intentional skips, zero focused Pyright
  diagnostics, Ruff, all eight import contracts, and diff checks.
- [x] (2026-08-05) Committed and pushed PostTrain E2B support as
  `7dd0ab370f708c75ebe52efc12994714db508ae1`.
- [x] (2026-08-05) Added, validated, committed, and pushed the Policy Prism E2B
  training binding and complete-run work package. The initial composition was
  `9ca19d0bbc8a20719bb1a58aff47d7518e217e64`; the corrected binding is
  `f28f16734abb7fca35e322bc9ca3abe5936441ee`.
- [x] (2026-08-05) Completed dataset/model/runtime/image/job preflight without a
  GPU smoke. The corrected isolated package is
  `109e29df668ecb5c725c980f40762a4f659827da0d774d7ee149eeeed35a788b`
  and the published job image is
  `sha256:64c09a418f57168d9d57b73c2adc78011cdf462e31d72f9777c5a1a1921a48eb`.
- [x] (2026-08-06) Ran and reconciled the complete 535-step SFT as
  `policy-prism-e2b-sft-r32-v1-r1`. Reconciliation is consistent, tracking
  succeeded, all required roles are retained, final loss is `0.0407577`, and
  validation loss is `0.0632281`.
- [x] (2026-08-06) Registered exact Trackio adapter
  `training-models-gemma4-e2b-it-bf16-sft-lora-adapter:v0`, committed the two
  qualification work packages in Policy Prism as `8ec8e90`, and packaged both
  isolated evaluation images successfully.
- [x] (2026-08-06) Ran scope and recovery sequentially and retained their
  complete scientific evidence. The
  scope attempts `policy-prism-e2b-sft-r32-v1-r1-scope` and
  `policy-prism-e2b-sft-r32-v1-r1-scope-r1` were externally stopped while
  managed vLLM was starting; neither produced a rollout. The second attempt has
  consistent cancelled reconciliation and retained tracking evidence. After a
  verified ten-minute continuous idle interval, unchanged attempt
  `policy-prism-e2b-sft-r32-v1-r1-scope-r2` was submitted as provider run
  `pt-69e72b0cc23543722d69983a` and completed consistently on the intended
  workstation with all 18 traces, zero harness failures, complete trace sync,
  and one genuine model-output truncation. The exact native artifact finalizes
  and validates as a standard Policy Prism run. Recovery was then submitted as
  `policy-prism-e2b-sft-r32-v1-r1-recovery` / provider
  `pt-022d8f871ae87b6f7abdc069`; it completed consistently with all 17 traces,
  zero harness failures, complete trace sync, and three genuine model-output
  truncations.
- [x] (2026-08-06) Published and independently redownloaded the private adapter
  at Hugging Face commit `c9b6e4a457902b99d8c3d9a6721afc0c79f574eb`; all seven
  required files are byte-identical to the staged publication and the
  repository is private.
- [x] (2026-08-06) Finalized and validated both standard five-file Policy Prism
  runs, then committed and pushed them as Policy Prism commit `d409dc3`.
- [x] (2026-08-06) Cleaned only the successful training, `scope-r2`, and recovery
  provider workspaces. Durable run identities and promoted artifacts remain;
  cancelled attempts and all Trackio/evaluation evidence were preserved.

## Surprises & Discoveries

- Observation: the authoritative E4B experiment used `max_length=49,152`, not
  the earlier provisional 65,536 value.
  Evidence: Policy Prism's tracked `sft-one-epoch-v1` settings and completed E4B
  run use 49,152; the longest released example is 46,381 tokens.

- Observation: E2B does not require another renderer implementation.
  Evidence: at the pinned revisions, E2B and E4B use identical
  `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` Git blobs,
  while both report `Gemma4ForConditionalGeneration` and a 131,072 context.

- Observation: existing E2B base evaluation runs are not strictly compatible
  with the current PostTrain-finalized SFT evaluations because their stored
  evaluator implementation digest is older.
  Evidence: the task, prompt, schema, Gold, and calibration inventories match,
  but the base scope evaluator digest is `f5c542...` while current E4B SFT
  evidence uses `6c5ab2...`. Do not claim a strict before/after delta unless the
  resulting compatibility digests match.

- Observation: the branch-wide static/test gates contain unrelated baseline
  failures outside the E2B change.
  Evidence: repository-wide Pyright reports 141 public-export diagnostics in
  CLI, Lab, Eval, Serve, Tracking, and Work code, while the two changed
  production modules report zero diagnostics. Repository-wide pytest reaches
  the pre-existing Observatory discovery test
  `test_success_removes_missing_projects_but_failure_retains_last_snapshot`
  and does not terminate; all E2B-affected suites pass. Repository-wide Ruff
  format also identifies three unchanged files. Do not modify those unrelated
  surfaces in this experiment branch.

- Observation: the first remote attempt failed safely before optimizer step
  zero because the new Policy Prism E2B YAML value contained two literal
  backslashes before `d`, while the proven E4B regex contains one.
  Evidence: `policy-prism-e2b-sft-r32-v1` reconciled consistently as failed with
  `Gemma 4 LoRA target expression matched no modules`. After correcting the
  YAML value, the E2B regex byte-matches E4B's parsed expression and matches 205
  language projections from layer 0 through 34 on a no-weight meta model while
  excluding vision and audio modules.

- Observation: the corrected full run passed the runtime-only target check
  that isolated packaging cannot perform without loading the model.
  Evidence: `policy-prism-e2b-sft-r32-v1-r1` produced finite step-one and
  subsequent metrics in Trackio instead of failing during model preparation.

- Observation: E2B completed the same 535-step workload materially faster than
  E4B while preserving the exact data and optimization configuration.
  Evidence: the corrected run started at `2026-08-05T17:22:28Z`, finished at
  `2026-08-05T20:56:08Z` (about 3 h 34 min), completed validation over all 220
  held-out examples, and retained an approximately 225.5 MB adapter.

- Observation: the complete E2B run retained exactly three promoted artifacts.
  Evidence: adapter manifest/content digests are `d3400565...`/`f58bac19...`,
  the step-535 recovery checkpoint digest is `5ace0b88...`, and the summary
  digest is `adc64530...`; reconciliation reported no missing roles.

- Observation: the first scope qualification attempt did not expose a model,
  Verifiers, Policy Prism, or GPU failure.
  Evidence: dstack recorded `STOPPED_BY_USER` exactly 20 minutes after the job
  entered `RUNNING`; PostTrain retained only the Trackio run placeholder, and
  the evaluation view contains zero expected, included, failed, or truncated
  rollouts. The workstation remained healthy and allocated to vLLM until the
  stop request. This attempt is operational cancellation evidence, not a
  scientific result.

- Observation: a second unchanged scope attempt proved the cancellation source
  is outside this PostTrain project's durable control state.
  Evidence: dstack recorded `STOPPED_BY_USER` for provider run
  `pt-89dce853d7e9a441d6b5ba11` at `2026-08-06T02:55:37+05:00`, but the project
  contains no `cancel-intent.json`, no local controller is running, and the
  attempt reconciled consistently as cancelled with zero rollouts. Thirty-five
  seconds after its GPU block was released, unrelated run
  `ambient-k1-dapo-smoke-50step-rtxpro-20260806-r5` claimed the same host. This
  is shared-workstation scheduling interference rather than an E2B evaluation
  failure.

- Observation: the workstation eventually remained continuously idle for the
  full ten-minute admission gate after the unrelated ambient-agent sequence.
  Evidence: only then was unchanged scope attempt `scope-r2` submitted; dstack
  assigned provider run `pt-69e72b0cc23543722d69983a` to
  `carbonteq-ai-workstation.lan` on its first attempt.

- Observation: stable scope attempt `scope-r2` completed operationally and
  retained all 18 traces, but one case is a real SFT quality regression rather
  than a framework failure. Evidence: case `scope-v2-533ea438bf1c1c89` emitted
  repetitive duplicate rules until the unchanged 16,384-token stage limit,
  returned `finish_reason=length`, skipped its dependent graph stage, and has
  no harness error. The immutable artifact passed Policy Prism preview
  finalization and `validate-runs` with `trace_count=18` and complete semantic
  status.

- Observation: the recovery evaluation exposed a stronger repetitive-output
  regression and therefore ran for about one hour rather than the E4B
  reference's 22 minutes. Evidence: all 17 traces completed with zero harness
  errors, but three rules stages reached the unchanged 65,536-token limit:
  `recall-006-ohio-5122-14-10-complete`,
  `recall-scope-audit-scope-v2-533ea438bf1c1c89`, and
  `recall-scope-audit-scope-v2-5ead6765d258cf42`. Each has
  `finish_reason=length` and complete retained trace evidence.

- Observation: PEFT's adapter configuration records the exact base repository
  but leaves its optional `revision` field null.
  Evidence: the original Trackio and fresh Hugging Face `adapter_config.json`
  files have identical SHA-256
  `de3f11beec8a8630f46e779cf4bc010b0cf929e78aabbde95b1526865ebfb10f`;
  the base revision is instead pinned in the model card and both evaluation
  manifests. Publication verification was corrected without uploading a second
  commit.

- Observation: repeated scope finalization is scientifically stable but not
  byte-deterministic at the final IEEE-754 bit for a small number of span-IoU
  sums. Evidence: native `traces.jsonl`, semantic diagnostics, inventories, and
  validation results are identical, while independent finalizations differed
  only by approximately `1e-16` in `node_span_f1`. The cause is the unsorted set
  intersection iteration in Policy Prism's `span_comparison()` before floating
  weights are accumulated. Recovery happened to be byte-identical across both
  finalizations. The pushed canonical evidence is internally hash-consistent
  and passes `validate-runs`; fixing the iteration order belongs in a separate
  Policy Prism change with an evaluator-digest update, not a silent mutation of
  this completed experiment.

## Decision Log

- Decision: add a separate E2B `ModelVariant` but share the existing pinned
  small-Gemma renderer contract and tokenizer fingerprint.
  Rationale: model topology and immutable weights differ, while tokenizer and
  chat-template behavior are byte-identical. Renaming the existing E4B renderer
  would invalidate stable catalog identities.
  Date/Author: 2026-08-05 / Codex.

- Decision: reuse the exact private dataset revision
  `92ff4cfca942f65214313416ae5f787cd19106e3` without rebuilding or uploading it.
  Rationale: rows are model-neutral and E2B rendering is identical to the
  already-audited E4B rendering.
  Date/Author: 2026-08-05 / Codex.

- Decision: run the complete one-epoch SFT without a GPU smoke.
  Rationale: this is an explicit user constraint. Offline rendering, exact model
  access, catalog validation, job planning, image packaging, and isolated
  container checks remain mandatory.
  Date/Author: 2026-08-05 / User.

- Decision: copy the completed E4B training configuration exactly: 535 steps,
  49,152 tokens, micro-batch one, gradient accumulation eight, rank/alpha/dropout
  32/64/0.05, checkpoint every 100 steps with two retained, and all 220 validation
  examples at the end.
  Rationale: the experiment changes only the selected foundation-model variant.
  Date/Author: 2026-08-05 / User and Codex.

- Decision: submit scope and recovery sequentially.
  Rationale: this site's dstack capacity wait is zero and a simultaneous second
  request can fail unassigned while the named workstation is occupied.
  Date/Author: 2026-08-05 / Codex.

- Decision: publish only the final PEFT adapter, not Gemma base weights or
  periodic checkpoints, to private
  `carbonteq/gemma-4-e2b-policy-prism-scope-sft-lora-v1`.
  Rationale: the adapter is the produced artifact; base weights remain pinned to
  their upstream repository and checkpoints remain Trackio recovery evidence.
  Date/Author: 2026-08-05 / User and Codex.

- Decision: retain the failed first attempt and relaunch the corrected capsule
  under `policy-prism-e2b-sft-r32-v1-r1`.
  Rationale: run IDs are immutable idempotency namespaces; changing project
  configuration requires a new identity, and the failed evidence must not be
  overwritten or misrepresented as training evidence.
  Date/Author: 2026-08-05 / Codex.

- Decision: retain the cancelled scope attempt, retry the unchanged qualified
  capsule with a fresh run ID, and monitor it without any cancel-on-timeout
  behavior.
  Rationale: the provider event timeline proves the process was externally
  stopped before the configured 1,800-second managed-vLLM startup budget, so no
  source or inference change is justified by this attempt. Recovery remains
  gated on 18 included scope traces with zero failures.
  Date/Author: 2026-08-06 / Codex.

- Decision: do not submit another scope run while the external ambient-agent
  workload owns or is actively reclaiming the workstation; require a stable
  idle interval before using a fresh scope run ID.
  Rationale: repeated submission during another project's experiment sequence
  wastes startup time, creates cancelled Trackio placeholders, and cannot
  produce scientific evidence. It must not be "fixed" by changing the already
  qualified E2B capsule.
  Date/Author: 2026-08-06 / Codex.

- Decision: retain the completed `scope-r2` evidence and proceed to recovery
  despite the former zero-truncation qualification check.
  Rationale: all infrastructure, inventory, sync, and artifact-integrity checks
  pass; the one truncation is deterministic model behavior under the same
  16,384-token policy used by the prior evaluations. Rerunning until that
  observed regression disappears would cherry-pick evidence, while increasing
  the budget would make the comparison a different evaluation.
  Date/Author: 2026-08-06 / Codex.

## Outcomes & Retrospective

The complete SFT finished successfully on `carbonteq-ai-workstation.lan` in
about 3 h 34 min. It produced final loss `0.0407577`, held-out validation loss
`0.0632281`, token accuracy `0.988076`, zero truncation, and exact adapter
`training-models-gemma4-e2b-it-bf16-sft-lora-adapter:v0` with artifact digest
`d34005655b8270d5a663d49a799bc19bb5fa101d6ccf3ad4aba8da812d8c73d5` and
content digest
`f58bac190bc25f1f37c6f2c013f74a29fc2fe2404e073fe8e51de5d9d0020028`.
Scope package/image are `c14e1de...`/`a48bcdf...`; recovery package/image are
`3a52b97e...`/`4781468e...`.

The stable scope qualification ran as
`policy-prism-e2b-sft-r32-v1-r1-scope-r2` on provider
`pt-69e72b0cc23543722d69983a` in about 15m57s. It retained 18/18 traces, zero
harness failures, one model-output truncation, complete semantic evidence, and
evaluation artifact digest
`9592d3291271d5bd51b3cd84475a29eaee17a17465d7ffd0847a3dd7d20a1132`.
Its absolute business results include 94.4% operational reliability, 5.6%
fully conformant interpretation, 70.8% expected-rules matched, and 79.1%
Claude-diagnostic predicted-rule support. Its compatibility digest is
`80b32e1429e5e663415bc3d109cb3cfdf6cfd5a9db8235f3f0c061e49c1336e8`.

Recovery ran as `policy-prism-e2b-sft-r32-v1-r1-recovery` on provider
`pt-022d8f871ae87b6f7abdc069` in about 1h00m20s. It retained 17/17 traces, zero
harness failures, three model-output truncations, complete semantic evidence,
and evaluation artifact digest
`e7c09869c0d468b748f38abc1ad48ff9ffba1c3d5e664a2287080de20f0f6dd9`.
Its absolute business results include 76.5% operational reliability, 27.4%
expected-rules found, 0% complete-provision exact recovery, and 20.9%
Claude-diagnostic recovered-rule meaning preservation. Its compatibility
digest is
`4077dd2f98fdd143d38f32acb42ecef4216b8d38c66b16b6f597a762c93534b4`.

The exact seven-file adapter is private at
`carbonteq/gemma-4-e2b-policy-prism-scope-sft-lora-v1@c9b6e4a457902b99d8c3d9a6721afc0c79f574eb`.
The fresh-download weights retain SHA-256
`2367b18f638096e501db1d5b2a66917b1d534ac256a9461ac9d84a19e3af2de9`.
Policy Prism permanently stores the two standard five-file directories and
catalog entries at commit `d409dc3`. Full-catalog validation passed with 81
runs and 1,418 traces. The prior E2B base runs have different evaluator
compatibility digests, so these absolute results must not be presented as a
strict before/after delta. The committed artifacts are the canonical evidence;
the known one-ULP span-summation ordering issue above should be fixed before a
future byte-for-byte finalization reproducibility gate is required.

Finally, PostTrain removed only the three successful E2B provider workspaces.
It reports no placements held and continues to list the reconciled training,
scope, and recovery identities with their promoted Trackio artifacts. The two
externally cancelled scope attempts remain retained as operational history.

## Context and Orientation

The repositories and immutable inputs are:

    PostTrain: /home/ali-awais-safdar/Post-Train/posttrain
    Policy Prism: /home/ali-awais-safdar/Policy Prism
    Live Kit: /home/ali-awais-safdar/Post-Train/posttrain-setup-v0.2.2-20260728/posttrain-setup
    Base model: google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7
    Dataset: carbonteq/policy-prism-scope-sft-luna-pro-v1@92ff4cfca942f65214313416ae5f787cd19106e3
    Target: targets/carbonteq-rtx-pro-6000-96gb
    Trackio project: policy-prism-scope-sft

The relevant PostTrain model implementation is
`packages/common/src/posttrain/common/variants/gemma4.py`; the framework base
catalog exposes variants from
`packages/catalog/src/posttrain/catalog/base/models.yaml`; and TRL already loads
all `gemma4` family variants through `AutoModelForMultimodalLM` while validating
that LoRA touches only seven language-model projections.

Policy Prism's `.posttrain/catalog/` directory composes the immutable dataset,
training, target, adapter, inference, environment, and evaluation selections.
Its `.posttrain/work_packages/` directory binds those selections to one training
job and two qualification jobs. `packages/training` owns dataset release
validation, while `packages/normative-verifiers` finalizes native Verifiers
artifacts into `evaluation-runs/<run-id>/`.

## Implementation Plan

### Milestone 1: PostTrain E2B support

Add `GEMMA_4_E2B_IT` with catalog ID `models/gemma4-e2b-it@bf16`, repository and
revision above, BF16 foundation form, family `gemma4`, exact parameter count
5,123,178,051, modalities text/image/audio/video, native context 131,072, MTP
disabled, and tokenizer fingerprint
`1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413`.
Expose a stable small-Gemma renderer alias without changing
`gemma4-e4b-tools@1`. Export the variant and alias from the common variants
package. Add catalog/model/loader/LoRA/rendering regression coverage.

Validate from PostTrain:

    uv sync --all-packages --locked --python 3.13
    uv run pytest packages/common/tests/test_model_variants.py \
      packages/catalog/tests/test_files.py \
      packages/train/tests/test_trl_common.py \
      packages/train/tests/test_rendering.py
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Commit and push before any project source references it. Record the full commit.

### Milestone 2: Policy Prism training composition

Reuse `policy-prism-scope/sft-one-epoch-v1` and add
`training/gemma4-e2b-policy-prism-scope-lora@1` with the E4B text-projection
regular expression and rank/alpha/dropout 32/64/0.05. Add
`gemma4_e2b_scope_one_epoch.yaml` binding E2B, the pinned training dataset, all
220 validation rows, the shared settings, and the new training binding. Do not
add or run an E2B smoke work package.

Validate the project catalog, work package, exact Hub dataset counts, sealed
leakage, and rendering equivalence. The gate is 4,279 train rows, 220 validation
rows, max 46,381 tokens, zero empty targets, zero over-limit records, and zero
truncated supervised tokens at 49,152. Commit and push before packaging.

### Milestone 3: Preflight and full SFT

Load the protected HF, Trackio, dstack, CA, registry, and OpenRouter environment
files without printing their values. Use the current PostTrain checkout through
one `pt` wrapper with `--project-root` pointing at Policy Prism and
`--env-file` pointing at `.env.posttrain`.

Run doctor, catalog validation, runtime-image verification, train/validation
dataset validation, work-package validation, job planning, and `job pack
--build-missing`. Verify the isolated container check and one idle reachable RTX
PRO 6000. Packaging must not reserve the GPU.

Submit only:

    run_id: policy-prism-e2b-sft-r32-v1
    work package: gemma4_e2b_scope_one_epoch.yaml
    provider: dstack
    timeout: 86400 seconds
    env: HF_TOKEN

Wait, reconcile, and inspect through `pt --json run show`, which uses
Observatory's normalized query path. Accept only 535 finite optimizer steps,
final validation, provider success, consistent reconciliation, successful
tracking, and retained adapter/recovery/summary roles. Resolve the exact adapter
Trackio name, immutable version, artifact digest, and content digest from the
run; never guess `v0`.

### Milestone 4: Adapter registration and qualification

After training, add the exact adapter model
`models/gemma4-e2b-policy-prism-scope-sft-lora@1`, its Trackio artifact,
provenance, and the inference binding
`inference/gemma4-e2b-policy-prism-scope-sft-lora-vllm-eval@1`. Serve BF16 with
vLLM 0.25.1, model length 131,072, chunked prefill, eager mode, text-only
profiling, max two sequences, and max LoRA rank 32.

Add scope and recovery work packages using the already-pinned successful sealed
environment revision `ef6f8e5a6bbbffce683afa08748878456913ab90`. Commit and
push, then pack both evaluation jobs while the GPU is free.

Submit scope as `policy-prism-e2b-sft-r32-v1-scope`; wait and reconcile. Require
18 included rollouts, zero failures/truncations/errors, complete trace sync,
complete Claude semantic evidence, and exactly one native evaluation artifact.
Only then submit recovery as `policy-prism-e2b-sft-r32-v1-recovery` and require
the analogous 17-case gate. A provider success with failed rollouts is not a
scientific success. Changed code/config requires a new `-rN` run ID.

### Milestone 5: Publication and permanent evidence

Materialize the exact adapter and two exact evaluation artifacts to new ignored
directories. Any temporary Trackio helper must require immutable `vN` references,
reject aliases and existing destinations, close its audit run, and verify
adapter configuration/weights or native `config.toml`/`traces.jsonl` as
appropriate.

Preview-finalize both evaluations into ignored state and use the deterministic
reports to create a model card containing base/dataset revisions, complete
training settings, source commits, Trackio lineage, scope/recovery results,
PEFT loading instructions, intended use, `teacher_generated_unverified`
limitations, and no legal-correctness claim.

Upload the final adapter and model card privately to
`carbonteq/gemma-4-e2b-policy-prism-scope-sft-lora-v1`, resolve the immutable HF
commit, download it freshly, and compare adapter config and weights with
Trackio. Stop if an existing repository contains conflicting adapter bytes.

Build secret-free serving metadata with exact base, adapter, HF, Trackio, vLLM,
GPU, and evaluation identities. Permanently finalize to:

    gemma-4-e2b-policy-prism-sft-r32-v1-v11-sealed-scope-20260805
    gemma-4-e2b-policy-prism-sft-r32-v1-v11-sealed-recovery-20260805

Each directory must contain `manifest.json`, `traces.jsonl`,
`business-kpis.json`, `engineering-metrics.json`, and
`semantic-diagnostics.json`. Run `validate-runs`, require 18/17 traces and
complete semantic status, verify the deterministic catalog contains both, then
commit and push the evidence.

Existing E2B base runs remain historical evidence and are not rerun. Their
evaluator implementation digest differs, so do not claim a strict numerical
before/after delta unless compatibility hashes match.

After every durable gate passes, clean only the full-SFT, scope, and recovery
provider workspaces. Preserve all Trackio artifacts, HF content, finalized
evidence, and prior failed evidence.

## Validation and Acceptance

PostTrain acceptance requires all focused and full checks above with no Qwen,
LFM, 12B, or E4B regression. Dataset acceptance requires the exact immutable
counts and identical small-Gemma rendering with no supervised truncation.
Training acceptance requires 535 steps, held-out validation, consistent
reconciliation, and all required artifact roles. Qualification acceptance
requires complete 18-case and 17-case scientific gates. Publication acceptance
requires a private immutable HF commit and fresh-download weight/config match.
Permanent evidence acceptance requires both standard five-file directories,
secret-free manifests, and a valid deterministic catalog.

## Idempotence and Recovery

Planning, packing, status, logs, wait, show, and reconciliation are safe to
repeat. After an ambiguous submission error, inspect the same run and use
`retry-submit`; do not create a duplicate blindly. A code or configuration
change requires a new run ID. Resume training only from an exact compatible
recovery artifact with unchanged model, dataset, package, and settings
identities. Do not clean a workspace until reconciliation, Trackio artifact
verification, HF verification, and permanent Policy Prism finalization all pass.

## Artifacts and Notes

Maintain these values during execution:

    E2B base revision / parameter count:
    Dataset repo / revision / counts:
    PostTrain implementation commit:
    Policy Prism pre-training commit:
    Training image digest / run ID / host / runtime:
    Final train loss / validation loss / peak VRAM:
    Recovery/checkpoint artifacts:
    Adapter Trackio project / name / vN / digests:
    Policy Prism adapter-config commit:
    Scope run / evaluation artifact / report / compatibility SHA:
    Recovery run / evaluation artifact / report / compatibility SHA:
    Adapter HF repo / immutable commit / verification digests:
    Final Policy Prism evidence commit:
    Provider cleanup results:

## Interfaces and Dependencies

The public PostTrain addition is `GEMMA_4_E2B_IT` plus catalog selection
`models/gemma4-e2b-it@bf16`; the stable E4B renderer identity remains unchanged.
Policy Prism adds one E2B training binding, one full training work package, one
materialized adapter model, one inference binding, and two domain-evaluation
work packages. No dataset schema, formatter, frozen product baseline, sealed
prompt, or generic framework API changes.
