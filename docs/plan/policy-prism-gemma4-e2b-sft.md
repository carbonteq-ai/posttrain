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
- [ ] Run scope and recovery sequentially and pass their scientific gates. The
  first scope attempt `policy-prism-e2b-sft-r32-v1-r1-scope` was cancelled by
  the controlling client after 20 minutes while managed vLLM was still
  starting; it produced no rollouts. Recovery remains intentionally
  unsubmitted, and the scope retry will use a fresh immutable run ID after the
  shared workstation is free.
- [ ] Publish and verify the private Hugging Face adapter.
- [ ] Finalize, validate, commit, and push both Policy Prism evaluation runs.
- [ ] Clean only the three new provider workspaces and record final outcomes.

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

## Outcomes & Retrospective

Execution is in progress. Record final source commits, job image digests, target
host, training runtime/loss/validation, checkpoint and adapter identities, HF
commit, evaluation metrics, compatibility digests, and cleanup results here.

The complete SFT finished successfully on `carbonteq-ai-workstation.lan` in
about 3 h 34 min. It produced final loss `0.0407577`, held-out validation loss
`0.0632281`, token accuracy `0.988076`, zero truncation, and exact adapter
`training-models-gemma4-e2b-it-bf16-sft-lora-adapter:v0` with artifact digest
`d34005655b8270d5a663d49a799bc19bb5fa101d6ccf3ad4aba8da812d8c73d5` and
content digest
`f58bac190bc25f1f37c6f2c013f74a29fc2fe2404e073fe8e51de5d9d0020028`.
Scope package/image are `c14e1de...`/`a48bcdf...`; recovery package/image are
`3a52b97e...`/`4781468e...`. Qualification, publication, permanent evidence,
and cleanup remain in progress.

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
