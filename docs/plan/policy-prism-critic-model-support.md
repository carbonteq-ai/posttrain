# Add exact small causal-model support for Policy Prism critics

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current while the work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Policy Prism needs exact small causal language-model subjects for a fair classification and relationship-linking comparison: Qwen3.5-0.8B in non-thinking and thinking modes, LFM2.5-350M, LFM2.5-1.2B-Instruct, and the separate LFM2.5-1.2B-Thinking checkpoint. After this work, PostTrain can identify the exact immutable weights and tokenizer, apply each model-card decoding and reasoning contract, render LFM Instruct conversations without pretending it has a thinking switch, serve the variants through vLLM, and fine-tune the supported Instruct bases with assistant-only TRL LoRA. This change supplies capability; it does not run the Policy Prism comparison or merge this feature branch into `main`.

## Progress

- [x] (2026-08-21 14:10Z) Fetched `origin`, recorded the release baseline, and created `feat/policy-prism-critic-model-support` directly from `origin/main`.
- [x] (2026-08-21 14:38Z) Verified the three upstream repositories, exact revisions, parameter counts, tokenizer bytes, templates, and operational context limits.
- [x] (2026-08-21 14:55Z) Added the two exact LFM BF16 variants and a dedicated package-owned Instruct renderer contract.
- [x] (2026-08-21 14:58Z) Added vLLM inference and TRL all-linear LoRA bindings for both LFM Instruct variants.
- [x] (2026-08-21 15:06Z) Completed renderer, assistant-mask, stop-token, no-truncation, command, adapter, and catalog tests.
- [x] (2026-08-21 15:13Z) Ran the locked full validation ladder: Ruff, Pyright, import contracts, and 1,272 tests passed; 25 environment-dependent tests skipped.
- [x] (2026-08-21 15:18Z) Created two focused commits and prepared both for delivery only to `origin/feat/policy-prism-critic-model-support` (`c974c7e`, `2ddbee8`).
- [ ] During the later Policy Prism model-comparison phase, capture real vLLM load/generation and post-SFT LoRA reload evidence on the selected GPU. No GPU allocation is part of this implementation phase.
- [x] (2026-08-21) Extended the frozen training-loop contract with finite non-negative weight decay and the cosine scheduler required by the selected Policy Prism critic SFT runs.
- [x] (2026-08-21) Added a standalone authenticated Trackio preflight that creates no run, matching the approved SFT handoff on the current main-derived branch.
- [x] (2026-08-21) Re-ran the complete locked repository gate after the SFT amendments: Ruff, Pyright, all 8 import contracts, and 1,279 tests passed; 25 environment-dependent tests skipped.
- [x] (2026-08-22) Extended resolved job-plan evidence to preserve tokenizer fingerprints, checkpoint/logging policy, gradient-checkpointing state, and loss-only validation settings before immutable packing.
- [x] (2026-08-22) Re-ran the complete framework gate after the plan-evidence change: Ruff, Pyright, all 8 import contracts, and 1,289 tests passed; 25 environment-dependent tests skipped.
- [ ] Submit and later reconcile the two Policy Prism critic SFT jobs; submission and qualification evidence remain outside the model-support-only commits above.
- [x] (2026-08-27) Connected each evaluation inference binding's complete sampling and reasoning-mode selection to the native Verifiers request, refreshed the exact LFM Thinking revision/template/parser, and validated the Policy Prism decoding path offline. Live GPU handshakes remain part of the later prompt-v3 model-selection run.

## Surprises & Discoveries

- Observation: the LFM2.5-350M and LFM2.5-1.2B-Instruct repositories contain byte-identical tokenizers and chat templates at the requested revisions.
  Evidence: `tokenizer.json` hashes to `df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd`; `chat_template.jinja` hashes to `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6` for both.
- Observation: the exact current LFM Thinking revision uses the same tokenizer and chat-template bytes as the pinned Instruct revision; PostTrain's older Thinking compatibility template was stale.
  Evidence: both exact artifacts hash to tokenizer `df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd` and chat template `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6`; mode and reasoning-parser behavior remain checkpoint/binding-specific.
- Observation: the upstream JSON configs advertise 128,000 positions, while both official model cards state a supported context length of 32,768 tokens.
  Evidence: this plan uses 32,768 as the operational and catalog capability limit instead of claiming the larger raw configuration value.
- Observation: the repository's default test environment intentionally omits Transformers and the training renderer dependency, so tokenizer tests skip in the ordinary full suite.
  Evidence: a locked minimal test environment with Transformers 5.14.1 and renderers 0.1.8 executed the new real-tokenizer cases; 7 applicable tests passed while unrelated uncached model cases skipped.
- Observation: the first full-suite run exposed one fixed-set inventory test that needed the two new model IDs.
  Evidence: after adding `lfm2.5-350m` and `lfm2.5-1.2b-instruct` to `IdentityContractTests`, the entire 1,294-collected-test run passed with 1,272 passed and 25 skipped.
- Observation: the approved Policy Prism SFT recipe requires cosine scheduling and weight decay, but the main-derived `TrainingLoop` exposed neither exact contract.
  Evidence: before this amendment the scheduler literal accepted only linear and constant variants, and TRL trainer arguments did not receive `weight_decay`.
- Observation: the handoff calls `posttrain run tracking-preflight`, but that command was not present on current `origin/main` even though the reusable Trackio readiness API was already maintained and tested.
  Evidence: the feature branch now adds only the CLI wrapper and its no-run unit test; it does not import the unrelated OPD recovery changes where the command first appeared.
- Observation: PostTrain already rejects non-finite loss or gradient metrics in its shared TRL observation callback.
  Evidence: the new SFT-specific first-step regression proves `loss=nan` at global step 1 raises `FloatingPointError`; no duplicate callback was introduced.
- Observation: the canonical product documents already assign purpose-specific generation defaults to `InferenceBinding.sampling`, but the Verifiers evaluation adapter currently takes generation values only from the environment cell.
  Evidence: `docs/post-training/02-primitives.md` assigns temperature and output limits to the inference binding, while `_native_sampling` in the Verifiers adapter currently reads `request.environment.sampling` and ignores the selected local inference binding's sampling mapping.

## Decision Log

- Decision: preserve the existing `lfm2.5-tools@1` contract for the Thinking variant and add `lfm2.5-instruct-tools@1` for the two requested Instruct variants.
  Rationale: changing the old renderer would mutate already-qualified behavior; the requested models must not inherit a reasoning parser or Thinking template.
  Date/Author: 2026-08-21 / Codex.
- Decision: package the exact official Instruct template and record both its SHA-256 and the tokenizer SHA-256 in immutable model identity.
  Rationale: a repository revision alone does not make rendered training bytes visible in traces. A package-owned template plus hashes makes train and serve behavior reproducible.
  Date/Author: 2026-08-21 / Codex.
- Decision: set the Instruct reasoning mode to `off` with `preserve_thinking=false` and omit a vLLM reasoning parser.
  Rationale: these are instruction models used for closed JSON decisions, not reasoning-model outputs.
  Date/Author: 2026-08-21 / Codex.
- Decision: use BF16 weights, 32,768 operational context, and all-linear LoRA for both LFM variants.
  Rationale: these are the requested contracts and match the official model cards' recommended native Transformers/vLLM and TRL paths.
  Date/Author: 2026-08-21 / Codex.
- Decision: treat real GPU loading and trained-adapter reload as a live qualification gate in the subsequent authorized evaluation/SFT phase, while unit-testing command and adapter lifecycle construction here.
  Rationale: this phase adds PostTrain support and does not authorize starting the RunPod evaluation or local RTX training job. Code must not claim live evidence that was not produced.
  Date/Author: 2026-08-21 / Codex.
- Decision: add `weight_decay` to the shared `TrainingLoop` with a behavior-preserving zero default and make cosine an explicit scheduler option.
  Rationale: optimization intent belongs to algorithm settings, and passing the exact frozen values through the existing TRL translation keeps the SFT work packages reproducible without Policy Prism-specific trainer code.
  Date/Author: 2026-08-21 / Codex.
- Decision: expose the existing authenticated Trackio readiness check through the primary CLI rather than carrying a temporary handoff script.
  Rationale: job submission must prove tracking writes before spending GPU time; a narrow reusable command is safer and matches the product CLI boundary.
  Date/Author: 2026-08-21 / Codex.
- Decision: make local evaluation sampling resolve from environment defaults followed by the selected inference binding, and make the inference binding own an optional explicit reasoning mode.
  Rationale: environment defaults retain task-owned output budgets, while the selected model/runtime binding must authoritatively supply model-card sampling and renderer mode. This implements the already-frozen product ownership without a Policy Prism-specific framework type.
  Date/Author: 2026-08-27 / Codex.
- Decision: defer real vLLM calls to the prompt-v3 model-selection execution and accept this slice using exact catalog contracts, a real HTTP-compatible fake endpoint, and offline tokenizer measurements.
  Rationale: live handshakes and the full model-selection evaluation can share the same GPU allocation and exact run manifests without weakening the offline runtime-contract tests.
  Date/Author: 2026-08-27 / Codex.

## Outcomes & Retrospective

The exact variants, dedicated Instruct renderer, vLLM bindings, TRL all-linear LoRA binding, SFT settings, and requested contract tests are complete. Evaluation sampling now resolves from task defaults followed by the selected inference binding, preserving exact Qwen/LFM card modes and renderer kwargs. The official LFM Thinking revision, tokenizer/template identity, BF16 runtime, and `qwen3` parser are corrected. Focused contracts, catalog validation, and import boundaries pass. Actual GPU loading and HTTP parameter acceptance remain a deliberately deferred handshake at the start of prompt-v3 model selection.

## Context and Orientation

`packages/common/src/posttrain/common/variants/lfm25.py` is the Python authority for built-in LFM model and conversation facts. `packages/common/src/posttrain/common/templates/` stores package-owned Jinja templates. `packages/catalog/src/posttrain/catalog/base/models.yaml`, `inference.yaml`, and `training.yaml` expose immutable selections to jobs. `packages/train/src/posttrain/train/rendering.py` uses the selected model conversation contract and a `TrainingRenderer` to produce token IDs and assistant-only loss masks. `packages/serve/src/posttrain/serve/backends/vllm/server.py` translates an `InferenceBinding` into the vLLM command and handles PEFT adapters without replacing base weights.

A renderer is the exact transformation from role-labelled messages into tokens. A tokenizer fingerprint is the SHA-256 of the pinned `tokenizer.json`. A chat-template fingerprint is the SHA-256 of the exact Jinja template. A LoRA adapter is a small set of trainable low-rank weights attached to the immutable base model; `all-linear` means PEFT targets every supported linear projection rather than a model-specific hand-maintained list.

The branch baseline is `origin/main` commit `42da687665d1661aef512eabb7adf59a5a5307d6`, reachable tag `v0.3.20`, with `uv.lock` SHA-256 `6418c5fbaa9d378af767d739b8ba407eb05bcb7595920ecde3401b26541a9d98`. The prior OPD branch remains preserved separately in Git.

The immutable model matrix is:

- `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`; existing PostTrain support is reused.
- `LiquidAI/LFM2.5-350M` at `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`; 354,483,968 BF16 parameters.
- `LiquidAI/LFM2.5-1.2B-Instruct` at `df58c174f05ff733f83f8cae10ea9298224c8006`; 1,170,340,608 BF16 parameters.
- `LiquidAI/LFM2.5-1.2B-Thinking` at `f313478934a7612d22991f752959d7a1a8756fec`; separate BF16 reasoning subject for evaluation only.

## Plan of Work

First, add the two exact LFM variants to `lfm25.py` and `models.yaml`, pin the shared tokenizer fingerprint, package the official Instruct template, and register a separate renderer whose only reasoning mode is off. Extend common tests to prove exact repository/revision/parameter identity, template separation, exact template bytes, deterministic ChatML rendering, and the 32K operational limit.

Second, add inference selections to `inference.yaml` for both LFM Instruct bases. Each selection uses vLLM, BF16, the Instruct renderer, a 32K maximum model length, the LFM tool parser, and no reasoning parser. Extend serve tests to prove the selected base revision, chat-template override, operational limit, BF16 dtype, parser separation, and PEFT adapter construction.

Third, add an Instruct `TrainingRenderer`, TRL SFT settings, and an all-linear LoRA training binding in `profiles.py` and `training.yaml`. Extend rendering tests so actual pinned tokenizers prove deterministic bytes, assistant-only mask boundaries, final stop-token supervision, and no target truncation at the configured limit. Extend TRL loader tests so the LFM causal factory receives the exact revision and the PEFT configuration uses `target_modules="all-linear"`; retain the existing adapter-resume/reload path.

Finally, run the locked dependency sync and all repository gates. Update this plan with exact results, commit the model/renderer work first, commit bindings/tests/documentation second, and push only this feature branch. Real GPU serve/generate and trained-adapter reload evidence will be added to the plan during the next phase when the comparison or SFT job is actually run.

## Concrete Steps

Run all commands from `/home/ali-awais-safdar/Post-Train/posttrain`.

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Before the first commit, run the focused common and catalog tests. Before the second commit, run the focused serve and train tests. After both commits, rerun the full ladder and push:

    git push -u origin feat/policy-prism-critic-model-support

## Validation and Acceptance

The feature is accepted when the catalog resolves both exact LFM models, inference bindings, SFT settings, and the all-linear LoRA training binding; the rendered Instruct template matches the official SHA-256; the Thinking and Instruct contracts are distinct; actual tokenizer rendering has deterministic bytes and assistant-only supervised tokens including the final stop marker; configured training examples do not truncate assistant targets; vLLM commands carry the exact revision, BF16 dtype, 32K limit, Instruct template, and LFM tool parser without a reasoning parser; and a materialized LFM PEFT variant generates the correct `--enable-lora` and `--lora-modules` arguments.

The full static and test ladder must pass. Live GPU qualification remains explicitly pending until the later model-evaluation/SFT phase and must record the exact model, renderer, tokenizer, inference binding, adapter digest, GPU, and trace IDs when performed.

## Idempotence and Recovery

Catalog additions and tests are deterministic. Dependency installation uses the checked-in lock and may be rerun. If a test fails after a partial edit, retain the branch, repair the smallest relevant file, and rerun focused tests before the full ladder. Do not reset or alter the preserved OPD branch. Do not merge this branch into `main` during Slice 5.

## Artifacts and Notes

Baseline evidence:

    origin/main 42da687665d1661aef512eabb7adf59a5a5307d6
    release tag v0.3.20
    uv.lock sha256 6418c5fbaa9d378af767d739b8ba407eb05bcb7595920ecde3401b26541a9d98

Official-artifact evidence:

    LFM Instruct tokenizer.json sha256 df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd
    LFM Instruct chat_template.jinja sha256 ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6
    LFM Thinking chat_template.jinja sha256 ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6

Validation evidence:

    uv sync --all-packages --locked --python 3.13: resolved 287, checked 106
    uv run ruff check .: passed
    uv run pyright: 0 errors, 0 warnings
    uv run lint-imports: 8 contracts kept, 0 broken
    uv run pytest: 1272 passed, 25 skipped
    exact LFM tokenizer/rendering subset: 7 passed, unrelated uncached variants skipped
    git diff --check: passed
    2026-08-27 focused decoding contracts: 51 passed, 2 skipped
    2026-08-27 catalog validation: 80 base entries, 94 project entries

## Interfaces and Dependencies

At completion, `posttrain.common.variants` exports `LFM_25_350M`, `LFM_25_12B_INSTRUCT`, and `LFM25_INSTRUCT_RENDERER_CONTRACT`. The base catalog exposes `models/lfm2.5-350m@bf16`, `models/lfm2.5-1.2b-instruct@bf16`, matching vLLM inference selections, one shared Instruct TRL LoRA binding, and Instruct SFT smoke settings. These reuse the existing `ModelVariant`, `RendererContract`, `InferenceBinding`, `TrainingBinding`, `SFTSettings`, and PEFT/vLLM adapter mechanisms; no Policy Prism-specific model profiles or provider integration are introduced.

Revision note (2026-08-21): created the living plan after verifying the repository baseline and official immutable artifacts so subsequent implementation and qualification can resume from this file alone.

Revision note (2026-08-21): recorded completed inference/SFT contracts, exact-tokenizer rendering evidence, the full validation result, and the remaining live-GPU qualification boundary.

Revision note (2026-08-21): extended the plan for the authorized critic SFT phase after discovering that the approved optimizer and Trackio preflight contracts were not representable on the current main-derived branch. The changes remain framework-generic and preserve existing defaults.
