# Add exact small causal-model support for Policy Prism critics

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current while the work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Policy Prism needs three small causal language-model bases for a fair classification and relationship-linking comparison: Qwen3.5-0.8B, LFM2.5-350M, and LFM2.5-1.2B-Instruct. After this work, PostTrain can identify the exact immutable weights and tokenizer, render LFM Instruct conversations without the LFM Thinking behavior, serve both LFM bases through vLLM, and fine-tune either with assistant-only TRL LoRA. This change supplies capability; it does not run the Policy Prism comparison or merge this feature branch into `main`.

## Progress

- [x] (2026-08-21 14:10Z) Fetched `origin`, recorded the release baseline, and created `feat/policy-prism-critic-model-support` directly from `origin/main`.
- [x] (2026-08-21 14:38Z) Verified the three upstream repositories, exact revisions, parameter counts, tokenizer bytes, templates, and operational context limits.
- [x] (2026-08-21 14:55Z) Added the two exact LFM BF16 variants and a dedicated package-owned Instruct renderer contract.
- [ ] Add vLLM inference and TRL all-linear LoRA bindings for both LFM Instruct variants.
- [ ] Complete renderer, assistant-mask, stop-token, no-truncation, command, adapter, and catalog tests.
- [ ] Run the locked full validation ladder and record results.
- [ ] Create two focused commits and push only `origin/feat/policy-prism-critic-model-support`.
- [ ] During the later Policy Prism model-comparison phase, capture real vLLM load/generation and post-SFT LoRA reload evidence on the selected GPU. No GPU allocation is part of this implementation phase.

## Surprises & Discoveries

- Observation: the LFM2.5-350M and LFM2.5-1.2B-Instruct repositories contain byte-identical tokenizers and chat templates at the requested revisions.
  Evidence: `tokenizer.json` hashes to `df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd`; `chat_template.jinja` hashes to `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6` for both.
- Observation: the official LFM Instruct template is not the Thinking template and is also more complete than PostTrain's older Thinking compatibility template.
  Evidence: the pinned Thinking template hashes to `f05bf4b967dc993bdc7a2fe6e43759ee218eb0eb340d68b063e1c4f8ad148176`; the Instruct template contains explicit Transformers `{% generation %}` boundaries that support assistant-only loss masks.
- Observation: the upstream JSON configs advertise 128,000 positions, while both official model cards state a supported context length of 32,768 tokens.
  Evidence: this plan uses 32,768 as the operational and catalog capability limit instead of claiming the larger raw configuration value.

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

## Outcomes & Retrospective

The branch and exact renderer/model identities are complete. Inference, SFT bindings, final tests, validation evidence, and delivery commits remain in progress.

## Context and Orientation

`packages/common/src/posttrain/common/variants/lfm25.py` is the Python authority for built-in LFM model and conversation facts. `packages/common/src/posttrain/common/templates/` stores package-owned Jinja templates. `packages/catalog/src/posttrain/catalog/base/models.yaml`, `inference.yaml`, and `training.yaml` expose immutable selections to jobs. `packages/train/src/posttrain/train/rendering.py` uses the selected model conversation contract and a `TrainingRenderer` to produce token IDs and assistant-only loss masks. `packages/serve/src/posttrain/serve/backends/vllm/server.py` translates an `InferenceBinding` into the vLLM command and handles PEFT adapters without replacing base weights.

A renderer is the exact transformation from role-labelled messages into tokens. A tokenizer fingerprint is the SHA-256 of the pinned `tokenizer.json`. A chat-template fingerprint is the SHA-256 of the exact Jinja template. A LoRA adapter is a small set of trainable low-rank weights attached to the immutable base model; `all-linear` means PEFT targets every supported linear projection rather than a model-specific hand-maintained list.

The branch baseline is `origin/main` commit `42da687665d1661aef512eabb7adf59a5a5307d6`, reachable tag `v0.3.20`, with `uv.lock` SHA-256 `6418c5fbaa9d378af767d739b8ba407eb05bcb7595920ecde3401b26541a9d98`. The prior OPD branch remains preserved separately in Git.

The immutable model matrix is:

- `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`; existing PostTrain support is reused.
- `LiquidAI/LFM2.5-350M` at `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`; 354,483,968 BF16 parameters.
- `LiquidAI/LFM2.5-1.2B-Instruct` at `df58c174f05ff733f83f8cae10ea9298224c8006`; 1,170,340,608 BF16 parameters.

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
    LFM Thinking chat_template.jinja sha256 f05bf4b967dc993bdc7a2fe6e43759ee218eb0eb340d68b063e1c4f8ad148176

## Interfaces and Dependencies

At completion, `posttrain.common.variants` exports `LFM_25_350M`, `LFM_25_12B_INSTRUCT`, and `LFM25_INSTRUCT_RENDERER_CONTRACT`. The base catalog exposes `models/lfm2.5-350m@bf16`, `models/lfm2.5-1.2b-instruct@bf16`, matching vLLM inference selections, one shared Instruct TRL LoRA binding, and Instruct SFT smoke settings. These reuse the existing `ModelVariant`, `RendererContract`, `InferenceBinding`, `TrainingBinding`, `SFTSettings`, and PEFT/vLLM adapter mechanisms; no Policy Prism-specific model profiles or provider integration are introduced.

Revision note (2026-08-21): created the living plan after verifying the repository baseline and official immutable artifacts so subsequent implementation and qualification can resume from this file alone.
