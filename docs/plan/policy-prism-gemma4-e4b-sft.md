# Release and run the Policy Prism Gemma 4 E4B SFT experiment

This ExecPlan is a living execution document maintained according to `docs/templates/PLAN.md`. It is the single authority for preparing the Policy Prism SFT dataset, running the experiment on the in-house GPU, qualifying the resulting adapter, and publishing the reproducible outputs. Update `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` whenever work advances.

## Purpose / Big Picture

After this work, CarbonTeq can reproduce one rank-32 LoRA SFT of the exact `google/gemma-4-E4B-it` revision on a private, immutable Policy Prism dataset release; inspect the run in Trackio and PostTrain Observatory; evaluate the exact retained adapter with the sealed Policy Prism Verifiers environments; and retrieve the adapter from both Trackio and the private CarbonTeq Hugging Face organization.

The experiment is deliberately split across two repositories. Policy Prism owns the meaning, curation, release, balance report, and sealed evaluations of the legal dataset. PostTrain owns model profiles, tokenizer-driven rendering, LoRA training, remote execution, tracking, artifact lineage, managed serving, and evaluation orchestration. The dataset contains model-neutral chat messages rather than Gemma-rendered strings, so the same release remains usable by another supported model.

## Progress

- [x] (2026-08-04 14:08Z) Audited the current PostTrain, Policy Prism, Live Kit, finalized trace file, GPU fleet, Trackio, registry, and existing base-model evaluations.
- [x] (2026-08-04 14:25Z) Fast-forwarded `feat/policy-prism-gemma4-e4b-sft` to teammate PR #12 head `81d5215063c926922a93250958201e5d7ef03815` and corrected its outstanding Ruff formatting failure.
- [x] (2026-08-04 14:32Z) Added the exact E4B variant, E4B-specific renderer contract, global catalog entry, multimodal Transformers loader, nested cache disabling, and safe seven-projection language-model LoRA validation with focused tests.
- [x] (2026-08-04 14:35Z) Added positive `max_lora_rank` support to vLLM keyword and CLI translation with focused tests.
- [x] (2026-08-04 14:40Z) Audited the 4,545 finalized records and found 46 duplicated complete conversations; selected deterministic exact-content deduplication before splitting.
- [x] (2026-08-04 15:06Z) Completed locked sync, live E4B tokenizer tests, Ruff check/format, Pyright, import boundaries, 936-test repository suite, and diff validation.
- [x] (2026-08-04) Added the missing `packages/environment` actual-job source root after the first live pack exposed the incomplete framework source closure; added a focused regression test.
- [ ] Commit and push the validated PostTrain changes, then record the resulting full commit in Policy Prism pins and release evidence.
- [ ] Make the current Policy Prism evaluation work green, then commit and push the existing 179-file change set before starting dataset-release work.
- [ ] Add the minimal Policy Prism `packages/training` package, release builder, validation CLI, tests, and PostTrain project files described here.
- [ ] Build and validate the deduplicated 4,499-row release, including sealed-data leakage and exact E4B rendering audits.
- [ ] Publish the private dataset, resolve its immutable Hub commit, and pin all three dataset selections to that commit.
- [ ] Initialize current PostTrain machine configuration and pass doctor, catalog, image, dataset, model-access, and job-plan preflights.
- [ ] Run and reconcile the one-step GPU smoke; record its adapter, package, tracking, and peak-memory evidence.
- [ ] Run and reconcile the 535-step SFT; record the exact final adapter and training evidence.
- [ ] Register and run the two adapter evaluation cells; finalize native Verifiers evidence and compare compatibility hashes with the existing base runs.
- [ ] Publish the private PEFT adapter to Hugging Face, verify a clean download, then clean provider workspaces.

## Surprises & Discoveries

- Observation: the first live actual-job image reached its isolated runtime smoke but failed to import `posttrain.environment` from `posttrain-catalog`.
  Evidence: source-based packing explicitly staged framework install roots and omitted `packages/environment`, even though both `posttrain-catalog` and `posttrain-runtime` declare `posttrain-environment`; local tests had the package installed and therefore masked the omission. The source closure and its regression test now include that package.

- Observation: current PostTrain already implements the general SFT normalization boundary. `packages/data/src/posttrain/data/catalog.py` accepts Hugging Face, JSONL, Parquet, fixture, built, and NeMo sources, while `packages/data/src/posttrain/data/adapters/huggingface.py` normalizes messages, prompt/completion, Alpaca, and ShareGPT rows into `SupervisedExample`.
  Evidence: Policy Prism needs a domain adapter from its trace schema to canonical `messages`; it does not need a second universal formatter or model-specific rendered text.

- Observation: teammate PR #12 supports Gemma 4 Unified 12B and supplies the shared `gemma4-tools@1` renderer contract, but it did not contain the E4B variant or rank-32 vLLM capacity.
  Evidence: the current branch now adds `models/gemma4-e4b-it@bf16` and `VllmEngineConfig.max_lora_rank` on top of exact PR head `81d5215`.

- Observation: the pinned E4B and Unified 12B tokenizers do not have identical generation-prompt behavior when thinking is disabled.
  Evidence: 12B appends an empty thought channel, whereas E4B ends directly at `<|turn>model\n`; E4B therefore uses the distinct `gemma4-e4b-tools@1` renderer contract while retaining the verified Gemma 4 family roles, flags, and tool syntax.

- Observation: the source file has 4,545 accepted rows and SHA-256 `0c53a9884e2abbfb01abb84a4be8ea835118dce6d4d9433039e9298f4c87651e`, but 46 pairs contain identical complete canonical conversations.
  Evidence: canonical hashing of `request.messages` plus `training_response` yields 4,499 unique conversations. Keeping both copies would silently give those targets twice the SFT weight.

- Observation: after deduplication and instrument-isolated splitting, the correct counts are 4,279 train and 220 validation, so one epoch at global batch eight is 535 optimizer steps, not the former 541.
  Evidence: `ceil(4279 / 8) == 535`; train and validation contain 228 and 26 regulatory instruments respectively.

- Observation: the exact PostTrain E4B renderer reports token lengths min 1,654, p50 5,164, p95 20,530, p99 31,662, and max 46,381 after deduplication, with zero empty targets and zero rows above 49,152.
  Evidence: the longest retained training example is `production-01443-independent_rule_inventory-peu_ecfr-42-cfr-488-488.61_936d9b4064/graph`, with 46,381 input tokens and 8,963 supervised tokens.

- Observation: Policy Prism currently has 179 changed paths and one known failing test, `test_missing_scope_rule_remains_in_qualification_and_span_denominators`; its old training scaffold exists only on `origin/develop` and depends on an obsolete benchmark package.
  Evidence: remote job packages must never snapshot that dirty state. Only the small `posttrain_entry.py` pattern should be adapted from the old scaffold.

- Observation: the Live Kit v0.2.2 contains valid internal CA and service secrets but installs an older framework and lacks the current dstack/Trackio Python environment.
  Evidence: use the kit only as a secret source; run every command through the current PostTrain checkout and a dedicated dstack 0.20.29 environment.

## Decision Log

- Decision: keep the canonical dataset model-neutral and let the selected PostTrain model profile own chat-template rendering and tokenization.
  Rationale: it preserves one portable SFT contract and prevents Policy Prism from duplicating or drifting from Gemma behavior.
  Date/Author: 2026-08-04 / Codex.

- Decision: give E4B its own `gemma4-e4b-tools@1` renderer identity rather than reuse Unified 12B's `gemma4-tools@1` identity.
  Rationale: the two pinned templates share family semantics but differ observably in the non-thinking generation prefix; a distinct contract prevents an unsupported claim of exact template equivalence.
  Date/Author: 2026-08-04 / Codex.

- Decision: exact-content-deduplicate before splitting, keeping the lexicographically smallest source ID and recording every contributing source ID plus `duplicate_count`.
  Rationale: this removes accidental double weighting while retaining auditable provenance. It changes the frozen raw input neither in place nor silently.
  Date/Author: 2026-08-04 / Codex.

- Decision: define “balanced” as deduplicated, leakage-free, instrument-isolated, coverage-audited, deterministically interleaved, and fully reported; do not force equal class or source-family counts.
  Rationale: rules legitimately contain both `full_scope` and `standalone_rules`, and eCFR reflects the curated corpus composition. Oversampling or downsampling without a target deployment distribution would distort the first faithful experiment.
  Date/Author: 2026-08-04 / Codex.

- Decision: publish the private dataset and pin its Hub commit before smoke or full training.
  Rationale: the uploaded release and the trained release then have one immutable identity, and the 369 MB raw trace never enters a remote job image.
  Date/Author: 2026-08-04 / Codex.

- Decision: use `max_length=49152`, global batch eight, rank/alpha/dropout `32/64/0.05`, and one 535-step epoch for the first full run.
  Rationale: the measured maximum fits with 2,771 tokens of headroom, while one epoch provides the fastest interpretable screening experiment.
  Date/Author: 2026-08-04 / Codex.

- Decision: the smoke uses eight training examples, the real gradient accumulation of eight, one optimizer step, and no validation seat. The full job alone uses `train/trl-sft-validated@1` with all 220 held-out examples at the end.
  Rationale: the smoke proves the complete forward/backward/artifact path without spending time evaluating the full validation set.
  Date/Author: 2026-08-04 / Codex.

- Decision: do not rerun the base model initially. Reuse the two existing immutable base reports and rerun a base cell only if the finalized adapter report has a different compatibility hash.
  Rationale: repeated base inference adds cost but no evidence when model, prompt, task set, sampling, judge, and compatibility hash are identical.
  Date/Author: 2026-08-04 / Codex.

- Decision: publish a PEFT adapter, not merged or copied Gemma foundation weights, to private `carbonteq/gemma-4-e4b-policy-prism-scope-sft-lora-v1`.
  Rationale: the adapter is the produced artifact, is much smaller, and retains explicit lineage to the gated pinned base model.
  Date/Author: 2026-08-04 / Codex.

## Outcomes & Retrospective

PostTrain framework implementation and its complete local validation ladder are complete: 936 tests passed with 19 expected skips, Pyright reported zero errors, all eight import contracts were kept, Ruff check/format passed, and the live pinned E4B tokenizer tests passed. The changes are intentionally uncommitted pending review, and Policy Prism has not been modified. Update this section after each remote milestone with the dataset Hub commit, PostTrain and Policy Prism commits, smoke and full run IDs, package digests, Trackio adapter reference/digest, peak GPU memory, final train and validation losses, finalized evaluation IDs and compatibility hashes, Hugging Face model commit, and the decision on a follow-up experiment.

## Context and Orientation

The relevant paths are:

    PostTrain checkout: /home/ali-awais-safdar/Post-Train/posttrain
    Policy Prism checkout: /home/ali-awais-safdar/Policy Prism
    Live Kit: /home/ali-awais-safdar/Post-Train/posttrain-setup-v0.2.2-20260728/posttrain-setup
    Raw input: /home/ali-awais-safdar/Policy Prism/trace-generation-runs/gpt-luna-pro-scope-5000-v1/accepted-stage-records.jsonl

At plan revision time, PostTrain is on `feat/policy-prism-gemma4-e4b-sft` at teammate head `81d5215063c926922a93250958201e5d7ef03815` plus uncommitted E4B, serving, test, formatting, and plan changes. Policy Prism is on `fea/on-policy-distill-env@posttrain` at `d969de68cc986b5a4fb4a872c18a7f26b2a22b77` with 179 changed paths. These hashes are observations, not the final source identities; remote execution must use later clean, pushed commits.

A model profile is the combination of a family renderer contract and one exact model variant. Gemma 4 Unified 12B and E4B remain in the `gemma4` family, but the pinned tokenizers have different non-thinking generation-prefix behavior, so they use separate renderer identities and separate model variants. The E4B facts used throughout this plan are:

    catalog id: models/gemma4-e4b-it@bf16
    repository: google/gemma-4-E4B-it
    revision: ee0ef6023621cff504d758262d4e04895a5af4a2
    architecture: Gemma4ForConditionalGeneration
    parameters: 7,996,156,490
    context window: 131,072
    renderer: gemma4-e4b-tools@1
    tokenizer fingerprint: 1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413

PostTrain already supports canonical supervised examples, assistant-only loss indices, Hugging Face dataset loading at a pinned commit, deterministic materialization, LoRA, BF16, gradient accumulation, checkpoint/recovery artifacts, Trackio tracking, Observatory, dstack execution, job planning, reconciliation, managed vLLM serving, and managed Verifiers evaluation. The current branch adds only the missing E4B-specific model/training qualification and rank-32 adapter serving.

Policy Prism owns the raw trace schema. Each accepted source row contains the two-message system/user request and the string `training_response`; the formatter appends the latter as the final assistant message. `training_response` must parse as JSON and satisfy the stage-specific schema, but its original string is preserved byte-for-byte. Raw provider responses, normalized intermediate responses, provider usage, sampling/wire data, and warnings are never copied to the released training rows.

## Dataset Release Contract

The builder is a deterministic release compiler, not a model renderer. For each retained record it emits exactly this shape:

    {
      "id": "<task_id>/<stage>",
      "messages": [
        <original system message>,
        <original user message>,
        {"role": "assistant", "content": "<training_response>"}
      ],
      "trainable_message_indices": [2],
      "metadata": {
        "task_id": "...",
        "stage": "evidence|rules|graph",
        "profile_id": "...",
        "regulatory_instrument_id": "...",
        "source_family": "...",
        "task_type": "...",
        "decision_class": "...",
        "acceptance_status": "...",
        "acceptance_reasons": [],
        "removed_span_count": 0,
        "resolved_span_count": 0,
        "legal_correctness": "teacher_generated_unverified",
        "source_task_ids": ["<task_id>/<stage>"],
        "duplicate_count": 1
      }
    }

The metadata object is an allowlist. Omit absent optional acceptance counts rather than inventing values. Do not include `raw_response`, `normalized_response`, `usage`, `sampling_tags`, request transport/sampling fields, `acceptance.training_text`, or provider warnings.

Canonical conversation content is compact, sorted-key UTF-8 JSON of the original `request.messages` and the assistant `training_response`. Hash that byte sequence with SHA-256. Group equal hashes, keep the lexicographically smallest `(task_id, stage)` row, and populate `source_task_ids` with all sorted source IDs and `duplicate_count` with the group size. Assert that 4,545 source rows become 4,499 unique conversations and that exactly 46 rows are removed. A different count means the input or transform changed and requires a new dataset revision.

After deduplication, group regulatory instrument IDs by `source_family`. Sort instruments within each family by SHA-256 of `20260710:{source_family}:{regulatory_instrument_id}`. For a singleton family select zero validation instruments; otherwise select the first `min(n - 1, max(1, (n + 5) // 10))`, which is ten percent rounded half-up while retaining training coverage. Put every row for a selected instrument in validation. Finally sort each split by the tuple `(sha256("20260728:" + id), id)` because the current TRL SFT backend deliberately sets `shuffle_dataset=False`.

The release must assert these exact results:

    unique rows: 4,499
    train rows / instruments: 4,279 / 228
    validation rows / instruments: 220 / 26
    train stages: evidence 1,084; rules 2,217; graph 978
    validation stages: evidence 61; rules 105; graph 54
    train profiles: complete_legal_interpretation 1,068
                    independent_rule_inventory 1,115
                    qualification_grounding_audit 1,086
                    source_faithful_responsibility 1,010

Validation must contain all three stages, all four profiles, both task types, all four decision classes, and every non-singleton source-family/stage cell. The `unknown` source family has one instrument and remains in training. Assert zero regulatory-instrument, source task ID, or canonical conversation-hash overlap between train and validation. Report source-family and per-instrument concentration in the manifest; do not oversample, undersample, or cap eCFR in this first experiment.

Select the eight smoke rows only from training after the renderer audit. Always include the 46,381-token longest row, then use deterministic greedy set cover with hash order as the tie-breaker to cover all three stages, all four profiles, both task types, and both acceptance statuses with eight distinct IDs. The smoke file is a training subset, not another held-out split.

Write `train.jsonl`, `validation.jsonl`, `smoke.jsonl`, `manifest.json`, and `README.md` into a temporary sibling directory and atomically rename it into place. Refuse to overwrite an existing release unless the caller passes `--force`; `--check` must rebuild into a temporary directory and byte-compare all five outputs without modifying the release.

The manifest records schema/transform version, input path/hash/row count, dedup algorithm and removed IDs, split and order seeds, selected validation instruments, output counts and SHA-256s, all distribution tables, duplicate audit, leakage-manifest SHA-256/result, E4B repository/revision/renderer/tokenizer fingerprint, exact token statistics, longest row ID, prohibited-field audit, and the Policy Prism/PostTrain Git commits when known. `README.md` is both a dataset card and Hugging Face split declaration for the `train`, `validation`, and `smoke` JSONL files. It must state that outputs are teacher-generated and not independently legally verified.

## Plan of Work

### Milestone 1 — Complete PostTrain support in the current branch

The teammate’s PR #12 code is the base. The current changes add `GEMMA_4_E4B_IT` and `GEMMA4_E4B_RENDERER_CONTRACT` in `packages/common/src/posttrain/common/variants/gemma4.py`, register both in `packages/common/src/posttrain/common/variants/__init__.py`, and add `models/gemma4-e4b-it@bf16` to `packages/catalog/src/posttrain/catalog/base/models.yaml`. Tests cover exact model facts, the shared family-level tool/thinking capabilities, and the distinct tokenizer-owned generation-prefix behavior of 12B and E4B.

`packages/train/src/posttrain/train/backends/trl/common.py` selects `AutoModelForMultimodalLM` for the Gemma 4 family, disables cache on the wrapper and nested text configuration, and validates LoRA targets before PEFT wrapping. The only allowed experiment regex is:

    ^model[.]language_model[.]layers[.]\d+[.](self_attn[.](q_proj|k_proj|v_proj|o_proj)|mlp[.](gate_proj|up_proj|down_proj))$

Validation must find all seven projection leaf names and no vision/audio modules; `all-linear`, zero matches, an incomplete projection set, or any non-language-model match must fail before training. `packages/serve/src/posttrain/serve/profiles/base.py` accepts an optional positive `max_lora_rank` and emits it to both the vLLM Python binding and `--max-lora-rank` command.

Run focused tests first, then the complete repository ladder in `Concrete Steps`. When all pass, commit and push these changes and record the full PostTrain commit. Policy Prism must use that exact commit in its optional PostTrain dependencies and remote environment lock; no published wheel or Live Kit refresh is required because current PostTrain snapshots its exact framework source into each job package.

### Milestone 2 — Establish a clean Policy Prism baseline

Before adding SFT release code, finish the current evaluation work on the existing Policy Prism branch. Make the known denominator test and the entire current suite pass, inspect the 179 changed paths, then commit and push them. Do not merge `origin/develop` wholesale and do not package a dirty checkout.

After that clean checkpoint, selectively adapt only the useful entry-point idea from `origin/develop:packages/training/src/policy_prism_training/posttrain_entry.py`. Do not restore its obsolete `policy-prism-benchmark` dependency or its independent model, renderer, trainer, sampling, or Maven dataset code.

### Milestone 3 — Add the Policy Prism release package

Create these maintained files:

    packages/training/pyproject.toml
    packages/training/README.md
    packages/training/src/policy_prism_training/__init__.py
    packages/training/src/policy_prism_training/cli.py
    packages/training/src/policy_prism_training/scope_sft.py
    packages/training/src/policy_prism_training/posttrain_entry.py
    packages/training/tests/test_scope_sft.py

`scope_sft.py` must expose constants for the source hash/count, split/order seeds, model revision, tokenizer fingerprint, and release schema version. Its public interfaces are:

    def build_scope_sft_release(
        source: Path,
        output: Path,
        exclusions: Path,
        *,
        force: bool = False,
    ) -> Mapping[str, object]: ...

    def validate_scope_sft_release(
        dataset: Path,
        exclusions: Path,
        *,
        check_rendering: bool = True,
    ) -> Mapping[str, object]: ...

    def build_scope_train_rows() -> Iterable[Mapping[str, object]]: ...
    def build_scope_validation_rows() -> Iterable[Mapping[str, object]]: ...
    def build_scope_smoke_rows() -> Iterable[Mapping[str, object]]: ...

The last three callables load the already built release for local tests only. Remote smoke/full jobs use the pinned Hugging Face selections, not a built source.

The CLI exposes `build-scope-sft` with `--source`, `--output`, `--exclusions`, `--force`, and `--check`, plus `validate-scope-sft` with `--dataset`, `--exclusions`, and `--skip-rendering`. Both print a compact JSON result and exit nonzero on any contract violation.

`posttrain_entry.py` defines `VALIDATED_SFT_DEFINITION = "train/trl-sft-validated@1"`, constructs `sft_definition(..., with_validation=True)`, and passes it as an extra definition to `build_job_runtime(request, tracking="trackio", ...)`. This preserves the existing standard no-validation SFT definition for smoke and adds the validation seat for the full job.

Update the root `pyproject.toml` workspace, pytest paths, mypy package/path, and pack configuration. The final pack boundary is explicit:

    [tool.posttrain.pack]
    project_packages = ["packages/training", "packages/normative-verifiers"]
    source_includes = [
      "README.md",
      "pyproject.toml",
      "uv.lock",
      "packages/training",
      "packages/normative-verifiers",
    ]

Add `data/sft/` to `.gitignore`; the immutable Hub commit, local file hashes, and manifest carry release identity, not Git storage of the generated 369 MB-derived files. Update and commit `uv.lock`. Use the exact pushed PostTrain commit for direct PostTrain package pins; never pin a branch or `main`.

Tests must cover raw SHA/count and schema, canonical IDs, exact allowlisted output keys, byte-preserved JSON targets, stage-schema validity, 46-content deduplication and provenance, exact split counts/distributions, no leakage across instrument/task/content hashes, deterministic ordering, coverage, eight-row smoke selection including the longest row, prohibited-field absence, atomic/refuse-overwrite behavior, manifest/file hashes, and exact E4B renderer statistics. A fixture should contain deliberate duplicates and instruments from multiple families so the algorithm is tested independently of the 369 MB file; one marked integration test checks the frozen file when present.

### Milestone 4 — Build, audit, and publish the private dataset

Build the release from the clean Policy Prism checkout. Run its validator, the standalone sealed leakage validator, and `--check`. The three commands must agree on 4,499 unique rows, 4,279/220/8 outputs, no holdout leakage, and zero truncated supervised tokens at 49,152.

Create private `carbonteq/policy-prism-scope-sft-luna-pro-v1`, upload the five release files, resolve the resulting 40-character Hub commit, and download it into a fresh temporary directory. Verify that the fresh file hashes equal `manifest.json`. Only that resolved commit may appear in `.posttrain/catalog/datasets.yaml`.

### Milestone 5 — Add Policy Prism PostTrain selections and work packages

Create `.posttrain/project.toml` with schema version 2, project ID `policy-prism-scope-sft`, the `catalog` and `work_packages` paths, Trackio tracking, and entry `policy_prism_training.posttrain_entry:configure`. Create the layer and catalog files:

    .posttrain/catalog/layer.yaml
    .posttrain/catalog/datasets.yaml
    .posttrain/catalog/training.yaml
    .posttrain/catalog/targets.yaml
    .posttrain/catalog/environments.yaml
    .posttrain/catalog/evaluations.yaml
    .posttrain/catalog/models.yaml
    .posttrain/catalog/inference.yaml

The layer lists all seven YAML files. `models.yaml` and `inference.yaml` may initially contain empty top-level maps, but the files and layer entries are present so registering the reconciled adapter is one controlled catalog edit. Do not redefine the global E4B foundation model.

`datasets.yaml` defines three supervised `format.kind: messages` selections using the same exact repo and Hub commit with splits `train`, `validation`, and `smoke`:

    dataset:
      datasets/policy-prism-scope-train@1:
        revision: "1"
        kind: supervised
        source:
          kind: huggingface
          repo: carbonteq/policy-prism-scope-sft-luna-pro-v1
          revision: <40-character HF_DATASET_REVISION>
          split: train
        format:
          kind: messages

The validation and smoke entries differ only in selection ID and split.

`training.yaml` defines separate smoke/full SFT settings and one shared binding. The smoke uses `max_steps: 1`, `logging_steps: 1`, `checkpoint_steps: 1`, and no validation object. The full settings are exactly:

    max_steps: 535
    max_length: 49152
    per_device_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 0.00005
    warmup_ratio: 0.05
    max_grad_norm: 1.0
    logging_steps: 5
    checkpoint_steps: 100
    checkpoint_limit: 2
    seed: 20260728
    gradient_checkpointing: true
    validation:
      steps: 535
      per_device_batch_size: 1
      on_start: false
      at_end: true

The shared training binding uses backend `trl@1.8.0`, renderer family `gemma4` with `implementation: default` and `reasoning_mode: "off"`, the exact seven-projection regex, LoRA rank 32, alpha 64, dropout 0.05, target `targets/carbonteq-rtx-pro-6000-96gb`, and runtime global batch eight. Do not enable packing or shuffling; PostTrain already forces both off. BF16 and non-reentrant gradient checkpointing are translated by the TRL backend.

`targets.yaml` contains only the reachable workstation target:

    target:
      targets/carbonteq-rtx-pro-6000-96gb:
        revision: "1"
        device_class: nvidia-cuda
        memory_gb: 96
        placement:
          world_size: 1
          instances:
            - hostname: carbonteq-ai-workstation.lan

Create `gemma4_e4b_scope_smoke.yaml` with seats `model`, `dataset`, `settings`, and `training`, definition `train/trl-sft@1`, and bindings to the global E4B model, smoke dataset/settings, and shared training binding. Create `gemma4_e4b_scope_one_epoch.yaml` with the additional `validation_dataset` seat, definition `train/trl-sft-validated@1`, and bindings to train, validation, full settings, and the same model/training binding.

### Milestone 6 — Configure the current framework and local credentials

Create a dedicated Python 3.13 dstack environment with dstack 0.20.29. Initialize current PostTrain machine configuration once. If it already exists, inspect it with `posttrain machine show` and repair the protected files rather than rerunning `machine init`, which intentionally refuses to overwrite state.

The generated protected files are:

    ~/.config/posttrain/credentials/trackio.env
    ~/.config/posttrain/credentials/dstack.env
    ~/.config/posttrain/credentials/huggingface.env

Transfer `TRACKIO_WRITE_TOKEN`, `DSTACK_TOKEN`, and `DSTACK_SERVER_URL` from the Live Kit without printing them; provide a Hugging Face write token as `HF_TOKEN`. Put `OPENROUTER_API_KEY` in Policy Prism’s ignored mode-0600 `posttrain.env`. No secret belongs in Git, catalog YAML, a command argument value, or the plan.

Prove gated model and private dataset access locally with `hf auth whoami`, `hf download ... config.json` at the exact E4B revision, and all three `posttrain dataset validate` calls before reserving the GPU.

### Milestone 7 — Run and reconcile the GPU smoke

Plan then submit `policy-prism-e4b-sft-r32-v1-smoke` with a 7,200-second timeout. The remote package must identify the exact PostTrain source digest, clean Policy Prism source digest, pinned dataset commit, and RTX PRO 6000 hostname. Follow logs and Observatory, wait for a terminal state, then reconcile before inspecting success.

The gate for the full run is strict: `Gemma4ForConditionalGeneration` loads in BF16; the seven text projection families and no multimodal tower are trainable; all eight examples complete one finite optimizer update at accumulation eight; the longest record has zero truncated supervised tokens; Trackio is terminal-success; adapter, recovery checkpoint, native training summary, and expected roles are retained; reconciliation reports consistent evidence and no missing roles.

If the smoke OOMs specifically on activation memory, verify that the intended model, one-example microbatch, gradient checkpointing, and text-only LoRA targets were used. Retry once with `max_length: 46592`, which remains above the measured 46,381 maximum. Do not lower below 46,381, silently discard the longest record, switch to QLoRA, or launch full after an OOM. Any further change creates a new settings revision and smoke run ID.

### Milestone 8 — Run and reconcile the full SFT

Only after the smoke gate, plan and submit `policy-prism-e4b-sft-r32-v1` with a 86,400-second timeout. Estimate expected duration from the smoke’s stable optimizer-step time as `535 * step_time + final_validation + loading_and_artifact_time`; the timeout is a safety ceiling, not an expected duration.

Observe loss, learning rate, gradient norm/clipping, tokens per second, sequence-length/truncation metrics, GPU memory, checkpoints, and final validation loss. Wait for terminal success and reconcile. Record the exact Trackio project, artifact name, resolved `vN`, content digest, recovery artifact, training summary, package identity, final loss, validation loss, and peak memory. Do not use `latest`, do not clean the provider workspace, and do not begin evaluation from an unreconciled path.

### Milestone 9 — Register and evaluate the exact adapter

After reconciliation, add a project model variant with `form: peft-adapter`, artifact kind `trackio`, the exact project/name/`vN`, the pinned E4B Hub base, parent `models/gemma4-e4b-it@bf16`, family/capabilities/renderer/tokenizer fingerprint copied from the base, and provenance containing the full run ID, dataset Hub commit, PostTrain/Policy commits, LoRA settings, and reconciled content digest. Never use a Trackio alias.

Add one adapter inference binding with backend `vllm@0.25.1`, model equal to that adapter variant, renderer `gemma4-e4b-tools@1`, target the local workstation, startup timeout 1,800 seconds, and:

    max_model_len: 131072
    gpu_memory_utilization: 0.90
    dtype: bfloat16
    load_format: auto
    enforce_eager: true
    enable_chunked_prefill: true
    disable_log_stats: false
    max_num_seqs: 2
    kv_cache_dtype: auto
    text_only: true
    skip_mm_profiling: true
    max_lora_rank: 32
    sampling.max_tokens: 65536
    sampling.temperature: 0.0

Pin both environment bindings to one full pushed Policy Prism commit and subdirectory `packages/normative-verifiers`. Scope activates `policy_prism_normative_verifiers.factories:build_policy_prism_scope_sealed`, uses 18 tasks, one rollout, concurrency two, 16,384 max tokens, and temperature zero. Recovery activates `build_policy_prism_rule_recovery_sealed`, uses 17 tasks, one rollout, concurrency two, 65,536 max tokens, and temperature zero. One domain evaluation plan lists both environment IDs.

Create two work packages, one for scope and one for recovery. Each has seats `model`, `evaluation_inference`, `target`, `evaluation_plan`, and `environment`; uses `eval/verifiers-managed@1`; binds the same exact adapter/inference/target/evaluation plan; and binds only its own environment cell. Forward both `HF_TOKEN` and `OPENROUTER_API_KEY`. Run each with a 21,600-second timeout, reconcile it, and retain the native `verifiers-evaluation` artifact.

The existing base comparison runs are:

    gemma-4-e4b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-scope-20260803
    gemma-4-e4b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-recovery-20260803

Download each exact evaluation artifact and finalize it through Policy Prism. The adapter report is paired with the existing base report only if its compatibility hash is exactly:

    scope: 3fe4471b643111e257ffce89cad32612bcb8f5c8e238e207ba48d807b99cef61
    recovery: 724117553feab78c833fe379df7f3e5cd7bba947e81d1b02fb46992ed528ce94

If one hash differs, rerun only that base cell through the same current work-package contract before comparing. Experiment execution succeeds when comparable native reports exist; model quality is a separate outcome. The initial quality target is schema compliance remaining 18/18 with gains over prior strict conformance 1/18, qualifications 0/53, and relationships 20/122, while inspecting rule fields, grounding, and source-family slices for regressions.

### Milestone 10 — Publish and verify the adapter, then clean up

Materialize the exact reconciled `model-adapter` Trackio artifact, verify `adapter_config.json` and at least one `.safetensors` file, and compute a deterministic tree hash equal to the reconciled digest. Write a model card containing the base revision, Trackio project/name/`vN`/digest, dataset repo/commit, both source commits, complete training configuration, run ID, two finalized evaluation IDs, compatibility hashes, metrics, license linkage, and `teacher_generated_unverified` limitation.

Upload only that directory and model card to private `carbonteq/gemma-4-e4b-policy-prism-scope-sft-lora-v1`. Resolve the model Hub commit, download it to a fresh temporary directory, and recheck the files and digest. Only then reconcile once more if necessary and clean smoke, full, scope-eval, and recovery-eval provider workspaces. Trackio, Hugging Face, finalized Policy Prism runs, and Observatory lineage remain durable after provider cleanup.

## Concrete Steps

Define paths in each shell without changing system variables:

    export POSTTRAIN_ROOT=/home/ali-awais-safdar/Post-Train/posttrain
    export POLICY_ROOT='/home/ali-awais-safdar/Policy Prism'
    export KIT_ROOT=/home/ali-awais-safdar/Post-Train/posttrain-setup-v0.2.2-20260728/posttrain-setup
    export UV_CACHE_DIR=/tmp/uv-cache

### PostTrain implementation validation

From `POSTTRAIN_ROOT`, run focused validation:

    cd "$POSTTRAIN_ROOT"
    uv sync --all-packages --locked --python 3.13
    uv run pytest \
      packages/common/tests/test_model_variants.py \
      packages/common/tests/test_model_chat_templates.py \
      packages/catalog/tests/test_files.py \
      packages/train/tests/test_trl_common.py \
      packages/serve/tests/test_online.py \
      packages/serve/tests/test_vllm_bindings.py \
      apps/lab/tests/test_catalog.py -q

Then run the repository ladder:

    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Record the eventual clean pushed commit:

    git rev-parse HEAD

### Policy Prism baseline and package validation

First make the existing Policy work green without mixing in SFT changes:

    cd "$POLICY_ROOT"
    uv sync --all-packages --locked
    uv run ruff check .
    uv run mypy
    uv run pytest
    git diff --check

After that state is committed and pushed, implement Milestones 3–5, update the lock, and rerun:

    uv lock
    uv sync --all-packages --locked
    uv run --package policy-prism-training pytest packages/training/tests -q
    uv run ruff check .
    uv run mypy
    uv run pytest
    git diff --check

### Dataset release, audit, and publication

Load `HF_TOKEN` securely into this shell without printing it; if current machine configuration has already been initialized, the protected Hugging Face credential file is the preferred source. Then build and validate:

    cd "$POLICY_ROOT"
    uv run --package policy-prism-training policy-prism-training build-scope-sft \
      --source "$POLICY_ROOT/trace-generation-runs/gpt-luna-pro-scope-5000-v1/accepted-stage-records.jsonl" \
      --output "$POLICY_ROOT/data/sft/policy-prism-scope-luna-pro-v1" \
      --exclusions "$POLICY_ROOT/data/benchmarks/normative-gold-v2/training-exclusions.json"

    uv run --package policy-prism-training policy-prism-training validate-scope-sft \
      --dataset "$POLICY_ROOT/data/sft/policy-prism-scope-luna-pro-v1" \
      --exclusions "$POLICY_ROOT/data/benchmarks/normative-gold-v2/training-exclusions.json"

    uv run --package policy-prism-training policy-prism-training build-scope-sft \
      --source "$POLICY_ROOT/trace-generation-runs/gpt-luna-pro-scope-5000-v1/accepted-stage-records.jsonl" \
      --output "$POLICY_ROOT/data/sft/policy-prism-scope-luna-pro-v1" \
      --exclusions "$POLICY_ROOT/data/benchmarks/normative-gold-v2/training-exclusions.json" \
      --check

    uv run --package policy-prism-normative-verifiers policy-prism-verifiers validate-leakage \
      --manifest data/benchmarks/normative-gold-v2/training-exclusions.json \
      --training-file data/sft/policy-prism-scope-luna-pro-v1/train.jsonl \
      --training-file data/sft/policy-prism-scope-luna-pro-v1/validation.jsonl

Publish and resolve the exact dataset revision from the PostTrain environment, which contains the pinned Hugging Face CLI:

    cd "$POSTTRAIN_ROOT"
    export HF_DATASET_REPO=carbonteq/policy-prism-scope-sft-luna-pro-v1
    export DATASET_DIR="$POLICY_ROOT/data/sft/policy-prism-scope-luna-pro-v1"

    uv run --package posttrain-train hf auth whoami
    uv run --package posttrain-train hf repos create "$HF_DATASET_REPO" \
      --repo-type dataset --private --exist-ok
    uv run --package posttrain-train hf upload "$HF_DATASET_REPO" \
      "$DATASET_DIR" . --repo-type dataset --private \
      --commit-message 'Publish deduplicated Policy Prism scope SFT v1'

    export HF_DATASET_REVISION="$(uv run --package posttrain-train python -c \
      'import os; from huggingface_hub import HfApi; print(HfApi().dataset_info(os.environ["HF_DATASET_REPO"]).sha)')"
    printf '%s\n' "$HF_DATASET_REVISION"

Pin that printed 40-character value into all three dataset selections. Commit and push Policy Prism again; record its full commit for the later Verifiers environment sources.

### Machine initialization and local preflight

From PostTrain:

    cd "$POSTTRAIN_ROOT"
    uv venv --python 3.13 .dstack-venv
    uv pip install --python .dstack-venv/bin/python --system-certs \
      --index-url https://pypi.lan/carbonteq/stable/+simple/ 'dstack==0.20.29'

If current machine configuration is absent, initialize it once:

    uv run --package posttrain posttrain machine init \
      --project "$POLICY_ROOT" \
      --default-provider dstack \
      --trackio-endpoint https://trackio.lan \
      --python-index-url https://pypi.lan/carbonteq/stable/+simple/ \
      --job-registry registry.lan/carbonteq \
      --dstack-project main \
      --dstack-python "$POSTTRAIN_ROOT/.dstack-venv/bin/python"

If it exists, inspect instead:

    uv run --package posttrain posttrain machine show

Populate the three protected credential files from the kit and the separate HF token without echoing values, then ensure directories/files are mode 0700/0600. Run:

    set -a
    . ~/.config/posttrain/credentials/huggingface.env
    set +a
    uv run --package posttrain-train hf auth whoami
    uv run --package posttrain-train hf download google/gemma-4-E4B-it config.json \
      --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
      --local-dir /tmp/policy-prism-e4b-model-preflight

Run all PostTrain preflights:

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" doctor
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" catalog validate
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" runtime images verify

    for dataset in \
      datasets/policy-prism-scope-train@1 \
      datasets/policy-prism-scope-validation@1 \
      datasets/policy-prism-scope-smoke@1
    do
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        dataset validate "$dataset"
    done

Start Observatory in a separate terminal and leave it running:

    cd "$POSTTRAIN_ROOT"
    UV_CACHE_DIR=/tmp/uv-cache uv run --package posttrain posttrain \
      --project-root "$POLICY_ROOT" observatory up --host 127.0.0.1 --port 7861

Open `http://127.0.0.1:7861`.

### Smoke submission and reconciliation

In the control terminal:

    cd "$POSTTRAIN_ROOT"
    export TARGET=targets/carbonteq-rtx-pro-6000-96gb
    export SMOKE_RUN=policy-prism-e4b-sft-r32-v1-smoke
    export SMOKE_WP=.posttrain/work_packages/gemma4_e4b_scope_smoke.yaml

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      job plan "$SMOKE_WP" --job train --provider dstack --target "$TARGET" \
      --env HF_TOKEN --timeout-seconds 7200 --run-id "$SMOKE_RUN"

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      job run "$SMOKE_WP" --job train --provider dstack --target "$TARGET" \
      --env HF_TOKEN --timeout-seconds 7200 --run-id "$SMOKE_RUN" --build-missing

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      run logs "$SMOKE_RUN" --follow
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      run wait "$SMOKE_RUN" --timeout-seconds 7200
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      run reconcile "$SMOKE_RUN"
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      --json run show "$SMOKE_RUN"

Do not proceed until every Milestone 7 gate is visible in the JSON, Trackio, logs, and Observatory.

### Full SFT submission and reconciliation

    export FULL_RUN=policy-prism-e4b-sft-r32-v1
    export FULL_WP=.posttrain/work_packages/gemma4_e4b_scope_one_epoch.yaml

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      job plan "$FULL_WP" --job train --provider dstack --target "$TARGET" \
      --env HF_TOKEN --timeout-seconds 86400 --run-id "$FULL_RUN"

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      job run "$FULL_WP" --job train --provider dstack --target "$TARGET" \
      --env HF_TOKEN --timeout-seconds 86400 --run-id "$FULL_RUN" --build-missing

    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      run logs "$FULL_RUN" --follow
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      run wait "$FULL_RUN" --timeout-seconds 86400
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      run reconcile "$FULL_RUN"
    uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
      --json run show "$FULL_RUN"

Stop here if reconciliation is inconsistent or any expected artifact role is missing. Resolve the exact adapter Trackio reference from the JSON and put it into `models.yaml`; do not guess it from the run name.

### Adapter evaluation

After committing and pushing the exact adapter catalog/work-package changes:

    export SCOPE_EVAL_RUN=policy-prism-e4b-sft-r32-v1-scope
    export RECOVERY_EVAL_RUN=policy-prism-e4b-sft-r32-v1-recovery
    export SCOPE_EVAL_WP=.posttrain/work_packages/gemma4_e4b_scope_adapter_eval.yaml
    export RECOVERY_EVAL_WP=.posttrain/work_packages/gemma4_e4b_recovery_adapter_eval.yaml

    for pair in \
      "$SCOPE_EVAL_WP:$SCOPE_EVAL_RUN" \
      "$RECOVERY_EVAL_WP:$RECOVERY_EVAL_RUN"
    do
      wp=${pair%%:*}
      run=${pair#*:}
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        job plan "$wp" --job evaluate --provider dstack --target "$TARGET" \
        --env HF_TOKEN --env OPENROUTER_API_KEY --timeout-seconds 21600 --run-id "$run"
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        job run "$wp" --job evaluate --provider dstack --target "$TARGET" \
        --env HF_TOKEN --env OPENROUTER_API_KEY --timeout-seconds 21600 \
        --run-id "$run" --build-missing
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        run logs "$run" --follow
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        run wait "$run" --timeout-seconds 21600
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        run reconcile "$run"
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        --json run show "$run"
    done

Materialize each exact reconciled `verifiers-evaluation` artifact with a small tested Trackio helper. The helper initializes a named audit run against `POSTTRAIN_TRACKIO_SERVER_URL`, calls `use_artifact("<name>:<vN>", type="verifiers-evaluation")`, downloads to a new destination, and finishes the audit run. It must reject aliases and existing destinations. Each downloaded native directory must contain `config.toml` and `traces.jsonl`.

Create secret-free serving metadata JSON for each cell, then finalize from Policy Prism:

    cd "$POLICY_ROOT"
    uv run --package policy-prism-normative-verifiers policy-prism-verifiers finalize-run \
      --input <downloaded-scope-native-directory> \
      --run-id "$SCOPE_EVAL_RUN" \
      --serving-metadata <scope-adapter-serving-metadata.json> \
      --output-root evaluation-runs

    uv run --package policy-prism-normative-verifiers policy-prism-verifiers finalize-run \
      --input <downloaded-recovery-native-directory> \
      --run-id "$RECOVERY_EVAL_RUN" \
      --serving-metadata <recovery-adapter-serving-metadata.json> \
      --output-root evaluation-runs

    uv run --package policy-prism-normative-verifiers policy-prism-verifiers validate-runs \
      --root evaluation-runs

The angle-bracket paths are not discretionary choices: copy the exact artifact paths printed by the materialization helper. Finalization output supplies the report directories and compatibility hashes used for comparison.

### Adapter publication and provider cleanup

Materialize the exact Trackio `model-adapter` artifact by project/name/`vN`, not alias, into a new directory. Verify it:

    export HF_MODEL_REPO=carbonteq/gemma-4-e4b-policy-prism-scope-sft-lora-v1
    export ADAPTER_DIR=<downloaded-exact-model-adapter-directory>
    test -f "$ADAPTER_DIR/adapter_config.json"
    find "$ADAPTER_DIR" -name '*.safetensors' -type f

After adding the required model card, publish privately:

    cd "$POSTTRAIN_ROOT"
    uv run --package posttrain-train hf repos create "$HF_MODEL_REPO" \
      --repo-type model --private --exist-ok
    uv run --package posttrain-train hf upload "$HF_MODEL_REPO" \
      "$ADAPTER_DIR" . --repo-type model --private \
      --commit-message 'Publish Policy Prism Gemma 4 E4B rank-32 SFT adapter'

    export HF_MODEL_REVISION="$(uv run --package posttrain-train python -c \
      'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["HF_MODEL_REPO"]).sha)')"
    uv run --package posttrain-train hf download "$HF_MODEL_REPO" \
      --revision "$HF_MODEL_REVISION" \
      --local-dir /tmp/policy-prism-e4b-adapter-hf-verification

Compare the fresh files/digest with Trackio. Then clean only the four provider workspaces:

    for run in "$SMOKE_RUN" "$FULL_RUN" "$SCOPE_EVAL_RUN" "$RECOVERY_EVAL_RUN"
    do
      uv run --package posttrain posttrain --project-root "$POLICY_ROOT" \
        run cleanup "$run"
    done

## Validation and Acceptance

PostTrain implementation is accepted when focused and full tests, Ruff check/format, Pyright, import boundaries, and `git diff --check` pass; the exact E4B tokenizer/template test is not skipped in the release environment; ordinary Qwen/LFM loaders remain unchanged; unsafe Gemma target patterns fail; and rank 32 appears in generated vLLM arguments only when configured.

The dataset release is accepted when raw hash/count are exact; 46 duplicate rows are audibly removed into provenance; train/validation/smoke counts are 4,279/220/8; all stated distributions and coverage checks pass; train/validation have no instrument, task, or content leakage; sealed leakage is zero; output contains no prohibited fields; every assistant target is valid stage JSON and byte-preserved; exact rendering has zero empty supervision and zero truncation at 49,152; `--check` reproduces identical bytes; and a clean Hub download matches manifest hashes.

The full SFT is allowed only after the smoke satisfies every Milestone 7 gate. The full run is accepted as an experiment when 535 finite optimizer steps and final validation complete, Trackio and provider states are terminal-success, reconciliation is consistent with no missing roles, and exact adapter/recovery/summary artifacts are retained. A disappointing model score does not invalidate a correctly run experiment.

Qualification is accepted when both adapter cells complete, reconcile, finalize into native Policy Prism runs, and match the corresponding base compatibility hashes or trigger the specified base rerun. Publication is accepted only when the exact PEFT adapter exists in Trackio and private Hugging Face at immutable versions, a fresh Hub download matches it, the dataset remains privately pinned, all lineage fields are recorded, and provider cleanup succeeds without deleting durable evidence.

## Idempotence and Recovery

Release building is deterministic and writes atomically. `--check` is read-only; an existing output is protected unless `--force` is explicit. Never edit a published dataset revision: any source, transform, split, dedup, or schema change requires a new release ID and Hub commit.

Job planning, dataset validation, logs, wait, show, and reconciliation are safe to repeat. If submission returns an ambiguous error, inspect the exact run first and use `posttrain run retry-submit <run-id>` rather than creating a duplicate. If cancellation leaves tracking open, run `posttrain run recover-cancelled-tracking <run-id>` and reconcile. After OOM, non-finite loss, code/config changes, or changed package identity, use a new settings revision and run ID; never overwrite evidence under the original ID.

Checkpoints are recovery evidence, not permission to mutate the original run. Resume only when PostTrain reports a compatible exact recovery artifact and unchanged package/settings/dataset identities. Reconcile before every cleanup. Trackio/Hugging Face downloads go to new empty directories and exact `vN`/commit values so reruns cannot silently replace local evidence.

## Artifacts and Notes

Maintain this record as execution proceeds:

    Raw input SHA / rows: 0c53a9884e2abbfb01abb84a4be8ea835118dce6d4d9433039e9298f4c87651e / 4545
    Transform version / duplicate groups removed:
    Train / validation / smoke file SHA and rows:
    Leakage manifest SHA and result:
    Dataset Hub repo / commit:
    PostTrain commit / framework package digest:
    Policy Prism commit / project package digest:
    Smoke run ID / host / runtime / peak VRAM / adapter reference:
    Full run ID / runtime / final train loss / validation loss / peak VRAM:
    Trackio adapter project / name / vN / digest:
    Scope adapter run / finalized report / compatibility SHA:
    Recovery adapter run / finalized report / compatibility SHA:
    Base report IDs used or rerun IDs:
    Adapter Hub repo / commit / verification digest:
    Provider cleanup results:
    Quality conclusion / next experiment:

## Interfaces and Dependencies

Policy Prism’s maintained public surface is `policy_prism_training.scope_sft.build_scope_sft_release`, `validate_scope_sft_release`, the three row builders, the `policy-prism-training` CLI, and `policy_prism_training.posttrain_entry:configure`. It depends on its local normative-verifiers package for the sealed exclusion contract and on exact PostTrain package commits for canonical data/rendering/job contracts. It does not own a tokenizer chat template, trainer, model loader, tracking client abstraction, or universal SFT schema registry.

PostTrain’s relevant stable surfaces are `DatasetLoadPlan` with Hugging Face `messages` format, `SupervisedExample`, `TrainingRenderer`, `SFTSettings`, `TrainingBinding`, `train/trl-sft@1`, the added `train/trl-sft-validated@1` project definition, `VllmEngineConfig`, Trackio artifact references, dstack execution, reconciliation, managed Verifiers evaluation, and Observatory. The external pinned services are Hugging Face Hub, Trackio at `https://trackio.lan`, registry at `registry.lan/carbonteq`, dstack project `main`, the in-house RTX PRO 6000 target, Prime Intellect Verifiers commit already pinned by Policy Prism, and OpenRouter for the sealed semantic judge.

Plan revision note (2026-08-04): consolidated the original launch plan and the later end-to-end audit into one execution authority; integrated teammate PR #12 as the base; recorded completed E4B and rank-32 framework work; separated E4B's renderer identity after a live pinned-tokenizer probe disproved exact Unified 12B template equivalence; corrected the dataset from 4,545 raw/4,321 train/541 steps to a deterministically deduplicated 4,499 unique/4,279 train/535-step release; made private dataset publication a pre-smoke reproducibility gate; separated no-validation smoke from final validation; reused existing compatible base evaluations; and added exact reconciliation, native-evidence finalization, adapter publication, recovery, and cleanup procedures.
