# Qualify and run Policy Prism Gemma 4 E2B from 12B OPD

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated whenever work advances. Maintain it according to `docs/templates/PLAN.md` and the frozen product baseline under `docs/post-training/`.

## Purpose / Big Picture

This plan delivers a reproducible Policy Prism on-policy distillation (OPD) experiment on the in-house RTX PRO 6000. Gemma 4 E2B is the trainable student and Gemma 4 12B is the frozen teacher. The student generates legal-interpretation responses, the teacher scores the exact generated token IDs, and a sparse completion-only reverse-KL loss updates a rank-16 LoRA adapter.

The previous attempts produced a valid intermediate checkpoint at optimizer step 96 but did not complete training. This plan preserves that checkpoint, qualifies batch sizes 1, 2, and 4 without multi-token prediction (MTP), then starts one fresh full run with the fastest stable configuration. Completion means that all 384 logical targets were optimized once, the final adapter and checkpoints are retained in Trackio, the adapter is privately published and freshly verified on Hugging Face, sealed scope and recovery evaluations succeed, and both evaluations are finalized in Policy Prism's normal five-file `evaluation-runs` format.

This plan does not change either `main` branch. All PostTrain commits use `feat/gemma-policy-prism-opd-e2b-12b`; all Policy Prism commits use `feat/scope-opd-e2b-12b-environment-v1`. Failed attempts and their Trackio evidence remain retained.

## Progress

- [x] (2026-08-07) Build and package the deterministic Policy Prism production plan containing 384 primary targets and 96 shared reserves, with sealed-family exclusion, exact target distributions, source review, and E2B token-budget validation.
- [x] (2026-08-07) Implement exact-token E2B-from-12B OPD, memory-safe sparse loss, structured XGrammar projection, stable target projection, Trackio checkpoints, and explicit checkpoint resume on the PostTrain feature branch.
- [x] (2026-08-08) Complete 66 finite updates in `opda06-e2b12b-r16-scope384-v1`; retain checkpoints through step 64 before an external dstack workstation-reachability failure.
- [x] (2026-08-10) Prove exact step-64 restore in `opda08rs-e2b12b-r16-resume64-scope384-v1`; diagnose its target-78 failure as overly narrow prompt-profile reserve matching.
- [x] (2026-08-10) Push Policy Prism plan-v2 reserve semantics in commits `3976d993a2e96dd4c91c3ad91481767299755364` and `47ee4f43366e3a1a68fa59db8ed3ab46613c677c`, allowing shared stage/quartile/decision reserves to inherit the logical target's profile and shape.
- [x] (2026-08-10) Resume from step 64 as `opda09rs-e2b12b-r16-resume64-scope384-v2`, pass target 78, retain checkpoints at steps 80 and 96, and diagnose target 98 before optimizer step 99.
- [x] (2026-08-10) Audit r9 attempt traces: ten of eleven reserve-triggering structural failures were exact `source_provision_id` copying mistakes; one was malformed long JSON.
- [x] (2026-08-10) Constrain each rules-stage wire schema to the exact task-owned `source_provision_id`, preserve canonical downstream validation, pass 20 focused Policy Prism tests, Ruff, and strict mypy, then push Policy Prism commit `147ac75997579f08154145ea9bdc6215b4aa7ec4`.
- [x] (2026-08-10) Include and push the user's local Policy Prism commit `85e0e12d102ea8e32e1b31fe9926179b21cbe2fb`; remote feature-branch tip `147ac759...` is its direct descendant.
- [x] (2026-08-10) Verify checkpoint 96 contains adapter, optimizer, scheduler, trainer, RNG, tokenizer, and SQLite allocation-ledger state; all inspected losses and gradients are finite and teacher failures are zero.
- [x] (2026-08-10) Publish checkpoint 96 privately to `carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-checkpoint-96`, fresh-download it, verify every uploaded file byte-for-byte, and resolve immutable Hugging Face revision `4f1fe9c75031396a11bcc44e2193f96df9003054`.
- [x] (2026-08-10) Cancel the failed r9 provider workspace after checkpoint preservation. It is terminal with provider state `terminated`; reconciliation records failed/inconsistent evidence and retains three training artifacts plus native rollout traces.
- [x] (2026-08-10) Stop before new GPU submissions and update this plan for a separate batch-qualification and full-run execution goal.
- [ ] Implement and offline-qualify narrowly bounded prompt batches 1, 2, and 4 in PostTrain; do not enable MTP.
- [ ] Commit Policy Prism's three deterministic four-target smoke selections and work packages pinned to source commit `147ac759...`.
- [ ] Run batch-1, batch-2, and batch-4 smokes sequentially, record scientific/runtime gates and normalized GPU evidence, and select the fastest stable configuration.
- [ ] Build and push one fresh 384-target work package using the selected batch and target-normalized schedule.
- [ ] Run the complete OPD experiment from the base E2B model, retain all milestones, and reconcile it consistently.
- [ ] Publish and freshly verify the final private LoRA adapter on Hugging Face.
- [ ] Run sealed 18-case scope and 17-case recovery evaluation jobs sequentially and apply their scientific gates.
- [ ] Finalize both native evaluation artifacts into Policy Prism `evaluation-runs`, validate them, and push the final evidence and living-plan updates.

## Surprises & Discoveries

- Observation: the step-78 and target-98 failures had different causes.
  Evidence: r8 found only one reserve because prompt profile was incorrectly part of reserve compatibility. Plan v2 removed that key and r9 passed target 78. At target 98, r9 had the primary plus two compatible reserves, but the model repeatedly misspelled task-owned source identifiers that the schema had left unconstrained.
- Observation: the dominant target-98 failure was preventable during generation without weakening scientific admission.
  Evidence: ten of eleven inspected structural rejections used a wrong `source_provision_id`, such as omitting a segment from an eCFR identifier. Policy Prism commit `147ac759...` adds a task-specific enum to the deep-copied rules schema. The canonical schema and post-generation admission checks remain unchanged.
- Observation: checkpoint 96 is healthy even though its producing run is not successful.
  Evidence: Trackio artifact `training-models-gemma4-e2b-it-bf16-distill-checkpoint-step-0096:v0` has provider digest `ddde25b171407c76fba02e1b44b23a6531bcede4bd99c7859cc9809496e7f7e7`, PostTrain tree digest `cc53e8330d2af3160d098341dece1c82cf06a559366950f689c72caf24f16dba`, global step 96, and complete resumable state. R9 itself failed before update 99 and was later cancelled to release the GPU.
- Observation: r9's reconciliation is correctly inconsistent rather than a checkpoint-integrity failure.
  Evidence: the provider was explicitly cancelled after the harness raised, while Trackio retained a failed outcome. PostTrain reports `provider cancelled but retained tracking outcome is failed`, no missing artifact role, and retained step-80, step-96, and rollout-trace outputs.
- Observation: batch size one was a branch-local safety guard, not a universal PostTrain constraint.
  Evidence: `_validate_memory_safe_sparse_request` in `packages/train/src/posttrain/train/backends/trl/distillation.py` was deliberately limited to one prompt, one generation, batch one, and accumulation one because no live multi-prompt sparse long-context configuration had been qualified and earlier instructions prohibited smoke tests.
- Observation: the RTX PRO 6000 was underutilized mainly during serial autoregressive student generation and dependency stages.
  Evidence: observed utilization was commonly about 20–22% while approximately 65 GiB of 96 GiB VRAM was allocated. Teacher scoring was usually sub-second; long student generations dominated wall time. More prompt concurrency may fill decode bubbles, but larger training batches also increase peak activation memory and must be measured live.
- Observation: loss variation alone does not show collapse or prove legal improvement.
  Evidence: all inspected losses and gradient norms through step 97 were finite, teacher failures were zero, and isolated spikes recovered immediately under gradient clipping. The median loss over steps 81–96 was lower than earlier windows, which is encouraging, but only sealed evaluations can establish legal quality.
- Observation: the measured batch-one rate is materially slower than the original estimate.
  Evidence: r9 advanced roughly 34 optimizer updates in about 2.5 hours including long heterogeneous generations, implying approximately 27–30 hours for a fresh 384-update batch-one run. Batch-two and batch-four duration must be derived from identical live smokes rather than extrapolated from H200 or E4B/31B runs.
- Observation: changing the optimizer batch invalidates exact continuation from checkpoint 96 for the production comparison.
  Evidence: a batch-two or batch-four run changes targets per optimizer update, scheduler steps, checkpoint step numbers, and sampler advancement. Resuming checkpoint 96 under that schedule would mix two optimization regimes. The checkpoint remains valuable evidence, but the final selected configuration starts from the frozen base E2B model.

## Decision Log

- Decision: preserve all work on the two feature branches and never modify or push either `main` branch.
  Rationale: this isolates the experiment and retains reproducible review boundaries.
  Date/Author: 2026-08-10 / Codex.
- Decision: publish checkpoint 96 as an explicitly intermediate private repository rather than call it the final model.
  Rationale: the weights are valid and useful, but only 96 of 384 targets were optimized and the producing run failed later.
  Date/Author: 2026-08-10 / Codex.
- Decision: keep r0-r9, checkpoints, Trackio traces, and provider evidence; do not delete or relabel failed attempts.
  Rationale: they explain each corrected boundary and are required for honest incident history.
  Date/Author: 2026-08-10 / Codex.
- Decision: fix target-owned identifier copying in Policy Prism's generated rules schema, not by sanitizing output or loosening admission.
  Rationale: the grammar can require the only scientifically valid identifier while canonical validation still evaluates the exact generated response.
  Date/Author: 2026-08-10 / Codex.
- Decision: qualify only prompt batches 1, 2, and 4, with one generation per prompt, accumulation one, and no MTP.
  Rationale: these bounded values test useful concurrency within the available VRAM while keeping the existing exact-token objective and avoiding another unqualified acceleration mechanism.
  Date/Author: 2026-08-10 / Codex.
- Decision: use the same four deterministic tasks for all smokes: indices 98, 151, 284, and 367 selected by seed 539.
  Rationale: the set contains the exact former target-98 boundary plus q4 rules, q1 evidence, and a q4 full graph trajectory. It exercises replacement-sensitive, short, long, standalone, and multi-stage paths without changing production data.
  Date/Author: 2026-08-10 / Codex.
- Decision: choose the fastest scientifically valid smoke by wall seconds per admitted target; use smaller batch when valid candidates differ by less than 10%.
  Rationale: throughput is the user goal, while the tie rule retains memory headroom and reduces long-context out-of-memory risk. GPU utilization is supporting evidence, not the primary objective.
  Date/Author: 2026-08-10 / Codex.
- Decision: run the final 384-target experiment from the frozen base E2B model after selecting the batch.
  Rationale: one consistent batch, learning-rate schedule, sampler, and ledger produces cleaner scientific lineage than mixing checkpoint-96 batch-one history with a new batch.
  Date/Author: 2026-08-10 / Codex.
- Decision: stop the current goal after checkpoint preservation and this launch-ready plan; submit no new GPU job until a new execution goal begins.
  Rationale: the revised user objective explicitly separates planning from the next experiment.
  Date/Author: 2026-08-10 / Codex.

## Outcomes & Retrospective

Checkpoint 96 is now a complete, independently downloadable intermediate result rather than an artifact trapped inside a failed run. Its private Hugging Face repository and immutable revision are verified. The previous run did not finish because reserve matching first over-fragmented candidates at target 78, and then the model was allowed to invent an exact source identifier at target 98. Both causes now have narrow, tested corrections: broader scientifically compatible reserve sharing and a task-specific schema enum.

No claim is made that checkpoint 96 is a completed or qualified model. No new smoke or production job has been submitted under this plan revision. The next execution starts with bounded live capacity qualification, then one fresh full run and normal sealed qualification.

## Context and Orientation

PostTrain is `/home/ali-awais-safdar/Post-Train/posttrain`. It owns model profiles, OPD request contracts, the TRL runtime, sparse loss, packaging, dstack execution, Trackio evidence, and Observatory. Policy Prism is `/home/ali-awais-safdar/Policy Prism`. It owns the source-only legal task plan, prompts, schemas, admission, replacement ledger, project catalog/work packages, and final five-file evaluation format.

OPD means that the current student generates a response and the frozen teacher scores the same token IDs. The teacher does not generate a replacement answer. A logical target is one selected evidence, rules, or graph output that receives loss. A reserve is a reviewed alternate source candidate used only when the primary attempt is structurally unusable. Schema-valid legal mistakes remain trainable; malformed JSON, truncation, unknown identifiers, and unusable dependencies are rejected.

The production plan has 384 targets: 77 evidence, 230 rules, and 77 graph. It has 96 shared reserve candidates. Plan v2 matches reserves by target stage, source-length quartile, and reviewed decision class; the logical target supplies prompt profile and task shape. The target-98 rules schema now enumerates the exact source identifier, so XGrammar cannot generate the historically dominant mismatch.

Immutable model inputs are:

    Student: google/gemma-4-E2B-it
    Student revision: 3e22461f65e89153144f8adb70e3b8c2cc9845a7
    Teacher: google/gemma-4-12B-it
    Teacher revision: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    Canonical token-ID fingerprint: 059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416
    GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GiB

The known-good OPD hyperparameters remain rank 16, alpha 32, dropout 0, learning rate `1e-5` at batch one, maximum prompt 32,768 tokens, per-call sequence cap 40,960, trainer maximum length 49,152, gradient clipping 1.0, gradient checkpointing enabled, and one student generation per prompt. MTP remains disabled.

## Plan of Work

### Milestone 1: freeze branch and checkpoint lineage

Fast-forward the ordinary Policy Prism feature checkout from `85e0e12...` to its already-pushed descendant `147ac759...`; do not merge or rebase `main`. Verify the PostTrain feature branch remains at the expected remote tip before editing. Record checkpoint 96's Trackio provider digest, PostTrain content digest, Hugging Face repository, immutable revision, and fresh-download digest in this plan. This milestone is accepted when both feature worktrees are clean, the Policy Prism source-ID regression passes, and the private HF repository resolves to `4f1fe9c...`.

### Milestone 2: add bounded batch qualification

In `packages/train/src/posttrain/train/backends/trl/distillation.py`, change only the memory-safe E2B sparse-OPD guard so it accepts prompt/device batches 1, 2, or 4, requires one generation per prompt, requires device batch to equal prompts per update, and keeps accumulation at one. Do not remove the E2B model, teacher mode, sparse loss, LoRA, or backend qualification checks. Do not add MTP or speculative decoding.

In `packages/train/tests/test_api.py`, parameterize the positive guard test for 1/2/4 and add failures for multiple generations, unsupported batch values, mismatched prompt/device batch, and accumulation above one. In `packages/train/tests/test_trl_sparse_distillation.py`, prove loss and gradients match the full-logit reference for batch 1/2/4. Add a batching test around the Verifiers bridge and TRL generator showing that four distinct selected task identities remain ordered and exact when gathered concurrently.

In Policy Prism `.posttrain/catalog/environments.yaml`, add three environment selections pinned to source commit `147ac759...`. Each selects four tasks with `sampling_seed: 539`; their `max_concurrent` values are 1, 2, and 4. The framework's deterministic selector must resolve indices `[98, 151, 284, 367]`. Add matching training settings, training bindings, rollout bindings, and three work packages:

    gemma4_e2b_scope_opd_from_12b_smoke_b1.yaml
    gemma4_e2b_scope_opd_from_12b_smoke_b2.yaml
    gemma4_e2b_scope_opd_from_12b_smoke_b4.yaml

The b1 package uses four optimizer updates with one prompt each. B2 uses two updates with two prompts each. B4 uses one update with four prompts. Every package therefore exposes the same four logical targets once. All use fresh base E2B weights and isolated ledger paths.

### Milestone 3: package and run three sequential smokes

Validate and package all three jobs before reserving the GPU. Submit b1, wait and reconcile it, then b2, then b4. Never queue them concurrently on the single workstation. Use distinguishable Trackio prefixes:

    opdsm1a1-e2b12b-r16-targets4-v1
    opdsm2a1-e2b12b-r16-targets4-v1
    opdsm4a1-e2b12b-r16-targets4-v1

A smoke is scientifically valid only when all four targets are admitted, every optimized loss and gradient norm is finite, scored-token count is positive, teacher failures are zero, no output ends in truncation, target 98 has no source-ID mismatch, native trace sync completes, and its final checkpoint is retained. An out-of-memory event, provider error, reserve exhaustion, missing target, NaN/Inf, or missing artifact rejects that batch.

For each valid smoke, calculate wall seconds per admitted target from recorded start/finish timestamps. Record median and peak `system/gpu_utilization`, peak `system/gpu_vram_used_bytes`, student generation throughput, teacher latency, scored tokens, replacement count, and target-level durations from Trackio/Observatory. Select the smallest batch within 10% of the fastest valid seconds-per-target result. If b2 and b4 fail, b1 is the proven fallback; the full run must not proceed if b1 fails under the corrected schema.

### Milestone 4: freeze the final 384-target configuration

Create a new production settings/binding/work-package revision only after the smoke winner is known. One epoch always means 384 distinct optimized targets. Therefore the optimizer schedule is batch-dependent:

| Winner | Prompts/device batch | Optimizer updates | Warmup updates | LR | Milestone steps |
| --- | ---: | ---: | ---: | ---: | --- |
| b1 | 1 | 384 | 20 | `1e-5` | 96, 192, 288, 384 |
| b2 | 2 | 192 | 10 | `2e-5` | 48, 96, 144, 192 |
| b4 | 4 | 96 | 5 | `4e-5` | 24, 48, 72, 96 |

The linear learning-rate scaling preserves approximately the same cumulative per-target update magnitude while the warmup and milestones preserve the same target-exposure fractions. Before selecting b2 or b4, the smoke must prove the scaled learning rate has finite loss and gradient norm. If the largest valid batch shows a clipped-gradient spike or materially worse numerical behavior in its first update, use the next smaller valid batch rather than changing additional hyperparameters.

Keep checkpoint publication at each quarter milestone and a rolling local limit of four. Record the winner, smoke evidence, final catalog IDs, exact PostTrain and Policy Prism commits, image digest, and schedule in this living plan before submission. The final run ID is:

    opdfull1-e2b12b-r16-scope384-b<1|2|4>-v1

The final run starts from base E2B, not checkpoint 96.

### Milestone 5: run and reconcile full OPD

Submit one full job with a 432,000-second timeout. Remain attached until model loading, the corrected XGrammar schema, at least one accepted target, one finite optimizer update, nonzero scored tokens, zero teacher failures, and the first retained checkpoint are visible. Thereafter monitor provider state, admitted-target count, loss, gradient norm, generation throughput, replacement pressure, trace synchronization, GPU utilization, and quarter milestones.

At completion, require provider success, PostTrain reconciliation `consistent`, exactly 384 optimized target identities, finite metrics, zero teacher failures, no duplicate logical target, complete native traces, the final adapter, and all four milestone checkpoints. A valid but higher-than-expected loss is retained for sealed qualification; an operational or structural failure is not silently accepted.

### Milestone 6: publish the final adapter

Materialize the exact final Trackio adapter into ignored Policy Prism state, verify its provider and PostTrain tree digests, PEFT configuration, rank/alpha/dropout, base repository/revision, safetensors loadability, and final trainer lineage. Add a model card containing training data/environment lineage, exact hyperparameters, smoke selection evidence, intended use, limitations, failed-attempt history, and pending or completed evaluation status.

Create and upload the private repository:

    carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1

Fresh-download at the returned 40-character commit into a separate ignored directory and require byte-identical model-card, config, and weight digests. Record both Trackio and Hugging Face identities in the Policy Prism model selection.

### Milestone 7: qualify sequentially and finalize evidence

Add an exact Trackio-backed adapter model selection, a vLLM LoRA inference binding with maximum LoRA rank 16 and 131,072 context, and two domain-evaluation work packages in project `policy-prism-scope-opd-e2b-12b`. Run scope first under `opdscope1-e2b12b-r16-sealed-v1`; require 18 expected/included traces, zero failures, zero truncations, zero trace errors, complete Claude judging, one native evaluation artifact, and consistent reconciliation. Only after it releases the GPU run recovery under `opdreco1-e2b12b-r16-sealed-v1`; require the same gates for 17 traces.

Materialize the exact native artifacts into ignored `.posttrain/state/native-evals/<run-id>/` directories. Generate serving metadata from PostTrain run evidence rather than hand-entering digests. Finalize directly into `Policy Prism/evaluation-runs` using stable IDs and require these five files per run:

    manifest.json
    traces.jsonl
    business-kpis.json
    engineering-metrics.json
    semantic-diagnostics.json

Run Policy Prism's normal `validate-runs`, compare scope only with compatible prior scope runs and recovery only with compatible prior recovery runs, update `evaluation-runs/catalog.json`, then commit and push final Policy Prism evidence. Update this plan with actual results and push the PostTrain feature branch.

## Concrete Steps

The commands below are run in a new execution goal. They deliberately do not print secrets. First configure the control shell:

    export POSTTRAIN_ROOT=/home/ali-awais-safdar/Post-Train/posttrain
    export POLICY_ROOT="/home/ali-awais-safdar/Policy Prism"
    export KIT=/home/ali-awais-safdar/Post-Train/posttrain-setup-v0.2.2-20260728/posttrain-setup
    export POSTTRAIN_ENV_FILE="$POLICY_ROOT/.env.posttrain"

    cd "$POSTTRAIN_ROOT"

    set -a
    . "$KIT/bundle/posttrain.env"
    . /home/ali-awais-safdar/.config/posttrain/credentials/huggingface.env
    . "$POSTTRAIN_ENV_FILE"
    set +a

    pt() {
      XDG_CONFIG_HOME=/tmp/posttrain-opd-empty-config \
      UV_CACHE_DIR=/tmp/posttrain-uv-cache \
      uv run --no-sync --package posttrain posttrain \
        --project-root "$POLICY_ROOT" \
        --env-file "$POSTTRAIN_ENV_FILE" \
        "$@"
    }

Verify branches before edits:

    git -C "$POSTTRAIN_ROOT" switch feat/gemma-policy-prism-opd-e2b-12b
    git -C "$POSTTRAIN_ROOT" status --short --branch
    git -C "$POLICY_ROOT" switch feat/scope-opd-e2b-12b-environment-v1
    git -C "$POLICY_ROOT" pull --ff-only origin feat/scope-opd-e2b-12b-environment-v1
    git -C "$POLICY_ROOT" rev-parse HEAD

The Policy Prism revision must be `147ac75997579f08154145ea9bdc6215b4aa7ec4` before new catalog edits. Do not use reset, rebase, or a main branch.

After implementing bounded batch support, run from PostTrain:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync pytest \
      packages/train/tests/test_api.py \
      packages/train/tests/test_trl_sparse_distillation.py \
      packages/train/tests/test_verifiers_grpo_bridge.py
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync ruff check \
      packages/train/src/posttrain/train/backends/trl/distillation.py \
      packages/train/tests/test_api.py \
      packages/train/tests/test_trl_sparse_distillation.py \
      packages/train/tests/test_verifiers_grpo_bridge.py
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync pyright \
      packages/train/src/posttrain/train/backends/trl/distillation.py \
      packages/train/tests/test_api.py \
      packages/train/tests/test_trl_sparse_distillation.py \
      packages/train/tests/test_verifiers_grpo_bridge.py
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync lint-imports
    git diff --check

Run Policy Prism's focused source-ID, reserve, deterministic-plan, and packaging tests from Policy Prism using the exact commands recorded by its branch. At minimum the focused OPD test file, Ruff, strict mypy, deterministic plan `--check`, and wheel resource audit must pass before catalog packaging.

Validate all three smoke packages from PostTrain:

    pt catalog validate
    for batch in b1 b2 b4; do
      package="gemma4_e2b_scope_opd_from_12b_smoke_${batch}.yaml"
      pt work-package validate "$package"
      pt job plan "$package" --job distill
      pt job pack "$package" --job distill --build-missing
    done

Expected validation identifies one `train.distill` job for each work package and each pack finishes with an immutable OCI image digest and isolated-runtime success. Do not submit if any command ends with `error:`.

Run smokes sequentially:

    export OPD_SMOKE_B1=opdsm1a1-e2b12b-r16-targets4-v1
    export OPD_SMOKE_B2=opdsm2a1-e2b12b-r16-targets4-v1
    export OPD_SMOKE_B4=opdsm4a1-e2b12b-r16-targets4-v1

    pt job run gemma4_e2b_scope_opd_from_12b_smoke_b1.yaml \
      --job distill --provider dstack --env HF_TOKEN \
      --timeout-seconds 21600 --run-id "$OPD_SMOKE_B1"
    pt run logs "$OPD_SMOKE_B1" --follow
    pt run wait "$OPD_SMOKE_B1" --timeout-seconds 21600
    pt run reconcile "$OPD_SMOKE_B1"

Apply the smoke gate before continuing. Then execute the same four commands for b2 and b4 by substituting the matching package and run variable. Never launch the next smoke before the previous run is terminal and the GPU is released. Record each `pt --json run show "$RUN_ID"` result in ignored `.posttrain/state/qualification/` for comparison.

After recording the winner in this plan and committing the final catalog/work package, validate and package the full job:

    pt catalog validate
    pt work-package validate gemma4_e2b_scope_opd_from_12b_final.yaml
    pt job plan gemma4_e2b_scope_opd_from_12b_final.yaml --job distill
    pt job pack gemma4_e2b_scope_opd_from_12b_final.yaml \
      --job distill --build-missing

    export OPD_BATCH=<1-or-2-or-4>
    export OPD_RUN="opdfull1-e2b12b-r16-scope384-b${OPD_BATCH}-v1"

    pt job run gemma4_e2b_scope_opd_from_12b_final.yaml \
      --job distill --provider dstack --env HF_TOKEN \
      --timeout-seconds 432000 --run-id "$OPD_RUN"
    pt run status "$OPD_RUN"
    pt run logs "$OPD_RUN" --follow

After terminal success:

    pt run wait "$OPD_RUN" --timeout-seconds 432000
    pt run reconcile "$OPD_RUN"
    pt --json run show "$OPD_RUN"

The last commands must show outcome `succeeded`, reconciliation `consistent`, complete required evidence, final adapter, all configured quarter checkpoints, native traces, and 384 unique optimized targets.

Publish only the exact final adapter. Use a protected-token shell, materialize to `$POLICY_ROOT/.posttrain/state/exports/$OPD_RUN/adapter`, validate the tree, create the private repository if missing, then upload:

    export HF_MODEL_REPO=carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync --package posttrain-train \
      hf repos create "$HF_MODEL_REPO" --repo-type model --private --exist-ok
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync --package posttrain-train \
      hf upload "$HF_MODEL_REPO" "$ADAPTER_DIR" . --repo-type model --private \
      --commit-message "Publish Policy Prism Gemma 4 E2B from 12B OPD adapter"
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync --package posttrain-train \
      python -c 'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["HF_MODEL_REPO"]).sha)'

Record the returned immutable revision, fresh-download that revision, and compare file hashes before evaluation.

Run sealed evaluations sequentially after catalog registration:

    export SCOPE_RUN=opdscope1-e2b12b-r16-sealed-v1
    export RECOVERY_RUN=opdreco1-e2b12b-r16-sealed-v1

    pt job pack gemma4_e2b_scope_opd_from_12b_scope_eval.yaml \
      --job evaluate --build-missing
    pt job run gemma4_e2b_scope_opd_from_12b_scope_eval.yaml \
      --job evaluate --provider dstack --env HF_TOKEN --env OPENROUTER_API_KEY \
      --timeout-seconds 21600 --run-id "$SCOPE_RUN"
    pt run logs "$SCOPE_RUN" --follow
    pt run wait "$SCOPE_RUN" --timeout-seconds 21600
    pt run reconcile "$SCOPE_RUN"

Require the 18-case gate, then run the analogous recovery package and require its 17-case gate. Do not submit both together.

Finally, from Policy Prism, materialize each native evaluation artifact and run:

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers finalize-run \
      --input "$SCOPE_NATIVE" \
      --run-id gemma-4-e2b-policy-prism-opd-r16-from-12b-v1-v11-sealed-scope \
      --serving-metadata "$SCOPE_SERVING_METADATA" \
      --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers finalize-run \
      --input "$RECOVERY_NATIVE" \
      --run-id gemma-4-e2b-policy-prism-opd-r16-from-12b-v1-v11-sealed-recovery \
      --serving-metadata "$RECOVERY_SERVING_METADATA" \
      --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers validate-runs --root "$POLICY_ROOT/evaluation-runs"

Only after validation passes should final evidence and plan updates be committed and pushed to their respective feature branches.

## Validation and Acceptance

Checkpoint preservation is accepted because the private HF repository is private, resolves to immutable revision `4f1fe9c75031396a11bcc44e2193f96df9003054`, and a fresh download matched all twelve uploaded checkpoint files byte-for-byte. The original Trackio artifact remains the lineage authority.

The reserve corrections are accepted offline only when plan v2 still has exactly 384 primaries and 96 reserves, all target distributions and immutable selection hashes remain valid, target 78 can claim the broader shared pool, every rules task schema enumerates its exact `source_provision_id`, the canonical source schema is not mutated, and malformed/unknown outputs are still rejected. Live acceptance additionally requires target 98 to complete in every valid smoke without the historical identifier mismatch.

A batch is qualified only by a successful real GPU smoke on the exact model, teacher, image, four selected tasks, no-MTP engine, and sparse loss. Local tensor tests alone do not qualify memory capacity. The final winner must be recorded with seconds per admitted target, GPU utilization, peak VRAM, loss/gradient evidence, teacher failures, replacements, and artifact digest.

The full run is accepted only with all 384 target identities optimized once, finite loss and gradient norms, zero teacher failures, complete native trace synchronization, final and milestone artifacts, and consistent PostTrain reconciliation. A provider `succeeded` state without these gates is insufficient.

Scope evaluation requires 18 expected and included traces; recovery requires 17. Both require zero failures, truncations, and trace errors, complete Claude semantic judging, one native `verifiers-evaluation` output, and consistent reconciliation. Finalized directories must contain exactly the five standard evidence files, update `evaluation-runs/catalog.json`, and pass `validate-runs`.

## Idempotence and Recovery

Validation and content-addressed packing may be rerun. Never reuse a run ID after provider admission. If a smoke fails operationally before model execution, fix the proven infrastructure cause and use an incremented suffix; do not reinterpret it as a batch failure. If it fails by OOM or numerical instability, reject that batch and continue with the next smaller candidate.

The final run starts from base weights. If infrastructure fails after a valid quarter checkpoint, resume only from the exact immutable checkpoint under a new run ID and only with identical batch, schedule, model, teacher, environment, plan, image, and ledger digests. Scientific failures such as exhausted reserves, invalid traces, NaN/Inf, or systematic teacher failure require diagnosis and a new fresh run; they are not automatically resumed.

Do not delete r0-r9, checkpoint 96, Trackio artifacts, native traces, finalized evaluation directories, or Hugging Face repositories. Cleanup is limited to new provider workspaces after all remote artifacts and five-file outputs are verified.

## Artifacts and Notes

Historical r9 identity:

    PostTrain run: opda09rs-e2b12b-r16-resume64-scope384-v2
    Trackio display: train.distill-opda09rs
    Trackio internal run ID: d9568a8d94e4433283b507635e341239
    Provider run: pt-5dfbcf7a92d8c14a24439628
    Terminal status: failed; provider terminated after explicit cancellation
    Retained: step-80 checkpoint, step-96 checkpoint, native rollout traces

Checkpoint 96 identity:

    Trackio project: policy-prism-scope-opd-e2b-12b
    Artifact: training-models-gemma4-e2b-it-bf16-distill-checkpoint-step-0096:v0
    Trackio manifest digest: ddde25b171407c76fba02e1b44b23a6531bcede4bd99c7859cc9809496e7f7e7
    PostTrain tree digest: cc53e8330d2af3160d098341dece1c82cf06a559366950f689c72caf24f16dba
    Size: 322,632,497 bytes
    HF repository: carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-checkpoint-96
    HF revision: 4f1fe9c75031396a11bcc44e2193f96df9003054
    Visibility: private

The four deterministic smoke tasks selected by seed 539 are:

    98: rules, q3, constructed-incomplete, standalone; former reserve failure boundary
    151: rules, q4, determinate, standalone
    284: evidence, q1, determinate, standalone
    367: graph, q4, determinate, full trajectory

The batch-one baseline estimate is 27–30 hours for training from measured r9 throughput. Batch-two and batch-four estimates are deliberately left open until their identical smokes produce observed seconds per target. After the smokes, estimate final training as:

    selected_smoke_seconds_per_target * 384 + measured initialization/checkpoint overhead

Add 2–6 hours for final image packaging, HF publication, two sequential sealed evaluations, and finalization. Update these estimates in the living plan before launching the full run.

## Interfaces and Dependencies

`posttrain.train.backends.trl.distillation._validate_memory_safe_sparse_request` owns the narrow supported batch envelope. The TRL generator in `packages/train/src/posttrain/train/backends/trl/distillation.py` owns generation batch size; `posttrain.train.integrations.verifiers.VerifiersEnvironmentRolloutBridge` owns task selection and concurrent exact-token rollouts; the sparse trainer owns batched completion-only loss without full sequence-vocabulary logits.

Policy Prism `scope_opd_tasks.py` owns task-specific generated schemas; `scope_opd_admission.py`, `harness.py`, and `scope_opd_ledger.py` own structural admission, reserve allocation, and resumable target identity. PostTrain must not import Policy Prism into reusable packages. Policy Prism must not weaken canonical validation to accommodate vLLM.

External services are Hugging Face for gated base models and private adapter publication, the Live Kit OCI registry for content-addressed images, dstack for GPU placement, Trackio for metrics/traces/artifacts, OpenRouter for Claude judging, and Observatory for read-only inspection. Credentials remain in permission-protected environment files and never enter commits or terminal output.

Revision note (2026-08-10): replaced the earlier resume-to-384 plan after r9 reached checkpoint 96 and failed at target 98. This revision records verified checkpoint publication, distinguishes the step-78 and target-98 causes, adds no-MTP batch 1/2/4 qualification, requires a fresh uniform-batch production run, and defers all new GPU submissions to the next execution goal.
