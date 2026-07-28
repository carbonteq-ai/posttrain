# Run the full Gemma 4 Halcyon GraphQL SFT profile

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this change, a developer can run a separate 98-step, validation-aware LoRA SFT scenario for `google/gemma-4-12B-it` through `posttrain-lab`. The run starts from the pinned foundation model, consumes the pinned 392-row train split twice at effective batch size eight, evaluates the 31-row held-out split at the start, midpoint, and end, and publishes the final adapter and latest recovery checkpoint to W&B. The existing one-step canary remains available and unchanged.

## Progress

- [x] (2026-07-28 08:17Z) Recorded the completed GPU canary evidence and selected a lab-local full profile.
- [x] (2026-07-28 08:20Z) Added the full settings and batch-eight training binding without changing the canary selections.
- [x] (2026-07-28 08:23Z) Added the full validation-aware CLI scenario and focused composition tests.
- [x] (2026-07-28 08:27Z) Passed the complete lab suite, Ruff, targeted Pyright, import contracts, and whitespace validation.
- [ ] Run the 98-step W&B job and the corrected final-adapter reload probe.

## Surprises & Discoveries

- Observation: The canary used 28.49 GiB peak allocated VRAM and rendered no sequence longer than 758 tokens.
  Evidence: W&B run `369fff7c-8434-4b5e-8979-7b675d90a598` recorded zero train and validation supervised-token truncation at length 2048.
- Observation: The actual base parameter count inferred from the loaded PEFT model is 11,959,730,176, 48 fewer than the lab model metadata.
  Evidence: the run reported 11,992,514,560 total parameters and 32,784,384 LoRA parameters.
- Observation: Posttrain retains checkpoints 49 and 98 inside the trainer workspace but promotes only the latest checkpoint before the ephemeral workspace is deleted.
  Evidence: `finish_training()` calls `get_last_checkpoint()` and returns one recovery checkpoint.

## Decision Log

- Decision: Add a new full scenario rather than mutate or parameterize the canary at launch time.
  Rationale: each run must retain an immutable settings and work-package identity.
  Date/Author: 2026-07-28 / user and Codex.
- Decision: Start the full run from the foundation model, not the one-step canary adapter.
  Rationale: the canary is feasibility evidence; the 98 steps represent the complete two-pass training budget.
  Date/Author: 2026-07-28 / Codex.
- Decision: Keep checkpoint behavior unchanged and promote only the latest recovery checkpoint.
  Rationale: the user selected the surgical lab-local option instead of expanding framework artifact contracts.
  Date/Author: 2026-07-28 / user.
- Decision: Keep learning rate `1e-4`, max gradient norm 1.0, and length 2048.
  Rationale: the canary had ample memory and length headroom; five warmup steps and observed clipping provide sufficient initial safeguards.
  Date/Author: 2026-07-28 / Codex.

## Outcomes & Retrospective

The full lab scenario is implemented and CPU-validated. The complete lab suite passes with 67 tests and 5 credential-gated skips; Ruff, targeted Pyright, all eight import contracts, and whitespace checks pass. The 98-step GPU/W&B run and final adapter reload remain the release gates.

## Context and Orientation

`apps/lab/src/posttrain_lab/gemma4_halcyon.py` owns the lab-only Gemma model, renderer, target, LoRA binding, and canary settings. `apps/lab/src/posttrain_lab/cli.py` turns those selections and the pinned train/test sources into a five-seat `train.sft` work package. TRL receives already-rendered token IDs and labels; it does not execute GraphQL. An effective batch is the number of examples contributing to one optimizer update. With per-device batch one and eight gradient-accumulation microbatches, 392 examples produce 49 optimizer updates per pass and 98 updates across two passes.

The backend disables dataset shuffling, so the second pass repeats the first pass order. Validation uses TRL's step schedule. `eval_on_start=True` produces step-zero evidence, `eval_steps=49` evaluates at the midpoint and final step, and Posttrain avoids a duplicate final evaluation when step 98 already has one.

## Plan of Work

In `apps/lab/src/posttrain_lab/gemma4_halcyon.py`, correct the model parameter fact to 11,959,730,176. Add a full training binding derived from the canary binding with a distinct identity and `global_batch_size=8`. Add full settings with 98 steps, length 2048, per-device batch one, accumulation eight, learning rate `1e-4`, warmup ratio 0.05, max gradient norm 1.0, logging every step, checkpoints every 49 steps with retention two, seed 42, and gradient checkpointing. Validation runs every 49 steps with batch one, on start, and at end.

In `apps/lab/src/posttrain_lab/cli.py`, add `gemma4-halcyon-graphql-sft`. Compose it with work-package ID `train/gemma4-12b/halcyon-graphql-sft`, a distinct validated SFT definition, the same model and data sources, and the new full settings and binding. Preserve the canary entry point and identities.

Extend `apps/lab/tests/test_gemma4_halcyon_sft.py` to prove exact full settings, effective and declared global batch eight, corrected parameter metadata, five-seat resolution, foundation-model input, and separation from the canary.

## Concrete Steps

From the repository root, run:

    uv run pytest -q apps/lab/tests/test_halcyon_graphql_data.py apps/lab/tests/test_gemma4_halcyon_sft.py
    uv run pytest -q apps/lab/tests
    uv run ruff check .
    uv run pyright packages/common/src/posttrain/common/models.py apps/lab/src/posttrain_lab/gemma4_halcyon.py apps/lab/src/posttrain_lab/data/halcyon_graphql.py apps/lab/src/posttrain_lab/cli.py apps/lab/tests/test_gemma4_halcyon_sft.py apps/lab/tests/test_halcyon_graphql_data.py
    uv run lint-imports
    git diff --check

On the existing 96 GB pod, after pulling the implementation commit, run:

    uv run --locked --package posttrain-lab --extra gpu-train posttrain-lab \
      gemma4-halcyon-graphql-sft \
      --project halcyon-graphql-sft \
      --project-root /workspace/posttrain \
      --scratch-root "$POSTTRAIN_SCRATCH_ROOT" \
      --tracking-backend wandb \
      --wandb-entity "$WANDB_ENTITY"

## Validation and Acceptance

CPU acceptance requires both scenarios to resolve all five SFT seats while selecting different settings and bindings. The canary remains one step with global batch one; the full scenario is 98 steps with effective and declared global batch eight.

GPU acceptance requires global step 98; finite train losses; finite validation losses at steps 0, 49, and 98; zero train and validation supervised-token truncation; no OOM or NaN/Inf; checkpoints configured at 49 and 98; and committed W&B artifacts for the final adapter, tokenizer, summary, and latest recovery checkpoint. Record peak VRAM, step time, gradient-clipping frequency, length percentiles, supervision ratio, and exact source/model/data/dependency provenance.

After training, run a W&B-tracked adapter reload using the final artifact. Load `Gemma4UnifiedForConditionalGeneration`, assert all 328 LoRA modules remain in the language model, perform a finite text-only forward pass, and generate with both the tokenizer EOS token and `<tool_call|>` as stop tokens. The output must contain one native `execute_graphql` call and no fabricated tool response after its end delimiter.

## Idempotence and Recovery

Tests and launches are safe to repeat and produce distinct W&B runs. Do not overwrite or resume from the canary adapter. A successful full run uploads only checkpoint 98 as the durable recovery artifact. Checkpoint 49 is transient trainer state, and this change does not claim crash recovery from it. If a run fails, retain its W&B evidence, diagnose it, and start a new run rather than changing the recorded settings identity.

## Artifacts and Notes

The canary W&B run is `369fff7c-8434-4b5e-8979-7b675d90a598`; its adapter reload run is `0hyffszz`. The canary validation loss changed from 2.64068 to 2.48794 after one update. The reload selected the expected unified class, loaded 328 language-model-only LoRA modules, and generated a correct named GraphQL operation. Its first probe continued past the tool call because `<tool_call|>` was not configured as a generation stop token; the full-run reload corrects that probe behavior.

## Interfaces and Dependencies

The new lab CLI scenario is `gemma4-halcyon-graphql-sft`. The new selection constants are `GEMMA4_HALCYON_SFT` and `GEMMA4_HALCYON_LORA_FULL`. No reusable package API, dependency version, external repository, catalog entry, or OCI image changes.

Plan revision note: created from the completed canary evidence, then updated after the lab scenario and CPU validation completed; GPU execution remains outstanding.
