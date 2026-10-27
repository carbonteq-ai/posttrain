# Score IW-OPD sampled tokens in row chunks

This ExecPlan is a living document maintained under `docs/templates/PLAN.md`.

## Purpose / Big Picture

The local-teacher IW-OPD loss takes a full-vocabulary log-softmax over every completion position for the student and the teacher; with a large vocabulary this dominates memory. After this change a run can set the training backend option `logits_chunk_size`, and the loss then runs both models to their final hidden states and scores only the sampled tokens, that many rows at a time. Observable result: with the option set, memory no longer scales with completion tokens times vocabulary, and the loss and student gradients equal those of the full path. With the option unset nothing changes.

## Progress

- [x] (2026-10-27) Ported the sampled-token scoring from the `policy_update_logprobs` module already used by GRPO.
- [x] `_compute_local_iw_opd_loss_chunked`, `_compute_iw_opd_from_actual_logprobs`, `_scores_local_iw_opd` and the `compute_loss` override in `packages/train/src/posttrain/train/backends/trl/distillation.py`.
- [x] Opt-in option, default unset (see Decision Log).
- [x] `packages/train/tests/test_trl_iw_opd_chunked_loss.py` (18 tests).
- [ ] Measure memory on a GPU with a real model. Not done.

## Surprises & Discoveries

- Observation: the TRL fork stamps the post-generation `num_items_in_batch` on each buffered input; the Trainer-level value is counted before generation and can be zero.
  Evidence: `IWOPDTrainer.compute_loss` reads `inputs.get("num_items_in_batch", num_items_in_batch)`. The override does the same, and a test fails if it does not.

## Decision Log

- Decision: opt-in through `logits_chunk_size` rather than default-on.
  Rationale: the chunked path replaces the model's full forward with a decoder forward, which would change behavior for every local-teacher run.
  Date/Author: 2026-10-27, owayys (confirmed by the requester).
- Decision: reuse `selected_token_logprobs` and `output_head`; do not carry a second chunked implementation.
  Rationale: smallest diff, one tested kernel.
  Date/Author: 2026-10-27, owayys.
- Decision: override only the local-teacher, `iw_opd`, non-Liger path; everything else calls `super()`.
  Rationale: the other paths do not take a full log-softmax of this shape.
  Date/Author: 2026-10-27, owayys.

## Outcomes & Retrospective

Value and gradient parity, the denominator and the dispatch are tested with tiny models. Not verified: a real Hugging Face model, PEFT or LoRA heads, multi-process `unwrap_model`, and the memory saving itself. The tests need torch and skip where it is absent.

## Context and Orientation

`ObservedIWOPDTrainer` subclasses TRL's `IWOPDTrainer` inside `run_distillation`. IW-OPD weights each sampled token's student log-probability by teacher minus rollout log-probability and a prefix weight. `selected_token_logprobs` in `packages/train/src/posttrain/train/backends/policy_update_logprobs.py` returns `log softmax(head(h)/T)[target]` per row with a custom backward that never stores full-vocabulary tensors.

## Plan of Work

Parse the option in `run_distillation` (a positive integer, or unset). Override `compute_loss` to call `_compute_local_iw_opd_loss_chunked` when the option is set and `_scores_local_iw_opd` is true. The loss function resolves the denominator from the inputs first, runs both decoders, scores the sampled tokens for each, and applies the IW-OPD weighting.

## Concrete Steps

From the repository root, with torch installed (it is not a dependency of the default test environment):

    UV_SYSTEM_CERTS=1 uv run --frozen --package posttrain-train pytest packages/train/tests/test_trl_iw_opd_chunked_loss.py -q

## Validation and Acceptance

The chunked loss and every student parameter gradient equal a per-token full-vocabulary reference for chunk sizes 1, 4 and 256, temperatures 1.0 and 0.7, and softcap off and on, with the teacher receiving no gradient. The stamped count wins over a zero argument. Non-finite rollout log-probabilities at required positions raise. Four configurations are checked for dispatch. Breaking the denominator, the prefix weighting or the student gradient each fails the tests.

## Idempotence and Recovery

Pure computation; repeatable. Unsetting the option restores the previous path.

## Artifacts and Notes

The release gate is one GPU run with the option set, comparing peak memory and the loss curve against the unset path; record it here.

## Interfaces and Dependencies

New backend option `logits_chunk_size`. No public type changes.

Revision note (2026-10-27): Initial plan.
