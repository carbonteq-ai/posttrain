"""Bounded, complete-group admission before logical-batch normalization.

All ranks exchange validation outcomes even after a local rollout exception.
Accepted groups are retained; a rejected group is replaced in full across ranks.
This is evidence admission, not scalar reward-variance filtering.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .online_rl import EnvironmentRollout, PartialRolloutBatchError, RolloutBatch
from .profiles import CAPOSettings, GDPOSettings, GRPOSettings
from .reward_evidence import InvalidRewardEvidence


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    rollouts: tuple[EnvironmentRollout, ...]
    retained_positions: tuple[int, ...]
    attempted_rollouts: int
    rounds: int
    rejected_groups: int
    failed_rollouts: int


def validate_reward_rollout(rollout: EnvironmentRollout, settings: GDPOSettings | CAPOSettings) -> None:
    evidence = rollout.reward_evidence
    if evidence is None or rollout.is_truncated:
        raise InvalidRewardEvidence("structured RL requires resolved evidence and nontruncated trajectories")
    if isinstance(settings, GDPOSettings):
        evidence.require_components(settings.component_names)
    else:
        (outcome,) = evidence.require_components((settings.outcome_component,))
        if outcome not in (0.0, 1.0) or evidence.process is None:
            raise InvalidRewardEvidence("CAPO requires verified binary outcome and process credit")
        evidence.process.error_mask(rollout.env_mask)


def admit_reward_groups(
    batch: RolloutBatch,
    settings: GDPOSettings | CAPOSettings,
    collect: Callable[[RolloutBatch], Sequence[EnvironmentRollout]],
    gather_failures: Callable[[list[tuple[str, str]]], list[tuple[str, str]]],
) -> AdmissionResult:
    return admit_rollout_groups(batch, settings, collect, gather_failures)


def admit_rollout_groups(
    batch: RolloutBatch,
    settings: GRPOSettings | GDPOSettings | CAPOSettings,
    collect: Callable[[RolloutBatch], Sequence[EnvironmentRollout]],
    gather_failures: Callable[[list[tuple[str, str]]], list[tuple[str, str]]],
    *,
    max_attempts: int | None = None,
    retain_complete_on_exhaustion: bool = False,
) -> AdmissionResult:
    """Run on every rank with its local rows; gather_failures must be collective.

    The caller establishes complete global groups before admission. All ranks
    enter one failure exchange per round, including those with no pending local
    rows. Exceptions carry only their type across ranks; full diagnostic chains
    remain local. Cancellation is propagated, never converted into reward zero.
    """
    if not batch.prompt_group_ids:
        raise InvalidRewardEvidence("group admission requires explicit occurrence and response identities")
    pending = set(batch.prompt_group_ids)
    accepted: dict[int, EnvironmentRollout] = {}
    attempted = 0
    rejected = 0
    failed_rollouts = 0
    attempt_limit = settings.max_admission_attempts if max_attempts is None else max_attempts
    if attempt_limit < 1:
        raise ValueError("rollout group admission attempts must be positive")
    for attempt in range(attempt_limit):
        positions = [i for i, group in enumerate(batch.prompt_group_ids) if group in pending]
        failures: list[tuple[str, str]] = []
        local_error: Exception | None = None
        if positions:
            current = RolloutBatch(
                example_ids=tuple(batch.example_ids[i] for i in positions),
                step=batch.step,
                model_id=batch.model_id,
                prompt_group_ids=tuple(batch.prompt_group_ids[i] for i in positions),
                rollout_ids=tuple(f"{batch.rollout_ids[i]}/attempt/{attempt}" for i in positions),
            )
            attempted += len(positions)

            def admit_rows(
                rows: Sequence[tuple[int, EnvironmentRollout]],
                *,
                current_batch: RolloutBatch,
                batch_positions: Sequence[int],
                round_failures: list[tuple[str, str]],
                admitted: dict[int, EnvironmentRollout],
            ) -> None:
                for ordinal, row in rows:
                    if ordinal < 0 or ordinal >= len(batch_positions):
                        raise InvalidRewardEvidence("rollout bridge returned an invalid occurrence ordinal")
                    position = batch_positions[ordinal]
                    evidence = row.reward_evidence
                    identity_mismatch = row.example_id != current_batch.example_ids[ordinal]
                    if not isinstance(settings, GRPOSettings):
                        identity_mismatch = identity_mismatch or (
                            evidence is None
                            or evidence.prompt_group_id != current_batch.prompt_group_ids[ordinal]
                            or evidence.rollout_id != current_batch.rollout_ids[ordinal]
                        )
                    if identity_mismatch:
                        round_failures.append((current_batch.prompt_group_ids[ordinal], "rollout_identity_mismatch"))
                        continue
                    if isinstance(settings, GRPOSettings):
                        admitted[position] = row
                    else:
                        try:
                            validate_reward_rollout(row, settings)
                        except InvalidRewardEvidence as error:
                            round_failures.append((current_batch.prompt_group_ids[ordinal], str(error)))
                        else:
                            admitted[position] = row

            try:
                rows = collect(current)
                if len(rows) != len(positions):
                    raise InvalidRewardEvidence("rollout bridge returned an incomplete batch")
                admit_rows(
                    tuple(enumerate(rows)),
                    current_batch=current,
                    batch_positions=positions,
                    round_failures=failures,
                    admitted=accepted,
                )
            except PartialRolloutBatchError as error:
                completed_ordinals = set(error.completed)
                failed_ordinals = set(error.failures)
                if completed_ordinals | failed_ordinals != set(range(len(positions))):
                    local_error = error
                    failures.extend((group, type(error).__name__) for group in set(current.prompt_group_ids))
                else:
                    admit_rows(
                        tuple(error.completed.items()),
                        current_batch=current,
                        batch_positions=positions,
                        round_failures=failures,
                        admitted=accepted,
                    )
                    failures.extend(
                        (current.prompt_group_ids[ordinal], reason) for ordinal, reason in error.failures.items()
                    )
            except Exception as error:
                local_error = error
                failures.extend((group, type(error).__name__) for group in set(current.prompt_group_ids))
        failed_rollouts += len(failures)
        global_failures = gather_failures(failures)
        pending = {group for group, _ in global_failures}
        if not pending:
            if len(accepted) != len(batch.example_ids):
                raise InvalidRewardEvidence("admission completed without every local response")
            return AdmissionResult(
                tuple(accepted[i] for i in range(len(batch.example_ids))),
                tuple(range(len(batch.example_ids))),
                attempted,
                attempt + 1,
                rejected,
                failed_rollouts,
            )
        rejected += len(pending)
        accepted = {
            position: row for position, row in accepted.items() if batch.prompt_group_ids[position] not in pending
        }
        if attempt + 1 == attempt_limit:
            if retain_complete_on_exhaustion and (
                accepted or (isinstance(settings, GRPOSettings) and settings.algorithm == "olmo3")
            ):
                retained_positions = tuple(sorted(accepted))
                return AdmissionResult(
                    tuple(accepted[position] for position in retained_positions),
                    retained_positions,
                    attempted,
                    attempt + 1,
                    rejected,
                    failed_rollouts,
                )
            error_type = RuntimeError if isinstance(settings, GRPOSettings) else InvalidRewardEvidence
            raise error_type(
                f"rollout group admission exhausted {attempt_limit} attempts; "
                f"{len(pending)} groups remain invalid: {sorted(set(reason for _, reason in global_failures))}"
            ) from local_error
    raise AssertionError("unreachable admission state")
