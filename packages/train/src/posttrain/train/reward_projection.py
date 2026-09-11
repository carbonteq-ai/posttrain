"""Versioned projections from retained trace evidence, independent of algorithms.

Environment/scorer code owns the meaning of a score. This module only selects
declared values and validates their coordinates; it never invents missing credit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from posttrain.common import TraceObservation
from pydantic import TypeAdapter, ValidationError

from .reward_evidence import InvalidRewardEvidence, ProcessCredit, RewardEvidence, RewardValue
from .turn_rewards import TURN_PROJECTION, TurnAssessment, TurnTokenMap, validate_turn_assessments


@dataclass(frozen=True, slots=True)
class RewardComponentProjection:
    name: str
    source: Literal["scalar", "contribution", "metric", "native_metric", "annotation", "turn_mean", "turn_sum"]
    key: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or self.source not in {
            "scalar",
            "contribution",
            "metric",
            "native_metric",
            "annotation",
            "turn_mean",
            "turn_sum",
        }:
            raise InvalidRewardEvidence("reward component projection requires a name and recognized source")
        if self.source != "scalar" and not self.key.strip():
            raise InvalidRewardEvidence("named reward projection requires a source key")
        if self.source == "scalar" and self.key:
            raise InvalidRewardEvidence("scalar projection does not accept a source key")


@dataclass(frozen=True, slots=True)
class RewardProjection:
    """Serializable selection; process evidence is an explicitly stamped trace field.

    The selected info field must contain a ProcessCredit JSON object, including its
    status and retained evidence reference. Coordinates are original completion
    tokens, never a judge's retokenization. Raw judge outputs belong to native trace
    annotations and are referenced by that object rather than copied here.
    """

    id: str
    revision: str
    components: tuple[RewardComponentProjection, ...]
    process_info_key: str | None = None
    scorer_digest: str | None = None
    turns_info_key: str | None = None
    turn_reward_key: str | None = None
    turn_error_key: str | None = None
    turn_reward_includes_terminal_outcome: bool | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.revision.strip() or not self.components:
            raise InvalidRewardEvidence("reward projection requires versioned identity and components")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise InvalidRewardEvidence("reward projection output names must be unique")
        if self.process_info_key is not None and not self.process_info_key.strip():
            raise InvalidRewardEvidence("process evidence field cannot be empty")
        if self.scorer_digest is not None and (
            len(self.scorer_digest) != 64 or any(c not in "0123456789abcdef" for c in self.scorer_digest)
        ):
            raise InvalidRewardEvidence("scorer digest must be a SHA-256")
        if self.turns_info_key is not None and (not self.turns_info_key.strip() or self.scorer_digest is None):
            raise InvalidRewardEvidence("turn rewards require an annotation key and frozen scorer digest")
        if any(component.source in {"turn_mean", "turn_sum"} for component in self.components):
            if self.turns_info_key is None:
                raise InvalidRewardEvidence("turn reductions require a declared turn evidence field")
        if self.turn_reward_key is not None and (not self.turn_reward_key.strip() or self.turns_info_key is None):
            raise InvalidRewardEvidence("direct turn rewards require a named score and turn evidence field")
        if self.turn_reward_key is not None and type(self.turn_reward_includes_terminal_outcome) is not bool:
            raise InvalidRewardEvidence("direct turn rewards must declare whether they include terminal outcome")
        if self.turn_reward_key is None and self.turn_reward_includes_terminal_outcome is not None:
            raise InvalidRewardEvidence("terminal outcome declaration requires direct turn reward selection")
        if self.turn_error_key is not None:
            if not self.turn_error_key.strip() or self.turns_info_key is None or self.process_info_key is not None:
                raise InvalidRewardEvidence(
                    "turn errors require turn evidence and cannot also select token process evidence"
                )

    def turn_assessments(
        self,
        observation: TraceObservation,
        turn_ids: tuple[str, ...],
    ) -> tuple[TurnAssessment, ...]:
        """Validate plugin evidence independently of an algorithm's group contract."""
        if self.turns_info_key is None:
            return ()
        info = observation.payload.get("info")
        if not isinstance(info, Mapping) or self._scorer_identity(info) != self.scorer_digest:
            raise InvalidRewardEvidence("trace scorer identity differs from the selected projection")
        payload = info.get(self.turns_info_key)
        if not isinstance(payload, Mapping) or (
            payload.get("trace_id") != observation.external_id
            or payload.get("branch_id") != "0"
            or payload.get("projection_id") != TURN_PROJECTION
            or payload.get("scorer_digest") != self.scorer_digest
        ):
            raise InvalidRewardEvidence("turn evidence identity differs from the retained native projection")
        try:
            assessments = TypeAdapter(tuple[TurnAssessment, ...]).validate_json(
                json.dumps(payload.get("assessments")),
                strict=True,
            )
        except (ValidationError, ValueError, TypeError) as error:
            raise InvalidRewardEvidence("malformed retained turn evidence") from error
        return validate_turn_assessments(assessments, turn_ids)

    def _scorer_identity(self, info: Mapping[str, object]) -> object:
        # Namespaced plugin annotations compose independently; the singular
        # field is retained only for existing trace enrichers.
        identities = info.get("posttrain_scorer_digests")
        if isinstance(identities, Mapping) and self.turns_info_key in identities:
            return identities[self.turns_info_key]
        return info.get("posttrain_scorer_digest")

    def project_turn_rewards(
        self,
        observation: TraceObservation,
        turn_ids: tuple[str, ...],
    ) -> tuple[float, ...] | None:
        """Select complete local rewards; None means an explicitly unselected path."""
        if self.turn_reward_key is None:
            return None
        return tuple(item.require(self.turn_reward_key) for item in self.turn_assessments(observation, turn_ids))

    def project(
        self,
        observation: TraceObservation,
        *,
        scalar_reward: float,
        turn_ids: tuple[str, ...] = (),
        native_turns: tuple[TurnTokenMap, ...] = (),
    ) -> RewardEvidence:
        info = observation.payload.get("info")
        if not isinstance(info, Mapping):
            raise InvalidRewardEvidence("structured reward trace requires stamped identities")
        if self.scorer_digest is not None and self._scorer_identity(info) != self.scorer_digest:
            raise InvalidRewardEvidence("trace scorer identity differs from the selected projection")
        group_id = info.get("posttrain_prompt_group_id")
        rollout_id = info.get("posttrain_rollout_id")
        if not isinstance(group_id, str) or not isinstance(rollout_id, str):
            raise InvalidRewardEvidence("structured reward trace lacks explicit group or rollout identity")
        values: list[RewardValue] = []
        assessments = self.turn_assessments(observation, turn_ids)
        for component in self.components:
            candidates: list[object] = []
            if component.source == "scalar":
                candidates = [scalar_reward]
            elif component.source in {"turn_mean", "turn_sum"}:
                total = math.fsum(assessment.require(component.key) for assessment in assessments)
                candidates = [total / len(assessments) if component.source == "turn_mean" else total]
            elif component.source == "annotation":
                candidates = [info[component.key]] if component.key in info else []
            elif component.source == "native_metric":
                metrics = observation.payload.get("metrics")
                if metrics is not None and not isinstance(metrics, Mapping):
                    raise InvalidRewardEvidence("native metrics must be a named mapping")
                candidates = (
                    [metrics[component.key]] if isinstance(metrics, Mapping) and component.key in metrics else []
                )
            elif component.source == "contribution":
                candidates = [
                    item.contribution
                    for facts in observation.facts
                    for item in facts.reward_components
                    if item.name == component.key
                ]
            else:
                candidates = [
                    facts.measures[component.key] for facts in observation.facts if component.key in facts.measures
                ]
            if len(candidates) > 1:
                raise InvalidRewardEvidence(f"ambiguous reward source {component.key!r}")
            value = candidates[0] if candidates else None
            if value is None:
                values.append(RewardValue(component.name, "failed"))
            elif isinstance(value, bool) or not isinstance(value, int | float):
                raise InvalidRewardEvidence(f"reward source {component.key!r} is not numeric")
            else:
                values.append(RewardValue(component.name, "valid", float(value)))
        process = None
        if self.turn_error_key is not None:
            if tuple(turn.id for turn in native_turns) != turn_ids or not native_turns:
                raise InvalidRewardEvidence("turn errors require the matching native token map")
            assert self.turns_info_key is not None
            payload = info[self.turns_info_key]
            if not isinstance(payload, Mapping):
                raise InvalidRewardEvidence("turn errors require retained turn evidence")
            errors = payload.get(self.turn_error_key)
            if (
                not isinstance(errors, list)
                or any(not isinstance(item, str) for item in errors)
                or len(set(errors)) != len(errors)
                or not set(errors).issubset(turn_ids)
            ):
                raise InvalidRewardEvidence("turn errors must name unique known native turns")
            process = ProcessCredit(
                "valid",
                TURN_PROJECTION,
                f"{observation.external_id}/info/{self.turns_info_key}",
                tuple(span for turn in native_turns if turn.id in errors for span in turn.token_spans),
            )
        if self.process_info_key is not None:
            payload = info.get(self.process_info_key)
            if payload is None:
                process = ProcessCredit("failed", f"{self.id}@{self.revision}", observation.external_id)
            else:
                try:
                    process = TypeAdapter(ProcessCredit).validate_json(json.dumps(payload), strict=True)
                except (ValidationError, ValueError, TypeError) as error:
                    raise InvalidRewardEvidence("malformed retained process evidence") from error
        return RewardEvidence(
            group_id,
            rollout_id,
            observation.external_id,
            "0",
            f"{self.id}@{self.revision}",
            tuple(values),
            process,
        )
