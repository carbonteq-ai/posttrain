"""Results-only training export; native assessment records remain replay authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from posttrain.common import ContractError


def _object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("assessment tracking projection requires an object")
    return value


def _rows(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError("assessment tracking projection requires a list")
    return value


def _fields(value: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    # Only scalar identifiers/values enter a result. Future nested schema fields
    # cannot silently become a transcript transport through an existing key.
    result = {name: value[name] for name in names if name in value}
    if any(item is not None and not isinstance(item, (str, int, float, bool)) for item in result.values()):
        raise ContractError("assessment result fields must be scalar")
    return result


def _subject(value: Any) -> dict[str, Any]:
    subject = _object(value)
    encoded = json.dumps(dict(subject), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    result = {
        "subject_digest": hashlib.sha256(encoded.encode()).hexdigest(),
        **_fields(subject, ("kind", "snapshot_id", "episode_id", "trace_id", "node_index", "call_index")),
    }
    if subject.get("kind") == "execution" and subject.get("execution") is not None:
        # Stable occurrence coordinates and evidence revision are distinct. Only
        # native coordinates enter tracking; payloads remain in the source archive.
        result["execution"] = _fields(
            _object(subject["execution"]),
            ("episode_id", "trace_id", "origin", "invocation_id", "prefix_digest", "event_count", "phase"),
        )
    return result


def _target(value: Any) -> dict[str, Any]:
    target = _object(value)
    return {
        "subject": _subject(target["subject"]),
        "signal": _fields(
            _object(target["signal"]),
            ("signal_id", "revision", "semantics", "units", "direction", "minimum", "maximum"),
        ),
    }


def _assessment(value: Any) -> dict[str, Any]:
    assessment = _object(value)
    result = {
        **_target(assessment),
        **_fields(assessment, ("assessment_id", "run_id", "view_id", "status", "value")),
    }
    if assessment.get("preference") is not None:
        preference = _object(assessment["preference"])
        result["preference"] = {
            **_fields(preference, ("relation", "preferred_subject_id")),
            "alternative_subject_ids": [
                _subject(alternative)["subject_digest"]
                for alternative in _rows(preference["alternatives"])
            ],
        }
    if assessment.get("derivation") is not None:
        derivation = _object(assessment["derivation"])
        parents = {}
        for name in ("required_parent_ids", "optional_parent_ids", "gate_decision_ids"):
            ids = _rows(derivation.get(name, []))
            if not all(isinstance(item, str) for item in ids):
                raise ContractError("assessment derivation identities must be strings")
            parents[name] = ids
        result["derivation"] = {
            **_fields(derivation, ("rule_id", "rule_revision", "allow_cross_snapshot")),
            **parents,
        }
    return result


def _view(value: Any) -> dict[str, Any]:
    view = _object(value)
    if "archive_view_ref" in view:
        if set(view) != {"archive_view_ref"}:
            raise ContractError("unsupported native view reference for tracking projection")
        view = _object(view["archive_view_ref"])
    return _fields(view, ("view_id", "snapshot_id", "builder_revision", "scope", "input_digest"))


def _batches(value: Any) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for item in _rows(value):
        batch = _object(item)
        if batch.get("schema_version") != 1:
            raise ContractError("unsupported native assessment schema for tracking projection")
        run = _object(batch["run"])
        key = tuple(run[name] for name in ("run_id", "producer_id", "snapshot_id", "invocation_id", "attempt_id"))
        latest[key] = batch
    results = []
    for batch in latest.values():
        run = _object(batch["run"])
        source = _object(batch["source"])
        expected = _rows(run["expected"])
        assessments = _rows(batch.get("assessments", []))
        results.append(
            {
                "schema_version": 1,
                "source": _fields(source, ("snapshot_id", "episode_id", "source_digest")),
                "run": _fields(
                    run,
                    (
                        "run_id",
                        "producer_id",
                        "producer_revision",
                        "rubric_revision",
                        "snapshot_id",
                        "invocation_id",
                        "attempt_id",
                        "status",
                    ),
                ),
                "configuration_digest": hashlib.sha256(str(run.get("configuration_json", "{}")).encode()).hexdigest(),
                "views": [
                    _view(view)
                    for view in _rows(batch.get("views", []))
                ],
                "coverage": {"requested": len(expected), "returned": len(assessments)},
                "expected": [{**_target(target), **_fields(_object(target), ("required",))} for target in expected],
                "assessments": [_assessment(assessment) for assessment in assessments],
            }
        )
    return results


def _assignments(value: Any) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for item in _rows(value):
        assignment = _object(item)
        request = _object(assignment["request"])
        latest[(request["invocation_id"], request["attempt_id"])] = assignment
    results = []
    for assignment in latest.values():
        if assignment.get("schema_version") != 1:
            raise ContractError("unsupported native assignment schema for tracking projection")
        request = _object(assignment["request"])
        rule = _object(request["rule"])
        contributions = []
        for item in _rows(assignment.get("contributions", [])):
            contribution = _object(item)
            parent_ids = _rows(contribution["parent_assessment_ids"])
            if not all(isinstance(parent, str) for parent in parent_ids):
                raise ContractError("assignment parents must be identities")
            contributions.append({
                **_fields(contribution, ("contribution_id", "branch_id", "channel", "status", "value", "weight", "allocation", "attribution")),
                "recipient": _subject(contribution["recipient"]),
                "signal": _fields(_object(contribution["signal"]), ("signal_id", "revision", "semantics", "units", "direction")),
                "parent_assessment_ids": parent_ids,
                "gates": [
                    _fields(_object(gate), ("gate_id", "outcome"))
                    for gate in _rows(contribution.get("gates", []))
                ],
            })
        results.append({
            **_fields(assignment, ("schema_version", "status")),
            **_fields(request, ("invocation_id", "attempt_id", "allocation", "overlap_policy")),
            "source": _fields(_object(request["source"]), ("snapshot_id", "episode_id", "source_digest")),
            "rule": _fields(rule, ("rule_id", "revision")),
            "configuration_digest": hashlib.sha256(str(rule.get("configuration_json", "{}")).encode()).hexdigest(),
            "contributions": contributions,
        })
    return results


def training_verifiers_results(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a solver record, replacing assessment bodies with compact results.

    This does not select a winning retry, add scores, or claim archive publication.
    It is called only after the writer establishes the run's training provenance.
    """
    result = deepcopy(dict(record))
    if "assessment_sources" in result:
        result["assessment_source_results"] = [
            _fields(_object(source), ("schema_version", "snapshot_id", "episode_id", "source_digest"))
            for source in _rows(result.pop("assessment_sources"))
        ]
        result["assessment_source_retention"] = "artifact_only"
    if "assessment_views" in result:
        result["assessment_view_results"] = [
            _fields(_object(view), ("view_id", "snapshot_id", "builder_revision", "scope", "input_digest"))
            for view in _rows(result.pop("assessment_views"))
        ]
        result["assessment_view_retention"] = "artifact_only"
    for field in ("tool_execution_events", "state_write_receipts", "tool_state_revision"):
        if field in result:
            result.pop(field)
            result["tool_execution_retention"] = "artifact_only"
    agent = result.get("agent")
    if isinstance(agent, Mapping) and agent.get("execution_purpose") == "assessment":
        result = {
            **_fields(result, ("id", "version", "episode_id", "ok", "is_completed", "stop_condition")),
            "agent": _fields(agent, ("execution_purpose", "trainable")),
            "assessment_trace_retention": "artifact_only",
            "assessment_batches": result.get("assessment_batches", []),
            "credit_assignments": result.get("credit_assignments", []),
        }
    if "assessment_batches" in result:
        result["assessment_results"] = _batches(result.pop("assessment_batches"))
    if "credit_assignments" in result:
        result["credit_assignment_results"] = _assignments(result.pop("credit_assignments"))
    info = result.get("info")
    if isinstance(info, dict):
        if "automationbench_capture" in info:
            info.pop("automationbench_capture")
            info["automationbench_capture_retention"] = "artifact_only"
        if "judge_calls" in info:
            calls = _rows(info.pop("judge_calls"))
            info["judge_call_results"] = [_fields(_object(call), ("name",)) for call in calls]
        if "posttrain_episode_reward_attempts" in info:
            attempts = _rows(info.pop("posttrain_episode_reward_attempts"))
            info["posttrain_episode_reward_attempt_results"] = [
                _fields(_object(attempt), ("attempt", "input_digest", "status", "error_type")) for attempt in attempts
            ]
        if "posttrain_episode_rewards" in info:
            verdict = _object(info["posttrain_episode_rewards"])
            assessments = _object(verdict.get("assessments", {}))
            info["posttrain_episode_rewards"] = {
                **_fields(verdict, ("trace_id", "scope", "scorer_digest")),
                "assessments": {
                    name: _fields(_object(value), ("status", "score")) for name, value in assessments.items()
                },
                "requirement_checks": [
                    _fields(_object(check), ("outcome",)) for check in _rows(verdict.get("requirement_checks", []))
                ],
            }
    return result
