"""Reward-source selection must be explicit, strict, and independent of algorithms."""

from dataclasses import asdict, replace

import pytest
from posttrain.common import JsonValue, TraceFactSet, TraceObservation
from posttrain.train.reward_evidence import InvalidRewardEvidence, ProcessCredit
from posttrain.train.reward_projection import RewardComponentProjection, RewardProjection
from pydantic import TypeAdapter


def observation(*, process=None, measures=None):
    info: dict[str, JsonValue] = {"posttrain_prompt_group_id": "occurrence/1", "posttrain_rollout_id": "response/1"}
    if process is not None:
        info["critic"] = process
    return TraceObservation(
        trace_type="verifiers",
        external_id="trace/1",
        payload={"info": info},
        facts=(TraceFactSet(namespace="project", calculator_version="1", measures=measures or {}),),
    )


def projection():
    return RewardProjection(
        "project/reward",
        "1",
        (RewardComponentProjection("outcome", "scalar"), RewardComponentProjection("quality", "metric", "score")),
        process_info_key="critic",
    )


def test_projection_roundtrip_preserves_selection_and_valid_zero():
    selected = projection()
    adapter = TypeAdapter(RewardProjection)
    assert adapter.validate_json(adapter.dump_json(selected)) == selected
    credit = ProcessCredit("valid", "tokens@1", "trace/1/critique/1", ((0, 1),))
    result = selected.project(observation(process=asdict(credit), measures={"score": 0.0}), scalar_reward=1.0)
    assert result.require_components(("outcome", "quality")) == (1.0, 0.0)
    assert result.process == credit
    assert result.projection_id == "project/reward@1"


def test_missing_evidence_remains_failed_not_zero():
    result = projection().project(observation(), scalar_reward=0.0)
    assert result.components[1].status == "failed"
    assert result.components[1].value is None
    assert result.process is not None
    assert result.process.status == "failed"
    with pytest.raises(InvalidRewardEvidence, match="unavailable"):
        result.require_components(("quality",))


@pytest.mark.parametrize("index", [True, "0", 0.5, -1])
def test_process_coordinates_are_not_coerced(index):
    credit = {
        "status": "valid",
        "projection_id": "tokens@1",
        "evidence_ref": "trace/1/critic",
        "error_spans": [[index, 1]],
    }
    with pytest.raises(InvalidRewardEvidence, match="malformed"):
        projection().project(observation(process=credit), scalar_reward=1.0)


def test_ambiguous_metric_projection_is_rejected():
    original = observation(measures={"score": 0.5})
    duplicate = TraceObservation(
        trace_type=original.trace_type,
        external_id=original.external_id,
        payload=original.payload,
        facts=original.facts * 2,
    )
    with pytest.raises(InvalidRewardEvidence, match="ambiguous"):
        projection().project(duplicate, scalar_reward=1.0)


def test_annotation_requires_selected_scorer_and_does_not_replace_native_reward():
    original = observation()
    selected = RewardProjection(
        "judge",
        "1",
        (
            RewardComponentProjection("outcome", "scalar"),
            RewardComponentProjection("quality", "annotation", "quality"),
        ),
        scorer_digest="a" * 64,
    )
    with pytest.raises(InvalidRewardEvidence, match="scorer identity"):
        selected.project(original, scalar_reward=0.25)
    native_info = original.payload["info"]
    assert isinstance(native_info, dict)
    info: dict[str, JsonValue] = dict(native_info)
    info.update(quality=0.75, posttrain_scorer_digest="a" * 64)
    evidence = selected.project(replace(original, payload={"info": info}), scalar_reward=0.25)
    assert evidence.require_components(("outcome", "quality")) == (0.25, 0.75)


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64])
def test_invalid_scorer_digest_is_rejected(digest):
    with pytest.raises(InvalidRewardEvidence, match="SHA-256"):
        replace(projection(), scorer_digest=digest)


def test_scorer_selection_roundtrips_with_annotation_source():
    selected = RewardProjection(
        "judge",
        "1",
        (RewardComponentProjection("quality", "annotation", "quality"),),
        scorer_digest="b" * 64,
    )
    adapter = TypeAdapter(RewardProjection)
    assert adapter.validate_json(adapter.dump_json(selected)) == selected


def test_native_metric_is_explicit_and_independent_of_scalar_and_summary():
    original = observation(measures={"task_completed_correctly": 0.75})
    selected = RewardProjection(
        "automationbench",
        "1",
        (RewardComponentProjection("outcome", "native_metric", "task_completed_correctly"),),
    )
    raw = replace(original, payload={**original.payload, "metrics": {"task_completed_correctly": 1.0}})
    assert selected.project(raw, scalar_reward=0.25).require_components(("outcome",)) == (1.0,)
    assert selected.project(original, scalar_reward=0.25).components[0].status == "failed"
    adapter = TypeAdapter(RewardProjection)
    assert adapter.validate_json(adapter.dump_json(selected)) == selected


@pytest.mark.parametrize("metrics", [[], {"task_completed_correctly": True}, {"task_completed_correctly": "1"}])
def test_malformed_native_metric_is_rejected(metrics):
    original = observation()
    selected = RewardProjection(
        "automationbench",
        "1",
        (RewardComponentProjection("outcome", "native_metric", "task_completed_correctly"),),
    )
    with pytest.raises(InvalidRewardEvidence):
        selected.project(replace(original, payload={**original.payload, "metrics": metrics}), scalar_reward=0.0)
