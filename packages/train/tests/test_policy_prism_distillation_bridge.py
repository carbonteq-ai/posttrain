"""End-to-end CPU coverage for Policy Prism staged distillation rollouts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, JsonValue
from posttrain.eval import EnvironmentBinding
from posttrain.train import PolicyTurnResult, RolloutBatch
from posttrain.train.integrations import create_verifiers_training_bridge

pytest.importorskip("verifiers")
policy_prism = pytest.importorskip("policy_prism_normative_verifiers")

build_environment = policy_prism.build_policy_prism_distillation_environment

_POLICY_PRISM_REPOSITORY = "https://github.com/alisafdar-carbonteq/policy-prism-monorepo.git"
_POLICY_PRISM_REVISION = "bfa7802f4e8250803f11fdba242608fb419acc8d"
_WORKSPACE = Path(__file__).resolve().parents[3]
_FORBIDDEN_TRAINING_LABELS = (
    "gold",
    "judge",
    "reference_answer",
    "reward",
    "teacher_response",
)


class SchemaAwareGenerator:
    def __init__(self, tangent: str) -> None:
        self.tangent = tangent
        self.requests: list[Any] = []

    async def generate(self, request: Any) -> PolicyTurnResult:
        self.requests.append(request)
        response_format = request.response_format
        assert response_format is not None
        payload = _valid_stage_payload(
            tangent=self.tangent,
            stage=response_format.name.removeprefix("policy_prism_"),
            schema=response_format.schema,
            messages=request.messages,
        )
        turn = len(self.requests)
        prompt_ids = (*request.previous_token_ids, 1000 + turn)
        completion_ids = (2000 + turn * 2, 2001 + turn * 2)
        message: dict[str, JsonValue] = {
            "role": "assistant",
            "content": json.dumps(payload, separators=(",", ":")),
        }
        raw_response: dict[str, JsonValue] = {
            "id": f"policy-prism-{self.tangent}-{turn}",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
        }
        return PolicyTurnResult(
            message=message,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=(-0.1, -0.2),
            finish_reason="stop",
            prompt_message_spans=(None,) * len(request.messages),
            prompt_is_content=(False,) * len(prompt_ids),
            raw_response=raw_response,
        )


@pytest.mark.parametrize(
    ("selection_id", "expected_count"),
    [("smoke", 1), ("qualification", 8), ("pilot", 64)],
)
def test_scope_catalog_bindings_match_policy_prism_builder_and_task_counts(
    tmp_path: Path,
    selection_id: str,
    expected_count: int,
) -> None:
    expected = build_environment(
        _POLICY_PRISM_REPOSITORY,
        _POLICY_PRISM_REVISION,
        "scope",
        selection_id,
    )
    catalog = open_catalog(
        scope="policy-prism",
        overlays=(_WORKSPACE / ".posttrain" / "catalog",),
    )
    actual = catalog.resolve(CatalogRef("environment", expected.id)).value

    assert isinstance(actual, EnvironmentBinding)
    assert actual == expected
    bridge = create_verifiers_training_bridge(
        actual,
        tmp_path / f"scope-{selection_id}-traces.jsonl",
        f"scope-{selection_id}-catalog-test",
        max_tokens=expected.sampling.max_tokens,
        temperature=expected.sampling.temperature,
        top_p=expected.sampling.top_p,
        purpose="distill",
    )
    assert len(bridge.dataset.examples) == expected_count
    for task in bridge.tasks.values():
        _assert_no_training_label_fields(task.data.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("tangent", "expected_stages", "expected_limits"),
    [
        ("scope", ("evidence", "rules", "graph"), (512, 1536, 768)),
        ("rule_recovery", ("evidence", "rules"), (512, 2048)),
    ],
)
def test_policy_prism_smoke_rollout_preserves_every_trainable_stage(
    tmp_path, tangent, expected_stages, expected_limits
) -> None:
    environment = build_environment(
        _POLICY_PRISM_REPOSITORY,
        _POLICY_PRISM_REVISION,
        tangent,
        "smoke",
    )
    bridge = create_verifiers_training_bridge(
        environment,
        tmp_path / f"{tangent}-traces.jsonl",
        f"{tangent}-bridge-test",
        max_tokens=12_288,
        temperature=1.0,
        top_p=1.0,
        purpose="distill",
    )
    generator = SchemaAwareGenerator(tangent)

    rollouts = asyncio.run(
        bridge.run(
            RolloutBatch(
                example_ids=(bridge.dataset.examples[0].id,),
                step=0,
                model_id="models/gemma4-e4b-it@bf16",
            ),
            generator,
        )
    )

    assert len(rollouts) == len(expected_stages)
    assert len({rollout.trace.external_id for rollout in rollouts}) == len(expected_stages)
    for rollout in rollouts:
        assert rollout.is_truncated is False
        assert len(rollout.completion_ids) == len(rollout.env_mask)
        assert len(rollout.completion_ids) == len(rollout.sampling_logprobs)
        assert sum(rollout.env_mask) > 0
        assert _sampled_group_count(rollout.env_mask) == 1
        assert rollout.trace.attributes["source_batch_position"] == 0
        assert rollout.trace.payload.get("error") is None
    assert len(generator.requests) == len(expected_stages)
    assert tuple(
        request.response_format.name.removeprefix("policy_prism_")
        for request in generator.requests
    ) == expected_stages
    assert tuple(request.sampling.max_tokens for request in generator.requests) == expected_limits
    for request in generator.requests:
        assert request.response_format is not None
        assert request.response_format.strict is True
        assert request.sampling.temperature == 1.0
        assert request.sampling.top_p == 1.0
        serialized = json.dumps(request.messages, sort_keys=True).lower()
        assert all(label not in serialized for label in _FORBIDDEN_TRAINING_LABELS)


def _valid_stage_payload(
    *,
    tangent: str,
    stage: str,
    schema: Any,
    messages: tuple[Any, ...],
) -> dict[str, object]:
    del schema
    segment_ids = _segment_ids(messages)
    completion = {"status": "complete", "reason": None}
    if stage == "evidence":
        return {
            "segments": [
                {"segment_id": segment_id, "status": "non_normative", "rule_count": 0}
                for segment_id in segment_ids
            ],
            "completion": completion,
        }
    if stage == "rules" and tangent == "scope":
        return {
            "resolution": {"status": "resolved", "reason": None},
            "rules": [],
            "completion": completion,
        }
    if stage == "rules":
        return {
            "rules": [],
            "coverage_ledger": [
                {"segment_id": segment_id, "status": "non_normative", "rule_count": 0}
                for segment_id in segment_ids
            ],
            "completion": completion,
        }
    if stage == "graph":
        return {"qualifiers": [], "attachments": [], "completion": completion}
    raise AssertionError(f"unexpected Policy Prism stage: {stage}")


def _segment_ids(messages: tuple[Any, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            segment_id = record.get("segment_id") if isinstance(record, dict) else None
            if isinstance(segment_id, str):
                values.append(segment_id)
    if not values:
        raise AssertionError("Policy Prism request contained no source segment IDs")
    return tuple(dict.fromkeys(values))


def _assert_no_training_label_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert str(key).lower() not in _FORBIDDEN_TRAINING_LABELS
            _assert_no_training_label_fields(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_no_training_label_fields(item)


def _sampled_group_count(mask: tuple[bool, ...]) -> int:
    return sum(value and (index == 0 or not mask[index - 1]) for index, value in enumerate(mask))
