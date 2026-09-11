"""Episode migration preserves native source identity and excludes judge agents."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.data import TraceSelection, supervised_from_verifiers, supervised_from_verifiers_jsonl


def _trace(*, trainable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        id="trace-a",
        agent=SimpleNamespace(trainable=trainable),
        reward=0.0,
        stop_condition="completed",
        has_error=False,
        is_truncated=False,
        tools=[],
        branches=[
            SimpleNamespace(
                index=0,
                nodes=[
                    SimpleNamespace(message={"role": "user", "content": "Question"}, sampled=False),
                    SimpleNamespace(message={"role": "assistant", "content": "Answer"}, sampled=True),
                ],
            )
        ],
    )


def test_episode_scope_prevents_collisions_and_keeps_zero_reward() -> None:
    records = [SimpleNamespace(id=name, ok=True, errors=[], traces=[_trace()]) for name in ("one", "two")]
    data = supervised_from_verifiers(records, dataset_id="toy", revision="1")
    assert len({row.id for row in data.examples}) == 2
    assert [row.metadata["episode_id"] for row in data.examples] == ["one", "two"]
    assert all(row.trainable_message_indices == (1,) for row in data.examples)


def test_episode_errors_and_nonpolicy_agents_are_not_training_targets() -> None:
    records = [
        SimpleNamespace(id="failed", ok=False, errors=[], traces=[_trace()]),
        SimpleNamespace(id="retried", ok=True, errors=[{"type": "timeout"}], traces=[_trace()]),
        SimpleNamespace(id="valid", ok=True, errors=[], traces=[_trace(trainable=False), _trace()]),
    ]
    data = supervised_from_verifiers(records, dataset_id="toy", revision="1")
    assert len(data.examples) == 1
    assert data.examples[0].metadata["episode_id"] == "valid"


def test_explicit_error_selection_does_not_admit_nonpolicy_agents() -> None:
    episode = SimpleNamespace(id="failed", ok=False, errors=[], traces=[_trace(trainable=False), _trace()])
    data = supervised_from_verifiers(
        [episode], dataset_id="toy", revision="1", selection=TraceSelection(drop_errors=False)
    )
    assert len(data.examples) == 1


def test_native_episode_jsonl_projects_without_mutating_source(tmp_path: Path) -> None:
    native = pytest.importorskip("verifiers.v1.episode", reason="requires modern native Verifiers episode API")
    if not hasattr(native, "WireEpisode"):
        pytest.skip("selected legacy Verifiers has no WireEpisode; run with the v0.3.1 candidate")
    task = {"type": "ToyTask", "data": {"question": "Question"}}
    record = {
        "id": "episode-a",
        "task": task,
        "ok": True,
        "traces": [
            {
                "id": "trace-a",
                "task": task,
                "agent": {"config": {}, "trainable": True},
                "ok": True,
                "is_completed": True,
                "rewards": {"outcome": {"score": 1.0}},
                "nodes": [
                    {"message": {"role": "user", "content": "Question"}},
                    {
                        "parent": 0,
                        "message": {"role": "assistant", "content": "Answer"},
                        "sampled": True,
                        "token_ids": [10, 11],
                        "mask": [False, True],
                        "logprobs": [-0.123456789123],
                    },
                ],
            }
        ],
    }
    native.WireEpisode.model_validate(record)
    path = tmp_path / "episodes.jsonl"
    original = json.dumps(record) + "\n"
    path.write_text(original)
    data = supervised_from_verifiers_jsonl(path, dataset_id="toy", revision="1")
    assert data.examples[0].metadata["episode_id"] == "episode-a"
    assert data.examples[0].metadata["reward"] == 1.0
    assert data.examples[0].trainable_message_indices == (1,)
    assert path.read_text() == original
