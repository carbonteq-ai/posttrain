"""Tests for the compact, research-only AutomationBench replay."""

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).parents[1] / "automationbench_replay.py"
SPEC = importlib.util.spec_from_file_location("automationbench_replay", MODULE)
assert SPEC and SPEC.loader
replay_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay_module
SPEC.loader.exec_module(replay_module)


def _group(group_id: str, task: str, domain: str, useful: int, step: int = 1) -> dict:
    return {
        "group_id": group_id,
        "task_id": task,
        "class": domain,
        "step": step,
        "size": 4,
        "useful": useful,
        "input_tokens": 400,
        "output_tokens": 100,
        "truncated_rollouts": 0,
        "error_rollouts": 0,
    }


def test_compact_trace_omits_raw_episode() -> None:
    raw = {
        "metadata": {"reward": 0.5, "example_id": "train/1", "domain": "sales", "optimizer_step": 2},
        "payload": {
            "info": {"posttrain_prompt_group_id": "step/2/group/1"},
            "input_tokens": 100,
            "completion_tokens": 20,
            "task": {"data": {"prompt": "private prompt", "initial_state": {"secret": "private"}}},
        },
    }
    compact = replay_module.compact_trace(raw)
    assert compact is not None
    assert "private" not in str(compact)
    assert compact["output_tokens"] == 20


def test_group_grain_and_variance() -> None:
    rows = [
        {
            "group_id": "g1",
            "task_id": "t1",
            "class": "c",
            "step": 1,
            "reward": reward,
            "input_tokens": 10,
            "output_tokens": 5,
            "truncated": False,
            "error": False,
        }
        for reward in (0.0, 0.0, 1.0, 1.0)
    ]
    groups, issues = replay_module.group_traces(rows)
    assert len(groups) == 1
    assert groups[0]["useful"] == 1
    assert groups[0]["input_tokens"] == 40
    assert issues["non_four_groups"] == 0


def test_prediction_same_scale_and_prequential_calibration() -> None:
    groups = [_group("g1", "a", "c", 1), _group("g2", "a", "c", 0, 2)]
    calibration = replay_module.calibration(groups)
    assert calibration["groups"] == 2
    assert calibration["known_task_groups"] == 1
    assert 0 <= calibration["brier_hierarchical"] <= 1
    cross_run = replay_module.cross_run_calibration(groups[:1], groups[1:])
    assert cross_run["target_groups"] == 1
    assert cross_run["shared_task_groups"] == 1
    assert 0 <= cross_run["brier_hierarchical"] <= 1
    profile = replay_module.observed_profile(groups)
    assert profile["new_groups"] == 1
    assert profile["known_groups"] == 1


def test_paired_replay_accounting_and_determinism() -> None:
    groups = [_group(f"g{i}", f"t{i % 6}", "c", i % 2) for i in range(12)]
    a = replay_module.replay(groups, "posterior_sample", 7, steps=3, target_groups=2, max_rounds=2)
    b = replay_module.replay(groups, "posterior_sample", 7, steps=3, target_groups=2, max_rounds=2)
    assert a == b
    assert a["candidate_groups"] == a["new_candidates"] + a["reuse_candidates"]
    assert a["useful_groups"] <= a["candidate_groups"]
    assert a["completed_steps"] <= 3


def test_incomplete_groups_are_excluded_from_fit() -> None:
    complete = [_group(f"g{i}", f"t{i}", "c", i % 2) for i in range(8)]
    incomplete = _group("g2", "b", "c", 0)
    incomplete["size"] = 1
    snapshot = {"runs": [{"name": "run", "groups": [*complete, incomplete], "group_count": 9}]}
    result = replay_module.simulate(snapshot, seeds=1)["runs"][0]
    assert result["observed_groups"] == 8
    assert result["excluded_incomplete_groups"] == 1
    assert not result["replayable_full_rounds"]
    assert result["policies"] == {}
