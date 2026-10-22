import copy

import pytest
from compose_sdpo_prepared_qualification import compose, runtime_sources
from sdpo_shared_base import RUNTIME_FILES


def fixture():
    tasks = [f"simple.new{i}" for i in range(20)]
    old = {"qualified": True, "optimizer_updates": 0, "method": "none",
           "prefix_caching": False, "chunked_prefill": False,
           "actor_capacity": {"qualified": True, "student_tokens": 32768, "scored_target_tokens": 4096},
           "stochastic_native": {"tasks": ["OLD"]}}
    greedy = {"qualified": True, "compared_pairs": 40}
    native = {"qualified": True, "optimizer_updates": 0, "policy_version": 0,
              "initial_adapter_sha256": "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c",
              "tasks": tasks, "native_episodes": 80, "greedy_episodes": 40, "stochastic_episodes": 40,
              "native_greedy_repeatability": greedy,
              "stochastic_native": {"qualified": True, "episodes": 40, "temperature": 1.,
                  "tasks": tasks, "optimizer_updates": 0, "native_greedy_repeatability": greedy}}
    return old, native, tasks


def test_new_population_replaces_old_population_without_mutating_inherited_capacity():
    old, native, tasks = fixture()
    before = copy.deepcopy(old)
    result = compose(old, native, tasks)
    assert result["stochastic_native"]["tasks"] == tasks
    assert result["actor_capacity"] == old["actor_capacity"] and old == before
    assert "separately replay-audited" in result["scope"]


@pytest.mark.parametrize("problem", ["old_tasks", "partial_population", "trained", "wrong_weights", "no_repeatability"])
def test_old_partial_trained_or_wrong_weight_evidence_rejected(problem):
    old, native, tasks = fixture()
    if problem == "old_tasks":
        native["stochastic_native"]["tasks"] = ["OLD"]
    elif problem == "partial_population":
        native["native_episodes"] = 79
    elif problem == "trained":
        native["policy_version"] = 1
    elif problem == "wrong_weights":
        native["initial_adapter_sha256"] = "trained-or-other-initialization"
    else:
        native["native_greedy_repeatability"]["compared_pairs"] = 0
    with pytest.raises(ValueError):
        compose(old, native, tasks)


def test_only_declared_preparation_changes_may_reuse_runtime_evidence(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for name in {*RUNTIME_FILES, "sdpo_capacity.py", "sdpo_agent_ablation.py", "sdpo_agent_config.py"}:
        for root in (old, new):
            (root / name).write_text("# immutable runtime\n")
    (old / "prepare_sdpo_guidance_five.py").write_text("# older preparation\n")
    (new / "prepare_sdpo_guidance_five.py").write_text("# new preparation\n")
    (new / "qualify_sdpo_prepared_tasks.py").write_text("# new task-only qualification\n")
    assert runtime_sources(old, new)["preparation_only_changes"] == [
        "prepare_sdpo_guidance_five.py", "qualify_sdpo_prepared_tasks.py"]
    (new / "sdpo_capacity.py").write_text("# changed capacity implementation\n")
    with pytest.raises(ValueError, match="runtime source differs"):
        runtime_sources(old, new)
