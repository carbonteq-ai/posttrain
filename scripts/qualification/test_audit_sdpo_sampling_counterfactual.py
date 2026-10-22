import json

import pytest
from audit_sdpo_sampling_counterfactual import replay


def fixture(tmp_path):
    tasks = [f"task{i}" for i in range(20)]
    episodes, calls = [], []
    for i, name in enumerate(tasks):
        user = {"role": "user", "content": name}
        nodes = [{"message": user}]
        for j in range(2):
            tool = {"id": f"id{i}-{j}", "function": {"name": "read", "arguments": "{}"}}
            message = {"role": "assistant", "tool_calls": [tool]}
            nodes += [{"message": message}, {"message": {"role": "tool", "tool_call_id": tool["id"],
                                                       "name": "read", "content": '{"success":true}'}}]
            calls.append({"complete": True, "messages": [user], "message": message, "tools": []})
        episodes.append({"id": f"episode{i}", "task": {"data": {"name": name}},
                         "traces": [{"nodes": nodes, "metrics": {"task_completed_correctly": 1}}]})
    (tmp_path / "native").mkdir()
    for file, rows in [("generation-calls.jsonl", calls), ("native/traces.jsonl", episodes)]:
        (tmp_path / file).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    (tmp_path / "training-admission.json").write_text(json.dumps({"admitted": 40, "exclusions": []}))
    indices = [2 * i for i in range(20)] + [2 * i + 1 for i in range(12)]
    selected = [{"task": tasks[index // 2], "episode_id": f"episode{index // 2}", "generation_index": index}
                for index in indices]
    (tmp_path / "selected-transitions.json").write_text(json.dumps(selected))
    return tasks, {name: {"check_roles": {}, "check_signals": {}} for name in tasks}, selected


def test_replay_requires_the_actual_saved_selection(tmp_path):
    tasks, guidance, selected = fixture(tmp_path)
    result = replay(tmp_path, guidance, tasks, 1)
    assert result["saved_selection_exact"] and result["changed_slots"] == []
    selected[0]["generation_index"] = 999
    (tmp_path / "selected-transitions.json").write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="historical selection was not reproduced"):
        replay(tmp_path, guidance, tasks, 1)


def test_reject_a_changed_native_admission_population(tmp_path):
    tasks, guidance, _ = fixture(tmp_path)
    (tmp_path / "training-admission.json").write_text(json.dumps({"admitted": 39, "exclusions": []}))
    with pytest.raises(ValueError, match="saved admission differs"):
        replay(tmp_path, guidance, tasks, 1)
