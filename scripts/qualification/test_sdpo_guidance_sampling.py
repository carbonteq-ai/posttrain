from collections import Counter

from sdpo_guidance_sampling import batch


def test_all_twenty_tasks_have_equal_weight_and_terminal_opportunity_over_ten_updates():
    tasks = [f"task{i}" for i in range(20)]
    rows = [{"task": name, "episode_id": f"{name}-{replica}", "generation_index": turn,
             "terminal_text": bool(turn), "native_success": 1, "executed_tool_responses": 1}
            for name in tasks for replica in range(2) for turn in range(2)]
    visits, terminals = Counter(), Counter()
    for step in range(1, 11):
        selected = batch(rows, tasks, step)
        assert len(selected) == 32 and {r["task"] for r in selected} == set(tasks)
        assert len({(r["episode_id"], r["generation_index"]) for r in selected}) == 32
        visits.update(r["task"] for r in selected)
        terminals.update(r["task"] for r in selected if r["terminal_text"])
    assert set(visits.values()) == {16}
    assert set(terminals.values()) == {6}
