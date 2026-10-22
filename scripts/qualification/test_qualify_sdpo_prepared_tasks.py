import copy
import json

import pytest
from qualify_sdpo_prepared_tasks import normalization, retained_population


def fixture():
    sources = [{"task": f"simple.task{i}", "material": {"native_data": {
        "initial_state": {"raw": i}}}} for i in range(20)]
    episodes = [{"id": f"{i}-{r}", "task": {"data": {
        "name": f"simple.task{i}", "initial_state": {"raw": i, "normalized": True}}}}
        for i in range(20) for r in range(2)]
    return {"rows": sources}, episodes


def normalize(value):
    value["normalized"] = True


def test_exact_new_population_and_actual_constructor_side_effects():
    materials, episodes = fixture()
    before = copy.deepcopy(materials)
    result = normalization(materials, episodes, normalize, "pinned-source-hash")
    assert result["qualified"] and result["native_episodes"] == 40
    assert materials == before
    assert len(result["normalized_states"]) == 20


def test_native_initial_state_is_read_from_reconciled_archive_not_sparse_wire(tmp_path):
    folder = tmp_path / "native"
    folder.mkdir()
    native = {"id": "episode1", "task": {"data": {"name": "task1", "initial_state": {"source": True}}}}
    wire = [{"id": "episode1", "task": {"data": {"name": "task1"}}}]
    (folder / "traces.jsonl").write_text(json.dumps(native) + "\n")
    checked = []

    def reconcile(sparse, retained):
        assert sparse == wire and retained == [native]
        checked.append(True)

    assert retained_population(tmp_path, wire, reconcile) == [native]
    assert checked == [True]


def test_rejected_native_identity_join_cannot_supply_initial_states(tmp_path):
    (tmp_path / "native").mkdir()
    (tmp_path / "native/traces.jsonl").write_text('{"id":"wrong"}\n')

    def reject(*args):
        raise ValueError("identity differs")

    with pytest.raises(ValueError, match="identity differs"):
        retained_population(tmp_path, [{"id": "expected"}], reject)


@pytest.mark.parametrize("problem", ["duplicate_id", "old_task", "missing", "different_state"])
def test_old_or_incomplete_or_drifting_native_population_rejected(problem):
    materials, episodes = fixture()
    if problem == "duplicate_id":
        episodes[1]["id"] = episodes[0]["id"]
    elif problem == "old_task":
        episodes[0]["task"]["data"]["name"] = "simple.old"
    elif problem == "missing":
        episodes.pop()
    else:
        episodes[0]["task"]["data"]["initial_state"]["normalized"] = False
    with pytest.raises(ValueError):
        normalization(materials, episodes, normalize, "pinned-source-hash")
