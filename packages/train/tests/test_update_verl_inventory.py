"""Resolved native collection rejects synthetic task duplication before launch."""

import sys
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.train.backends.verl.contracts import VerlPayload
from posttrain.train.backends.verl.worker import _write_dataset


@pytest.mark.parametrize(
    ("identities", "budget", "message"),
    [(["a"], 2, "cannot fill distinct complete groups"),
     (["a", "a"], 2, "contains duplicate tasks"),
     ([""], 1, "nonempty task identities")],
)
def test_resolved_inventory_rejects_before_writing(monkeypatch, tmp_path, identities, budget, message):
    def unexpected_write(rows):
        pytest.fail("invalid inventory reached native dataset materialization")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        Dataset=SimpleNamespace(from_list=unexpected_write)))
    payload = SimpleNamespace(
        policy=SimpleNamespace(id="policy"),
        algorithm=SimpleNamespace(policy_updates=object(), num_prompts_per_step=budget),
        environment=SimpleNamespace(examples=[
            SimpleNamespace(id=identity, prompt="task", metadata={}) for identity in identities]),
    )
    with pytest.raises(ValueError, match=message):
        _write_dataset(cast(VerlPayload, payload), tmp_path / "population.parquet")


def test_resolved_inventory_preserves_distinct_native_rows(monkeypatch, tmp_path):
    captured = []

    def write(rows):
        captured.extend(rows)
        return SimpleNamespace(to_parquet=lambda path: None)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(from_list=write)))
    payload = SimpleNamespace(
        policy=SimpleNamespace(id="policy"),
        algorithm=SimpleNamespace(policy_updates=object(), num_prompts_per_step=2, adaptive_curriculum=None),
        environment=SimpleNamespace(examples=[
            SimpleNamespace(id=identity, prompt=identity, metadata={}) for identity in ("a", "b", "c")]),
    )
    _write_dataset(cast(VerlPayload, payload), tmp_path / "population.parquet")
    assert [row["example_id"] for row in captured] == ["a", "b", "c"]
