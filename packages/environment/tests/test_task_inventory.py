from __future__ import annotations

import pytest
from posttrain.environment import TaskDescriptor, TaskFacetValue, task_inventory_digest


def task(key: str, content: str = "a") -> TaskDescriptor:
    return TaskDescriptor(
        key=key,
        fingerprint="sha256:" + content * 64,
        split="test",
        facets=(TaskFacetValue("subject", "algebra"),),
        source_reference={"row": key},
    )


def test_inventory_digest_is_order_independent() -> None:
    assert task_inventory_digest((task("b", "b"), task("a"))) == task_inventory_digest(
        (task("a"), task("b", "b"))
    )


def test_inventory_rejects_duplicate_keys_and_content_collisions() -> None:
    with pytest.raises(ValueError, match="duplicate task keys"):
        task_inventory_digest((task("a"), task("a")))
    with pytest.raises(ValueError, match="different content"):
        task_inventory_digest((task("a"), task("a", "b")))


def test_task_facets_are_sorted_and_multi_valued() -> None:
    value = TaskDescriptor(
        key="task-1",
        fingerprint="c" * 64,
        facets=(
            TaskFacetValue("skill", "geometry"),
            TaskFacetValue("language", "English"),
            TaskFacetValue("skill", "algebra"),
        ),
    )
    assert value.values("skill") == ("algebra", "geometry")
    assert value.to_payload()["fingerprint"] == "sha256:" + "c" * 64
