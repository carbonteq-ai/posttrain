from __future__ import annotations

import pytest
from posttrain_release.image_plan import DesiredImage, ObservedImage, plan_release_images


def _base(*, forced: bool = False) -> DesiredImage:
    return DesiredImage("base", "posttrain-base", "a" * 64, "b" * 64, forced=forced)


def _kind(*, forced: bool = False) -> DesiredImage:
    return DesiredImage(
        "kind.trl", "posttrain-kind-online-rl-trl-py312", "c" * 64, "d" * 64, parent="base", forced=forced
    )


def _present(desired: DesiredImage, marker: str, *, bytes: int = 0, missing: int | None = None) -> ObservedImage:
    return ObservedImage(
        "present",
        digest="sha256:" + marker * 64,
        identity=desired.identity,
        logical_bytes=bytes,
        missing_blob_bytes=missing,
    )


def test_plan_reuses_matching_destination_without_transfer() -> None:
    base = _base()
    kind = _kind()
    plan = plan_release_images(
        (base, kind),
        source={"base": _present(base, "a"), "kind.trl": _present(kind, "b")},
        destination={"base": _present(base, "a"), "kind.trl": _present(kind, "b")},
    )

    assert [node.action for node in plan.nodes] == ["reuse-remote", "reuse-remote"]
    assert plan.known_transfer_bytes == 0
    assert not plan.has_unknown_transfer_bytes


def test_plan_copies_only_the_missing_destination_nodes() -> None:
    base = _base()
    kind = _kind()
    plan = plan_release_images(
        (base, kind),
        source={
            "base": _present(base, "a", bytes=100, missing=60),
            "kind.trl": _present(kind, "b", bytes=200, missing=75),
        },
        destination={"base": _present(base, "a"), "kind.trl": ObservedImage("missing")},
    )

    assert [node.action for node in plan.nodes] == ["reuse-remote", "copy"]
    assert plan.node("kind.trl").estimated_transfer_bytes == 75
    assert plan.known_transfer_bytes == 75


def test_plan_makes_base_change_fan_out_explicit() -> None:
    old_base = _base()
    new_base = DesiredImage("base", "posttrain-base", "e" * 64, "b" * 64)
    kind = _kind()
    plan = plan_release_images(
        (new_base, kind),
        source={"base": _present(old_base, "a"), "kind.trl": _present(kind, "b", bytes=200)},
        destination={},
    )

    assert [node.action for node in plan.nodes] == ["build", "build"]
    assert "different immutable inputs" in plan.node("base").reason
    assert "parent base" in plan.node("kind.trl").reason
    assert plan.has_unknown_transfer_bytes


def test_plan_blocks_without_build_permission() -> None:
    base = _base()
    plan = plan_release_images(
        (base,),
        source={"base": ObservedImage("missing")},
        destination={"base": ObservedImage("missing")},
        allow_build=False,
    )

    assert plan.node("base").action == "blocked"
    assert "building is disabled" in plan.node("base").reason
    assert plan.blocked


def test_plan_rejects_children_before_their_parent() -> None:
    with pytest.raises(ValueError, match="must appear first"):
        plan_release_images((_kind(), _base()), source={}, destination={})
