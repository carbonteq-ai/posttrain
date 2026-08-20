"""Tests for backend-neutral media references on supervised examples."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from posttrain.data import (
    SupervisedDataset,
    SupervisedExample,
    SupervisedMedia,
    supervised_from_huggingface,
    supervised_from_nemo,
    to_huggingface_sft_rows,
    to_nemo_sft_rows,
)

_FIRST_DIGEST = "1" * 64
_SECOND_DIGEST = "2" * 64


def _example(*media: SupervisedMedia) -> SupervisedExample:
    return SupervisedExample(
        id="documents/example",
        messages=(
            {"role": "user", "content": "Extract this policy."},
            {"role": "assistant", "content": '{"title":"Example"}'},
        ),
        trainable_message_indices=(1,),
        media=media,
    )


def test_supervised_media_is_immutable_and_preserves_page_order() -> None:
    first = SupervisedMedia(
        path="assets/documents/example/page-0001.png",
        sha256=_FIRST_DIGEST,
        mime_type="image/png",
        metadata={"page_number": 1},
    )
    second = SupervisedMedia(
        path="assets/documents/example/page-0002.png",
        sha256=_SECOND_DIGEST,
        mime_type="image/png",
        metadata={"page_number": 2},
    )

    example = _example(second, first)

    assert example.media == (second, first)
    assert [item.path for item in example.media] == [second.path, first.path]
    with pytest.raises(FrozenInstanceError):
        first.path = "assets/replaced.png"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.metadata["page_number"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "path",
    (
        "",
        "/assets/page.png",
        "page.png",
        "data/page.png",
        "assets/../page.png",
        "assets//page.png",
        "assets/page.png/",
        "assets\\page.png",
    ),
)
def test_supervised_media_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative POSIX paths below assets"):
        SupervisedMedia(path=path, sha256=_FIRST_DIGEST, mime_type="image/png")


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ({"sha256": "A" * 64}, "64 lowercase hexadecimal"),
        ({"sha256": "1" * 63}, "64 lowercase hexadecimal"),
        ({"mime_type": "application/pdf"}, "image/jpeg, image/png, or image/webp"),
        ({"kind": "audio"}, "kind must be image"),
    ),
)
def test_supervised_media_rejects_invalid_identity(values: dict[str, str], message: str) -> None:
    arguments = {
        "path": "assets/documents/example/page-0001.png",
        "sha256": _FIRST_DIGEST,
        "mime_type": "image/png",
        **values,
    }
    with pytest.raises(ValueError, match=message):
        SupervisedMedia(**arguments)  # type: ignore[arg-type]


def test_supervised_example_rejects_duplicate_media_paths() -> None:
    first = SupervisedMedia("assets/documents/example/page.png", _FIRST_DIGEST, "image/png")
    conflicting = SupervisedMedia("assets/documents/example/page.png", _SECOND_DIGEST, "image/png")

    with pytest.raises(ValueError, match="paths must be unique"):
        _example(first, conflicting)


def test_huggingface_media_round_trip_preserves_order_and_metadata() -> None:
    dataset = SupervisedDataset(
        "datasets/visual",
        "revision",
        (
            _example(
                SupervisedMedia(
                    "assets/documents/example/page-0002.webp",
                    _SECOND_DIGEST,
                    "image/webp",
                    metadata={"page_number": 2},
                ),
                SupervisedMedia(
                    "assets/documents/example/page-0001.jpg",
                    _FIRST_DIGEST,
                    "image/jpeg",
                    metadata={"page_number": 1},
                ),
            ),
        ),
    )

    rows = to_huggingface_sft_rows(dataset)
    restored = supervised_from_huggingface(
        rows,
        dataset_id="datasets/restored",
        revision="revision",
    )

    assert rows[0]["media"] == dataset.examples[0].media_records()
    assert restored.examples[0].media == dataset.examples[0].media


def test_nemo_media_round_trip_preserves_order() -> None:
    dataset = SupervisedDataset(
        "datasets/visual",
        "revision",
        (
            _example(
                SupervisedMedia("assets/documents/example/page-0001.png", _FIRST_DIGEST, "image/png"),
                SupervisedMedia("assets/documents/example/page-0002.png", _SECOND_DIGEST, "image/png"),
            ),
        ),
    )

    restored = supervised_from_nemo(
        to_nemo_sft_rows(dataset),
        dataset_id="datasets/restored",
        revision="revision",
    )

    assert restored.examples[0].media == dataset.examples[0].media


def test_media_serialization_is_deterministic() -> None:
    dataset = SupervisedDataset(
        "datasets/visual",
        "revision",
        (
            _example(
                SupervisedMedia(
                    "assets/documents/example/page-0001.png",
                    _FIRST_DIGEST,
                    "image/png",
                    metadata={"page_number": 1, "source": "rendered-pdf"},
                ),
            ),
        ),
    )

    row = to_huggingface_sft_rows(dataset)[0]
    first = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    second = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    assert first == second


def test_text_only_huggingface_export_keeps_the_existing_row_shape() -> None:
    dataset = SupervisedDataset("datasets/text", "revision", (_example(),))

    row = to_huggingface_sft_rows(dataset)[0]

    assert "media" not in row
