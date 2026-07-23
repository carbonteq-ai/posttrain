"""Tests for the pinned bounded Smol-SmolTalk source."""

from __future__ import annotations

import posttrain_lab.data.smol_smoltalk as smol_smoltalk
from posttrain_lab.data import SMOL_SMOLTALK_REVISION, SmolSmolTalkSupervisedSource


def test_smoltalk_source_normalizes_messages_and_preserves_source_label(monkeypatch) -> None:
    monkeypatch.setattr(
        smol_smoltalk,
        "_rows",
        lambda offset, count: [
            {
                "id": f"smol-smoltalk/train/{offset:06d}",
                "messages": (
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Answer"},
                ),
                "source": "fixture-source",
                "metadata": {"source_row": offset, "source": "fixture-source"},
            }
        ],
    )
    source = SmolSmolTalkSupervisedSource(count=1, offset=4)

    dataset = source.load()

    assert dataset.revision == SMOL_SMOLTALK_REVISION
    assert dataset.id == "smol-smoltalk/train-prefix-filtered-4-5-v1"
    assert dataset.examples[0].id == "smol-smoltalk/train/000004"
    assert dataset.examples[0].trainable_message_indices == (1,)
    assert dataset.examples[0].metadata["source"] == "fixture-source"
    assert dataset.metadata["repository"] == "HuggingFaceTB/smol-smoltalk"
    assert dataset.metadata["selection_filter"] == "bounded-conversation-characters-v1"
