"""Compatibility proof for the pinned TRL vision-language SFT contract."""

from __future__ import annotations

import importlib.metadata
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_PROJECT = tomllib.loads((_REPOSITORY_ROOT / "packages/train/pyproject.toml").read_text(encoding="utf-8"))
_PINNED_TRL_VERSION = _TRAIN_PROJECT["tool"]["posttrain"]["trl"]["version"]


def _pinned_multimodal_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    except ImportError:
        pytest.skip("TRL optional dependency is not installed")

    assert importlib.metadata.version("trl") == _PINNED_TRL_VERSION
    return torch, DataCollatorForVisionLanguageModeling


class _RecordingProcessor:
    """Small processor double that leaves TRL's real collator behavior visible."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self.templates: list[list[dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.templates.append(deepcopy(messages))
        if len(messages) == 1:
            assert kwargs["add_generation_prompt"] is True
            return "<prompt>"
        return "<prompt><completion>"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if "images" in kwargs:
            return {
                "input_ids": self._torch.tensor([[11, 12, 13]]),
                "attention_mask": self._torch.ones((1, 3), dtype=self._torch.long),
                "pixel_values": self._torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
            }
        return {
            "input_ids": self._torch.tensor([[21, 22]]),
            "attention_mask": self._torch.ones((1, 2), dtype=self._torch.long),
        }


def test_pinned_trl_collates_ordered_images_and_masks_only_the_prompt() -> None:
    torch, collator_type = _pinned_multimodal_runtime()
    processor = _RecordingProcessor(torch)
    first_page = object()
    second_page = object()
    example = {
        "images": [first_page, second_page],
        "prompt": [{"role": "user", "content": "Extract the policy as canonical JSON."}],
        "completion": [{"role": "assistant", "content": '{"title":"Example"}'}],
    }

    collator = collator_type(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )
    batch = collator([example])

    assert collator.max_length is None
    assert processor.calls[0]["images"] == [[first_page, second_page]]
    assert "max_length" not in processor.calls[0]
    prompt_blocks = processor.templates[0][0]["content"]
    assert [block["image"] for block in prompt_blocks[:2]] == [first_page, second_page]
    assert prompt_blocks[2] == {"type": "text", "text": "Extract the policy as canonical JSON."}
    assert batch["pixel_values"].shape == (1, 1, 2, 2)
    assert batch["input_ids"].tolist() == [[11, 12, 13, 21, 22]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 21, 22]]


@pytest.mark.network
def test_pinned_trl_collates_with_the_exact_gemma_e4b_processor() -> None:
    if os.environ.get("POSTTRAIN_RUN_GEMMA4_MULTIMODAL_PROBE") != "1":
        pytest.skip("set POSTTRAIN_RUN_GEMMA4_MULTIMODAL_PROBE=1 to run the exact Gemma E4B processor gate")

    torch, collator_type = _pinned_multimodal_runtime()
    transformers = pytest.importorskip("transformers")
    image_module = pytest.importorskip("PIL.Image")
    from posttrain.common.variants import GEMMA_4_E4B_IT

    processor = transformers.AutoProcessor.from_pretrained(
        GEMMA_4_E4B_IT.base.repo_id,
        revision=GEMMA_4_E4B_IT.base.revision,
        trust_remote_code=False,
    )
    pages = [
        image_module.new("RGB", (32, 32), color="red"),
        image_module.new("RGB", (32, 32), color="blue"),
    ]
    collator = collator_type(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )

    batch = collator(
        [
            {
                "images": pages,
                "prompt": [{"role": "user", "content": "Extract the policy as canonical JSON."}],
                "completion": [{"role": "assistant", "content": '{"title":"Example"}'}],
            }
        ]
    )

    labels = batch["labels"]
    assert collator.max_length is None
    assert isinstance(batch["pixel_values"], torch.Tensor)
    assert batch["pixel_values"].numel() > 0
    assert torch.any(labels == -100)
    assert torch.any(labels != -100)
