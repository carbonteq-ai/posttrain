"""Renderer-owned tokenization and loss attribution for training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from posttrain.common import ModelProfile

from .data import PreferenceDataset, SupervisedDataset
from .profiles import RendererProfile


@dataclass(frozen=True, slots=True)
class RenderedSFTExample:
    id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RenderedPreferenceExample:
    id: str
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]


def create_renderer(tokenizer: Any, model: ModelProfile, profile: RendererProfile) -> Any:
    """Create the pinned renderer while honoring the shared conversation contract."""

    try:
        from renderers import (  # pyright: ignore[reportMissingImports]
            DefaultRendererConfig,
            Qwen35RendererConfig,
        )
        from renderers import (
            create_renderer as create,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    if model.family != profile.model_family:
        raise ValueError("renderer profile is incompatible with the model family")
    template = model.conversation.chat_template.text()
    if template is not None:
        tokenizer.chat_template = template
    mode = model.conversation.reasoning_mode(profile.reasoning_mode)
    if profile.implementation == "qwen3.5":
        enable_thinking = mode.kwargs().get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError("Qwen enable_thinking must be a boolean")
        config = Qwen35RendererConfig(enable_thinking=enable_thinking)
    else:
        template_kwargs = cast(dict[str, Any], mode.kwargs())
        config = DefaultRendererConfig(
            **template_kwargs,
        )
    return create(tokenizer, config)


def render_supervised(
    tokenizer: Any,
    model: ModelProfile,
    dataset: SupervisedDataset,
    profile: RendererProfile,
    *,
    max_length: int,
) -> tuple[RenderedSFTExample, ...]:
    try:
        from renderers import build_training_sample  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    renderer = create_renderer(tokenizer, model, profile)
    rendered: list[RenderedSFTExample] = []
    for example in dataset.examples:
        sample = build_training_sample(
            renderer,
            cast(list[Any], example.messages()),
            role_to_mask=lambda message: message.get("role") == "assistant",
            ensure_final_stop=True,
        )
        input_ids = tuple(sample.token_ids[:max_length])
        loss_mask = tuple(sample.loss_mask[:max_length])
        if len(input_ids) != len(loss_mask) or not any(loss_mask):
            raise ValueError(f"supervised example {example.id!r} has no trainable tokens after rendering")
        labels = tuple(token if include else -100 for token, include in zip(input_ids, loss_mask, strict=True))
        rendered.append(RenderedSFTExample(example.id, input_ids, labels))
    return tuple(rendered)


def render_preferences(
    tokenizer: Any,
    model: ModelProfile,
    dataset: PreferenceDataset,
    profile: RendererProfile,
    *,
    max_length: int,
) -> tuple[RenderedPreferenceExample, ...]:
    renderer = create_renderer(tokenizer, model, profile)
    rendered: list[RenderedPreferenceExample] = []
    for example in dataset.examples:
        prompt_messages = example.prompt_messages()
        prompt_ids = tuple(renderer.render_ids(prompt_messages, add_generation_prompt=True))
        chosen_full = tuple(renderer.render_ids([*prompt_messages, {"role": "assistant", "content": example.chosen}]))
        rejected_full = tuple(
            renderer.render_ids([*prompt_messages, {"role": "assistant", "content": example.rejected}])
        )
        if chosen_full[: len(prompt_ids)] != prompt_ids or rejected_full[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"preference example {example.id!r} violates renderer prompt-prefix equality")
        chosen_ids = chosen_full[len(prompt_ids) :]
        rejected_ids = rejected_full[len(prompt_ids) :]
        if not chosen_ids or not rejected_ids:
            raise ValueError(f"preference example {example.id!r} rendered an empty completion")
        if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_length:
            raise ValueError(f"preference example {example.id!r} exceeds max_length; curate or raise the profile limit")
        rendered.append(RenderedPreferenceExample(example.id, prompt_ids, chosen_ids, rejected_ids))
    return tuple(rendered)


__all__ = [
    "RenderedPreferenceExample",
    "RenderedSFTExample",
    "create_renderer",
    "render_preferences",
    "render_supervised",
]
