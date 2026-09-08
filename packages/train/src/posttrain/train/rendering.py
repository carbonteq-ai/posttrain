"""Renderer-owned tokenization and loss attribution for training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from posttrain.common import ModelVariant
from posttrain.data import PreferenceDataset, SupervisedDataset

from .profiles import TrainingRenderer


@dataclass(frozen=True, slots=True)
class RenderedSFTExample:
    id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    source_length: int
    source_supervised_tokens: int


@dataclass(frozen=True, slots=True)
class RenderedPreferenceExample:
    id: str
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]


def bridge_lfm25_tool_cycle(
    renderer: Any,
    tokenizer: Any,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[dict[str, Any]],
) -> Any | None:
    """Extend an LFM tool cycle without retokenizing sampled assistant tokens.

    LFM's Jinja template appends a newline after an assistant ``im_end`` when it
    renders history. That newline is not part of a stop-terminated completion,
    so a full rerender changes the final BPE boundary and forks Verifiers' token
    graph. Render only the new tool messages, remove their standalone BOS, and
    join them to the retained sampled prefix with the template's exact newline.
    """

    if not new_messages or any(message.get("role") != "tool" for message in new_messages):
        return None
    stop_ids = frozenset(int(value) for value in renderer.get_stop_token_ids())
    if not previous_completion_ids or previous_completion_ids[-1] not in stop_ids:
        return None
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if not isinstance(bos_token_id, int):
        return None
    suffix = renderer.render(new_messages, tools=None, add_generation_prompt=True)
    if not suffix.token_ids or int(suffix.token_ids[0]) != bos_token_id:
        return None
    newline_ids = [int(value) for value in tokenizer.encode("\n", add_special_tokens=False)]
    if not newline_ids:
        return None
    try:
        from renderers import RenderedTokens  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error
    prefix_length = len(previous_prompt_ids) + len(previous_completion_ids) + len(newline_ids)
    suffix_is_content = list(suffix.is_content[1:]) if suffix.is_content else []
    suffix_sampled = list(suffix.sampled_mask[1:]) if suffix.sampled_mask else []
    return RenderedTokens(
        token_ids=[*previous_prompt_ids, *previous_completion_ids, *newline_ids, *suffix.token_ids[1:]],
        message_indices=[-1] * prefix_length + list(suffix.message_indices[1:]),
        sampled_mask=([False] * prefix_length + suffix_sampled) if suffix_sampled else [],
        is_content=([False] * prefix_length + suffix_is_content) if suffix_is_content else [],
        message_roles=list(suffix.message_roles),
        message_tool_names=list(suffix.message_tool_names),
    )


def create_renderer_config(model: ModelVariant, renderer: TrainingRenderer) -> Any:
    """Create the serializable renderer config for a model contract."""
    try:
        from renderers import (  # pyright: ignore[reportMissingImports]
            DefaultRendererConfig,
            Qwen35RendererConfig,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    if model.family != renderer.model_family:
        raise ValueError("training renderer is incompatible with the model family")
    mode = model.conversation.reasoning_mode(renderer.reasoning_mode)
    if renderer.implementation == "qwen3.5":
        enable_thinking = mode.kwargs().get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError("Qwen enable_thinking must be a boolean")
        config = Qwen35RendererConfig(enable_thinking=enable_thinking)
    else:
        template_kwargs = cast(dict[str, Any], mode.kwargs())
        config = DefaultRendererConfig(
            **template_kwargs,
        )
    return config


def create_renderer(tokenizer: Any, model: ModelVariant, renderer: TrainingRenderer) -> Any:
    """Create the pinned renderer while honoring the shared conversation contract."""

    try:
        from renderers import (  # pyright: ignore[reportMissingImports]
            create_renderer as create,
        )
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    template = model.conversation.chat_template.text()
    if template is not None:
        tokenizer.chat_template = template
    config = create_renderer_config(model, renderer)
    return create(tokenizer, config)


def render_supervised(
    tokenizer: Any,
    model: ModelVariant,
    dataset: SupervisedDataset,
    renderer: TrainingRenderer,
    *,
    max_length: int,
) -> tuple[RenderedSFTExample, ...]:
    try:
        from renderers import build_training_sample  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    active_renderer = create_renderer(tokenizer, model, renderer)
    rendered: list[RenderedSFTExample] = []
    for example in dataset.examples:
        messages = example.message_records()
        tools = example.tool_records()
        trainable = {id(messages[index]) for index in example.trainable_message_indices}
        sample = build_training_sample(
            active_renderer,
            cast(list[Any], messages),
            role_to_mask=lambda message, indices=trainable: id(message) in indices,
            tools=cast(list[Any], tools) or None,
            ensure_final_stop=True,
        )
        source_ids = tuple(sample.token_ids)
        source_loss_mask = tuple(sample.loss_mask)
        input_ids = source_ids[:max_length]
        loss_mask = source_loss_mask[:max_length]
        if len(input_ids) != len(loss_mask) or not any(loss_mask):
            raise ValueError(f"supervised example {example.id!r} has no trainable tokens after rendering")
        labels = tuple(token if include else -100 for token, include in zip(input_ids, loss_mask, strict=True))
        rendered.append(
            RenderedSFTExample(
                example.id,
                input_ids,
                labels,
                len(source_ids),
                sum(source_loss_mask),
            )
        )
    return tuple(rendered)


def render_preferences(
    tokenizer: Any,
    model: ModelVariant,
    dataset: PreferenceDataset,
    renderer: TrainingRenderer,
    *,
    max_length: int,
) -> tuple[RenderedPreferenceExample, ...]:
    active_renderer = create_renderer(tokenizer, model, renderer)
    rendered: list[RenderedPreferenceExample] = []
    for example in dataset.examples:
        prompt_messages = example.prompt_records()
        chosen_messages = example.chosen_records()
        rejected_messages = example.rejected_records()
        tools = cast(list[Any], example.tool_records()) or None
        prompt_ids = tuple(active_renderer.render_ids(prompt_messages, tools=tools, add_generation_prompt=True))
        chosen_full = tuple(active_renderer.render_ids([*prompt_messages, *chosen_messages], tools=tools))
        rejected_full = tuple(active_renderer.render_ids([*prompt_messages, *rejected_messages], tools=tools))
        if chosen_full[: len(prompt_ids)] != prompt_ids or rejected_full[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"preference example {example.id!r} violates renderer prompt-prefix equality")
        chosen_ids = chosen_full[len(prompt_ids) :]
        rejected_ids = rejected_full[len(prompt_ids) :]
        if not chosen_ids or not rejected_ids:
            raise ValueError(f"preference example {example.id!r} rendered an empty completion")
        if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_length:
            raise ValueError(
                f"preference example {example.id!r} exceeds max_length; curate or raise the settings limit"
            )
        rendered.append(RenderedPreferenceExample(example.id, prompt_ids, chosen_ids, rejected_ids))
    return tuple(rendered)


__all__ = [
    "RenderedPreferenceExample",
    "RenderedSFTExample",
    "bridge_lfm25_tool_cycle",
    "create_renderer",
    "create_renderer_config",
    "render_preferences",
    "render_supervised",
]
