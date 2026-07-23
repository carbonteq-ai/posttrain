"""Adapters for common row schemas found in Hugging Face datasets."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from posttrain.common import JsonValue

from ..models import MessageRecord, PreferenceDataset, PreferenceExample, SupervisedDataset, SupervisedExample

type SFTFormat = Literal["auto", "messages", "prompt-completion", "alpaca", "sharegpt"]
type PreferenceFormat = Literal["auto", "trl", "tulu", "nemo-ranked"]

_CANONICAL_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _messages(value: Any, *, default_role: str, field: str) -> tuple[MessageRecord, ...]:
    if isinstance(value, str):
        return (cast(MessageRecord, {"role": default_role, "content": value}),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be text or a sequence of messages")
    output: list[MessageRecord] = []
    role_aliases = {"human": "user", "gpt": "assistant", "model": "assistant"}
    for item in value:
        message = _mapping(item, field=field)
        role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        if not isinstance(role, str):
            raise ValueError(f"{field} message is missing role/from")
        normalized = dict(message)
        normalized.pop("from", None)
        normalized.pop("value", None)
        normalized["role"] = role_aliases.get(role.lower(), role.lower())
        normalized["content"] = content
        output.append(cast(MessageRecord, normalized))
    return tuple(output)


def _tools(row: Mapping[str, Any]) -> tuple[Mapping[str, JsonValue], ...]:
    value = row.get("tools")
    if value is None:
        return ()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("tools must be a sequence of tool definitions")
    return tuple(cast(Mapping[str, JsonValue], _mapping(tool, field="tools")) for tool in value)


def _identity(row: Mapping[str, Any], index: int) -> tuple[str, dict[str, JsonValue]]:
    external_id = row.get("id")
    identifier = external_id if isinstance(external_id, str) and _CANONICAL_ID.fullmatch(external_id) else None
    raw_metadata = row.get("metadata")
    metadata = dict(cast(Mapping[str, JsonValue], raw_metadata)) if isinstance(raw_metadata, Mapping) else {}
    if identifier is None and isinstance(external_id, (str, int)):
        metadata["external_id"] = external_id
    return identifier or f"rows/{index:06d}", metadata


def _sft_format(row: Mapping[str, Any], requested: SFTFormat) -> SFTFormat:
    if requested != "auto":
        return requested
    if row.get("messages") is not None:
        return "messages"
    if row.get("prompt") is not None and row.get("completion") is not None:
        return "prompt-completion"
    if row.get("conversations") is not None:
        return "sharegpt"
    if row.get("instruction") is not None and row.get("output") is not None:
        return "alpaca"
    raise ValueError("could not detect SFT row format")


def supervised_from_huggingface(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    revision: str,
    format: SFTFormat = "auto",
    metadata: Mapping[str, JsonValue] | None = None,
) -> SupervisedDataset:
    """Normalize HF Dataset rows without making HF Dataset the domain contract."""

    examples: list[SupervisedExample] = []
    for index, row in enumerate(rows):
        selected = _sft_format(row, format)
        if selected == "messages":
            messages = _messages(row["messages"], default_role="user", field="messages")
        elif selected == "prompt-completion":
            messages = (
                *_messages(row["prompt"], default_role="user", field="prompt"),
                *_messages(row["completion"], default_role="assistant", field="completion"),
            )
        elif selected == "sharegpt":
            messages = _messages(row["conversations"], default_role="user", field="conversations")
        else:
            instruction = str(row["instruction"])
            input_text = str(row.get("input") or "").strip()
            prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
            messages = (
                cast(MessageRecord, {"role": "user", "content": prompt}),
                cast(MessageRecord, {"role": "assistant", "content": str(row["output"])}),
            )
        raw_trainable = row.get("trainable_message_indices")
        if raw_trainable is None:
            trainable = tuple(i for i, message in enumerate(messages) if message.get("role") == "assistant")
        elif isinstance(raw_trainable, Sequence) and not isinstance(raw_trainable, (str, bytes)):
            trainable = tuple(int(value) for value in raw_trainable)
        else:
            raise ValueError("trainable_message_indices must be a sequence of integers")
        if not trainable:
            raise ValueError(f"SFT row {index} has no assistant target")
        identifier, row_metadata = _identity(row, index)
        row_metadata.update({"source_row": index, "source_format": selected})
        examples.append(
            SupervisedExample(
                id=identifier,
                messages=messages,
                trainable_message_indices=trainable,
                tools=_tools(row),
                metadata=row_metadata,
            )
        )
    return SupervisedDataset(dataset_id, revision, tuple(examples), metadata=metadata or {})


def _preference_format(row: Mapping[str, Any], requested: PreferenceFormat) -> PreferenceFormat:
    if requested != "auto":
        return requested
    if row.get("context") is not None and row.get("completions") is not None:
        return "nemo-ranked"
    if row.get("prompt") is not None and row.get("chosen") is not None and row.get("rejected") is not None:
        return "trl"
    if row.get("chosen") is not None and row.get("rejected") is not None:
        return "tulu"
    raise ValueError("could not detect preference row format")


def _split_common_prefix(
    chosen: tuple[MessageRecord, ...], rejected: tuple[MessageRecord, ...]
) -> tuple[tuple[MessageRecord, ...], tuple[MessageRecord, ...], tuple[MessageRecord, ...]]:
    prefix = 0
    for preferred, dispreferred in zip(chosen, rejected, strict=False):
        if dict(preferred) != dict(dispreferred):
            break
        prefix += 1
    if prefix == 0 or prefix == len(chosen) or prefix == len(rejected):
        raise ValueError("implicit preference rows require a non-empty shared prompt and distinct continuations")
    return chosen[:prefix], chosen[prefix:], rejected[prefix:]


def preferences_from_huggingface(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    revision: str,
    format: PreferenceFormat = "auto",
    metadata: Mapping[str, JsonValue] | None = None,
) -> PreferenceDataset:
    """Normalize TRL, Tulu, or NeMo-ranked preference rows."""

    examples: list[PreferenceExample] = []
    for index, row in enumerate(rows):
        selected = _preference_format(row, format)
        chosen_score: float | None = None
        rejected_score: float | None = None
        if selected == "nemo-ranked":
            prompt = _messages(row["context"], default_role="user", field="context")
            completions = row["completions"]
            if not isinstance(completions, Sequence) or len(completions) < 2:
                raise ValueError("NeMo preference rows require at least two ranked completions")
            ranked = sorted(
                (_mapping(value, field="completions") for value in completions), key=lambda value: value["rank"]
            )
            chosen = _messages(ranked[0]["completion"], default_role="assistant", field="completion")
            rejected = _messages(ranked[-1]["completion"], default_role="assistant", field="completion")
            chosen_score = float(-int(ranked[0]["rank"]))
            rejected_score = float(-int(ranked[-1]["rank"]))
        elif selected == "trl":
            prompt = _messages(row["prompt"], default_role="user", field="prompt")
            chosen = _messages(row["chosen"], default_role="assistant", field="chosen")
            rejected = _messages(row["rejected"], default_role="assistant", field="rejected")
            if row.get("chosen_score") is not None and row.get("rejected_score") is not None:
                chosen_score = float(row["chosen_score"])
                rejected_score = float(row["rejected_score"])
        else:
            full_chosen = _messages(row["chosen"], default_role="assistant", field="chosen")
            full_rejected = _messages(row["rejected"], default_role="assistant", field="rejected")
            prompt, chosen, rejected = _split_common_prefix(full_chosen, full_rejected)
        identifier, row_metadata = _identity(row, index)
        row_metadata.update({"source_row": index, "source_format": selected})
        if isinstance(row.get("task_name"), str):
            row_metadata["task_name"] = row["task_name"]
        examples.append(
            PreferenceExample(
                id=identifier,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                tools=_tools(row),
                metadata=row_metadata,
            )
        )
    return PreferenceDataset(dataset_id, revision, tuple(examples), metadata=metadata or {})


def to_huggingface_sft_rows(dataset: SupervisedDataset) -> list[dict[str, Any]]:
    return [
        {
            "id": example.id,
            "messages": example.message_records(),
            "tools": example.tool_records(),
            "trainable_message_indices": list(example.trainable_message_indices),
            "metadata": dict(example.metadata),
        }
        for example in dataset.examples
    ]


def to_huggingface_preference_rows(dataset: PreferenceDataset) -> list[dict[str, Any]]:
    return [
        {
            "id": example.id,
            "prompt": example.prompt_records(),
            "chosen": example.chosen_records(),
            "rejected": example.rejected_records(),
            "tools": example.tool_records(),
            "chosen_score": example.chosen_score,
            "rejected_score": example.rejected_score,
            "metadata": dict(example.metadata),
        }
        for example in dataset.examples
    ]


__all__ = [
    "PreferenceFormat",
    "SFTFormat",
    "preferences_from_huggingface",
    "supervised_from_huggingface",
    "to_huggingface_preference_rows",
    "to_huggingface_sft_rows",
]
