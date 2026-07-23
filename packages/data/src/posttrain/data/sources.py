"""Reusable transformations between canonical post-training data sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from posttrain.common import JsonValue

from .models import (
    DatasetDescriptor,
    MessageRecord,
    PreferenceDataset,
    PreferenceExample,
    SupervisedDataSource,
    SupervisedExample,
)


@dataclass(frozen=True, slots=True)
class ScoredContinuation:
    """A candidate produced by any evaluator, model, or human source."""

    example_id: str
    messages: tuple[MessageRecord, ...]
    score: float
    trace_id: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages or not self.example_id.strip():
            raise ValueError("scored continuations require an example id and messages")
        if self.trace_id is not None and not self.trace_id.strip():
            raise ValueError("trace lineage cannot be an empty string")
        object.__setattr__(self, "messages", tuple(MappingProxyType(dict(message)) for message in self.messages))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PreferencePairSource:
    """Pair demonstration targets with lower-scoring continuations by stable ID."""

    demonstrations: SupervisedDataSource
    candidates: tuple[ScoredContinuation, ...]
    chosen_score: float = 1.0
    id_suffix: str = "preferences"

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("preference pairing requires at least one candidate")
        ids = tuple(candidate.example_id for candidate in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("preference candidates require unique example ids")
        if not self.id_suffix or "/" in self.id_suffix:
            raise ValueError("preference source id suffix must be one stable path segment")

    @property
    def descriptor(self) -> DatasetDescriptor:
        source = self.demonstrations.descriptor
        digest = hashlib.sha256(source.revision.encode())
        for candidate in sorted(self.candidates, key=lambda value: value.example_id):
            digest.update(candidate.example_id.encode())
            digest.update(json.dumps([dict(message) for message in candidate.messages], sort_keys=True).encode())
            digest.update(repr(candidate.score).encode())
            digest.update((candidate.trace_id or "").encode())
        return DatasetDescriptor(
            id=f"{source.id}/{self.id_suffix}",
            revision=digest.hexdigest(),
            kind="preference",
            metadata={
                "derivation": "supervised-target-vs-scored-continuation",
                "source_dataset_id": source.id,
                "source_dataset_revision": source.revision,
                "candidate_trace_ids": [
                    candidate.trace_id for candidate in self.candidates if candidate.trace_id is not None
                ],
                "candidate_sources": [dict(candidate.metadata) for candidate in self.candidates],
            },
            num_examples=len(self.candidates),
        )

    @property
    def id(self) -> str:
        return self.descriptor.id

    @property
    def revision(self) -> str:
        return self.descriptor.revision

    def load(self) -> PreferenceDataset:
        demonstrations = self.demonstrations.load()
        if demonstrations.descriptor != self.demonstrations.descriptor:
            raise ValueError("supervised source descriptor changed while materializing preference data")
        candidates = {candidate.example_id: candidate for candidate in self.candidates}
        examples: list[PreferenceExample] = []
        for demonstration in demonstrations.examples:
            candidate = candidates.get(demonstration.id)
            if candidate is None:
                continue
            prompt, chosen = _split_supervised_target(demonstration)
            if self.chosen_score <= candidate.score:
                raise ValueError(f"candidate {candidate.example_id!r} is not worse than the chosen target")
            examples.append(
                PreferenceExample(
                    id=demonstration.id,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=candidate.messages,
                    chosen_score=self.chosen_score,
                    rejected_score=candidate.score,
                    rejected_trace_id=candidate.trace_id,
                    tools=demonstration.tools,
                    metadata=candidate.metadata,
                )
            )
        if not examples:
            raise ValueError("no scored continuations matched the supervised source")
        if len(examples) != len(self.candidates):
            raise ValueError("one or more scored continuations did not match the supervised source")
        descriptor = self.descriptor
        return PreferenceDataset(
            descriptor.id,
            descriptor.revision,
            tuple(examples),
            metadata=descriptor.metadata,
            schema_version=descriptor.schema_version,
        )


def _split_supervised_target(example: SupervisedExample) -> tuple[tuple[MessageRecord, ...], tuple[MessageRecord, ...]]:
    first = example.trainable_message_indices[0]
    expected = tuple(range(first, len(example.messages)))
    if example.trainable_message_indices != expected or first == 0:
        raise ValueError(
            f"supervised example {example.id!r} must expose one contiguous trainable suffix for preference pairing"
        )
    return example.messages[:first], example.messages[first:]


__all__ = ["PreferencePairSource", "ScoredContinuation"]
