"""Pure metadata for the representative serving prompt population."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeneralServingSource:
    """Pinned source metadata retained in the corpus definition."""

    id: str
    revision: str
    split: str
    license_id: str
    config: str | None = None


@dataclass(frozen=True, slots=True)
class GeneralServingCorpusDefinition:
    """Reproducibility contract for one packaged serving corpus revision.

    ``builder`` is an inert import reference.  Consumers may inspect this
    value while planning a workload without importing the module that performs
    network I/O and record transformation.
    """

    id: str
    revision: str
    builder: str
    expected_content_sha256: str
    record_count: int
    category_counts: tuple[tuple[str, int], ...]
    selection_algorithm: str
    sources: tuple[GeneralServingSource, ...]


GENERAL_SERVING_V1 = GeneralServingCorpusDefinition(
    id="general-serving-v1",
    revision="1",
    builder="posttrain.serve.benchmarks.general_serving.build:build",
    expected_content_sha256="9a9467fd8a5e744968d09a4d8fd6f4d92a089c50a84e1e6e7e5c5520a9f4e50e",
    record_count=128,
    category_counts=(
        ("chat", 8),
        ("code", 32),
        ("extraction", 8),
        ("reasoning", 64),
        ("structured-output", 8),
        ("tool-use", 8),
    ),
    selection_algorithm="lowest-sha256-over-source-revision-key-and-normalized-prompt-v1",
    sources=(
        GeneralServingSource(
            id="openai/gsm8k",
            revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
            config="main",
            split="train",
            license_id="MIT",
        ),
        GeneralServingSource(
            id="openai/openai_humaneval",
            revision="e9b53e1677523f1e61e4d0960fd7502694a24bd4",
            split="test",
            license_id="MIT",
        ),
        GeneralServingSource(
            id="carbonteq-ai/posttrain",
            revision="1",
            split="first-party",
            license_id="Apache-2.0",
        ),
    ),
)


__all__ = ["GENERAL_SERVING_V1", "GeneralServingCorpusDefinition", "GeneralServingSource"]
