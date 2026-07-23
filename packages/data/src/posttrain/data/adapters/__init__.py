"""Framework boundary adapters for canonical post-training data."""

from .huggingface import (
    PreferenceFormat,
    SFTFormat,
    preferences_from_huggingface,
    supervised_from_huggingface,
    to_huggingface_preference_rows,
    to_huggingface_sft_rows,
)
from .nemo import preferences_from_nemo, supervised_from_nemo, to_nemo_preference_rows, to_nemo_sft_rows
from .verifiers import TraceSelection, supervised_from_verifiers, supervised_from_verifiers_jsonl

__all__ = [
    "PreferenceFormat",
    "SFTFormat",
    "TraceSelection",
    "preferences_from_huggingface",
    "preferences_from_nemo",
    "supervised_from_huggingface",
    "supervised_from_nemo",
    "supervised_from_verifiers",
    "supervised_from_verifiers_jsonl",
    "to_huggingface_preference_rows",
    "to_huggingface_sft_rows",
    "to_nemo_preference_rows",
    "to_nemo_sft_rows",
]
