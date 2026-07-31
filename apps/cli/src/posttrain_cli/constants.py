"""Shared CLI constants."""

from __future__ import annotations

DISTRIBUTION = "posttrain"

DATASET_FORMAT_CHOICES = (
    "auto",
    "messages",
    "prompt-completion",
    "alpaca",
    "sharegpt",
    "trl",
    "tulu",
    "nemo-ranked",
)

NEMO_FORMAT_CHOICES = ("auto", "messages", "nemo-ranked")

DATASET_KIND_CHOICES = ("supervised", "preference")

TEMPLATE_CHOICES = ("sft", "grpo")

RUN_MODE_CHOICES = ("auto", "job", "generic")
