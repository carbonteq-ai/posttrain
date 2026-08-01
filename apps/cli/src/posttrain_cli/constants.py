"""Shared CLI constants."""

from __future__ import annotations

from posttrain.common.selections import SelectionFamily

DISTRIBUTION = "posttrain"

CATALOG_FAMILIES: tuple[SelectionFamily, ...] = (
    "model",
    "dataset",
    "environment",
    "inference",
    "training",
    "quantization",
    "evaluation",
    "remote-evaluation",
    "workload",
    "target",
    "recipe",
)

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
