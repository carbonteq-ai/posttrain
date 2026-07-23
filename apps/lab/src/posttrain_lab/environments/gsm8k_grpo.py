"""GSM8K qualification policy: optional reward shaping (not a parallel bridge stack).

Standard GRPO/distill jobs use ``create_verifiers_training_bridge`` from
``posttrain.train.integrations``. Lab may attach ``add_gsm8k_shaping`` as an
enricher only in scenario-specific code; the standard path does not.
"""

from __future__ import annotations

import re
from typing import Any

VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"
_FINAL_ANSWER = re.compile(r"(?m)^####\s*[+-]?(?:\d[\d,]*)(?:\.\d+)?\s*$")
_SHAPING_WEIGHT = 0.1


def add_gsm8k_shaping(trace: Any) -> None:
    """Record the lab-only final-answer conciseness reward component."""

    trace.record_reward(
        "final_answer_conciseness",
        final_answer_conciseness(trace.last_reply, trace.num_output_tokens),
        weight=_SHAPING_WEIGHT,
    )


def final_answer_conciseness(completion: str, completion_tokens: int) -> float:
    if completion_tokens < 1 or _FINAL_ANSWER.search(completion) is None:
        return 0.0
    return 1.0 / (1.0 + completion_tokens / 256.0)


# Compatibility for existing tests that imported the private name.
_final_answer_conciseness = final_answer_conciseness
_add_gsm8k_shaping = add_gsm8k_shaping


__all__ = [
    "VERIFIERS_REVISION",
    "add_gsm8k_shaping",
    "final_answer_conciseness",
]
