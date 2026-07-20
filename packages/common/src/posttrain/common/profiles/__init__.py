"""Reusable pinned foundation profiles."""

from .lfm25 import LFM_25_12B_THINKING
from .qwen35 import QWEN_35_2B

FOUNDATION_PROFILES = {
    QWEN_35_2B.id: QWEN_35_2B,
    LFM_25_12B_THINKING.id: LFM_25_12B_THINKING,
}

__all__ = ["FOUNDATION_PROFILES", "LFM_25_12B_THINKING", "QWEN_35_2B"]
