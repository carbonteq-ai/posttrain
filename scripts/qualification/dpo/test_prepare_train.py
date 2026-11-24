"""prepare_train.py: steps are set in the main settings block only (the memory-proof block keeps its own)."""
from pathlib import Path

import pytest

from prepare_train import CATALOG, set_steps

TEXT = (Path(__file__).resolve().parents[3] / "apps/lab" / CATALOG).read_text()


def block(text, name):
    start = text.index(f"  {name}:\n")
    return text[start:text.find("\n\n", start)]


def test_set_steps_changes_only_the_main_settings():
    new = set_steps(TEXT, 63, 21)
    assert "max_steps: 63" in block(new, "qwen3.5-4b/agent-dpo-p9-e3")
    assert "checkpoint_steps: 21" in block(new, "qwen3.5-4b/agent-dpo-p9-e3")
    assert block(new, "qwen3.5-4b/agent-dpo-p9-proof") == block(TEXT, "qwen3.5-4b/agent-dpo-p9-proof")
    assert len(new.splitlines()) == len(TEXT.splitlines())


def test_set_steps_refuses_a_catalog_without_the_block():
    with pytest.raises(ValueError):
        set_steps(TEXT.replace("  qwen3.5-4b/agent-dpo-p9-e3:\n", "  other:\n"), 63, 21)
