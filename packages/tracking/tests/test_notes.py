from __future__ import annotations

import asyncio

from posttrain.tracking import InMemoryRunNoteStore
from posttrain.tracking.notes import verify_note_store


def test_in_memory_store_keeps_the_contract() -> None:
    asyncio.run(verify_note_store(InMemoryRunNoteStore(), "run-a", "run-b"))
