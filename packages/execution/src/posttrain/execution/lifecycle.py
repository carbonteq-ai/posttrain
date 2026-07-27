"""Provider-neutral waiting and terminal-state reconciliation."""

from __future__ import annotations

import time
from collections.abc import Callable

from .contracts import ExecutionHandle, ExecutionProvider, ExecutionRecord
from .receipts import ExecutionJournal

_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "lost"})


def wait_for_terminal(
    provider: ExecutionProvider,
    handle: ExecutionHandle,
    *,
    timeout_seconds: float,
    journal: ExecutionJournal | None = None,
    poll_interval_seconds: float = 5.0,
    cancel_on_timeout: bool = True,
    on_transition: Callable[[ExecutionRecord], None] | None = None,
    _monotonic: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], None] = time.sleep,
) -> ExecutionRecord:
    """Poll one detached execution, journal observations, and cancel on timeout.

    ``on_transition`` receives only meaningful state/native-state/message
    changes, so CLIs can remain observable without printing every poll.
    """

    if timeout_seconds <= 0:
        raise ValueError("execution wait timeout must be positive")
    if poll_interval_seconds < 0:
        raise ValueError("execution poll interval cannot be negative")

    deadline = _monotonic() + timeout_seconds
    previous: tuple[str, str, str | None] | None = None
    while True:
        record = provider.status(handle)
        if journal is not None:
            journal.append(record)
        transition = (record.state, record.native_state, record.message)
        if transition != previous:
            previous = transition
            if on_transition is not None:
                on_transition(record)
        if record.state in _TERMINAL_STATES:
            return record

        remaining = deadline - _monotonic()
        if remaining <= 0:
            if cancel_on_timeout:
                provider.cancel(handle)
            raise TimeoutError(
                f"execution {handle.provider_id} exceeded {timeout_seconds:g}s "
                f"while {record.state}"
            )
        _sleep(min(poll_interval_seconds, remaining))
