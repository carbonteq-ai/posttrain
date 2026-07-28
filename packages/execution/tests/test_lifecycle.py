from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest
from posttrain.execution import (
    ExecutionHandle,
    ExecutionProvider,
    ExecutionRecord,
    ExecutionState,
    wait_for_terminal,
)


class _Provider:
    def __init__(self, states: list[ExecutionState]) -> None:
        self.handle = ExecutionHandle("fake", "job-1", "key-1")
        self.records = [
            ExecutionRecord(
                self.handle,
                state,
                1,
                "targets/gpu",
                datetime.now(UTC),
                state,
            )
            for state in states
        ]
        self.cancelled = False

    def status(self, handle: ExecutionHandle) -> ExecutionRecord:
        assert handle == self.handle
        if len(self.records) == 1:
            return self.records[0]
        return self.records.pop(0)

    def cancel(self, handle: ExecutionHandle) -> None:
        assert handle == self.handle
        self.cancelled = True


def test_wait_reports_only_transitions() -> None:
    provider = _Provider(["queued", "queued", "running", "succeeded"])
    transitions: list[str] = []

    terminal = wait_for_terminal(
        cast(ExecutionProvider, provider),
        provider.handle,
        timeout_seconds=10,
        poll_interval_seconds=0,
        on_transition=lambda record: transitions.append(record.state),
    )

    assert terminal.state == "succeeded"
    assert transitions == ["queued", "running", "succeeded"]
    assert not provider.cancelled


def test_wait_cancels_at_deadline() -> None:
    provider = _Provider(["running"])
    current = 0.0

    def monotonic() -> float:
        return current

    def sleep(seconds: float) -> None:
        nonlocal current
        current += seconds

    with pytest.raises(TimeoutError, match="job-1.*running"):
        wait_for_terminal(
            cast(ExecutionProvider, provider),
            provider.handle,
            timeout_seconds=3,
            poll_interval_seconds=2,
            _monotonic=monotonic,
            _sleep=sleep,
        )

    assert provider.cancelled


def test_wait_can_leave_execution_running_at_deadline() -> None:
    provider = _Provider(["running"])
    current = 0.0

    def monotonic() -> float:
        return current

    def sleep(seconds: float) -> None:
        nonlocal current
        current += seconds

    with pytest.raises(TimeoutError, match="job-1.*running"):
        wait_for_terminal(
            cast(ExecutionProvider, provider),
            provider.handle,
            timeout_seconds=3,
            poll_interval_seconds=2,
            cancel_on_timeout=False,
            _monotonic=monotonic,
            _sleep=sleep,
        )

    assert not provider.cancelled


def test_wait_reports_native_state_or_message_changes() -> None:
    provider = _Provider(["running", "running", "succeeded"])
    provider.records[1] = replace(provider.records[1], native_state="pulling")
    observed: list[tuple[str, str]] = []

    wait_for_terminal(
        cast(ExecutionProvider, provider),
        provider.handle,
        timeout_seconds=10,
        poll_interval_seconds=0,
        on_transition=lambda record: observed.append((record.state, record.native_state)),
    )

    assert observed == [
        ("running", "running"),
        ("running", "pulling"),
        ("succeeded", "succeeded"),
    ]
