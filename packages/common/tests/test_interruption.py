from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

import pytest
from posttrain.common import HostCancellation


@contextmanager
def _armed(gate: HostCancellation, *, max_deferral_seconds: float = 60.0) -> Iterator[HostCancellation]:
    """Install the same handler shape as posttrain-runtime for SIGTERM."""

    previous = signal.getsignal(signal.SIGTERM)

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        gate.request(signum)

    gate.arm(max_deferral_seconds=max_deferral_seconds)
    signal.signal(signal.SIGTERM, handler)
    try:
        yield gate
    finally:
        signal.signal(signal.SIGTERM, previous)
        gate.disarm()


def test_signal_outside_a_critical_section_exits_immediately() -> None:
    gate = HostCancellation()
    with _armed(gate), pytest.raises(SystemExit) as captured:
        signal.raise_signal(signal.SIGTERM)
    assert captured.value.code == 128 + signal.SIGTERM


def test_signal_inside_a_critical_section_is_delivered_when_it_closes() -> None:
    gate = HostCancellation()
    completed: list[str] = []
    with _armed(gate):
        with pytest.raises(SystemExit) as captured:
            with gate.critical("optimizer_update"):
                signal.raise_signal(signal.SIGTERM)
                # The update keeps running after the signal arrived.
                completed.append("optimizer.step")
                completed.append("scheduler.step")
            completed.append("after the section")
        request = gate.requested
        assert request is not None
        assert request.deferred_by == "optimizer_update"
        assert not gate.forced
    assert captured.value.code == 128 + signal.SIGTERM
    assert completed == ["optimizer.step", "scheduler.step"]


def test_nested_sections_defer_until_the_outermost_closes() -> None:
    gate = HostCancellation()
    order: list[str] = []
    with _armed(gate), pytest.raises(SystemExit):
        outer = gate.critical("update")
        outer.open()
        with gate.critical("save"):
            signal.raise_signal(signal.SIGTERM)
            order.append("save finished")
        order.append("inner closed without exit")
        outer.close()
        order.append("unreachable")
    assert order == ["save finished", "inner closed without exit"]


def test_a_repeated_signal_is_ignored_while_the_exit_unwinds() -> None:
    gate = HostCancellation()
    with _armed(gate):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)
        # The finalizer path keeps running when the provider repeats the signal.
        signal.raise_signal(signal.SIGTERM)
        assert gate.requested is not None


def test_closing_without_delivery_supersedes_a_pending_request() -> None:
    gate = HostCancellation()
    with _armed(gate):
        section = gate.critical("optimizer_update")
        section.open()
        signal.raise_signal(signal.SIGTERM)
        assert gate.pending
        section.close(deliver=False)
        assert not gate.pending
        assert gate.requested is not None
        # Another section later does not replay the superseded request.
        with gate.critical("save"):
            pass


def test_failure_inside_a_section_keeps_its_own_error() -> None:
    gate = HostCancellation()
    with _armed(gate), pytest.raises(RuntimeError, match="optimizer failed") as captured:
        with gate.critical("optimizer_update"):
            signal.raise_signal(signal.SIGTERM)
            raise RuntimeError("optimizer failed")
    assert "superseded" in captured.value.__notes__[0]


def test_deferral_is_bounded_and_marks_the_update_as_torn() -> None:
    gate = HostCancellation()
    started = time.monotonic()
    with _armed(gate, max_deferral_seconds=0.2):
        with pytest.raises(SystemExit) as captured:
            with gate.critical("stuck_weight_sync"):
                signal.raise_signal(signal.SIGTERM)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    time.sleep(0.01)
        assert gate.forced
    assert captured.value.code == 128 + signal.SIGTERM
    assert time.monotonic() - started < 5.0


def test_sections_are_inert_when_no_host_armed_the_gate() -> None:
    gate = HostCancellation()
    with gate.critical("optimizer_update") as section:
        assert not section.is_open
    assert gate.requested is None
    assert gate.open_sections == ()


def test_sections_on_worker_threads_do_not_defer() -> None:
    gate = HostCancellation()
    opened: list[bool] = []

    def worker() -> None:
        with gate.critical("background") as section:
            opened.append(section.is_open)

    with _armed(gate):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert opened == [False]


def test_disarm_forgets_the_request() -> None:
    gate = HostCancellation()
    with _armed(gate), pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGTERM)
    assert gate.requested is None
    assert not gate.armed
