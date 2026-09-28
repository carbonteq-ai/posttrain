"""Deferred delivery of a host cancellation signal around atomic state updates.

A worker host (``posttrain-runtime``) turns SIGTERM and SIGINT into
``SystemExit(128 + signal)`` so tracked-run finalization records the run as
cancelled. Python delivers a signal between two bytecodes of the main thread,
so without help the exit could land in the middle of an update that must be
all-or-nothing, such as a training optimizer step followed by its scheduler
step and step counter. A reusable package marks such an update as a
:class:`CriticalSection`; the host routes each signal through
:meth:`HostCancellation.request`, which delivers the exit immediately outside a
section and at the end of the outermost section inside one.

Deferral is bounded. When a section is still open ``max_deferral_seconds``
after the request, a watchdog re-signals the main thread and the exit is
forced where it lands. :attr:`HostCancellation.forced` then records that an
atomic update may have been torn, so no consumer should persist that state.

Only sections on the main thread defer anything, because Python runs signal
handlers only on the main thread. A section opened on another thread is a
no-op. When no host has armed the gate, sections are no-ops as well and no
request is ever recorded.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from types import TracebackType

DEFAULT_MAX_DEFERRAL_SECONDS = 60.0
_WATCHDOG_POLL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class CancellationRequest:
    """The first cancellation signal a host received for this process."""

    signal_number: int
    requested_at: float
    deferred_by: str | None

    @property
    def exit_code(self) -> int:
        return 128 + self.signal_number


class CriticalSection:
    """One atomic update during which host cancellation is deferred.

    Use it as a context manager for a lexical update, or call :meth:`open` and
    :meth:`close` when the update spans framework callbacks. ``close`` delivers a
    deferred cancellation when it closes the outermost section; ``close(deliver=False)``
    releases the section without raising, for an error path that is already
    unwinding with another exception.
    """

    def __init__(self, gate: HostCancellation, name: str) -> None:
        if not name.strip():
            raise ValueError("critical section name cannot be empty")
        self._gate = gate
        self.name = name
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    def open(self) -> None:
        if self._opened:
            return
        self._opened = self._gate._enter(self.name)

    def close(self, *, deliver: bool = True) -> None:
        if not self._opened:
            return
        self._opened = False
        self._gate._exit(self.name, deliver=deliver)

    def __enter__(self) -> CriticalSection:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if exc is not None and self._gate.pending:
            exc.add_note(f"host cancellation deferred by {self.name!r} was superseded by this failure")
        self.close(deliver=exc is None)


class HostCancellation:
    """Process-wide cancellation gate shared by a host and reusable packages."""

    def __init__(self) -> None:
        self._armed = False
        self._max_deferral_seconds = DEFAULT_MAX_DEFERRAL_SECONDS
        self._open: list[str] = []
        self._request: CancellationRequest | None = None
        self._pending = False
        self._forced = False
        self._force_now = False
        self._deadline: float | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_watchdog = threading.Event()

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def requested(self) -> CancellationRequest | None:
        """The recorded cancellation request, delivered or still pending."""

        return self._request

    @property
    def pending(self) -> bool:
        """Whether a request is waiting for the open sections to close."""

        return self._pending

    @property
    def forced(self) -> bool:
        """Whether a request was delivered inside a section after its deferral bound."""

        return self._forced

    @property
    def open_sections(self) -> tuple[str, ...]:
        return tuple(self._open)

    def arm(self, *, max_deferral_seconds: float = DEFAULT_MAX_DEFERRAL_SECONDS) -> None:
        """Start honoring sections; a host calls this when it installs its signal handlers."""

        if max_deferral_seconds <= 0:
            raise ValueError("maximum cancellation deferral must be positive")
        self.disarm()
        self._max_deferral_seconds = max_deferral_seconds
        self._armed = True
        # The watchdog runs for the whole armed scope so a signal handler never
        # has to start a thread or take a lock the interrupted code might hold.
        stop = threading.Event()
        self._stop_watchdog = stop
        watchdog = threading.Thread(
            target=self._watch,
            args=(stop,),
            name="posttrain-cancellation-watchdog",
            daemon=True,
        )
        self._watchdog = watchdog
        watchdog.start()

    def disarm(self) -> None:
        """Forget all state; a host calls this when it restores its previous handlers."""

        self._stop_watchdog.set()
        watchdog = self._watchdog
        self._watchdog = None
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=1.0)
        self._armed = False
        self._open = []
        self._request = None
        self._pending = False
        self._forced = False
        self._force_now = False
        self._deadline = None

    def critical(self, name: str) -> CriticalSection:
        return CriticalSection(self, name)

    def request(self, signal_number: int) -> None:
        """Deliver or defer one host cancellation signal; call only from a signal handler.

        The first request raises ``SystemExit(128 + signal)`` unless a section is
        open. A repeated request is ignored so it cannot interrupt the bounded
        cancellation path, except when the deferral watchdog forces delivery.
        """

        if self._request is not None:
            if self._pending and self._force_now:
                self._pending = False
                self._force_now = False
                self._forced = bool(self._open)
                raise SystemExit(self._request.exit_code)
            return
        deferred_by = self._open[-1] if self._armed and self._open else None
        self._request = CancellationRequest(signal_number, time.monotonic(), deferred_by)
        if deferred_by is None:
            raise SystemExit(128 + signal_number)
        self._deadline = time.monotonic() + self._max_deferral_seconds
        self._pending = True

    def _enter(self, name: str) -> bool:
        if not self._armed or threading.current_thread() is not threading.main_thread():
            return False
        self._open.append(name)
        return True

    def _exit(self, name: str, *, deliver: bool) -> None:
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index] == name:
                del self._open[index]
                break
        if self._open or not self._pending:
            return
        self._pending = False
        self._deadline = None
        request = self._request
        if deliver and request is not None:
            raise SystemExit(request.exit_code)

    def _watch(self, stop: threading.Event) -> None:
        main = threading.main_thread().ident
        while not stop.wait(_WATCHDOG_POLL_SECONDS):
            deadline = self._deadline
            request = self._request
            if main is None or request is None or not self._pending or deadline is None:
                continue
            if time.monotonic() < deadline:
                continue
            self._deadline = None
            self._force_now = True
            # A real signal, unlike ``_thread.interrupt_main``, also interrupts a
            # blocking system call on the main thread.
            signal.pthread_kill(main, request.signal_number)


_HOST_CANCELLATION = HostCancellation()


def host_cancellation() -> HostCancellation:
    """Return the process-wide gate; a single process has one set of signal handlers."""

    return _HOST_CANCELLATION


__all__ = [
    "DEFAULT_MAX_DEFERRAL_SECONDS",
    "CancellationRequest",
    "CriticalSection",
    "HostCancellation",
    "host_cancellation",
]
