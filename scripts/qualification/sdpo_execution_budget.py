"""One non-resettable host deadline for experiment10 fitting and reserve phases."""

import datetime
import json
import math
import time
from pathlib import Path


def boot_identity():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def boot_seconds():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def start_shared_budget(path, run_name, preflight_seconds, *, now=None, boot_id=None):
    if not 0 <= preflight_seconds < 7200:
        raise ValueError("invalid preflight reservation")
    now = boot_seconds() if now is None else now
    if not math.isfinite(now) or now < 0:
        raise ValueError("invalid execution clock")
    budget = {"experiment_slot": 10, "total_wall_cap_seconds": 7200,
              "maximum_combined_optimizer_updates": 20, "fitting_run_name": run_name,
              "preflight_reservation_seconds": preflight_seconds,
              "clock": "CLOCK_BOOTTIME", "boot_id": boot_identity() if boot_id is None else boot_id,
              "created_boot_seconds": now, "deadline_boot_seconds": now + 7200 - preflight_seconds,
              "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat()}
    # Exclusive creation prevents a second launcher from resetting the budget.
    with path.open("x") as stream:
        json.dump(budget, stream, indent=2)
        stream.write("\n")
    return budget


def remaining_runtime(budget, phase_cap, *, now=None, boot_id=None):
    now = boot_seconds() if now is None else now
    boot_id = boot_identity() if boot_id is None else boot_id
    created, deadline = budget["created_boot_seconds"], budget["deadline_boot_seconds"]
    preflight = budget["preflight_reservation_seconds"]
    if (budget["experiment_slot"] != 10 or budget["total_wall_cap_seconds"] != 7200
            or budget["maximum_combined_optimizer_updates"] != 20 or budget["clock"] != "CLOCK_BOOTTIME"
            or budget["boot_id"] != boot_id or not all(math.isfinite(x) for x in (created, deadline, now))
            or not 0 <= preflight < 7200 or abs(deadline - created - 7200 + preflight) > 1e-6
            or now < created or phase_cap <= 0):
        raise ValueError("shared execution budget identity/clock mismatch")
    seconds = min(int(phase_cap), math.floor(deadline - now - 30))
    if seconds < 1:
        raise TimeoutError("experiment10 combined deadline exhausted")
    return seconds


def reserve_profile(selection):
    profile = {k: selection[k] for k in ("context_limit", "reply_limit", "thinking")}
    if ((profile["context_limit"], profile["reply_limit"]) not in ((16384, 2048), (32768, 4096))
            or type(profile["thinking"]) is not bool
            or (profile["thinking"] and selection["tool_parser_mode"] != "schema-bound-v1")):
        raise ValueError("unqualified frozen reserve mode/profile")
    return profile


def install_deadline_timer(budget):
    """Kernel timer rechecks the absolute deadline after container startup latency."""
    import signal

    seconds = remaining_runtime(budget, 7100)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return seconds
