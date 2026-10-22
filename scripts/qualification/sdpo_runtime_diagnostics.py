"""Bounded worker diagnostics; no policy, sampling or numerical changes."""


def trace_interval(environment):
    value = int(environment.get("POSTTRAIN_QUALIFICATION_TRACE_SECONDS", "45"))
    if not 0 <= value <= 3600:
        raise ValueError("invalid diagnostic traceback interval")
    return value


def engine_status(rollout):
    try:
        manager = rollout.engine.engine_core.resources.engine_manager
        return {"failed_proc_name": manager.failed_proc_name,
            "processes": [{"name": p.name, "pid": p.pid, "exitcode": p.exitcode}
                          for p in manager.processes]}
    except (AttributeError, ValueError) as error:
        return {"inspection_error": type(error).__name__}
