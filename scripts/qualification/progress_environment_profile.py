"""Explicit local environment selection for fresh ten-task controls."""

FROZEN = "frozen-a0c"
REPAIRED = "local-clock-strict-v1"
ACCESS = "local-clock-strict-access-v2"
DISCOVERY = "local-clock-strict-access-v3"


def task_options(profile=FROZEN):
    if profile == FROZEN:
        return {}
    if profile in (REPAIRED, ACCESS, DISCOVERY):
        return {"world_time_context_if_declared": True, "strict_tool_arguments": True}
    raise ValueError(f"unknown environment profile: {profile}")


def taskset_options(profile, names):
    task_options(profile)
    # Native taskset rejects unused additions. Scope this project selection to
    # the actual allocated population, not every subsequent training batch.
    additions = {"operations.calendly_equipment_inspection": ("calendly_list_event_types",)}
    if profile == DISCOVERY:
        additions["support.hiver_escalation_patterns"] = ("jira_project",)
    return {"additional_tools_by_task": {n: additions[n] for n in names if n in additions}} if profile in (ACCESS, DISCOVERY) else {}


def is_repaired(profile):
    task_options(profile)
    return profile != FROZEN


def validate_reference(profile, reference):
    task_options(profile)
    if is_repaired(profile) and reference is not None:
        raise ValueError("repaired environment requires fresh controls, not historical references")


def verified_rejection(name, arguments, declared, observed):
    """Match only the exact tested middleware error, never arbitrary tool failures."""
    extra = set(arguments) - set(declared)
    if not extra:
        return False
    expected = (f"Unsupported arguments for {name}: {', '.join(sorted(extra))}. "
                "Use the declared tool parameter names. No action was executed.")
    if isinstance(observed, dict) and observed.get("isError") is True:
        content = observed.get("content", [])
        observed = content[0].get("text") if len(content) == 1 else None
    return observed == expected


def validate_native_source(current, previous, receipt):
    profile = current.get("environment_profile")
    if (profile not in (REPAIRED, ACCESS, DISCOVERY)
            or previous.get("environment_profile") != profile
            or receipt.get("environment_profile") != profile
            or receipt.get("runtime_completed") is not True
            or any(not current.get(k) or current[k] != previous.get(k)
                   for k in ("environment_base_revision", "environment_patch_sha256"))):
        raise ValueError("native reference is not a completed identical repaired-environment control")
