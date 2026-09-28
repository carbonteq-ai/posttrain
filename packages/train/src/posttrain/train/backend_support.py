"""Online-RL settings a training backend does not implement.

A setting that a backend never reads must be rejected, not silently ignored:
the run would otherwise train with different behaviour than its settings and
recorded configuration say.
"""

from __future__ import annotations

from dataclasses import fields

from .profiles import GRPOSettings

# Top-level GRPOSettings fields the veRL GRPO launch plan does not pass to veRL.
# Only their defaults are accepted, so a selection cannot claim behaviour veRL
# never applies.
_VERL_GRPO_FIXED_DEFAULTS: tuple[str, ...] = (
    "advantage_scaling",
    "importance_sampling_mode",
    "importance_sampling_clip_min",
    "importance_sampling_clip_max",
    "max_admission_attempts",
)


def verl_grpo_settings_problem(settings: GRPOSettings) -> str | None:
    """Explain the first GRPO setting the veRL backend would silently ignore, or None."""

    if settings.algorithm == "olmo3":
        return "the OLMo 3 GRPO recipe is currently supported by the TRL backend only"
    if settings.truncation_penalty is not None:
        return "GRPO truncation_penalty is currently supported by the TRL backend only"
    if settings.adaptive_curriculum is not None:
        return "adaptive_curriculum is currently supported by the TRL backend only"
    if settings.active_sampling is not None:
        return "GRPO active_sampling is currently supported by the TRL backend only"
    defaults = {item.name: item.default for item in fields(GRPOSettings) if item.name in _VERL_GRPO_FIXED_DEFAULTS}
    changed = [name for name in _VERL_GRPO_FIXED_DEFAULTS if getattr(settings, name) != defaults[name]]
    if changed:
        selected = ", ".join(f"{name}={getattr(settings, name)!r}" for name in changed)
        return (
            f"GRPO {selected} is currently supported by the TRL backend only; the veRL backend does not "
            "receive this setting, so keep the default " + ", ".join(f"{name}={defaults[name]!r}" for name in changed)
        )
    return None


__all__ = ["verl_grpo_settings_problem"]
