"""Project-owned priorities for the matched episode-judge qualification runs.

Weights apply to normalized component advantages, not raw judge scores.
They express relative priorities, not guaranteed shares of optimizer influence.
"""

EPISODE_COMPONENT_WEIGHTS = {
    "partial_credit": 0.55,
    "problem_understanding_planning": 0.05,
    "logical_correctness": 0.05,
    "verification_self_correction": 0.03,
    "progress_efficiency": 0.07,
    "action_quality": 0.15,
    "answer_quality": 0.10,
}


def episode_component_weights(rubric_names: tuple[str, ...]) -> tuple[float, ...]:
    """Align priorities by name; reject silently added or removed dimensions."""
    names = ("partial_credit", *rubric_names)
    if len(names) != len(set(names)) or set(names) != set(EPISODE_COMPONENT_WEIGHTS):
        raise ValueError("episode rubric names differ from the selected reward priority profile")
    return tuple(EPISODE_COMPONENT_WEIGHTS[name] for name in names)
