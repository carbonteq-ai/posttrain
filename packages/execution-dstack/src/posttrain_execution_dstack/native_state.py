"""Dependency-free classification of dstack's native assignment evidence."""

from __future__ import annotations

from typing import Any, Literal

type AssignmentState = Literal["assigned", "never-assigned", "ambiguous"]


def assignment_state(run: Any) -> AssignmentState:
    """Classify whether a native dstack run ever acquired an instance.

    dstack retains every job submission in the terminal run model. Provisioning
    data contains the concrete instance identity and runtime data is populated
    only after admission. Absence is conclusive only when every native job has
    at least one retained submission and all retained submissions lack both
    fields.
    """

    jobs = getattr(run, "jobs", None)
    if not isinstance(jobs, list) or not jobs:
        return "ambiguous"

    for job in jobs:
        submissions = getattr(job, "job_submissions", None)
        if not isinstance(submissions, list) or not submissions:
            return "ambiguous"
        if getattr(job, "job_connection_info", None) is not None:
            return "assigned"
        for submission in submissions:
            if (
                getattr(submission, "job_provisioning_data", None) is not None
                or getattr(submission, "job_runtime_data", None) is not None
            ):
                return "assigned"

    return "never-assigned"


__all__ = ["AssignmentState", "assignment_state"]
