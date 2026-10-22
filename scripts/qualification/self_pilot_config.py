"""CPU-only admission and sampling helpers for the bounded SELF pilot."""

import hashlib
import json
import random
from collections import defaultdict


def resolve_pilot_tasks(protocol_path, heldout_path):
    protocol = json.loads(protocol_path.read_text())
    cohorts = protocol["cohorts"]
    names = cohorts["pilot"]["task_names"]
    development = set(cohorts["development"]["task_names"])
    eligible = set(cohorts["new_update_eligible"]["task_names"])
    heldout = json.loads(heldout_path.read_text())
    digest = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    if len(names) != len(set(names)) or len(names) != 32 or digest != cohorts["pilot"]["sorted_names_sha256"]:
        raise ValueError("pilot identity/count/digest mismatch")
    if not set(names) <= eligible or set(names) & (development | set(heldout["task_names"])):
        raise ValueError("pilot leaks development or heldout tasks, or exceeds the admitted training pool")
    if protocol["environment_revision"] != heldout["environment_revision"]:
        raise ValueError("pilot environment differs from frozen task selection")
    return names


def sample_transitions(rows, count, seed):
    """Balance trajectory exposure, selecting a uniform random turn per visit."""
    if not rows or count < 1:
        raise ValueError("nonempty transitions and positive exposure count required")
    groups = defaultdict(list)
    for row in rows:
        groups[row["trajectory_id"]].append(row)
    rng, sampled = random.Random(seed), []
    while len(sampled) < count:
        identities = sorted(groups)
        rng.shuffle(identities)
        for identity in identities:
            sampled.append(rng.choice(groups[identity]))
            if len(sampled) == count:
                break
    return sampled


def validate_sequences(rows, limit):
    for row in rows:
        for prefix, target in (("student_prefix_ids", "action_ids"), ("teacher_prefix_ids", "action_ids"),
                               ("feedback_prefix_ids", "feedback_ids")):
            if not row[target] or len(row[prefix]) + len(row[target]) > limit:
                raise ValueError(f"unaltered {prefix}+{target} exceeds qualified sequence limit {limit}")


def admit_sequences(rows, limit):
    """Retain exact sequences or exclude the entire transition; never slice tokens."""
    admitted, excluded = [], []
    for index, row in enumerate(rows):
        try:
            validate_sequences([row], limit)
        except ValueError as error:
            excluded.append({"transition_index": index, "trajectory_id": row["trajectory_id"],
                             "reason": str(error), "token_lengths": {key: len(value) for key, value in row.items()
                                                                       if key.endswith("_ids")}})
        else:
            admitted.append(row)
    return admitted, excluded
