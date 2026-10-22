"""Twenty-task coverage with fair additional twelve-action rotation."""

from sdpo_agent_config import balanced_rows
from sdpo_long_campaign_config import distinct_campaign_batch


def batch(rows, tasks, step, mode="first-error-frontier"):
    if len(tasks) != len(set(tasks)) or len(tasks) != 20 or not 1 <= step <= 10:
        raise ValueError("frozen20task/tenupdate sampling required")
    # Twelve extra slots per update: stride12 covers each task3times in5steps.
    start = (step - 1) * 12 % 20
    order = tasks[start:] + tasks[:start]
    selected = balanced_rows(rows, 32, offset=step - 1, tasks=order, mode=mode)
    distinct_campaign_batch(selected, 32)
    return selected
