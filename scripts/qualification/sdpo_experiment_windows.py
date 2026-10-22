"""Explicit renewed-window authorization; retries never renew their own clock."""

AUTHORIZATION = "user-fresh-plain-four-windows-20261019"


def validate_window(config, previous, now):
    deadline = config["experiment_deadline_unix"]
    authorization = config.get("budget_authorization")
    if config.get("speculation") == "none" and authorization != AUTHORIZATION:
        raise ValueError("plain-decoding window requires the recorded user authorization")
    if authorization == AUTHORIZATION:
        if config["number"] not in (2, 3, 4, 5) or config.get("speculation") != "none":
            raise ValueError("renewal only authorizes remaining plain-decoding experiments")
        started = config["budget_window_started_unix"]
        if deadline != started + 7140 or started > now or not now < deadline:
            raise ValueError("invalid or exhausted renewed window")
        matching = [a for a in previous if a.get("budget_authorization") == authorization]
        if matching and any(a["experiment_deadline_unix"] != deadline for a in matching):
            raise ValueError("retry cannot renew an authorized window")
    elif previous and deadline != previous[0]["experiment_deadline_unix"]:
        raise ValueError("deadline changed without explicit authorization")
    return deadline
