import pytest
from sdpo_experiment_windows import AUTHORIZATION, validate_window


def config():
    return dict(number=2, speculation="none", budget_authorization=AUTHORIZATION,
                budget_window_started_unix=1000, experiment_deadline_unix=8140)


def test_explicit_new_window_preserves_old_attempt_deadline():
    old = [dict(experiment_deadline_unix=500)]
    assert validate_window(config(), old, 1100) == 8140
    assert old[0]["experiment_deadline_unix"] == 500


def test_retry_must_keep_renewed_deadline():
    prior = [dict(budget_authorization=AUTHORIZATION, experiment_deadline_unix=8140)]
    c = config()
    assert validate_window(c, prior, 1200) == 8140
    c.update(budget_window_started_unix=1200, experiment_deadline_unix=8340)
    with pytest.raises(ValueError, match="retry cannot renew"):
        validate_window(c, prior, 1200)


def test_plain_new_arm_cannot_omit_the_authorized_window_identity():
    c = config()
    c.pop("budget_authorization")
    with pytest.raises(ValueError, match="recorded user authorization"):
        validate_window(c, [], 1100)


@pytest.mark.parametrize("changes", [dict(number=1), dict(speculation="mtp"),
    dict(experiment_deadline_unix=9000), dict(budget_window_started_unix=1200)])
def test_no_unapproved_scope_or_extended_duration(changes):
    c = config()
    c.update(changes)
    with pytest.raises(ValueError):
        validate_window(c, [], 1100)
