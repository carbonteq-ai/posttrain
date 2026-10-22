from types import SimpleNamespace
from unittest.mock import patch

from sdpo_mtp_metrics import NAMES, delta, snapshot


class Registry:
    def __init__(self, values):
        self.values = values

    def collect(self):
        yield SimpleNamespace(samples=[SimpleNamespace(name=NAMES[k], value=v)
                                       for k, v in self.values.items()])


def test_actual_counters_are_differenced_not_read_as_whole_run_acceptance():
    a = snapshot(Registry(dict(drafts=10, draft_tokens=20, accepted_tokens=10)))
    b = snapshot(Registry(dict(drafts=20, draft_tokens=40, accepted_tokens=25)))
    r = delta(a, b)
    assert r["available"] and r["draft_token_acceptance"] == .75
    assert r["counters"]["drafts"] == 10


def test_missing_zero_and_reset_counters_stay_distinct():
    missing = snapshot(Registry({}))
    zero = snapshot(Registry(dict(drafts=0, draft_tokens=0, accepted_tokens=0)))
    previous = snapshot(Registry(dict(drafts=2, draft_tokens=4, accepted_tokens=2)))
    assert not delta(missing, zero)["available"]
    assert delta(zero, zero)["available"] and delta(zero, zero)["draft_token_acceptance"] is None
    assert not delta(previous, zero)["available"]


def test_optional_metrics_dependency_failure_is_retained_not_a_training_failure():
    with patch("builtins.__import__", side_effect=ImportError("optional metrics unavailable")):
        result = snapshot()
    assert not result["available"] and result["error_type"] == "ImportError"
