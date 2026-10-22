import pytest
from sdpo_runtime_diagnostics import trace_interval


def test_periodic_dumps_can_be_disabled_without_disabling_runtime_gates():
    assert trace_interval({}) == 45
    assert trace_interval({"POSTTRAIN_QUALIFICATION_TRACE_SECONDS": "0"}) == 0
    with pytest.raises(ValueError):
        trace_interval({"POSTTRAIN_QUALIFICATION_TRACE_SECONDS": "-1"})
