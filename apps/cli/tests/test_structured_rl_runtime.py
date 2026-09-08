import pytest
from posttrain_cli.execution_planning import _runtime_profile_for_job_kind


@pytest.mark.parametrize("kind", ["train.gdpo", "train.capo"])
def test_structured_reward_jobs_select_online_runtime(kind):
    assert _runtime_profile_for_job_kind(kind) == "framework/online-rl@1"
