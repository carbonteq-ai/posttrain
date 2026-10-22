from analyze_sdpo_guidance_five import absolute_uncertainty


def test_incomplete_task_sensitivity_is_never_labelled_full_population_uncertainty():
    summary = {"metrics": {"success": {"complete_task_means": {"a": 1.}}}}
    value = absolute_uncertainty(summary, ["a", "b"], 5)["success"]
    assert value["complete_tasks"] == 1 and value["expected_tasks"] == 2
    assert value["full_population_absolute_ci95"] is None
    assert value["complete_task_absolute_sensitivity_ci95"] == [1., 1.]


def test_complete_absolute_interval_is_distinct_from_a_gain_interval():
    summary = {"metrics": {"success": {"complete_task_means": {"a": 0., "b": 1.}}}}
    value = absolute_uncertainty(summary, ["a", "b"], 5)["success"]
    assert value["full_population_absolute_ci95"] == [0., 1.]
    assert "paired" not in value
