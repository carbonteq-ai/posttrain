import copy

from sdpo_greedy_repeatability import audit


def call():
    return {"complete": True, "temperature": 0., "prefix_ids": [1, 2],
            "requested_completion_budget": 4096, "adapter_sha256": "a",
            "action_ids": [3, 4], "sampled_logprobs": [-.1, -.2]}


def test_identical_native_inputs_compare_tokens_and_report_logprob_drift():
    a = call()
    b = copy.deepcopy(a)
    b["sampled_logprobs"][0] += .01
    result = audit([a, b])
    assert result["qualified"] and result["compared_pairs"] == 1
    assert result["comparisons"][0]["common_prefix_max_logp_delta"] > 0
    b["action_ids"][1] = 9
    assert not audit([a, b])["qualified"]


def test_different_policy_budget_stochastic_and_incomplete_are_not_controls():
    a = call()
    variants = []
    for key, value in [("adapter_sha256", "b"), ("requested_completion_budget", 20),
                       ("temperature", 1.), ("complete", False)]:
        b = copy.deepcopy(a)
        b[key] = value
        variants.append(b)
    result = audit([a, *variants])
    assert not result["qualified"] and result["compared_pairs"] == 0
