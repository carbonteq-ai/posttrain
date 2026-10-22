import pytest
from audit_sdpo_teacher_directions import audit, regions


def test_original_marker_tokens_are_excluded_from_both_evidence_regions():
    result = regions([1, 2, 8, 9, 3, 4], [8, 9], False)
    assert list(result["reasoning"]) == [0, 1]
    assert list(result["native_tool_action"]) == [4, 5]
    assert "final_answer" in regions([1, 8, 9, 4], [8, 9], True)


def test_missing_thinking_marker_is_rejected_instead_of_misclassifying_all_tokens():
    with pytest.raises(ValueError, match="closing marker"):
        regions([1, 2, 3], [8, 9], False)


def test_actor_rollout_diagnostics_use_only_the_actual_region():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [8, 9]

        def decode(self, ids, **kwargs):
            return "feedback" if ids == [99] else str(ids)

    ids = [1, 8, 9, 4, 5]
    row = {"task": "t", "episode_id": "e", "generation_index": 0, "action_ids": ids,
           "student_prefix_ids": [42], "teacher_prefix_ids": [99], "teacher_feedback": "feedback",
           "terminal_text": False}
    support = {"teacher_top1_ids": ids, "teacher_top1_student_ranks": [1]*5,
               "teacher_mass_on_student_support": [.9]*5,
               "actual_adam_teacher_top1_logprob_deltas": [.2]*5,
               "actor_rollout_logp_differences": [9., 8., 7., 1., -1.],
               "importance_weights": [2., 2., 2., 2., .3]}
    direction = {"task": "t", "generation_index": 0, "teacher_sampled_advantages": [0.]*5,
                 "actual_adam_logprob_deltas": [.1]*5, "teacher_support_diagnostic": support}
    call = {"prefix_ids": [42], "action_ids": ids, "tools": [], "message": {}, "raw": "native"}
    result = audit([row], [direction], [call], Tokenizer())
    native = result["rows"][0]["regions"]["native_tool_action"]
    assert native["actor_rollout_max_abs_logp_difference"] == 1.
    assert native["importance_weight_at_cap2_fraction"] == .5
    support.pop("importance_weights")
    result = audit([row], [direction], [call], Tokenizer())
    assert result["rows"][0]["regions"]["native_tool_action"]["importance_weight_at_cap2_fraction"] is None
