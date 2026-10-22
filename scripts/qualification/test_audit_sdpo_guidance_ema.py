import pytest
import torch
from audit_sdpo_guidance_ema import compare


def test_small_b_updates_are_verified_instead_of_hidden_by_absolute_tolerance():
    key = "projection.lora_B.weight"
    initial = {key: torch.zeros(3)}
    students = [{key: torch.full((3,), value)} for value in (1e-6, 2e-6)]
    expected = initial[key].lerp(students[0][key], .01).lerp(students[1][key], .01)
    assert compare(initial, students, {key: expected}, .01)["qualified"]
    with pytest.raises(ValueError, match="EMA recurrence"):
        compare(initial, students, initial, .01)


def test_student_only_teacher_and_missing_export_cannot_pass_as_ema():
    key = "projection.lora_B.weight"
    initial = {key: torch.zeros(3)}
    students = [{key: torch.ones(3)}]
    with pytest.raises(ValueError, match="EMA recurrence"):
        compare(initial, students, students[-1], .01)
    with pytest.raises(ValueError, match="keys"):
        compare(initial, [{}], initial, .01)


def test_nonfinite_export_is_rejected_before_comparison():
    key = "projection.lora_B.weight"
    with pytest.raises(ValueError, match="finite"):
        compare({key: torch.zeros(2)}, [{key: torch.tensor([float("nan"), 0.])}],
                {key: torch.zeros(2)}, .01)
    with pytest.raises(ValueError, match="initial.*finite"):
        compare({key: torch.tensor([float("nan"), 0.])}, [{key: torch.zeros(2)}],
                {key: torch.zeros(2)}, .01)
