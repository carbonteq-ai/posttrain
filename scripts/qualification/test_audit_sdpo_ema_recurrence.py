"""A nonzero but stale or hard-copied teacher must fail recurrence validation."""

import json

import pytest
import torch
from audit_sdpo_ema_recurrence import audit
from safetensors.torch import save_file


def test_actual_recurrence_rejects_hard_copy_and_stale_teacher(tmp_path):
    (tmp_path / 'inputs.json').write_text(json.dumps({'teacher_rate': .01}))
    (tmp_path / 'updates.json').write_text(json.dumps([{'step': 1}, {'step': 2}]))
    initial = {'layer.lora_A.weight': torch.tensor([.5]), 'layer.lora_B.weight': torch.tensor([0.])}
    (tmp_path / 'initial-student').mkdir()
    save_file(initial, tmp_path / 'initial-student/adapter_model.safetensors')
    previous = initial
    for step in (1, 2):
        student = {k: v + .2 * step for k, v in initial.items()}
        teacher = {k: v.lerp(student[k], .01) for k, v in previous.items()}
        folder = tmp_path / f'checkpoint-{step}'
        (folder / 'sdpo_teacher').mkdir(parents=True)
        save_file(student, folder / 'adapter_model.safetensors')
        save_file(teacher, folder / 'sdpo_teacher/adapter_model.safetensors')
        if step == 1:
            first_teacher = teacher
        previous = teacher
    assert audit(tmp_path)['actual_updates'] == 2
    for incorrect in (student, first_teacher):
        save_file(incorrect, tmp_path / 'checkpoint-2/sdpo_teacher/adapter_model.safetensors')
        with pytest.raises(AssertionError, match='step2'):
            audit(tmp_path)
