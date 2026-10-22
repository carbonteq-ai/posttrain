"""CPU-only independent check of saved EMA teachers against every student update."""

import argparse
import json
from pathlib import Path


def audit(root):
    import torch
    from safetensors.torch import load_file

    config = json.loads((root / 'inputs.json').read_text())
    rate = config['teacher_rate']
    if not 0 < rate <= 1:
        raise ValueError('invalid configured teacher update rate')
    updates = json.loads((root / 'updates.json').read_text())
    if not updates or [u['step'] for u in updates] != list(range(1, len(updates) + 1)):
        raise ValueError('no contiguous applied update population')
    previous = load_file(root / 'initial-student/adapter_model.safetensors', device='cpu')
    results = []
    for update in updates:
        step = update['step']
        checkpoint = root / f'checkpoint-{step}'
        student = load_file(checkpoint / 'adapter_model.safetensors', device='cpu')
        teacher = load_file(checkpoint / 'sdpo_teacher/adapter_model.safetensors', device='cpu')
        if previous.keys() != student.keys() or student.keys() != teacher.keys():
            raise ValueError('student and teacher adapter tensor identities differ')
        maximum_error = 0.
        for key, old in previous.items():
            # A tiny tolerance allows CPU/GPU FP32 fused-operation rounding;
            # it must not hide a missing update on the small LoRA B weights.
            expected = old.lerp(student[key].to(old.dtype), rate)
            if not torch.isfinite(expected).all() or not torch.isfinite(teacher[key]).all():
                raise ValueError('nonfinite saved EMA tensor')
            torch.testing.assert_close(teacher[key], expected, atol=1e-10, rtol=1e-6,
                                       msg=lambda m, step=step, key=key: f'step{step} tensor{key}: {m}')
            maximum_error = max(maximum_error, float((teacher[key] - expected).abs().max()))
        results.append({'step': step, 'tensors_checked': len(teacher), 'max_abs_error': maximum_error})
        previous = teacher
    return {'qualified': True, 'actual_updates': len(updates), 'teacher_update_rate': rate,
            'steps': results, 'GPU_used': False,
            'scope': 'Saved EMA recurrence independently matches every saved student update. '
                     'Optimizer moments, runtime token alignment and task rewards are separate audits.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + '\n')
