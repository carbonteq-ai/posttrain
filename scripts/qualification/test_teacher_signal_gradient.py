"""Verify the signal diagnostic against the actual full-vocabulary KL gradient."""

import ast
import unittest
from pathlib import Path

import torch


class GradientChecks(unittest.TestCase):
    def compare(self, student, teacher):
        source = ast.parse(Path(__file__).with_name('self_teacher_signal_probe.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'compare')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'signal-comparison', 'exec'), namespace)
        logits = torch.tensor(student, dtype=torch.float64).log().requires_grad_()
        p = logits.log_softmax(-1)
        q = torch.tensor(teacher, dtype=torch.float64).log()
        loss = (p.exp() * (p - q)).sum()
        grad, = torch.autograd.grad(loss, logits)
        result = namespace['compare'](p.detach(), q, 0, 1)
        self.assertAlmostEqual(result['descent_correct_vs_bad_log_odds_direction'], float(grad[0] - grad[1]), places=10)
        return result

    def test_teacher_preference_alone_does_not_prove_correct_gradient(self):
        result = self.compare([0.1, 0.01, 0.89], [0.49, 0.5099, 0.0001])
        self.assertTrue(result['teacher_prefers_correct_token'])
        self.assertLess(result['descent_correct_vs_bad_log_odds_direction'], 0)
        self.assertFalse(result['gate_passed'])

    def test_clear_corrective_distribution_passes(self):
        result = self.compare([0.8, 0.1, 0.1], [0.001, 0.98, 0.019])
        self.assertTrue(result['gate_passed'])


if __name__ == '__main__':
    unittest.main()
