"""Verify forward-KL orientation, full-vocabulary gradient and teacher detachment."""

import ast
from pathlib import Path

import torch


def implementation():
    source = ast.parse(Path(__file__).with_name("self_batch_signal_diagnosis.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "forward_kl")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "forward-kl", "exec"), namespace)
    return namespace["forward_kl"]


def test_forward_kl_matches_reference_and_detaches_teacher():
    p = torch.tensor([[1., -2., .3], [.2, 1., -1.]], requires_grad=True)
    q = torch.tensor([[-2., 3., .7], [1., -.3, .1]], requires_grad=True)
    actual = implementation()(p, q)
    reference = torch.nn.functional.kl_div(p.log_softmax(-1), q.detach().softmax(-1), reduction="batchmean")
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert q.grad is None
    torch.testing.assert_close(p.grad, (p.detach().softmax(-1) - q.detach().softmax(-1)) / 2)


def test_equal_distributions_have_zero_loss_and_gradient():
    p = torch.tensor([[.1, .3, -.8]], requires_grad=True)
    loss = implementation()(p, p.clone())
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.))
    torch.testing.assert_close(p.grad, torch.zeros_like(p), atol=1e-7, rtol=0)
