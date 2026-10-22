"""Check task weighting and privileged-teacher gradient isolation."""

import ast
from pathlib import Path

import pytest
import torch


def load(name, **globals_):
    tree = ast.parse(Path(__file__).with_name("self_argument_repair_control.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, **globals_}
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
    return namespace[name]


def test_equal_task_norms_with_unequal_original_magnitudes():
    groups = [[torch.tensor([100., 0.])], [torch.tensor([0., 1.])]]
    combined, norms, scales = load("equalize_task_gradients")(groups)
    torch.testing.assert_close(combined[0], torch.tensor([50.5, 50.5]))
    assert norms == [100., 1.]
    assert norms[0] * scales[0] == pytest.approx(norms[1] * scales[1])


def test_zero_task_gradient_is_not_silently_dropped():
    with pytest.raises(ValueError, match="nonzero"):
        load("equalize_task_gradients")([[torch.zeros(3)]])


def test_teacher_position_loss_matches_distribution_not_hard_label():
    class Model:
        adapter = "default"

        def set_adapter(self, value):
            self.adapter = value

        def train(self):
            pass

        def eval(self):
            pass

    model = Model()
    p = torch.tensor([[0., 1., 2.], [1., 2., 3.]], requires_grad=True)
    q = torch.tensor([[2., 3., 1.], [0., 1., 1.]], requires_grad=True)
    function = load("repair_teacher_kl", corrected_action=lambda *_: ([2, 2], 1),
                    logits=lambda m, *_: q if m.adapter == "self_teacher" else p)
    loss = function(model, {"row": {"student_prefix_ids": [5], "teacher_prefix_ids": [8, 9],
                                   "action_ids": [0, 1]}}, None)
    expected = torch.nn.functional.kl_div(p[1].log_softmax(-1), q[1].detach().softmax(-1), reduction="sum")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert q.grad is None
    torch.testing.assert_close(p.grad[0], torch.zeros(3))
    torch.testing.assert_close(p.grad[1], p[1].detach().softmax(-1) - q[1].detach().softmax(-1))
    assert model.adapter == "default"


def test_full_repair_span_uses_corrected_branch_and_excludes_unchanged_syntax():
    class Model:
        adapter = "default"

        def set_adapter(self, value):
            self.adapter = value

        def train(self):
            pass

        def eval(self):
            pass

    model = Model()
    p = torch.arange(20, dtype=torch.float32).reshape(4, 5).requires_grad_()
    q = torch.flip(p.detach(), dims=[1]).requires_grad_()
    targets = []

    def logits(m, prefix, target):
        targets.append(target)
        return q if m.adapter == "self_teacher" else p

    function = load("repair_teacher_kl", corrected_action=lambda *_: ([0, 2, 3, 4], 1),
                    changed_span=load("changed_span"), logits=logits)
    loss = function(model, {"row": {"student_prefix_ids": [5], "teacher_prefix_ids": [8, 9],
                                   "action_ids": [0, 1, 4]}}, None, full_span=True)
    expected = torch.nn.functional.kl_div(p[1:3].log_softmax(-1), q[1:3].detach().softmax(-1), reduction="batchmean")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert targets == [[0, 2, 3, 4], [0, 2, 3, 4]]
    assert q.grad is None
    torch.testing.assert_close(p.grad[[0, 3]], torch.zeros(2, 5))


def test_full_repair_span_refuses_overlong_teacher_context():
    function = load("repair_teacher_kl", corrected_action=lambda *_: ([0, 2, 3, 4], 1),
                    changed_span=load("changed_span"))
    with pytest.raises(ValueError, match="no truncation"):
        function(None, {"row": {"student_prefix_ids": [5], "teacher_prefix_ids": [8] * 16384,
                                "action_ids": [0, 1, 4]}}, None, full_span=True)


def test_teacher_span_audit_checks_later_tokens_and_restores_student():
    class Model:
        adapter = "default"

        def set_adapter(self, value):
            self.adapter = value

        def eval(self):
            pass

    class Tokenizer:
        def decode(self, tokens):
            return ",".join(map(str, tokens))

    model = Model()
    scores = torch.zeros(4, 5)
    scores[1, 2], scores[2, 0] = 5., 5.  # First corrected token right, later token wrong.
    function = load("teacher_span_audit", corrected_action=lambda *_: ([0, 2, 3, 4], 1),
                    changed_span=load("changed_span"), logits=lambda *_: scores)
    result = function(model, [{"task": "task", "call": {"messages": []},
                     "row": {"teacher_prefix_ids": [8], "action_ids": [0, 1, 4]}}], Tokenizer())
    assert result[0]["tokens"][0]["correct_rank"] == 1
    assert result[0]["tokens"][1]["teacher_top1"] == "0"
    assert result[0]["tokens"][1]["correct_probability"] < .01
    assert model.adapter == "default"


def test_prefix_anchor_matches_span_plus_first_token_kl():
    class Model:
        adapter = "default"

        def set_adapter(self, value):
            self.adapter = value

        def train(self):
            pass

        def eval(self):
            pass

    p = torch.arange(20, dtype=torch.float32).reshape(4, 5).requires_grad_()
    q = torch.flip(p.detach(), dims=[1]).requires_grad_()
    function = load("repair_teacher_kl", corrected_action=lambda *_: ([0, 2, 3, 4], 1),
                    changed_span=load("changed_span"), logits=lambda m, *_: q if m.adapter == "self_teacher" else p)
    loss, parts = function(Model(), {"row": {"student_prefix_ids": [5], "teacher_prefix_ids": [8],
                           "action_ids": [0, 1, 4]}}, None, full_span=True, prefix_anchor=True, return_parts=True)
    per_token = torch.nn.functional.kl_div(p[1:3].log_softmax(-1), q[1:3].detach().softmax(-1), reduction="none").sum(-1)
    expected = per_token.mean() + per_token[0]
    torch.testing.assert_close(loss, expected)
    assert parts["prefix_anchor_weight"] == 1
    loss.backward()
    assert q.grad is None
    torch.testing.assert_close(p.grad[1], 1.5 * (p[1].detach().softmax(-1) - q[1].detach().softmax(-1)))
    torch.testing.assert_close(p.grad[2], .5 * (p[2].detach().softmax(-1) - q[2].detach().softmax(-1)))
