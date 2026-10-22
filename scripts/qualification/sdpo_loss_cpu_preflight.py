"""Check diagnostic instrumentation against the pinned SDPO objective on CPU."""

import json
from types import SimpleNamespace

import sdpo_agent_ablation as runtime
import torch


def main():
    torch.manual_seed(19)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.randn(3, 32))

    model = Model()
    teacher = torch.randn(3, 32)
    teacher[:, 1] += 5
    action = [2, 3, 4]
    row = {"student_prefix_ids": [0], "teacher_prefix_ids": [0, 1],
           "action_ids": action,
           "sampled_logprobs": model.scores.detach().log_softmax(-1)[range(3), action].tolist()}
    original_logits, original_tensor = runtime.logits, torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    runtime.logits = lambda model, prefix, actions: model.scores
    torch.tensor = cpu_tensor
    try:
        expected = runtime.compute_topk_self_distillation_loss(
            model.scores, teacher, distillation_topk=20,
            distillation_alpha=1., distillation_add_tail=True).mean()
        expected_gradient = torch.autograd.grad(expected, model.scores)[0]
        loss, advantages = runtime.loss_for(model, row, SimpleNamespace(topk=20, alpha=1.), teacher=teacher)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(model.scores.grad, expected_gradient)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        optimizer.step()
        runtime.score_sampled(model, row)
        diagnostic = row["teacher_support_diagnostic"]
        assert len(advantages) == 3
        assert all(1 <= rank <= 32 for rank in diagnostic["teacher_top1_student_ranks"])
        assert all(0 <= mass <= 1 for mass in diagnostic["teacher_mass_on_student_support"])
        assert len(diagnostic["actual_adam_teacher_top1_logprob_deltas"]) == 3
        assert len(diagnostic["sampled_token_student_ranks"]) == 3
        assert diagnostic["importance_weights"] == [1., 1., 1.]

        # Deliberately different training/evaluation logits expose a mixed-mode
        # baseline bug; compare both preferred and original tokens in eval mode.
        bias = torch.zeros_like(model.scores)
        bias[:, 0] = 2.
        runtime.logits = lambda model, prefix, actions: model.scores + (bias if model.training else 0)
        preferred_ids = teacher.argmax(-1).tolist()
        eval_before = model.scores.detach().log_softmax(-1)[range(3), preferred_ids].clone()
        train_before = (model.scores.detach() + bias).log_softmax(-1)[range(3), preferred_ids].clone()
        before, preferred_before = runtime.score_sampled(model, row, preferred_ids=preferred_ids)
        torch.testing.assert_close(preferred_before, eval_before)
        model.train()
        row["sampled_logprobs"] = (model.scores.detach() + bias).log_softmax(-1)[range(3), action].tolist()
        model.zero_grad(set_to_none=True)
        expected = runtime.compute_topk_self_distillation_loss(
            model.scores + bias, teacher, distillation_topk=20,
            distillation_alpha=1., distillation_add_tail=True).mean()
        expected_gradient = torch.autograd.grad(expected, model.scores)[0]
        loss, _ = runtime.loss_for(model, row, SimpleNamespace(topk=20, alpha=1.), teacher=teacher)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(model.scores.grad, expected_gradient)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        after, preferred_after = runtime.score_sampled(model, row, preferred_ids=preferred_ids)
        eval_after = model.scores.detach().log_softmax(-1)[range(3), preferred_ids]
        torch.testing.assert_close(preferred_after - preferred_before, eval_after - eval_before)
        assert (train_before - eval_before).abs().max() > 1e-4
        assert len(after - before) == 3
        print(json.dumps({"passed": True, "objective_and_gradient_unchanged": True,
                          "support_and_adam_diagnostic_shapes_valid": True,
                          "same_mode_preferred_token_baseline_verified": True,
                          "full_importance_and_sampled_rank_diagnostics_valid": True,
                          "scope": "Synthetic CPU tensor check, zero native rollouts or GPU updates"}))
    finally:
        runtime.logits, torch.tensor = original_logits, original_tensor


if __name__ == "__main__":
    main()
