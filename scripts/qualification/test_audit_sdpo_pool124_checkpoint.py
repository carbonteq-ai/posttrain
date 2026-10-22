"""Detect mislabeled moments and changed checkpoint files before continuation."""

import hashlib
import json

import pytest
import torch
from audit_sdpo_pool124_checkpoint import audit
from safetensors.torch import save_file
from sdpo_agent_config import update_learning_rate


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "inputs.json").write_text(json.dumps({"run_profile": "pool124-50", "steps": 50,
                                                    "lr": 1e-5, "lr_end_ratio": .1}))
    folder = tmp_path / "checkpoint-10"
    (folder / "sdpo_teacher").mkdir(parents=True)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    for _ in range(10):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    torch.save({"torch": torch.get_rng_state()}, folder / "rng_state.pth")
    for relative in ("adapter_model.safetensors", "sdpo_teacher/adapter_model.safetensors"):
        save_file({"linear.lora_B.weight": torch.ones(1)}, folder / relative)
    data = {"step": 10, "ema_updates": 10, "batch": 32, "run_profile": "pool124-50",
            "resume_qualified": False, "learning_rate": 1e-5,
            "files_sha256": {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in folder.rglob("*") if p.is_file()}}
    (folder / "campaign-checkpoint.json").write_text(json.dumps(data))
    return tmp_path


def test_live_audit_requires_only_the_requested_saved_checkpoint(checkpoint):
    assert audit(checkpoint, 10)["actual_saved_optimizer_moments"] == [10]


def test_changed_file_is_rejected(checkpoint):
    (checkpoint / "checkpoint-10/rng_state.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file hash mismatch"):
        audit(checkpoint, 10)


def test_mislabeled_actual_optimizer_moment_is_rejected(checkpoint):
    folder = checkpoint / "checkpoint-10"
    optimizer = torch.load(folder / "optimizer.pt", weights_only=True)
    for state in optimizer["state"].values():
        state["step"] = torch.tensor(9.)
    torch.save(optimizer, folder / "optimizer.pt")
    metadata = json.loads((folder / "campaign-checkpoint.json").read_text())
    metadata["files_sha256"]["optimizer.pt"] = hashlib.sha256((folder / "optimizer.pt").read_bytes()).hexdigest()
    (folder / "campaign-checkpoint.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="actual saved moment"):
        audit(checkpoint, 10)


@pytest.mark.parametrize("step", [1, 6])
def test_bounded_checkpoint_checks_real_moments_and_cosine_lr(tmp_path, step):
    inputs = dict(run_profile="pool124-bounded", steps=6, batch_size=32, online_rollouts=2,
                  lr=1e-5, lr_end_ratio=.1, lr_schedule="cosine-decay", context_limit=32768,
                  reply_limit=4096, qualify_capacity=True, thinking=False,
                  teacher_feedback="solution-or-feedback", tool_parser_mode="schema-bound-v1",
                  selection_mode="failure-frontier", alpha=1., topk=20, teacher_rate=.01)
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    folder = tmp_path / f"checkpoint-{step}"
    (folder / "sdpo_teacher").mkdir(parents=True)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    for index in range(1, step + 1):
        optimizer.param_groups[0]["lr"] = update_learning_rate(1e-5, index, 6, "cosine-decay", .1)
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    expected = optimizer.param_groups[0]["lr"]
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    torch.save({"torch": torch.get_rng_state()}, folder / "rng_state.pth")
    for relative in ("adapter_model.safetensors", "sdpo_teacher/adapter_model.safetensors"):
        save_file({"linear.lora_B.weight": torch.ones(1)}, folder / relative)
    metadata = dict(step=step, ema_updates=step, batch=32, run_profile="pool124-bounded",
                    resume_qualified=False, learning_rate=expected,
                    files_sha256={str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in folder.rglob("*") if p.is_file()})
    (folder / "campaign-checkpoint.json").write_text(json.dumps(metadata))
    result = audit(tmp_path, step)
    assert result["actual_saved_optimizer_moments"] == [step]
    assert result["learning_rate"] == pytest.approx(expected)
    (tmp_path / "inputs.json").write_text(json.dumps(inputs | {"steps": 20}))
    with pytest.raises(ValueError, match="authorized matched recipe"):
        audit(tmp_path, step)
