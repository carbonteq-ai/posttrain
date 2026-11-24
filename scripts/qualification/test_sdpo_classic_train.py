"""CPU tests of the arms 2/3 training pass (sdpo_classic.train_pass). The 15 Nov memory proof crashed because the loss
returns per-token importance weights as a list; these tests run the loop with that exact return shape.
Needs torch (tools/torchcpu); skipped without it."""
import json

import pytest

torch = pytest.importorskip("torch")

import sdpo_classic  # noqa: E402


def fake_loss(model, student, teacher):
    """The runner's episode_loss shape: (scalar loss, per-token weights as a list)."""
    out = model(torch.tensor(student["x"], dtype=torch.float32))
    return ((out - torch.tensor(teacher["y"], dtype=torch.float32)) ** 2).mean(), [1.0, 0.5] * len(student["x"])


def prepared(n):
    return [({"episode_id": f"e{i}", "task": "t"}, {"student": {"x": [[1.0, 2.0]]}, "teacher": {"y": [[3.0]]}})
            for i in range(n)]


def test_list_weights_train_and_log_every_step(tmp_path):
    model = torch.nn.Linear(2, 1)
    before = model.weight.detach().clone()
    logged = []
    steps, losses, oom, peak = sdpo_classic.train_pass(model, prepared(5), fake_loss, 1e-2, 2, tmp_path / "s.jsonl",
                                                       lambda **k: logged.append(k), peak_mib=lambda: 7)
    assert [s["attempts"] for s in steps] == [2, 4, 5]  # a short last batch still steps
    assert len(losses) == 5 and oom == 0 and peak == 7
    assert all(s["ratio"] == pytest.approx(0.75) for s in steps)
    assert not torch.equal(before, model.weight.detach())
    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert rows == steps and [k["stage"] for k in logged] == ["classic_step"] * 3


def test_out_of_memory_is_skipped_and_counted(tmp_path):
    def flaky(model, student, teacher):
        if student.get("big"):
            raise torch.OutOfMemoryError("fake")
        return fake_loss(model, student, teacher)

    data = prepared(3)
    data[1] = (data[1][0], {"student": {"x": [[1.0, 2.0]], "big": True}, "teacher": {"y": [[3.0]]}})
    logged = []
    steps, losses, oom, _ = sdpo_classic.train_pass(torch.nn.Linear(2, 1), data, flaky, 1e-2, 16, tmp_path / "s.jsonl",
                                                    lambda **k: logged.append(k), peak_mib=lambda: 0)
    assert oom == 1 and len(losses) == 2 and len(steps) == 1
    assert {"stage": "classic_oom", "episode_id": "e1", "task": "t"} in logged


def test_no_weights_gives_no_ratio(tmp_path):
    steps, *_ = sdpo_classic.train_pass(torch.nn.Linear(2, 1), prepared(1), lambda m, s, t: (fake_loss(m, s, t)[0], []),
                                        1e-2, 1, tmp_path / "s.jsonl", lambda **k: None, peak_mib=lambda: 0)
    assert steps[0]["ratio"] is None
