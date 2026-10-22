"""A final phase must preserve the original time and frozen adapter identities."""

import hashlib
import json

import pytest
import stage_sdpo_reserve_phase as runtime
from sdpo_agent_config import MODEL_REVISION, fingerprint
from sdpo_execution_budget import start_shared_budget


def fixture(tmp_path, monkeypatch):
    fitting = tmp_path / "fit"
    (fitting / "driver").mkdir(parents=True)
    (fitting / "evidence").mkdir()
    (fitting / "driver/run_sdpo_reserve_eval.sh").write_text("source identity fixture")
    (fitting / "experiment.json").write_text(json.dumps({"budget_slot": 10, "steps": 10,
        "shared_slot10_budget": True, "run_name": "owned-thinking-fit", "validation_run_name": "owned-validation"}))
    (fitting / "cohorts.json").write_text(json.dumps({"reserve": ["sealed-task"]}))
    (fitting / "source-sha256.txt").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(fitting)}\n"
        for p in sorted(fitting.rglob("*")) if p.is_file()))
    start_shared_budget(fitting / "evidence/shared-budget.json", "owned-thinking-fit", 600)
    (fitting / "evidence/run").mkdir()
    (fitting / "evidence/run/receipt.json").write_text(json.dumps({"completed": True, "optimizer_updates": 10}))
    (fitting / "evidence/run/updates.json").write_text(json.dumps([
        {"step": step, "batch": 16, "optimizer_moment_steps": [step], "ema_updates": step}
        for step in range(1, 11)]))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"immutable identity, not a GPU model fixture")
    (adapter / "adapter_config.json").write_text("{}")
    primary = tmp_path / "selection.json"
    primary.write_text(json.dumps({"approved_for_reserve": True, "frozen_before_reserve": True,
        "model_revision": MODEL_REVISION, "budget_slot": 8, "thinking": False,
        "context_limit": 16384, "reply_limit": 2048, "tool_parser_mode": "schema-bound-v1",
        "adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "reserve_sha256": fingerprint(["sealed-task"])}))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "exited:0\n")
    return fitting, primary, adapter, tmp_path / "validation"


def test_staging_keeps_exact_original_deadline_and_source_identities(tmp_path, monkeypatch):
    fitting, primary, adapter, destination = fixture(tmp_path, monkeypatch)
    result = runtime.stage(fitting, primary, adapter, destination)
    assert result["staged"] and not result["GPU_submitted"]
    assert (destination / "parent-budget.json").read_bytes() == (fitting / "evidence/shared-budget.json").read_bytes()
    assert (destination / "primary-selection.json").read_bytes() == primary.read_bytes()
    assert json.loads((destination / "experiment.json").read_text())["run_name"] == "owned-validation"
    assert 0 < result["remaining_seconds"] <= 6570
    with pytest.raises(FileExistsError):
        runtime.stage(fitting, primary, adapter, destination)


def test_tampered_adapter_and_unfinished_parent_cannot_start_validation(tmp_path, monkeypatch):
    fitting, primary, adapter, destination = fixture(tmp_path, monkeypatch)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="adapter"):
        runtime.stage(fitting, primary, adapter, destination)
    assert not destination.exists()
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "running:0\n")
    with pytest.raises(ValueError, match="not succeeded"):
        runtime.stage(fitting, primary, adapter, destination)


def test_altered_fitting_source_is_not_silently_resealed(tmp_path, monkeypatch):
    fitting, primary, adapter, destination = fixture(tmp_path, monkeypatch)
    (fitting / "driver/run_sdpo_reserve_eval.sh").write_text("unreviewed edit")
    with pytest.raises(ValueError, match="source manifest changed"):
        runtime.stage(fitting, primary, adapter, destination)
    assert not destination.exists()


def test_provider_success_without_all_actual_updates_cannot_validate(tmp_path, monkeypatch):
    fitting, primary, adapter, destination = fixture(tmp_path, monkeypatch)
    (fitting / "evidence/run/updates.json").write_text("[]")
    with pytest.raises(ValueError, match="actual optimizer"):
        runtime.stage(fitting, primary, adapter, destination)
    assert not destination.exists()
