import hashlib
import json
from types import SimpleNamespace

import pytest
from qualify_sdpo_prepared_tasks import reuse_native_reference


def fixture(tmp_path, monkeypatch):
    import compose_sdpo_prepared_qualification as composition

    monkeypatch.setattr(composition, "runtime_sources", lambda *_: {"verified": True})
    old, output = tmp_path / "old", tmp_path / "new"
    root = old / "evidence/preparation/task-qualification"
    root.mkdir(parents=True)
    output.mkdir()
    materials, cohort = tmp_path / "materials.json", tmp_path / "cohort.json"
    materials.write_text("materials")
    cohort.write_text("cohort")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    files = {}
    for group in ("train", "online_train"):
        folder = root / "step-0000" / group
        (folder / "native").mkdir(parents=True)
        (folder / "native/traces.jsonl").write_text(group)
        (folder / "generation-calls.jsonl").write_text(group + " calls")
        files[group] = {"traces_sha256": sha(folder / "native/traces.jsonl"),
                        "calls_sha256": sha(folder / "generation-calls.jsonl")}
    receipt = {"qualified": True, "native_episodes": 80, "optimizer_updates": 0,
        "initial_adapter_sha256": "initial", "materials_sha256": sha(materials), "cohorts_sha256": sha(cohort)}
    (root / "receipt.json").write_text(json.dumps(receipt))
    proof = {"qualified": True, "changed_cohort_composition": {
        "new_native_receipt_sha256": sha(root / "receipt.json"), "native_files": files}}
    folder = old / "evidence/component-audit-r10-not-guide-approval"
    folder.mkdir()
    (folder / "mtp-gate-receipt.json").write_text(json.dumps(proof))
    (old / "preparation.json").write_text(json.dumps({"run_name": "native-origin"}))
    for path in (old / "evidence/preparation/decoding-request.json", output / "decoding-request.json"):
        path.write_text(json.dumps({"speculative_config": None, "concurrency": 40}))
    return SimpleNamespace(native_reference=old, materials=materials, native_cohorts=cohort, output=output), (
        SimpleNamespace(version=0, refreshes=[{"weights_sha256": "initial"}]))


def test_entire_native_component_is_referenced_without_new_episode_claim(tmp_path, monkeypatch):
    args, rollout = fixture(tmp_path, monkeypatch)
    receipt = reuse_native_reference(args, rollout)
    assert receipt["referenced_from"]["native_episodes_generated_this_preparation"] == 0
    assert receipt["referenced_from"]["original_run"] == "native-origin"
    for group in ("train", "online_train"):
        assert (args.output / f"task-qualification/step-0000/{group}/native/traces.jsonl").read_text() == group


@pytest.mark.parametrize("change", ["weights", "materials", "native_file"])
def test_changed_weights_data_or_native_files_reject_reference(tmp_path, monkeypatch, change):
    args, rollout = fixture(tmp_path, monkeypatch)
    if change == "weights":
        rollout.refreshes[-1]["weights_sha256"] = "other"
    elif change == "materials":
        args.materials.write_text("other population")
    else:
        (args.native_reference / "evidence/preparation/task-qualification/step-0000/train/native/traces.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="referenced native"):
        reuse_native_reference(args, rollout)
    assert not (args.output / "task-qualification").exists()
