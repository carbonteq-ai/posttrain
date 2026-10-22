import hashlib
import json

import pytest
from compose_sdpo_plain_qualification import IMAGE, MODEL, RUNTIME_FILES, compose
from sdpo_greedy_repeatability import audit
from self_campaign_config import summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    n, c = tmp_path / "native", tmp_path / "capacity"
    names = [f"task{i}" for i in range(20)]
    weights = hashlib.sha256(b"same initial adapter").hexdigest()
    for root in (n, c):
        for name in set(RUNTIME_FILES) | {"sdpo_agent_ablation.py", "sdpo_capacity.py", "sdpo_agent_config.py"}:
            f = root / "driver" / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("# matched runtime\n")
        files = sorted((root / "driver").glob("*.py"))
        (root / "source-sha256.txt").write_text("".join(
            hashlib.sha256(f.read_bytes()).hexdigest() + "  " + str(f.relative_to(root)) + "\n" for f in files))
        evidence = root / "evidence/mtp"
        (evidence / "initial-student").mkdir(parents=True)
        (evidence / "initial-student/adapter_model.safetensors").write_bytes(b"same initial adapter")
        write(evidence / "requested.json", dict(method="none", speculative_config=None,
            concurrency=40, context=65536, optimizer_updates=0, prefix_caching=False,
            chunked_prefill=False, enforce_eager=False))
        write(root / "qualification.json", dict(method="none", gdn_decode_kernel="triton",
            experiment_deadline_unix=9000, budget_window_started_unix=1860,
            budget_authorization="user-fresh-plain-four-windows-20261019", logical_experiment=2))
    batch = dict(exact_batch_invariance=True, records=[dict(peak_in_flight=40)])
    write(n / "cohorts.json", {"train": names})
    for group, temperature in (("train", 0.), ("online_train", 1.)):
        folder = n / "evidence/mtp/step-0000" / group
        episodes = [dict(id=f"{group}-{task}-{rep}", task={"data": {"name": task}}, traces=[dict(
            stop_condition="agent_completed", metrics={"task_completed_correctly": 1},
            rewards={"partial_credit": {"score": 1}})]) for task in names for rep in range(2)]
        folder.joinpath("native").mkdir(parents=True)
        (folder / "native/traces.jsonl").write_text("\n".join(map(json.dumps, episodes)))
        write(folder / "outcomes.json", summarize(episodes, names, 2))
        calls = [dict(complete=True, temperature=temperature, policy_version=0, adapter_sha256=weights,
            prefix_ids=[i//2], requested_completion_budget=4096, action_ids=[2], sampled_logprobs=[-.1])
            for i in range(40)]
        (folder / "generation-calls.jsonl").write_text("\n".join(map(json.dumps, calls)))
        if group == "train":
            write(n / "evidence/mtp/native-greedy-repeatability.json", audit(calls))
    write(n / "evidence/mtp/native-tool-qualification/receipt.json", {"qualified": True})
    write(n / "evidence/mtp/batch-invariance-v0.json", batch)
    write(c / "evidence/mtp/receipt.json", dict(qualified=True, component="plain-capacity-only-v1",
        method="none", optimizer_updates=0, initial_adapter_sha256=weights,
        actor_capacity=dict(qualified=True, weights_unchanged=True, finite_nonzero_student_gradients=True,
            optimizer_updates=0, student_tokens=32768, teacher_tokens=32768, scored_target_tokens=4096, loss=.1),
        rollout_capacity=dict(context_limit=32768, reply_limit=4096, generation_prompt_tokens=28672,
            generated_tokens=4096, parity={"max_abs_delta": .125}, batch_invariance=batch)))
    state = dict(Running=False, OOMKilled=False, ExitCode=124)
    inspect_n = dict(State=state, Config=dict(Image=IMAGE, Cmd=["--model", MODEL]))
    inspect_c = dict(State={**state, "ExitCode": 0}, Config=dict(Image=IMAGE, Cmd=["--model", MODEL]))
    return n, c, inspect_n, inspect_c


def test_qualified_composition_preserves_native_timeout_as_provenance(tmp_path):
    result = compose(*fixture(tmp_path))
    assert result["qualified"] and result["optimizer_updates"] == 0
    assert result["components"]["native_terminal_state"]["ExitCode"] == 124
    assert result["actor_capacity"]["scored_target_tokens"] == 4096
    assert result["stochastic_native"]["episodes"] == 40


@pytest.mark.parametrize("failure", ["weights", "source", "capacity-exit", "image", "missing-score", "repeatability", "gradient"])
def test_incompatible_or_incomplete_components_never_qualify(tmp_path, failure):
    n, c, ns, cs = fixture(tmp_path)
    if failure == "weights":
        (c / "evidence/mtp/initial-student/adapter_model.safetensors").write_bytes(b"different")
    elif failure == "source":
        (c / "driver/sdpo_capacity.py").write_text("# altered\n")
    elif failure == "capacity-exit":
        cs["State"]["ExitCode"] = 124
    elif failure == "image":
        cs["Config"]["Image"] = "different image"
    elif failure == "missing-score":
        f = n / "evidence/mtp/step-0000/online_train/native/traces.jsonl"
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        rows[0]["traces"][0]["metrics"] = {}
        f.write_text("\n".join(map(json.dumps, rows)))
    elif failure == "repeatability":
        f = n / "evidence/mtp/native-greedy-repeatability.json"
        value = json.loads(f.read_text())
        value["compared_pairs"] += 1
        write(f, value)
    else:
        f = c / "evidence/mtp/receipt.json"
        value = json.loads(f.read_text())
        value["actor_capacity"]["finite_nonzero_student_gradients"] = False
        write(f, value)
    with pytest.raises(ValueError):
        compose(n, c, ns, cs)
