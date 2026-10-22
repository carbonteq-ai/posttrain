"""Explicit shared untrained control; never label reused episodes as a new run."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

RUNTIME_FILES = ("self_feedback_vllm.py", "sdpo_speculation.py", "sdpo_schema_parser.py",
                 "self_feedback_native.py", "self_campaign.py",
                 "sdpo_prompting.py", "self_feedback_signals.py", "sdpo_greedy_repeatability.py")
INPUT_KEYS = ("model", "model_family", "run_profile", "thinking", "concurrency", "speculation",
              "mtp_tokens", "tool_parser_mode", "disable_prefix_caching", "disable_chunked_prefill", "enforce_eager")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def student_code(source):
    tree = ast.parse(source)
    result = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if (cls.name, getattr(fn, "name", None)) not in {
                    ("SamplingRollout", "generate"), ("Agent", "completions"), ("Agent", "collect")}:
                continue
            if fn.name == "collect":
                # Ignore only the opt-in counter instrumentation, preserving
                # every student sampling, budget, environment and native call.
                fn.body = [node for node in fn.body if not (
                    isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "mtp_before"
                                                        for t in node.targets)
                    or isinstance(node, ast.If) and ast.unparse(node.test) in (
                        "getattr(self.args, 'capture_mtp_metrics', False)", "mtp_before is not None"))]
            result[f"{cls.name}.{fn.name}"] = hashlib.sha256(ast.dump(fn).encode()).hexdigest()
    if len(result) != 3:
        raise ValueError("student native interface source not found")
    return result


def fingerprint(driver, inputs, adapter_sha256):
    driver = Path(driver)
    nodes = ast.parse((driver / "sdpo_agent_config.py").read_text()).body
    helpers = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest() for node in nodes
               if isinstance(node, ast.FunctionDef) and node.name in ("aligned_messages", "population_rollouts")}
    if set(helpers) != {"aligned_messages", "population_rollouts"}:
        raise ValueError("student message/population source not found")
    return {"inputs": {k: inputs.get(k) for k in INPUT_KEYS}, "initial_adapter_sha256": adapter_sha256,
            "runtime_sources": {name: sha(driver / name) for name in RUNTIME_FILES},
            "student_message_population_helpers": helpers,
            "student_code": student_code((driver / "sdpo_agent_ablation.py").read_text())}


def inspect_population(folder, heldout):
    from analyze_sdpo_pool124 import bounded_scores, comparable_summary, generation_evidence
    from analyze_self_repair_run import native_tool_observations
    from sdpo_greedy_repeatability import audit

    folder = Path(folder)
    names = heldout["task_names"]
    if len(names) != len(set(names)) or len(names) != 62:
        raise ValueError("shared control requires the full heldout62 fixture")
    episodes = [json.loads(line) for line in (folder / "traces.jsonl").read_text().splitlines() if line]
    ids = [e["id"] for e in episodes]
    if len(ids) != len(set(ids)) or len(ids) != 310:
        raise ValueError("shared control lacks310unique native episodes")
    summary, bounds = bounded_scores(episodes, names, 5)
    if comparable_summary(summary) != comparable_summary(read(folder / "outcomes.json")):
        raise ValueError("shared control native/saved scores disagree")
    calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
    repeatability = audit(calls)
    if (not repeatability["qualified"]
            or repeatability != read(folder / "native-greedy-repeatability.json")):
        raise ValueError("shared baseline native repeatability failed or evidence differs")
    generated = [c for c in calls if c["action_ids"]]
    hashes = {c["adapter_sha256"] for c in generated}
    if (len(calls) < 310 or not generated or len(hashes) != 1
            or any(c["policy_version"] != 0 or not c["thinking_enabled"] for c in calls)
            or any(c["temperature"] != 0. for c in generated)
            or any(not c["action_ids"] and c.get("limit_reason") != "context_limit" for c in calls)):
        raise ValueError("shared control is not one untrained greedy thinking policy")
    return {"summary": summary, "bounded_metrics": bounds, "episode_ids": ids,
            "initial_adapter_sha256": next(iter(hashes)),
            "native_sha256": sha(folder / "traces.jsonl"),
            "tool_observations": native_tool_observations(episodes),
            "generation": generation_evidence(folder / "generation-calls.jsonl", 0)}


def export(root, driver, heldout_file, destination):
    root, destination = Path(root), Path(destination)
    source = root / "step-0000/heldout62"
    if not (source / "outcomes.json").exists():
        raise ValueError("only a closed complete baseline population may be shared")
    inputs, heldout = read(root / "inputs.json"), read(heldout_file)
    # Refuse open populations; audit the full raw snapshot after copying it.
    layout = {"traces.jsonl": source / "native/traces.jsonl",
              "outcomes.json": source / "outcomes.json",
              "generation-calls.jsonl": source / "generation-calls.jsonl",
              "native-greedy-repeatability.json": source / "native-greedy-repeatability.json",
              "heldout62.json": Path(heldout_file)}
    if not read(source / "native-greedy-repeatability.json")["qualified"]:
        raise ValueError("unqualified native repeatability cannot be exported as a shared control")
    destination.mkdir(exist_ok=False)
    for name, path in layout.items():
        shutil.copyfile(path, destination / name)
        if sha(path) != sha(destination / name):
            raise ValueError("baseline source changed during export; preserve export and reject")
    audited = inspect_population(destination, heldout)
    manifest = {"version": "guidance-five-shared-base-v1", "origin_run": inputs["run_name"],
                "origin_population": str(source), "files_sha256": {k: sha(destination / k) for k in layout},
                "fingerprint": fingerprint(driver, inputs, audited["initial_adapter_sha256"]),
                "native_episodes": 310, "scope": "Shared fresh base control. Reused native IDs are not new consumer episodes; no successful episodes are pooled across training attempts."}
    (destination / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(reference, driver, inputs, adapter_sha256, heldout_file):
    reference = Path(reference)
    manifest = read(reference)
    if manifest["version"] != "guidance-five-shared-base-v1":
        raise ValueError("unknown shared baseline version")
    if set(manifest["files_sha256"]) != {"traces.jsonl", "outcomes.json", "generation-calls.jsonl",
                                        "heldout62.json", "native-greedy-repeatability.json"}:
        raise ValueError("shared baseline file coverage differs")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha(reference.parent / name) != expected:
            raise ValueError("shared baseline file identity differs")
    if (manifest["files_sha256"]["heldout62.json"] != sha(heldout_file)
            or manifest["fingerprint"] != fingerprint(driver, inputs, adapter_sha256)):
        raise ValueError("shared control fixture, weights, decoding or student runtime differs")
    audited = inspect_population(reference.parent, read(heldout_file))
    return {**audited, "reference_sha256": sha(reference), "origin_run": manifest["origin_run"],
            "origin_population": manifest["origin_population"], "reused": True}
