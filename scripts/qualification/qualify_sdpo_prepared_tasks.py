"""Fresh native task evidence for a changed cohort, with no optimizer updates."""

import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reuse_native_reference(args, rollout):
    """Reuse the entire unchanged student component, never successful subsets."""
    import shutil

    from compose_sdpo_prepared_qualification import runtime_sources

    old = args.native_reference
    root = old / "evidence/preparation/task-qualification"
    receipt_file = root / "receipt.json"
    receipt = json.loads(receipt_file.read_text())
    proof = json.loads((old / "evidence/component-audit-r10-not-guide-approval/mtp-gate-receipt.json").read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    if (not proof["qualified"] or not receipt["qualified"] or receipt["native_episodes"] != 80
            or receipt["optimizer_updates"] != 0 or rollout.version != 0
            or receipt["initial_adapter_sha256"] != rollout.refreshes[-1]["weights_sha256"]
            or receipt["materials_sha256"] != sha(args.materials)
            or receipt["cohorts_sha256"] != sha(args.native_cohorts)
            or proof["changed_cohort_composition"]["new_native_receipt_sha256"] != sha(receipt_file)
            or json.loads((old / "evidence/preparation/decoding-request.json").read_text()) !=
               json.loads((args.output / "decoding-request.json").read_text())):
        raise ValueError("referenced native proof does not match current model/data/operating point")
    source = runtime_sources(old / "driver", Path(__file__).parent)
    files = proof["changed_cohort_composition"]["native_files"]
    for group in ("train", "online_train"):
        folder = root / "step-0000" / group
        if (sha(folder / "native/traces.jsonl") != files[group]["traces_sha256"]
                or sha(folder / "generation-calls.jsonl") != files[group]["calls_sha256"]):
            raise ValueError("referenced native files changed after independent component audit")
    target = args.output / "task-qualification"
    shutil.copytree(root, target)
    origin = {"capsule": str(old), "original_run": json.loads((old / "preparation.json").read_text())["run_name"],
        "receipt_sha256": sha(receipt_file), "source_compatibility": source,
        "native_episodes_generated_this_preparation": 0,
        "scope": "All80 unchanged previously qualified student episodes, referenced once, not new populations. "
                 "No guide-quality approval is inherited; current guides require separate full-source review."}
    receipt["referenced_from"] = origin
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "reused-native-proof.json").write_text(json.dumps(origin, indent=2) + "\n")
    return receipt


def retained_population(folder, wire, reconcile=None):
    """WireTaskData omits initial state; join the actual native archive explicitly."""
    if reconcile is None:
        from sdpo_luna_context import native_task_bindings

        reconcile = native_task_bindings
    retained = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines() if line]
    reconcile(wire, retained)
    return retained


def normalization(materials, episodes, normalizer, source_hash):
    """Require every retained native state to equal the actual pinned constructor."""
    sources = {r["task"]: r["material"]["native_data"]["initial_state"] for r in materials["rows"]}
    names = [e["task"]["data"]["name"] for e in episodes]
    ids = [e["id"] for e in episodes]
    if (len(sources) != 20 or len(ids) != len(set(ids)) or len(ids) != 40
            or Counter(names) != Counter({name: 2 for name in sources}) or not source_hash):
        raise ValueError("normalization needs40unique native episodes, exactly2per new20task cohort")
    canonical = {}
    for name, source in sources.items():
        value = copy.deepcopy(source)
        normalizer(value)
        canonical[name] = value
    for episode in episodes:
        name = episode["task"]["data"]["name"]
        if episode["task"]["data"]["initial_state"] != canonical[name]:
            raise ValueError(f"new native state differs from pinned constructor: {name}")
    return {"qualified": True, "native_episodes": len(ids), "native_episode_ids": ids,
            "mismatch": [], "normalized_states": canonical,
            "raw_source_hashes": {name: digest(source) for name, source in sources.items()},
            "canonical_hashes": {name: digest(state) for name, state in canonical.items()},
            "normalizer_sha256": source_hash,
            "scope": "Pinned constructor side effects on deep copies, checked against every fresh native task. "
                     "No fixture changes; original values/instructions retained."}


async def qualify(args, rollout, model, tokenizer):
    from aiohttp import web
    from automationbench_v1.world_codec import load_world
    from sdpo_agent_ablation import Agent
    from sdpo_greedy_repeatability import audit
    from sdpo_schema_parser import parser_class
    from verifiers_preinstalled import configure_preinstalled_runtime

    materials = json.loads(args.materials.read_text())
    cohort = json.loads(args.native_cohorts.read_text())
    names = cohort["train"]
    if (len(names) != len(set(names)) or len(names) != 20 or cohort["development"] or cohort["reserve"]
            or set(names) != {r["task"] for r in materials["rows"]}):
        raise ValueError("fresh native task qualification must use the exact new20-task training cohort")
    initial_sha = rollout.refreshes[-1]["weights_sha256"]
    if (rollout.version != 0
            or initial_sha != "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c"):
        raise ValueError("new-task qualification is not the exact pinned fresh initial student")
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)
    tokenizer._sdpo_model_family = "qwen3.5"
    output = args.output / "task-qualification"
    output.mkdir(exist_ok=False)
    agent = Agent(SimpleNamespace(output=output, context_limit=32768, reply_limit=4096,
        thinking=True, feedback_mode="minimal", model_family="qwen3.5", tool_parser_mode="schema-bound-v1",
        run_profile="guidance-five", concurrency=40, run_name=args.output.parent.name + "-new-tasks",
        teacher_feedback="native-feedback", online_rollouts=2, native_repeatability=True))
    agent.rollout, agent.tokenizer, agent.model = rollout, tokenizer, model
    agent.schema_parser, _ = parser_class()
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 18774).start()
        greedy_episodes, greedy_calls = await agent.collect(0, "train", names, temperature=0.)
        greedy_episodes = retained_population(agent.folder, greedy_episodes)
        repeatability = audit(greedy_calls)
        if not repeatability["qualified"] or repeatability["compared_pairs"] <= 0:
            raise ValueError("new-cohort native greedy repeatability failed")
        stochastic_episodes, calls = await agent.collect(0, "online_train", names, temperature=1.)
        stochastic_episodes = retained_population(agent.folder, stochastic_episodes)
        # load_world intentionally retains its actual constructor mutations in
        # the copied raw mapping, rather than replacing it by a full world dump.
        import inspect

        import automationbench.schema.google_sheets.base as sheets

        source_hash = hashlib.sha256(Path(inspect.getsourcefile(sheets)).read_bytes()).hexdigest()
        normalized = normalization(materials, greedy_episodes, load_world, source_hash)
        normalization(materials, stochastic_episodes, load_world, source_hash)
        (output / "state-normalization.json").write_text(json.dumps(normalized, indent=2) + "\n")
        receipt = {"qualified": True, "optimizer_updates": 0, "tasks": names,
            "initial_adapter_sha256": initial_sha, "policy_version": rollout.version,
            "native_episodes": 80, "greedy_episodes": 40, "stochastic_episodes": 40,
            "native_greedy_repeatability": repeatability,
            "stochastic_native": {"qualified": True, "episodes": len(stochastic_episodes),
                "calls": len(calls), "temperature": 1., "tasks": names, "optimizer_updates": 0,
                "native_greedy_repeatability": repeatability},
            "cohorts_sha256": hashlib.sha256(args.native_cohorts.read_bytes()).hexdigest(),
            "materials_sha256": hashlib.sha256(args.materials.read_bytes()).hexdigest(),
            "normalization_sha256": digest(normalized),
            "scope": "Fresh new-cohort native generation and state binding; separate from inherited model/runtime "
                     "capacity evidence. No learning or complete semantic goal/harm qualification claim."}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        await runner.cleanup()
