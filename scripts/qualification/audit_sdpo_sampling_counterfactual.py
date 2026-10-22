"""CPU replay of saved selection metadata; never claims counterfactual learning."""

import argparse
import json
from pathlib import Path

from sdpo_action_risks import action_risks
from sdpo_agent_config import observed_tool_error
from sdpo_guidance_context import native_findings
from sdpo_guidance_sampling import batch


def replay(folder, guidance, tasks, step):
    def read(name):
        return json.loads((folder / name).read_text())

    calls = [json.loads(x) for x in (folder / "generation-calls.jsonl").read_text().splitlines() if x]
    episodes = [json.loads(x) for x in (folder / "native/traces.jsonl").read_text().splitlines() if x]
    admission = read("training-admission.json")
    excluded = {x["generation_index"] for x in admission["exclusions"]}
    tools, histories, prompts = {}, {}, {}
    for episode in episodes:
        messages = [n["message"] for n in episode["traces"][0]["nodes"] if n.get("message")]
        user = next(m["content"] for m in messages if m["role"] == "user")
        prompts.setdefault(user, []).append(episode)
        for message in messages:
            for tool in message.get("tool_calls") or []:
                if tool["id"] in tools:
                    raise ValueError("duplicate native tool identity")
                tools[tool["id"]] = episode
            if message["role"] == "tool":
                histories[message["tool_call_id"]] = episode
    rows = []
    for index, call in enumerate(calls):
        if index in excluded:
            continue
        if not call["complete"]:
            raise ValueError("saved admission included an incomplete generation")
        current = call["message"].get("tool_calls") or []
        history = [m for m in call["messages"] if m["role"] == "tool"]
        if current and current[0]["id"] in tools:
            episode = tools[current[0]["id"]]
        elif history:
            episode = histories[history[-1]["tool_call_id"]]
        else:
            candidates = prompts[next(m["content"] for m in call["messages"] if m["role"] == "user")]
            if len(candidates) != 1:
                raise ValueError("ambiguous admitted first-turn final action")
            episode = candidates[0]
        trace = episode["traces"][0]
        name = episode["task"]["data"]["name"]
        messages = [n["message"] for n in trace["nodes"] if n.get("message")]
        ids = {t["id"] for t in current}
        responses = [m for m in messages if m["role"] == "tool" and m.get("tool_call_id") in ids]
        if not current and history:
            responses = history[-1:]
        findings = native_findings(episode, guidance[name])
        rows.append({"task": name, "episode_id": episode["id"], "generation_index": index,
            "terminal_text": not bool(current), "native_success": trace.get("metrics", {}).get("task_completed_correctly"),
            "executed_tool_responses": len(responses), "observed_tool_error": bool(current) and observed_tool_error(responses),
            "observed_obligation_failure": findings["observed_obligation_failure"],
            "observed_action_risks": action_risks(call, responses) if current else []})
    if len(rows) != admission["admitted"]:
        raise ValueError("reconstructed saved admission differs")
    ordinary = batch(rows, tasks, step, mode="first-error-frontier")
    actual = read("selected-transitions.json")

    def identity(row):
        return row["task"], row["episode_id"], row["generation_index"]

    if [identity(x) for x in ordinary] != [identity(x) for x in actual]:
        raise ValueError("historical selection was not reproduced; reject counterfactual")
    proposed = batch(rows, tasks, step, mode="first-risk-frontier")
    return {"step": step, "admitted": len(rows), "saved_selection_exact": True,
            "changed_slots": [{"slot": i, "old": old, "proposed": new}
                              for i, (old, new) in enumerate(zip(ordinary, proposed, strict=True))
                              if identity(old) != identity(new)],
            "old_risk_actions": sum(bool(x["observed_action_risks"]) for x in ordinary),
            "proposed_risk_actions": sum(bool(x["observed_action_risks"]) for x in proposed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guidance = json.loads(args.guidance.read_text())["guidance"]
    tasks = json.loads(args.cohorts.read_text())["train"]
    results = [replay(args.root / f"step-{step:04d}/online_train", guidance, tasks, step)
               for step in range(1, 11)]
    args.output.write_text(json.dumps({"steps": results, "model_calls": 0, "optimizer_updates": 0,
        "scope": "Retrospective selection on the same retained trajectories/admission. "
                 "Risk is not causal correctness credit. Different training would change later "
                 "trajectories; this is not a counterfactual reward or gradient estimate."}, indent=2) + "\n")
