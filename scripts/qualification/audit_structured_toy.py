"""Audit a completed toy against raw native reward evidence and a Decimal oracle."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

from safetensors.torch import load_file
from verifiers.v1.trace import WireTrace


def normalize(values, epsilon):
    with localcontext() as context:
        context.prec = 60
        values = [Decimal(str(value)) for value in values]
        mean = sum(values, Decimal(0)) / len(values)
        variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / (len(values) - 1)
        return [float((value - mean) / (variance.sqrt() + Decimal(str(epsilon)))) for value in values]


def native_reward(record):
    # Independent native weighted-score oracle, including the retained legacy format.
    return math.fsum(
        value["score"] * value.get("weight", 1.0) if isinstance(value, dict) else value
        for value in record["rewards"].values()
    )


DIMENSION_NAMES = (
    "understanding_planning",
    "logical_correctness",
    "evidence_state_grounding",
    "verification_self_correction",
    "progress_efficiency",
)


def dimension_mean(dimensions):
    assert tuple(dimensions) == DIMENSION_NAMES
    values = [dimensions[name] for name in DIMENSION_NAMES]
    assert all(0 <= value <= 1 for value in values)
    return math.fsum(values) / len(values)


def mechanics(root, selection, records, identities, expected):
    """Reconstruct actual judge inputs and explain each admitted trajectory."""
    from transformers import AutoTokenizer

    artifact = selection["policy"]["artifact"]
    tokenizer = AutoTokenizer.from_pretrained(artifact["repo_id"], revision=artifact["revision"], local_files_only=True)
    if selection["judge"].get("assessment_scope") == "episode":
        return episode_mechanics(tokenizer, selection, records, identities, expected)
    spec = importlib.util.spec_from_file_location("qualification_saved_judge", root / "judge-source.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("saved judge source is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = []
    for identity in identities:
        record = records[identity]
        turns = []
        trace = WireTrace.model_validate(record)
        attempts = record["info"]["posttrain_turn_rewards_attempts"]
        valid_attempts = [attempt for attempt in attempts if attempt["status"] == "valid"]
        assert len(valid_attempts) == 1
        attempt = valid_attempts[0]
        request = attempt["assessment_request"]
        targets = request["target_turn_ids"]
        by_turn = {entry["turn_id"]: entry for entry in request["trajectory"] if "turn_id" in entry}
        assert request["context_scope"] == "retrospective"
        assert set(by_turn) == set(targets)
        ordinal = 0
        for node in trace.branches[0].nodes:
            if node.sampled and any(node.mask):
                turn_id = f"assistant-{ordinal}"
                ordinal += 1
                entry = by_turn[turn_id]
                ids = [token for token, selected in zip(node.token_ids, node.mask, strict=True) if selected]
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
                reason = entry.get("reasoning_content") or ""
                assert not reason or reason in decoded, "judge reasoning is not present in sampled text"
                content = entry.get("content") or ""
                if "<tool_call>" in decoded and not entry.get("tool_calls"):
                    assert "<tool_call>" in content, "rejected tool attempt was dropped before judging"
                    rejected = entry.get("provider_state", [])
                    assert rejected and rejected[0]["type"] == "posttrain.rejected_tool_call"
                    assert rejected[0]["status"] != "ok"
                turns.append(
                    {
                        "turn_id": turn_id,
                        "sampled_tokens": len(ids),
                        "reasoning_chars": len(reason),
                        "content_chars": len(content),
                        "tool_names": [call["name"] for call in entry.get("tool_calls", [])],
                        "nonexecuted_tool_syntax": "<tool_call>" in content,
                        "decoded_sampled_text": decoded,
                    }
                )
        panel = record["info"]["posttrain_turn_rewards"]
        assert panel["context_scope"] == "retrospective"
        prompt = selection["judge"]["rubric"] + "\nASSESSMENT REQUEST:\n" + json.dumps(request, ensure_ascii=False)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        assert selection["judge"]["input_budget_tokens"] >= selection["environment"]["parameters"]["max_total_tokens"]
        assert (
            selection["judge"]["input_budget_tokens"] + selection["judge"]["sampling"]["max_tokens"]
            <= selection["judge_context_window"]
        )
        assert attempts and all(attempt["input_digest"] == digest for attempt in attempts)
        raw = json.loads(next(attempt["raw_response"] for attempt in attempts if attempt["status"] == "valid"))
        by_turn = {turn["turn_id"]: turn for turn in raw["turns"]}
        for turn in turns:
            turn["judgment"] = by_turn[turn["turn_id"]]
        rows.append(
            {
                "trace_id": identity,
                "group": record["info"]["posttrain_prompt_group_id"],
                "task": record["info"]["automationbench"]["task_name"],
                "partial_credit": native_reward(record),
                "expected_advantage": expected[identity],
                "turns": turns,
                "judge_input_digest_verified": digest,
                "judge_usage": [item.get("usage") for item in record["info"].get("judge", [])],
            }
        )
    return rows


def episode_mechanics(tokenizer, selection, records, identities, expected):
    """Check exact stored prompts against native messages, not reserialized dict order."""
    from automationbench_v1.episode_prompt import (
        EPISODE_RUBRICS,
        EpisodeVerdict,
        build_episode_judge_messages,
        validate_episode_verdict,
    )
    from automationbench_v1.judge import project_tool_observation
    from automationbench_v1.limited_tools import selected_tool_definitions

    assert selection["episode_rubrics"] == EPISODE_RUBRICS
    assert selection["judge"]["input_budget_tokens"] >= selection["environment"]["parameters"]["max_total_tokens"]
    assert (
        selection["judge"]["input_budget_tokens"] + selection["judge"]["sampling"]["max_tokens"]
        <= selection["judge_context_window"]
    )
    rows = []
    for identity in identities:
        record = records[identity]
        trace = WireTrace.model_validate(record)
        trajectory, turns = [], []
        for index, node in enumerate(trace.branches[0].nodes):
            entry = node.message.model_dump(mode="json", exclude_none=True)
            entry["message_id"] = f"message-{index}"
            if entry["role"] == "tool" and isinstance(entry.get("content"), str):
                entry["content"] = project_tool_observation(entry["content"])
            trajectory.append(entry)
            if node.sampled and any(node.mask):
                assert node.message.role == "assistant"
                ids = [token for token, eligible in zip(node.token_ids, node.mask, strict=True) if eligible]
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
                reason = entry.get("reasoning_content") or ""
                assert not reason or reason in decoded
                if "<tool_call>" in decoded and not entry.get("tool_calls"):
                    assert "<tool_call>" in (entry.get("content") or ""), "rejected action missing from judge input"
                turns.append(
                    {"message_id": entry["message_id"], "sampled_tokens": len(ids), "reasoning_chars": len(reason)}
                )
        messages, request, input_digest = build_episode_judge_messages(
            trace_id=identity,
            trajectory=trajectory,
            available_tools=selected_tool_definitions(tuple(record["task"]["data"].get("zapier_tools", ()))),
            system_prompt=selection["judge"]["rubric"],
        )
        wire = [message.model_dump(mode="json", exclude_none=True) for message in messages]
        attempts = record["info"]["posttrain_episode_reward_attempts"]
        for attempt in attempts:
            assert attempt["messages"] == wire, "judge messages differ from native trajectory"
            assert attempt["assessment_request"] == request
            assert attempt["input_digest"] == input_digest
        [valid] = [attempt for attempt in attempts if attempt["status"] == "valid"]
        verdict = EpisodeVerdict.model_validate_json(valid["raw_response"])
        validate_episode_verdict(verdict, {entry["message_id"] for entry in trajectory})
        assessments = verdict.model_dump(mode="json")["assessments"]
        assert assessments == record["info"]["posttrain_episode_rewards"]["assessments"]
        rows.append(
            {
                "trace_id": identity,
                "partial_credit": native_reward(record),
                "expected_advantage": expected[identity],
                "assessments": assessments,
                "turns": turns,
                "judge_input_digest_verified": valid["input_digest"],
                "judge_usage": [item.get("usage") for item in record["info"].get("judge", [])],
            }
        )
    return rows


def audit(root, *, reload_export=False, include_mechanics=False):
    selection = json.loads((root / "selection.json").read_text())
    settings = selection["settings"]
    expected_steps = int(settings["loop"]["max_steps"])
    native_judge = "judge" in selection
    episode_judge = selection.get("judge", {}).get("assessment_scope") == "episode"
    scorer_digest = (
        selection["projection"]["scorer_digest"]
        if native_judge
        else hashlib.sha256(json.dumps(selection["scorer"], sort_keys=True).encode()).hexdigest()
    )
    algorithm = "gdpo" if "component_names" in settings else "capo"
    records = {row["id"]: row for row in map(json.loads, (root / "traces.jsonl").read_text().splitlines())}
    observations = list(map(json.loads, (root / "observations.jsonl").read_text().splitlines()))
    facts = {
        row["value"]["external_id"]: row["value"]
        for row in observations
        if row["kind"] == "trace_fact_update"
        and row["value"]["facts"]["namespace"] == "posttrain.train.structured_reward"
    }
    groups = defaultdict(list)
    masks, errors, quality = {}, {}, {}
    for identity in facts:
        record = records[identity]
        assert record["is_completed"] and not record.get("errors", []) and record.get("ok", True)
        trace = WireTrace.model_validate(record)
        assert len(trace.branches) == 1
        original_mask = tuple(trace.branches[0].sampled_mask)
        masks[identity] = original_mask[original_mask.index(True) :]
        info = record["info"]
        sampled_errors = [False] * len(masks[identity])
        if episode_judge:
            assert info["posttrain_scorer_digest"] == scorer_digest
            panel = info["posttrain_episode_rewards"]
            assert panel["scope"] == "episode" and panel["trace_id"] == identity
            assert panel["scorer_digest"] == scorer_digest
            assert set(panel["assessments"]) == set(settings["component_names"][1:])
            for name, rating in panel["assessments"].items():
                assert rating["status"] == "valid" and 0 <= rating["score"] <= 1
                assert rating["evidence"] and set(rating["evidence"]).issubset(
                    {f"message-{i}" for i in range(len(trace.branches[0].nodes))}
                )
                assert info[f"episode_reward/{name}"] == rating["score"]
        elif native_judge:
            assert info["posttrain_scorer_digests"]["posttrain_turn_rewards"] == scorer_digest
            panel = info["posttrain_turn_rewards"]
            assert panel["scorer_digest"] == scorer_digest and panel["trace_id"] == identity
            turns = {}
            offset, ordinal = 0, 0
            first = original_mask.index(True)
            for node in trace.branches[0].nodes:
                positions = [offset + index - first for index, eligible in enumerate(node.mask) if eligible]
                if positions:
                    assert node.sampled and node.message.role == "assistant"
                    turns[f"assistant-{ordinal}"] = positions
                    ordinal += 1
                offset += len(node.token_ids)
            assert set(panel["erroneous_turn_ids"]).issubset(turns)
            for turn_id in panel["erroneous_turn_ids"]:
                for position in turns[turn_id]:
                    assert masks[identity][position]
                    sampled_errors[position] = True
            assessments = {item["turn_id"]: item for item in panel["assessments"]}
            assert len(assessments) == len(panel["assessments"]) and set(assessments) == set(turns)
            scores = []
            for turn_id, assessment in assessments.items():
                [component] = assessment["components"]
                assert component["name"] == "quality" and component["status"] == "valid"
                attempt = info["posttrain_turn_rewards_attempts"][int(assessment["evidence_ref"].rsplit("/", 1)[1])]
                assert attempt["status"] == "valid"
                raw = json.loads(attempt["raw_response"])
                [rated] = [item for item in raw["turns"] if item["turn_id"] == turn_id]
                dimensions = rated["dimensions"]
                assert assessment["rubric_dimensions"] == dimensions
                assert math.isclose(dimension_mean(dimensions), component["value"], abs_tol=1e-12, rel_tol=1e-12)
                assert rated["erroneous"] == (turn_id in panel["erroneous_turn_ids"])
                assert 0 <= component["value"] <= 1
                scores.append(component["value"])
            quality[identity] = math.fsum(scores) / len(scores)
        else:
            assert info["posttrain_scorer_digest"] == scorer_digest
            credit = info["posttrain_process_credit"]
            assert credit["status"] == "valid"
            for start, end in credit["error_spans"]:
                assert 0 <= start < end <= len(sampled_errors)
                assert all(masks[identity][start:end])
                sampled_errors[start:end] = [True] * (end - start)
            assert info["posttrain_judge_panel"]["scorer"] == selection["scorer"]
            quality[identity] = info["posttrain_quality"]
        errors[identity] = sampled_errors
        groups[info["posttrain_prompt_group_id"]].append(identity)
    means = {}
    for identities in groups.values():
        assert len(identities) == settings["num_generations"]
        if algorithm == "gdpo":
            columns = [[native_reward(records[i]) for i in identities]]
            if episode_judge:
                columns.extend(
                    [records[i]["info"][f"episode_reward/{name}"] for i in identities]
                    for name in settings["component_names"][1:]
                )
            else:
                columns.append([quality[i] for i in identities])
            normalized = [normalize(values, settings["epsilon"]) for values in columns]
            combined = [
                sum(
                    weight * values[row]
                    for weight, values in zip(settings["component_weights"], normalized, strict=True)
                )
                for row in range(len(identities))
            ]
            expected = normalize(combined, settings["epsilon"])
            means.update(zip(identities, expected, strict=True))
        else:
            raw, sizes = [], []
            for identity in identities:
                outcome = records[identity]["metrics"]["task_completed_correctly"]
                assert outcome in (0.0, 1.0)
                row = [
                    settings["outcome_weight"] * outcome - settings["process_weight"] * erroneous
                    for erroneous, sampled in zip(errors[identity], masks[identity], strict=True)
                    if sampled
                ]
                sizes.append(len(row))
                raw.extend(row)
            expected = normalize(raw, settings["epsilon"])
            offset = 0
            for identity, size in zip(identities, sizes, strict=True):
                means[identity] = math.fsum(expected[offset : offset + size]) / size
                offset += size
    for identity, expected in means.items():
        actual = facts[identity]["facts"]["measures"]["sampled_advantage_mean"]
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (identity, actual, expected)
    values = [row["value"] for row in observations if row["kind"] in {"metric", "metrics"}]
    gradients = []
    for value in values:
        if value.get("name") == "train/grad_norm":
            gradients.append(value["value"])
        elif "train/grad_norm" in value.get("values", {}):
            gradients.append(value["values"]["train/grad_norm"])
    result = json.loads((root / "result.json").read_text())
    assert result["summary"]["global_step"] == expected_steps
    adapters = list((root / "training" / algorithm / "adapter").glob("adapter_model.safetensors"))
    assert len(adapters) == 1
    tensors = load_file(adapters[0])
    b_norm = sum(float(value.float().square().sum()) for name, value in tensors.items() if "lora_B" in name) ** 0.5
    assert len(gradients) == expected_steps and all(math.isfinite(value) for value in gradients)
    expected_groups = expected_steps * settings["num_prompts_per_step"]
    assert len(groups) == expected_groups
    assert len(facts) == expected_groups * settings["num_generations"]
    assert b_norm > 0 and any(value > 0 for value in gradients), "run did not exercise a learning signal"
    report = {
        "algorithm": algorithm,
        "optimizer_steps": expected_steps,
        "native_traces": len(records),
        "admitted_traces": len(facts),
        "nonadmitted_traces": [
            {"trace_id": identity, "stop_condition": record["stop_condition"], "errors": record.get("errors", [])}
            for identity, record in records.items()
            if identity not in facts
        ],
        "prompt_groups": len(groups),
        "partial_credit": [native_reward(records[i]) for i in facts],
        "binary_success": [records[i]["metrics"]["task_completed_correctly"] for i in facts],
        "judge_quality": None if episode_judge else [quality[i] for i in facts],
        "sampled_tokens": [sum(masks[i]) for i in facts],
        "process_error_tokens": [sum(errors[i]) for i in facts],
        "grad_norms": gradients,
        "lora_B_norm": b_norm,
        "decimal_advantage_mean_parity": True,
        "native_token_mask_alignment": True,
        "learning_signal_exercised": b_norm > 0 and any(value > 0 for value in gradients),
        "scope": "local TRL toy; not veRL, benchmark improvement, or expert validation of judge judgments",
    }
    if episode_judge:
        report["episode_components"] = {
            name: [records[i]["info"][f"episode_reward/{name}"] for i in facts]
            for name in settings["component_names"][1:]
        }
        report["component_weights"] = dict(zip(settings["component_names"], settings["component_weights"], strict=True))
        report["quality_reduced_from_dimensions"] = False
        report["groups_with_component_variation"] = {
            name: sum(
                len({records[i]["info"][f"episode_reward/{name}"] for i in group}) > 1 for group in groups.values()
            )
            for name in settings["component_names"][1:]
        }
    elif native_judge:
        dimension_rows = [
            assessment["rubric_dimensions"]
            for identity in facts
            for assessment in records[identity]["info"]["posttrain_turn_rewards"]["assessments"]
        ]
        report["judge_dimensions"] = {name: [row[name] for row in dimension_rows] for name in DIMENSION_NAMES}
        report["quality_reduced_from_dimensions"] = True
        report["groups_with_quality_variation"] = sum(
            len({quality[identity] for identity in identities}) > 1 for identities in groups.values()
        )
    if native_judge:
        from verifiers.v1.episode import WireEpisode  # pyright: ignore[reportAttributeAccessIssue]

        native_traces = {}
        for line in (root / "episodes.jsonl").read_text().splitlines():
            episode = WireEpisode.model_validate_json(line)
            for trace in episode.traces:
                native_traces[trace.id] = trace
        for identity in facts:
            original = native_traces[identity].branches[0]
            derived = WireTrace.model_validate(records[identity]).branches[0]
            assert original.token_ids == derived.token_ids
            assert original.sampled_mask == derived.sampled_mask
            assert original.logprobs == derived.logprobs
        report["native_episode_provenance"] = True
        report["scope"] = "installed TRL and managed native judge; not OCI, benchmark improvement, or judge calibration"
    if reload_export:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        artifact = selection["policy"]["artifact"]
        tokenizer = AutoTokenizer.from_pretrained(adapters[0].parent, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            artifact["repo_id"],
            revision=artifact["revision"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, adapters[0].parent, local_files_only=True).to("cuda").eval()
        template_kwargs = (
            {"enable_thinking": False}
            if selection["policy"]["family"] == "qwen3.5"
            else {"preserve_thinking": False}
        )
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with one word: ready."}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        completion = generated[0, inputs["input_ids"].shape[-1] :].tolist()
        assert completion
        report["export_reload_generation"] = {"token_ids": completion, "text": tokenizer.decode(completion)}
    if include_mechanics:
        mechanics_rows = mechanics(root, selection, records, facts, means)
        (root / "mechanics.json").write_text(json.dumps(mechanics_rows, indent=2) + "\n")
        report["mechanics"] = {
            "file": "mechanics.json",
            "trajectories": len(mechanics_rows),
            "judge_input_digests_verified": True,
        }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reload-export", action="store_true", help="Also load the saved LoRA adapter on an idle GPU")
    parser.add_argument(
        "--mechanics", action="store_true", help="Verify exact judge inputs and retained reasoning/tool evidence"
    )
    args = parser.parse_args()
    audit(args.output, reload_export=args.reload_export, include_mechanics=args.mechanics)
