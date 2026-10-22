"""Read-only tool-span gradient/Adam audit of a completed paired qualifier."""

import argparse
import json
from pathlib import Path


def tool_positions(ids, opening, closing):
    positions, active = [], False
    for index, token in enumerate(ids):
        if token == opening:
            if active:
                raise ValueError("nested tool-call boundary")
            active = True
        if active:
            positions.append(index)
        if token == closing:
            if not active:
                raise ValueError("orphan closing tool boundary")
            active = False
    if active:
        raise ValueError("incomplete tool-call boundary")
    return positions


def forbidden_positions(ids, tokenizer, checks, allowed):
    """Exact native-token matches to failed, non-excluded forbidden-text checks.

    Diagnostic labels only. No target is added to the training loss. Missing
    subsequence matches remain missing rather than asserting the text is safe.
    """
    terms = sorted({a["params"]["text_contains"] for a in checks
                    if a.get("type") == "slack_message_not_in_channel" and a.get("passed") is False
                    and not a.get("excluded") and isinstance(a.get("params", {}).get("text_contains"), str)})
    positions, matched = set(), []
    allowed = set(allowed)
    for term in terms:
        found = False
        for prefix in ["", " ", "\n"]:
            needle = tokenizer.encode(prefix + term, add_special_tokens=False)
            for start in range(len(ids) - len(needle) + 1):
                covered = set(range(start, start + len(needle)))
                if ids[start:start + len(needle)] == needle and covered <= allowed:
                    positions.update(covered)
                    found = True
        if found:
            matched.append(term)
    return sorted(positions), matched


def run(args):
    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.root / "receipt.json").read_text())
    if not receipt["runtime_completed"] or receipt["total_diagnostic_updates"] != 2:
        raise ValueError("requires completed paired one-update diagnostic")
    cases = json.loads(args.cases.read_text())["cases"]
    proposals = json.loads((args.root / "same-decision-proposals.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True),
        args.root / "initial-student", is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.eval()
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError("gradient probe requires zero dropout for comparable train/eval scoring")
    params = [(name, p) for name, p in model.named_parameters() if ".default." in name and p.requires_grad]
    initial = load_file(args.root / "initial-student/adapter_model.safetensors")
    opening, closing = [tokenizer.convert_tokens_to_ids(t) for t in ["<tool_call>", "</tool_call>"]]
    results = []

    def selected_logits(prefix, target, positions):
        # Keep the ENTIRE context/action input, emit only measured tool logits.
        # This is selective diagnostic support, never sequence truncation.
        ids = torch.tensor([prefix + target[:-1]], device="cuda")
        support = torch.tensor([len(prefix) - 1 + i for i in positions], device="cuda")
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                     logits_to_keep=support).logits[0].float()

    for arm in ["original_teacher", "combined_teacher"]:
        optimizer = torch.load(args.root / arm / "optimizer.pt", map_location="cpu", weights_only=True)
        identifiers = optimizer["param_groups"][0]["params"]
        beta1 = optimizer["param_groups"][0]["betas"][0]
        if len(optimizer["param_groups"]) != 1 or len(identifiers) != len(params):
            raise ValueError("optimizer/default-adapter parameter order mismatch")
        for pid, (_, parameter) in zip(identifiers, params, strict=True):
            state = optimizer["state"][pid]
            if int(state["step"].item()) != 1 or state["exp_avg"].shape != parameter.shape:
                raise ValueError("optimizer moment shape/step mismatch")
        for case in cases:
            row = case["row"]
            proposal = next(p for p in proposals if p["task"] == row["task"] and p["mode"] == "combined_teacher")
            record = {"arm": arm, "task": row["task"], "episode_id": row["episode_id"],
                      "proposal_parse_error": proposal.get("native_error"), "proposal_complete": proposal["complete"]}
            if not proposal["complete"]:
                record["measurable"] = False
                results.append(record)
                continue
            target = proposal["action_ids"]
            selected = tool_positions(target, opening, closing)
            if not selected:
                record.update(measurable=False, reason="combined teacher produced no tool call")
                results.append(record)
                continue
            checks = case["episode"]["traces"][0].get("info", {}).get("automationbench", {}).get("assertions", [])
            forbidden, forbidden_terms = forbidden_positions(target, tokenizer, checks, selected)
            local_forbidden = [selected.index(i) for i in forbidden]
            set_peft_model_state_dict(model, initial, adapter_name="default")
            model.eval()
            tokens = torch.tensor(target, device="cuda")[torch.tensor(selected, device="cuda")]
            with torch.no_grad():
                before_values = selected_logits(row["student_prefix_ids"], target, selected)
                before = float(before_values.log_softmax(-1).gather(-1, tokens[:, None]).mean())
                if forbidden:
                    bad_pos = torch.tensor(local_forbidden, device="cuda")
                    bad_tokens = tokens[bad_pos]
                    forbidden_before = float(before_values[bad_pos].log_softmax(-1).gather(-1, bad_tokens[:, None]).mean())
            del before_values
            model.train()  # enable activation checkpointing; dropout is explicitly zero
            values = selected_logits(row["student_prefix_ids"], target, selected)
            mean = values.log_softmax(-1).gather(-1, tokens[:, None]).mean()
            reference = torch.autograd.grad(mean, [p for _, p in params], allow_unused=True, retain_graph=bool(forbidden))
            projection = 0.
            for pid, gradient in zip(identifiers, reference, strict=True):
                if gradient is not None:
                    accumulated = optimizer["state"][pid]["exp_avg"].to("cuda").float() / (1 - beta1)
                    projection -= float((accumulated * gradient.float()).sum())
            if forbidden:
                bad_mean = values[bad_pos].log_softmax(-1).gather(-1, bad_tokens[:, None]).mean()
                bad_reference = torch.autograd.grad(bad_mean, [p for _, p in params], allow_unused=True)
                forbidden_projection = 0.
                for pid, gradient in zip(identifiers, bad_reference, strict=True):
                    if gradient is not None:
                        accumulated = optimizer["state"][pid]["exp_avg"].to("cuda").float() / (1 - beta1)
                        forbidden_projection -= float((accumulated * gradient.float()).sum())
                del bad_mean, bad_reference
            del mean, values, reference
            set_peft_model_state_dict(model, load_file(args.root / arm / "student/adapter_model.safetensors"), adapter_name="default")
            model.eval()
            with torch.no_grad():
                values = selected_logits(row["student_prefix_ids"], target, selected)
                after = float(values.log_softmax(-1).gather(-1, tokens[:, None]).mean())
                if forbidden:
                    forbidden_after = float(values[bad_pos].log_softmax(-1).gather(-1, bad_tokens[:, None]).mean())
            del values
            record.update(measurable=True, proposal_tool_tokens=len(selected),
                          full_batch_gradient_direction_toward_proposal_tool_logprob=projection,
                          tool_mean_logprob_before=before, tool_mean_logprob_after=after,
                          actual_adam_tool_mean_logprob_delta=after - before,
                          correctness="Requires source/native review; tool-span likelihood is not proof of correct action",
                          gradient_source="Moment-derived clipped-gradient approximation: first-step Adam exp_avg/(1-beta1), with stored-moment rounding; positive global clip factor preserves exact-gradient sign",
                          scoring_modes="before/after eval; gradient probe train with zero dropout and activation checkpointing")
            record["matched_forbidden_text_terms"] = forbidden_terms
            if forbidden:
                record["forbidden_text_diagnostic"] = {
                    "tokens": len(forbidden), "gradient_direction": forbidden_projection,
                    "actual_adam_logprob_delta": forbidden_after - forbidden_before,
                    "desired_direction": "negative: these strings violate decided non-excluded native checks in this task's outgoing messages",
                    "scope": "conditional token probability within the generated proposal; not probability of whole-task violation"}
            results.append(record)
            (args.output / "tool-gradient-directions.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({"arm": arm, "task": row["task"], "tool_delta": after - before,
                              "gradient_direction": projection}), flush=True)
    (args.output / "receipt.json").write_text(json.dumps({"completed": True, "optimizer_updates": 0,
        "results": results, "scope": "Tool-call tokens only; excludes generated reasoning. Read-only saved-adapter/moment audit."}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
