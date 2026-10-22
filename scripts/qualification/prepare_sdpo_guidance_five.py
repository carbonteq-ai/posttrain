"""Generate source-separated teacher guidance with only the pinned local4B."""

import argparse
import asyncio
import json
from pathlib import Path


class BoundedRequests:
    """Bound preparation requests across every task and independent check."""

    def __init__(self, maximum=40):
        if not 1 <= maximum <= 40:
            raise ValueError("preparation concurrency must be between1and40")
        self.maximum = maximum
        self.semaphore = asyncio.Semaphore(maximum)
        self.active = self.peak = self.submitted = 0

    async def call(self, operation):
        async with self.semaphore:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.submitted += 1
            try:
                return await operation()
            finally:
                self.active -= 1

    def evidence(self):
        return {"limit": self.maximum, "peak_in_flight": self.peak,
                "submitted": self.submitted, "in_flight_at_snapshot": self.active,
                "scope": "Model request concurrency; not a GPU throughput or generation-quality estimate."}


def parallel_checks_enabled(args):
    enabled = getattr(args, "parallel_checks", False)
    if enabled and args.recipe not in ("checkwise", "private-checkwise"):
        raise ValueError("independent-check parallelism requires a checkwise recipe")
    return enabled


def preparation_prompt(tokenizer, system, data):
    from sdpo_prompting import prompt_ids

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]
    prefix = prompt_ids(tokenizer, messages, [], thinking=False)
    if not prefix or not all(type(token) is int for token in prefix):
        raise ValueError("preparation requires actual original integer token IDs")
    if len(prefix) + 4096 > 32768:
        raise ValueError("preparation exceeds full context; no clipping")
    return messages, prefix


def preparation_schema(stage, data):
    if stage.startswith(("check_", "workflow_brief", "workflow_review", "bundle_review")):
        from sdpo_checkwise_preparation import schema

        return schema(stage, data)
    from luna_solver_schema import instruction_schema

    schema = instruction_schema(stage, data)
    if stage.removesuffix("_repair") == "assembly":
        fields = schema["properties"]["steps"]["items"]["properties"]
        fields["step_id"] = {"type": "string", "pattern": "^s[1-9][0-9]*$"}
        fields["depends_on"]["items"] = {"type": "string", "pattern": "^s[1-9][0-9]*$"}
    return schema


async def cpu_preflight(args):
    from luna_solver_stages import prepare_instructions
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = json.loads(args.materials.read_text())["rows"]
    captured = []
    parallel = parallel_checks_enabled(args)

    class Captured(Exception):
        pass

    for row in rows:
        async def capture(stage, system, data, task_name=row["task"]):
            _, prefix = preparation_prompt(tokenizer, system, data)
            schema = preparation_schema(stage, data)
            SamplingParams(temperature=0., max_tokens=4096,
                           structured_outputs=StructuredOutputsParams(json=schema))
            captured.append({"task": task_name, "stage": stage,
                             "prefix_tokens": len(prefix), "integer_ids": True,
                             "check_id": data.get("check", {}).get("check_id"),
                             "historical_reference_scope": data.get("historical_reference_scope")})
            raise Captured()

        try:
            if args.recipe in ("checkwise", "private-checkwise"):
                from sdpo_checkwise_preparation import prepare

                await prepare(row["material"], capture, lambda *_: None,
                              private_preparation=args.recipe == "private-checkwise",
                              historical_binding=row.get("binding"), parallel_checks=parallel,
                              abstract_private_briefs=args.recipe == "private-checkwise")
            else:
                await prepare_instructions(row["material"], capture, lambda *_: None,
                    "scope-repair-natural-policy-provenance-v5", schema_argument_roles=True,
                    repair_invalid_assembly=True)
        except Captured:
            pass
    expected = ({(row["task"], check["check_id"]) for row in rows
                 for check in row["material"]["goal_manifest"]["checks"]
                 if check["role"] in ("goal", "harm")} if parallel else None)
    if (parallel and ({(c["task"], c["check_id"]) for c in captured} != expected
                      or len(captured) != len(expected))
            or not parallel and len(captured) != len(rows)):
        actual = {(c["task"], c["check_id"]) for c in captured}
        raise ValueError(f"preflight task coverage mismatch: captured={len(captured)}, "
                         f"expected={len(expected) if expected is not None else len(rows)}, "
                         f"missing={sorted(expected - actual) if expected is not None else []}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"qualified": True, "tasks": captured,
        "parallel_checks": parallel,
        "optimizer_updates": 0, "model_calls": 0, "scope": "actual pinned tokenizer and first-stage schemas; not generation quality"}, indent=2))


def decoding_config(args):
    from sdpo_speculation import speculative_config

    method = getattr(args, "speculation", "mtp")
    if method == "none" and not getattr(args, "plain_decoding_authorized", False):
        raise ValueError("plain preparation requires the explicit user amendment")
    return speculative_config(method, args.mtp_tokens)


async def run(args):
    import torch
    from generate_luna_ssr_pilot import parsed_json
    from luna_solver_stages import prepare_instructions
    from peft import LoraConfig, get_peft_model
    from sdpo_guidance_context import digest
    from self_feedback_vllm import LocalVLLMRollout
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    parallel = parallel_checks_enabled(args)
    native_cohorts = getattr(args, "native_cohorts", None)
    args.output.mkdir(parents=True, exist_ok=False)
    speculation = decoding_config(args)
    (args.output / "decoding-request.json").write_text(json.dumps({
        "speculative_config": speculation, "concurrency": 40,
        "context": 65536 if native_cohorts is not None else 32768,
        "preparation_context_limit": 32768,
        "prefix_caching": not getattr(args, "disable_prefix_caching", False),
        "chunked_prefill": not getattr(args, "disable_chunked_prefill", False),
        "enforce_eager": getattr(args, "enforce_eager", False),
        "parallel_checks": parallel, "preparation_request_concurrency": 40,
        "optimizer_updates": 0}, indent=2))
    rows = json.loads(args.materials.read_text())["rows"]
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    engine = LocalVLLMRollout
    if native_cohorts is not None:
        from sdpo_agent_ablation import SamplingRollout

        engine = SamplingRollout
    rollout = await engine.create(args.model, model, tokenizer, args.output,
        65536 if native_cohorts is not None else 32768, 40, 16, speculative_config=speculation,
        prefix_caching=not getattr(args, "disable_prefix_caching", False),
        chunked_prefill=not getattr(args, "disable_chunked_prefill", False),
        enforce_eager=getattr(args, "enforce_eager", False))
    guidance, rejected = {}, {}
    requests = BoundedRequests(40)
    if args.accepted_guidance is not None:
        from sdpo_guidance_context import teacher_messages

        parent = json.loads(args.accepted_guidance.read_text())
        by_name = {r["task"]: r for r in rows}
        if not set(parent["guidance"]) <= set(by_name):
            raise ValueError("accepted guidance contains tasks outside the frozen population")
        for name, bundle in parent["guidance"].items():
            source = by_name[name]["material"]
            expected_checks = [c for c in source["goal_manifest"]["checks"] if c["role"] in ("goal", "harm")]
            if (bundle.get("preparation_recipe") != args.recipe
                    or bundle["model_revision"] != Path(args.model).name
                    or bundle["manifest_sha256"] != digest(source["goal_manifest"])
                    or bundle["check_roles"] != {c["check_id"]: c["role"] for c in expected_checks}
                    or bundle["check_signals"] != {c["check_id"]: c["signal_id"] for c in expected_checks}):
                raise ValueError("accepted guidance recipe/model/manifest binding differs")
            teacher_messages({"messages": []}, source["native_data"], bundle, [], [])
            guidance[name] = bundle
        (args.output / "reused-accepted-guidance.json").write_text(json.dumps({
            "source": str(args.accepted_guidance), "sha256": digest(parent),
            "tasks": sorted(guidance), "scope": "Previously individually approved same4B/checkwise contexts; rejected candidates never reused"}, indent=2))

    def append(path, value):
        with path.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    async def prepare(row):
        material, stages = row["material"], []

        def record(stage, value):
            stages.append({"stage": stage, **value})
            append(args.output / "stages.jsonl", {"task": row["task"], "stage": stage, **value})

        async def generate(stage, system, data):
            messages, prefix = preparation_prompt(tokenizer, system, data)
            params = SamplingParams(temperature=0., max_tokens=4096,
                structured_outputs=StructuredOutputsParams(json=preparation_schema(stage, data)))
            result = await requests.call(lambda: rollout._generate(prefix, params))
            completion = result.outputs[0]
            text = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
            append(args.output / "generations.jsonl", {"task": row["task"], "stage": stage,
                "messages": messages, "prefix_ids": prefix, "action_ids": list(completion.token_ids),
                "text": text, "finish_reason": completion.finish_reason,
                "generator": "Qwen/Qwen3.5-4B", "policy_version": rollout.version})
            if completion.finish_reason != "stop":
                raise ValueError("incomplete preparation; no truncation or automatic retry")
            return parsed_json(text)

        try:
            if row["task"] in guidance:
                from sdpo_checkwise_preparation import (
                    GROUNDED_REVIEW,
                    grounded_review_payload,
                    preparation_rules,
                )

                bundle = guidance[row["task"]]
                payload = grounded_review_payload(material, bundle)
                review = await generate("bundle_review", preparation_rules(material) + "\n" + GROUNDED_REVIEW, payload)
                record("reused_bundle_source_review", {"input": {"guidance_sha256": digest(bundle),
                    "private_source_provenance": payload["private_current_source_provenance"]}, "output": review})
                if review.get("approved") is not True or review.get("issues") != []:
                    del guidance[row["task"]]
                    raise ValueError("reused guide failed current grounded source review")
                return
            if args.recipe in ("checkwise", "private-checkwise"):
                from sdpo_checkwise_preparation import prepare

                result = await prepare(material, generate, record,
                                       private_preparation=args.recipe == "private-checkwise",
                                       historical_binding=row.get("binding"), parallel_checks=parallel,
                                       abstract_private_briefs=args.recipe == "private-checkwise")
                lines = result["instructions"]
                approved = result["review_approved"]
                requirements = {"requirements": result["requirements"]}
            else:
                lines = await prepare_instructions(material, generate, record,
                    "scope-repair-natural-policy-provenance-v5", schema_argument_roles=True,
                    repair_invalid_assembly=True)
                reviews = [r["output"] for r in stages if r["stage"] in ("review", "review_repair")]
                approved = bool(reviews and reviews[-1].get("approved"))
                requirements = [r["output"] for r in stages if r["stage"] in ("requirements", "requirements_repair")][-1]
            expected = [c["check_id"] for c in material["goal_manifest"]["checks"] if c["role"] in ("goal", "harm")]
            actual = [r["check_id"] for r in requirements["requirements"]]
            if (not approved or set(actual) != set(expected)
                    or len(actual) != len(expected)):
                raise ValueError("missing approval or exact requirement coverage; no diagnostic-only admission")
            guidance[row["task"]] = {"task": row["task"], "instructions": lines,
                "user_request_sha256": digest(material["user_request"]),
                "check_ids": actual, "current_state_sha256": digest(material["native_data"]["initial_state"]),
                "check_roles": {c["check_id"]: c["role"] for c in material["goal_manifest"]["checks"]
                                if c["role"] in ("goal", "harm")},
                "check_signals": {c["check_id"]: c["signal_id"] for c in material["goal_manifest"]["checks"]
                                  if c["role"] in ("goal", "harm")},
                "manifest_sha256": digest(material["goal_manifest"]), "review_approved": True,
                "prepared_by_model": "Qwen/Qwen3.5-4B", "model_revision": Path(args.model).name,
                "preparation_recipe": args.recipe,
                "source": row["luna_source"], "historical_binding": row["binding"],
                "preparation_reference_scope": (result["historical_reference_scope"]
                    if args.recipe in ("checkwise", "private-checkwise") else None),
                "stages_sha256": digest(stages)}
        except (ValueError, KeyError, TypeError, IndexError) as error:
            guidance.pop(row["task"], None)
            rejected[row["task"]] = {"type": type(error).__name__, "message": str(error)}
        (args.output / "guidance.json").write_text(json.dumps({"guidance": guidance,
            "rejected": rejected, "expected": [r["task"] for r in rows],
            "optimizer_updates": 0, "external_calls": 0}, indent=2) + "\n")

    try:
        await asyncio.gather(*(prepare(row) for row in rows))
        (args.output / "guidance.json").write_text(json.dumps({"guidance": guidance,
            "rejected": rejected, "expected": [r["task"] for r in rows],
            "optimizer_updates": 0, "external_calls": 0}, indent=2) + "\n")
        if rejected or len(guidance) != len(rows):
            raise ValueError("incomplete guidance population; inspect rejected stages before training")
        task_qualification = None
        if native_cohorts is not None:
            from qualify_sdpo_prepared_tasks import qualify, reuse_native_reference

            task_qualification = (reuse_native_reference(args, rollout) if args.native_reference is not None
                                  else await qualify(args, rollout, model, tokenizer))
        (args.output / "receipt.json").write_text(json.dumps({"qualified": True,
            "tasks": len(guidance), "optimizer_updates": 0, "external_calls": 0,
            "parallel_checks": parallel, "request_concurrency": requests.evidence(),
            "native_task_qualification": task_qualification,
            "guidance_sha256": digest(guidance)}, indent=2))
    finally:
        (args.output / "request-concurrency.json").write_text(json.dumps(requests.evidence(), indent=2))
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp-tokens", type=int, choices=(1, 2), required=True)
    parser.add_argument("--speculation", choices=("none", "mtp"), default="mtp")
    parser.add_argument("--plain-decoding-authorized", action="store_true")
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--recipe", choices=("staged", "checkwise", "private-checkwise"), default="staged")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--accepted-guidance", type=Path)
    parser.add_argument("--parallel-checks", action="store_true")
    parser.add_argument("--native-cohorts", type=Path)
    parser.add_argument("--native-reference", type=Path)
    args = parser.parse_args()
    asyncio.run(cpu_preflight(args) if args.cpu_preflight else run(args))
