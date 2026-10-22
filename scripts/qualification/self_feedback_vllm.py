"""Bounded SELF rollout qualification using the selected CarbonTeq vLLM fork."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from trl.generation.async_vllm_session import AsyncGenerationRequest, AsyncVllmSession
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM


class SelfQualificationWorkerExtension:
    """Named diagnostic method; uses the normal safe worker RPC serializer."""

    def self_qualification_diagnostics(self):
        import faulthandler
        import os

        from sdpo_runtime_diagnostics import trace_interval
        from vllm.compilation.counter import compilation_counter

        interval = trace_interval(os.environ)
        faulthandler.enable(all_threads=True)
        if interval:
            faulthandler.dump_traceback_later(interval, repeat=True)
        else:
            faulthandler.cancel_dump_traceback_later()
        return {"captured_graphs": compilation_counter.num_cudagraph_captured,
                "trace_interval_seconds": interval, "fatal_tracebacks_enabled": True}


class LocalVLLMRollout:
    """Single-owner rollout engine; no collection may overlap adapter refresh."""

    engine: Any
    session: AsyncVllmSession

    def __init__(self, tokenizer, output: Path):
        self.tokenizer, self.output = tokenizer, output
        self.version = -1
        self.request = None
        self.refreshes: list[dict[str, Any]] = []
        self.request_counter = 0
        self.in_flight = self.peak_in_flight = 0

    @classmethod
    async def create(cls, model_path, model, tokenizer, output, context_limit=8192, server_sequences=4,
                     kv_cache_gib=2, *, speculative_config=None, prefix_caching=True, chunked_prefill=True,
                     enforce_eager=False):
        self = cls(tokenizer, output)
        from posttrain.common.cuda import activate_cuda_toolkit

        activate_cuda_toolkit(torch)
        if importlib.metadata.version("vllm") != "0.29.1.dev4":
            raise ValueError("qualification requires selected vLLM 0.29.1.dev4")
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.35,
            max_model_len=context_limit,
            max_num_seqs=server_sequences,
            max_num_batched_tokens=context_limit,
            kv_cache_memory_bytes=kv_cache_gib * 1024**3,
            # The default hybrid page is 528 tokens on this model. Composing
            # that boundary with invariant GDN's 128-token split can return
            # zero forever. Request 128 alignment; state-page sizing rounds
            # the resolved hybrid block to 640, preserving both constraints.
            block_size=128,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=16,
            enable_prefix_caching=prefix_caching,
            enable_chunked_prefill=chunked_prefill,
            enforce_eager=enforce_eager,
            language_model_only=True,
            seed=17,
            generation_config="vllm",
            override_generation_config={"eos_token_id": tokenizer.eos_token_id},
            disable_log_stats=False,
            enable_sleep_mode=True,
            worker_extension_cls="self_feedback_vllm.SelfQualificationWorkerExtension",
            speculative_config=speculative_config,
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)
        (output / "engine-config.txt").write_text(repr(self.engine.vllm_config))
        if self.engine.vllm_config.model_config.enforce_eager != enforce_eager:
            raise ValueError("resolved eager-execution setting differs from the requested qualification")
        if self.engine.vllm_config.scheduler_config.enable_chunked_prefill != chunked_prefill:
            raise ValueError("resolved chunked-prefill setting differs from the requested qualification")
        if self.engine.vllm_config.cache_config.block_size % 128:
            raise ValueError("resolved hybrid cache blocks must align with invariant GDN prefill")
        graph_status = await self.engine.collective_rpc("self_qualification_diagnostics")
        (output / "graph-captures.json").write_text(json.dumps(graph_status, indent=2))

        async def synchronize(_version):
            await self._load_student(model)

        self.session = AsyncVllmSession(self.engine, synchronize, sleep_level=1)
        await self.refresh()
        smoke = await self._generate(
            tokenizer.encode("Hello", add_special_tokens=False),
            SamplingParams(
                temperature=0.0,
                max_tokens=4,
            ),
        )
        (output / "short-generation.json").write_text(json.dumps({"tokens": list(smoke.outputs[0].token_ids)}))
        return self

    async def refresh(self):
        version = str(self.version + 1)
        await self.session.synchronize_policy(version)
        await self.session.open_policy(version)

    async def suspend_for_update(self):
        await self.session.suspend_for_update()

    async def aclose(self):
        await self.session.aclose()

    async def _load_student(self, model):
        version = self.version + 1
        directory = self.output / f"rollout-policy-v{version}"
        directory.mkdir()
        if model.active_adapter != "default":
            raise ValueError("rollout export requires the active student adapter")
        model.save_pretrained(directory, selected_adapters=["default"], safe_serialization=True)
        weights = directory / "adapter_model.safetensors"
        if not weights.is_file() or (directory / "self_teacher").exists():
            raise ValueError("rollout export must contain only student adapter weights")
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if self.refreshes and digest == self.refreshes[-1]["weights_sha256"]:
            raise ValueError("applied SELF updates did not change the exported adapter")
        if self.request is not None:
            if not await self.engine.remove_lora(self.request.lora_int_id):
                raise ValueError("engine did not remove the previous policy adapter")
        request = LoRARequest(f"self-policy-v{version}", version + 1, str(directory))
        if not await self.engine.add_lora(request):
            raise ValueError("engine did not acknowledge the student adapter")
        if not await self.engine.reset_prefix_cache():
            raise ValueError("engine did not acknowledge prefix-cache invalidation")
        self.request, self.version = request, version
        self.refreshes.append({"policy_version": version, "weights_sha256": digest, "path": str(directory)})

    async def _generate(self, prefix, params):
        self.request_counter += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            return await self.session.generate(
                AsyncGenerationRequest(
                    request_id=f"self-v{self.version}-{self.request_counter}",
                    prompt_token_ids=tuple(prefix),
                    sampling_params=params,
                    lora_request=self.request,
                )
            )
        finally:
            self.in_flight -= 1

    async def generate(self, prefix, max_tokens=512):
        if not 1 <= max_tokens <= 2048:
            raise ValueError("reply budget must be in1..2048 tokens")
        print(json.dumps({"stage": "engine_generate", "policy_version": self.version}), flush=True)
        result = await self._generate(
            prefix,
            SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[self.tokenizer.eos_token_id], logprobs=1),
        )
        if list(result.prompt_token_ids) != prefix:
            raise ValueError("engine changed the supplied prompt token IDs")
        completion = result.outputs[0]
        ids = list(completion.token_ids)
        if not ids or not completion.logprobs:
            raise ValueError("engine returned no original action tokens/logprobs")
        logprobs = [float(item[token].logprob) for item, token in zip(completion.logprobs, ids, strict=True)]
        if not all(math.isfinite(v) for v in logprobs):
            raise ValueError("nonfinite rollout log probabilities")
        return (
            ids,
            completion.finish_reason == "stop",
            {
                "policy_version": self.version,
                "adapter_sha256": self.refreshes[-1]["weights_sha256"],
                "sampled_logprobs": logprobs,
                "finish_reason": completion.finish_reason,
                "num_cached_tokens": result.num_cached_tokens,
            },
        )

    async def score_probe(self, model, prefix, targets):
        sequence = prefix + targets
        if not await self.engine.reset_prefix_cache():
            raise ValueError("probe prefix-cache reset was rejected")
        result = await self._generate(
            sequence,
            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1),
        )
        rollout = [float(result.prompt_logprobs[i][sequence[i]].logprob) for i in range(len(prefix), len(sequence))]
        model.eval()
        with torch.no_grad():
            ids = torch.tensor([sequence[:-1]], device="cuda")
            positions = torch.arange(len(prefix) - 1, len(sequence) - 1, device="cuda")
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=positions).logits[0]
            selected = torch.tensor(targets, device="cuda")
            actor = logits.float().log_softmax(-1).gather(-1, selected[:, None]).flatten().tolist()
        maximum = max(abs(a - b) for a, b in zip(actor, rollout, strict=True))
        record = {"policy_version": self.version, "actor": actor, "rollout": rollout, "max_abs_delta": maximum}
        (self.output / f"parity-v{self.version}.json").write_text(json.dumps(record, indent=2))
        if not all(math.isfinite(v) for v in actor + rollout) or maximum > 0.15:
            raise ValueError(f"actor/rollout selected-token discrepancy {maximum} exceeds 0.15 nats")
        return record

    async def batch_probe(self, prefix, maximum=4):
        params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, logprobs=1)
        variants = [prefix, prefix + self.tokenizer.encode("\n", add_special_tokens=False)]
        records = []
        from sdpo_speculation import batch_orders

        orders = batch_orders(maximum)
        for order in orders:
            if not await self.engine.reset_prefix_cache():
                raise ValueError("batch probe prefix-cache reset rejected")
            self.peak_in_flight = 0
            started = time.perf_counter()
            outputs = await asyncio.gather(*(self._generate(variants[i], params) for i in order))
            completion = outputs[order.index(0)].outputs[0]
            ids = list(completion.token_ids)
            records.append(
                {
                    "order": order,
                    "elapsed_s": time.perf_counter() - started,
                    "peak_in_flight": self.peak_in_flight,
                    "tokens": ids,
                    "logprobs": [float(item[t].logprob) for item, t in zip(completion.logprobs, ids, strict=True)],
                }
            )
        passed = all(r["tokens"] == records[0]["tokens"] and r["logprobs"] == records[0]["logprobs"] for r in records)
        receipt = {"policy_version": self.version, "exact_batch_invariance": passed, "records": records}
        (self.output / f"batch-invariance-v{self.version}.json").write_text(json.dumps(receipt, indent=2))
        if not passed:
            raise ValueError("rollout tokens/logprobs changed across diagnostic batch mixes")
        if maximum > 4 and max(row["peak_in_flight"] for row in records) != maximum:
            raise ValueError("batch qualification did not reach requested concurrent admission")
        return receipt
