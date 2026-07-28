"""Tests for offline vLLM benchmark helpers."""

from __future__ import annotations

import unittest

from posttrain.common import InferenceBinding, Workload
from posttrain.serve import ServeBenchmarkRequest
from posttrain.serve.backends.vllm.bindings import benchmark_config
from posttrain.serve.backends.vllm.offline import (
    _controlled_prompt_ids,
    _duration,
    _KvCacheTracker,
    _percentile,
    _point_failure,
    _representative_prompt_batches,
    _shutdown_llm,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **_: object,
    ) -> list[int]:
        content = str(messages[-1]["content"])
        return [ord(character) for character in content]


class ServeBenchmarkTest(unittest.TestCase):
    def test_controlled_prompts_have_exact_input_length(self) -> None:
        prompts = _controlled_prompt_ids(_Tokenizer(), 257, 8)

        self.assertEqual(len(prompts), 8)
        self.assertTrue(all(len(prompt["prompt_token_ids"]) == 257 for prompt in prompts))

    def test_percentile_handles_single_and_multiple_samples(self) -> None:
        self.assertEqual(_percentile([0.2], 0.95), 0.2)
        self.assertEqual(_percentile([0.1, 0.2, 0.3], 0.50), 0.2)

    def test_request_stage_duration_requires_one_monotonic_clock(self) -> None:
        self.assertEqual(_duration(10.0, 10.25), 0.25)
        self.assertIsNone(_duration(0.0, 10.25))
        self.assertIsNone(_duration(10.25, 10.0))

    def test_kv_cache_tracker_emits_only_measured_scheduler_pressure(self) -> None:
        observed: list[tuple[dict[str, float], dict[str, str | int]]] = []
        tracker = _KvCacheTracker(lambda values, attributes: observed.append((values, attributes)), interval_seconds=0)

        class Stats:
            kv_cache_usage = 0.625
            num_running_reqs = 8
            num_waiting_reqs = 2

        tracker.record(Stats(), None)
        tracker.start()
        tracker.record(Stats(), None)
        tracker.stop()
        tracker.record(Stats(), None)

        self.assertEqual(tracker.peak_usage_ratio, 0.625)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0]["serve/backend/kv_cache_usage_ratio"], 0.625)

    def test_point_failure_classifies_resource_and_unsupported_boundaries(self) -> None:
        oom = _point_failure(RuntimeError("CUDA out of memory"), sweep_index=2, concurrency=8)
        unsupported = _point_failure(
            NotImplementedError("mode is not supported"),
            sweep_index=3,
            concurrency=16,
        )

        self.assertEqual(oom.status, "resource_exhausted")
        self.assertEqual(unsupported.status, "unsupported")
        self.assertEqual(oom.message, "CUDA out of memory")


def test_representative_batches_are_seeded_and_cover_corpus_once(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    request = benchmark_config(ServeBenchmarkRequest(qwen_screen_binding, representative_workload))

    batches, record_ids = _representative_prompt_batches(_Tokenizer(), request, request.cells[0])
    repeated_batches, repeated_record_ids = _representative_prompt_batches(
        _Tokenizer(),
        request,
        request.cells[0],
    )

    assert record_ids == repeated_record_ids
    assert batches == repeated_batches
    assert len(batches) == 32
    assert all(len(batch) == 4 for batch in batches)
    assert len(set(record_ids)) == 128


def test_shutdown_llm_closes_vllm_engine_client() -> None:
    calls: list[bool] = []

    class Client:
        def shutdown(self) -> None:
            calls.append(True)

    class Engine:
        engine_core = Client()

    class Llm:
        llm_engine = Engine()

    _shutdown_llm(Llm())

    assert calls == [True]


if __name__ == "__main__":
    unittest.main()
