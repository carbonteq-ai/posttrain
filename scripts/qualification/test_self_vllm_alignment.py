"""Exercise the pinned scheduler's composed hybrid/invariant prefill splits on CPU."""

import unittest
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def chunks(block_size, length):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_config = SimpleNamespace(block_size=block_size)
    scheduler.use_eagle_block_drop = False
    scheduler.hash_block_size = block_size
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_partial_cache_hit = False
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.batch_invariant_prefill_alignment = 128
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=length,
        num_tokens=length,
        shared_prefix_boundary=0,
    )
    result = []
    while request.num_computed_tokens < length:
        step = scheduler._mamba_block_aligned_split(request, length - request.num_computed_tokens)
        step = scheduler._batch_invariant_prefill_split(request, step)
        result.append(step)
        if step == 0:
            break
        request.num_computed_tokens += step
    return result


class AlignmentChecks(unittest.TestCase):
    def test_default_528_block_stalls_at_observed_prompt_shape(self):
        self.assertEqual(chunks(528, 1864), [1536, 0])

    def test_640_blocks_progress_across_prompt_lengths(self):
        for length in (1, 127, 128, 129, 512, 1024, 1864, 4097, 7680):
            with self.subTest(length=length):
                values = chunks(640, length)
                self.assertNotIn(0, values)
                self.assertEqual(sum(values), length)


if __name__ == "__main__":
    unittest.main()
