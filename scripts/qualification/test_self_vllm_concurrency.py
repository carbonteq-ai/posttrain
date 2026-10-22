"""Exercise async request identity/accounting without allocating a model."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from self_feedback_vllm import LocalVLLMRollout


class Session:
    async def generate(self, request):
        await asyncio.sleep(0.01)
        return request.request_id


class ConcurrencyChecks(unittest.IsolatedAsyncioTestCase):
    async def test_32_requests_have_unique_ids_and_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout = LocalVLLMRollout(None, Path(directory))
            rollout.session = Session()
            identities = await asyncio.gather(*(rollout._generate([1], None) for _ in range(32)))
            self.assertEqual(len(set(identities)), 32)
            self.assertEqual(rollout.peak_in_flight, 32)
            self.assertEqual(rollout.in_flight, 0)

    async def test_cancelled_request_is_removed_from_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout = LocalVLLMRollout(None, Path(directory))
            rollout.session = Session()
            task = asyncio.create_task(rollout._generate([1], None))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(rollout.in_flight, 0)


if __name__ == "__main__":
    unittest.main()
