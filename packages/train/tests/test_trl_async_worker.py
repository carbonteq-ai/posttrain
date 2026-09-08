import asyncio
import queue
import threading
import time

import pytest
from posttrain.train.backends.trl.async_worker import AsyncGroupRejected, TrlAsyncRolloutWorker
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample


def sample(group_id: int, version: int) -> RolloutSample:
    return RolloutSample([], [], [1, 2], [0, 1], [0.0, -0.2], 0.5, version, group_id, {"reward": 1.0})


class OrderedProducer:
    def __init__(self):
        self.calls = 0
        self.release_slow = threading.Event()
        self.closed = False

    async def astart(self):
        pass

    async def produce_group(self, target_policy_version):
        self.calls += 1
        call = self.calls
        if call == 1:
            while not self.release_slow.is_set():
                await asyncio.sleep(0.01)
        elif call > 2:
            while True:
                await asyncio.sleep(1)
        return (sample(call, target_policy_version),)

    async def prepare_model_update(self, model_version):
        pass

    async def activate_model_version(self, model_version):
        pass

    async def aclose(self):
        self.closed = True


def test_short_group_publishes_while_unrelated_group_is_waiting():
    producer = OrderedProducer()
    worker = TrlAsyncRolloutWorker(
        producer,
        initial_model_version=3,
        max_inflight_groups=2,
        queue_maxsize=2,
        shutdown_timeout_s=2,
    )
    worker.start()
    try:
        first = worker.rollout_buffer.get(timeout=1)
        assert first.group_id == 2
        producer.release_slow.set()
        second = worker.rollout_buffer.get(timeout=1)
        assert second.group_id == 1
        worker.check_health(1)
    finally:
        worker.stop()
    assert producer.closed


class FailingProducer:
    async def astart(self):
        pass

    async def produce_group(self, target_policy_version):
        del target_policy_version
        raise ValueError("broken environment source")

    async def prepare_model_update(self, model_version):
        pass

    async def activate_model_version(self, model_version):
        pass

    async def aclose(self):
        pass


class FailingStartupProducer(FailingProducer):
    def __init__(self):
        self.closed = False

    async def astart(self):
        raise ValueError("broken producer startup")

    async def aclose(self):
        self.closed = True


def test_startup_failure_closes_producer_and_does_not_leave_worker_started():
    producer = FailingStartupProducer()
    worker = TrlAsyncRolloutWorker(
        producer, max_inflight_groups=1, queue_maxsize=1, shutdown_timeout_s=1
    )

    with pytest.raises(RuntimeError, match="broken producer startup"):
        worker.start()

    assert producer.closed
    with pytest.raises(RuntimeError, match="broken producer startup"):
        worker.check_health(1)


class RejectingProducer:
    def __init__(self, *, always=False):
        self.calls = 0
        self.always = always

    async def astart(self):
        pass

    async def produce_group(self, target_policy_version):
        self.calls += 1
        if self.always or self.calls == 1:
            raise AsyncGroupRejected("invalid verifier evidence")
        return (sample(self.calls, target_policy_version),)

    async def prepare_model_update(self, model_version):
        pass

    async def activate_model_version(self, model_version):
        pass

    async def aclose(self):
        pass


def test_failure_reaches_native_health_contract_without_fake_sample():
    worker = TrlAsyncRolloutWorker(
        FailingProducer(), max_inflight_groups=1, queue_maxsize=1, shutdown_timeout_s=2
    )
    worker.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with pytest.raises(queue.Empty):
            worker.rollout_buffer.get_nowait()
        try:
            worker.check_health(1)
        except RuntimeError as error:
            assert "broken environment source" in str(error)
            break
        time.sleep(0.01)
    else:
        pytest.fail("worker failure was not reported")
    with pytest.raises(RuntimeError, match="broken environment source"):
        worker.stop()


def test_rejected_group_is_dropped_and_next_group_can_publish():
    producer = RejectingProducer()
    worker = TrlAsyncRolloutWorker(
        producer, max_inflight_groups=1, queue_maxsize=1, max_consecutive_rejections=2
    )
    worker.start()
    try:
        published = worker.rollout_buffer.get(timeout=1)
        assert published.group_id == 2
        assert worker.metrics_queue.get_nowait() == {"rollout/rejected_groups": 1.0}
    finally:
        worker.stop()


def test_repeated_group_rejection_fails_at_explicit_bound():
    worker = TrlAsyncRolloutWorker(
        RejectingProducer(always=True),
        max_inflight_groups=1,
        queue_maxsize=1,
        max_consecutive_rejections=2,
    )
    worker.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            worker.check_health(1)
        except RuntimeError as error:
            assert "rejection limit" in str(error)
            break
        time.sleep(0.01)
    else:
        pytest.fail("bounded group rejection failure was not reported")
    with pytest.raises(RuntimeError, match="rejection limit"):
        worker.stop()


def test_rejects_unbounded_queue_and_version_regression():
    producer = OrderedProducer()
    with pytest.raises(ValueError, match="positive bound"):
        TrlAsyncRolloutWorker(producer, max_inflight_groups=1, queue_maxsize=0)
    with pytest.raises(ValueError, match="rejection limit"):
        TrlAsyncRolloutWorker(
            producer, max_inflight_groups=1, queue_maxsize=1, max_consecutive_rejections=0
        )
    worker = TrlAsyncRolloutWorker(producer, initial_model_version=4, max_inflight_groups=1, queue_maxsize=1)
    with pytest.raises(ValueError, match="backwards"):
        worker.update_model_version(3)


class FullQueueProducer:
    def __init__(self):
        self.closed = False

    async def astart(self):
        pass

    async def produce_group(self, target_policy_version):
        return (sample(1, target_policy_version), sample(1, target_policy_version))

    async def prepare_model_update(self, model_version):
        pass

    async def activate_model_version(self, model_version):
        pass

    async def aclose(self):
        self.closed = True


def test_shutdown_remains_responsive_when_native_queue_is_full():
    producer = FullQueueProducer()
    worker = TrlAsyncRolloutWorker(
        producer, max_inflight_groups=1, queue_maxsize=1, shutdown_timeout_s=1
    )
    worker.start()
    deadline = time.monotonic() + 1
    while worker.rollout_buffer.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not worker.rollout_buffer.empty()

    worker.stop()

    assert producer.closed


class UpdateGateProducer:
    def __init__(self):
        self.calls = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.second_started = threading.Event()
        self.lifecycle = []

    async def astart(self):
        pass

    async def produce_group(self, target_policy_version):
        self.calls += 1
        call = self.calls
        if call == 1:
            self.first_started.set()
            while not self.release_first.is_set():
                await asyncio.sleep(0.01)
        elif call == 2:
            self.second_started.set()
        else:
            while True:
                await asyncio.sleep(1)
        return (sample(call, target_policy_version),)

    async def prepare_model_update(self, model_version):
        self.lifecycle.append(("prepare", model_version))

    async def activate_model_version(self, model_version):
        self.lifecycle.append(("activate", model_version))

    async def aclose(self):
        pass


def test_weight_update_gates_new_groups_until_version_activation():
    producer = UpdateGateProducer()
    worker = TrlAsyncRolloutWorker(
        producer, initial_model_version=1, max_inflight_groups=1, queue_maxsize=2, shutdown_timeout_s=2
    )
    worker.start()
    try:
        assert producer.first_started.wait(timeout=1)
        worker.prepare_model_update(2)
        producer.release_first.set()
        first = worker.rollout_buffer.get(timeout=1)
        assert first.model_version == 1
        assert not producer.second_started.wait(timeout=0.1)

        worker.update_model_version(2)
        assert producer.second_started.wait(timeout=1)
        second = worker.rollout_buffer.get(timeout=1)
        assert second.model_version == 2
    finally:
        worker.stop()

    assert producer.lifecycle == [("prepare", 2), ("activate", 2)]
