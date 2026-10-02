"""Durable candidate accounting around veRL's native active-sampling hooks.

The native buffer owns dispatch sizes, spread filtering, eviction and selection.
This adapter publishes original episode artifacts before native eviction and
records actual selected keys, including surplus and unused reserved candidates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from posttrain.common import RunContext

from ...update_records import InvalidPolicyUpdate
from ..policy_collection_evidence import publish_collection_snapshot
from .policy_rollouts import NativeEpisodeReceipt


class NativeActiveCollectionBuffer:
    """Observe one native buffer without implementing another refill scheduler."""

    def __init__(self, native: Any, context: RunContext, destination: Path, generations: int):
        if native.active_observe_metric != native.active_metric:
            raise InvalidPolicyUpdate("resolved native collection has not qualified a separate selector observation metric")
        self.native, self.context = native, context
        self.destination, self.generations = destination.resolve(), generations
        self._dispatch = native.dispatch_fn
        self._observe = native.observe_fn
        self._state: dict[str, Any] | None = None
        self._pending: tuple[str, ...] = ()
        native.dispatch_fn = self._dispatch_round
        native.observe_fn = self._observe_round

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def begin(self, tasks: tuple[str, ...], uids: tuple[str, ...], sampler_step: int) -> None:
        if self._pending or (self._state is not None and self._state["status"] not in {"selected", "failed"}):
            raise InvalidPolicyUpdate("native active reservation cannot replace an unfinished collection")
        if (len(tasks) != len(uids) or not tasks or len(set(tasks)) != len(tasks)
                or len(set(uids)) != len(uids) or any(not uid or "_" in uid for uid in uids)
                or type(sampler_step) is not int or sampler_step < 0):
            raise InvalidPolicyUpdate("native active collection requires distinct reserved task/UID identities")
        self._state = {"schema": "posttrain.native-active-collection@1", "sampler_step": sampler_step,
            "native_metric": self.native.active_metric, "native_reward_std_epsilon": self.native.active_reward_std_epsilon,
            "reserved": [{"uid": uid, "task": task} for task, uid in zip(tasks, uids, strict=True)],
            "rounds": [], "groups": [], "selected": None, "status": "reserved"}
        self._pending = ()
        self._publish()

    def _publish(self) -> None:
        if self._state is None:
            raise InvalidPolicyUpdate("native active collection lacks its checked reservation")
        publish_collection_snapshot(self.context, self.destination, self._state)

    def _dispatch_round(self, count: int, round_index: int | None = None) -> list[str]:
        if self._state is None or self._pending:
            raise InvalidPolicyUpdate("native active dispatch requires a finished preceding round")
        cursor = sum(len(item["uids"]) for item in self._state["rounds"])
        expected = [item["uid"] for item in self._state["reserved"][cursor:cursor + count]]
        if len(expected) != count:
            raise InvalidPolicyUpdate("native active dispatch exceeds the checked reservation")
        self._state["rounds"].append({"index": round_index, "uids": expected})
        self._state["status"] = "dispatching"
        self._publish()  # Retain intent before handing tasks to native rollout.
        dispatched = list(self._dispatch(count, round_index=round_index))
        if dispatched != expected:
            raise InvalidPolicyUpdate("native dispatch changed reserved candidate order")
        self._pending = tuple(dispatched)
        return dispatched

    def _observe_round(self, groups: list[tuple[str, list[float]]]) -> None:
        import transfer_queue as tq  # pyright: ignore[reportMissingImports]

        if self._state is None or not self._pending:
            raise InvalidPolicyUpdate("native active observation lacks its dispatched round")
        finished = [uid for uid, _ in groups]
        metrics = dict(groups)
        if len(set(finished)) != len(finished) or any(uid not in self._pending for uid in finished):
            raise InvalidPolicyUpdate("native active observation changes round membership")
        tasks = {item["uid"]: item["task"] for item in self._state["reserved"]}
        seen_episodes = set()
        seen_traces = set()
        for item in self._state["groups"]:
            for value in item["receipts"]:
                receipt = NativeEpisodeReceipt.model_validate_json(value)
                seen_traces.add(receipt.rollout.trace.external_id)
                info = receipt.rollout.trace.payload.get("info")
                episode = info.get("posttrain_episode_id") if isinstance(info, Mapping) else None
                if not isinstance(episode, str):
                    raise InvalidPolicyUpdate("observed native collection lost episode identity")
                seen_episodes.add(episode)
        captured = []
        for uid in self._pending:
            complete = uid in finished
            if not complete and uid not in self.native.failure_keys["train"]:
                raise InvalidPolicyUpdate("native active round includes a nonterminal candidate")
            keys = [key for key in self.native.partitions["train"] if key.split("_")[0] == uid]
            data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["extra_fields"]) if keys else {}
            values = list(data.get("extra_fields", []))
            if len(values) != len(keys) or (complete and len(values) != self.generations):
                raise InvalidPolicyUpdate("native active observation requires complete finished episode groups")
            if complete and len(metrics[uid]) != self.generations:
                raise InvalidPolicyUpdate("native active classification differs from complete receipt coverage")
            encoded = []
            for value in values:
                extra = getattr(value, "data", value)
                receipt_value = extra.get("posttrain_native_episode_receipt") if isinstance(extra, Mapping) else None
                if receipt_value is None and not complete:
                    continue  # Native failure may have no completed episode.
                if not isinstance(receipt_value, str):
                    raise InvalidPolicyUpdate("native active group lacks its original episode receipt")
                receipt = NativeEpisodeReceipt.model_validate_json(receipt_value)
                receipt.verify_artifact()
                rollout = receipt.rollout
                info = rollout.trace.payload.get("info")
                episode = info.get("posttrain_episode_id") if isinstance(info, Mapping) else None
                group = info.get("posttrain_prompt_group_id") if isinstance(info, Mapping) else None
                step = self._state["sampler_step"]
                if (rollout.example_id != tasks[uid] or group != f"{self.context.run_id}/{step}/{uid}"
                        or not isinstance(episode, str) or not episode or episode in seen_episodes
                        or rollout.trace.external_id in seen_traces
                        or rollout.behavior_policy is None or rollout.behavior_policy.start != step
                        or rollout.behavior_policy.end != step):
                    raise InvalidPolicyUpdate("native active receipt changes reserved task/group/policy identity")
                seen_episodes.add(episode)
                seen_traces.add(rollout.trace.external_id)
                self.context.artifact(receipt.artifact)
                encoded.append(receipt_value)
            scores = metrics[uid] if complete else []
            captured.append({"uid": uid, "terminal": "finished" if complete else "failed", "receipts": encoded,
                "native_metric_values": [value if math.isfinite(value) else None for value in scores],
                "native_spread_eligible": self.native._keeps_group(scores) if complete else False})  # noqa: SLF001
        # The native predicate/filter is still evaluated by the native buffer.
        self._state["groups"].extend(captured)
        self._state["status"] = "round-observed"
        self._publish()  # Callback occurs before native rejected-group eviction.
        if self._observe is not None:
            self._observe(groups)
        self._pending = ()

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> Any:
        if partition_id == "val":
            return self.native.sample(global_steps, partition_id, batch_size)
        if self._state is None or global_steps != self._state["sampler_step"]:
            raise InvalidPolicyUpdate("native active sample differs from its reserved policy boundary")
        try:
            batch, metrics = self.native.sample(global_steps, partition_id, batch_size)
            selected = list(dict.fromkeys(key.split("_")[0] for key in batch.keys))
            finished = {item["uid"] for item in self._state["groups"] if item["terminal"] == "finished"}
            if len(selected) != batch_size or any(uid not in finished for uid in selected):
                raise InvalidPolicyUpdate("native active selection lacks observed complete groups")
            self._state["selected"] = selected
            self._state["status"] = "selected"
            self._publish()
            return batch, metrics
        except Exception:
            self._state["status"] = "failed"
            self._publish()
            raise
