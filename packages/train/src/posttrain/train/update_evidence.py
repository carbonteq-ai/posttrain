"""Assemble admitted native trajectory views without reconstructing transcripts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from .online_rl import EnvironmentRollout
from .sampo_advantages import anchor_identities
from .update_credit import NativeCreditRows
from .update_records import (
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
    SemanticSpan,
    require_identity,
)


def population_from_rollouts(
    rollouts: tuple[EnvironmentRollout, ...],
    *,
    population_id: str,
    native_evidence_ref: str,
    native_evidence_digest: str,
    template_revision: str,
    versions: PolicyVersions,
    sampler_step: int,
    num_generations: int,
    selector_digest: str,
    spans: tuple[SemanticSpan, ...] = (),
    anchor_fallback: bool = False,
) -> tuple[PopulationSnapshot, NativeCreditRows]:
    """Require complete admitted groups and original native conditioning maps.

    Native references must still be checked against retained bytes by the scorer.
    This validates transport, population coverage and version coherence. Legacy
    flattened rows without view provenance cannot enter this explicit path.
    Rows are grouped by first-seen prompt identity, preserving order within each
    group, because the existing SAMPO estimator requires contiguous groups.
    """
    require_identity(population_id, native_evidence_ref, native_evidence_digest, template_revision, selector_digest)
    if type(num_generations) is not int or num_generations < 2 or type(sampler_step) is not int or sampler_step < 0:
        raise InvalidPolicyUpdate("native population requires generation count and synchronous sampler step")
    if not rollouts:
        raise InvalidPolicyUpdate("native population requires admitted trajectories")
    grouped: dict[str, list[EnvironmentRollout]] = {}
    for rollout in rollouts:
        info = rollout.trace.payload.get("info")
        group = info.get("posttrain_prompt_group_id") if isinstance(info, Mapping) else None
        if not isinstance(group, str):
            raise InvalidPolicyUpdate("native rollout lacks retained episode/group identities")
        grouped.setdefault(group, []).append(rollout)
    seen_tasks: set[str] = set()
    for members in grouped.values():
        if len({rollout.example_id for rollout in members}) != 1:
            raise InvalidPolicyUpdate("native prompt group must share one example identity")
        task = members[0].example_id
        if task in seen_tasks:
            raise InvalidPolicyUpdate("fresh native population cannot collect one task in multiple prompt groups")
        seen_tasks.add(task)
    rollouts = tuple(rollout for members in grouped.values() for rollout in members)
    # Anchor relations record the same groups SAMPO's credit uses (sampo_advantages.anchor_identities).
    identities_of: dict[int, list[str]] = {}
    for members in grouped.values():
        for rollout, identities in zip(members, anchor_identities(members, fallback=anchor_fallback), strict=True):
            identities_of[id(rollout)] = identities
    views: list[ConditioningView] = []
    rows: list[np.ndarray] = []
    prompt_members: dict[str, list[str]] = defaultdict(list)
    anchor_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    group_episodes: dict[str, list[str]] = defaultdict(list)
    seen_episodes: set[str] = set()
    position = 0
    for rollout in rollouts:
        info = rollout.trace.payload.get("info")
        if not isinstance(info, Mapping):
            raise InvalidPolicyUpdate("native rollout lacks retained episode/group identities")
        episode_id, group_id = info.get("posttrain_episode_id"), info.get("posttrain_prompt_group_id")
        if not isinstance(episode_id, str) or not isinstance(group_id, str):
            raise InvalidPolicyUpdate("native rollout lacks retained episode/group identities")
        require_identity(episode_id, group_id)
        evidence = rollout.reward_evidence
        if evidence is not None and (
            evidence.prompt_group_id != group_id or evidence.rollout_id != info.get("posttrain_rollout_id")
        ):
            raise InvalidPolicyUpdate("native reward evidence disagrees with admitted group/rollout")
        if episode_id in seen_episodes:
            raise InvalidPolicyUpdate("initial native population admits one policy trajectory per episode")
        seen_episodes.add(episode_id)
        group_episodes[group_id].append(episode_id)
        policy = rollout.behavior_policy
        if policy is None or policy.start != sampler_step or policy.end != sampler_step:
            raise InvalidPolicyUpdate("native population must prove one synchronous sampler version")
        if not rollout.conditioning_records or rollout.selected_branch_id is None:
            raise InvalidPolicyUpdate("legacy flattened rollout has no proven original conditioning views")
        # Completion index -> population position, -1 where the token was not sampled.
        row = np.full(len(rollout.completion_ids), -1, dtype=np.int64)
        if rollout.turns and len(rollout.turns) != len(rollout.conditioning_records):
            raise InvalidPolicyUpdate("native turn credit and conditioning views disagree")
        for turn_index, (record, indices) in enumerate(
            zip(
                rollout.conditioning_records,
                rollout.conditioning_completion_indices,
                strict=True,
            )
        ):
            if rollout.turns:
                turn = rollout.turns[turn_index]
                if indices != tuple(range(turn.completion_start, turn.completion_end)):
                    raise InvalidPolicyUpdate("native turn credit and sampled action coordinates disagree")
            if len(indices) != len(record.sampled_token_indices):
                raise InvalidPolicyUpdate("native sampled actions and completion coordinates disagree")
            if any(index < 0 or index >= len(row) or row[index] != -1 for index in indices):
                raise InvalidPolicyUpdate("native sampled actions address invalid completion coordinates")
            view_id = f"{record.trace_id}/node-{record.node_index}"
            coordinates = json.dumps(
                {
                    "trace_id": record.trace_id,
                    "prefix_nodes": record.prefix_node_indices,
                    "node_index": record.node_index,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            views.append(
                ConditioningView(
                    view_id,
                    native_evidence_ref,
                    coordinates,
                    f"{record.context_contract}/attention",
                    f"{record.context_contract}/positions",
                    template_revision,
                    record.input_digest,
                    record.context_tokens,
                    episode_id,
                    rollout.selected_branch_id,
                    tuple(record.sampled_token_indices),
                )
            )
            row[list(indices)] = np.arange(position, position + len(indices), dtype=np.int64)
            position += len(indices)
            prompt_members[group_id].append(view_id)
            if rollout.turns:
                anchor_members[(group_id, identities_of[id(rollout)][turn_index])].append(view_id)
        if [bool(value) for value in rollout.env_mask] != [index >= 0 for index in row.tolist()]:
            raise InvalidPolicyUpdate("native credit coordinates must match original sampled eligibility")
        row.setflags(write=False)
        rows.append(row)
    if any(len(episodes) != num_generations for episodes in group_episodes.values()):
        raise InvalidPolicyUpdate("native population requires complete prompt groups before credit")
    relations = tuple(
        PopulationRelation(f"prompt/{identity}", "prompt-group", tuple(members), "complete", tuple(members))
        for identity, members in prompt_members.items()
    ) + tuple(
        PopulationRelation(
            json.dumps(("anchor", *identity), separators=(",", ":")),
            "anchor-state",
            tuple(members),
            "complete",
            tuple(members),
        )
        for identity, members in anchor_members.items()
    )
    snapshot = PopulationSnapshot(
        population_id,
        native_evidence_ref,
        native_evidence_digest,
        tuple(views),
        spans,
        relations,
        versions,
        selector_digest,
    )
    return snapshot, NativeCreditRows(rollouts, tuple(rows), (native_evidence_digest,))
