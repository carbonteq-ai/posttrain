"""Translate native assignments without choosing reward channels or advantages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..reward_evidence import AssignedCredit, AssignedCreditEvidence, CreditAttempt, InvalidRewardEvidence


def align_native_credit(
    trace: Any,
    branch: Any,
    first_sampled: int,
    *,
    assignments: Iterable[Any] = (),
    traces: Mapping[str, Any] | None = None,
) -> AssignedCreditEvidence:
    """Retain latest journal entry per attempt, with no selection between retries.

    Named semantic occurrences require an explicit branch resolver and remain
    unsupported here. A native branch ordinal is never used as that resolver.
    """
    records = tuple(getattr(trace, "credit_assignments", ())) + tuple(assignments)
    if not records:
        return AssignedCreditEvidence()
    from verifiers.v1.assessment_projection import project_assignment

    native_positions = {id(node): index for index, node in enumerate(trace.nodes)}
    offsets: dict[int, int] = {}
    offset = -first_sampled
    path = []
    for node in branch.nodes:
        index = native_positions[id(node)]
        offsets[index] = offset
        path.append({"node": index, "tokens": list(node.token_ids), "mask": list(node.mask)})
        offset += len(node.token_ids)
    branch_digest = hashlib.sha256(
        json.dumps({"trace": str(trace.id), "path": path}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    latest = {}
    for assignment in records:
        request = assignment.request
        key = (request.invocation_id, request.attempt_id)
        previous = latest.get(key)
        if previous is not None:
            if previous.request.request_id != request.request_id:
                raise InvalidRewardEvidence("credit journal changed request within an attempt")
            if previous.status in {"complete", "failed", "interrupted"} and previous != assignment:
                raise InvalidRewardEvidence("credit journal changed after terminal assignment")
            prior = {item.contribution_id: item for item in previous.contributions}
            current = {item.contribution_id: item for item in assignment.contributions}
            if any(current.get(identity) != item for identity, item in prior.items()):
                raise InvalidRewardEvidence("credit journal rewrote accepted contributions")
        latest[key] = assignment
    result = []
    attempts = []
    for assignment in latest.values():
        aligned = project_assignment(assignment, traces or {str(trace.id): trace})
        assignment = aligned.assignment
        assignment_id = assignment.assignment_id
        rule_digest = assignment.request.rule.digest
        attempts.append(
            CreditAttempt(
                assignment_id=assignment_id,
                rule_digest=rule_digest,
                invocation_id=assignment.request.invocation_id,
                attempt_id=assignment.request.attempt_id,
                status=assignment.status,
                requested_count=len(assignment.request.targets),
                missing_count=len(assignment.missing),
                overlap_policy=assignment.request.overlap_policy,
            )
        )
        projections = {item.contribution_id: item.projection for item in aligned.contributions}
        for contribution in aligned.assignment.contributions:
            projection = projections[contribution.contribution_id]
            intervals = []
            reason = projection.reason
            status = "unsupported"
            if contribution.branch_id is not None:
                reason = "semantic_branch_resolver_required"
            elif contribution.recipient.episode_id != trace.episode_id or (
                contribution.recipient.kind == "trace" and contribution.recipient.trace_id != str(trace.id)
            ):
                status, reason = "absent", "recipient_not_on_selected_episode"
            elif projection.status == "failed":
                status = "failed"
            elif projection.status in {"exact_turn", "exact_span", "exact_call"}:
                status = "exact"
                for span in projection.intervals:
                    if span.trace_id != str(trace.id) or span.node_index not in offsets:
                        status, reason = "absent", "recipient_not_on_selected_path"
                        break
                    start = offsets[span.node_index] + span.start
                    end = offsets[span.node_index] + span.end
                    if start < 0:
                        status, reason = "absent", "recipient_precedes_completion"
                        break
                    intervals.append((start, end))
            result.append(
                AssignedCredit(
                    assignment_id=assignment_id,
                    contribution_id=contribution.contribution_id,
                    rule_digest=rule_digest,
                    attempt_id=aligned.assignment.request.attempt_id,
                    channel=contribution.channel,
                    semantics=contribution.signal.semantics,
                    branch_digest=branch_digest,
                    status=contribution.status,
                    alignment=status,
                    allocation=contribution.allocation,
                    value=contribution.value,
                    weight=contribution.weight,
                    signal_digest=hashlib.sha256(contribution.signal.model_dump_json().encode()).hexdigest(),
                    units=contribution.signal.units,
                    transformation=contribution.transformation,
                    parent_assessment_ids=contribution.parent_assessment_ids,
                    attribution=contribution.attribution,
                    recipient_kind=contribution.recipient.kind,
                    recipient_id=contribution.recipient.subject_id,
                    source_snapshot_id=contribution.recipient.snapshot_id,
                    recipient_trace_id=contribution.recipient.trace_id,
                    gate_ids=tuple(gate.gate_id for gate in contribution.gates),
                    intervals=tuple(intervals) if status == "exact" else (),
                    reason=reason,
                )
            )
    return AssignedCreditEvidence(
        attempts=tuple(attempts),
        contributions=tuple(result),
        branch_digest=branch_digest,
    )
