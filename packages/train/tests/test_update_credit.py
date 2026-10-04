from dataclasses import replace

import pytest
from posttrain.train.assigned_rewards import CreditSelection, episode_reward, local_token_rewards, with_turn_rewards
from posttrain.train.profiles import CAPOSettings, GDPOSettings, GRPOSettings
from posttrain.train.reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from posttrain.train.reward_evidence import (
    AssignedCredit,
    AssignedCreditEvidence,
    CreditAttempt,
    InvalidRewardEvidence,
    ProcessCredit,
    RewardEvidence,
    RewardValue,
)
from posttrain.train.update_credit import (
    ActionCredit,
    NativeCreditRows,
    PreparedCredit,
    SampoCreditEstimator,
    ScalarGroupCreditEstimator,
    StructuredCreditEstimator,
    prepare_credit,
)
from posttrain.train.update_records import (
    ActionRecord,
    ActionRef,
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
    record_digest,
)

from .test_sampo import _rollout, _settings
from .test_update_records import population


class ExternalEstimator:
    """Declared external estimator fixture, not a recommended PRM recipe."""

    id = "external-fixture@1"
    required_relations = ("prompt",)

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit:
        return PreparedCredit(
            snapshot.digest,
            self.id,
            tuple(
                ActionCredit(record.action, (-1.0, 0.0, 1.0)[index]) for index, record in enumerate(snapshot.actions)
            ),
            self.required_relations,
            (("step-return", 1.0),),
            "externally-prepared@1",
            "prefix",
            ("retained-step-assessment-digest",),
        )


def test_external_estimator_preserves_detached_credit_and_identity() -> None:
    snapshot = population()
    credit = prepare_credit(snapshot, ExternalEstimator())
    assert tuple(value.advantage for value in credit.values) == (-1.0, 0.0, 1.0)
    assert credit.digest == prepare_credit(snapshot, ExternalEstimator()).digest


def test_partial_population_rejected_before_estimator_is_called() -> None:
    snapshot = population()
    partial = replace(snapshot.relations[0], completeness="partial", members=snapshot.relations[0].members[:1])
    snapshot = replace(snapshot, relations=(partial, snapshot.relations[1]))
    with pytest.raises(InvalidPolicyUpdate, match="complete population"):
        prepare_credit(snapshot, ExternalEstimator())


def test_credit_must_cover_original_actions_and_frozen_evidence() -> None:
    snapshot = population()
    credit = prepare_credit(snapshot, ExternalEstimator())
    with pytest.raises(InvalidPolicyUpdate, match="cover eligible"):
        replace(credit, values=credit.values[:1]).validate(snapshot)
    with pytest.raises(InvalidPolicyUpdate, match="different frozen"):
        credit.validate(replace(snapshot, native_evidence_digest="different"))


@pytest.mark.parametrize("advantage", [float("nan"), float("inf"), True])
def test_credit_rejects_nonfinite_and_boolean_values(advantage: float) -> None:
    with pytest.raises(InvalidPolicyUpdate, match="finite numeric"):
        ActionCredit(population().actions[0].action, advantage)


def test_scorer_quality_cannot_be_relabelled_as_advantage() -> None:
    credit = prepare_credit(population(), ExternalEstimator())
    with pytest.raises(InvalidPolicyUpdate, match="quality scores"):
        replace(credit, meaning="quality")  # type: ignore[arg-type]


def native_rows() -> tuple[PopulationSnapshot, NativeCreditRows]:
    rollouts = tuple(
        replace(
            _rollout(float(index == 0), str(index)),
            reward_evidence=RewardEvidence(
                "prompt",
                f"episode-{index}",
                f"trace-{index}",
                "branch",
                "fixture@1",
                (RewardValue("outcome", "valid", float(index == 0)),),
                ProcessCredit("valid", "assistant-turns@1", f"critique:{index}", ((0, 1),)),
            ),
        )
        for index in range(2)
    )
    coordinates = tuple(
        tuple(
            ActionRef(f"episode-{index}", "branch", "first" if position < 2 else "second", position)
            if eligible
            else None
            for position, eligible in enumerate(rollout.env_mask)
        )
        for index, rollout in enumerate(rollouts)
    )
    actions = tuple(action for row in coordinates for action in row if action is not None)
    snapshot = PopulationSnapshot(
        "native-fixture",
        "native:fixture",
        "native-fixture-digest",
        tuple(ActionRecord(action, "context", "native:node") for action in actions),
        (ConditioningView("context", "native:fixture", "tokens", "attention", "positions", "template@1", "digest", 8),),
        (),
        (PopulationRelation("prompt", "prompt-group", actions, "complete", actions),),
        PolicyVersions("sampler@1", "old@1", "current@1", None),
        "selector@1",
    )
    return snapshot, NativeCreditRows(rollouts, coordinates, ("native-fixture-digest",))


def test_sampo_adapter_preserves_existing_sparse_terminal_credit() -> None:
    snapshot, rows = native_rows()
    credit = prepare_credit(snapshot, SampoCreditEstimator(_settings(), rows, ("prompt",)))
    assert [value.advantage for value in credit.values] == pytest.approx(
        [0.975, 0.975, 1.0, 1.0, -0.975, -0.975, -1.0, -1.0]
    )
    assert credit.normalization == "sampo-mean@1"


def test_sampo_native_adapter_applies_truncation_recipe_before_centering():
    snapshot, rows = native_rows()
    # Equal raw rewards must become unequal algorithm rewards when one native
    # rollout is truncated. Previously this adapter returned all-zero credit.
    rows = replace(
        rows, rollouts=(replace(rows.rollouts[0], reward=1.0), replace(rows.rollouts[1], reward=1.0, is_truncated=True))
    )
    selected = _settings(truncation_penalty=0.5)
    credit = prepare_credit(snapshot, SampoCreditEstimator(selected, rows, ("prompt",)))
    assert [value.advantage for value in credit.values] == pytest.approx(
        [0.4875, 0.4875, 0.5, 0.5, -0.4875, -0.4875, -0.5, -0.5]
    )
    assert rows.rollouts[1].reward == 1.0
    assert credit.estimator_id.startswith("sampo-credit@2:")


@pytest.mark.parametrize("algorithm", ["gdpo", "capo"])
def test_structured_adapter_delegates_without_changing_existing_estimator(algorithm) -> None:
    snapshot, rows = native_rows()
    loop = _settings().loop
    evidence: list[RewardEvidence] = []
    for rollout in rows.rollouts:
        assert rollout.reward_evidence is not None
        evidence.append(rollout.reward_evidence)
    masks = [rollout.env_mask for rollout in rows.rollouts]
    if algorithm == "gdpo":
        settings = GDPOSettings(
            id="fixture-gdpo",
            loop=loop,
            max_prompt_length=2,
            max_completion_length=6,
            component_names=("outcome",),
            component_weights=(1.0,),
        )
        expected = compute_gdpo_advantages(
            evidence, masks, component_names=("outcome",), component_weights=(1.0,), group_size=2
        )
    else:
        settings = CAPOSettings(id="fixture-capo", loop=loop, max_prompt_length=2, max_completion_length=6)
        expected = compute_capo_advantages(evidence, masks, group_size=2)
    credit = prepare_credit(snapshot, StructuredCreditEstimator(settings, rows, ("prompt",)))
    assert [value.advantage for value in credit.values] == pytest.approx(
        [
            value
            for row, mask in zip(expected.token_advantages, masks, strict=True)
            for value, eligible in zip(row, mask, strict=True)
            if eligible
        ]
    )


def test_native_projection_rejects_observation_credit_and_coordinate_loss() -> None:
    _, rows = native_rows()
    with pytest.raises(InvalidPolicyUpdate, match="sampled eligibility"):
        replace(rows, actions=(rows.actions[0][:-1], rows.actions[1]))
    with pytest.raises(InvalidPolicyUpdate, match="ineligible native"):
        rows.project(((1.0,) * 6,) * 2)


def _assignment_selection(channel: str) -> CreditSelection:
    return CreditSelection(
        "assignment-rule-digest", f"assignment-invocation:{channel}", f"assignment-attempt:{channel}", channel
    )


def _assigned_row(rollout, specifications, *, status="complete", missing_count=0, overlap_policy="reject"):
    """Retained bridge shape; negative cases explicitly corrupt fields for admission checks."""
    values = []
    grouped = {}
    for index, specification in enumerate(specifications):
        channel = specification.get("channel", "progress")
        allocation = specification.get("allocation", "turn_boundary")
        assignment_id = f"assignment:{rollout.trace.external_id}:{channel}:{allocation}"
        fields = {
            "assignment_id": assignment_id,
            "contribution_id": f"contribution:{index}",
            "rule_digest": "assignment-rule-digest",
            "attempt_id": f"assignment-attempt:{channel}",
            "channel": "progress",
            "semantics": "progress",
            "branch_digest": "fixture-path-digest",
            "recipient_kind": "turn",
            "recipient_id": f"recipient:{index}",
            "source_snapshot_id": "fixture-source-snapshot",
            "recipient_trace_id": rollout.trace.external_id,
            "status": "valid",
            "alignment": "exact",
            "allocation": "turn_boundary",
            "value": 0.0,
            "weight": 1.0,
            "signal_digest": "progress-signal@1",
            "units": "reward",
            "transformation": "explicit-domain-rule@1",
            "parent_assessment_ids": (f"finding:{index}",),
            "attribution": "coarse",
            "intervals": ((0, 2),),
        }
        fields.update(specification)
        value = AssignedCredit(**fields)
        values.append(value)
        grouped.setdefault((channel, allocation), []).append(value)
    assert len({channel for channel, _ in grouped}) == len(grouped), (
        "fixture selection needs one allocation per channel"
    )
    return replace(
        rollout,
        assigned_credit=AssignedCreditEvidence(
            attempts=tuple(
                CreditAttempt(
                    records[0].assignment_id,
                    "assignment-rule-digest",
                    f"assignment-invocation:{channel}",
                    f"assignment-attempt:{channel}",
                    status,
                    len(records) + missing_count,
                    missing_count,
                    overlap_policy=overlap_policy,
                )
                for (channel, _), records in grouped.items()
            ),
            contributions=tuple(values),
            branch_digest="fixture-path-digest",
        ),
    )


def test_selected_episode_assignments_feed_group_estimator_without_changing_official_reward():
    snapshot, rows = native_rows()
    selected = _assignment_selection("outcome")
    assigned = tuple(
        _assigned_row(
            rollout,
            (
                {
                    "channel": "outcome",
                    "semantics": "outcome",
                    "recipient_kind": "trace",
                    "value": float(index == 1),
                    "alignment": "unsupported",
                    "intervals": (),
                    "signal_digest": "outcome-signal@1",
                    "reason": "explicit_turn_or_span_required",
                },
            ),
        )
        for index, rollout in enumerate(rows.rollouts)
    )
    admitted = tuple(replace(rollout, reward=episode_reward(rollout, selected)) for rollout in assigned)
    settings = GRPOSettings(
        id="assigned-outcome",
        loop=_settings().loop,
        max_prompt_length=2,
        max_completion_length=6,
        advantage_scaling="none",
    )
    credit = prepare_credit(
        snapshot, ScalarGroupCreditEstimator(settings, replace(rows, rollouts=admitted), ("prompt",))
    )
    assert [value.advantage for value in credit.values] == pytest.approx([-0.5] * 4 + [0.5] * 4)
    assert [rollout.reward for rollout in assigned] == [1.0, 0.0]
    assert [rollout.reward for rollout in admitted] == [0.0, 1.0]


def test_selected_turn_assignments_feed_sampo_returns_without_broadcasting_raw_rewards():
    snapshot, rows = native_rows()
    assigned = tuple(
        _assigned_row(
            replace(rollout, reward=0),
            (
                {"value": float(index == 0), "intervals": ((0, 2),)},
                {"value": float(index == 1), "intervals": ((4, 6),)},
            ),
        )
        for index, rollout in enumerate(rows.rollouts)
    )
    admitted = tuple(
        with_turn_rewards(rollout, _assignment_selection("progress"), terminal_outcome="separate")
        for rollout in assigned
    )
    assert [tuple(turn.step_reward for turn in rollout.turns) for rollout in admitted] == [(1.0, 0.0), (0.0, 1.0)]
    assert all(turn.step_reward is None for rollout in assigned for turn in rollout.turns)
    credit = prepare_credit(snapshot, SampoCreditEstimator(_settings(), replace(rows, rollouts=admitted), ("prompt",)))
    # Discounted returns (1, 0) versus (.95, 1), centered within each original anchor.
    assert [value.advantage for value in credit.values] == pytest.approx(
        [0.025, 0.025, -0.5, -0.5, -0.025, -0.025, 0.5, 0.5]
    )


def test_turn_consumer_requires_explicit_terminal_outcome_composition():
    _, rows = native_rows()
    rollout = _assigned_row(
        rows.rollouts[0],
        ({"value": 0.25, "intervals": ((0, 2),)}, {"value": 0, "intervals": ((4, 6),)}),
    )
    selected = _assignment_selection("progress")
    separate = with_turn_rewards(rollout, selected, terminal_outcome="separate")
    combined = with_turn_rewards(rollout, selected, terminal_outcome="include_in_last_turn")
    assert tuple(turn.step_reward for turn in separate.turns) == (0.25, 0)
    assert tuple(turn.step_reward for turn in combined.turns) == (0.25, 1)
    assert separate.reward == combined.reward == rollout.reward == 1
    assert all(turn.step_reward is None for turn in rollout.turns)


def test_local_assignment_allocation_keeps_guard_channel_separate_from_useful_progress():
    _, rows = native_rows()
    rollout = _assigned_row(
        rows.rollouts[0],
        (
            {"value": 2, "allocation": "fixed_mass", "intervals": ((0, 2),)},
            {"value": 6, "allocation": "fixed_mass", "intervals": ((4, 6),)},
            {
                "channel": "guard",
                "semantics": "cost",
                "signal_digest": "guard-cost@1",
                "value": -1,
                "weight": 0.5,
                "allocation": "broadcast",
                "intervals": ((4, 6),),
            },
        ),
    )
    assert local_token_rewards(rollout, _assignment_selection("progress")) == (1, 1, 0, 0, 3, 3)
    assert local_token_rewards(rollout, _assignment_selection("guard")) == (0, 0, 0, 0, -0.5, -0.5)
    assert rollout.reward == 1.0
    assert all(turn.step_reward is None for turn in rollout.turns)
    assert rollout.assigned_credit.contributions[0].value == 2  # Fixed mass is allocated once, not rewritten.


def test_local_assignment_channels_cross_explicit_estimator_seam_with_original_action_ids():
    snapshot, rows = native_rows()
    assigned = tuple(
        _assigned_row(
            rollout,
            (
                {"value": 2 if index == 0 else 0, "allocation": "fixed_mass", "intervals": ((0, 2),)},
                {"value": 6 if index == 0 else 0, "allocation": "fixed_mass", "intervals": ((4, 6),)},
                {
                    "channel": "guard",
                    "semantics": "cost",
                    "signal_digest": "guard-cost@1",
                    "value": -1 if index == 0 else 0,
                    "weight": 0.5,
                    "allocation": "broadcast",
                    "intervals": ((4, 6),),
                },
            ),
        )
        for index, rollout in enumerate(rows.rollouts)
    )
    native = replace(rows, rollouts=assigned)
    progress = tuple(local_token_rewards(rollout, _assignment_selection("progress")) for rollout in assigned)
    guards = tuple(local_token_rewards(rollout, _assignment_selection("guard")) for rollout in assigned)
    assert progress == ((1, 1, 0, 0, 3, 3), (0, 0, 0, 0, 0, 0))
    assert guards == ((0, 0, 0, 0, -0.5, -0.5), (0, 0, 0, 0, 0, 0))

    class AssignmentTransportFixture:
        """Named contract fixture: center progress, retain a separate guard cost.

        This exercises the external estimator seam, not a proposed learning recipe.
        """

        id = "assignment-local-transport-fixture@1"
        required_relations = ("prompt",)

        def prepare(self, population: PopulationSnapshot) -> PreparedCredit:
            centered = tuple(
                tuple(
                    value - sum(column) / len(column)
                    for value, column in zip(row, zip(*progress, strict=True), strict=True)
                )
                for row in progress
            )
            advantages = tuple(
                tuple(value + guard for value, guard in zip(row, cost, strict=True))
                for row, cost in zip(centered, guards, strict=True)
            )
            assert all(
                value == 0
                for row, mask in zip(advantages, (item.env_mask for item in assigned), strict=True)
                for value, eligible in zip(row, mask, strict=True)
                if not eligible
            )
            return PreparedCredit(
                population.digest,
                self.id,
                native.project(advantages),
                self.required_relations,
                (("centered-progress", 1.0), ("guard-cost", 1.0)),
                "fixture-center-progress-plus-guard@1",
                "full-trajectory",
                rows.evidence_digests + tuple(record_digest(item.assigned_credit) for item in assigned),
            )

    credit = prepare_credit(snapshot, AssignmentTransportFixture())
    assert [item.advantage for item in credit.values] == pytest.approx([0.5, 0.5, 1, 1, -0.5, -0.5, -1.5, -1.5])
    assert tuple(item.action for item in credit.values) == tuple(record.action for record in snapshot.actions)
    assert credit.meaning == "detached-advantage"
    assert credit.component_weights == (("centered-progress", 1.0), ("guard-cost", 1.0))
    assert progress[0] != (0.5, 0.5, 0, 0, 1, 1)  # Assignment values remain distinct from prepared advantages.


@pytest.mark.parametrize("kind", ["absent-channel", "incomplete-attempt", "missing-turn", "partial-turn", "overlap"])
def test_turn_assignment_consumer_rejects_missing_or_ambiguous_credit(kind):
    _, rows = native_rows()
    specs = [{"value": 1, "intervals": ((0, 2),)}, {"value": 0, "intervals": ((4, 6),)}]
    status, missing_count = "complete", 0
    if kind == "missing-turn":
        specs = specs[:1]
    elif kind == "partial-turn":
        specs[0]["intervals"] = ((0, 1),)
        specs[0]["recipient_kind"] = "span"
    elif kind == "overlap":
        specs[1]["intervals"] = ((0, 2),)
    elif kind == "incomplete-attempt":
        status, missing_count = "partial", 1
    rollout = _assigned_row(rows.rollouts[0], specs, status=status, missing_count=missing_count)
    selected = _assignment_selection("missing" if kind == "absent-channel" else "progress")
    with pytest.raises(InvalidRewardEvidence):
        with_turn_rewards(rollout, selected, terminal_outcome="separate")
    assert all(turn.step_reward is None for turn in rollout.turns)


@pytest.mark.parametrize("kind", ["probability", "unavailable", "unaligned", "boundary-allocation"])
def test_local_consumer_rejects_nonreward_or_unavailable_assignments(kind):
    _, rows = native_rows()
    spec = {"value": 0.9, "allocation": "broadcast"}
    if kind == "probability":
        spec["semantics"] = "probability"
    elif kind == "unavailable":
        spec.update(status="unavailable", value=None, alignment="unsupported", intervals=(), reason="judge_failed")
    elif kind == "unaligned":
        spec.update(alignment="failed", intervals=(), reason="rewritten_source")
    else:
        spec["allocation"] = "turn_boundary"
    rollout = _assigned_row(rows.rollouts[0], (spec,))
    with pytest.raises(InvalidRewardEvidence):
        local_token_rewards(rollout, _assignment_selection("progress"))


def test_episode_consumer_rejects_outcome_from_another_native_trace():
    _, rows = native_rows()
    rollout = _assigned_row(
        rows.rollouts[0],
        (
            {
                "channel": "outcome",
                "semantics": "outcome",
                "recipient_kind": "trace",
                "recipient_trace_id": "another-trace",
                "alignment": "unsupported",
                "intervals": (),
                "reason": "explicit_turn_or_span_required",
            },
        ),
    )
    with pytest.raises(InvalidRewardEvidence, match="another trace"):
        episode_reward(rollout, _assignment_selection("outcome"))


@pytest.mark.parametrize("consumer", [episode_reward, with_turn_rewards, local_token_rewards])
def test_assignment_consumers_never_turn_absent_native_evidence_into_zero(consumer):
    _, rows = native_rows()
    with pytest.raises(InvalidRewardEvidence, match="absent or ambiguous"):
        if consumer is with_turn_rewards:
            consumer(rows.rollouts[0], _assignment_selection("outcome"), terminal_outcome="separate")
        else:
            consumer(rows.rollouts[0], _assignment_selection("outcome"))


@pytest.mark.parametrize("consumer", [with_turn_rewards, local_token_rewards])
def test_local_consumers_reject_credit_copied_from_another_trace(consumer):
    _, rows = native_rows()
    rollout = _assigned_row(
        rows.rollouts[0],
        ({"value": 1, "recipient_trace_id": "other-trace", "allocation": "broadcast"},),
    )
    with pytest.raises(InvalidRewardEvidence, match="another trace"):
        if consumer is with_turn_rewards:
            consumer(rollout, _assignment_selection("progress"), terminal_outcome="separate")
        else:
            consumer(rollout, _assignment_selection("progress"))


def test_token_overlap_is_combined_only_when_receipt_declares_sum():
    _, rows = native_rows()
    specs = (
        {"value": 2, "allocation": "fixed_mass", "intervals": ((0, 2),)},
        {"value": 1, "allocation": "fixed_mass", "intervals": ((1, 2),), "recipient_kind": "span"},
    )
    rejected = _assigned_row(rows.rollouts[0], specs)
    with pytest.raises(InvalidRewardEvidence, match="explicit sum policy"):
        local_token_rewards(rejected, _assignment_selection("progress"))
    summed = _assigned_row(rows.rollouts[0], specs, overlap_policy="sum")
    assert local_token_rewards(summed, _assignment_selection("progress")) == (1, 2, 0, 0, 0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap_policy", ["reject", "sum"])
async def test_native_retry_execution_credit_overlap_preserves_domain_provenance(overlap_policy):
    """Host-linked retries share a test-qualified original call span, not identities."""
    vf = pytest.importorskip("verifiers.v1")
    if not hasattr(vf, "resolve_execution_parent"):
        pytest.skip("requires native execution-parent candidate checkout")
    from posttrain.common import TraceObservation
    from posttrain.train.integrations.verifiers import _project_training_branch
    from posttrain.train.integrations.verifiers_credit import align_native_credit
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.interception.tool import MCPDispatch, ToolHookRequest
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.types import generated_completion_digest

    call = vf.ToolCall(id="provider-call", name="counter_bump", arguments="{}")
    tokens = [101, 102, 103, 104, 105]
    revision = "fixture-original-execution-span@1"
    trace = vf.Trace(
        episode_id="consumer-overlap",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="bump")),
        nodes=[
            vf.MessageNode(message=vf.UserMessage(content="bump"), token_ids=[1], mask=[False]),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="", tool_calls=[call]),
                sampled=True,
                token_ids=tokens,
                mask=[False, True, True, True, True],
                generated_call_producer=vf.GeneratedCallProducer.capture(revision, {"kind": "test_fixture"}),
                generated_calls=(
                    vf.GeneratedCallAttempt(
                        attempt_index=0,
                        emitted_call_index=0,
                        provider_call_id=call.id,
                        parse_status="parsed",
                        raw="counter_bump({})",
                        name=call.name,
                        arguments="{}",
                        coordinate_system="node_local_full_tokens",
                        parser_revision=revision,
                        token_span=(2, 4),
                        span_fidelity="exact",
                        completion_token_digest=generated_completion_digest(tokens[1:]),
                    ),
                ),
            ),
        ],
        calls=[vf.ModelCall(node=1, finish_reason="tool_calls")],
    )
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    message = vf.ToolMessage(tool_call_id=call.id, name=call.name, content="")
    for index, phase in enumerate(("before", "dispatch")):
        decision = await session.handle_tool(
            phase,
            message,
            request=ToolHookRequest(
                phase=phase,
                message=message,
                execution_id="parent",
                call=call,
                event_index=index,
                mcp_dispatch=MCPDispatch(server_name="counter", tool_name="bump", arguments_json="{}")
                if phase == "dispatch"
                else None,
            ),
        )
    for attempt in range(2):
        receipt = vf.ToolServerReceipt(
            invocation_id=f"physical-{attempt}",
            event_index=0,
            phase="dispatch",
            tool_name="bump",
            arguments_json='{"args":[],"kwargs":{}}',
            parent_execution_id="parent",
            dispatch_ticket=decision["mcp_dispatch_ticket"],
            transport_attempt_index=attempt,
            server_name="counter",
        )
        session.retain_tool_server_receipt(receipt)
        session.retain_tool_server_receipt(
            receipt.model_copy(
                update={
                    "event_index": 1,
                    "phase": "returned",
                    "result_json": '"observed"',
                }
            )
        )
    await session.handle_tool(
        "after",
        message,
        request=ToolHookRequest(
            phase="after",
            message=message,
            execution_id="parent",
            call=call,
            event_index=2,
            raw_result="observed",
        ),
    )
    source = capture_trace_source(trace)
    recipients = tuple(
        vf.SubjectRef(
            kind="execution",
            snapshot_id=source.snapshot_id,
            episode_id=source.episode_id,
            trace_id=trace.id,
            execution=execution,
        )
        for execution in source.executions
        if execution.invocation_id.startswith("physical-")
    )
    assert len(recipients) == 2 and len({recipient.subject_id for recipient in recipients}) == 2
    assert {vf.resolve_execution_parent(source, recipient.execution).invocation_id for recipient in recipients} == {
        "parent"
    }
    # Signed values are already domain rewards; this consumer does not derive their policy meaning.
    signal = vf.SignalDefinition(
        signal_id="signed-progress",
        revision="1",
        semantics="progress",
        description="signed verified change",
        units="reward",
        minimum=-1,
        maximum=1,
    )
    parents = tuple(
        vf.Assessment(
            assessment_id=f"finding-{index}",
            run_id="verified-effects",
            subject=recipient,
            view_id="effect-view",
            signal=signal,
            status="valid",
            value=value,
        )
        for index, (recipient, value) in enumerate(zip(recipients, (1, -1), strict=True))
    )
    request = vf.CreditRequest(
        source=source,
        invocation_id="assign",
        attempt_id="attempt",
        rule=vf.CreditRule(rule_id="signed-effects", revision="1"),
        accepted=parents,
        targets=tuple(vf.CreditTarget(recipient=recipient, channel="progress") for recipient in recipients),
        allocation="fixed_mass",
        overlap_policy=overlap_policy,
    )
    assignment = vf.CreditAssignment(
        request=request,
        contributions=tuple(
            vf.CreditContribution(
                contribution_id=f"effect-{index}",
                parent_assessment_ids=(parent.assessment_id,),
                recipient=parent.subject,
                channel="progress",
                signal=signal,
                transformation="identity",
                status="valid",
                value=parent.value,
                allocation="fixed_mass",
                attribution="exact",
            )
            for index, parent in enumerate(parents)
        ),
    )
    alignment = vf.project_assignment(assignment, trace)
    assert [item.projection.status for item in alignment.contributions] == ["exact_call", "exact_call"]
    assert alignment.contributions[0].projection.intervals == alignment.contributions[1].projection.intervals
    restored = vf.WireTrace.model_validate(trace.to_record())
    aligned = align_native_credit(restored, _project_training_branch(restored), 2, assignments=(assignment,))
    assert [item.intervals for item in aligned.contributions] == [((1, 3),), ((1, 3),)]
    assert [item.value for item in aligned.contributions] == [1, -1]
    assert [item.parent_assessment_ids for item in aligned.contributions] == [("finding-0",), ("finding-1",)]
    assert {item.assignment_id for item in aligned.contributions} == {assignment.assignment_id}
    assert {item.recipient_id for item in aligned.contributions} == {item.subject_id for item in recipients}
    rollout = replace(
        _rollout(0.75, "overlap"),
        completion_ids=tuple(tokens[1:]),
        env_mask=(True,) * 4,
        sampling_logprobs=(-0.1,) * 4,
        turns=(),
        trace=TraceObservation("verifiers.rollout", trace.id, {}),
        assigned_credit=aligned,
    )
    selected = CreditSelection(request.rule.digest, "assign", "attempt", "progress")
    if overlap_policy == "reject":
        with pytest.raises(InvalidRewardEvidence, match="explicit sum policy"):
            local_token_rewards(rollout, selected)
    else:
        assert local_token_rewards(rollout, selected) == (0, 0, 0, 0)
        assert [item.token_values(rollout.env_mask) for item in aligned.contributions] == [
            (0, 0.5, 0.5, 0),
            (0, -0.5, -0.5, 0),
        ]
    assert rollout.reward == 0.75 and assignment.contributions[0].value == 1
    assert assignment.contributions[1].value == -1
