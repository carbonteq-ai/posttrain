"""Recovery restores credit and resolved occurrences without an estimator."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.train.profiles import SAMPOSettings, TrainingLoop
from posttrain.train.update_plan import (
    ExecutionCapabilities,
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    PolicyUpdateSettings,
    plan_packs,
)
from posttrain.train.update_records import InvalidPolicyUpdate, PolicyVersions
from posttrain.train.update_resolution import ResolvedPolicyPopulation, resolve_rollout_population
from posttrain.train.update_transport import decode_population_payload, population_payload

from .test_update_evidence import native_rollout


def settings():
    return SAMPOSettings(
        id="transport-test",
        loop=TrainingLoop(max_steps=2, per_device_batch_size=1),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(2, 100, 1000)),
    )


def capabilities():
    return ExecutionCapabilities(
        ("sampo@1", "sampo-spans@1", "grpo@1", "dapo@1"), ("sampled-logp", "old-logp", "reference-logp"), 100, True
    )


def fixture(offset=0):
    resolved = resolve_rollout_population(
        (native_rollout("a", 0, 1), native_rollout("a", 1, 0)),
        settings(),
        capabilities(),
        population_id="admitted",
        native_evidence_ref="native:receipts",
        native_evidence_digest="native-receipt-digest",
        template_revision="native-template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="original-actions@1",
    )
    population = SimpleNamespace(
        updates=resolved.updates,
        credit=resolved.credit,
        spec=resolved.spec,
        execution=resolved.execution,
        capabilities=resolved.capabilities,
        max_overflow_retries=2,
        applied_update_offset=offset,
        attempt_offset=offset + 1,
    )
    return resolved, json.loads(json.dumps(population_payload(population)))


def assert_same(actual: ResolvedPolicyPopulation, expected: ResolvedPolicyPopulation) -> None:
    """Population-sized records compare by digest, not structural equality."""
    assert actual.snapshot.digest == expected.snapshot.digest
    assert actual.credit.digest == expected.credit.digest
    assert actual.spec == expected.spec
    assert [update.digest for update in actual.updates] == [update.digest for update in expected.updates]
    assert [update.objective.digest for update in actual.updates] == [
        update.objective.digest for update in expected.updates
    ]
    assert actual.packs == expected.packs
    assert actual.execution == expected.execution and actual.capabilities == expected.capabilities


def test_typed_reconstruction_keeps_all_original_digests_and_packs():
    expected, payload = fixture(3)
    assert payload["schema"] == "posttrain.resolved-population.v4"
    retained = decode_population_payload(payload)
    assert_same(retained.resolved, expected)
    assert (retained.max_overflow_retries, retained.applied_update_offset, retained.attempt_offset) == (2, 3, 4)
    assert retained.resolved.credit.advantages.tolist() == expected.credit.advantages.tolist()


def _sampled(payload):
    payload["population"]["conditioning"][0]["sampled"][0] = True


def _advantage(payload):
    payload["credit"]["advantages"][0] = "1.0"


# The decoder now re-raises its own specific rejection instead of wrapping
# every InvalidPolicyUpdate in one generic "retained population" message.
@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda x: x.update(extra_field="ignored?"), "retained population fields"),
        (lambda x: x.update(schema="future-schema"), "unsupported retained population schema"),
        (lambda x: x.update(applied_update_offset=True), "retained population offsets"),
        (lambda x: x.update(attempt_offset=-1), "retained population offsets"),
        (_advantage, "retained advantages"),
        (_sampled, "retained population value"),
        (lambda x: x.update(objective_digest="changed"), "retained objective differs"),
        (lambda x: x["updates"][0].update(digest="changed"), "retained occurrence digest"),
        (lambda x: x["updates"].append(x["updates"][0]), "retained population has no or duplicate"),
        (lambda x: x["spec"].update(policy_selection={"mode": "unknown", "span_ids": [], "roles": []}), "selector"),
    ],
)
def test_reconstruction_rejects_malformed_or_changed_records(mutation, message):
    _, payload = fixture()
    mutation(payload)
    with pytest.raises(InvalidPolicyUpdate, match=message):
        decode_population_payload(payload)


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_legacy_per_token_transport_is_rejected(version):
    # Rewritten by design: v1-v3 sidecars from the per-token engine (formerly
    # restorable at zero offsets) are no longer readable by this engine.
    _, payload = fixture()
    payload["schema"] = f"posttrain.resolved-population.{version}"
    with pytest.raises(InvalidPolicyUpdate, match="unsupported retained population schema"):
        decode_population_payload(payload)


@pytest.mark.parametrize("layout", ["dense-pack", "dense-population"])
def test_layout_transport_preserves_logical_digests_and_physical_packs(layout):
    resolved, _ = fixture()
    capabilities = replace(resolved.capabilities, context_layout=layout)
    expected = replace(
        resolved,
        capabilities=capabilities,
        packs=tuple(plan_packs(update, resolved.execution, capabilities) for update in resolved.updates),
    )
    payload = json.loads(json.dumps(population_payload(expected)))
    assert payload["schema"] == "posttrain.resolved-population.v4"
    assert payload["capabilities"]["context_layout"] == layout
    retained = decode_population_payload(payload)
    assert_same(retained.resolved, expected)
    assert [update.digest for update in retained.resolved.updates] == [update.digest for update in resolved.updates]
