"""Recovery restores credit and resolved occurrences without an estimator."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.train.update_plan import plan_packs
from posttrain.train.update_records import InvalidPolicyUpdate
from posttrain.train.update_transport import decode_population_payload, population_payload

from .test_update_evidence import native_rollout
from .test_update_resolution import native_resolution, settings


def fixture(offset=0):
    resolved = native_resolution((native_rollout("a", 0, 1), native_rollout("a", 1, 0)), settings())
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


def test_typed_reconstruction_keeps_all_original_digests_and_packs():
    expected, payload = fixture(3)
    retained = decode_population_payload(payload)
    assert retained.resolved == expected
    assert (retained.max_overflow_retries, retained.applied_update_offset, retained.attempt_offset) == (2, 3, 4)
    assert [value.digest for value in retained.resolved.updates] == [value.digest for value in expected.updates]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(extra_field="ignored?"),
        lambda x: x.update(schema="future-schema"),
        lambda x: x.update(applied_update_offset=True),
        lambda x: x.update(attempt_offset=-1),
        lambda x: x["credit"]["values"][0].update(advantage="1.0"),
        lambda x: x["credit"]["values"][0]["action"].update(token_index=True),
        lambda x: x["updates"][0].update(digest="changed"),
        lambda x: x["updates"].append(x["updates"][0]),
        lambda x: x["spec"].update(policy_selection={"mode": "unknown", "span_ids": []}),
    ],
)
def test_reconstruction_rejects_malformed_or_changed_records(mutation):
    _, payload = fixture()
    mutation(payload)
    with pytest.raises(InvalidPolicyUpdate, match="retained population"):
        decode_population_payload(payload)


def test_legacy_transport_is_only_supported_at_zero_offsets():
    expected, payload = fixture()
    payload.update(schema="posttrain.resolved-population.v1")
    payload["capabilities"].pop("context_layout")
    payload.pop("applied_update_offset")
    payload.pop("attempt_offset")
    retained = decode_population_payload(payload)
    assert retained.resolved == expected
    assert retained.applied_update_offset == retained.attempt_offset == 0
    payload["applied_update_offset"] = 3
    with pytest.raises(InvalidPolicyUpdate):
        decode_population_payload(payload)


@pytest.mark.parametrize("layout", ["dense-pack", "dense-population"])
def test_v3_layout_transport_preserves_logical_digests_and_physical_packs(layout):
    resolved, _ = fixture()
    capabilities = replace(resolved.capabilities, context_layout=layout)
    expected = replace(
        resolved,
        capabilities=capabilities,
        packs=tuple(plan_packs(update, resolved.execution, capabilities) for update in resolved.updates),
    )
    payload = json.loads(json.dumps(population_payload(expected)))
    assert payload["schema"] == "posttrain.resolved-population.v3"
    retained = decode_population_payload(payload)
    assert retained.resolved == expected
    assert [update.digest for update in retained.resolved.updates] == [update.digest for update in resolved.updates]


def test_v2_transport_preserves_ragged_contract_and_rejects_new_layout_fields():
    expected, payload = fixture()
    payload["schema"] = "posttrain.resolved-population.v2"
    with pytest.raises(InvalidPolicyUpdate, match="retained population"):
        decode_population_payload(payload)
    payload["capabilities"].pop("context_layout")
    assert decode_population_payload(payload).resolved == expected
