import copy

import pytest
from sdpo_effect_brief import expression, operational_effect, population_binding


def test_population_id_is_owned_by_contact_record_and_source_is_unchanged():
    check = {"population": "contacts", "role": "harm"}
    sources = {"contacts": {"adapter": "initial.records@1", "fields": {"id": ["id"]}}}
    before = copy.deepcopy(sources)
    binding = operational_effect(check, sources)["population_binding"]
    assert binding["source"] == sources["contacts"]
    assert "current RECORD" in binding["meaning"] and "not an email ID" in binding["meaning"]
    assert sources == before
    with pytest.raises(ValueError, match="unknown check population"):
        population_binding({"population": "invented"}, sources)


def test_unbound_population_is_explicit_unknown_not_email_request():
    assert population_binding({}, {})["declared"] is False


def test_typed_sequence_index_is_preserved_in_field_path():
    assert expression({"kind": "field", "path": ["candidate", "identity", 3]}) == "candidate.identity[3]"


def test_mentions_harm_and_required_before_read_join_are_explicit():
    check = {"role": "harm", "effect_match": {"op": "mentions", "mode": "verbatim",
        "text": {"kind": "field", "path": ["effect", "body"]},
        "value": {"kind": "field", "path": ["request", "rejected_description"]}},
        "effect_joins": [{"source": "policy_reads", "timing": "before"}]}
    result = operational_effect(check)
    assert "effect.body contains request.rejected_description" in result["matched_effect"]
    assert result["dependency_constraints"] == check["effect_joins"]
    assert result["meaning"].startswith("When applicable, prohibit ONLY")


def test_wrong_recipient_guard_does_not_prohibit_requested_message():
    check = {"role": "harm", "prohibited_when": {"op": "eq",
        "left": {"kind": "field", "path": ["request", "request_key"]},
        "right": {"kind": "literal", "value": "internal-request-selector"}},
        "effect_match": {"op": "not", "arg": {"op": "eq",
            "left": {"kind": "field", "path": ["effect", "recipient_user_id"]},
            "right": {"kind": "field", "path": ["request", "recipient_id"]}}}}
    result = operational_effect(check)
    assert result["matched_effect"] == "effect.recipient_user_id differs from request.recipient_id"
    assert "not an API parameter or a prohibition" in result["applicability"]
    assert "prohibit ONLY effects matching" in result["meaning"]
    assert "internal-request-selector" not in str(result)


def test_not_equal_and_boolean_negation_preserve_polarity():
    eq = {"op": "eq", "left": {"kind": "field", "path": ["effect", "id"]},
          "right": {"kind": "field", "path": ["request", "id"]}}
    assert "differs" in expression({"op": "not", "arg": eq})
    assert "equals" in expression({"op": "not", "arg": {"op": "not", "arg": eq}})
    assert " OR " in expression({"op": "all", "args": [eq, eq]}, negate=True)


def test_duplicate_guard_requires_matching_earlier_effect():
    result = operational_effect({"role": "harm", "effect_match": {"op": "eq",
        "left": {"kind": "field", "path": ["effect", "name"]},
        "right": {"kind": "field", "path": ["request", "name"]}},
        "operational_dependencies": [{"source": "earlier_creates", "order": "earlier"}]})
    assert result["dependency_constraints"][0]["source"] == "earlier_creates"
    assert "only together with all declared dependency constraints" in result["dependency_interpretation"]
    assert "does not prohibit the first required action" in result["dependency_interpretation"]
