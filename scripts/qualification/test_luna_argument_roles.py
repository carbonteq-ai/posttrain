from luna_argument_roles import public_equivalence, public_query, query_structure
from luna_solver_separation import argument_provenance


def test_api_object_is_not_misidentified_as_email_content():
    material = {"user_request": "Update the requested record", "tool_definitions": [],
        "reference_actions": [
            {"call": {"name": "gmail_find_email", "arguments": {}},
             "result": {"observed_result": {"subject": "Contact Update"}}},
            {"call": {"name": "salesforce_find_records", "arguments": {"object_type": "Contact"}},
             "result": {"observed_result": {}}}]}
    old = argument_provenance(material)[1]["argument_provenance"]["object_type"]
    new = argument_provenance(material, schema_roles=True)[1]["argument_provenance"]["object_type"]
    assert old["origin"] == "prior_tool_result"
    assert new["origin"] == "demonstrated_api_option" and new["value"] == "Contact"


def test_query_structure_preserves_operators_without_reference_answers():
    query = "SELECT Id,Name FROM Contact WHERE Email LIKE '%hidden-domain.example' AND Phone = '555-9000'"
    result = query_structure(query)
    assert result["select"] == ["Id", "Name"]
    assert result["object"] == "Contact"
    assert result["boolean_structure"] == "predicate AND predicate"
    assert result["predicates"][0]["operator"] == "LIKE"
    assert "hidden-domain" not in str(result) and "555-9000" not in str(result)


def test_unsupported_query_is_not_passed_through_as_structure():
    assert query_structure("SELECT Id FROM Contact WHERE Name = unquoted_secret") is None


def test_public_word_composition_is_distinct_from_hidden_resource_title():
    assert public_query('from:(Natalie Wong) "project timeline"',
                        "Find the email from Natalie Wong about the project timeline")
    assert not public_query('"Revenue Adjustments Tracker"', "Draft a renewal reply")


def test_public_date_normalization_is_not_hidden_answer_disclosure():
    public = "Create a card due March 5, 2026. First list the board's lists."
    assert public_equivalence("2026-03-05", public) == "public_date_normalization"
    assert public_equivalence("2026-03-02", public) is None
    assert public_equivalence("hidden_00301", public) is None
    assert public_equivalence("Phone Update Processed", "Update the phone and add a note") is None


def test_json_from_public_fields_is_not_hidden_answer_disclosure():
    public = "Create Name='Jordan Lee', Email='jordan@example.com', Status='VIP'."
    assert public_equivalence('{"Name":"Jordan Lee","Email":"jordan@example.com","Status":"VIP"}',
                              public) == "public_json_composition"
    assert public_equivalence('{"Email":"hidden@example.com"}', public) is None
