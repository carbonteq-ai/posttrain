import pytest
from sdpo_guidance_context import digest
from sdpo_guidance_state_binding import bind


def fixture():
    raw, native = {"nested": True}, {"flattened": True}
    return ({"guidance": {"task": {"current_state_sha256": digest(raw), "instructions": ["Read."]}}},
        {"rows": [{"task": "task", "material": {"native_data": {"initial_state": raw}}}]},
        {"qualified": True, "mismatch": [], "native_episodes": 40, "normalizer_sha256": "pinned",
         "normalized_states": {"task": native}, "raw_source_hashes": {"task": digest(raw)},
         "canonical_hashes": {"task": digest(native)}})


def test_only_verified_representation_hash_changes_without_mutating_source():
    guidance, materials, normalization = fixture()
    result = bind(guidance, materials, normalization)
    assert result["guidance"]["task"]["instructions"] == ["Read."]
    assert result["guidance"]["task"]["source_state_sha256"] == guidance["guidance"]["task"]["current_state_sha256"]
    assert result["guidance"]["task"]["current_state_sha256"] == normalization["canonical_hashes"]["task"]
    assert "source_state_sha256" not in guidance["guidance"]["task"]


def test_native_mismatch_cannot_be_rebound():
    guidance, materials, normalization = fixture()
    normalization["mismatch"] = ["task"]
    with pytest.raises(ValueError, match="complete native"):
        bind(guidance, materials, normalization)
