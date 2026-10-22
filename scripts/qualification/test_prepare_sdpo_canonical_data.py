import copy

from prepare_sdpo_guidance20 import canonical_data, digest
from self_feedback_signals import authority_errors


def test_constructor_projection_can_satisfy_original_binding_without_changing_digest():
    data = {"initial_state": {"nested": [1, 2]}, "prompt": []}
    before = copy.deepcopy(data)

    def constructor(state):
        state["rows"] = list(state["nested"])

    canonical, provenance = canonical_data(data, constructor, "actual-pinned-source")
    import hashlib
    import json

    expected = hashlib.sha256(json.dumps(canonical["initial_state"], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    manifest = {"bindings": [{"path": ["task_evidence", "initial"], "canonical_sha256": expected}]}
    manifest_before = copy.deepcopy(manifest)
    assert authority_errors(manifest, data["initial_state"], [])
    assert authority_errors(manifest, canonical["initial_state"], []) == []
    assert manifest == manifest_before and data == before
    assert provenance["raw_initial_state_sha256"] == digest(data["initial_state"])


def test_projection_does_not_waive_a_real_source_mismatch():
    data = {"initial_state": {"actual": "wrong"}, "prompt": []}
    canonical, _ = canonical_data(data, lambda state: None, "actual-pinned-source")
    manifest = {"bindings": [{"path": ["task_evidence", "initial"], "canonical_sha256": "different"}]}
    assert authority_errors(manifest, canonical["initial_state"], [])
