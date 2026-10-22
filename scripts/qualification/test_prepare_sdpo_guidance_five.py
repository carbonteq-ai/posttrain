from types import SimpleNamespace

import pytest
from prepare_sdpo_guidance_five import decoding_config, preparation_prompt, preparation_schema


def test_preparation_plain_decoding_is_explicit_and_has_no_draft_head():
    args = SimpleNamespace(speculation="none", mtp_tokens=2, plain_decoding_authorized=False)
    with pytest.raises(ValueError, match="explicit user amendment"):
        decoding_config(args)
    args.plain_decoding_authorized = True
    assert decoding_config(args) is None


class Batch:
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, index):
        assert index == 0
        return self

    def tolist(self):
        return self.ids


class Tokenizer:
    def __init__(self, ids):
        self.ids = ids

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is True
        assert kwargs["enable_thinking"] is False
        return {"input_ids": Batch(self.ids)}


def test_batch_encoding_is_resolved_to_original_integer_ids():
    messages, prefix = preparation_prompt(Tokenizer([1, 2, 3]), "Prepare", {"task": "example"})
    assert prefix == [1, 2, 3]
    assert messages[0]["content"] == "Prepare"


def test_preparation_json_compaction_preserves_all_evidence():
    import json

    data = {"observations": [{"text": "preserve these spaces", "values": [1, None]}]}
    messages, _ = preparation_prompt(Tokenizer([1]), "Prepare", data)
    assert json.loads(messages[1]["content"]) == data
    assert messages[1]["content"] == json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize("ids", [["input_ids"], [], [1] * 30000])
def test_noninteger_empty_or_overlong_prefixes_are_rejected_whole(ids):
    with pytest.raises(ValueError):
        preparation_prompt(Tokenizer(ids), "Prepare", {})


def test_assembly_schema_distinguishes_step_dependencies_from_action_evidence():
    data = {"requirements": {"requirements": [{"check_id": "goal"}]},
            "workflow": {"actions": [{"action_id": "a1"}]}}
    fields = preparation_schema("assembly", data)["properties"]["steps"]["items"]["properties"]
    assert fields["depends_on"]["items"]["pattern"] == "^s[1-9][0-9]*$"
    assert fields["action_ids"]["items"]["enum"] == ["a1"]
