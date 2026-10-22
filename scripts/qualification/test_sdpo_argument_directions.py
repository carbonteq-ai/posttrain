from audit_sdpo_teacher_directions import argument_schema_directions


class Tokenizer:
    pieces = {1: "</think>", 2: '{"due_date":', 3: '"tomorrow"}'}

    def encode(self, text, **kwargs):
        assert text == "</think>"
        return [1]

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(i, "alternative") for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 8), (8, 20), (20, len(text))]}


def values():
    call = {"tools": [{"function": {"name": "create", "parameters": {"properties": {"dueDate": {}}}}}],
            "message": {"tool_calls": [{"function": {"name": "create", "arguments": '{"due_date":"tomorrow"}'}}]}}
    direction = {"teacher_sampled_advantages": [0., -.4, .1], "actual_adam_logprob_deltas": [0., -.2, .3],
                 "teacher_support_diagnostic": {"teacher_top1_ids": [1, 99, 3]}}
    return {"action_ids": [1, 2, 3]}, call, direction


def test_field_positions_keep_actual_original_token_directions():
    row, call, direction = values()
    result = argument_schema_directions(row, call, direction, Tokenizer())
    cue = result["cues"][0]
    assert result["available"] and cue["field"] == "due_date"
    token = cue["original_argument_tokens"][0]
    assert token["position"] == 1
    assert token["teacher_minus_student_sampled_logp"] == -.4
    assert token["actual_adam_sampled_logp_delta"] == -.2
    assert cue["forbidden_by_schema"] is False


def test_nonroundtripping_decoding_cannot_claim_token_alignment():
    class Wrong(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [4, 5, 6]}

    row, call, direction = values()
    result = argument_schema_directions(row, call, direction, Wrong())
    assert result["available"] is False


def test_shared_field_across_calls_cannot_be_assigned_to_one_tool():
    row, call, direction = values()
    call["message"]["tool_calls"].append({"function": {"name": "other", "arguments": '{"due_date":"tomorrow"}'}})
    result = argument_schema_directions(row, call, direction, Tokenizer())
    cue = result["cues"][0]
    assert cue["token_attribution_available"] is False
    assert cue["original_argument_tokens"] == []


def test_native_xml_parameters_use_exact_action_offsets():
    class XML(Tokenizer):
        pieces = {1: "</think>", 2: "<parameter=due_date>", 3: "tomorrow</parameter>"}

        def __call__(self, text, **kwargs):
            middle = 8 + len(self.pieces[2])
            return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 8), (8, middle), (middle, len(text))]}

    row, call, direction = values()
    cue = argument_schema_directions(row, call, direction, XML())["cues"][0]
    assert cue["token_attribution_available"]
    assert cue["original_argument_tokens"][0]["position"] == 1
