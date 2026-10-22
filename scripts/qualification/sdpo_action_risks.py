"""Factual candidate sampling cues; never tool execution or per-action reward."""

import json


def action_risks(call, responses):
    definitions = {tool["function"]["name"]: tool["function"].get("parameters", {})
                   for tool in call.get("tools", [])}
    result = []
    for tool in call["message"].get("tool_calls") or []:
        function = tool.get("function", tool)
        name = function.get("name")
        schema = definitions.get(name)
        if schema is None:
            continue  # Existing native unknown-tool feedback is its authority.
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                continue
        properties = schema.get("properties")
        if (not isinstance(arguments, dict) or not isinstance(properties, dict)
                or not properties and schema.get("additionalProperties") is not False):
            continue
        undeclared = sorted(set(arguments) - set(properties))
        if undeclared:
            result.append({"kind": "undeclared_arguments", "tool": name,
                           "fields": undeclared,
                           "forbidden_by_schema": schema.get("additionalProperties") is False,
                           "scope": "A declared-schema mismatch cue, not proof of an API error or failed goal."})
    for response in responses:
        content = response.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                continue
        if (isinstance(content, dict) and content.get("success") is not True
                and content.get("error")):
            result.append({"kind": "structured_error_response",
                           "scope": "The observed response reports an error; this is not causal task credit."})
    return result
