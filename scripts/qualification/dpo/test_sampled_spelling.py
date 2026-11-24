"""sampled_spelling.restore: non-string arguments take P9's sampled spelling only when it parses to the same value."""
from sampled_spelling import restore


def row(args):
    return {"messages": [{"role": "system"}, {"role": "user"},
                         {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": args}}]},
                         {"role": "tool", "content": "ok"}, {"role": "assistant", "content": "done"}]}


def action(params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<think>\nx\n</think>\n\n<tool_call>\n<function=f>\n{body}</function>\n</tool_call>"


def test_same_value_takes_sampled_spelling():
    r, n, prob = restore(row({"a": 4200.0, "b": None, "c": True, "d": "text", "e": 0}),
                         [action({"a": "4200.00", "b": "null", "c": "true", "d": "text", "e": "0.0"}), "done"])
    args = r["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert args == {"a": "4200.00", "b": "null", "c": "true", "d": "text", "e": "0.0"} and n == 4 and not prob


def test_different_value_is_left_and_counts_mismatch():
    r, n, prob = restore(row({"a": 4200.0, "b": [1, 2]}), [action({"a": "4300.00", "b": "[1,2]"}), "done"])
    args = r["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert args["a"] == 4200.0 and args["b"] == "[1,2]" and n == 1
    r, n, prob = restore(row({"a": 1}), ["only one"])
    assert prob
