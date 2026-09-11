"""The episode audit rejects prompt drift independently of the scorer."""

import hashlib
import json
from types import SimpleNamespace

import audit_structured_toy as audit
import pytest


@pytest.mark.parametrize("tamper", [None, "prompt", "digest", "response"])
def test_exact_episode_prompt_and_raw_verdict_are_audited(monkeypatch, tamper):
    message = {"role": "assistant", "content": "Done", "reasoning_content": "Checked"}
    node = SimpleNamespace(
        message=SimpleNamespace(role="assistant", model_dump=lambda **kwargs: dict(message)),
        sampled=True,
        mask=[True],
        token_ids=[1],
    )
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=[node])])
    monkeypatch.setattr(audit, "WireTrace", SimpleNamespace(model_validate=lambda value: trace))
    assessments = {"logic": {"status": "valid", "score": 0.75, "reason": "Checked", "evidence": ["message-0"]}}
    request = {"scope": "episode", "trace_id": "t", "trajectory": [{**message, "message_id": "message-0"}]}
    prompt = "Rubric\nASSESSMENT REQUEST:\n" + json.dumps(request, ensure_ascii=False, sort_keys=True)
    attempt = {
        "status": "valid",
        "prompt": prompt,
        "input_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_response": json.dumps({"assessments": assessments}),
    }
    if tamper == "prompt":
        attempt["prompt"] += " altered"
        attempt["input_digest"] = hashlib.sha256(attempt["prompt"].encode()).hexdigest()
    elif tamper == "digest":
        attempt["input_digest"] = "0" * 64
    elif tamper == "response":
        attempt["raw_response"] = '{"assessments": {}}'
    module = SimpleNamespace(
        EPISODE_RUBRICS={"logic": "Check logic"},
        EPISODE_RUBRIC="Rubric",
        EpisodeVerdict=SimpleNamespace(
            model_validate_json=lambda raw: SimpleNamespace(model_dump=lambda **kwargs: json.loads(raw))
        ),
    )
    selection = {
        "episode_rubrics": module.EPISODE_RUBRICS,
        "judge": {"input_budget_tokens": 12288, "sampling": {"max_tokens": 6144}},
        "judge_context_window": 18432,
        "environment": {"parameters": {"max_total_tokens": 8192}},
    }
    record = {
        "rewards": {"task": 1.0},
        "info": {
            "posttrain_episode_reward_attempts": [attempt],
            "posttrain_episode_rewards": {"assessments": assessments},
        },
    }
    # Simulate the recorder changing object key order before audit.
    records = json.loads(json.dumps({"t": record}, sort_keys=True))
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "Checked Done")
    if tamper:
        with pytest.raises(AssertionError):
            audit.episode_mechanics(module, tokenizer, selection, records, ["t"], {"t": 0.7})
    else:
        [row] = audit.episode_mechanics(module, tokenizer, selection, records, ["t"], {"t": 0.7})
        assert row["assessments"] == assessments
        assert row["turns"][0]["reasoning_chars"] == 7
