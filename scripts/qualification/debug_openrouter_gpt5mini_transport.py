"""Emit a bounded, credential-free diagnostic for one frozen judge request."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from automationbench_v1.episode_prompt import WireEpisodeVerdict
from episode_judge_vllm_benchmark import _response_format


def main() -> None:
    key = os.environ["OPENROUTER_API_KEY"]
    record = json.loads(
        next(line for line in Path("/opt/posttrain/qualification/judge-inputs.jsonl").read_text().splitlines() if line)
    )
    payload = {
        "model": "google/gemini-2.5-flash-lite",
        "messages": record["messages"]
        + [{"role": "user", "content": 'Return exactly one JSON object with {"ready": true}.'}],
        "max_tokens": 4096,
        "response_format": _response_format(
            "WireEpisodeVerdict", WireEpisodeVerdict.model_json_schema(), "strict_schema"
        ),
        "provider": {
            "order": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "zdr": False,
            "data_collection": "allow",
        },
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            print(json.dumps({"status": response.status, "body": response.read(512).decode("utf-8", "replace")[:512]}))
    except urllib.error.HTTPError as error:
        print(json.dumps({"status": error.code, "body": error.read(512).decode("utf-8", "replace")[:512]}))


if __name__ == "__main__":
    main()
