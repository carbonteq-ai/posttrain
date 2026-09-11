"""Replay retained judge messages against a local OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import hashlib
import io
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import httpx
from automationbench_v1.episode_prompt import (
    EPISODE_PROMPT_VERSION,
    EPISODE_RUBRICS,
    GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
    MODEL_NATIVE_VERDICT_REQUEST,
    EpisodeAssessmentFrame,
    EpisodeVerdict,
    EpisodeVocabularyProfile,
    WireEpisodeVerdict,
    model_native_frame_request,
    normalize_wire_verdict,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Optional frozen model-authored evaluator instruction replacing the corpus system message.",
    )
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        help=(
            "Optional frozen model-authored JSON artifact with system_prompt and rubrics. "
            "It replaces both human-authored text surfaces while preserving the wire schema."
        ),
    )
    parser.add_argument(
        "--vocabulary-profile",
        type=Path,
        help=(
            "Optional model-authored vocabulary-profile artifact. It guides the model-native "
            "assessment frame without replacing rubric semantics or the output schema."
        ),
    )
    parser.add_argument(
        "--current-production-protocol",
        action="store_true",
        help=(
            "Rebuild retained evaluator messages with the current production system prompt, "
            "contract version, and explicit zero-based trajectory evidence indexes."
        ),
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Replay only the first N retained cases for a bounded transport preflight.",
    )
    parser.add_argument(
        "--source-input-digests-json",
        help="Optional JSON array selecting reviewed source-input digests from a larger capture.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8123")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--response-format-mode",
        choices=("strict_schema", "json_object"),
        default="strict_schema",
        help="Use provider constrained decoding or a JSON object validated locally against the same schema.",
    )
    parser.add_argument(
        "--frame-response-format-mode",
        choices=("strict_schema", "json_object"),
        help="Optional transport mode for the intermediate assessment frame; defaults to verdict mode.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--omit-top-p", action="store_true")
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--omit-top-k",
        action="store_true",
        help="Do not send vLLM-specific top_k to a generic OpenAI-compatible endpoint.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--api-key-env",
        help="Optional environment variable holding an OpenAI-compatible endpoint credential.",
    )
    parser.add_argument(
        "--extra-body-json",
        help="Optional provider-specific request fields as a JSON object.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--omit-chat-template-kwargs",
        action="store_true",
        help="Do not send vLLM-specific chat_template_kwargs to a generic OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--restatement-first",
        action="store_true",
        help=(
            "Ask the same judge to build a structured, episode-specific assessment frame in its "
            "own vocabulary before scoring. The final verdict consumes that frame unchanged."
        ),
    )
    parser.add_argument("--restatement-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--review-after-verdict",
        action="store_true",
        help="Have the same judge reconcile its frame and provisional verdict before emitting the admitted verdict.",
    )
    parser.add_argument(
        "--retain-content",
        action="store_true",
        help="Retain raw model content in the report for qualitative prompt debugging",
    )
    parser.add_argument(
        "--wire-evidence-indexes",
        action="store_true",
        help="Use the compact judge wire schema and normalize evidence indexes before validation",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        help="Model-specific chat-template kwargs as a JSON object",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    args = parser.parse_args()
    if (
        sum(value is not None for value in (args.system_prompt_file, args.prompt_artifact))
        + int(args.current_production_protocol)
        > 1
    ):
        parser.error("system-prompt-file, prompt-artifact, and current-production-protocol are mutually exclusive")
    if args.concurrency < 1 or args.repetitions < 1 or args.max_tokens < 1 or args.restatement_max_tokens < 1:
        parser.error("concurrency, repetitions, and max tokens must be positive")
    if args.review_after_verdict and not args.restatement_first:
        parser.error("review-after-verdict requires restatement-first")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive when supplied")
    if args.source_input_digests_json is not None:
        try:
            selected = json.loads(args.source_input_digests_json)
        except json.JSONDecodeError as error:
            parser.error(f"invalid source-input digests JSON: {error}")
        if not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected):
            parser.error("source-input digests must be a non-empty JSON string array")
        args.source_input_digests = frozenset(selected)
    else:
        args.source_input_digests = None
    if args.temperature < 0:
        parser.error("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("top-p must be in (0, 1]")
    if args.chat_template_kwargs_json is not None:
        try:
            args.chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
        except json.JSONDecodeError as error:
            parser.error(f"invalid chat-template kwargs JSON: {error}")
        if not isinstance(args.chat_template_kwargs, dict):
            parser.error("chat-template kwargs must be a JSON object")
    else:
        args.chat_template_kwargs = {"enable_thinking": args.enable_thinking}
    if args.extra_body_json is not None:
        try:
            args.extra_body = json.loads(args.extra_body_json)
        except json.JSONDecodeError as error:
            parser.error(f"invalid extra body JSON: {error}")
        if not isinstance(args.extra_body, dict):
            parser.error("extra body must be a JSON object")
    else:
        args.extra_body = {}
    if args.api_key_env is not None:
        args.api_key = os.environ.get(args.api_key_env)
        if not args.api_key:
            parser.error(f"endpoint credential variable {args.api_key_env!r} is not set")
    else:
        args.api_key = None
    return args


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages")
        input_digest = record.get("input_digest")
        if not isinstance(messages, list) or not messages:
            raise ValueError("every replay record must contain a non-empty messages list")
        if not isinstance(input_digest, str) or not input_digest:
            raise ValueError("every replay record must contain an input digest")
        cases.append(
            {
                "input_digest": input_digest,
                "source_input_digest": record.get("source_input_digest", input_digest),
                "case_id": record.get("case_id"),
                "messages": messages,
            }
        )
    if not cases:
        raise ValueError("replay corpus is empty")
    return cases


def _use_current_production_protocol(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rehydrate retained traces under the exact current evidence-index contract."""

    refreshed: list[dict[str, Any]] = []
    for case in cases:
        messages = case["messages"]
        if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
            raise ValueError("replay case must contain exactly the evaluator system and request messages")
        request = json.loads(messages[1]["content"])
        trajectory = request.get("trajectory")
        message_ids = request.get("valid_message_ids")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("replay request requires a non-empty trajectory")
        if not isinstance(message_ids, list) or any(not isinstance(value, str) or not value for value in message_ids):
            raise ValueError("replay request requires valid message IDs")
        observed_ids = [entry.get("message_id") for entry in trajectory if isinstance(entry, dict)]
        if observed_ids != message_ids or len(observed_ids) != len(trajectory):
            raise ValueError("replay trajectory must match its valid message IDs exactly")
        indexed = [{**entry, "evidence_index": index} for index, entry in enumerate(trajectory)]
        refreshed_request = {**request, "contract": EPISODE_PROMPT_VERSION, "trajectory": indexed}
        content = json.dumps(refreshed_request, ensure_ascii=False, sort_keys=True)
        refreshed.append(
            {
                **case,
                "input_digest": hashlib.sha256(content.encode()).hexdigest(),
                "messages": [
                    {"role": "system", "content": GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            }
        )
    return refreshed


def _replace_system_prompt(cases: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    if not prompt.strip():
        raise ValueError("model-authored system prompt must not be empty")
    replaced = []
    for case in cases:
        messages = case["messages"]
        if not messages or messages[0].get("role") != "system":
            raise ValueError("replay case must begin with the evaluator system message")
        replaced.append({**case, "messages": [{**messages[0], "content": prompt}, *messages[1:]]})
    return replaced


def _replace_prompt_artifact(cases: list[dict[str, Any]], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply frozen model-native wording without changing the verdict schema."""

    prompt = artifact.get("system_prompt")
    rubrics = artifact.get("rubrics")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt artifact requires a non-empty system_prompt")
    if not isinstance(rubrics, dict) or set(rubrics) != set(EPISODE_RUBRICS):
        raise ValueError("prompt artifact must preserve exactly the fixed rubric dimension identifiers")
    if any(not isinstance(value, str) or not value.strip() for value in rubrics.values()):
        raise ValueError("prompt artifact rubric descriptions must be non-empty strings")
    replaced = []
    for case in cases:
        messages = case["messages"]
        if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
            raise ValueError("replay case must begin with evaluator system and user messages")
        try:
            request = json.loads(messages[1]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("replay case user message must contain a JSON judge request") from error
        if not isinstance(request, dict):
            raise ValueError("replay case judge request must be an object")
        request["rubrics"] = rubrics
        replaced.append(
            {
                **case,
                "messages": [
                    {**messages[0], "content": prompt},
                    {**messages[1], "content": json.dumps(request, ensure_ascii=False, sort_keys=True)},
                    *messages[2:],
                ],
            }
        )
    return replaced


async def _async_json_request(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send one request without imposing a deadline of its own."""

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload, headers=headers)
    if response.is_error:
        raise HTTPError(
            url,
            response.status_code,
            response.reason_phrase,
            dict(response.headers),
            io.BytesIO(response.content),
        )
    return response.json()


def _json_request(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send one request with a wall-clock deadline, including response decoding.

    ``urllib`` interprets its timeout as an idle socket timeout: a provider that
    periodically transfers bytes can therefore hold a calibration worker forever.
    ``asyncio.wait_for`` cancels the async client request at the declared deadline,
    which closes the response stream before the caller releases its worker slot.
    """

    try:
        return asyncio.run(
            asyncio.wait_for(
                _async_json_request(url, payload, api_key=api_key),
                timeout=timeout,
            )
        )
    except TimeoutError as error:
        raise TimeoutError(f"request exceeded {timeout:.1f}s wall-clock deadline") from error


def _chat_completions_url(base_url: str) -> str:
    """Support both a server root and an Endpoint OpenAI API root."""

    root = base_url.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


def _chat_completions_urls(base_url: str) -> tuple[str, str]:
    """Return the preferred and alternate OpenAI-compatible chat routes.

    A hosted endpoint proxy may expose either its root or its already-versioned
    OpenAI API root.  Retrying only a 404 is safe: it corrects that unambiguous
    route mismatch without masking authentication, quota, or provider errors.
    """

    preferred = _chat_completions_url(base_url)
    root = base_url.rstrip("/")
    alternate = (
        f"{root}/v1/chat/completions"
        if preferred.endswith("/chat/completions") and not preferred.endswith("/v1/chat/completions")
        else f"{root}/chat/completions"
    )
    return preferred, alternate


def _chat_completion_request(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    api_key: str | None,
) -> dict[str, Any]:
    """POST to an OpenAI-compatible chat endpoint with one route-only retry."""

    preferred, alternate = _chat_completions_urls(base_url)
    try:
        return _json_request(preferred, payload, timeout, api_key=api_key)
    except HTTPError as error:
        if error.code != 404:
            _append_http_error_detail(error)
            raise
        return _json_request(alternate, payload, timeout, api_key=api_key)


def _append_http_error_detail(error: HTTPError) -> None:
    """Add bounded provider diagnostics without retaining credentials or payloads."""

    headers = getattr(error, "headers", None)
    response_source = None
    request_id = None
    if headers is not None:
        response_source = headers.get("server")
        request_id = headers.get("x-request-id") or headers.get("x-openrouter-request-id")
    try:
        raw = error.read(512)
        decoded = raw.decode("utf-8", errors="replace")
        parsed = json.loads(decoded)
        if isinstance(parsed, dict):
            candidate = parsed.get("error", parsed)
            detail = json.dumps(candidate, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        else:
            detail = decoded
    except (OSError, UnicodeError, json.JSONDecodeError):
        detail = ""
    metadata = ", ".join(
        value
        for value in (
            f"server={response_source}" if response_source else None,
            f"request_id={request_id}" if request_id else None,
        )
        if value
    )
    detail = detail.strip().replace("\n", " ")[:300]
    if metadata:
        detail = f"{detail}; {metadata}" if detail else metadata
    if detail:
        error.msg = f"{error.msg}: {detail}"


_ASSESSMENT_FRAME_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "episode_understanding",
        "success_conditions",
        "requirement_observations",
        "observed_state",
        "claims_to_verify",
        "open_discrepancies",
        "observed_defects",
        "perfection_blockers",
        "dimension_language",
    ],
    "properties": {
        "episode_understanding": {"type": "string", "minLength": 24, "maxLength": 800},
        "success_conditions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "minItems": 1,
            "maxItems": 8,
        },
        "requirement_observations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["requirement", "requested", "attempted", "observed", "discrepancy"],
                "properties": {
                    name: {"type": "string", "minLength": 1, "maxLength": 240}
                    for name in ("requirement", "requested", "attempted", "observed", "discrepancy")
                },
            },
        },
        "observed_state": {
            "type": "array",
            "items": {"type": "string", "maxLength": 280},
            "minItems": 1,
            "maxItems": 12,
        },
        "claims_to_verify": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 8},
        "open_discrepancies": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 8},
        "observed_defects": {
            "type": "array",
            "items": {"type": "string", "minLength": 8, "maxLength": 300},
            "maxItems": 8,
        },
        "perfection_blockers": {
            "type": "object",
            "additionalProperties": False,
            "required": list(EPISODE_RUBRICS),
            "properties": {
                name: {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 4}
                for name in EPISODE_RUBRICS
            },
        },
        "dimension_language": {
            "type": "object",
            "additionalProperties": False,
            "required": list(EPISODE_RUBRICS),
            "properties": {name: {"type": "string", "minLength": 8, "maxLength": 160} for name in EPISODE_RUBRICS},
        },
    },
}


def _choice_content(response: dict[str, Any]) -> tuple[str, str, str]:
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return str(choice.get("finish_reason") or ""), content, reasoning if isinstance(reasoning, str) else ""


def _sampling_payload(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    omit_temperature: bool,
    omit_top_p: bool,
    omit_top_k: bool,
    chat_template_kwargs: dict[str, Any],
    omit_chat_template_kwargs: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not omit_temperature:
        payload["temperature"] = temperature
    if not omit_top_p:
        payload["top_p"] = top_p
    if not omit_top_k:
        payload["top_k"] = top_k
    if not omit_chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload


def _response_format(name: str, schema: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": _inline_local_json_refs(schema), "strict": True},
    }


def _inline_local_json_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic-local `$defs` for strict providers that reject `$ref`.

    The resulting schema has identical semantics for our non-recursive wire
    models. Validation still occurs against the canonical Pydantic model after
    generation; this only makes constrained decoding portable.
    """

    root = copy.deepcopy(schema)
    definitions = root.pop("$defs", {})
    if not isinstance(definitions, dict):
        return root

    def expand(value: Any, ancestors: frozenset[str] = frozenset()) -> Any:
        if isinstance(value, list):
            return [expand(item, ancestors) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.removeprefix("#/$defs/")
            definition = definitions.get(name)
            if isinstance(definition, dict) and name not in ancestors:
                expanded = expand(definition, ancestors | {name})
                siblings = {key: expand(item, ancestors) for key, item in value.items() if key != "$ref"}
                return {**expanded, **siblings}
        return {key: expand(item, ancestors) for key, item in value.items()}

    return expand(root)


def _complete(
    *,
    base_url: str,
    model: str,
    case: dict[str, Any],
    max_tokens: int,
    timeout: float,
    chat_template_kwargs: dict[str, Any],
    temperature: float,
    top_p: float,
    top_k: int,
    omit_temperature: bool,
    omit_top_p: bool,
    wire_evidence_indexes: bool,
    retain_content: bool,
    restatement_first: bool,
    restatement_max_tokens: int,
    review_after_verdict: bool,
    api_key: str | None,
    extra_body: dict[str, Any],
    omit_top_k: bool,
    omit_chat_template_kwargs: bool,
    response_format_mode: str,
    frame_response_format_mode: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    schema = WireEpisodeVerdict if wire_evidence_indexes else EpisodeVerdict
    assessment_frame = None
    assessment_frame_usage: dict[str, Any] = {}
    assessment_frame_finish_reason: str | None = None
    assessment_frame_content: str | None = None
    assessment_frame_validation_error: str | None = None
    messages = case["messages"]
    raw_vocabulary_profile = case.get("vocabulary_profile")
    vocabulary_profile = None
    if raw_vocabulary_profile is not None:
        vocabulary_profile = EpisodeVocabularyProfile.model_validate(raw_vocabulary_profile)
    if restatement_first:
        first = _chat_completion_request(
            base_url,
            {
                **extra_body,
                "model": model,
                "messages": messages
                + [
                    {
                        "role": "user",
                        "content": model_native_frame_request(vocabulary_profile),
                    }
                ],
                **_sampling_payload(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    omit_temperature=omit_temperature,
                    omit_top_p=omit_top_p,
                    omit_top_k=omit_top_k,
                    chat_template_kwargs=chat_template_kwargs,
                    omit_chat_template_kwargs=omit_chat_template_kwargs,
                ),
                "max_tokens": restatement_max_tokens,
                "response_format": _response_format(
                    "EpisodeAssessmentFrame",
                    _ASSESSMENT_FRAME_SCHEMA,
                    frame_response_format_mode or response_format_mode,
                ),
            },
            timeout,
            api_key=api_key,
        )
        first_finish, first_content, _ = _choice_content(first)
        assessment_frame_finish_reason = first_finish
        assessment_frame_content = first_content
        try:
            raw_assessment_frame = json.loads(first_content) if first_finish == "stop" else None
            assessment_frame = EpisodeAssessmentFrame.model_validate(raw_assessment_frame).model_dump(mode="json")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            assessment_frame = None
            assessment_frame_validation_error = str(error)
        assessment_frame_usage = first.get("usage") if isinstance(first.get("usage"), dict) else {}
        if assessment_frame is None:
            # A model-native verdict is meaningful only if the judge supplied
            # the vocabulary it is supposed to use.  Do not fall back to the
            # direct prompt, but retain the first response so a qualification
            # failure remains diagnosable rather than becoming an opaque
            # transport-style error.
            elapsed = time.perf_counter() - started
            result: dict[str, Any] = {
                "case_id": case["case_id"],
                "input_digest": case["input_digest"],
                "source_input_digest": case["source_input_digest"],
                "elapsed_seconds": elapsed,
                "prompt_tokens": int(assessment_frame_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(assessment_frame_usage.get("completion_tokens") or 0),
                "finish_reason": "invalid_assessment_frame",
                "content_sha256": None,
                "reasoning_sha256": None,
                "content_characters": 0,
                "reasoning_characters": 0,
                "structured_output_valid": False,
                "assessment_frame": None,
                "assessment_frame_finish_reason": assessment_frame_finish_reason,
                "assessment_frame_validation_error": assessment_frame_validation_error,
                "verdict": None,
                "scores": None,
            }
            if retain_content:
                result["content"] = None
                result["reasoning"] = None
                result["assessment_frame_content"] = assessment_frame_content
            return result
        messages = messages + [
            {"role": "assistant", "content": first_content},
            {
                "role": "user",
                "content": MODEL_NATIVE_VERDICT_REQUEST,
            },
        ]
    response = _chat_completion_request(
        base_url,
        {
            **extra_body,
            "model": model,
            "messages": messages,
            **_sampling_payload(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                omit_temperature=omit_temperature,
                omit_top_p=omit_top_p,
                omit_top_k=omit_top_k,
                chat_template_kwargs=chat_template_kwargs,
                omit_chat_template_kwargs=omit_chat_template_kwargs,
            ),
            "max_tokens": max_tokens,
            "response_format": _response_format(schema.__name__, schema.model_json_schema(), response_format_mode),
        },
        timeout,
        api_key=api_key,
    )
    provisional_content: str | None = None
    provisional_usage: dict[str, Any] = {}
    if review_after_verdict:
        provisional_finish, provisional_content, _ = _choice_content(response)
        try:
            if provisional_finish != "stop":
                raise ValueError("provisional verdict did not stop normally")
            if wire_evidence_indexes:
                WireEpisodeVerdict.model_validate_json(provisional_content)
            else:
                EpisodeVerdict.model_validate_json(provisional_content)
        except ValueError:
            provisional_content = None
        if provisional_content is not None:
            provisional_usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            response = _chat_completion_request(
                base_url,
                {
                    **extra_body,
                    "model": model,
                    "messages": messages
                    + [
                        {"role": "assistant", "content": provisional_content},
                        {
                            "role": "user",
                            "content": (
                                "Review your provisional verdict against the original trajectory and the assessment "
                                "frame you authored. Use the frame's own vocabulary and return a complete replacement "
                                "wire verdict. Reconstruct every material request; do not omit requirements because an "
                                "action was attempted. For each open discrepancy, observed defect, perfection blocker, "
                                "or unresolved requirement observation, make each relevant requirement and assessment "
                                "consistent. A score of 1 needs affirmative evidence for the whole dimension and cannot "
                                "coexist with a relevant unresolved blocker. Do not invent facts, tool contracts, or a "
                                "critique; return only the replacement structured object."
                            ),
                        },
                    ],
                    **_sampling_payload(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        omit_temperature=omit_temperature,
                        omit_top_p=omit_top_p,
                        omit_top_k=omit_top_k,
                        chat_template_kwargs=chat_template_kwargs,
                        omit_chat_template_kwargs=omit_chat_template_kwargs,
                    ),
                    "max_tokens": max_tokens,
                    "response_format": _response_format(
                        schema.__name__, schema.model_json_schema(), response_format_mode
                    ),
                },
                timeout,
                api_key=api_key,
            )
    elapsed = time.perf_counter() - started
    usage = response.get("usage") or {}
    finish_reason, content, reasoning = _choice_content(response)
    verdict = None
    verdict_validation_error: str | None = None
    try:
        if wire_evidence_indexes:
            wire = WireEpisodeVerdict.model_validate_json(content)
            request = json.loads(case["messages"][1]["content"])
            verdict = normalize_wire_verdict(wire, request["valid_message_ids"])
        else:
            verdict = EpisodeVerdict.model_validate_json(content)
    except ValueError as error:
        verdict_validation_error = str(error)
    result = {
        "case_id": case["case_id"],
        "input_digest": case["input_digest"],
        "source_input_digest": case["source_input_digest"],
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0)
        + int(provisional_usage.get("prompt_tokens") or 0)
        + int(assessment_frame_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0)
        + int(provisional_usage.get("completion_tokens") or 0)
        + int(assessment_frame_usage.get("completion_tokens") or 0),
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "content_characters": len(content),
        "reasoning_characters": len(reasoning),
        "structured_output_valid": verdict is not None and (not restatement_first or assessment_frame is not None),
        "assessment_frame": assessment_frame,
        "assessment_frame_finish_reason": assessment_frame_finish_reason,
        "assessment_frame_validation_error": assessment_frame_validation_error,
        "verdict_validation_error": verdict_validation_error,
        "verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        "scores": (
            {name: assessment.score for name, assessment in verdict.assessments.items()}
            if verdict is not None
            else None
        ),
    }
    if retain_content:
        result["content"] = content
        result["reasoning"] = reasoning
        result["assessment_frame_content"] = assessment_frame_content
        result["provisional_content"] = provisional_content
    return result


def _failed_completion(case: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """Keep calibration evidence complete when one provider request fails."""

    return {
        "case_id": case["case_id"],
        "input_digest": case["input_digest"],
        "source_input_digest": case["source_input_digest"],
        "elapsed_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "finish_reason": "error",
        "content_sha256": None,
        "reasoning_sha256": None,
        "content_characters": 0,
        "reasoning_characters": 0,
        "structured_output_valid": False,
        "assessment_frame": None,
        "assessment_frame_finish_reason": None,
        "verdict": None,
        "scores": None,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main() -> None:
    args = _arguments()
    corpus = _load_cases(args.inputs)
    system_prompt_digest = None
    prompt_artifact_digest = None
    vocabulary_profile_digest = None
    if args.current_production_protocol:
        corpus = _use_current_production_protocol(corpus)
    if args.system_prompt_file is not None:
        prompt = args.system_prompt_file.read_text()
        corpus = _replace_system_prompt(corpus, prompt)
        system_prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
    if args.prompt_artifact is not None:
        artifact = json.loads(args.prompt_artifact.read_text())
        if not isinstance(artifact, dict):
            raise ValueError("prompt artifact must be a JSON object")
        corpus = _replace_prompt_artifact(corpus, artifact)
        prompt_artifact_digest = hashlib.sha256(
            json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    if args.vocabulary_profile is not None:
        artifact = json.loads(args.vocabulary_profile.read_text())
        if not isinstance(artifact, dict) or artifact.get("kind") != "episode-judge-vocabulary-profile":
            raise ValueError("vocabulary profile must be an episode-judge-vocabulary-profile artifact")
        profile = EpisodeVocabularyProfile.model_validate(artifact.get("profile"))
        corpus = [{**case, "vocabulary_profile": profile.model_dump(mode="json")} for case in corpus]
        vocabulary_profile_digest = hashlib.sha256(
            json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    if args.source_input_digests is not None:
        corpus = [case for case in corpus if case["source_input_digest"] in args.source_input_digests]
        if len(corpus) != len(args.source_input_digests):
            found = {case["source_input_digest"] for case in corpus}
            missing = sorted(args.source_input_digests - found)
            raise ValueError(f"reviewed source-input digests missing from replay capture: {missing}")
    if args.case_limit is not None:
        corpus = corpus[: args.case_limit]
    requests = corpus * args.repetitions
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda case: _run_one(case, args),
                requests,
            )
        )
    wall_time = time.perf_counter() - started
    completion_tokens = sum(result["completion_tokens"] for result in results)
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    latencies = [result["elapsed_seconds"] for result in results]
    report = {
        "schema_version": 1,
        "label": args.label,
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime_identity": args.runtime_identity,
        "requests": len(results),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "response_format_mode": args.response_format_mode,
        "frame_response_format_mode": args.frame_response_format_mode or args.response_format_mode,
        "current_production_protocol": args.current_production_protocol,
        "system_prompt_file": str(args.system_prompt_file.resolve()) if args.system_prompt_file else None,
        "system_prompt_sha256": system_prompt_digest,
        "prompt_artifact": str(args.prompt_artifact.resolve()) if args.prompt_artifact else None,
        "prompt_artifact_sha256": prompt_artifact_digest,
        "vocabulary_profile": str(args.vocabulary_profile.resolve()) if args.vocabulary_profile else None,
        "vocabulary_profile_sha256": vocabulary_profile_digest,
        "enable_thinking": args.chat_template_kwargs.get("enable_thinking"),
        "chat_template_kwargs": args.chat_template_kwargs,
        "extra_body": args.extra_body,
        "wire_evidence_indexes": args.wire_evidence_indexes,
        "restatement_first": args.restatement_first,
        "review_after_verdict": args.review_after_verdict,
        "content_retained": args.retain_content,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "omit_temperature": args.omit_temperature,
            "omit_top_p": args.omit_top_p,
        },
        "wall_time_seconds": wall_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / wall_time,
        "request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "max": max(latencies),
        },
        "finish_reasons": {
            reason: sum(result["finish_reason"] == reason for result in results)
            for reason in sorted({str(result["finish_reason"]) for result in results})
        },
        "structured_output_valid": sum(result["structured_output_valid"] for result in results),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_one(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        return _complete(
            base_url=args.base_url,
            model=args.model,
            case=case,
            max_tokens=args.max_tokens,
            timeout=args.timeout_seconds,
            chat_template_kwargs=args.chat_template_kwargs,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            omit_temperature=args.omit_temperature,
            omit_top_p=args.omit_top_p,
            wire_evidence_indexes=args.wire_evidence_indexes,
            retain_content=args.retain_content,
            restatement_first=args.restatement_first,
            restatement_max_tokens=args.restatement_max_tokens,
            review_after_verdict=args.review_after_verdict,
            api_key=args.api_key,
            extra_body=args.extra_body,
            omit_top_k=args.omit_top_k,
            omit_chat_template_kwargs=args.omit_chat_template_kwargs,
            response_format_mode=args.response_format_mode,
            frame_response_format_mode=args.frame_response_format_mode or args.response_format_mode,
        )
    except Exception as error:  # A calibration cell must retain every failed request.
        result = _failed_completion(case, error)
        result["elapsed_seconds"] = time.perf_counter() - started
        return result


if __name__ == "__main__":
    main()
