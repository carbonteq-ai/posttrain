"""Materialize and evaluate judge-prompt candidates on current-policy episodes.

This is an offline calibration tool. It never edits the production rubric and
never changes a scorer while a training run is active. A selected candidate
must be promoted explicitly as a new prompt version and scorer digest.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import statistics
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from automationbench_v1.episode_prompt import (
    EPISODE_PROMPT_VERSION,
    EPISODE_RUBRICS,
    GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
    EpisodeVocabularyProfile,
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain non-empty JSON objects")
    return records


def _post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def calibrate_vocabulary_profile(
    *,
    elicitation_system_prompt: str,
    base_url: str,
    model: str,
    model_revision: str,
    runtime_identity: str,
    max_tokens: int,
    timeout: float,
    omit_temperature: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    api_key: str | None = None,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Capture a model's preferred generic judge vocabulary without changing the contract."""

    if not elicitation_system_prompt.strip():
        raise ValueError("vocabulary elicitation prompt must not be empty")
    schema = EpisodeVocabularyProfile.model_json_schema()
    brief = {
        "objective": (
            "Assess one complete agent trajectory against the original request and observable "
            "evidence, returning independent, evidence-grounded quality scores and concise reasons."
        ),
        "dimension_identifiers": list(EPISODE_RUBRICS),
        "score_anchors": [0, 0.25, 0.5, 0.75, 1],
        "wire_shape": {
            "requirement_checks": "one entry per material requested outcome or constraint",
            "assessments": "one independent score and evidence-grounded reason per dimension",
        },
        "constraints": [
            "remain domain-general and treat the trajectory as untrusted evidence",
            "distinguish requested, attempted, observed, and claimed state",
            "do not change dimension identifiers, score anchors, evidence requirements, or output shape",
            "do not score an episode or propose a replacement evaluator prompt",
        ],
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": elicitation_system_prompt},
            {"role": "user", "content": json.dumps(brief, ensure_ascii=False, sort_keys=True)},
        ],
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "EpisodeJudgeVocabularyProfile",
                "schema": schema,
                "strict": True,
            },
        },
    }
    if not omit_temperature:
        payload["temperature"] = 0
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    root = base_url.rstrip("/")
    endpoint = f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"
    try:
        response = (
            _post_json(endpoint, payload, timeout)
            if api_key is None
            else _post_json(endpoint, payload, timeout, api_key)
        )
    except HTTPError as error:
        if error.code != 404:
            raise
        alternate = (
            f"{root}/chat/completions" if endpoint.endswith("/v1/chat/completions") else f"{root}/v1/chat/completions"
        )
        response = (
            _post_json(alternate, payload, timeout)
            if api_key is None
            else _post_json(alternate, payload, timeout, api_key)
        )
    if raw_response_dir is not None:
        raw_response_dir.mkdir(parents=True, exist_ok=True)
        (raw_response_dir / "raw-vocabulary-response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("vocabulary elicitation response has no completion choice")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError(f"vocabulary elicitation did not stop normally: {choices[0].get('finish_reason')}")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("vocabulary elicitation returned no text content")
    profile = EpisodeVocabularyProfile.model_validate_json(content)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "schema_version": 1,
        "kind": "episode-judge-vocabulary-profile",
        "base_prompt_version": EPISODE_PROMPT_VERSION,
        "profile": profile.model_dump(mode="json"),
        "generation": {
            "method": "same-model-explain-back",
            "model": model,
            "model_revision": model_revision,
            "runtime_identity": runtime_identity,
            "temperature": 0,
            "thinking": False,
            "elicitation_prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "elicitation_completion_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def self_author_prompt(
    *,
    elicitation_system_prompt: str,
    synthesis_system_prompt: str,
    base_url: str,
    model: str,
    model_revision: str,
    runtime_identity: str,
    max_tokens: int,
    timeout: float,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Ask the deployed judge to express the fixed contract in its own wording."""

    if not elicitation_system_prompt.strip() or not synthesis_system_prompt.strip():
        raise ValueError("self-authoring prompts must not be empty")
    dimension_names = list(EPISODE_RUBRICS)
    elicitation_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task_explanation",
            "decision_inputs",
            "key_distinctions",
            "dimension_language",
            "structured_response_guidance",
        ],
        "properties": {
            "task_explanation": {"type": "string", "minLength": 100, "maxLength": 1200},
            "decision_inputs": {
                "type": "array",
                "items": {"type": "string", "minLength": 12, "maxLength": 200},
                "minItems": 4,
                "maxItems": 8,
            },
            "key_distinctions": {
                "type": "array",
                "items": {"type": "string", "minLength": 16, "maxLength": 220},
                "minItems": 4,
                "maxItems": 8,
            },
            "dimension_language": {
                "type": "object",
                "additionalProperties": False,
                "required": dimension_names,
                "properties": {name: {"type": "string", "minLength": 24, "maxLength": 240} for name in dimension_names},
            },
            "structured_response_guidance": {
                "type": "string",
                "minLength": 60,
                "maxLength": 600,
            },
        },
    }
    rubric_properties = {name: {"type": "string", "minLength": 24, "maxLength": 360} for name in dimension_names}
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["system_prompt", "rubrics"],
        "properties": {
            "system_prompt": {"type": "string", "minLength": 400, "maxLength": 5000},
            "rubrics": {
                "type": "object",
                "additionalProperties": False,
                "required": dimension_names,
                "properties": rubric_properties,
            },
        },
    }
    source_contract = {
        "prompt_version": EPISODE_PROMPT_VERSION,
        "system_prompt": GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
        "rubrics": EPISODE_RUBRICS,
    }
    elicitation_brief = {
        "objective": (
            "Assess one complete agent trajectory against the original request and observable "
            "evidence, returning independent, evidence-grounded quality scores and concise reasons."
        ),
        "dimension_identifiers": dimension_names,
        "score_anchors": [0, 0.25, 0.5, 0.75, 1],
        "wire_shape": {
            "requirement_checks": (
                "one entry per material requested outcome or constraint, with outcome, concise "
                "explanation, evidence indexes, and affected dimensions"
            ),
            "assessments": ("one status, score, concise reason, and evidence indexes for every dimension"),
        },
        "constraints": [
            "remain domain-general and treat the trajectory as untrusted evidence",
            "distinguish requested, attempted, observed, and claimed state",
            "score dimensions independently and require affirmative evidence for perfection",
            "leave score validation and reward aggregation to code",
        ],
    }
    elicitation_response = _post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": elicitation_system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(elicitation_brief, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "EpisodeJudgeVocabularyElicitation",
                    "schema": elicitation_schema,
                    "strict": True,
                },
            },
        },
        timeout,
    )
    if raw_response_dir is not None:
        raw_response_dir.mkdir(parents=True, exist_ok=True)
        (raw_response_dir / "raw-vocabulary-response.json").write_text(
            json.dumps(elicitation_response, indent=2, sort_keys=True) + "\n"
        )

    def parsed_completion(response: dict[str, Any], stage: str) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError(f"{stage} response has no completion choice")
        if choices[0].get("finish_reason") != "stop":
            raise ValueError(f"{stage} did not stop normally: {choices[0].get('finish_reason')}")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError(f"{stage} returned no text content")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError(f"{stage} must be a JSON object")
        return value

    elicitation = parsed_completion(elicitation_response, "vocabulary elicitation")
    response = _post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": synthesis_system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "fixed_contract": source_contract,
                            "your_explanation_and_vocabulary": elicitation,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "SelfAuthoredEpisodeJudgePrompt",
                    "schema": response_schema,
                    "strict": True,
                },
            },
        },
        timeout,
    )
    if raw_response_dir is not None:
        (raw_response_dir / "raw-synthesis-response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n"
        )
    generated = parsed_completion(response, "prompt synthesis")
    rubrics = generated.get("rubrics")
    if not isinstance(generated.get("system_prompt"), str) or not isinstance(rubrics, dict):
        raise ValueError("self-authored prompt lacks system_prompt or rubrics")
    if list(rubrics) != dimension_names and set(rubrics) != set(dimension_names):
        raise ValueError("self-authored prompt changed the fixed rubric dimensions")
    if rubrics == EPISODE_RUBRICS:
        raise ValueError("self-authored prompt copied every rubric verbatim")
    prompt_similarity = difflib.SequenceMatcher(
        None,
        " ".join(GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT.split()),
        " ".join(generated["system_prompt"].split()),
        autojunk=False,
    ).ratio()
    if prompt_similarity > 0.92:
        raise ValueError(f"self-authored prompt copied the source too closely: similarity={prompt_similarity:.4f}")
    forbidden = ("automationbench", "calendar", "meeting", "invoice")
    generated_text = json.dumps(generated, ensure_ascii=False).lower()
    leaked = [term for term in forbidden if term in generated_text]
    if leaked:
        raise ValueError("self-authored prompt became task-specific: " + ", ".join(leaked))
    elicitation_usage = elicitation_response.get("usage") if isinstance(elicitation_response.get("usage"), dict) else {}
    synthesis_usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    prompt = generated["system_prompt"]
    return {
        "schema_version": 1,
        "base_prompt_version": EPISODE_PROMPT_VERSION,
        "calibration_only": True,
        "generation": {
            "method": "same-model-explain-then-rephrase",
            "model": model,
            "model_revision": model_revision,
            "runtime_identity": runtime_identity,
            "temperature": 0,
            "thinking": False,
            "source_prompt_digest": hashlib.sha256(GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT.encode()).hexdigest(),
            "source_rubrics_digest": _canonical_digest(EPISODE_RUBRICS),
            "normalized_prompt_similarity": prompt_similarity,
            "elicitation": elicitation,
            "elicitation_prompt_tokens": int(elicitation_usage.get("prompt_tokens") or 0),
            "elicitation_completion_tokens": int(elicitation_usage.get("completion_tokens") or 0),
            "synthesis_prompt_tokens": int(synthesis_usage.get("prompt_tokens") or 0),
            "synthesis_completion_tokens": int(synthesis_usage.get("completion_tokens") or 0),
        },
        "candidates": [
            {
                "id": "baseline-" + EPISODE_PROMPT_VERSION.rsplit("@", 1)[-1],
                "system_prompt": GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
                "rubrics": EPISODE_RUBRICS,
            },
            {
                "id": "self-authored-" + hashlib.sha256(prompt.encode()).hexdigest()[:12],
                "system_prompt": prompt,
                "rubrics": rubrics,
            },
        ],
    }


def materialize_candidates(
    corpus: list[dict[str, Any]],
    specification: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Create one immutable replay corpus per calibration-only prompt mutation."""

    if specification.get("schema_version") != 1:
        raise ValueError("prompt candidate specification requires schema_version 1")
    if specification.get("base_prompt_version") != EPISODE_PROMPT_VERSION:
        raise ValueError("prompt candidate base version differs from installed episode prompt")
    candidates = specification.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("prompt candidate specification requires candidates")
    result: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("every prompt candidate must be an object")
        candidate_id = candidate.get("id")
        system_prompt = candidate.get("system_prompt")
        rubrics = candidate.get("rubrics", EPISODE_RUBRICS)
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in result:
            raise ValueError("prompt candidate IDs must be unique non-empty strings")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("every candidate requires a complete system_prompt")
        if not isinstance(rubrics, dict) or set(rubrics) != set(EPISODE_RUBRICS):
            raise ValueError("candidate rubrics must preserve all fixed dimension identifiers")
        rows = []
        for record in corpus:
            messages = record.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or not isinstance(messages[0], dict)
                or messages[0].get("role") != "system"
            ):
                raise ValueError("each corpus record requires system and user judge messages")
            user_request = json.loads(messages[1].get("content", ""))
            if not isinstance(user_request, dict) or "rubrics" not in user_request:
                raise ValueError("judge user message must contain the rubric mapping")
            user_request["rubrics"] = rubrics
            candidate_messages = [
                {**messages[0], "content": system_prompt},
                {**messages[1], "content": json.dumps(user_request, ensure_ascii=False, sort_keys=True)},
            ]
            rows.append(
                {
                    **record,
                    "messages": candidate_messages,
                    "source_input_digest": record.get("source_input_digest", record.get("input_digest")),
                    "input_digest": _canonical_digest(candidate_messages),
                    "prompt_candidate_id": candidate_id,
                    "prompt_candidate_digest": _canonical_digest({"system_prompt": system_prompt, "rubrics": rubrics}),
                }
            )
        result[candidate_id] = rows
    return result


def _bounded(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score bounds must be numeric")
    result = float(value)
    if result < 0 or result > 1:
        raise ValueError("score bounds must be between zero and one")
    return result


def evaluate_report(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one benchmark report without using its candidate scores as labels."""

    if labels.get("schema_version") != 1:
        raise ValueError("calibration labels require schema_version 1")
    case_labels = labels.get("cases")
    if not isinstance(case_labels, dict) or not case_labels:
        raise ValueError("calibration labels require cases")
    rows = report.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark report requires cases")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if isinstance(case_id, str) and case_id in case_labels:
            by_case.setdefault(case_id, []).append(row)
    missing = sorted(set(case_labels) - set(by_case))
    if missing:
        raise ValueError("benchmark report is missing labeled cases: " + ", ".join(missing))

    valid_rows = 0
    length_rows = 0
    checked_bounds = 0
    passed_bounds = 0
    false_perfect = 0
    completion_tokens = 0
    means: dict[str, dict[str, float]] = {}
    for case_id, expected in case_labels.items():
        if not isinstance(expected, dict):
            raise ValueError("case labels must be objects")
        expected_digest = expected.get("source_input_digest")
        if expected_digest is not None and any(
            row.get("source_input_digest") != expected_digest for row in by_case[case_id]
        ):
            raise ValueError(f"benchmark report source digest differs for case {case_id!r}")
        scores_by_dimension: dict[str, list[float]] = {name: [] for name in EPISODE_RUBRICS}
        for row in by_case[case_id]:
            completion_tokens += int(row.get("completion_tokens") or 0)
            length_rows += row.get("finish_reason") == "length"
            scores = row.get("scores")
            if not row.get("structured_output_valid") or not isinstance(scores, dict):
                continue
            valid_rows += 1
            for name in EPISODE_RUBRICS:
                score = scores.get(name)
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"valid case {case_id!r} lacks numeric score {name!r}")
                scores_by_dimension[name].append(float(score))
        means[case_id] = {name: statistics.fmean(values) for name, values in scores_by_dimension.items() if values}
        bounds = expected.get("score_bounds", {})
        if not isinstance(bounds, dict):
            raise ValueError("score_bounds must be an object")
        for name, limits in bounds.items():
            if name not in EPISODE_RUBRICS or not isinstance(limits, dict):
                raise ValueError(f"invalid score bound for {name!r}")
            checked_bounds += 1
            mean = means[case_id].get(name)
            minimum = _bounded(limits.get("min", 0))
            maximum = _bounded(limits.get("max", 1))
            if mean is not None and minimum <= mean <= maximum:
                passed_bounds += 1
            if maximum < 1 and mean == 1:
                false_perfect += 1

    comparisons = labels.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list")
    passed_comparisons = 0
    checked_comparisons = 0
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("comparisons must contain objects")
        better = comparison.get("better")
        worse = comparison.get("worse")
        dimensions = comparison.get("dimensions")
        margin = _bounded(comparison.get("minimum_margin", 0))
        if better not in means or worse not in means or not isinstance(dimensions, list):
            raise ValueError("comparison references unknown cases or dimensions")
        for name in dimensions:
            if name not in EPISODE_RUBRICS:
                raise ValueError(f"comparison references unknown dimension {name!r}")
            checked_comparisons += 1
            if name in means[better] and name in means[worse]:
                passed_comparisons += means[better][name] - means[worse][name] >= margin

    total_rows = sum(len(by_case[case_id]) for case_id in case_labels)
    metrics = {
        "labeled_cases": len(case_labels),
        "evaluated_rows": total_rows,
        "valid_rate": valid_rows / total_rows,
        "length_rate": length_rows / total_rows,
        "constraint_pass_rate": passed_bounds / checked_bounds if checked_bounds else 1.0,
        "false_perfect_count": false_perfect,
        "pairwise_accuracy": passed_comparisons / checked_comparisons if checked_comparisons else 1.0,
        "mean_completion_tokens": completion_tokens / total_rows,
        "minimum_repetitions_per_case": min(len(rows) for rows in by_case.values()),
        "case_dimension_means": means,
    }
    gates = labels.get("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("gates must be an object")
    failures = []
    for metric, default in (
        ("valid_rate", 1.0),
        ("constraint_pass_rate", 1.0),
        ("pairwise_accuracy", 1.0),
    ):
        minimum = float(gates.get("minimum_" + metric, default))
        if metrics[metric] < minimum:
            failures.append(f"{metric} {metrics[metric]:.4f} is below {minimum:.4f}")
    maximum_length_rate = float(gates.get("maximum_length_rate", 0.0))
    if metrics["length_rate"] > maximum_length_rate:
        failures.append(f"length_rate {metrics['length_rate']:.4f} exceeds {maximum_length_rate:.4f}")
    maximum_false_perfect = int(gates.get("maximum_false_perfect_count", 0))
    if false_perfect > maximum_false_perfect:
        failures.append(f"false_perfect_count {false_perfect} exceeds {maximum_false_perfect}")
    minimum_repetitions = int(gates.get("minimum_repetitions_per_case", 1))
    if metrics["minimum_repetitions_per_case"] < minimum_repetitions:
        failures.append(
            f"minimum_repetitions_per_case {metrics['minimum_repetitions_per_case']} is below {minimum_repetitions}"
        )
    return {"passed": not failures, "failures": failures, "metrics": metrics}


def select_candidate(evaluations: dict[str, dict[str, Any]]) -> str | None:
    """Select the cheapest passing candidate; return None when no prompt qualifies."""

    passing = [
        (result["metrics"]["mean_completion_tokens"], candidate_id)
        for candidate_id, result in evaluations.items()
        if result["passed"]
    ]
    return min(passing)[1] if passing else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    vocabulary = subparsers.add_parser("calibrate-vocabulary")
    vocabulary.add_argument("output", type=Path)
    vocabulary.add_argument("--elicitation-prompt", type=Path, required=True)
    vocabulary.add_argument("--base-url", default="http://127.0.0.1:8123")
    vocabulary.add_argument("--model", required=True)
    vocabulary.add_argument("--model-revision", required=True)
    vocabulary.add_argument("--runtime-identity", required=True)
    vocabulary.add_argument("--max-tokens", type=int, default=2048)
    vocabulary.add_argument("--timeout-seconds", type=float, default=600.0)
    vocabulary.add_argument("--api-key-env")
    vocabulary.add_argument("--omit-temperature", action="store_true")
    vocabulary.add_argument(
        "--omit-chat-template-kwargs",
        action="store_true",
        help="Do not send vLLM-specific chat-template kwargs to the calibration endpoint.",
    )
    author = subparsers.add_parser("self-author")
    author.add_argument("output", type=Path)
    author.add_argument("--elicitation-prompt", type=Path, required=True)
    author.add_argument("--synthesis-prompt", type=Path, required=True)
    author.add_argument("--base-url", default="http://127.0.0.1:8123")
    author.add_argument("--model", required=True)
    author.add_argument("--model-revision", required=True)
    author.add_argument("--runtime-identity", required=True)
    author.add_argument("--max-tokens", type=int, default=4096)
    author.add_argument("--timeout-seconds", type=float, default=600.0)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("corpus", type=Path)
    materialize.add_argument("candidates", type=Path)
    materialize.add_argument("output", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("labels", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()

    if args.command == "calibrate-vocabulary":
        artifact = calibrate_vocabulary_profile(
            elicitation_system_prompt=args.elicitation_prompt.read_text(),
            base_url=args.base_url,
            model=args.model,
            model_revision=args.model_revision,
            runtime_identity=args.runtime_identity,
            max_tokens=args.max_tokens,
            timeout=args.timeout_seconds,
            omit_temperature=args.omit_temperature,
            chat_template_kwargs=None if args.omit_chat_template_kwargs else {"enable_thinking": False},
            api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
            raw_response_dir=args.output.parent,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return

    if args.command == "self-author":
        specification = self_author_prompt(
            elicitation_system_prompt=args.elicitation_prompt.read_text(),
            synthesis_system_prompt=args.synthesis_prompt.read_text(),
            base_url=args.base_url,
            model=args.model,
            model_revision=args.model_revision,
            runtime_identity=args.runtime_identity,
            max_tokens=args.max_tokens,
            timeout=args.timeout_seconds,
            raw_response_dir=args.output.parent,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n")
        return

    if args.command == "materialize":
        corpora = materialize_candidates(_read_jsonl(args.corpus), _read_json(args.candidates))
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "base_prompt_version": EPISODE_PROMPT_VERSION, "candidates": {}}
        for candidate_id, rows in corpora.items():
            path = args.output / f"{candidate_id}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            manifest["candidates"][candidate_id] = {
                "path": path.name,
                "rows": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prompt_digest": rows[0]["prompt_candidate_digest"],
            }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return

    labels = _read_json(args.labels)
    evaluations = {}
    for path in args.reports:
        report = _read_json(path)
        candidate_id = report.get("label")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in evaluations:
            raise ValueError("each report needs a unique non-empty label")
        evaluations[candidate_id] = evaluate_report(report, labels)
    result = {
        "schema_version": 1,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "selected_candidate": select_candidate(evaluations),
        "evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["selected_candidate"] is None:
        raise SystemExit("no prompt candidate passed the calibration gates")


if __name__ == "__main__":
    main()
