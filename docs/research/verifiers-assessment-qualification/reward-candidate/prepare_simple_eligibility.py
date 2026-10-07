"""Offline Simple eligibility qualification; never collect or mutate references.

Run with the isolated environment interpreter and both candidate source roots on
PYTHONPATH. Inspection does not seal a redesign or execute the historical driver.
Execution requires an explicitly released source-stable window. The composition
approval below is restricted to four exact public requests and a reviewed EMPTY
task-specific guard inventory, not universal or environment-wide guard closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import verifiers
import verifiers.v1 as vf
from automationbench_v1.calibration.collector import read_retained_artifacts
from automationbench_v1.calibration.eligibility import (
    TaskRewardContract,
    build_eligibility_proof,
    validate_eligibility_proof,
)
from automationbench_v1.calibration.frozen_verification import (
    FrozenVerification,
    build_frozen_verification,
    capture_frozen_verifier_binding,
)
from automationbench_v1.calibration.inventory import content_digest, load_inventory
from automationbench_v1.calibration.reassessment import TaskScoreReassessmentVerifier
from automationbench_v1.calibration.reward_revision import capture_redesign_revision
from automationbench_v1.simple_assessments import ReviewedSimpleTask
from automationbench_v1.simple_evidence import REQUESTS
from automationbench_v1.taskset import AutomationBenchData, AutomationBenchTaskConfig
from verifiers.v1.assessment_source import capture_trace_source

PROJECT = Path("/home/hammad/projects/rl")
STATE = PROJECT / ".posttrain/state/verifiers-assessment-qualification"
DOCS = PROJECT / "docs/research/verifiers-assessment-qualification"
ORIGINAL = STATE / "luna-reference-campaign-01/remaining700/untimed"
CANDIDATE = Path(
    "/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1"
)
NATIVE = Path("/home/hammad/projects/verifiers-credit-candidate-20261003/verifiers")
APPROVED_BINDING = "5d67f0266cc01f37ede66f1871d44202b58a2d1a7a79bb57ebdba64a19c8b6ca"
REVIEW_REVISION = "root-reviewed-empty-task-specific-guards-20261003-public-direct-requests-v1"
BINDINGS = {
    "simple.sf_case_priority_high": (
        "b5a9b50ff1ad6992a9f6c8fec175585f6c27c57ea913256ce70a59239a9edddd",
        "249015db21420e535f668c90c3181a9803e28b601f42b990b18ba15c63846296",
    ),
    "simple.hs_update_contact_phone": (
        "dd0f6d93d5062a72bbe620c3bfe191bb029f9b33464416cfc4a190627bc736ed",
        "dd10d63c5313fe166f9e24d621da2017e79985550710774b6f3eee6ce049f9aa",
    ),
    "simple.asana_api_docs_task": (
        "923dd7d4a69c8b0c83266fb9eecfcd3f9fa888e6de08686b22661b0bd471557c",
        "b1cee3cf075f8b7248a9ab829174b4f37f51a96a25afb41535a9cd833ecfab47",
    ),
    "simple.gcal_one_on_one": (
        "7515cdfd8b31c56a94e06975bbb68b178022baeb6d7f698686c80824318418bc",
        "b7ee3c8a36e8418a11b942473c3651f3b63092eadad31298b25846503812dfdd",
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError("qualification_output_already_exists")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_closure() -> tuple[str, str]:
    if Path(next(iter(verifiers.__path__))).resolve() != NATIVE:
        raise ValueError("candidate_native_import_root_differs")
    native = {str(path.relative_to(NATIVE)): sha(path) for path in sorted(NATIVE.rglob("*.py"))}
    root = CANDIDATE / "src/automationbench_v1"
    adapter = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*.py"))}
    return content_digest(native), content_digest(
        {"native": native, "adapter": adapter, "lock": sha(CANDIDATE / "uv.lock")}
    )


def approved_binding():
    # Independent composition selection: a computed embedded digest alone is not
    # approval. Restrict to the previously qualified fixed local original driver.
    artifact = FrozenVerification.model_validate_json(
        (DOCS / "frozen-official-verification-candidate/simple.airtable_create_contact.json").read_bytes()
    )
    binding = artifact.binding
    roots = {
        "automationbench": "/home/hammad/projects/verifiers-environments/environments/automationbench_v1/src/automationbench",
        "automationbench_environment": "/home/hammad/projects/verifiers-environments/environments/automationbench_v1/src/automationbench_v1",
        "verifiers": "/home/hammad/projects/verifiers/verifiers",
    }
    if (
        binding.digest != APPROVED_BINDING
        or binding.package_roots != roots
        or binding.interpreter
        != "/home/hammad/projects/verifiers-environments/environments/automationbench_v1/.venv/bin/python"
    ):
        raise ValueError("independently_approved_frozen_binding_differs")
    manifest = json.loads((ORIGINAL / "manifest.json").read_bytes())
    live = capture_frozen_verifier_binding(
        interpreter=Path(binding.interpreter),
        package_roots={key: Path(value) for key, value in roots.items()},
        source_archive=Path(binding.archive_path),
        source_identity=manifest["source_identity"],
    )
    if live != binding:
        raise ValueError("approved_original_source_closure_changed")
    return binding


def contract_for(task, scorer):
    prompt = " ".join(item.get("content", "") for item in task.data["prompt"] if item.get("role") == "user")
    if prompt != REQUESTS[task.task_name] or task.config.get("capture_actions") is not True:
        raise ValueError("exact_public_request_or_capture_revision_differs")
    public_policy = {
        "task_name": task.task_name,
        "user_request": prompt,
        "policy_revision": scorer.policy_revision,
        "task_specific_guard_inventory": [],
        "scope": "task_specific",
        "review_revision": REVIEW_REVISION,
    }
    policy_digest = content_digest(public_policy)
    body = {
        "schema_version": 2,
        "task_name": task.task_name,
        "task_digest": task.digest,
        "policy_source_digest": policy_digest,
        "admission_policy": "accepted_redesign",
        "guard_set": {
            "status": "reviewed_closed",
            "check_keys": [],
            "policy_source_digest": policy_digest,
            "scope": "task_specific",
            "review_revision": REVIEW_REVISION,
        },
        "checks": [
            {
                "key": purpose,
                "purpose": purpose,
                "signal": scorer.signal_definition(signal).model_dump(mode="json"),
                "producer_id": scorer.producer_id,
                "producer_revision": scorer.producer_revision,
                "rubric_revision": scorer.policy_revision,
                "subject_kind": "trace",
                "operator": "eq",
                "expected_value": 1.0,
            }
            for purpose, signal in (("goal", "simple.requested_state"), ("coverage", "simple.recording_coverage"))
        ],
    }
    return TaskRewardContract.model_validate({**body, "digest": content_digest(body)}), public_policy


def prepare(*, output: Path | None):
    inventory_path, manifest_path, journal_path = (
        ORIGINAL / name for name in ("inventory.json", "manifest.json", "attempts.jsonl")
    )
    inventory = load_inventory(inventory_path)
    binding = approved_binding()
    registry = json.loads((DOCS / "luna-development-policy-contracts.json").read_bytes())
    tasks, contracts, policies, episodes = {}, {}, {}, {}
    for name, (attempt, digest) in BINDINGS.items():
        entry = next(item for item in registry["tasks"] if item["task_name"] == name)
        path = ORIGINAL / attempt / "episode.json"
        if (
            entry["source_binding"]["source_episode_path"] != str(path)
            or entry["source_binding"]["source_episode_sha256"] != digest
            or sha(path) != digest
        ):
            raise ValueError("approved_development_episode_binding_differs")
        task = next(task for task in inventory.tasks if task.task_name == name)
        episode = vf.WireEpisode.model_validate_json(path.read_bytes())
        if episode.task.data.model_dump(mode="json") != task.data or len(episode.traces) != 1:
            raise ValueError("original_episode_task_or_trace_differs")
        scorer = ReviewedSimpleTask(
            AutomationBenchData.model_validate(task.data), AutomationBenchTaskConfig.model_validate(task.config)
        )
        contract, policy = contract_for(task, scorer)
        tasks[name], contracts[name], policies[name], episodes[name] = task, contract, policy, episode
    inspection = {
        "mode": "inspect" if output is None else "execute",
        "model_calls": 0,
        "approved_frozen_binding": binding.digest,
        "guard_scope": "task_specific",
        "guard_inventory": [],
        "guard_review_revision": REVIEW_REVISION,
        "task_names": list(tasks),
        "original_episode_sha256": {name: digest for name, (_, digest) in BINDINGS.items()},
    }
    if output is None:
        return inspection
    output = output.resolve()
    if not output.is_relative_to(STATE) or output == STATE:
        raise ValueError("qualification_outputs_must_use_new_ignored_state_directory")
    output.mkdir(parents=True, exist_ok=False)
    native_digest, producer_digest = source_closure()
    key = (
        f"{ReviewedSimpleTask.producer_id}@{ReviewedSimpleTask.producer_revision}/{ReviewedSimpleTask.policy_revision}"
    )
    configuration = {
        "task_contracts": {name: contract.digest for name, contract in contracts.items()},
        "frozen_verifiers": [binding.digest],
        "assessment_verifiers": {key: producer_digest},
        "review_revision": REVIEW_REVISION,
    }
    revision = capture_redesign_revision(configuration, native_digest)
    revision_path = output / "redesign-revision.json"
    write(revision_path, revision.model_dump(mode="json"))
    verifier = TaskScoreReassessmentVerifier(ReviewedSimpleTask, source_digest=producer_digest)
    results = []
    for name, task in tasks.items():
        episode_path = ORIGINAL / BINDINGS[name][0] / "episode.json"
        directory = output / name
        directory.mkdir()
        write(directory / "public-policy-review.json", policies[name])
        write(directory / "reward-contract.json", contracts[name].model_dump(mode="json"))
        row = {"task_name": name, "original_episode_sha256": BINDINGS[name][1], "status": "not_admitted"}
        try:
            if source_closure() != (native_digest, producer_digest):
                raise ValueError("candidate_sources_changed_before_replay")
            official = build_frozen_verification(
                approved_binding=binding,
                inventory_path=inventory_path,
                manifest_path=manifest_path,
                journal_path=journal_path,
                episode_path=episode_path,
            )
            official_path = directory / "frozen-verification.json"
            write(official_path, official.model_dump(mode="json"))
            episode = episodes[name]
            trace = episode.traces[0]
            trace.state.artifacts.update(read_retained_artifacts(episode_path.parent, trace))
            scorer = ReviewedSimpleTask(
                AutomationBenchData.model_validate(task.data), AutomationBenchTaskConfig.model_validate(task.config)
            )
            source = capture_trace_source(trace, task_evidence=scorer.assessment_source(trace))
            request = scorer.assessment_requests(source)[0][1].run
            replay = verifier.replay(episode=episode, task=task, contract=contracts[name], request=request)
            assessment_path = directory / "reassessment.json"
            write(
                assessment_path,
                {
                    "schema_version": 1,
                    "original_episode_digest": BINDINGS[name][1],
                    "redesign_digest": revision.digest,
                    "contract_digest": contracts[name].digest,
                    "selected_run_ids": [replay.batch.run.run_id],
                    "batches": [replay.batch.model_dump(mode="json")],
                    "credit_assignments": [item.model_dump(mode="json") for item in replay.credit_assignments],
                },
            )
            proof = build_eligibility_proof(
                inventory_path=inventory_path,
                manifest_path=manifest_path,
                journal_path=journal_path,
                episode_path=episode_path,
                assessment_path=assessment_path,
                revision_path=revision_path,
                frozen_verification_path=official_path,
                approved_binding=binding,
                current_redesign=revision,
                reassessment_verifiers={key: verifier},
                contract=contracts[name],
            )
            validate_eligibility_proof(
                proof,
                task_name=name,
                task_digest=task.digest,
                redesign_digest=revision.digest,
                current_redesign=revision,
                approved_bindings={binding.digest: binding},
                reassessment_verifiers={key: verifier},
            )
            if source_closure() != (native_digest, producer_digest) or sha(episode_path) != BINDINGS[name][1]:
                raise ValueError("source_closure_changed_after_validation")
            write(directory / "eligibility-proof.json", proof.model_dump(mode="json"))
            row.update(
                status="admitted_task_specific_candidate",
                proof_digest=proof.digest,
                reported_output_tokens=proof.reported_output_tokens,
                required_checks=len(proof.checks),
                proof_path=str(directory / "eligibility-proof.json"),
            )
        except Exception as error:
            row.update(reason=str(error), error_type=type(error).__name__)
        results.append(row)
    summary = {
        **inspection,
        "redesign_digest": revision.digest,
        "state_directory": str(output),
        "producer_source_digest": producer_digest,
        "native_source_digest": native_digest,
        "results": results,
        "accepted_count": sum(row["status"] == "admitted_task_specific_candidate" for row in results),
        "limitations": [
            "Only four exact reviewed public requests",
            "Empty task-specific guard inventory excludes unreviewed environment-wide guards",
            "No Qwen collection, model calls, publication, or dependency pin changes",
        ],
    }
    write(output / "qualification-summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Execute offline replay into a new ignored state directory after source-window release",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(output=args.output_dir), indent=2))
