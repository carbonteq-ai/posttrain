"""Audit offline source states without replaying tools or calling a model."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from inventory_sdpo_luna_pool124 import inspect
from sdpo_luna_preflight import render_events


def load_native_initialization(world_type, runner_path):
    """Load exact pure initialization helpers without importing the legacy runner API."""
    raw = runner_path.read_bytes()
    tree = ast.parse(raw, filename=str(runner_path))
    wanted = {"strip_none_values", "_service_for_name", "compute_allowed_services"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
        elif (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
              and t.id == "_SERVICE_FIELDS" for t in node.targets)):
            selected.append(node)
    if len(selected) != 4:
        raise ValueError("pinned runner initialization helper contract changed")
    namespace = {"WorldState": world_type}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(runner_path), "exec"), namespace)
    return namespace, hashlib.sha256(raw).hexdigest()


def bind_generated_ids(model, captured, path=""):
    """Bind only omitted, factory-created IDs; seeded IDs and all other state stay strict."""
    changes = []
    if hasattr(model, "model_fields_set") and isinstance(captured, dict):
        for name, field in type(model).model_fields.items():
            value = getattr(model, name)
            target = captured.get(name)
            location = f"{path}.{name}"
            if (name == "id" and name not in model.model_fields_set and field.default_factory
                    and isinstance(value, str) and isinstance(target, str)):
                setattr(model, name, target)
                changes.append({"path": location, "captured_id": target})
            else:
                changes.extend(bind_generated_ids(value, target, location))
    elif isinstance(model, list) and isinstance(captured, list) and len(model) == len(captured):
        for index, (value, target) in enumerate(zip(model, captured, strict=True)):
            changes.extend(bind_generated_ids(value, target, f"{path}[{index}]"))
    return changes


def capture_states(trace, normalize, initialize=None):
    capture = trace["info"]["automationbench_capture"]
    events, snapshots = capture["events"], capture["snapshots"]
    if isinstance(snapshots, str):
        snapshots = json.loads(snapshots)
    for key, raw in snapshots.items():
        if hashlib.sha256(raw.encode()).hexdigest() != key:
            raise ValueError("snapshot bytes do not match retained digest")
    if not events or capture["initial_digest"] != events[0]["before_digest"]:
        raise ValueError("missing initial capture boundary")
    for before, after in zip(events, events[1:], strict=False):
        if before["after_digest"] != after["before_digest"]:
            raise ValueError("discontinuous retained state chain")
    for event in events:
        if event["before_digest"] not in snapshots or event["after_digest"] not in snapshots:
            raise ValueError("missing event boundary snapshot")
    initial = json.loads(snapshots[capture["initial_digest"]])
    final = json.loads(snapshots[events[-1]["after_digest"]])
    expected = (initialize(trace["task"]["data"], initial, events) if initialize
                else normalize(trace["task"]["data"]["initial_state"]))
    if normalize(initial) != expected:
        raise ValueError("captured initial state differs from bound task state")
    if final != trace["info"]["automationbench"]["end_state"]:
        raise ValueError("last captured state differs from recorded endpoint")
    return final


def review(root, check, normalize, initialize=None):
    manifest = json.loads((root / "manifest.json").read_text())
    bindings = manifest["bindings"]
    audits, references = [], {}
    # Among passing duplicate sources, choose smallest SHA before any model outcome.
    for source in sorted(manifest["candidates"], key=lambda e: e["source_episode_sha256"]):
        row = {"task": source["task"], "source_episode_sha256": source["source_episode_sha256"],
               "qualified": False, "checks": [], "errors": []}
        try:
            raw = (root / source["episode_file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["source_episode_sha256"]:
                raise ValueError("candidate source bytes changed")
            episode = json.loads(raw)
            admission = inspect(episode, bindings[source["task"]])
            if not admission["candidate"]:
                raise ValueError(f"candidate no longer admitted: {admission['reason']}")
            trace = episode["traces"][0]
            data = trace["task"]["data"]
            final = capture_states(trace, normalize, initialize)
            for spec in data["assertions"]:
                if spec.get("scored") is False or spec.get("excluded") is True:
                    continue
                row["checks"].append({"spec": spec,
                    "initial_passed": check(data["initial_state"], spec),
                    "final_passed": check(final, spec)})
            row["qualified"] = bool(row["checks"]) and all(c["final_passed"] for c in row["checks"])
            row["preservation_checks"] = sum(c["initial_passed"] for c in row["checks"])
            row["broken_preservation_checks"] = sum(c["initial_passed"] and not c["final_passed"]
                                                     for c in row["checks"])
            row["capture_coverage"] = admission["capture_coverage"]
            if row["qualified"] and source["task"] not in references:
                references[source["task"]] = {"episode_id": episode["id"],
                    "source_episode_sha256": source["source_episode_sha256"],
                    **{k: v for k, v in bindings[source["task"]].items() if k.endswith("sha256")},
                    "observations": render_events(trace["info"]["automationbench_capture"]["events"]),
                    "current_native_checks_pass": True,
                    "warning": "Retained historical execution observations; not hidden thinking, "
                               "per-action gold, native re-execution or complete failed/concurrent-call capture"}
        except Exception as exc:
            row["errors"].append(f"{type(exc).__name__}: {exc}")
        audits.append(row)
    return {"audits": audits, "references": references, "training_tasks": len(bindings),
            "tasks_without_admitted_reference": sorted(set(bindings) - references.keys()),
            "duplicate_selection": "Smallest source SHA among current-rubric-passing candidates",
            "external_model_calls": 0, "hidden_thinking_available": False,
            "teacher_context_length_and_live_binding_qualified": False,
            "scope": "Current rubric plus hashed retained state-chain review; not tool replay, "
                     "full failed/concurrent-call coverage, gradient qualification or reward improvement"}


def reference_bundle(audit):
    """Export only explicitly passing sources; never promote rejected candidates."""
    references = audit["references"]
    if not references:
        raise ValueError("no admitted references")
    for task, reference in references.items():
        matches = [row for row in audit["audits"] if row["task"] == task
                   and row["source_episode_sha256"] == reference["source_episode_sha256"]]
        if (len(matches) != 1 or not matches[0]["qualified"] or matches[0]["errors"]
                or not matches[0]["checks"] or not all(c["final_passed"] for c in matches[0]["checks"])
                or reference.get("current_native_checks_pass") is not True):
            raise ValueError("reference lacks unique complete passing source audit")
    return {"references": references, "current_rubric_state_checks_complete": True,
            "coverage_scope": "Flag covers admitted references only; other candidates remain rejected",
            "training_tasks": audit["training_tasks"],
            "tasks_without_admitted_reference": audit["tasks_without_admitted_reference"],
            "external_model_calls": 0, "teacher_context_length_and_live_binding_qualified": False,
            "source_review_sha256": hashlib.sha256(
                json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "scope": audit["scope"], "initialization_policy": audit["initialization_policy"],
            "pinned_runner_source_sha256": audit["pinned_runner_source_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["AUTOMATIONBENCH_STRICT_ASSERTIONS"] = "1"
    from automationbench.rubric import AssertionRegistry, WorldState
    package = importlib.util.find_spec("automationbench")
    helpers, runner_sha = load_native_initialization(
        WorldState, Path(package.origin).parent / "runner.py")
    compute_allowed_services = helpers["compute_allowed_services"]
    strip_none_values = helpers["strip_none_values"]

    def normalize(state):
        return WorldState.model_validate(state).model_dump(mode="json")

    def check(state, spec):
        return bool(AssertionRegistry.check(WorldState.model_validate(state), dict(spec)))

    def initialize(data, captured, events):
        raw = strip_none_values(data["initial_state"])
        world = WorldState.model_validate(raw)
        world.meta.allowed_services = compute_allowed_services(
            raw, [strip_none_values(a) for a in data["assertions"]], data.get("zapier_tools", []))
        generated = bind_generated_ids(world, captured)
        arguments = "\n".join(e["arguments_json"] for e in events)
        if any(row["captured_id"] in arguments for row in generated):
            raise ValueError("historical action depends on an unbound generated runtime ID")
        return world.model_dump(mode="json")

    result = review(args.root, check, normalize, initialize)
    result["pinned_runner_source_sha256"] = runner_sha
    result["helper_loading"] = (
        "Exact AST definitions of four pure runner initialization helpers; full legacy runner "
        "cannot import against installed Verifiers. No copied or altered initialization algorithm.")
    result["initialization_policy"] = (
        "Pinned runner strip_none_values and compute_allowed_services; captured identities bind only "
        "omitted factory-created id fields. All remaining initialized state must match exactly. "
        "Reject sources whose tool arguments reuse these generated historical IDs; no action remapping.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sources": len(result["audits"]), "references": len(result["references"]),
                      "errors": sum(bool(r["errors"]) for r in result["audits"])}))
