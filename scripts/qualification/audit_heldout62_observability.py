"""Inventory exact heldout tasks through an explicitly selected local environment.

Run with the environment package interpreter; its imports own native task loading.
This is an audit inventory, not an automatic semantic-solvability classifier.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main():
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overlay = yaml.safe_load(args.catalog.read_text())
    environment = next(iter(overlay["environment"].values()))
    config = AutomationBenchConfig.model_validate(environment["activation"]["config"]["taskset"])
    tasks = AutomationBenchTaskset(config).load()
    rows = []
    for task in tasks:
        data = task.data.model_dump(mode="json")
        initial = data["initial_state"]
        resources = {}

        def walk(value, path, resources=resources):
            if isinstance(value, dict):
                if str(value.get("id", "")).startswith("noise"):
                    return  # Distractor payload remains covered by the full state digest.
                for key, item in value.items():
                    if (
                        key
                        in {
                            "body_plain",
                            "text",
                            "description",
                            "notes",
                            "Notes",
                            "subject",
                            "title",
                            "name",
                            "content",
                        }
                        and isinstance(item, str)
                        and item
                    ):
                        resources["/".join([*path, key])] = item
                    else:
                        walk(item, [*path, key])
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, [*path, str(index)])

        walk(initial, [])
        rows.append(
            {
                "task": data["task_name"],
                "prompt": data["prompt"],
                "tools": data["zapier_tools"],
                "clock": initial.get("meta", {}).get("current_time"),
                "services": list(initial),
                "resources": resources,
                "assertions": data["assertions"],
                "initial_state_sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Inventoried {len(rows)} tasks into {args.output}")


if __name__ == "__main__":
    main()
