"""Prepare a local, content-addressed observability variant; never publish Git."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ADDITIONS = {
    "finance.sales_tax_remittance": ["gmail_find_email", "gmail_get_email_by_id"],
    "finance.prepaid_amortization": ["slack_list_channels"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-repo", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    package = args.environment_repo / "environments/automationbench_v1"
    args.evidence.mkdir(parents=True, exist_ok=True)
    patch = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "environments/automationbench_v1/src"], cwd=args.environment_repo
    )
    (args.evidence / "environment-source.patch").write_bytes(patch)
    files = {}
    for root in [package / "src"]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ["pyproject.toml", "uv.lock", "README.md"]:
        files[name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
    tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    relative = Path(f".posttrain/state/local-environments/heldout62-observable-{tree[:16]}/automationbench_v1")
    snapshot = args.project / relative
    if snapshot.exists():
        for name, digest in files.items():
            if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Retained snapshot mismatch: {name}")
    else:
        snapshot.mkdir(parents=True)
        shutil.copytree(package / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ["pyproject.toml", "uv.lock", "README.md"]:
            shutil.copy2(package / name, snapshot / name)
    catalog = args.project / ".posttrain/catalog"
    original = yaml.safe_load((catalog / "automationbench-manifest-steps.yaml").read_text())
    old_env = "automationbench-manifest-heldout62-eval-64k20t-qwen-v1"
    new_env = "automationbench-heldout62-observable-local-qwen-v1"
    new_eval = "automationbench-heldout62-observable-local-qwen"
    env = copy.deepcopy(original["environment"][old_env])
    env["source"] = {"kind": "project-path", "package": "automationbench-v1", "path": str(relative)}
    taskset = env["activation"]["config"]["taskset"]
    taskset["additional_tools_by_task"] = ADDITIONS
    taskset["task"]["world_time_context_if_declared"] = True
    env["parameters"]["observability_profile"] = "heldout62-observable-local-v1"
    env["parameters"]["source_tree_sha256"] = tree
    evaluation = copy.deepcopy(original["evaluation"]["automationbench-manifest-heldout62-64k20t-qwen"])
    evaluation["environments"] = [new_env]
    for key in ["success", "selection", "measurement"]:
        evaluation[key] = {new_env: evaluation[key][old_env]}
    (catalog / "heldout62-observable-local.yaml").write_text(
        yaml.safe_dump({"environment": {new_env: env}, "evaluation": {new_eval: evaluation}}, sort_keys=False)
    )
    packages = args.project / ".posttrain/work_packages"
    work = yaml.safe_load((packages / "qwen35_27b_heldout62_eval.yaml").read_text())
    work["work_package_id"] = "qualify/qwen3.5-27b/heldout62-observable-local"
    work["description"] = (
        "Separate local repaired heldout62: declared task clocks, audited retrieval additions, LinkedIn roundtrip fix; original task names, checks, model and budgets retained."
    )
    work["recipe"]["id"] = "recipes/qwen3.5-27b-heldout62-observable-local@1"
    work["bindings"]["evaluation_plan"]["id"] = new_eval
    work["bindings"]["environment"]["id"] = new_env
    work["metadata"]["comparison_family"] = new_eval
    (packages / "qwen35_27b_heldout62_observable_local.yaml").write_text(yaml.safe_dump(work, sort_keys=False))
    receipt = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.environment_repo, text=True
        ).strip(),
        "tree_sha256": tree,
        "files": files,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "snapshot": str(snapshot),
        "additional_tools_by_task": ADDITIONS,
        "world_time_context_if_declared": True,
        "publication": "local experiment snapshot only; no Git commit or push",
    }
    (args.evidence / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ["tree_sha256", "snapshot", "patch_sha256"]}))


if __name__ == "__main__":
    main()
