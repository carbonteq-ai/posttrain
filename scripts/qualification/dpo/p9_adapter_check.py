"""Check, in the container, that the adapter the DPO job downloads is the evaluated SFT adapter, byte for byte.

The job's model is run q35-4b-sft-v4-e3-20261113-r2's final adapter, stored as a Trackio artifact whose recorded tree
digest (posttrain_content_digest, lab trackio_observer._tree_sha256) the catalog pins. P9 was evaluated with
/evidence/sft-v4-20261113-r2/e3, copied from that run's training/sft/adapter folder. Checked here, inside a container
that mounts the run folder read-only by convention (/ptruns/<run>) and the evidence folder:
  1. the run folder's tree digest equals the pinned digest (so the stored artifact is this folder);
  2. its adapter_model.safetensors and adapter_config.json equal P9's evaluated copies (sha256).
usage: p9_adapter_check.py RUN_ADAPTER_DIR EVALUATED_ADAPTER_DIR EXPECTED_TREE_DIGEST OUT.json
"""
import hashlib
import json
import sys
from pathlib import Path

FILES = ("adapter_model.safetensors", "adapter_config.json")


def tree_sha256(path: Path) -> str:
    """The lab's _tree_sha256: every file under path, sorted, as relative-name NUL content."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(run_dir: Path, evaluated_dir: Path, expected_tree: str) -> dict:
    tree = tree_sha256(run_dir)
    files = {name: {"run": file_sha256(run_dir / name), "evaluated": file_sha256(evaluated_dir / name)}
             for name in FILES}
    ok = tree == expected_tree and all(v["run"] == v["evaluated"] for v in files.values())
    return {"run_dir": str(run_dir), "evaluated_dir": str(evaluated_dir), "tree_sha256": tree,
            "expected_tree_sha256": expected_tree, "files": files, "ok": ok}


def main(run_dir, evaluated_dir, expected_tree, out):
    result = check(Path(run_dir), Path(evaluated_dir), expected_tree)
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
