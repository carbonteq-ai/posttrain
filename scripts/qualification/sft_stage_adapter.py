"""Phase 2A arm T: copy one framework SFT adapter into the evaluation volume (queue_sft.sh stage_adapter).

The framework writes adapters under /var/lib/posttrain/runs/<id>/scratch/<id>/training/sft/{adapter,
checkpoints/step-NNNNNNNN/model} as root, adapter_model.safetensors with mode 600, so the workstation user cannot read
them. This runs as a short dstack task through sdpo_ab_launch.sh with SDPO_AB_PTRUNS=<id> (that run's folder mounted at
/ptruns/<id>) and copies exactly adapter_model.safetensors and adapter_config.json to --dst under /evidence with mode
0644. It never writes under /ptruns. Writes --sha-out (default ./stage-sha.json, i.e. the task's own evidence folder):
sha256 of each source and destination file; exits 1 if any pair differs.

Usage: sft_stage_adapter.py --src /ptruns/<id>/scratch/<id>/training/sft/adapter --dst /evidence/<F>/<name>
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

FILES = ("adapter_model.safetensors", "adapter_config.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sha-out", default="stage-sha.json")
    a = ap.parse_args()
    src, dst = os.path.realpath(a.src), os.path.realpath(a.dst)
    if not src.startswith("/ptruns/") or "/training/sft/" not in src:
        sys.exit(f"--src must be a training/sft folder under /ptruns: {src}")
    if not dst.startswith("/evidence/") or dst.count("/") < 3:
        sys.exit(f"--dst must be a folder two levels under /evidence: {dst}")
    for name in FILES:
        p = os.path.join(src, name)
        if not os.path.isfile(p) or os.path.islink(os.path.join(a.src, name)):
            sys.exit(f"missing source file: {p}")
    os.makedirs(dst, exist_ok=True)
    os.chmod(dst, 0o755)
    out = {"src": src, "dst": dst, "files": {}}
    ok = True
    for name in FILES:
        s, d = os.path.join(src, name), os.path.join(dst, name)
        tmp = d + ".tmp"
        shutil.copyfile(s, tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, d)
        rec = {"src_sha256": sha256(s), "dst_sha256": sha256(d), "bytes": os.path.getsize(d),
               "dst_mode": oct(os.stat(d).st_mode & 0o777)}
        rec["equal"] = rec["src_sha256"] == rec["dst_sha256"] and rec["dst_mode"] == "0o644"
        ok &= rec["equal"]
        out["files"][name] = rec
    out["ok"] = ok
    with open(a.sha_out, "w") as f:
        json.dump(out, f, indent=1)
    os.chmod(a.sha_out, 0o644)
    print(json.dumps(out))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
