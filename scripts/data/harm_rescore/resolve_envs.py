# ruff: noqa: E701, E702, E741, B905
"""environment_id -> verifiers-environments revision from the rl repo's catalog history (read-only git).

usage: python3 resolve_envs.py run_envids.json > env_revisions.json
"""

import json
import os
import subprocess
import sys

import yaml

RL = os.environ.get("POSTTRAIN_REPO", ".")


def git(*a):
    return subprocess.run(["git", "-C", RL, *a], capture_output=True, text=True).stdout


def find(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, dict) and isinstance(v.get("source"), dict) and v["source"].get("revision"):
                return v["source"]
            r = find(v, key)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v, key)
            if r:
                return r


envs = sorted({e for _, e, _ in json.load(open(sys.argv[1])) if e})
out = {}
for env in envs:
    for c in reversed(git("log", "--all", "--format=%H", "-S", env).split()):  # oldest first: the definition
        for path in git("grep", "-l", f"{env}:", c, "--", "*.yaml").split():
            path = path.split(":", 1)[1]
            try:
                src = find(yaml.safe_load(git("show", f"{c}:{path}")), env)
            except Exception:
                src = None
            if src:
                out[env] = {"revision": src["revision"], "rl_commit": c[:10], "path": path}
                break
        if env in out:
            break
    print(env, out.get(env, {}).get("revision"), file=sys.stderr)
print(json.dumps(out, indent=1))
