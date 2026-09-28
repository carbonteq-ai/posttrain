"""Adopt the published causal-conv1d rebuild into every Qwen3.5 training image.

Run once, from the repository root, after the retained wheel is on carbonteq/dev:

    uv run python tools/kernel-wheels/causal-conv1d/adopt.py

It refuses to change anything until the index serves the exact retained bytes.
Then it adds causal-conv1d next to fla-core in the TRL extra, the supervised and
transform profiles, the quantization tool and the veRL backend project; adds
`causal_conv1d` to each image's smoke import; regenerates every lock the
repository derives from those inputs (never pointing a lock at a local path);
and updates the recorded lock digests. Re-running it is a no-op. Review with
`git diff`, run the validation ladder, rebuild the kind images and commit.
The index check uses the system trust store; set SSL_CERT_FILE to the
CarbonTeq CA bundle if pypi.lan's certificate is not trusted system-wide.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

VERSION = "1.7.0+cu130torch2.13"
WHEEL = f"causal_conv1d-{VERSION}-cp313-cp313-linux_x86_64.whl"
WHEEL_SHA256 = "b69f39142ac88cac91cba5f954cb616c50bc49933cd84f349319420470a4947a"
SIMPLE = "https://pypi.lan/carbonteq/dev/+simple/causal-conv1d/"
INDEX_FILE = f"https://pypi.lan/carbonteq/dev/+f/b69/f39142ac88cac/{WHEEL}"  # devpi path: digest prefix
MARKER = "sys_platform == 'linux' and platform_machine == 'x86_64'"
REQUIREMENT = f'"causal-conv1d=={VERSION}; {MARKER}",'

ROOT = Path(__file__).resolve().parents[3]
KINDS = ROOT / "packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds"
VERL = KINDS / "verl-py313"
CATALOG_LOCKS = ROOT / "packages/catalog/src/posttrain/catalog/base/locks.toml"
LAB_OVERLAY = ROOT / "apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml"
DEV_INDEX = (
    '[[tool.uv.index]]\nname = "carbonteq-dev"\nurl = "https://pypi.lan/carbonteq/dev/+simple/"\nexplicit = true\n'
)


def verify_published() -> None:
    page = urllib.request.urlopen(SIMPLE, timeout=30).read().decode()
    hrefs = [h for h in re.findall(r'href="([^"]+)"', page) if WHEEL in urllib.parse.unquote(h)]
    if not hrefs:
        raise SystemExit(f"{WHEEL} is not on {SIMPLE}; publish it first (docs/tooling/linear-attention-kernels)")
    url = urllib.parse.urljoin(SIMPLE, hrefs[0].split("#")[0])
    if urllib.parse.unquote(url) != INDEX_FILE:
        raise SystemExit(f"carbonteq/dev stores the wheel at {url}, not {INDEX_FILE}")
    data = urllib.request.urlopen(url, timeout=600).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != WHEEL_SHA256:
        raise SystemExit(f"carbonteq/dev serves {digest}, expected the retained {WHEEL_SHA256}")
    print(f"verified {WHEEL} on carbonteq/dev ({digest})")


def edit(path: Path, old: str, new: str, done: str) -> None:
    text = path.read_text(encoding="utf-8")
    if done in text:
        return
    if old not in text:
        raise SystemExit(f"{path.relative_to(ROOT)}: anchor not found: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"edited {path.relative_to(ROOT)}")


def edit_sources() -> None:
    fla = '    "fla-core>=0.5.2,<0.6",\n'
    # transform.lock.txt is exported without index URLs (like carbonteq-trackio there), so the quantization
    # project references the immutable index file directly.
    edit(
        ROOT / "tools/quantization/pyproject.toml",
        fla,
        fla
        + "    # Exported to transform.lock.txt without index URLs, so internal wheels use immutable direct references.\n"
        f'    "causal-conv1d @ {INDEX_FILE}#sha256={WHEEL_SHA256} ; {MARKER}",\n',
        "causal-conv1d @",
    )
    for project in (ROOT / "packages/train/pyproject.toml", VERL / "release/pyproject.toml"):
        edit(project, fla, fla + f"    {REQUIREMENT}\n", "causal-conv1d==")
        edit(
            project,
            "[tool.uv.sources]\n",
            '[tool.uv.sources]\ncausal-conv1d = { index = "carbonteq-dev" }\n',
            'causal-conv1d = { index = "carbonteq-dev" }',
        )
    edit(
        VERL / "release/pyproject.toml",
        '[[tool.uv.index]]\nname = "pytorch-cu130"',
        DEV_INDEX + '\n[[tool.uv.index]]\nname = "pytorch-cu130"',
        'name = "carbonteq-dev"',
    )
    # pypi.lan uses the CarbonTeq CA; the root and quantization projects already trust the system store.
    edit(
        VERL / "release/pyproject.toml",
        "[tool.uv]\npackage = false\n",
        "[tool.uv]\npackage = false\nsystem-certs = true\n",
        "system-certs = true",
    )
    for profile in (KINDS / "profiles/supervised.txt", KINDS / "profiles/transform.txt"):
        edit(profile, "fla-core==0.5.2\n", f"causal-conv1d=={VERSION}\nfla-core==0.5.2\n", "causal-conv1d==")
    smoke = "import fla.modules, fla.ops.gated_delta_rule, "
    with_conv = "import causal_conv1d, fla.modules, fla.ops.gated_delta_rule, "
    for dockerfile in (KINDS / "Dockerfile", KINDS / "Dockerfile.vllm", VERL / "Dockerfile"):
        text = dockerfile.read_text(encoding="utf-8")
        if smoke in text:
            dockerfile.write_text(text.replace(smoke, with_conv), encoding="utf-8")
            print(f"edited {dockerfile.relative_to(ROOT)}")
    edit(
        KINDS / "Dockerfile",
        "import compressed_tensors, fla.modules",
        "import causal_conv1d, compressed_tensors, fla.modules",
        "import causal_conv1d, compressed_tensors",
    )
    test = ROOT / "packages/runtime-images/tests/test_verl_release_gate.py"
    edit(
        test,
        '"import fla.modules, fla.ops.gated_delta_rule, ray,',
        '"import causal_conv1d, fla.modules, fla.ops.gated_delta_rule, ray,',
        '"import causal_conv1d, fla.modules',
    )


def run(*command: str, cwd: Path = ROOT) -> str:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout


def relock() -> None:
    # The catalog lock record and its lab overlay copy carry the SHA-256 of uv.lock. Read the recorded value
    # (not the current file) so a retry after a partial run still rewrites the overlay copy.
    recorded = re.search(r'dependency_lock_sha256 = "([0-9a-f]{64})"', CATALOG_LOCKS.read_text(encoding="utf-8"))
    if recorded is None:
        raise SystemExit(f"{CATALOG_LOCKS.relative_to(ROOT)} records no uv.lock digest")
    old_digest = recorded.group(1)
    run("uv", "lock")
    run("uv", "lock", "--project", "tools/quantization")
    run(
        "uv",
        "export",
        "--project",
        "tools/quantization",
        "--frozen",
        "--python",
        "3.13.12",
        "--no-emit-project",
        "--no-dev",
        "--format",
        "requirements.txt",
        "--output-file",
        str((KINDS / "locks/transform.lock.txt").relative_to(ROOT)),
    )
    release = VERL / "release"
    run("uv", "lock", "--python", "3.13.12", cwd=release)
    run(
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--no-annotate",
        "--format",
        "requirements-txt",
        "--output-file",
        "backend-constraints.txt",
        cwd=release,
    )
    profile = VERL / "profile.toml"
    text = profile.read_text(encoding="utf-8")
    for key, name in (("dependency_lock_sha256", "uv.lock"), ("backend_constraints_sha256", "backend-constraints.txt")):
        digest = hashlib.sha256((release / name).read_bytes()).hexdigest()
        text = re.sub(rf'^{key} = "[0-9a-f]{{64}}"', f'{key} = "{digest}"', text, count=1, flags=re.M)
    profile.write_text(text, encoding="utf-8")
    run("uv", "sync", "--all-packages", "--locked", "--group", "dev", "--extra", "trl", "--python", "3.13")
    run("uv", "run", "posttrain-release", "lock-runtime-dependencies")
    run("uv", "run", "posttrain-release", "lock-dependencies")
    new_digest = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    if new_digest not in CATALOG_LOCKS.read_text(encoding="utf-8"):
        raise SystemExit("posttrain-release lock-dependencies did not record the new uv.lock digest")
    LAB_OVERLAY.write_text(LAB_OVERLAY.read_text(encoding="utf-8").replace(old_digest, new_digest), encoding="utf-8")
    for path in sorted((KINDS / "locks").glob("*.lock.txt")):
        if any(line.startswith("causal-conv1d") for line in path.read_text(encoding="utf-8").splitlines()):
            print(f"causal-conv1d locked in {path.name}")


def main() -> int:
    verify_published()
    edit_sources()
    relock()
    print("done: review git diff, run the validation ladder, rebuild the kind images, then commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
