"""Worker entry point that verifies an immutable bundle before executing it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .bundles import verify_bundle
from .contracts import BundleRef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a job command is required after --")
    bundle = BundleRef(args.bundle.resolve(), args.digest)
    verify_bundle(bundle)
    os.chdir(bundle.path)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
