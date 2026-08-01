"""Command-line entrypoint installed in every framework job image."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .execute import execute_manifest, qualify_manifest
from .trust import install_additional_trust


def main() -> None:
    # Before anything verifies a certificate, including the runtime's own
    # tracking calls.
    install_additional_trust()

    parser = argparse.ArgumentParser(prog="posttrain-runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--manifest", type=Path, required=True)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--timeout-seconds", type=float, default=60.0)
    qualify.add_argument("--allow-deferred", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "execute":
        result = execute_manifest(arguments.manifest)
        print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    if arguments.command == "qualify":
        result = qualify_manifest(
            arguments.manifest,
            timeout_seconds=arguments.timeout_seconds,
            allow_deferred=arguments.allow_deferred,
        )
        print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
