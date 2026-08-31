"""Command-line entrypoint installed in every framework job image."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
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
        # This must run before importing execute or a selected backend: either
        # may load libcuda into the process and make library-path changes too
        # late to affect CUDA initialization.
        from .cuda_compat import activate_cuda_compatibility

        activate_cuda_compatibility()
        from .trust import install_additional_trust

        install_additional_trust()
        from .execute import execute_manifest

        result = execute_manifest(arguments.manifest)
        print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    if arguments.command == "qualify":
        # Offline package qualification deliberately does not require a GPU.
        from .trust import install_additional_trust

        install_additional_trust()
        from .execute import qualify_manifest

        result = qualify_manifest(
            arguments.manifest,
            timeout_seconds=arguments.timeout_seconds,
            allow_deferred=arguments.allow_deferred,
        )
        print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
