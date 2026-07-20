"""Command-line validation and resolution for checked-in profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common import PROFILES_DIR
from common.profiles import ProfileError, ProfileResolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and validate a reusable profile")
    parser.add_argument("kind", choices=("models", "train", "eval", "serve"))
    parser.add_argument("reference", help="Profile reference relative to its kind directory")
    parser.add_argument("--root", type=Path, default=PROFILES_DIR)
    args = parser.parse_args()

    try:
        resolved = ProfileResolver(args.root).resolve(args.kind, args.reference)
    except ProfileError as error:
        parser.error(str(error))

    print(yaml.safe_dump(resolved.data, sort_keys=False), end="")


if __name__ == "__main__":
    main()
