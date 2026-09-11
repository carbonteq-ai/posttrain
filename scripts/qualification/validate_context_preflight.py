"""Reject a policy preflight that still loses trajectories to context capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--max-context-truncations", type=int, default=0)
    args = parser.parse_args()
    if args.max_context_truncations < 0:
        parser.error("max context truncations must be non-negative")
    summary = json.loads(args.summary.read_text())
    assert summary["trajectories"] == 20
    assert summary["failed"] == 0
    assert summary["truncation_reasons"].get("context_length", 0) <= args.max_context_truncations
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
