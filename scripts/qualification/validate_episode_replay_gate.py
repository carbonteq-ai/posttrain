"""Fail closed unless the frozen episode-judge replay is complete and differentiated."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["episodes"] == 8
    assert manifest["valid_episodes"] == 8
    assert manifest["invalid_episodes"] == 0
    summary = manifest["summary"]
    assert summary["all_dimensions_perfect_episodes"] < manifest["episodes"]
    for values in summary["dimension_scores"].values():
        assert len(values) == manifest["episodes"]
        assert len(set(values)) > 1
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
