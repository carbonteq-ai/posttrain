from __future__ import annotations

import json
import os
from pathlib import Path

from posttrain.execution import latest_runtime_image


def test_latest_runtime_image_uses_write_time_not_digest_filename(
    tmp_path: Path,
) -> None:
    older = tmp_path / f"{'f' * 64}.json"
    newer = tmp_path / f"{'0' * 64}.json"
    older.write_text(json.dumps({"image": (f"registry.lan/posttrain@sha256:{'a' * 64}")}))
    newer.write_text(json.dumps({"image": (f"registry.lan/posttrain@sha256:{'b' * 64}")}))
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert latest_runtime_image(tmp_path).value.endswith("b" * 64)


def test_latest_runtime_image_filters_by_profile(tmp_path: Path) -> None:
    framework = tmp_path / "framework.json"
    unrelated = tmp_path / "unrelated.json"
    framework.write_text(
        json.dumps(
            {
                "profile": "framework/job@1",
                "image": f"registry.lan/framework@sha256:{'a' * 64}",
            }
        )
    )
    unrelated.write_text(
        json.dumps(
            {
                "profile": "training/verl@1",
                "image": f"registry.lan/verl@sha256:{'b' * 64}",
            }
        )
    )
    os.utime(framework, ns=(1_000_000_000, 1_000_000_000))
    os.utime(unrelated, ns=(2_000_000_000, 2_000_000_000))

    assert latest_runtime_image(
        tmp_path,
        profile="framework/job@1",
    ).value.endswith("a" * 64)
