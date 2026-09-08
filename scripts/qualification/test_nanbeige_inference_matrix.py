import json
import sys

import nanbeige_inference_matrix
import pytest
from nanbeige_inference_matrix import DRAFT_REVISION, PROFILES, TARGET_REVISION, _command


def test_matrix_has_four_orthogonal_profiles() -> None:
    assert [
        (profile.name, profile.turboquant, profile.dspark, profile.supported)
        for profile in PROFILES
    ] == [
        ("standard", False, False, True),
        ("turboquant", True, False, True),
        ("dspark", False, True, True),
        ("dspark-turboquant", True, True, False),
    ]


def test_commands_pin_target_and_compose_dspark_with_turboquant() -> None:
    command = _command(PROFILES[-1], 8123)

    assert command[command.index("--revision") + 1] == TARGET_REVISION
    assert command[command.index("--kv-cache-dtype") + 1] == "turboquant_k8v4"
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative == {
        "method": "dspark",
        "num_speculative_tokens": 7,
        "model": "Nanbeige/Nanbeige4.2-3B-DSpark",
        "revision": DRAFT_REVISION,
    }


def test_runner_rejects_mutable_runtime_image(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nanbeige_inference_matrix.py",
            str(tmp_path / "evidence"),
            "--runtime-image",
            "registry.example/runtime:latest",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        nanbeige_inference_matrix.main()
