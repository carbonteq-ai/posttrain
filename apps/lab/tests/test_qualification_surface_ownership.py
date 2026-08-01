"""Keep the temporary qualification harness migration inventory complete."""

from __future__ import annotations

import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
INVENTORY = WORKSPACE / "apps" / "lab" / "qualification-surfaces.toml"
SCRIPTS = WORKSPACE / "scripts" / "qualification"
EXECUTION_FIXTURES = WORKSPACE / "packages" / "execution" / "tests" / "fixtures"
REMOTE_GPU_FIXTURE = WORKSPACE / "apps" / "lab" / "tests" / "fixtures" / "remote_gpu_project"
OBSERVATORY_QUALIFICATION = (
    WORKSPACE / "apps" / "observatory" / "src" / "posttrain_observatory" / "deployment_qualification.py"
)
LAB_SCENARIOS = WORKSPACE / "apps" / "lab" / "src" / "posttrain_lab" / "qualification" / "scenarios.py"

_KINDS = {
    "scenario-policy",
    "direct-launcher",
    "evidence-validator",
    "public-observatory-contract",
    "operator-preflight",
    "provider-fixture",
    "product-specific-launcher",
    "lab-dependent-fixture",
}
_OWNERS = {"lab", "execution", "observatory", "ai-infra", "ambient-agent"}
_TRANSITIONS = {"promote", "move", "retire"}
_FIELDS = {"id", "path", "kind", "owner", "transition", "replacement", "tests", "parity", "deletion_condition"}


def _inventory() -> list[dict[str, object]]:
    document = tomllib.loads(INVENTORY.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    surfaces = document["surfaces"]
    assert isinstance(surfaces, list)
    assert all(isinstance(surface, dict) for surface in surfaces)
    return surfaces


def test_every_temporary_script_and_lab_dependent_fixture_has_one_owner_and_exit_criterion() -> None:
    surfaces = _inventory()
    assert {surface["id"] for surface in surfaces} == {
        "algorithm-scenarios",
        "algorithm-scenario-launcher",
        "algorithm-evidence-validator",
        "observatory-http-qualification",
        "artifact-lifecycle-preflight",
        "runtime-smoke-payload",
        "queue-probe-payload",
        "local-runtime-smoke-launcher",
        "dstack-runtime-smoke-launcher",
        "ambient-model-backward-launcher",
        "ambient-artifact-consumer-launcher",
        "ambient-serving-launcher",
        "remote-gpu-lab-fixture",
    }
    assert len({surface["path"] for surface in surfaces}) == len(surfaces)

    expected_paths = {
        path.relative_to(WORKSPACE).as_posix() for path in SCRIPTS.glob("*.py") if path.name != "__init__.py"
    } | {
        REMOTE_GPU_FIXTURE.relative_to(WORKSPACE).as_posix(),
        OBSERVATORY_QUALIFICATION.relative_to(WORKSPACE).as_posix(),
        LAB_SCENARIOS.relative_to(WORKSPACE).as_posix(),
        *(path.relative_to(WORKSPACE).as_posix() for path in EXECUTION_FIXTURES.glob("*_job.py")),
    }
    assert {surface["path"] for surface in surfaces} == expected_paths

    for surface in surfaces:
        assert set(surface) == _FIELDS
        assert isinstance(surface["id"], str) and surface["id"]
        assert isinstance(surface["path"], str) and (WORKSPACE / surface["path"]).exists()
        assert surface["kind"] in _KINDS
        assert surface["owner"] in _OWNERS
        assert surface["transition"] in _TRANSITIONS
        assert isinstance(surface["replacement"], str) and surface["replacement"]
        assert isinstance(surface["deletion_condition"], str) and surface["deletion_condition"]
        parity = surface["parity"]
        assert isinstance(parity, list) and len(parity) >= 2
        assert all(isinstance(check, str) and check for check in parity)
        tests = surface["tests"]
        assert isinstance(tests, list)
        assert all(isinstance(test, str) and (WORKSPACE / test).is_file() for test in tests)


def test_current_script_unit_tests_are_owned_by_the_surface_they_characterize() -> None:
    documented_script_tests = {
        test
        for surface in _inventory()
        for test in surface["tests"]  # type: ignore[index]
        if test.startswith("scripts/qualification/tests/")
    }
    actual_tests = {path.relative_to(WORKSPACE).as_posix() for path in (SCRIPTS / "tests").glob("test_*.py")}

    assert documented_script_tests == actual_tests
