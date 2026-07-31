from posttrain.catalog import ProjectExecutionDefaults
from posttrain.project import ExecutionOverrides, PackageOverrides, resolve_execution_settings


def test_public_execution_settings_preserve_layer_precedence_and_provenance() -> None:
    resolved = resolve_execution_settings(
        ProjectExecutionDefaults(provider="project", environment_names=("PROJECT_TOKEN",)),
        local=ExecutionOverrides(provider="local", target="targets/local@1", environment_names=("LOCAL_TOKEN",)),
        cli=ExecutionOverrides(provider="cli", environment_names=("CLI_TOKEN",)),
        job=ExecutionOverrides(
            provider="job",
            runtime_profile="framework/supervised@1",
            timeout_seconds=30,
            max_attempts=2,
            priority=1,
            environment_names=("JOB_TOKEN",),
        ),
    )

    assert resolved.provider == "cli"
    assert resolved.target == "targets/local@1"
    assert resolved.environment_names == ("JOB_TOKEN", "PROJECT_TOKEN", "LOCAL_TOKEN", "CLI_TOKEN")
    assert resolved.sources == {
        "provider": "cli",
        "target": "local",
        "runtime_profile": "job",
        "timeout_seconds": "job",
        "max_attempts": "job",
        "priority": "job",
        "environment_names": "cli",
    }
    assert PackageOverrides(target="targets/override@1").as_execution_overrides().target == "targets/override@1"
