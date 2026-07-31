"""The environment domain has one public owner and a compatibility re-export."""

from posttrain.environment import EnvironmentBinding, EnvironmentBindingSchema, EnvironmentSource
from posttrain.eval import EnvironmentBinding as LegacyEnvironmentBinding
from posttrain.eval import EnvironmentBindingSchema as LegacyEnvironmentBindingSchema


def test_eval_reexports_the_environment_owner_for_one_release() -> None:
    assert LegacyEnvironmentBinding is EnvironmentBinding
    assert LegacyEnvironmentBindingSchema is EnvironmentBindingSchema


def test_environment_source_remains_available_without_the_eval_package() -> None:
    source = EnvironmentSource(
        package="example-env",
        repository="https://github.com/example/environments",
        revision="a" * 40,
        subdirectory="example_env",
    )

    assert source.package == "example-env"
