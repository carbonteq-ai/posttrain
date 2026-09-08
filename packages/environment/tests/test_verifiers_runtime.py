from __future__ import annotations

import sys
from types import ModuleType

from posttrain.environment.verifiers_runtime import (
    materialize_verifiers_environment,
    verifiers_environment_types,
)


def test_verifiers_v03_env_name_is_supported(monkeypatch) -> None:
    class EnvConfig:
        pass

    class Env:
        def __init__(self, config: EnvConfig) -> None:
            self.config = config

    module = ModuleType("verifiers.v1.env")
    module.EnvConfig = EnvConfig
    module.Env = Env
    monkeypatch.setitem(sys.modules, "verifiers.v1.env", module)
    loaders = ModuleType("verifiers.v1.utils.loaders")
    loaders.load_environment = Env
    monkeypatch.setitem(sys.modules, "verifiers.v1.utils.loaders", loaders)

    assert verifiers_environment_types() == (EnvConfig, Env)
    config = EnvConfig()
    environment = materialize_verifiers_environment(config)
    assert isinstance(environment, Env)
    assert environment.config is config
