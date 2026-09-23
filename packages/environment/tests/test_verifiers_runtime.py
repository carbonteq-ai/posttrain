from __future__ import annotations

import os
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
    module.__dict__["EnvConfig"] = EnvConfig
    module.__dict__["Env"] = Env
    monkeypatch.setitem(sys.modules, "verifiers.v1.env", module)
    loaders = ModuleType("verifiers.v1.utils.loaders")
    loaders.__dict__["load_environment"] = Env
    monkeypatch.setitem(sys.modules, "verifiers.v1.utils.loaders", loaders)

    assert verifiers_environment_types() == (EnvConfig, Env)
    config = EnvConfig()
    environment = materialize_verifiers_environment(config)
    assert isinstance(environment, Env)
    assert environment.config is config


def test_fork_server_is_enabled_by_default_and_respects_opt_out(monkeypatch):
    from posttrain.environment.verifiers_runtime import (
        FORK_SERVER_PRELOAD,
        enable_verifiers_fork_server,
    )

    monkeypatch.delenv("VF_FORK_SERVER", raising=False)
    monkeypatch.delenv("VF_FORK_SERVER_PRELOAD", raising=False)
    enable_verifiers_fork_server()
    assert os.environ["VF_FORK_SERVER"] == "1"
    assert os.environ["VF_FORK_SERVER_PRELOAD"].split(",") == list(FORK_SERVER_PRELOAD)

    monkeypatch.setenv("VF_FORK_SERVER", "0")
    monkeypatch.setenv("VF_FORK_SERVER_PRELOAD", "openai")
    enable_verifiers_fork_server()
    assert os.environ["VF_FORK_SERVER"] == "0"
    assert os.environ["VF_FORK_SERVER_PRELOAD"] == "openai"
