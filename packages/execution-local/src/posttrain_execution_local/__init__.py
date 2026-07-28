"""Local Docker adapter for the provider-neutral execution lifecycle."""

from .adapter import DockerCli, LocalDockerExecutionProvider

__all__ = ["DockerCli", "LocalDockerExecutionProvider"]
