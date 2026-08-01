"""Project initialization scaffolding."""

from __future__ import annotations

from .init_machine import add_machine_project, initialize_machine
from .init_project import initialize, install_starter

__all__ = ["add_machine_project", "initialize", "initialize_machine", "install_starter"]
