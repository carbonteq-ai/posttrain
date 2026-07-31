"""CLI session state passed through Typer contexts."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from posttrain.catalog import ProjectLayout, discover_project, open_catalog


@dataclass
class CliState:
    project_root: Path | None = None
    env_file: Path | None = None
    json_output: bool = False
    traceback: bool = False
    json_stream: TextIO = field(default_factory=lambda: sys.stdout)

    def layout(self) -> ProjectLayout:
        return discover_project(Path.cwd(), explicit_root=self.project_root)

    def open_catalog(self) -> tuple[ProjectLayout, Any]:
        layout = self.layout()
        return (
            layout,
            open_catalog(
                scope=layout.project_id,
                overlays=layout.catalog_overlays,
                catalog_root=layout.base_catalog,
                required_plugin_distributions=layout.catalog_plugin_requirements,
            ),
        )
